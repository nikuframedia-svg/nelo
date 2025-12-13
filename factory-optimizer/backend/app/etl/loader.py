import json
import logging
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from app.ml.inventory import InventoryPredictor


logger = logging.getLogger(__name__)

DB_FILENAME = "factory_optimizer.db"
STATUS_FILENAME = "etl_status.json"
MAPPINGS_FILENAME = "etl_mappings.json"


def _normalize_header(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("%", " percent ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


COLUMN_SYNONYMS_RAW: Dict[str, set[str]] = {
    "sku": {
        "sku",
        "artigo",
        "codigo",
        "codigo_produto",
        "codigo_do_produto",
        "ref",
        "referencia",
        "produto",
        "item",
        "cod",
        "codigo_sku",
        "codigo_artigo",
    },
    "setor": {"setor", "sector", "linha", "departamento", "area"},
    "ordem_grupo": {"ordem_grupo", "ordem grupo", "grupo_ordem", "ordemgrupo", "id_grupo", "grupo"},
    "grupo_operacao": {"grupo_operacao", "operacao_grupo", "grupoop", "operation_group", "grupo_operacoes"},
    "alternativa": {"alternativa", "alt", "rota", "rota_alternativa"},
    "operacao": {"operacao", "operation", "operacao_nome", "tarefa", "etapa", "op"},
    "maquinas_possiveis": {
        "maquinas_possiveis",
        "maquinas_possiveis",
        "maquinas_elegiveis",
        "maquinas",
        "equipamentos",
        "recursos",
        "machines",
        "resources",
    },
    "ratio_pch": {
        "ratio_pch",
        "racio_pc_h",
        "racio_pch",
        "pcs_por_h",
        "pecas_por_hora",
        "pc_h",
        "throughput",
        "capacidade_h",
        "pcs_h",
        "racio",
    },
    "overlap_prev": {
        "overlap_prev",
        "overlap",
        "overlap_pct",
        "percent_anterior",
        "percentual_anterior",
        "percentual_predecessor",
        "percent_anterior_para_iniciar",
        "percent_predecessor",
    },
    "pessoas": {
        "pessoas",
        "num_pessoas",
        "qtd_pessoas",
        "colaboradores",
        "equipa",
        "people",
        "pessoas_necessarias",
        "pessoas_necessárias",
    },
    "qty": {"qty", "quantidade", "qtd", "quant", "quantity", "volume"},
    "data_prometida": {"data_prometida", "data_entrega", "prazo", "due_date", "data_prevista", "promessa"},
    "prioridade": {"prioridade", "priority", "critico", "nivel_prioridade"},
    "recurso_pref": {"recurso_pref", "recurso_preferido", "maquina_preferida", "recurso", "resource_pref", "preferred_resource"},
    "data": {"data", "data_movimento", "data_registo", "data_lancamento", "date"},
    "tipo_mov": {
        "tipo_mov",
        "tipo",
        "movimento",
        "tipo_movimento",
        "mov_type",
        "movimentacao",
        "codigo_do_",
        "codigo_do",
    },
    "entradas": {"entradas", "entrada", "rececao", "recebido", "input", "qty_in"},
    "saidas": {"saidas", "saida", "saída", "consumo", "output", "qty_out", "retiradas"},
    "saldo": {"saldo", "stock", "stock_atual", "saldo_final", "saldo_atual", "saldo_total"},
    "armazem": {
        "armazem",
        "armazem_origem",
        "warehouse",
        "deposito",
        "local",
        "localizacao",
        "location",
        "arm_",
        "arm",
    },
    "valor_unit": {"valor_unit", "valor_unitario", "preco_unit", "custo_unit", "valor_un", "valor"},
    "lote": {"lote", "batch", "lote_id", "nr_lote", "lote_prod"},
    "stock_atual": {"stock_atual", "stock", "saldo", "saldo_atual", "inventario"},
    "cobertura_dias": {"cobertura_dias", "cobertura", "dias_cobertura", "coverage_days", "coverage"},
    "num_pessoas": {"num_pessoas", "pessoas", "qtd_pessoas", "colaboradores"},
    "horario": {"horario", "turno", "schedule", "shift", "horario_trabalho"},
}


def _build_synonyms() -> Dict[str, set[str]]:
    synonyms: Dict[str, set[str]] = {}
    for canonical, aliases in COLUMN_SYNONYMS_RAW.items():
        normalized_canonical = _normalize_header(canonical)
        items = {normalized_canonical}
        for alias in aliases:
            items.add(_normalize_header(alias))
        synonyms[normalized_canonical] = items
    return synonyms


COLUMN_SYNONYMS = _build_synonyms()


CANONICAL_SCHEMA: Dict[str, Dict[str, Any]] = {
    "roteiros": {
        "canonical": [
            "sku",
            "setor",
            "ordem_grupo",
            "grupo_operacao",
            "alternativa",
            "operacao",
            "maquinas_possiveis",
            "ratio_pch",
            "overlap_prev",
            "pessoas",
        ],
        "required": ["sku", "ordem_grupo", "grupo_operacao", "maquinas_possiveis"],
        "defaults": {"overlap_prev": 0.0, "pessoas": 1},
        "min_matches": 4,
    },
    "ordens": {
        "canonical": ["sku", "qty", "data_prometida", "prioridade", "recurso_pref"],
        "required": ["sku", "qty"],
        "defaults": {"prioridade": "Media", "recurso_pref": ""},
        "min_matches": 3,
    },
    "stocks_mov": {
        "canonical": ["sku", "data", "tipo_mov", "entradas", "saidas", "saldo", "armazem", "valor_unit", "lote"],
        "required": ["sku", "data", "saldo"],
        "defaults": {"tipo_mov": "Desconhecido", "entradas": 0.0, "saidas": 0.0, "armazem": "DEFAULT", "valor_unit": 0.0, "lote": ""},
        "min_matches": 2,  # Reduced to allow sheets with just sku+data+saldo
    },
    "stocks_snap": {
        "canonical": ["sku", "data", "stock_atual", "cobertura_dias"],
        "required": ["sku", "stock_atual"],
        "defaults": {"cobertura_dias": 0.0},
        "min_matches": 2,
    },
    "staffing": {
        "canonical": ["setor", "num_pessoas", "horario"],
        "required": ["setor", "num_pessoas"],
        "defaults": {"horario": "08:00-16:00"},
        "min_matches": 2,
    },
}

ALL_CANONICALS = sorted({col for schema in CANONICAL_SCHEMA.values() for col in schema["canonical"]})

DEFAULT_MAPPINGS: Dict[str, Any] = {"version": 1, "columns": {}, "sheets": {}}


@dataclass
class ColumnMapping:
    original: str
    normalized: str
    canonical: Optional[str]
    confidence: float
    source: str
    suggested: Optional[str] = None


@dataclass
class SheetReport:
    file: str
    sheet: str
    inferred_type: str
    score: float
    mapped_pct: float
    rows: int
    columns: int
    status: str
    mapped: List[str] = field(default_factory=list)
    pending: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    columns_details: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "sheet": self.sheet,
            "type": self.inferred_type,
            "score": self.score,
            "mapped_pct": self.mapped_pct,
            "rows": self.rows,
            "columns": self.columns,
            "status": self.status,
            "mapped": self.mapped,
            "pending": self.pending,
            "missing_required": self.missing_required,
            "warnings": self.warnings,
            "columns_details": self.columns_details,
        }


class DataLoader:
    """Responsável pelo pipeline ETL, estado e dados em cache."""

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        base_dir = Path(__file__).parent.parent
        if data_dir is None:
            data_dir = base_dir / "data"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.data_dir / DB_FILENAME
        self.status_path = self.data_dir / STATUS_FILENAME
        self.mappings_path = self.data_dir / MAPPINGS_FILENAME

        self.roteiros: Optional[pd.DataFrame] = None
        self.staffing: Optional[pd.DataFrame] = None
        self.ordens: Optional[pd.DataFrame] = None
        self.stocks_mov: Optional[pd.DataFrame] = None
        self.stocks_snap: Optional[pd.DataFrame] = None

        self.mappings: Dict[str, Any] = self._load_mappings()
        self.status: Dict[str, Any] = {
            "last_run": None,
            "last_import_at": None,
            "files": [],
            "sheets": [],
            "warnings": [],
            "ready_flags": {"planning_ready": False, "inventory_ready": False},
            "has_data": False,
            "mappings_version": self.mappings.get("version", 1),
        }
        self._load_status_from_disk()

        self._current_sheet_reports: List[Dict[str, Any]] = []
        self._current_warnings: List[str] = []

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def process_existing_files(self) -> Dict[str, int]:
        self._ensure_sample_files_if_needed()
        files = sorted(self.data_dir.glob("*.xlsx"))
        self._start_run()

        summary_total = {"artigos": 0, "skus": 0}
        for path in files:
            logger.info("Processing existing file: %s", path)
            try:
                result = self._process_file(path)
                summary_total["artigos"] += result.get("artigos", 0)
                summary_total["skus"] += result.get("skus", 0)
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Failed to process %s", path)
                self._record_file_error(path, str(exc))
                self._current_warnings.append(f"Erro ao processar {path.name}: {exc}")

        self._refresh_cache_from_db()
        self._finalize_run()
        return summary_total

    def process_uploaded_files(self, saved_files: List[Path]) -> Dict[str, int]:
        import time
        total_start = time.time()
        self._start_run()
        summary_total = {"artigos": 0, "skus": 0}
        for path in saved_files:
            file_start = time.time()
            logger.info("Processing uploaded file: %s", path)
            try:
                result = self._process_file(path)
                summary_total["artigos"] += result.get("artigos", 0)
                summary_total["skus"] += result.get("skus", 0)
                file_elapsed = time.time() - file_start
                logger.info(f"File {path.name} processed in {file_elapsed:.2f}s")
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Failed to process %s", path)
                self._record_file_error(path, str(exc))
                self._current_warnings.append(f"Erro ao processar {path.name}: {exc}")

        refresh_start = time.time()
        self._refresh_cache_from_db()
        refresh_elapsed = time.time() - refresh_start
        logger.info(f"_refresh_cache_from_db() took {refresh_elapsed:.2f}s")
        
        finalize_start = time.time()
        self._finalize_run()
        finalize_elapsed = time.time() - finalize_start
        logger.info(f"_finalize_run() took {finalize_elapsed:.2f}s")
        
        total_elapsed = time.time() - total_start
        logger.info(f"process_uploaded_files() TOTAL: {total_elapsed:.2f}s")
        return summary_total

    def get_status(self) -> Dict[str, Any]:
        return self.status

    def generate_preview(self, filename: str, sheet: Optional[str] = None, limit: int = 5) -> Dict[str, Any]:
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Ficheiro {filename} não encontrado em {self.data_dir}")

        workbook = pd.ExcelFile(file_path)
        sheet_names = [sheet] if sheet else workbook.sheet_names
        previews: List[Dict[str, Any]] = []

        for sheet_name in sheet_names:
            df_raw = workbook.parse(sheet_name)
            mapping_result = self._map_sheet(file_path.name, sheet_name, df_raw, record_warnings=False)
            sample_raw = df_raw.head(limit).replace({np.nan: None}).to_dict(orient="records")
            canonical_sample = None
            if mapping_result.get("dataframe") is not None:
                canonical_sample = (
                    mapping_result["dataframe"]
                    .head(limit)
                    .replace({np.nan: None})
                    .to_dict(orient="records")
                )

            previews.append(
                {
                    "sheet": sheet_name,
                    "report": mapping_result["report"],
                    "sample": sample_raw,
                    "canonical_sample": canonical_sample,
                }
            )

        return {"file": filename, "sheets": previews}

    def update_mappings(
        self,
        file_name: str,
        sheet_name: str,
        overrides: Dict[str, str],
        global_overrides: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        sheet_key = self._sheet_key(file_name, sheet_name)
        sheet_map = self.mappings.setdefault("sheets", {}).setdefault(sheet_key, {})
        updated_sheet: Dict[str, str] = {}

        for original, canonical in overrides.items():
            normalized_original = _normalize_header(original)
            canonical_key = _normalize_header(canonical)
            self._validate_canonical(canonical_key)
            sheet_map[normalized_original] = canonical_key
            updated_sheet[normalized_original] = canonical_key

        if global_overrides:
            global_map = self.mappings.setdefault("columns", {})
            for original, canonical in global_overrides.items():
                normalized_original = _normalize_header(original)
                canonical_key = _normalize_header(canonical)
                self._validate_canonical(canonical_key)
                global_map[normalized_original] = canonical_key

        self.mappings["version"] = int(self.mappings.get("version", 1)) + 1
        self._save_mappings()

        self.status["mappings_version"] = self.mappings["version"]
        self._save_status_to_disk()

        return {"sheet_key": sheet_key, "updated": updated_sheet, "version": self.mappings["version"]}

    # Compatibilidade com módulos existentes --------------------------------

    def get_roteiros(self) -> pd.DataFrame:
        self._refresh_cache_from_db()
        return self.roteiros if self.roteiros is not None else pd.DataFrame()

    def get_staffing(self) -> pd.DataFrame:
        self._refresh_cache_from_db()
        return self.staffing if self.staffing is not None else pd.DataFrame()

    def get_ordens(self) -> pd.DataFrame:
        self._refresh_cache_from_db()
        return self.ordens if self.ordens is not None else pd.DataFrame()

    def get_stocks_mov(self) -> pd.DataFrame:
        self._refresh_cache_from_db()
        return self.stocks_mov if self.stocks_mov is not None else pd.DataFrame()

    def get_stocks_snap(self) -> pd.DataFrame:
        self._refresh_cache_from_db()
        return self.stocks_snap if self.stocks_snap is not None else pd.DataFrame()

    # ------------------------------------------------------------------
    # Core ETL
    # ------------------------------------------------------------------

    def _start_run(self):
        self._current_sheet_reports = []
        self._current_warnings = []

    def _process_file(self, file_path: Path) -> Dict[str, int]:
        workbook = pd.ExcelFile(file_path)
        summary = {"artigos": 0, "skus": 0}
        sheet_reports: List[SheetReport] = []
        aggregated_warnings: List[str] = []
        
        # Consolidate stocks_mov from all sheets before writing
        stocks_mov_frames: List[pd.DataFrame] = []

        for sheet_name in workbook.sheet_names:
            df_raw = workbook.parse(sheet_name)
            try:
                mapping_result = self._map_sheet(file_path.name, sheet_name, df_raw, record_warnings=True)
            except Exception as exc:  # pylint: disable=broad-except
                warning = f"Falha ao mapear folha {sheet_name} ({file_path.name}): {exc}"
                aggregated_warnings.append(warning)
                sheet_reports.append(
                    SheetReport(
                        file=file_path.name,
                        sheet=sheet_name,
                        inferred_type="error",
                        score=0.0,
                        mapped_pct=0.0,
                        rows=len(df_raw),
                        columns=len(df_raw.columns),
                        status="error",
                        warnings=[warning],
                    )
                )
                continue

            sheet_reports.append(mapping_result["report"])
            aggregated_warnings.extend(mapping_result["warnings"])

            dataframe = mapping_result.get("dataframe")
            table = mapping_result.get("table")
            if dataframe is not None and table:
                if table == "stocks_mov":
                    # Collect all stocks_mov dataframes to consolidate later
                    stocks_mov_frames.append(dataframe)
                else:
                    # Write other tables immediately
                    self._write_table(dataframe, table)
                    if table in {"roteiros", "ordens"} and "sku" in dataframe.columns:
                        summary["artigos"] = max(summary["artigos"], dataframe["sku"].dropna().nunique())
        
        # Consolidate and write all stocks_mov sheets together
        if stocks_mov_frames:
            consolidated_stocks = pd.concat(stocks_mov_frames, ignore_index=True)
            # Remove duplicates if any (same sku+data combination)
            if "sku" in consolidated_stocks.columns and "data" in consolidated_stocks.columns:
                consolidated_stocks = consolidated_stocks.drop_duplicates(subset=["sku", "data"], keep="last")
            self._write_table(consolidated_stocks, "stocks_mov")
            if "sku" in consolidated_stocks.columns:
                summary["skus"] = max(summary["skus"], consolidated_stocks["sku"].dropna().nunique())

        self._current_sheet_reports.extend(report.as_dict() for report in sheet_reports)
        self._current_warnings.extend(aggregated_warnings)
        self._record_file_entry(file_path, summary, [report.as_dict() for report in sheet_reports], aggregated_warnings)
        return summary

    def _map_sheet(
        self,
        file_name: str,
        sheet_name: str,
        dataframe: pd.DataFrame,
        record_warnings: bool,
    ) -> Dict[str, Any]:
        sheet_label = sheet_name.strip()
        dataframe = self._prepare_sheet_dataframe(sheet_name, dataframe)

        sheet_key = self._sheet_key(file_name, sheet_name)
        normalized_headers = {col: _normalize_header(col) for col in dataframe.columns}
        normalized_cols = list(normalized_headers.values())

        suggestions_any = {
            normalized: self._resolve_column(normalized, sheet_type=None, sheet_key=sheet_key)
            for normalized in normalized_cols
        }

        sheet_type, score = self._classify_sheet(normalized_cols, suggestions_any)
        if sheet_type == "unknown" and self._is_stock_sheet_name(sheet_label):
            sheet_type = "stocks_mov"
            score = max(score, 80.0)
            # Force re-resolution with stocks_mov context for better column mapping
            suggestions_any = {
                normalized: self._resolve_column(normalized, sheet_type="stocks_mov", sheet_key=sheet_key)
                for normalized in normalized_cols
            }

        table_name = self._table_for_sheet(sheet_type)
        schema = CANONICAL_SCHEMA.get(sheet_type)

        columns_details: List[Dict[str, Any]] = []
        mapped_canonicals: List[str] = []
        pending_headers: List[str] = []
        warnings: List[str] = []

        if sheet_type == "unknown" or schema is None:
            warning = (
                f"Folha {sheet_name} ({file_name}) não foi reconhecida automaticamente. "
                "Utilize /api/etl/preview para mapear manualmente."
            )
            warnings.append(warning)
            report = SheetReport(
                file=file_name,
                sheet=sheet_name,
                inferred_type="unknown",
                score=0.0,
                mapped_pct=0.0,
                rows=len(dataframe),
                columns=len(dataframe.columns),
                status="unknown",
                warnings=warnings,
            )
            return {"report": report, "table": None, "dataframe": None, "warnings": warnings}

        used_canonicals: set[str] = set()
        column_mapping: Dict[str, ColumnMapping] = {}

        for original, normalized in normalized_headers.items():
            canonical, confidence, source = self._resolve_column(normalized, sheet_type=sheet_type, sheet_key=sheet_key)
            suggested = suggestions_any.get(normalized, (None, 0.0, ""))[0]

            if canonical and canonical in used_canonicals:
                # Avoid duplicate assignment of the same canonical column
                canonical = None
                source = "duplicate"
                confidence = 0.0

            if canonical:
                used_canonicals.add(canonical)
                mapped_canonicals.append(canonical)
            else:
                pending_headers.append(original)

            column_mapping[original] = ColumnMapping(
                original=original,
                normalized=normalized,
                canonical=canonical,
                confidence=confidence,
                source=source,
                suggested=suggested,
            )

            columns_details.append(
                {
                    "original": original,
                    "normalized": normalized,
                    "canonical": canonical,
                    "suggested": suggested,
                    "confidence": confidence,
                    "source": source,
                }
            )

        canonical_set = schema["canonical"]
        mapped_set = set(mapped_canonicals)
        backfilled_sku = False
        
        # For stocks_mov sheets with SKU code as name, always backfill SKU
        if sheet_type == "stocks_mov" and self._is_stock_sheet_name(sheet_label):
            if "sku" in canonical_set and "sku" not in mapped_set:
                backfilled_sku = True
                mapped_set.add("sku")
                columns_details.append(
                    {
                        "original": "__sheet_name__",
                        "normalized": "__sheet_name__",
                        "canonical": "sku",
                        "suggested": "sku",
                        "confidence": 1.0,
                        "source": "sheet_name",
                    }
                )
        elif "sku" in canonical_set and "sku" not in mapped_set:
            # For other sheet types, backfill SKU if missing
            backfilled_sku = True
            mapped_set.add("sku")
            columns_details.append(
                {
                    "original": "__sheet_name__",
                    "normalized": "__sheet_name__",
                    "canonical": "sku",
                    "suggested": "sku",
                    "confidence": 1.0,
                    "source": "sheet_name",
                }
            )

        required_for_ok: set[str] = set()
        if sheet_type == "roteiros":
            required_for_ok = {
                "sku",
                "setor",
                "ordem_grupo",
                "grupo_operacao",
                "operacao",
                "maquinas_possiveis",
                "ratio_pch",
                "overlap_prev",
                "pessoas",
            }
        elif sheet_type == "stocks_mov":
            # For stocks_mov, only require the absolute minimum: sku, data, saldo
            # Other columns (tipo_mov, entradas, saidas, armazem) are optional
            required_for_ok = {"sku", "data", "saldo"}

        pending_canonicals = [col for col in canonical_set if col not in mapped_set]
        missing_required = [col for col in schema["required"] if col not in mapped_set]
        missing_for_ok = [col for col in required_for_ok if col not in mapped_set]
        
        if sheet_type == "stocks_mov" and required_for_ok:
            mapped_pct = round(len(mapped_set.intersection(required_for_ok)) / len(required_for_ok) * 100, 2)
        else:
            mapped_pct = round(len(mapped_set) / len(canonical_set) * 100, 2) if canonical_set else 0.0

        status = "partial"
        if missing_required or missing_for_ok:
            status = "needs_mapping"
        elif mapped_pct >= 95.0:
            status = "ok"
        elif sheet_type == "stocks_mov" and mapped_pct >= 66.0:  # At least 2 of 3 required (sku, data, saldo)
            status = "ok"  # Accept partial stocks_mov if we have the core columns

        if missing_required:
            warnings.append(
                f"Folha {sheet_name} ({file_name}) sem colunas críticas: {', '.join(sorted(missing_required))}"
            )
        elif missing_for_ok:
            warnings.append(
                f"Folha {sheet_name} ({file_name}) incompleta para planeamento: {', '.join(sorted(missing_for_ok))}"
            )

        canonical_df: Optional[pd.DataFrame] = None
        if status == "ok":
            canonical_df, clean_warnings = self._build_canonical_dataframe(
                dataframe,
                column_mapping,
                schema,
                sheet_type,
                sheet_name,
                backfilled_sku,
            )
            warnings.extend(clean_warnings)
        else:
            canonical_df = None

        report = SheetReport(
            file=file_name,
            sheet=sheet_name,
            inferred_type=sheet_type,
            score=score,
            mapped_pct=mapped_pct,
            rows=len(dataframe),
            columns=len(dataframe.columns),
            status=status,
            mapped=sorted(mapped_set),
            pending=sorted(pending_canonicals),
            missing_required=sorted(missing_required),
            warnings=warnings,
            columns_details=columns_details,
        )

        if record_warnings:
            self._current_warnings.extend(warnings)

        return {
            "report": report,
            "dataframe": canonical_df,
            "table": table_name,
            "warnings": warnings,
        }

    def _build_canonical_dataframe(
        self,
        dataframe: pd.DataFrame,
        column_mapping: Dict[str, ColumnMapping],
        schema: Dict[str, Any],
        sheet_type: str,
        sheet_name: str,
        sku_backfilled: bool,
    ) -> Tuple[pd.DataFrame, List[str]]:
        rename_map = {
            mapping.original: mapping.canonical
            for mapping in column_mapping.values()
            if mapping.canonical is not None
        }
        df = dataframe.rename(columns=rename_map).copy()
        df = df.loc[:, ~df.columns.duplicated()]

        for column in schema["canonical"]:
            if column not in df.columns:
                df[column] = np.nan

        df = df[schema["canonical"]]

        if "sku" in df.columns:
            sheet_value = sheet_name.strip()
            df["sku"] = df["sku"].apply(lambda value: value if isinstance(value, str) and value.strip() else value)
            df["sku"] = df["sku"].where(df["sku"].notna() & df["sku"].astype(str).str.strip().ne(""), sheet_value)
        elif sheet_type == "stocks_mov" and self._is_stock_sheet_name(sheet_name):
            # Force SKU from sheet name if missing and sheet name is a SKU code
            df["sku"] = sheet_name.strip()

        warnings = self._clean_canonical_dataframe(df, sheet_type, schema, sheet_name, sku_backfilled)
        return df, warnings

    def _clean_canonical_dataframe(
        self,
        df: pd.DataFrame,
        sheet_type: str,
        schema: Dict[str, Any],
        sheet_name: str,
        sku_backfilled: bool,
    ) -> List[str]:
        warnings: List[str] = []

        if sheet_type in {"roteiros", "ordens", "stocks_mov", "stocks_snap"} and "sku" in df.columns:
            if sku_backfilled:
                df["sku"] = df["sku"].apply(lambda value: str(value).strip() if pd.notna(value) else sheet_name.strip())
            else:
                df["sku"] = self._normalize_sku_series(df["sku"])
            df.dropna(subset=["sku"], inplace=True)

        if sheet_type == "roteiros":
            df["ordem_grupo"] = self._coerce_numeric(df["ordem_grupo"]).fillna(0).astype(int)
            df["grupo_operacao"] = df["grupo_operacao"].astype(str).str.strip()
            df["alternativa"] = df["alternativa"].astype(str).str.strip()
            if "operacao" in df.columns:
                df["operacao"] = df["operacao"].fillna(df["grupo_operacao"]).astype(str).str.strip()
            df["maquinas_possiveis"] = df["maquinas_possiveis"].astype(str).str.strip()

            ratio_series = self._coerce_numeric(df["ratio_pch"]).where(lambda s: s > 0)
            positive_median = ratio_series.dropna().median()
            if pd.isna(positive_median) or positive_median <= 0:
                positive_median = 120.0
                warnings.append("ratio_pch ausente. Aplicado default 120 pc/h.")
            df["ratio_pch"] = ratio_series.fillna(positive_median).clip(lower=1.0)

            overlap_series = self._coerce_percentage(df["overlap_prev"])
            if overlap_series.isna().all():
                warnings.append("overlap_prev ausente. Planeamento conservador (0% overlap).")
            overlap = overlap_series.fillna(0.0)
            df["overlap_prev"] = overlap.clip(lower=0.0)

            pessoas_series = self._coerce_numeric(df["pessoas"]).round().fillna(1)
            pessoas_series = pessoas_series.apply(lambda value: max(int(value) if not pd.isna(value) else 1, 1))
            df["pessoas"] = pessoas_series

        elif sheet_type == "ordens":
            df["qty"] = self._coerce_numeric(df["qty"]).fillna(0).round().astype(int)
            df["data_prometida"] = self._coerce_datetime(df["data_prometida"]).fillna(
                datetime.utcnow() + timedelta(days=7)
            )
            df["prioridade"] = df["prioridade"].fillna(schema["defaults"].get("prioridade", "Media"))
            df["recurso_pref"] = df["recurso_pref"].fillna(schema["defaults"].get("recurso_pref", ""))

        elif sheet_type == "stocks_mov":
            df["data"] = self._coerce_datetime(df["data"])
            df.dropna(subset=["data"], inplace=True)
            for column in ["entradas", "saidas", "saldo", "valor_unit"]:
                df[column] = self._coerce_numeric(df[column]).fillna(schema["defaults"].get(column, 0.0))
            df["tipo_mov"] = df["tipo_mov"].fillna(schema["defaults"].get("tipo_mov", "Desconhecido")).astype(str).str.strip()
            df["armazem"] = df["armazem"].fillna(schema["defaults"].get("armazem", "DEFAULT")).astype(str).str.upper()
            df["lote"] = df["lote"].fillna(schema["defaults"].get("lote", "")).astype(str).str.strip()

        elif sheet_type == "stocks_snap":
            df["data"] = self._coerce_datetime(df["data"])
            df["stock_atual"] = self._coerce_numeric(df["stock_atual"]).fillna(0.0)
            df["cobertura_dias"] = self._coerce_numeric(df["cobertura_dias"]).fillna(
                schema["defaults"].get("cobertura_dias", 0.0)
            )

        elif sheet_type == "staffing":
            df["setor"] = df["setor"].astype(str).str.strip()
            df["num_pessoas"] = self._coerce_numeric(df["num_pessoas"]).fillna(0).round().astype(int)
            df["horario"] = df["horario"].fillna(schema["defaults"].get("horario", "08:00-16:00"))

        # Apply defaults for remaining columns
        for column, default_value in schema.get("defaults", {}).items():
            if column in df.columns:
                df[column] = df[column].fillna(default_value)

        return warnings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _coerce_numeric(self, series: pd.Series) -> pd.Series:
        cleaned = (
            series.astype(str)
            .str.replace(r"[^\d,\.\-]", "", regex=True)
            .str.replace(",", ".", regex=False)
        )
        coerced = pd.to_numeric(cleaned, errors="coerce")
        return coerced

    def _coerce_percentage(self, series: pd.Series) -> pd.Series:
        numeric = (
            series.astype(str)
            .str.replace(r"[^\d,\.\-]", "", regex=True)
            .str.replace(",", ".", regex=False)
        )
        values = pd.to_numeric(numeric, errors="coerce")
        if values.notna().any():
            divided = values.where(values <= 1.0, values / 100.0)
        else:
            divided = values
        return divided

    def _coerce_datetime(self, series: pd.Series) -> pd.Series:
        coerced = pd.to_datetime(series, errors="coerce", dayfirst=True)
        return coerced

    def _normalize_sku_series(self, series: pd.Series) -> pd.Series:
        def _normalize(value: Any) -> Optional[str]:
            if pd.isna(value):
                return None
            text = unicodedata.normalize("NFKD", str(value))
            text = "".join(ch for ch in text if not unicodedata.combining(ch))
            text = text.upper().strip()
            text = re.sub(r"[^A-Z0-9]", "", text)
            return text or None

        return series.apply(_normalize)

    def _classify_sheet(
        self,
        normalized_cols: List[str],
        suggestions_any: Dict[str, Tuple[Optional[str], float, str]],
    ) -> Tuple[str, float]:
        best_type = "unknown"
        best_score = 0.0
        best_matches = 0

        for sheet_type, schema in CANONICAL_SCHEMA.items():
            matches = 0
            for normalized in normalized_cols:
                canonical = suggestions_any.get(normalized, (None, 0.0, ""))[0]
                confidence = suggestions_any.get(normalized, (None, 0.0, ""))[1]
                if canonical in schema["canonical"] and confidence >= 0.6:
                    matches += 1

            coverage = matches / max(len(schema["canonical"]), 1)
            if matches > best_matches or (matches == best_matches and coverage > best_score):
                best_matches = matches
                best_score = coverage
                best_type = sheet_type

        min_matches = CANONICAL_SCHEMA.get(best_type, {}).get("min_matches", 999)
        if best_matches < min_matches:
            return "unknown", 0.0

        return best_type, round(best_score * 100, 2)

    def _table_for_sheet(self, sheet_type: str) -> Optional[str]:
        return {
            "roteiros": "roteiros",
            "ordens": "ordens",
            "stocks_mov": "stocks_mov",
            "stocks_snap": "stocks_snap",
            "staffing": "staffing",
        }.get(sheet_type)

    def _candidate_canonicals(self, sheet_type: Optional[str]) -> List[str]:
        if sheet_type and sheet_type in CANONICAL_SCHEMA:
            return CANONICAL_SCHEMA[sheet_type]["canonical"]
        return ALL_CANONICALS

    def _resolve_column(
        self,
        normalized_header: str,
        sheet_type: Optional[str],
        sheet_key: Optional[str],
    ) -> Tuple[Optional[str], float, str]:
        normalized_header = _normalize_header(normalized_header)

        if sheet_key:
            sheet_overrides = self.mappings.get("sheets", {}).get(sheet_key, {})
            if normalized_header in sheet_overrides:
                canonical = sheet_overrides[normalized_header]
                return canonical, 1.0, "sheet_override"

        global_overrides = self.mappings.get("columns", {})
        if normalized_header in global_overrides:
            canonical = global_overrides[normalized_header]
            return canonical, 1.0, "global_override"

        candidates = self._candidate_canonicals(sheet_type)

        for canonical in candidates:
            canonical_norm = _normalize_header(canonical)
            if normalized_header == canonical_norm:
                return canonical_norm, 0.99, "exact"

        for canonical in candidates:
            canonical_norm = _normalize_header(canonical)
            synonyms = COLUMN_SYNONYMS.get(canonical_norm, {canonical_norm})
            if normalized_header in synonyms:
                return canonical_norm, 0.95, "synonym"

        best_canonical = None
        best_score = 0.0
        for canonical in candidates:
            canonical_norm = _normalize_header(canonical)
            synonyms = COLUMN_SYNONYMS.get(canonical_norm, {canonical_norm})
            for synonym in synonyms:
                score = SequenceMatcher(None, normalized_header, synonym).ratio()
                if score > best_score:
                    best_score = score
                    best_canonical = canonical_norm

        if best_canonical and best_score >= 0.75:
            return best_canonical, float(best_score), "fuzzy"

        return None, 0.0, "unknown"

    def _write_table(self, df: pd.DataFrame, table: str):
        if df is None or df.empty:
            return
        # Otimização: usar batch writes e desativar índices temporariamente
        with sqlite3.connect(self.db_path) as conn:
            # Desativar índices temporariamente para escrita mais rápida
            conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging - mais rápido
            conn.execute("PRAGMA synchronous=NORMAL")  # Menos seguro mas mais rápido
            df.to_sql(table, conn, if_exists="replace", index=False, method="multi", chunksize=1000)

    def _refresh_cache_from_db(self):
        if not self.db_path.exists():
            return
        with sqlite3.connect(self.db_path) as conn:
            self.roteiros = self._read_table(conn, "roteiros")
            if self.roteiros is not None and not self.roteiros.empty and "sku" in self.roteiros.columns:
                self.roteiros["SKU"] = self.roteiros["sku"]
            self.staffing = self._read_table(conn, "staffing")
            self.ordens = self._read_table(conn, "ordens", parse_dates=["data_prometida"])
            if self.ordens is not None and not self.ordens.empty and "sku" in self.ordens.columns:
                self.ordens["SKU"] = self.ordens["sku"]
            self.stocks_mov = self._read_table(conn, "stocks_mov", parse_dates=["data"])
            if self.stocks_mov is not None and not self.stocks_mov.empty:
                if "sku" in self.stocks_mov.columns:
                    self.stocks_mov["SKU"] = self.stocks_mov["sku"]
            self.stocks_snap = self._read_table(conn, "stocks_snap", parse_dates=["data"])
            if self.stocks_snap is not None and not self.stocks_snap.empty:
                if "sku" in self.stocks_snap.columns:
                    self.stocks_snap["SKU"] = self.stocks_snap["sku"]

    def _read_table(
        self,
        conn: sqlite3.Connection,
        table: str,
        parse_dates: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        try:
            return pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=parse_dates)
        except Exception:
            return None

    def _record_file_entry(
        self,
        file_path: Path,
        summary: Dict[str, int],
        sheet_reports: List[Dict[str, Any]],
        warnings: List[str],
    ):
        timestamp = datetime.utcnow().isoformat()
        try:
            stat = file_path.stat()
            size_bytes = stat.st_size
            modified_at = datetime.utcfromtimestamp(stat.st_mtime).isoformat()
        except FileNotFoundError:
            size_bytes = 0
            modified_at = None

        entry = {
            "filename": file_path.name,
            "path": str(file_path),
            "size_bytes": size_bytes,
            "modified_at": modified_at,
            "status": "processed",
            "summary": summary,
            "sheets": [
                {
                    "sheet": report.get("sheet"),
                    "type": report.get("type"),
                    "status": report.get("status"),
                    "mapped_pct": report.get("mapped_pct"),
                }
                for report in sheet_reports
            ],
            "warnings": warnings,
            "timestamp": timestamp,
        }

        files = self.status.get("files", [])
        files.append(entry)
        self.status["files"] = files[-12:]
        self.status["last_run"] = timestamp
        self.status["last_import_at"] = timestamp

    def _record_file_error(self, file_path: Path, error: str):
        timestamp = datetime.utcnow().isoformat()
        entry = {
            "filename": file_path.name,
            "path": str(file_path),
            "status": "error",
            "summary": {"artigos": 0, "skus": 0},
            "error": error,
            "timestamp": timestamp,
        }
        files = self.status.get("files", [])
        files.append(entry)
        self.status["files"] = files[-12:]
        self.status["last_run"] = timestamp

    def _build_inventory_snapshot(self):
        import time
        start_time = time.time()
        stocks_mov_df = self.get_stocks_mov()
        if stocks_mov_df is None or stocks_mov_df.empty:
            empty_snap = pd.DataFrame(columns=["sku", "SKU", "data", "stock_atual", "ads_180", "cobertura_dias"])
            self._write_table(empty_snap, "stocks_snap")
            self.stocks_snap = empty_snap
            self._inventory_insights = self._default_inventory_insights()
            return

        df = stocks_mov_df.copy()
        if "SKU" in df.columns and "sku" not in df.columns:
            df["sku"] = df["SKU"]
        elif "sku" not in df.columns:
            df["sku"] = df.iloc[:, 0].astype(str).str.strip()

        if "data" in df.columns:
            df["data"] = pd.to_datetime(df["data"], errors="coerce", dayfirst=True)
        elif "Data" in df.columns:
            df["data"] = pd.to_datetime(df["Data"], errors="coerce", dayfirst=True)
        else:
            df["data"] = pd.NaT

        for numeric_column in ["entradas", "saidas", "saldo"]:
            if numeric_column in df.columns:
                df[numeric_column] = self._coerce_numeric(df[numeric_column])

        df = df.dropna(subset=["sku", "data"])
        if df.empty:
            empty_snap = pd.DataFrame(columns=["sku", "SKU", "data", "stock_atual", "ads_180", "cobertura_dias"])
            self._write_table(empty_snap, "stocks_snap")
            self.stocks_snap = empty_snap
            self._inventory_insights = self._default_inventory_insights()
            return

        df.sort_values(["sku", "data"], inplace=True)

        latest = df.groupby("sku", as_index=False).last()
        latest["stock_atual"] = latest.get("saldo", pd.Series(dtype=float)).fillna(0.0)

        horizon = datetime.utcnow() - timedelta(days=180)
        recent = df[df["data"] >= horizon]
        if "saidas" in recent.columns and not recent.empty:
            ads = recent.groupby("sku")["saidas"].sum() / 180
        else:
            ads = pd.Series(dtype=float)

        latest["ads_180"] = latest["sku"].map(ads).fillna(0.0)
        # Vectorização em vez de apply() - muito mais rápido
        latest["cobertura_dias"] = np.where(
            latest["ads_180"] > 0,
            latest["stock_atual"] / latest["ads_180"],
            np.inf
        )

        latest["data"] = pd.to_datetime(latest["data"], errors="coerce").dt.tz_localize(None)
        snapshot = latest[["sku", "data", "stock_atual", "ads_180", "cobertura_dias"]].copy()
        self._write_table(snapshot, "stocks_snap")

        snapshot_cache = snapshot.copy()
        snapshot_cache["SKU"] = snapshot_cache["sku"].astype(str)
        self.stocks_snap = snapshot_cache
        # Adiar cálculo de insights durante upload - só calcular quando necessário
        # self._inventory_insights = self._compute_inventory_insights()  # Removido para performance
        self._inventory_insights = None  # Será calculado lazy quando get_inventory_insights() for chamado
        
        elapsed = time.time() - start_time
        logger.info(f"_build_inventory_snapshot() completed in {elapsed:.2f}s")

    def _finalize_run(self):
        import time
        start_time = time.time()
        planning_ready = False
        inventory_ready = False
        summary = {"artigos": 0, "skus": 0}

        roteiros_df = self.get_roteiros()
        if roteiros_df is not None and not roteiros_df.empty:
            if "sku" in roteiros_df.columns:
                summary["artigos"] = int(pd.Series(roteiros_df["sku"]).dropna().nunique())
            planning_ready = summary["artigos"] > 0

        ordens_df = self.get_ordens()
        if ordens_df.empty and planning_ready:
            if self._generate_seed_orders():
                self._current_warnings.append(
                    "Ordens reais não encontradas. Geradas ordens seed para 7 dias."
                )
                ordens_df = self.get_ordens()

        if ordens_df is not None and not ordens_df.empty:
            planning_ready = True

        self._build_inventory_snapshot()
        stocks_snap_df = self.get_stocks_snap()
        stocks_mov_df = self.get_stocks_mov()

        if stocks_snap_df is not None and not stocks_snap_df.empty:
            inventory_ready = True
            sku_column = "sku" if "sku" in stocks_snap_df.columns else "SKU" if "SKU" in stocks_snap_df.columns else None
            if sku_column:
                summary["skus"] = int(pd.Series(stocks_snap_df[sku_column]).dropna().nunique())
        elif stocks_mov_df is not None and not stocks_mov_df.empty:
            inventory_ready = True
            sku_column = "sku" if "sku" in stocks_mov_df.columns else "SKU" if "SKU" in stocks_mov_df.columns else None
            if sku_column:
                summary["skus"] = int(pd.Series(stocks_mov_df[sku_column]).dropna().nunique())
        else:
            self._current_warnings.append(
                "Stocks não carregados. Inventário permanece em estado vazio."
            )

        self.status["ready_flags"] = {"planning_ready": planning_ready, "inventory_ready": inventory_ready}
        # Definir has_data: True se houver dados de planeamento OU inventário
        self.status["has_data"] = planning_ready or inventory_ready
        self.status["summary"] = summary
        self.status["sheets"] = self._current_sheet_reports
        self.status["warnings"] = list(dict.fromkeys(self._current_warnings))
        self.status["mappings_version"] = self.mappings.get("version", 1)
        # Não guardar insights completos no status durante upload - muito pesado
        # self.status["inventory"] = self._inventory_insights  # Removido para performance
        self.status["inventory"] = {"skus_total": summary.get("skus", 0)}  # Apenas contagem
        
        save_start = time.time()
        self._save_status_to_disk()
        save_elapsed = time.time() - save_start
        logger.info(f"_save_status_to_disk() took {save_elapsed:.2f}s")
        
        total_elapsed = time.time() - start_time
        logger.info(f"_finalize_run() completed in {total_elapsed:.2f}s")

    def _generate_seed_orders(self) -> bool:
        roteiros_df = self.get_roteiros()
        if roteiros_df.empty or "sku" not in roteiros_df.columns:
            return False

        unique_skus = roteiros_df["sku"].dropna().unique()
        if len(unique_skus) == 0:
            return False

        today = datetime.utcnow()
        seed_orders: List[Dict[str, Any]] = []
        priorities = ["Alta", "Media", "Baixa"]

        maquinas_map = {}
        if "maquinas_possiveis" in roteiros_df.columns:
            maquinas_map = (
                roteiros_df.groupby("sku")["maquinas_possiveis"].first().to_dict()
            )

        ratio_median = None
        if "ratio_pch" in roteiros_df.columns:
            ratio_series = self._coerce_numeric(roteiros_df["ratio_pch"]).dropna()
            if not ratio_series.empty:
                ratio_median = float(ratio_series.median())

        base_qty = max(int((ratio_median or 40) * 4), 60)

        for idx, sku in enumerate(unique_skus):
            for offset in range(1, 4):
                due_date = today + timedelta(days=offset * 3 + idx % 4)
                qty = base_qty + int((idx + offset) * 5)
                seed_orders.append(
                    {
                        "sku": sku,
                        "qty": qty,
                        "data_prometida": due_date,
                        "prioridade": priorities[(idx + offset) % len(priorities)],
                        "recurso_pref": (maquinas_map.get(sku) or "").split(",")[0].strip(),
                    }
                )

        df_orders = pd.DataFrame(seed_orders)
        self._write_table(df_orders, "ordens")
        return True

    def _sheet_key(self, file_name: str, sheet_name: str) -> str:
        return f"{file_name}::{sheet_name}"

    def _validate_canonical(self, canonical: str):
        canonical_key = _normalize_header(canonical)
        if canonical_key not in ALL_CANONICALS:
            raise ValueError(f"Coluna canónica desconhecida: {canonical}")

    def _ensure_sample_files_if_needed(self):
        prod_sample = self.data_dir / "Nikufra DadosProducao.xlsx"
        stock_sample = self.data_dir / "Nikufra Stocks.xlsx"
        if not prod_sample.exists():
            self._create_sample_production_data(prod_sample)
        if not stock_sample.exists():
            self._create_sample_stock_data(stock_sample)

    def _save_status_to_disk(self):
        # Otimização: escrever de forma mais eficiente
        try:
            # Escrever para ficheiro temporário primeiro, depois renomear (atomic)
            temp_path = self.status_path.with_suffix(".tmp")
            with temp_path.open("w", encoding="utf-8") as fp:
                json.dump(self.status, fp, indent=2, default=str, ensure_ascii=False)
            temp_path.replace(self.status_path)  # Atomic rename
        except Exception as exc:
            logger.warning(f"Erro ao guardar status: {exc}")
            # Fallback: escrever diretamente
            with self.status_path.open("w", encoding="utf-8") as fp:
                json.dump(self.status, fp, indent=2, default=str, ensure_ascii=False)

    def _load_status_from_disk(self):
        if not self.status_path.exists():
            return
        try:
            with self.status_path.open("r", encoding="utf-8") as fp:
                loaded = json.load(fp)
        except Exception:
            return

        self.status.update(loaded)
        self.status.setdefault("files", [])
        self.status.setdefault("sheets", [])
        self.status.setdefault("warnings", [])
        self.status.setdefault("ready_flags", {"planning_ready": False, "inventory_ready": False})
        self.status.setdefault("mappings_version", self.mappings.get("version", 1))
        self.status.setdefault("last_run", None)
        self.status.setdefault("last_import_at", None)

    def _load_mappings(self) -> Dict[str, Any]:
        if not self.mappings_path.exists():
            return DEFAULT_MAPPINGS.copy()
        try:
            with self.mappings_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception:
            return DEFAULT_MAPPINGS.copy()

        data.setdefault("version", 1)
        data.setdefault("columns", {})
        data.setdefault("sheets", {})
        return data

    def _save_mappings(self):
        with self.mappings_path.open("w", encoding="utf-8") as fp:
            json.dump(self.mappings, fp, indent=2)

    def _coerce_numeric_value(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------------
    # Dados de exemplo
    # ------------------------------------------------------------------

    def _create_sample_production_data(self, filepath: Path):
        start_date = datetime.utcnow()
        skus = [f"SKU-{i:03d}" for i in range(1, 11)]
        roteiros = {
            "SKU": skus,
            "setor": ["Transformação", "Acabamentos", "Transformação", "Acabamentos", "Embalagem"] * 2,
            "ordem_grupo": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
            "grupo_operacao": [
                "Transformação",
                "Acabamentos",
                "Transformação",
                "Acabamentos",
                "Embalagem",
                "Transformação",
                "Acabamentos",
                "Embalagem",
                "Transformação",
                "Acabamentos",
            ],
            "alternativa": ["A", "A", "B", "A", "A", "A", "B", "A", "A", "A"],
            "operacao": [
                "Corte",
                "Polimento",
                "Soldadura",
                "Pintura",
                "Embalagem",
                "Corte",
                "Polimento",
                "Embalagem",
                "Soldadura",
                "Pintura",
            ],
            "maquinas_possiveis": [f"M-{i:02d},M-{i+1:02d}" for i in range(1, 11)],
            "ratio_pch": np.random.randint(80, 180, size=10),
            "overlap_prev": np.random.uniform(0.0, 0.35, size=10),
            "pessoas": np.random.randint(1, 4, size=10),
        }
        staffing = {
            "setor": ["Transformação", "Acabamentos", "Embalagem"],
            "num_pessoas": [18, 12, 6],
            "horario": ["08:00-16:00", "08:00-16:00", "08:00-16:00"],
        }
        ordens = {
            "sku": skus,
            "qty": np.random.randint(50, 400, size=10),
            "data_prometida": [start_date + timedelta(days=i + 5) for i in range(10)],
            "prioridade": np.random.choice(["Alta", "Media", "Baixa"], size=10),
            "recurso_pref": np.random.choice([f"M-{i:02d}" for i in range(1, 6)], size=10),
        }

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            pd.DataFrame(roteiros).to_excel(writer, sheet_name="Roteiros", index=False)
            pd.DataFrame(staffing).to_excel(writer, sheet_name="Staffing", index=False)
            pd.DataFrame(ordens).to_excel(writer, sheet_name="Ordens", index=False)

    def _create_sample_stock_data(self, filepath: Path):
        dates = pd.date_range(end=datetime.utcnow(), periods=30, freq="D")
        skus = [f"SKU-{i:03d}" for i in range(1, 11)]

        mov_rows = []
        for date in dates:
            for sku in skus[:8]:
                mov_rows.append(
                    {
                        "sku": sku,
                        "data": date,
                        "tipo_mov": np.random.choice(["Venda", "Compra", "Ajuste"]),
                        "entradas": int(np.random.randint(0, 80)),
                        "saidas": int(np.random.randint(0, 50)),
                        "saldo": int(np.random.randint(100, 800)),
                        "armazem": np.random.choice(["A1", "B1", "C1"]),
                        "valor_unit": round(float(np.random.uniform(1.2, 5.0)), 2),
                        "lote": f"L{np.random.randint(100, 999)}",
                    }
                )
        snap_rows = []
        for date in dates:
            for sku in skus:
                snap_rows.append(
                    {
                        "sku": sku,
                        "data": date,
                        "stock_atual": int(np.random.randint(80, 600)),
                        "cobertura_dias": float(np.random.uniform(5, 30)),
                    }
                )

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            pd.DataFrame(mov_rows).to_excel(writer, sheet_name="Stocks_mov", index=False)
            pd.DataFrame(snap_rows).to_excel(writer, sheet_name="Stocks_snap", index=False)

    def _prepare_sheet_dataframe(self, sheet_name: str, dataframe: pd.DataFrame) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe

        df = dataframe.copy()
        first_column = df.columns[0]
        header_series = df[first_column].astype(str).str.strip()
        header_idx_candidates = header_series[header_series.str.lower().str.startswith("data")]

        if header_idx_candidates.any():
            header_idx = header_idx_candidates.index[0]
            header_row = df.iloc[header_idx].fillna("").map(lambda value: str(value).strip())
            df = df.iloc[header_idx + 1 :].copy()
            df.columns = [str(col).strip() for col in header_row]
            df = df.loc[:, ~df.columns.duplicated()]
            df.dropna(how="all", inplace=True)
            df.reset_index(drop=True, inplace=True)

            if "Codigo_do_" in df.columns:
                df = df[df["Codigo_do_"].astype(str).str.strip().str.lower() != "stock anterior"].copy()
            if "Data" in df.columns:
                df = df[df["Data"].astype(str).str.strip().str.lower() != "."].copy()

        return df

    def _is_stock_sheet_name(self, sheet_name: str) -> bool:
        clean = sheet_name.strip()
        if not clean:
            return False
        return bool(re.fullmatch(r"\d{6,}", clean))

    def _default_inventory_insights(self) -> Dict[str, Any]:
         return {
             "kpis": {
                 "total_stock": 0.0,
                 "average_coverage_days": 0.0,
                 "global_risk_score": 0.0,
                 "sku_count": 0,
             },
             "matrix": {
                 "A": {"X": 0, "Y": 0, "Z": 0},
                 "B": {"X": 0, "Y": 0, "Z": 0},
                 "C": {"X": 0, "Y": 0, "Z": 0},
             },
             "skus": [],
             "top_risks": [],
             "generated_at": datetime.utcnow().isoformat(),
         }

    def _compute_inventory_insights(self, compute_rop: bool = False) -> Dict[str, Any]:
        snapshot = self.stocks_snap if self.stocks_snap is not None else pd.DataFrame()
        if snapshot is None or snapshot.empty:
            return self._default_inventory_insights()

        df = snapshot.copy()
        df["SKU"] = df.get("SKU", df.get("sku", "")).astype(str)
        df["sku"] = df.get("sku", df["SKU"])
        df["stock_atual"] = df.get("stock_atual", 0.0).fillna(0.0)
        df["ads_180"] = df.get("ads_180", 0.0).fillna(0.0)
        df["cobertura_dias"] = (
            df.get("cobertura_dias", 0.0)
            .replace({np.inf: np.nan, -np.inf: np.nan})
            .fillna(0.0)
        )

        stocks_mov = self.get_stocks_mov()
        predictor = InventoryPredictor()

        if stocks_mov is not None and not stocks_mov.empty:
            mov_df = stocks_mov.copy()
            mov_df["SKU"] = mov_df.get("SKU", mov_df.get("sku", "")).astype(str)
            mov_df["data"] = pd.to_datetime(mov_df.get("data", mov_df.get("Data")), errors="coerce")
            mov_df = mov_df.dropna(subset=["SKU"])
            sales_totals = mov_df.groupby("SKU")["saidas"].sum().sort_values(ascending=False)
        else:
            sales_totals = pd.Series(dtype=float)

        total_sales = float(sales_totals.sum()) if not sales_totals.empty else 0.0

        abc_class: Dict[str, str] = {}
        if total_sales > 0:
            # Otimização: usar vectorização para calcular ABC
            cumulative_pct = (sales_totals / total_sales * 100).cumsum()
            # Usar cut() para classificar em batch
            abc_class_series = pd.cut(
                cumulative_pct,
                bins=[0, 80, 95, 100],
                labels=["A", "B", "C"],
                include_lowest=True
            )
            abc_class = abc_class_series.to_dict()

        xyz_class: Dict[str, str] = {}
        if stocks_mov is not None and not stocks_mov.empty:
            grouped = stocks_mov.copy()
            grouped["SKU"] = grouped.get("SKU", grouped.get("sku", "")).astype(str)
            # Otimização: calcular mean e std em batch usando groupby agg
            if "saidas" in grouped.columns:
                grouped_stats = grouped.groupby("SKU")["saidas"].agg(["mean", "std"]).fillna(0)
                for sku, row in grouped_stats.iterrows():
                    mean_val = float(row["mean"])
                    if mean_val <= 0:
                        xyz_class[sku] = "Z"
                        continue
                    std_val = float(row["std"])
                    cv = std_val / mean_val if mean_val else 1.0
                    if cv < 0.25:
                        xyz_class[sku] = "X"
                    elif cv < 0.5:
                        xyz_class[sku] = "Y"
                    else:
                        xyz_class[sku] = "Z"

        matrix = {
            "A": {"X": 0, "Y": 0, "Z": 0},
            "B": {"X": 0, "Y": 0, "Z": 0},
            "C": {"X": 0, "Y": 0, "Z": 0},
        }

        sku_rows: List[Dict[str, Any]] = []
        risk_entries: List[Dict[str, Any]] = []

        # Lazy ROP calculation: só calcular Monte Carlo completo se compute_rop=True
        # Por padrão, usar aproximação rápida para performance
        
        # Otimização: calcular ROP aproximado em batch usando vectorização
        df["ads_180_safe"] = df["ads_180"].fillna(0.0).clip(lower=0.1)
        df["mu_lt"] = df["ads_180_safe"] * 7  # demanda durante 7 dias
        df["safety_stock"] = df["mu_lt"] * 0.3
        df["rop_approx"] = df["mu_lt"] + df["safety_stock"]
        df["stock_atual_safe"] = df["stock_atual"].fillna(0.0)
        
        # Stockout probability aproximada (vectorizada)
        df["stockout_prob_approx"] = np.where(
            df["stock_atual_safe"] < df["rop_approx"],
            np.clip((df["rop_approx"] - df["stock_atual_safe"]) / df["rop_approx"] * 0.5, 0.01, 0.5),
            0.01
        )
        
        # Gap e risk_score (vectorizados)
        df["gap"] = df["rop_approx"] - df["stock_atual_safe"]
        df["risk_score"] = np.maximum(df["gap"], 0.0) * df["stockout_prob_approx"]
        
        # Iterar apenas para construir estruturas finais e calcular ROP completo se necessário
        for idx, row in df.iterrows():
            sku = str(row.get("SKU", ""))
            stock_atual = float(row.get("stock_atual_safe", 0.0))
            ads_180 = float(row.get("ads_180_safe", 0.1))
            cobertura = float(row.get("cobertura_dias", 0.0) or 0.0)
            rop_approx = float(row.get("rop_approx", 0.0))
            stockout_prob_approx = float(row.get("stockout_prob_approx", 0.01))
            gap = float(row.get("gap", 0.0))
            risk_score = float(row.get("risk_score", 0.0))

            # Só calcular ROP completo se compute_rop=True
            if compute_rop:
                predictor.update_demand(sku, ads_180)
                rop_result = predictor.calculate_rop(sku, lead_time=7, service_level=0.95)
                rop = float(rop_result.get("rop", 0.0))
                stockout_prob = float(rop_result.get("stockout_prob", 0.0))
            else:
                # Usar aproximação (já calculada)
                rop = rop_approx
                stockout_prob = stockout_prob_approx

            abc = abc_class.get(sku, "C")
            xyz = xyz_class.get(sku, "Z")
            matrix.setdefault(abc, {}).setdefault(xyz, 0)
            matrix[abc][xyz] += 1

            if gap > 0:
                acao = "Comprar agora"
            elif cobertura > 90:
                acao = "Excesso"
            else:
                acao = "Acompanhar"

            insight = (
                f"Classe {abc}{xyz}. Cobertura {cobertura:.0f} dias. "
                f"Prob. ruptura {stockout_prob * 100:.1f}%."
            )

            sku_entry = {
                "sku": sku,
                "classe": f"{abc}{xyz}",
                "abc": abc,
                "xyz": xyz,
                "stock_atual": round(stock_atual, 1),
                "ads_180": round(ads_180, 2),
                "cobertura_dias": round(cobertura, 1),
                "rop": round(rop, 1),
                "risco_30d": round(stockout_prob * 100, 1),
                "risk_score": round(risk_score, 2),
                "acao": acao,
                "insight": insight,
            }
            sku_rows.append(sku_entry)

            risk_entries.append(
                {
                    "sku": sku,
                    "risk_score": risk_score,
                    "probability": round(stockout_prob * 100, 1),
                    "impact": round(max(gap, 0.0), 1),
                    "classe": f"{abc}{xyz}",
                    "message": (
                        f"SKU {sku} classe {abc}{xyz} com cobertura {cobertura:.0f} dias "
                        f"e risco {stockout_prob * 100:.1f}% de rutura."
                    ),
                }
            )

        total_stock = float(df["stock_atual"].sum())
        coverage_series = df["cobertura_dias"].replace({np.inf: np.nan})
        avg_coverage = float(coverage_series.mean(skipna=True) or 0.0)
        global_risk = float(np.mean([item["risk_score"] for item in risk_entries]) if risk_entries else 0.0)

        top_risks = sorted(risk_entries, key=lambda item: item["risk_score"], reverse=True)[:10]

        insights = {
            "kpis": {
                "total_stock": round(total_stock, 1),
                "average_coverage_days": round(avg_coverage, 1),
                "global_risk_score": round(global_risk, 2),
                "sku_count": len(sku_rows),
            },
            "matrix": matrix,
            "skus": sku_rows,
            "top_risks": top_risks,
            "generated_at": datetime.utcnow().isoformat(),
        }

        return insights

    def get_inventory_insights(self, recalculate_rop: bool = False) -> Dict[str, Any]:
        """
        Retorna insights de inventário.
        Se recalculate_rop=True, recalcula ROP completo (Monte Carlo) para todos os SKUs.
        """
        if not hasattr(self, "_inventory_insights"):
            self._inventory_insights = self._default_inventory_insights()
        if not self._inventory_insights or not self._inventory_insights.get("skus"):
            # Recalcular com ROP completo se solicitado
            self._inventory_insights = self._compute_inventory_insights(compute_rop=recalculate_rop)
        elif recalculate_rop:
            # Recalcular apenas ROP para SKUs existentes
            self._inventory_insights = self._compute_inventory_insights(compute_rop=True)
        return self._inventory_insights


_loader: Optional[DataLoader] = None


def get_loader() -> DataLoader:
    global _loader  # pylint: disable=global-statement
    if _loader is None:
        data_dir = Path(__file__).parent.parent / "data"
        _loader = DataLoader(data_dir=data_dir)
    return _loader


def run_startup_etl() -> Dict[str, int]:
    return get_loader().process_existing_files()


def process_uploaded_files(saved_files: List[Path]) -> Dict[str, int]:
    return get_loader().process_uploaded_files(saved_files)


def get_etl_status() -> Dict[str, Any]:
    return get_loader().get_status()

