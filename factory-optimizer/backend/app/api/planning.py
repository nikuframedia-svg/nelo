import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.aps.scheduler import APSScheduler

router = APIRouter()


logger = logging.getLogger(__name__)

# Cache para planos APS (evita recalcular a cada carregamento)
_plan_cache_dir = Path(__file__).parent.parent.parent / "data" / "plan_cache"
_plan_cache_dir.mkdir(parents=True, exist_ok=True)
_plan_cache_memory: dict = {}  # Cache em memória para acesso rápido


def _get_plan_cache_key(start_date: str, end_date: str, cell: Optional[str] = None) -> str:
    """Gera chave de cache baseada em datas e cell."""
    key_str = f"{start_date}:{end_date}:{cell or ''}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cached_plan(start_date: str, end_date: str, cell: Optional[str] = None) -> Optional[dict]:
    """Recupera plano do cache."""
    cache_key = _get_plan_cache_key(start_date, end_date, cell)
    
    # Tentar memória primeiro
    if cache_key in _plan_cache_memory:
        return _plan_cache_memory[cache_key]
    
    # Tentar disco
    cache_path = _plan_cache_dir / f"{cache_key}.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Carregar em memória
                _plan_cache_memory[cache_key] = data
                return data
        except Exception as exc:
            logger.warning(f"Erro ao ler cache de plano: {exc}")
    
    return None


def _set_cached_plan(start_date: str, end_date: str, cell: Optional[str], plan_data: dict):
    """Guarda plano no cache."""
    cache_key = _get_plan_cache_key(start_date, end_date, cell)
    _plan_cache_memory[cache_key] = plan_data
    
    cache_path = _plan_cache_dir / f"{cache_key}.json"
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(plan_data, f, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        logger.warning(f"Erro ao escrever cache de plano: {exc}")


DEFAULT_KPIS = {
    "otd_pct": 0.0,
    "lead_time_h": 0.0,
    "gargalo_ativo": "N/A",
    "horas_setup_semana": 0.0,
}


def _plan_block() -> dict:
    return {
        "kpis": DEFAULT_KPIS.copy(),
        "operations": [],
        "explicacoes": [],
    }


def _empty_payload() -> dict:
    return {"antes": _plan_block(), "depois": _plan_block()}


@router.get("/plano")
async def get_plan(
    start_date: str = Query(..., description="YYYY-MM-DD"),
    end_date: str = Query(..., description="YYYY-MM-DD"),
    cell: Optional[str] = Query(None, description="Célula/Linha")
):
    """Retorna plano Antes e Depois"""
    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Datas inválidas.") from exc

    if end < start:
        raise HTTPException(status_code=400, detail="end_date deve ser >= start_date")

    scheduler = APSScheduler()
    loader = scheduler.loader

    roteiros = loader.get_roteiros()
    required_route_cols = {
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
    if roteiros is None or roteiros.empty:
        logger.warning("Planeamento indisponível: roteiros vazios.")
        return _empty_payload()

    missing_route_cols = sorted(required_route_cols.difference(roteiros.columns))
    if missing_route_cols:
        logger.warning("Planeamento indisponível: roteiros sem colunas %s", missing_route_cols)
        return _empty_payload()

    ordens = loader.get_ordens()
    if ordens is None or ordens.empty:
        logger.info("Planeamento vazio: nenhuma ordem carregada.")
        return _empty_payload()

    data_prometida = pd.to_datetime(ordens.get("data_prometida"), errors="coerce")
    window_mask = data_prometida.notna() & (data_prometida >= start) & (data_prometida <= end)
    if not window_mask.any():
        logger.info(
            "Planeamento vazio: nenhuma ordem entre %s e %s.",
            start_date,
            end_date,
        )
        return _empty_payload()

    # ✅ Verificar cache primeiro (performance)
    cached_plan = _get_cached_plan(start_date, end_date, cell)
    if cached_plan:
        logger.info(f"Cache hit para plano {start_date}-{end_date}")
        return cached_plan

    # Cache miss - gerar planos
    logger.info(f"Cache miss para plano {start_date}-{end_date}, gerando...")
    try:
        plano_antes = scheduler.generate_baseline_plan(start, end, cell)
        plano_depois = scheduler.generate_optimized_plan(start, end, cell)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Falha ao gerar plano para %s-%s", start_date, end_date)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def serialize(plan) -> dict:
        if plan is None:
            return _plan_block()

        block = _plan_block()
        block["kpis"].update(plan.kpis or {})

        operations = []
        for op in getattr(plan, "operations", []) or []:
            start_ts = getattr(op, "start_time", None)
            end_ts = getattr(op, "end_time", None)
            operations.append(
                {
                    "ordem": str(getattr(op, "ordem", "")),
                    "artigo": str(getattr(op, "artigo", "")),
                    "operacao": str(getattr(op, "operacao", "")),
                    "recurso": str(getattr(op, "recurso", "")),
                    "start_time": start_ts.isoformat() if start_ts else None,
                    "end_time": end_ts.isoformat() if end_ts else None,
                    "setor": str(getattr(op, "setor", "")),
                    "overlap": float(getattr(op, "overlap", 0.0) or 0.0),
                    "rota": str(getattr(op, "rota", "")),
                    "explicacao": str(getattr(op, "explicacao", "")),
                }
            )
        block["operations"] = operations
        block["explicacoes"] = list(getattr(plan, "explicacoes", []) or [])
        return block

    result = {"antes": serialize(plano_antes), "depois": serialize(plano_depois)}
    
    # ✅ Guardar no cache
    _set_cached_plan(start_date, end_date, cell, result)
    
    return result


@router.post("/plano/aplicar")
async def apply_plan(payload: Optional[dict] = None):
    """Define o plano otimizado atual como novo baseline."""
    scheduler = APSScheduler()
    loader = scheduler.loader

    committed_at = datetime.utcnow().isoformat()
    loader.status.setdefault("baseline_history", [])
    loader.status["baseline_history"].append(
        {
            "committed_at": committed_at,
            "context": payload or {},
        }
    )
    ready_flags = loader.status.setdefault("ready_flags", {})
    ready_flags["planning_ready"] = True
    loader.status["last_baseline_commit"] = committed_at
    loader._save_status_to_disk()

    return {"status": "ok", "committed_at": committed_at}

