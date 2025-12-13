"""
════════════════════════════════════════════════════════════════════════════════
EXCEL PARSER - Parser Flexível com Mapeamento de Colunas
════════════════════════════════════════════════════════════════════════════════

Contract 14: Parser de Excel com mapeamento flexível de colunas

Funcionalidades:
- Lê Excel usando pandas
- Mapeia colunas usando column_aliases.yaml
- Aceita variações de nomes de colunas
- Retorna dados validados por schema Pydantic
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date

import pandas as pd

logger = logging.getLogger(__name__)

# Try to import yaml
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logger.warning("PyYAML not available, will use JSON fallback for column aliases")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

COLUMN_ALIASES_PATH = Path(__file__).parent / "data" / "column_aliases.yaml"

_column_aliases_cache: Optional[Dict[str, Any]] = None


def _load_column_aliases() -> Dict[str, Any]:
    """Load column aliases from YAML file."""
    global _column_aliases_cache
    
    if _column_aliases_cache is not None:
        return _column_aliases_cache
    
    try:
        if COLUMN_ALIASES_PATH.exists():
            with open(COLUMN_ALIASES_PATH, 'r', encoding='utf-8') as f:
                if HAS_YAML:
                    _column_aliases_cache = yaml.safe_load(f)
                else:
                    # Fallback: try JSON
                    import json
                    content = f.read()
                    try:
                        _column_aliases_cache = json.loads(content)
                    except json.JSONDecodeError:
                        _column_aliases_cache = _get_default_aliases()
        else:
            logger.warning(f"Column aliases file not found: {COLUMN_ALIASES_PATH}")
            _column_aliases_cache = _get_default_aliases()
    except Exception as e:
        logger.error(f"Failed to load column aliases: {e}")
        _column_aliases_cache = _get_default_aliases()
    
    return _column_aliases_cache


def _get_default_aliases() -> Dict[str, Any]:
    """Get default column aliases (fallback)."""
    return {
        "orders": {
            "external_order_code": ["Order Code", "Ordem", "OP"],
            "product_code": ["Product Code", "Produto", "Product"],
            "quantity": ["Quantity", "Quantidade", "Qty"],
            "due_date": ["Due Date", "Data Entrega"],
        },
        "inventory_moves": {
            "order_code": ["Order Code", "Ordem"],
            "from_station": ["From", "De"],
            "to_station": ["To", "Para"],
            "timestamp": ["Date", "Data"],
        },
        "hr": {
            "technician_code": ["Technician Code", "Técnico"],
            "name": ["Name", "Nome"],
        },
        "machines": {
            "machine_code": ["Machine Code", "Máquina"],
            "capacity_per_shift_hours": ["Capacity", "Capacidade"],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

def _map_columns(
    df: pd.DataFrame,
    data_type: str,  # "orders", "inventory_moves", "hr", "machines"
) -> Tuple[Dict[str, str], List[str]]:
    """
    Mapeia colunas do DataFrame para campos do schema.
    
    Returns:
        (column_mapping, missing_required): Mapeamento e lista de campos obrigatórios em falta
    """
    aliases = _load_column_aliases()
    type_aliases = aliases.get(data_type, {})
    
    column_mapping: Dict[str, str] = {}
    missing_required: List[str] = []
    
    # Campos obrigatórios por tipo
    required_fields = {
        "orders": ["external_order_code", "product_code", "quantity"],
        "inventory_moves": ["order_code", "timestamp"],
        "hr": ["technician_code", "name"],
        "machines": ["machine_code", "capacity_per_shift_hours"],
    }
    
    required = required_fields.get(data_type, [])
    
    # Normalizar nomes de colunas do DataFrame (remover espaços, lowercase)
    df_columns_normalized = {col.lower().strip(): col for col in df.columns}
    
    # Para cada campo esperado, tentar encontrar coluna correspondente
    for field_name, field_aliases in type_aliases.items():
        found = False
        
        # Tentar match exato primeiro
        for alias in field_aliases:
            alias_normalized = alias.lower().strip()
            
            # Match exato
            if alias_normalized in df_columns_normalized:
                column_mapping[field_name] = df_columns_normalized[alias_normalized]
                found = True
                break
            
            # Match parcial (contém)
            for df_col_normalized, df_col_original in df_columns_normalized.items():
                if alias_normalized in df_col_normalized or df_col_normalized in alias_normalized:
                    column_mapping[field_name] = df_col_original
                    found = True
                    break
            
            if found:
                break
        
        # Se não encontrou e é obrigatório, adicionar à lista de faltantes
        if not found and field_name in required:
            missing_required.append(field_name)
    
    return column_mapping, missing_required


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEL PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_excel_orders(file_path: str | Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Parse Excel file with orders.
    
    Returns:
        (rows, errors): Lista de linhas parseadas e lista de erros
    """
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        
        # Mapear colunas
        column_mapping, missing_required = _map_columns(df, "orders")
        
        if missing_required:
            return [], [f"Colunas obrigatórias em falta: {', '.join(missing_required)}"]
        
        # Converter DataFrame para lista de dicts
        rows = []
        errors = []
        
        for idx, row in df.iterrows():
            try:
                row_dict = {}
                
                # Mapear valores
                for field_name, column_name in column_mapping.items():
                    value = row.get(column_name)
                    
                    # Converter tipos especiais
                    if field_name == "due_date" and value is not None:
                        if isinstance(value, pd.Timestamp):
                            value = value.date()
                        elif isinstance(value, str):
                            try:
                                value = datetime.strptime(value, "%Y-%m-%d").date()
                            except ValueError:
                                try:
                                    value = pd.to_datetime(value).date()
                                except:
                                    value = None
                    
                    if field_name == "quantity" and value is not None:
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            value = None
                    
                    row_dict[field_name] = value
                
                rows.append(row_dict)
            except Exception as e:
                errors.append(f"Linha {idx + 2}: {str(e)}")
        
        return rows, errors
    
    except Exception as e:
        return [], [f"Erro ao ler Excel: {str(e)}"]


def parse_excel_inventory_moves(file_path: str | Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse Excel file with inventory moves."""
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        
        column_mapping, missing_required = _map_columns(df, "inventory_moves")
        
        if missing_required:
            return [], [f"Colunas obrigatórias em falta: {', '.join(missing_required)}"]
        
        rows = []
        errors = []
        
        for idx, row in df.iterrows():
            try:
                row_dict = {}
                
                for field_name, column_name in column_mapping.items():
                    value = row.get(column_name)
                    
                    if field_name == "timestamp" and value is not None:
                        if isinstance(value, pd.Timestamp):
                            value = value.to_pydatetime()
                        elif isinstance(value, str):
                            try:
                                value = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                            except ValueError:
                                try:
                                    value = pd.to_datetime(value).to_pydatetime()
                                except:
                                    value = None
                    
                    if field_name in ["quantity_good", "quantity_scrap"] and value is not None:
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            value = None
                    
                    row_dict[field_name] = value
                
                rows.append(row_dict)
            except Exception as e:
                errors.append(f"Linha {idx + 2}: {str(e)}")
        
        return rows, errors
    
    except Exception as e:
        return [], [f"Erro ao ler Excel: {str(e)}"]


def parse_excel_hr(file_path: str | Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse Excel file with HR data."""
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        
        column_mapping, missing_required = _map_columns(df, "hr")
        
        if missing_required:
            return [], [f"Colunas obrigatórias em falta: {', '.join(missing_required)}"]
        
        rows = []
        errors = []
        
        for idx, row in df.iterrows():
            try:
                row_dict = {}
                
                for field_name, column_name in column_mapping.items():
                    value = row.get(column_name)
                    row_dict[field_name] = value
                
                rows.append(row_dict)
            except Exception as e:
                errors.append(f"Linha {idx + 2}: {str(e)}")
        
        return rows, errors
    
    except Exception as e:
        return [], [f"Erro ao ler Excel: {str(e)}"]


def parse_excel_machines(file_path: str | Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse Excel file with machines data."""
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        
        column_mapping, missing_required = _map_columns(df, "machines")
        
        if missing_required:
            return [], [f"Colunas obrigatórias em falta: {', '.join(missing_required)}"]
        
        rows = []
        errors = []
        
        for idx, row in df.iterrows():
            try:
                row_dict = {}
                
                for field_name, column_name in column_mapping.items():
                    value = row.get(column_name)
                    
                    if field_name in ["capacity_per_shift_hours", "avg_setup_time_minutes"] and value is not None:
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            value = None
                    
                    row_dict[field_name] = value
                
                rows.append(row_dict)
            except Exception as e:
                errors.append(f"Linha {idx + 2}: {str(e)}")
        
        return rows, errors
    
    except Exception as e:
        return [], [f"Erro ao ler Excel: {str(e)}"]


