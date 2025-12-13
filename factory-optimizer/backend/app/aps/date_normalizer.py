"""
Normalizador de Datas - Converte datas humanas para ISO 8601

Camada intermédia que normaliza datas antes de enviar para o backend.
Rejeita datas não ISO e converte datas humanas automaticamente.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def normalize_date(date_str: str, reference_time: Optional[datetime] = None) -> Optional[str]:
    """
    Normaliza uma string de data para formato ISO 8601.
    
    Aceita:
    - ISO 8601: "2025-11-19T14:00:00"
    - Datas humanas: "hoje", "amanhã", "esta tarde", "das 14h às 18h", etc.
    
    Args:
        date_str: String de data a normalizar
        reference_time: Tempo de referência (default: agora UTC)
    
    Returns:
        String ISO 8601 ou None se não conseguir normalizar
    """
    if not date_str or not isinstance(date_str, str):
        return None
    
    date_str = date_str.strip()
    
    # Se já é ISO, validar e retornar
    if _is_iso_format(date_str):
        try:
            datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return date_str
        except (ValueError, AttributeError):
            pass
    
    # Normalizar datas humanas
    reference = reference_time or datetime.utcnow()
    
    # "hoje" -> data atual
    if date_str.lower() in ["hoje", "today", "agora", "now"]:
        return reference.isoformat()
    
    # "amanhã" -> data atual + 1 dia
    if date_str.lower() in ["amanhã", "amanha", "tomorrow"]:
        return (reference + timedelta(days=1)).isoformat()
    
    # "esta tarde" -> 14:00-18:00 hoje
    if "tarde" in date_str.lower() or "afternoon" in date_str.lower():
        start = reference.replace(hour=14, minute=0, second=0, microsecond=0)
        if reference.hour >= 18:
            start = start + timedelta(days=1)
        return start.isoformat()
    
    # "das Xh às Yh" -> extrair horas
    time_range_match = re.search(r'das?\s+(\d{1,2})h?\s+às?\s+(\d{1,2})h?', date_str.lower())
    if time_range_match:
        start_hour = int(time_range_match.group(1))
        end_hour = int(time_range_match.group(2))
        
        start = reference.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        if start < reference:
            start = start + timedelta(days=1)
        
        return start.isoformat()
    
    # "às Xh" -> hora específica hoje
    time_match = re.search(r'às?\s+(\d{1,2})h?', date_str.lower())
    if time_match:
        hour = int(time_match.group(1))
        result = reference.replace(hour=hour, minute=0, second=0, microsecond=0)
        if result < reference:
            result = result + timedelta(days=1)
        return result.isoformat()
    
    # Tentar parsear como data relativa (ex: "em 2 horas", "daqui a 3 dias")
    relative_match = re.search(r'(\d+)\s*(hora|horas|dia|dias|semana|semanas)', date_str.lower())
    if relative_match:
        amount = int(relative_match.group(1))
        unit = relative_match.group(2)
        
        if "hora" in unit:
            delta = timedelta(hours=amount)
        elif "dia" in unit:
            delta = timedelta(days=amount)
        elif "semana" in unit:
            delta = timedelta(weeks=amount)
        else:
            return None
        
        return (reference + delta).isoformat()
    
    logger.warning(f"Não foi possível normalizar data: '{date_str}'")
    return None


def normalize_time_range(
    start_str: Optional[str],
    end_str: Optional[str],
    reference_time: Optional[datetime] = None,
    default_duration_hours: int = 4,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Normaliza um intervalo de tempo para ISO 8601.
    
    Args:
        start_str: String de início
        end_str: String de fim
        reference_time: Tempo de referência (default: agora UTC)
        default_duration_hours: Duração padrão se só tiver start
    
    Returns:
        Tupla (start_iso, end_iso) ou (None, None) se não conseguir normalizar
    """
    reference = reference_time or datetime.utcnow()
    
    # Normalizar start
    start_iso = None
    if start_str:
        start_iso = normalize_date(start_str, reference)
    else:
        start_iso = reference.isoformat()
    
    # Normalizar end
    end_iso = None
    if end_str:
        end_iso = normalize_date(end_str, reference)
    else:
        # Se não tiver end, usar start + default_duration
        if start_iso:
            try:
                start_dt = datetime.fromisoformat(start_iso.replace('Z', '+00:00'))
                end_dt = start_dt + timedelta(hours=default_duration_hours)
                end_iso = end_dt.isoformat()
            except (ValueError, AttributeError):
                pass
    
    # Validar que end > start
    if start_iso and end_iso:
        try:
            start_dt = datetime.fromisoformat(start_iso.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_iso.replace('Z', '+00:00'))
            
            if end_dt <= start_dt:
                logger.warning(f"end_time <= start_time. Ajustando end_time para start_time + {default_duration_hours}h")
                end_dt = start_dt + timedelta(hours=default_duration_hours)
                end_iso = end_dt.isoformat()
        except (ValueError, AttributeError) as e:
            logger.error(f"Erro ao validar intervalo: {e}")
            return None, None
    
    return start_iso, end_iso


def _is_iso_format(date_str: str) -> bool:
    """Verifica se uma string está em formato ISO 8601."""
    iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
    return bool(re.match(iso_pattern, date_str))


def validate_iso_date(date_str: str) -> bool:
    """
    Valida se uma string é uma data ISO 8601 válida.
    
    Args:
        date_str: String a validar
    
    Returns:
        True se for ISO válida, False caso contrário
    """
    if not date_str or not isinstance(date_str, str):
        return False
    
    if not _is_iso_format(date_str):
        return False
    
    try:
        datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return True
    except (ValueError, AttributeError):
        return False

