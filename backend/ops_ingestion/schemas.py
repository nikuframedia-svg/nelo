"""
════════════════════════════════════════════════════════════════════════════════
OPS INGESTION SCHEMAS - Pydantic Models para Validação de Dados Excel
════════════════════════════════════════════════════════════════════════════════

Contract 14: Schemas Pydantic para validação de linhas Excel

Schemas:
- OrderRowSchema: Validação de linha de ordem
- InventoryMoveRowSchema: Validação de linha de movimento
- HRRowSchema: Validação de linha de RH
- MachineRowSchema: Validação de linha de máquina
"""

from __future__ import annotations

from datetime import datetime, date
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, validator, field_validator
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class MovementTypeEnum(str, Enum):
    """Tipo de movimento de inventário."""
    TRANSFER = "TRANSFER"
    GOOD_OUTPUT = "GOOD_OUTPUT"
    SCRAP_OUTPUT = "SCRAP_OUTPUT"
    HOLD = "HOLD"
    OTHER = "OTHER"


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class OrderRowSchema(BaseModel):
    """
    Schema para validação de linha de ordem de produção.
    
    As specified in Contract 14:
    - external_order_code: Código externo (obrigatório)
    - product_code: Código do produto (obrigatório)
    - quantity: Quantidade (obrigatório, > 0)
    - due_date: Data de entrega (opcional)
    - routing: Lista de operações (opcional, JSON ou string)
    - line_or_center: Linha ou centro (opcional)
    """
    
    external_order_code: str = Field(..., description="Código externo da ordem")
    product_code: str = Field(..., description="Código do produto")
    quantity: float = Field(..., gt=0, description="Quantidade (deve ser > 0)")
    due_date: Optional[date] = Field(None, description="Data de entrega")
    routing: Optional[Union[str, List[Dict[str, Any]]]] = Field(None, description="Routing (JSON string ou lista)")
    line_or_center: Optional[str] = Field(None, description="Linha ou centro de produção")
    
    @field_validator('routing', mode='before')
    @classmethod
    def parse_routing(cls, v):
        """Parse routing from string or list."""
        if v is None:
            return None
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                import json
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return None
        return None
    
    class Config:
        use_enum_values = True


class InventoryMoveRowSchema(BaseModel):
    """
    Schema para validação de linha de movimento de inventário.
    
    As specified in Contract 14:
    - order_code: Código da ordem (obrigatório)
    - from_station: Estação de origem (opcional)
    - to_station: Estação de destino (opcional)
    - movement_type: Tipo de movimento (obrigatório, enum)
    - quantity_good: Quantidade de peças boas (opcional, >= 0)
    - quantity_scrap: Quantidade de refugo (opcional, >= 0)
    - timestamp: Data/hora do movimento (obrigatório)
    """
    
    order_code: str = Field(..., description="Código da ordem")
    from_station: Optional[str] = Field(None, description="Estação de origem")
    to_station: Optional[str] = Field(None, description="Estação de destino")
    movement_type: MovementTypeEnum = Field(MovementTypeEnum.TRANSFER, description="Tipo de movimento")
    quantity_good: Optional[float] = Field(None, ge=0, description="Quantidade de peças boas")
    quantity_scrap: Optional[float] = Field(None, ge=0, description="Quantidade de refugo")
    timestamp: datetime = Field(..., description="Data/hora do movimento")
    
    @field_validator('movement_type', mode='before')
    @classmethod
    def parse_movement_type(cls, v):
        """Parse movement_type from string."""
        if isinstance(v, str):
            v_upper = v.upper()
            for mt in MovementTypeEnum:
                if mt.value == v_upper or v_upper in mt.value:
                    return mt
            # Fallback: tentar mapear variações comuns
            if "TRANSFER" in v_upper or "TRANSFERÊNCIA" in v_upper:
                return MovementTypeEnum.TRANSFER
            if "GOOD" in v_upper or "BOA" in v_upper:
                return MovementTypeEnum.GOOD_OUTPUT
            if "SCRAP" in v_upper or "REFUGO" in v_upper:
                return MovementTypeEnum.SCRAP_OUTPUT
            if "HOLD" in v_upper or "ESPERA" in v_upper:
                return MovementTypeEnum.HOLD
        return MovementTypeEnum.OTHER
    
    class Config:
        use_enum_values = True


class HRRowSchema(BaseModel):
    """
    Schema para validação de linha de RH.
    
    As specified in Contract 14:
    - technician_code: Código do técnico (obrigatório)
    - name: Nome (obrigatório)
    - role: Função/cargo (opcional)
    - skills: Skills matrix (opcional, JSON ou dict)
    - shift_pattern: Padrão de turnos (opcional)
    - home_cell: Célula base (opcional)
    """
    
    technician_code: str = Field(..., description="Código do técnico")
    name: str = Field(..., description="Nome")
    role: Optional[str] = Field(None, description="Função/cargo")
    skills: Optional[Union[str, Dict[str, float]]] = Field(None, description="Skills matrix (JSON string ou dict)")
    shift_pattern: Optional[str] = Field(None, description="Padrão de turnos")
    home_cell: Optional[str] = Field(None, description="Célula base")
    
    @field_validator('skills', mode='before')
    @classmethod
    def parse_skills(cls, v):
        """Parse skills from string or dict."""
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                import json
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return None
        return None
    
    class Config:
        use_enum_values = True


class MachineRowSchema(BaseModel):
    """
    Schema para validação de linha de máquina.
    
    As specified in Contract 14:
    - machine_code: Código da máquina (obrigatório)
    - description: Descrição (opcional)
    - line: Linha de produção (opcional)
    - capacity_per_shift_hours: Capacidade por turno (obrigatório, > 0)
    - avg_setup_time_minutes: Tempo médio de setup (opcional, >= 0)
    - maintenance_windows: Janelas de manutenção (opcional, JSON ou lista)
    """
    
    machine_code: str = Field(..., description="Código da máquina")
    description: Optional[str] = Field(None, description="Descrição")
    line: Optional[str] = Field(None, description="Linha de produção")
    capacity_per_shift_hours: float = Field(..., gt=0, description="Capacidade por turno (horas)")
    avg_setup_time_minutes: Optional[float] = Field(None, ge=0, description="Tempo médio de setup (minutos)")
    maintenance_windows: Optional[Union[str, List[Dict[str, Any]]]] = Field(None, description="Janelas de manutenção (JSON ou lista)")
    
    @field_validator('maintenance_windows', mode='before')
    @classmethod
    def parse_maintenance_windows(cls, v):
        """Parse maintenance_windows from string or list."""
        if v is None:
            return None
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                import json
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return None
        return None
    
    class Config:
        use_enum_values = True


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class ImportResult(BaseModel):
    """Resultado de uma importação."""
    success: bool
    imported_count: int = 0
    failed_count: int = 0
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    record_ids: List[int] = Field(default_factory=list)
    source_file: str

