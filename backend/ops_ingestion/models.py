"""
════════════════════════════════════════════════════════════════════════════════
OPS INGESTION MODELS - Tabelas Raw para Ingestão de Dados Operacionais
════════════════════════════════════════════════════════════════════════════════

Contract 14: Operational Data Engine

Tabelas:
- ops_raw_orders: Ordens de produção brutas do Excel
- ops_raw_inventory_moves: Movimentos internos de inventário/WIP
- ops_raw_hr: Recursos Humanos
- ops_raw_machines: Máquinas e Linhas

Estas tabelas são ADITIVAS - não substituem dados existentes.
"""

from __future__ import annotations

import json
from datetime import datetime, date
from enum import Enum
from typing import Optional, Dict, Any

from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, Enum as SQLEnum, Index
from sqlalchemy.orm import relationship

# Import Base and engine from duplios.models to use the same database
try:
    from duplios.models import Base, engine
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy import create_engine
    import os
    Base = declarative_base()
    DATABASE_URL = os.getenv("DUPLIOS_DATABASE_URL", "sqlite:///duplios.db")
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})

# Initialize tables on import
def init_ops_ingestion_tables():
    """Initialize Ops Ingestion tables."""
    try:
        # Use checkfirst=True to avoid errors if tables already exist
        Base.metadata.create_all(bind=engine, checkfirst=True)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to create Ops Ingestion tables: {e}")

# Create tables when module is imported
init_ops_ingestion_tables()


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class MovementType(str, Enum):
    """Tipo de movimento de inventário."""
    TRANSFER = "TRANSFER"          # Transferência entre estações
    GOOD_OUTPUT = "GOOD_OUTPUT"    # Saída de peças boas
    SCRAP_OUTPUT = "SCRAP_OUTPUT"  # Saída de refugo
    HOLD = "HOLD"                  # Retenção/espera
    OTHER = "OTHER"                 # Outro tipo


# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class OpsRawOrder(Base):
    """
    Ordens de produção brutas do Excel.
    
    As specified in Contract 14:
    - external_order_code: Código externo da ordem
    - product_code: Código do produto
    - quantity: Quantidade
    - due_date: Data de entrega
    - routing_json: Lista de operações e tempos (JSON)
    - line_or_center: Linha ou centro de produção
    - source_file: Nome do ficheiro Excel
    - imported_at: Data/hora de importação
    """
    __tablename__ = "ops_raw_orders"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Dados da ordem
    external_order_code = Column(String(100), nullable=False, index=True)
    product_code = Column(String(100), nullable=False, index=True)
    quantity = Column(Float, nullable=False)
    due_date = Column(Date, nullable=True, index=True)
    
    # Routing (sequência de operações)
    routing_json = Column(Text, nullable=True)  # JSON: [{"operation": "OP1", "machine": "M1", "time_min": 30}, ...]
    
    # Localização
    line_or_center = Column(String(100), nullable=True)
    
    # Metadados de importação
    source_file = Column(String(255), nullable=False)
    imported_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Flags de qualidade
    quality_flags = Column(Text, nullable=True)  # JSON: {"warnings": [...], "errors": [...]}
    
    __table_args__ = (
        Index('ix_ops_raw_orders_external_code', 'external_order_code'),
        Index('ix_ops_raw_orders_product', 'product_code'),
        Index('ix_ops_raw_orders_due_date', 'due_date'),
    )
    
    def get_routing(self) -> list[Dict[str, Any]]:
        """Parse routing_json to list."""
        if not self.routing_json:
            return []
        try:
            return json.loads(self.routing_json)
        except (json.JSONDecodeError, TypeError):
            return []
    
    def set_routing(self, routing: list[Dict[str, Any]]) -> None:
        """Set routing_json from list."""
        self.routing_json = json.dumps(routing) if routing else None


class OpsRawInventoryMove(Base):
    """
    Movimentos internos de inventário/WIP do Excel.
    
    As specified in Contract 14:
    - order_code: Código da ordem
    - from_station: Estação de origem
    - to_station: Estação de destino
    - movement_type: Tipo de movimento (enum)
    - quantity_good: Quantidade de peças boas
    - quantity_scrap: Quantidade de refugo
    - timestamp: Data/hora do movimento
    - source_file: Nome do ficheiro Excel
    - imported_at: Data/hora de importação
    """
    __tablename__ = "ops_raw_inventory_moves"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Dados do movimento
    order_code = Column(String(100), nullable=False, index=True)
    from_station = Column(String(100), nullable=True)
    to_station = Column(String(100), nullable=True, index=True)
    movement_type = Column(SQLEnum(MovementType), nullable=False, default=MovementType.TRANSFER)
    
    # Quantidades
    quantity_good = Column(Float, nullable=True)
    quantity_scrap = Column(Float, nullable=True)
    
    # Timestamp
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Metadados de importação
    source_file = Column(String(255), nullable=False)
    imported_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Flags de qualidade
    quality_flags = Column(Text, nullable=True)  # JSON: {"warnings": [...], "errors": [...]}
    
    __table_args__ = (
        Index('ix_ops_raw_moves_order', 'order_code'),
        Index('ix_ops_raw_moves_timestamp', 'timestamp'),
        Index('ix_ops_raw_moves_to_station', 'to_station'),
    )


class OpsRawHR(Base):
    """
    Recursos Humanos do Excel.
    
    As specified in Contract 14:
    - technician_code: Código do técnico
    - name: Nome
    - role: Função/cargo
    - skills_json: Skills matrix (JSON)
    - shift_pattern: Padrão de turnos
    - home_cell: Célula/célula base
    - source_file: Nome do ficheiro Excel
    - imported_at: Data/hora de importação
    """
    __tablename__ = "ops_raw_hr"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Dados do colaborador
    technician_code = Column(String(100), nullable=False, unique=True, index=True)
    name = Column(String(255), nullable=False)
    role = Column(String(100), nullable=True)
    
    # Skills matrix
    skills_json = Column(Text, nullable=True)  # JSON: {"welding": 0.8, "cnc": 0.4, ...}
    
    # Turnos
    shift_pattern = Column(String(100), nullable=True)  # ex: "M-F 08:00-16:00"
    
    # Localização
    home_cell = Column(String(100), nullable=True)
    
    # Metadados de importação
    source_file = Column(String(255), nullable=False)
    imported_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Flags de qualidade
    quality_flags = Column(Text, nullable=True)  # JSON: {"warnings": [...], "errors": [...]}
    
    def get_skills(self) -> Dict[str, float]:
        """Parse skills_json to dict."""
        if not self.skills_json:
            return {}
        try:
            return json.loads(self.skills_json)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def set_skills(self, skills: Dict[str, float]) -> None:
        """Set skills_json from dict."""
        self.skills_json = json.dumps(skills) if skills else None


class OpsRawMachine(Base):
    """
    Máquinas e Linhas do Excel.
    
    As specified in Contract 14:
    - machine_code: Código da máquina
    - description: Descrição
    - line: Linha de produção
    - capacity_per_shift_hours: Capacidade por turno (horas)
    - avg_setup_time_minutes: Tempo médio de setup (minutos)
    - maintenance_windows_json: Janelas de manutenção (JSON)
    - source_file: Nome do ficheiro Excel
    - imported_at: Data/hora de importação
    """
    __tablename__ = "ops_raw_machines"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Dados da máquina
    machine_code = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(String(255), nullable=True)
    line = Column(String(100), nullable=True, index=True)
    
    # Capacidade
    capacity_per_shift_hours = Column(Float, nullable=False)
    avg_setup_time_minutes = Column(Float, nullable=True)
    
    # Manutenção
    maintenance_windows_json = Column(Text, nullable=True)  # JSON: [{"start": "...", "end": "...", "type": "..."}, ...]
    
    # Metadados de importação
    source_file = Column(String(255), nullable=False)
    imported_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Flags de qualidade
    quality_flags = Column(Text, nullable=True)  # JSON: {"warnings": [...], "errors": [...]}
    
    __table_args__ = (
        Index('ix_ops_raw_machines_code', 'machine_code'),
        Index('ix_ops_raw_machines_line', 'line'),
    )
    
    def get_maintenance_windows(self) -> list[Dict[str, Any]]:
        """Parse maintenance_windows_json to list."""
        if not self.maintenance_windows_json:
            return []
        try:
            return json.loads(self.maintenance_windows_json)
        except (json.JSONDecodeError, TypeError):
            return []
    
    def set_maintenance_windows(self, windows: list[Dict[str, Any]]) -> None:
        """Set maintenance_windows_json from list."""
        self.maintenance_windows_json = json.dumps(windows) if windows else None


# ═══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY FLAGS
# ═══════════════════════════════════════════════════════════════════════════════

class OpsDataQualityFlag(Base):
    """
    Flags de qualidade de dados.
    
    Armazena warnings e erros detectados durante a importação.
    """
    __tablename__ = "ops_data_quality_flags"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Referência ao registo
    table_name = Column(String(50), nullable=False, index=True)  # "ops_raw_orders", "ops_raw_inventory_moves", etc.
    record_id = Column(Integer, nullable=False, index=True)
    
    # Tipo de flag
    flag_type = Column(String(20), nullable=False)  # "WARNING", "ERROR", "ANOMALY"
    
    # Detalhes
    field_name = Column(String(100), nullable=True)
    message = Column(String(500), nullable=False)
    severity = Column(Integer, default=1)  # 1=low, 2=medium, 3=high
    
    # Metadados
    detected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    detected_by = Column(String(50), nullable=True)  # "data_quality", "ml_anomaly", etc.
    
    __table_args__ = (
        Index('ix_ops_quality_flags_table_record', 'table_name', 'record_id'),
        Index('ix_ops_quality_flags_type', 'flag_type'),
    )

