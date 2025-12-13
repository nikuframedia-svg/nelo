"""
════════════════════════════════════════════════════════════════════════════════════════════════════
MAINTENANCE MODELS - Modelos de Dados para Ordens de Manutenção
════════════════════════════════════════════════════════════════════════════════════════════════════

SQLAlchemy models e Pydantic schemas para gestão de ordens de trabalho de manutenção.

Tabelas:
- maintenance_work_orders: Ordens de manutenção
- maintenance_history: Histórico de manutenções realizadas

R&D / SIFIDE: WP1 - Digital Twin para manutenção preditiva
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import json

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class MaintenancePriority(str, Enum):
    """Prioridade da ordem de manutenção."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class MaintenanceType(str, Enum):
    """Tipo de manutenção."""
    PREDICTIVE = "PREDICTIVE"      # Baseada em condição (PredictiveCare)
    PREVENTIVE = "PREVENTIVE"      # Calendário/horas de uso
    CORRECTIVE = "CORRECTIVE"      # Reparação após falha
    IMPROVEMENT = "IMPROVEMENT"    # Melhoria/upgrade


class MaintenanceStatus(str, Enum):
    """Estado da ordem de manutenção."""
    DRAFT = "DRAFT"            # Rascunho
    OPEN = "OPEN"              # Aberta, aguardando planeamento
    PLANNED = "PLANNED"        # Planeada/Agendada
    IN_PROGRESS = "IN_PROGRESS"  # Em execução
    ON_HOLD = "ON_HOLD"        # Em espera (peças, técnico)
    COMPLETED = "COMPLETED"    # Concluída
    CANCELLED = "CANCELLED"    # Cancelada
    OVERDUE = "OVERDUE"        # Atrasada


class WorkOrderSource(str, Enum):
    """Origem da ordem de trabalho."""
    PREDICTIVECARE = "PREDICTIVECARE"  # Gerada automaticamente por PredictiveCare
    ZDM = "ZDM"                        # Gerada por Zero Defect Manufacturing
    MANUAL = "MANUAL"                  # Criada manualmente
    PREVENTIVE_SCHEDULE = "PREVENTIVE_SCHEDULE"  # Plano preventivo
    CMMS = "CMMS"                      # Sincronizada de CMMS externo


# ═══════════════════════════════════════════════════════════════════════════════
# SQLAlchemy MODELS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from sqlalchemy import (
        Column, Integer, String, Float, DateTime, Text, Boolean, 
        Enum as SQLEnum, ForeignKey, Index
    )
    from sqlalchemy.orm import relationship
    from sqlalchemy.ext.declarative import declarative_base
    
    # Use existing Base if available
    try:
        from models_common import Base
    except ImportError:
        Base = declarative_base()
    
    class MaintenanceWorkOrder(Base):
        """
        SQLAlchemy model for maintenance work orders.
        
        Representa uma ordem de trabalho de manutenção, desde a criação
        até a conclusão ou cancelamento.
        """
        __tablename__ = "maintenance_work_orders"
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        
        # Identification
        work_order_number = Column(String(32), unique=True, nullable=False, index=True)
        machine_id = Column(String(64), nullable=False, index=True)
        
        # Description
        title = Column(String(256), nullable=False)
        description = Column(Text, nullable=True)
        
        # Classification
        priority = Column(String(16), nullable=False, default=MaintenancePriority.MEDIUM.value)
        maintenance_type = Column(String(16), nullable=False, default=MaintenanceType.CORRECTIVE.value)
        status = Column(String(16), nullable=False, default=MaintenanceStatus.OPEN.value, index=True)
        source = Column(String(32), nullable=False, default=WorkOrderSource.MANUAL.value)
        
        # Scheduling
        suggested_start = Column(DateTime(timezone=True), nullable=True)
        suggested_end = Column(DateTime(timezone=True), nullable=True)
        scheduled_start = Column(DateTime(timezone=True), nullable=True)
        scheduled_end = Column(DateTime(timezone=True), nullable=True)
        actual_start = Column(DateTime(timezone=True), nullable=True)
        actual_end = Column(DateTime(timezone=True), nullable=True)
        
        # Duration
        estimated_duration_hours = Column(Float, nullable=True)
        actual_duration_hours = Column(Float, nullable=True)
        
        # PredictiveCare metrics (when source=PREDICTIVECARE)
        shi_at_creation = Column(Float, nullable=True)  # Health Index at creation time
        rul_at_creation = Column(Float, nullable=True)  # RUL at creation time (hours)
        risk_at_creation = Column(Float, nullable=True)  # Risk probability at creation
        
        # Assignment
        assigned_technician_id = Column(String(64), nullable=True)
        assigned_team = Column(String(64), nullable=True)
        
        # Spare parts
        spare_parts_json = Column(Text, nullable=True)  # JSON list of required parts
        spare_parts_ready = Column(Boolean, default=False)
        
        # CMMS Integration
        external_cmms_id = Column(String(128), nullable=True, index=True)
        external_cmms_system = Column(String(64), nullable=True)  # e.g., "SAP_PM", "Odoo"
        last_sync_at = Column(DateTime(timezone=True), nullable=True)
        
        # Completion
        resolution_notes = Column(Text, nullable=True)
        failure_code = Column(String(64), nullable=True)
        root_cause = Column(Text, nullable=True)
        
        # Timestamps
        created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
        updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), 
                          onupdate=lambda: datetime.now(timezone.utc))
        
        # Metadata
        metadata_json = Column(Text, nullable=True)
        
        # Indexes
        __table_args__ = (
            Index('ix_maint_machine_status', 'machine_id', 'status'),
            Index('ix_maint_scheduled', 'scheduled_start', 'status'),
            Index('ix_maint_priority_status', 'priority', 'status'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary."""
            return {
                "id": self.id,
                "work_order_number": self.work_order_number,
                "machine_id": self.machine_id,
                "title": self.title,
                "description": self.description,
                "priority": self.priority,
                "maintenance_type": self.maintenance_type,
                "status": self.status,
                "source": self.source,
                "suggested_start": self.suggested_start.isoformat() if self.suggested_start else None,
                "suggested_end": self.suggested_end.isoformat() if self.suggested_end else None,
                "scheduled_start": self.scheduled_start.isoformat() if self.scheduled_start else None,
                "scheduled_end": self.scheduled_end.isoformat() if self.scheduled_end else None,
                "actual_start": self.actual_start.isoformat() if self.actual_start else None,
                "actual_end": self.actual_end.isoformat() if self.actual_end else None,
                "estimated_duration_hours": self.estimated_duration_hours,
                "actual_duration_hours": self.actual_duration_hours,
                "shi_at_creation": self.shi_at_creation,
                "rul_at_creation": self.rul_at_creation,
                "risk_at_creation": self.risk_at_creation,
                "assigned_technician_id": self.assigned_technician_id,
                "assigned_team": self.assigned_team,
                "spare_parts": json.loads(self.spare_parts_json) if self.spare_parts_json else [],
                "spare_parts_ready": self.spare_parts_ready,
                "external_cmms_id": self.external_cmms_id,
                "external_cmms_system": self.external_cmms_system,
                "resolution_notes": self.resolution_notes,
                "failure_code": self.failure_code,
                "root_cause": self.root_cause,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            }
    
    class MaintenanceHistory(Base):
        """
        Historical record of maintenance activities.
        
        Registo imutável de manutenções concluídas para análise e KPIs.
        """
        __tablename__ = "maintenance_history"
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        work_order_id = Column(Integer, nullable=True)  # May be null for legacy data
        machine_id = Column(String(64), nullable=False, index=True)
        
        # Classification
        maintenance_type = Column(String(16), nullable=False)
        priority = Column(String(16), nullable=False)
        
        # Timing
        started_at = Column(DateTime(timezone=True), nullable=False)
        completed_at = Column(DateTime(timezone=True), nullable=False)
        duration_hours = Column(Float, nullable=False)
        
        # PredictiveCare metrics
        shi_before = Column(Float, nullable=True)
        shi_after = Column(Float, nullable=True)
        failure_prevented = Column(Boolean, default=False)  # Was this a prevented failure?
        
        # Outcome
        was_planned = Column(Boolean, default=True)
        was_successful = Column(Boolean, default=True)
        downtime_hours = Column(Float, nullable=True)
        cost_estimate = Column(Float, nullable=True)
        
        # Details
        work_performed = Column(Text, nullable=True)
        parts_replaced_json = Column(Text, nullable=True)
        technician_id = Column(String(64), nullable=True)
        
        # Timestamps
        created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
        
        __table_args__ = (
            Index('ix_maint_hist_machine_date', 'machine_id', 'completed_at'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                "id": self.id,
                "work_order_id": self.work_order_id,
                "machine_id": self.machine_id,
                "maintenance_type": self.maintenance_type,
                "priority": self.priority,
                "started_at": self.started_at.isoformat(),
                "completed_at": self.completed_at.isoformat(),
                "duration_hours": self.duration_hours,
                "shi_before": self.shi_before,
                "shi_after": self.shi_after,
                "failure_prevented": self.failure_prevented,
                "was_planned": self.was_planned,
                "was_successful": self.was_successful,
                "downtime_hours": self.downtime_hours,
                "cost_estimate": self.cost_estimate,
                "work_performed": self.work_performed,
                "parts_replaced": json.loads(self.parts_replaced_json) if self.parts_replaced_json else [],
                "technician_id": self.technician_id,
            }
    
    SQLALCHEMY_AVAILABLE = True

except ImportError:
    MaintenanceWorkOrder = None
    MaintenanceHistory = None
    SQLALCHEMY_AVAILABLE = False
    logger.warning("SQLAlchemy not available - using Pydantic models only")


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS (for API)
# ═══════════════════════════════════════════════════════════════════════════════

class WorkOrderCreate(BaseModel):
    """Schema for creating a work order."""
    machine_id: str = Field(..., description="Machine identifier")
    title: str = Field(..., max_length=256, description="Work order title")
    description: Optional[str] = Field(None, description="Detailed description")
    priority: MaintenancePriority = Field(MaintenancePriority.MEDIUM, description="Priority level")
    maintenance_type: MaintenanceType = Field(MaintenanceType.CORRECTIVE, description="Type of maintenance")
    estimated_duration_hours: Optional[float] = Field(None, ge=0, description="Estimated duration")
    scheduled_start: Optional[datetime] = Field(None, description="Scheduled start time")
    scheduled_end: Optional[datetime] = Field(None, description="Scheduled end time")
    assigned_technician_id: Optional[str] = Field(None, description="Assigned technician")
    spare_parts: List[str] = Field(default_factory=list, description="Required spare parts SKUs")
    
    class Config:
        json_schema_extra = {
            "example": {
                "machine_id": "MACH-001",
                "title": "Replace worn bearings",
                "description": "Bearing vibration exceeds threshold",
                "priority": "HIGH",
                "maintenance_type": "PREDICTIVE",
                "estimated_duration_hours": 4.0,
            }
        }


class WorkOrderUpdate(BaseModel):
    """Schema for updating a work order."""
    title: Optional[str] = Field(None, max_length=256)
    description: Optional[str] = None
    priority: Optional[MaintenancePriority] = None
    status: Optional[MaintenanceStatus] = None
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    assigned_technician_id: Optional[str] = None
    spare_parts_ready: Optional[bool] = None
    resolution_notes: Optional[str] = None
    failure_code: Optional[str] = None
    root_cause: Optional[str] = None


class WorkOrderResponse(BaseModel):
    """Schema for work order response."""
    id: int
    work_order_number: str
    machine_id: str
    title: str
    description: Optional[str]
    priority: str
    maintenance_type: str
    status: str
    source: str
    suggested_start: Optional[datetime]
    suggested_end: Optional[datetime]
    scheduled_start: Optional[datetime]
    scheduled_end: Optional[datetime]
    estimated_duration_hours: Optional[float]
    shi_at_creation: Optional[float]
    rul_at_creation: Optional[float]
    risk_at_creation: Optional[float]
    assigned_technician_id: Optional[str]
    spare_parts: List[str]
    spare_parts_ready: bool
    external_cmms_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class MaintenanceWindowSuggestion(BaseModel):
    """Suggestion for optimal maintenance window."""
    machine_id: str
    window_start: datetime
    window_end: datetime
    duration_hours: float
    impact_on_plan: str  # "none", "low", "medium", "high"
    plan_delay_hours: float  # Estimated delay if maintenance is done
    risk_if_postponed: float  # Increased failure risk if postponed
    reason: str
    alternative_windows: List[Dict[str, Any]] = Field(default_factory=list)


