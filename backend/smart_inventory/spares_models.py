"""
════════════════════════════════════════════════════════════════════════════════════════════════════
SPARE PARTS MODELS & FORECASTING - Previsão de Peças Sobressalentes
════════════════════════════════════════════════════════════════════════════════════════════════════

Modelos e serviço para gestão e previsão de peças sobressalentes.

Integra-se com:
- PredictiveCare (RUL/SHI) para prever substituições
- MRP para garantir disponibilidade
- SmartInventory para stock management

R&D / SIFIDE: WP1 - Digital Twin para manutenção preditiva
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class SpareCriticality(str, Enum):
    """Criticidade da peça sobressalente."""
    LOW = "LOW"          # Não afeta produção (pode esperar)
    MEDIUM = "MEDIUM"    # Afeta produção parcialmente
    HIGH = "HIGH"        # Para produção se faltar
    CRITICAL = "CRITICAL"  # Segurança ou regulamentação


# ═══════════════════════════════════════════════════════════════════════════════
# SQLAlchemy MODELS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from sqlalchemy import (
        Column, Integer, String, Float, DateTime, Text, Boolean,
        ForeignKey, Index
    )
    from sqlalchemy.ext.declarative import declarative_base
    
    try:
        from models_common import Base
    except ImportError:
        Base = declarative_base()
    
    class SparePart(Base):
        """
        SQLAlchemy model for spare parts.
        
        Links SKUs to machines/components for replacement tracking.
        """
        __tablename__ = "spare_parts"
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        
        # Identification
        sku_id = Column(String(64), nullable=False, index=True)
        component_name = Column(String(128), nullable=False)
        description = Column(Text, nullable=True)
        
        # Machine association (can be specific or group)
        machine_id = Column(String(64), nullable=True, index=True)  # Specific machine
        machine_group = Column(String(64), nullable=True)  # Or machine type/group
        
        # Replacement statistics
        mtbr_hours = Column(Float, nullable=True)  # Mean Time Between Replacement
        mtbr_confidence = Column(Float, default=0.5)  # Confidence in MTBR (0-1)
        last_replacement_at = Column(DateTime(timezone=True), nullable=True)
        total_replacements = Column(Integer, default=0)
        
        # Criticality
        criticality = Column(String(16), default=SpareCriticality.MEDIUM.value)
        lead_time_days = Column(Float, default=7.0)  # Procurement lead time
        
        # Stock info
        safety_stock = Column(Integer, default=1)
        reorder_point = Column(Integer, default=2)
        
        # Costs
        unit_cost = Column(Float, nullable=True)
        
        # Timestamps
        created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
        updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
        
        __table_args__ = (
            Index('ix_spare_machine_sku', 'machine_id', 'sku_id'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                "id": self.id,
                "sku_id": self.sku_id,
                "component_name": self.component_name,
                "description": self.description,
                "machine_id": self.machine_id,
                "machine_group": self.machine_group,
                "mtbr_hours": self.mtbr_hours,
                "mtbr_confidence": self.mtbr_confidence,
                "last_replacement_at": self.last_replacement_at.isoformat() if self.last_replacement_at else None,
                "total_replacements": self.total_replacements,
                "criticality": self.criticality,
                "lead_time_days": self.lead_time_days,
                "safety_stock": self.safety_stock,
                "reorder_point": self.reorder_point,
                "unit_cost": self.unit_cost,
            }
    
    class SpareReplacementHistory(Base):
        """History of spare part replacements."""
        __tablename__ = "spare_replacement_history"
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        spare_part_id = Column(Integer, ForeignKey("spare_parts.id"), nullable=False)
        machine_id = Column(String(64), nullable=False)
        work_order_id = Column(Integer, nullable=True)
        
        replaced_at = Column(DateTime(timezone=True), nullable=False)
        operating_hours_at_replacement = Column(Float, nullable=True)
        shi_at_replacement = Column(Float, nullable=True)
        
        failure_type = Column(String(32), nullable=True)  # "preventive", "predictive", "corrective"
        was_unplanned = Column(Boolean, default=False)
        
        notes = Column(Text, nullable=True)
        
        __table_args__ = (
            Index('ix_spare_hist_part_date', 'spare_part_id', 'replaced_at'),
        )
    
    SQLALCHEMY_AVAILABLE = True

except ImportError:
    SparePart = None
    SpareReplacementHistory = None
    SQLALCHEMY_AVAILABLE = False
    logger.warning("SQLAlchemy not available - using Pydantic models only")


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class SparePartCreate(BaseModel):
    """Schema for creating a spare part."""
    sku_id: str = Field(..., description="SKU identifier")
    component_name: str = Field(..., description="Component name")
    description: Optional[str] = None
    machine_id: Optional[str] = None
    machine_group: Optional[str] = None
    mtbr_hours: Optional[float] = Field(None, ge=0, description="Mean Time Between Replacement")
    criticality: SpareCriticality = SpareCriticality.MEDIUM
    lead_time_days: float = Field(7.0, ge=0)
    safety_stock: int = Field(1, ge=0)
    unit_cost: Optional[float] = Field(None, ge=0)


class SpareNeed(BaseModel):
    """Predicted spare part need."""
    sku_id: str
    component_name: str
    machine_id: str
    expected_replacements: float  # Can be fractional (probability)
    recommended_date: datetime
    confidence: float = Field(ge=0, le=1)
    reason: str
    criticality: str
    current_stock: Optional[int] = None
    reorder_needed: bool = False


class SpareForecastResult(BaseModel):
    """Result of spare parts forecast."""
    horizon_days: int
    total_needs: int
    critical_needs: int
    reorder_recommendations: int
    needs: List[SpareNeed]
    generated_at: datetime


# ═══════════════════════════════════════════════════════════════════════════════
# SPARE FORECAST SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SparePartInfo:
    """Internal spare part information."""
    sku_id: str
    component_name: str
    machine_id: str
    mtbr_hours: float
    mtbr_confidence: float
    last_replacement: Optional[datetime]
    operating_hours: float
    criticality: str
    lead_time_days: float
    safety_stock: int
    current_stock: int = 0


class SmartSpareForecastService:
    """
    Service for forecasting spare part needs.
    
    Uses:
    - MTBR (Mean Time Between Replacement) statistics
    - RUL from PredictiveCare
    - Production plan (future load)
    
    To predict when spare parts will need to be replaced.
    """
    
    def __init__(
        self,
        predictive_care_service=None,
        db_session=None,
    ):
        """
        Initialize the forecast service.
        
        Args:
            predictive_care_service: PredictiveCareService for RUL data
            db_session: SQLAlchemy session
        """
        self._pc_service = predictive_care_service
        self._db_session = db_session
        
        # Demo spare parts (in real system, would come from database)
        self._demo_spares: List[SparePartInfo] = [
            SparePartInfo(
                sku_id="SPARE-BRG-001",
                component_name="Main Spindle Bearing",
                machine_id="MACH-001",
                mtbr_hours=2000,
                mtbr_confidence=0.8,
                last_replacement=datetime.now(timezone.utc) - timedelta(hours=1500),
                operating_hours=1500,
                criticality="HIGH",
                lead_time_days=14,
                safety_stock=1,
                current_stock=2,
            ),
            SparePartInfo(
                sku_id="SPARE-BLT-001",
                component_name="Drive Belt",
                machine_id="MACH-001",
                mtbr_hours=1000,
                mtbr_confidence=0.7,
                last_replacement=datetime.now(timezone.utc) - timedelta(hours=800),
                operating_hours=800,
                criticality="MEDIUM",
                lead_time_days=3,
                safety_stock=2,
                current_stock=3,
            ),
            SparePartInfo(
                sku_id="SPARE-FLT-001",
                component_name="Hydraulic Filter",
                machine_id="MACH-002",
                mtbr_hours=500,
                mtbr_confidence=0.9,
                last_replacement=datetime.now(timezone.utc) - timedelta(hours=450),
                operating_hours=450,
                criticality="MEDIUM",
                lead_time_days=2,
                safety_stock=3,
                current_stock=4,
            ),
            SparePartInfo(
                sku_id="SPARE-SRV-001",
                component_name="Servo Motor",
                machine_id="CNC-001",
                mtbr_hours=5000,
                mtbr_confidence=0.6,
                last_replacement=datetime.now(timezone.utc) - timedelta(hours=4200),
                operating_hours=4200,
                criticality="CRITICAL",
                lead_time_days=30,
                safety_stock=1,
                current_stock=1,
            ),
        ]
        
        logger.info("SmartSpareForecastService initialized")
    
    def predict_spare_needs(
        self,
        horizon_days: int = 30,
        machine_ids: Optional[List[str]] = None,
    ) -> SpareForecastResult:
        """
        Predict spare part needs for the given horizon.
        
        Args:
            horizon_days: How far ahead to forecast
            machine_ids: Optional filter by machines
            
        Returns:
            SpareForecastResult with predicted needs
        """
        now = datetime.now(timezone.utc)
        horizon_hours = horizon_days * 24
        
        needs: List[SpareNeed] = []
        
        # Get spare parts (from DB or demo)
        spares = self._get_spare_parts(machine_ids)
        
        for spare in spares:
            need = self._predict_single_spare(spare, horizon_hours, now)
            if need is not None:
                needs.append(need)
        
        # Sort by urgency (recommended_date, then criticality)
        criticality_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        needs.sort(key=lambda n: (
            n.recommended_date,
            criticality_order.get(n.criticality, 2),
        ))
        
        # Count stats
        critical_needs = len([n for n in needs if n.criticality == "CRITICAL"])
        reorder_needed = len([n for n in needs if n.reorder_needed])
        
        return SpareForecastResult(
            horizon_days=horizon_days,
            total_needs=len(needs),
            critical_needs=critical_needs,
            reorder_recommendations=reorder_needed,
            needs=needs,
            generated_at=now,
        )
    
    def get_mrp_demands(
        self,
        horizon_days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get spare part demands for MRP integration.
        
        Returns demands in format compatible with MRP engine.
        """
        forecast = self.predict_spare_needs(horizon_days)
        
        demands = []
        for need in forecast.needs:
            if need.expected_replacements > 0.5:  # >50% probability
                demands.append({
                    "sku_id": need.sku_id,
                    "quantity": max(1, int(math.ceil(need.expected_replacements))),
                    "required_date": need.recommended_date,
                    "source": "PREDICTIVECARE",
                    "priority": "HIGH" if need.criticality in ("CRITICAL", "HIGH") else "MEDIUM",
                    "machine_id": need.machine_id,
                    "reason": need.reason,
                })
        
        return demands
    
    def _get_spare_parts(
        self,
        machine_ids: Optional[List[str]] = None,
    ) -> List[SparePartInfo]:
        """Get spare parts from database or demo."""
        spares = self._demo_spares
        
        if machine_ids:
            spares = [s for s in spares if s.machine_id in machine_ids]
        
        return spares
    
    def _predict_single_spare(
        self,
        spare: SparePartInfo,
        horizon_hours: float,
        now: datetime,
    ) -> Optional[SpareNeed]:
        """
        Predict if a single spare part will be needed.
        
        Uses:
        1. Time since last replacement vs MTBR
        2. RUL from PredictiveCare (if available)
        3. Machine load profile
        """
        # Calculate hours until next expected replacement
        if spare.last_replacement:
            hours_since_replacement = (now - spare.last_replacement).total_seconds() / 3600
        else:
            hours_since_replacement = spare.operating_hours
        
        # Basic MTBR-based prediction
        hours_remaining = max(0, spare.mtbr_hours - hours_since_replacement)
        
        # Get RUL adjustment from PredictiveCare
        rul_factor = self._get_rul_adjustment(spare.machine_id)
        
        # Adjust remaining hours based on RUL
        # If machine is degrading faster, spare will fail sooner
        adjusted_hours_remaining = hours_remaining * rul_factor
        
        # Will it be needed within horizon?
        if adjusted_hours_remaining > horizon_hours:
            # Not expected within horizon, but check confidence
            if spare.mtbr_confidence < 0.5 and adjusted_hours_remaining < horizon_hours * 1.5:
                # Low confidence + close to horizon = include with lower probability
                pass
            else:
                return None
        
        # Calculate probability of replacement
        if adjusted_hours_remaining <= 0:
            probability = 0.95  # Almost certain
        else:
            # Exponential distribution-based probability
            rate = 1 / spare.mtbr_hours
            probability = 1 - math.exp(-rate * (spare.mtbr_hours - adjusted_hours_remaining + horizon_hours / 2))
        
        probability = min(0.99, max(0.1, probability * spare.mtbr_confidence))
        
        # Calculate recommended date
        recommended_date = now + timedelta(hours=adjusted_hours_remaining)
        
        # Adjust for lead time (order before needed)
        order_by_date = recommended_date - timedelta(days=spare.lead_time_days)
        if order_by_date < now:
            order_by_date = now
        
        # Check if reorder is needed
        reorder_needed = spare.current_stock <= spare.safety_stock
        
        # Generate reason
        if adjusted_hours_remaining <= 0:
            reason = "Overdue for replacement"
        elif adjusted_hours_remaining <= 168:  # 1 week
            reason = f"Due in {adjusted_hours_remaining:.0f} hours"
        else:
            reason = f"Due in ~{adjusted_hours_remaining / 24:.0f} days"
        
        if rul_factor < 0.8:
            reason += " (accelerated by machine degradation)"
        
        return SpareNeed(
            sku_id=spare.sku_id,
            component_name=spare.component_name,
            machine_id=spare.machine_id,
            expected_replacements=probability,
            recommended_date=recommended_date,
            confidence=spare.mtbr_confidence * rul_factor,
            reason=reason,
            criticality=spare.criticality,
            current_stock=spare.current_stock,
            reorder_needed=reorder_needed,
        )
    
    def _get_rul_adjustment(self, machine_id: str) -> float:
        """
        Get RUL-based adjustment factor for spare prediction.
        
        Returns:
            Factor 0-1, where <1 means faster degradation (shorter spare life)
        """
        if self._pc_service is None:
            try:
                from digital_twin.predictive_care import get_predictive_care_service
                self._pc_service = get_predictive_care_service()
            except:
                return 1.0
        
        try:
            state = self._pc_service.get_machine_state(machine_id)
            
            # Convert SHI to adjustment factor
            # SHI 100% = factor 1.0 (normal wear)
            # SHI 50% = factor 0.7 (faster wear)
            # SHI 20% = factor 0.4 (much faster wear)
            shi_norm = state.shi_percent / 100
            factor = 0.4 + 0.6 * shi_norm
            
            return factor
            
        except Exception as e:
            logger.debug(f"Could not get RUL adjustment for {machine_id}: {e}")
            return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

_spare_service: Optional[SmartSpareForecastService] = None


def get_spare_forecast_service() -> SmartSpareForecastService:
    """Get or create the spare forecast service singleton."""
    global _spare_service
    if _spare_service is None:
        _spare_service = SmartSpareForecastService()
    return _spare_service


