"""
════════════════════════════════════════════════════════════════════════════════════════════════════
TRUST INDEX MODELS - Advanced Trust Index for Duplios DPP
════════════════════════════════════════════════════════════════════════════════════════════════════

Contract D1 Implementation: Field-level Trust Index (0-100)

Models:
- DataSourceType: Classification of data source (measured, reported, estimated, unknown)
- FieldTrustMeta: Metadata for trust calculation per field
- DPPTrustResult: Complete trust index result with breakdown

R&D Integration: Logs trust index evolutions for WPX_TRUST_EVOLUTION experiments.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class DataSourceType(str, Enum):
    """Type of data source for a field."""
    MEDIDO = "MEDIDO"           # Measured directly
    REPORTADO = "REPORTADO"     # Reported by supplier/manufacturer
    ESTIMADO = "ESTIMADO"       # Estimated via LCA engine or models
    DESCONHECIDO = "DESCONHECIDO"  # Unknown/missing


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class FieldTrustMeta(BaseModel):
    """
    Metadata for trust calculation per field.
    
    As specified in Contract D1:
    - field_key: str (ex: "carbon_footprint_kg_co2eq")
    - base_class: DataSourceType
    - measured_fraction, reported_fraction, estimated_fraction, unknown_fraction: 0-1
    - recency_days: int
    - third_party_verified: bool
    - uncertainty_relative: float (ex: 0.2 = ±20%)
    - materiality_weight: float (0-1; carbono ~0.4, água ~0.25, etc.)
    - consistency_zscore: float | None (z-score vs peers, if available)
    """
    
    field_key: str = Field(..., description="Field identifier (e.g., 'carbon_footprint_kg_co2eq')")
    base_class: DataSourceType = Field(..., description="Primary data source type")
    
    # Fraction breakdown (must sum to ~1.0)
    measured_fraction: float = Field(0.0, ge=0.0, le=1.0, description="Fraction of value that is measured")
    reported_fraction: float = Field(0.0, ge=0.0, le=1.0, description="Fraction of value that is reported")
    estimated_fraction: float = Field(0.0, ge=0.0, le=1.0, description="Fraction of value that is estimated")
    unknown_fraction: float = Field(0.0, ge=0.0, le=1.0, description="Fraction of value that is unknown")
    
    # Recency
    recency_days: int = Field(..., ge=0, description="Days since last update")
    
    # Verification
    third_party_verified: bool = Field(False, description="Whether data is third-party verified/audited")
    
    # Uncertainty
    uncertainty_relative: float = Field(0.0, ge=0.0, le=1.0, description="Relative uncertainty (e.g., 0.2 = ±20%)")
    
    # Materiality (used in global weighting)
    materiality_weight: float = Field(0.1, ge=0.0, le=1.0, description="Materiality weight for this field (sum should be ~1.0 across all fields)")
    
    # Consistency vs peers
    consistency_zscore: Optional[float] = Field(None, description="Z-score vs peer products (None if not available)")
    
    # Calculated score (populated by service)
    field_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Calculated trust score for this field")
    
    # Last update timestamp
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp for this field")
    
    class Config:
        use_enum_values = True


class DPPTrustResult(BaseModel):
    """
    Complete trust index result with field-level breakdown.
    
    As specified in Contract D1:
    - dpp_id: UUID
    - overall_trust_index: float (0-100)
    - field_scores: dict[str, float] (field_key -> score)
    - field_metas: dict[str, FieldTrustMeta] (field_key -> metadata)
    """
    
    dpp_id: UUID = Field(..., description="DPP identifier")
    overall_trust_index: float = Field(..., ge=0.0, le=100.0, description="Overall trust index (0-100)")
    
    field_scores: Dict[str, float] = Field(default_factory=dict, description="Field-level scores (field_key -> score)")
    field_metas: Dict[str, FieldTrustMeta] = Field(default_factory=dict, description="Field-level metadata")
    
    # Summary info
    calculated_at: datetime = Field(default_factory=datetime.utcnow, description="Calculation timestamp")
    calculation_version: str = Field("1.0", description="Trust index calculation version")
    
    # Key messages for UI (simplified, non-technical)
    key_messages: list[str] = Field(default_factory=list, description="Key messages for UI (e.g., 'Carbono: base medido + auditado')")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

