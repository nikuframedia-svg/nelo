"""
════════════════════════════════════════════════════════════════════════════════
COMPLIANCE MODELS - Compliance Radar for Duplios DPP
════════════════════════════════════════════════════════════════════════════════

Contract D3 Implementation: Compliance Radar (ESPR / CBAM / CSRD Lite)

Models:
- RegulationType: ESPR, CBAM, CSRD
- ComplianceStatus: COMPLIANT, PARTIAL, MISSING
- ComplianceItemStatus: Status of individual compliance items
- ComplianceRadarResult: Complete compliance analysis result
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class RegulationType(str, Enum):
    """Type of regulation."""
    ESPR = "ESPR"  # Ecodesign for Sustainable Products Regulation
    CBAM = "CBAM"  # Carbon Border Adjustment Mechanism
    CSRD = "CSRD"  # Corporate Sustainability Reporting Directive


class ComplianceStatus(str, Enum):
    """Compliance status for an item."""
    COMPLIANT = "COMPLIANT"  # Fully compliant
    PARTIAL = "PARTIAL"      # Partially compliant
    MISSING = "MISSING"      # Missing required fields


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ComplianceItemStatus(BaseModel):
    """
    Status of an individual compliance item.
    
    As specified in Contract D3:
    - key: str (ex: "espr.identification", "espr.composition")
    - description: str
    - required: bool
    - present: bool
    - severity: int (1=low, 3=critical)
    - notes: str | None
    """
    
    key: str = Field(..., description="Compliance item key (e.g., 'espr.identification')")
    description: str = Field(..., description="Human-readable description")
    required: bool = Field(..., description="Whether this item is required")
    present: bool = Field(..., description="Whether the required fields are present")
    severity: int = Field(..., ge=1, le=3, description="Severity: 1=low, 2=medium, 3=critical")
    notes: Optional[str] = Field(None, description="Additional notes or recommendations")
    
    @property
    def status(self) -> ComplianceStatus:
        """Calculate compliance status."""
        if self.required:
            return ComplianceStatus.COMPLIANT if self.present else ComplianceStatus.MISSING
        else:
            return ComplianceStatus.COMPLIANT if self.present else ComplianceStatus.PARTIAL


class ComplianceRadarResult(BaseModel):
    """
    Complete compliance radar analysis result.
    
    As specified in Contract D3:
    - dpp_id: UUID
    - espr_score: float (0-100)
    - cbam_score: float | None (0-100, None if not applicable)
    - csrd_score: float (0-100)
    - espr_items: list[ComplianceItemStatus]
    - cbam_items: list[ComplianceItemStatus]
    - csrd_items: list[ComplianceItemStatus]
    """
    
    dpp_id: UUID = Field(..., description="DPP identifier")
    
    # Scores (0-100)
    espr_score: float = Field(..., ge=0.0, le=100.0, description="ESPR compliance score (0-100)")
    cbam_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="CBAM compliance score (0-100, None if not applicable)")
    csrd_score: float = Field(..., ge=0.0, le=100.0, description="CSRD compliance score (0-100)")
    
    # Item-level status
    espr_items: list[ComplianceItemStatus] = Field(default_factory=list, description="ESPR compliance items")
    cbam_items: list[ComplianceItemStatus] = Field(default_factory=list, description="CBAM compliance items")
    csrd_items: list[ComplianceItemStatus] = Field(default_factory=list, description="CSRD compliance items")
    
    # Summary
    critical_gaps: list[str] = Field(default_factory=list, description="Critical gaps (severity=3, missing)")
    recommended_actions: list[str] = Field(default_factory=list, description="Recommended actions to improve compliance")
    
    # Metadata
    analyzed_at: str = Field(..., description="Analysis timestamp (ISO format)")
    regulation_version: str = Field("2024", description="Regulation version/date")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            UUID: lambda v: str(v),
        }


