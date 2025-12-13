"""
════════════════════════════════════════════════════════════════════════════════
COMPLIANCE API - REST Endpoints for Compliance Radar
════════════════════════════════════════════════════════════════════════════════

Contract D3 Implementation: Compliance Radar API

Endpoints:
- GET /duplios/dpp/{dpp_id}/compliance-radar - Full compliance analysis
- GET /duplios/dpp/{dpp_id}/compliance-summary - Light summary (scores only)
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

from duplios.compliance_radar import get_compliance_radar_service
from duplios.compliance_models import ComplianceRadarResult
from duplios.dpp_models import DppRecord
from duplios.service import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/duplios", tags=["Compliance"])


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ComplianceSummary(BaseModel):
    """Light compliance summary (scores only)."""
    dpp_id: str
    espr_score: float
    cbam_score: Optional[float]
    csrd_score: float


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/dpp/{dpp_id}/compliance-radar", response_model=ComplianceRadarResult)
async def get_compliance_radar(
    dpp_id: int,
    db: Session = Depends(get_db),
):
    """
    Get full compliance radar analysis for a DPP.
    
    As specified in Contract D3:
    - Analyzes ESPR, CBAM, CSRD compliance
    - Returns scores (0-100) and item-level status
    - Identifies critical gaps and recommended actions
    
    Returns:
        ComplianceRadarResult with complete analysis
    """
    service = get_compliance_radar_service()
    
    # Get DPP
    dpp = db.query(DppRecord).filter(DppRecord.id == dpp_id).first()
    if not dpp:
        raise HTTPException(status_code=404, detail=f"DPP {dpp_id} not found")
    
    # Analyze compliance
    result = service.analyze_dpp(dpp, db_session=db)
    
    return result


@router.get("/dpp/{dpp_id}/compliance-summary", response_model=ComplianceSummary)
async def get_compliance_summary(
    dpp_id: int,
    db: Session = Depends(get_db),
):
    """
    Get light compliance summary (scores only).
    
    As specified in Contract D3:
    - Returns only ESPR, CBAM, CSRD scores
    - Faster endpoint for dashboards and lists
    
    Returns:
        ComplianceSummary with scores only
    """
    service = get_compliance_radar_service()
    
    # Get DPP
    dpp = db.query(DppRecord).filter(DppRecord.id == dpp_id).first()
    if not dpp:
        raise HTTPException(status_code=404, detail=f"DPP {dpp_id} not found")
    
    # Analyze compliance
    result = service.analyze_dpp(dpp, db_session=db)
    
    return ComplianceSummary(
        dpp_id=str(result.dpp_id),
        espr_score=result.espr_score,
        cbam_score=result.cbam_score,
        csrd_score=result.csrd_score,
    )

