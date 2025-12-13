"""
════════════════════════════════════════════════════════════════════════════════════════════════════
TRUST INDEX API - REST Endpoints for Advanced Trust Index
════════════════════════════════════════════════════════════════════════════════════════════════════

Contract D1 Implementation: Field-level Trust Index API

Endpoints:
- GET /duplios/dpp/{dpp_id}/trust-index - Get trust index for a DPP
- POST /duplios/dpp/{dpp_id}/trust-index/recalculate - Force recalculation
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from duplios.trust_index_service import get_trust_index_service
from duplios.trust_index_models import DPPTrustResult
from duplios.dpp_models import DppRecord
from duplios.service import get_db  # Same as api_duplios.py

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/duplios", tags=["Trust Index"])


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/dpp/{dpp_id}/trust-index", response_model=DPPTrustResult)
async def get_trust_index(
    dpp_id: int,
    db: Session = Depends(get_db),
):
    """
    Get trust index for a DPP.
    
    As specified in Contract D1:
    - Returns DPPTrustResult with overall_trust_index, field_scores, field_metas
    - Field-level breakdown available for advanced UIs or debugging
    
    Returns:
        DPPTrustResult with overall trust index and field-level breakdown
    """
    service = get_trust_index_service()
    
    # Get DPP
    dpp = db.query(DppRecord).filter(DppRecord.id == dpp_id).first()
    if not dpp:
        raise HTTPException(status_code=404, detail=f"DPP {dpp_id} not found")
    
    # Calculate trust index
    result = service.calculate_for_dpp(dpp, db_session=db)
    
    return result


@router.post("/dpp/{dpp_id}/trust-index/recalculate", response_model=DPPTrustResult)
async def recalculate_trust_index(
    dpp_id: int,
    db: Session = Depends(get_db),
):
    """
    Force recalculation of trust index for a DPP.
    
    Used after editing DPP data or when metadata changes.
    """
    service = get_trust_index_service()
    
    # Get DPP
    dpp = db.query(DppRecord).filter(DppRecord.id == dpp_id).first()
    if not dpp:
        raise HTTPException(status_code=404, detail=f"DPP {dpp_id} not found")
    
    # Recalculate trust index
    result = service.calculate_for_dpp(dpp, db_session=db)
    
    logger.info(f"Recalculated trust index for DPP {dpp_id}: {result.overall_trust_index}")
    
    return result

