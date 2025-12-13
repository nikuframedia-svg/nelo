"""
════════════════════════════════════════════════════════════════════════════════
GAP FILLING API - REST Endpoints for Gap Filling Lite
════════════════════════════════════════════════════════════════════════════════

Contract D2 Implementation: Gap Filling Lite API

Endpoints:
- POST /duplios/dpp/{dpp_id}/gap-fill-lite - Fill missing environmental fields
"""

from __future__ import annotations

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

from duplios.gap_filling_lite import get_gap_filling_lite_service
from duplios.dpp_models import DppRecord
from duplios.service import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/duplios", tags=["Gap Filling"])


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class GapFillRequest(BaseModel):
    """Request for gap filling."""
    force: bool = False  # If True, overwrite existing values


class GapFillResponse(BaseModel):
    """Response from gap filling."""
    success: bool
    filled_fields: list[str]
    values: Dict[str, float]
    uncertainty: Dict[str, float]
    context: Dict[str, Any]
    message: str


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/dpp/{dpp_id}/gap-fill-lite", response_model=GapFillResponse)
async def gap_fill_lite(
    dpp_id: int,
    request: GapFillRequest = GapFillRequest(),
    db: Session = Depends(get_db),
):
    """
    Fill missing environmental fields using Gap Filling Lite.
    
    As specified in Contract D2:
    - Uses internal factor tables (no Ecoinvent)
    - Applies contextual adjustments (country, tech age)
    - Updates DPP with filled values and metadata
    - Recalculates Trust Index automatically
    - Logs to R&D (WPX_GAPFILL_LITE)
    
    Args:
        dpp_id: DPP identifier
        request: GapFillRequest with force flag
        db: Database session
    
    Returns:
        GapFillResponse with filled fields and metadata
    """
    service = get_gap_filling_lite_service()
    
    # Get DPP
    dpp = db.query(DppRecord).filter(DppRecord.id == dpp_id).first()
    if not dpp:
        raise HTTPException(status_code=404, detail=f"DPP {dpp_id} not found")
    
    try:
        # Perform gap filling
        result = service.fill_for_dpp(dpp, db_session=db, force=request.force)
        
        if not result["filled_fields"]:
            message = "No fields were filled (all fields already have values or no composition found)"
        else:
            message = f"Filled {len(result['filled_fields'])} field(s): {', '.join(result['filled_fields'])}"
        
        return GapFillResponse(
            success=True,
            filled_fields=result["filled_fields"],
            values=result["values"],
            uncertainty=result["uncertainty"],
            context=result["context"],
            message=message,
        )
    
    except Exception as e:
        logger.error(f"Gap filling failed for DPP {dpp_id}: {e}", exc_info=True)
        # Never block user - return error but don't crash
        return GapFillResponse(
            success=False,
            filled_fields=[],
            values={},
            uncertainty={},
            context={},
            message=f"Gap filling failed: {str(e)}",
        )


