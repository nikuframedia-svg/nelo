"""
════════════════════════════════════════════════════════════════════════════════════════════════════
API XAI-DT - Digital Twin Product API
════════════════════════════════════════════════════════════════════════════════════════════════════

Contract 6 Implementation: XAI-DT Product Endpoints

Endpoints for:
- Analyzing product scans vs CAD
- Getting conformance history
- Golden runs and process optimization
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from digital_twin.xai_dt_geometry import (
    get_deviation_engine,
    DeviationField,
    XaiDtExplanation,
    create_test_deviation_field,
)
from digital_twin.process_optimization import (
    compute_golden_runs,
    suggest_process_params,
    get_golden_runs,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/digital-twin", tags=["Digital Twin Product"])


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class AnalyzeScanRequest(BaseModel):
    """Request for scan analysis."""
    scan_id: str = Field(..., description="ID/reference of the scan")
    process_params: Dict[str, float] = Field(
        default_factory=dict,
        description="Process parameters at time of scan"
    )
    machine_id: Optional[str] = Field(None, description="Machine ID")
    operator_id: Optional[str] = Field(None, description="Operator ID")


class ConformanceSnapshot(BaseModel):
    """Product conformance snapshot."""
    id: int
    scan_id: str
    max_dev: float
    mean_dev: float
    rms_dev: float
    scalar_error_score: float
    conformity_status: str
    explanation: Optional[Dict]
    machine_id: Optional[str]
    created_at: str


class AnalyzeScanResponse(BaseModel):
    """Response from scan analysis."""
    snapshot_id: int
    scan_id: str
    max_dev: float
    mean_dev: float
    rms_dev: float
    scalar_error_score: float
    conformity_status: str
    probable_causes: List[Dict]
    recommendations: List[str]
    dominant_modes: List[int]


class GoldenRunResponse(BaseModel):
    """Golden run data."""
    id: int
    revision_id: int
    operation_id: Optional[int]
    machine_id: Optional[str]
    process_params: Dict
    kpis: Optional[Dict]
    score: float


class ProcessParamSuggestion(BaseModel):
    """Suggested process parameters."""
    revision_id: int
    operation_id: Optional[int]
    machine_id: Optional[str]
    suggested_params: Dict
    based_on_golden_runs: int
    expected_quality_score: float


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE SESSION
# ═══════════════════════════════════════════════════════════════════════════════

def get_db():
    """Get database session."""
    from duplios.models import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SCAN ANALYSIS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/product/{revision_id}/analyze-scan", response_model=AnalyzeScanResponse)
def api_analyze_scan(
    revision_id: int,
    request: AnalyzeScanRequest,
    db: Session = Depends(get_db),
):
    """
    Analyze a product scan against CAD reference.
    
    Steps:
    1. Load scan as DeviationField
    2. Run XAI-DT analysis
    3. Store conformance snapshot
    4. Return analysis results
    """
    from duplios.dpp_models import ProductConformanceSnapshot, ConformityStatus
    
    logger.info(f"Analyzing scan {request.scan_id} for revision {revision_id}")
    
    # Load scan as deviation field (for now, create synthetic)
    # In production: load_scan_as_deviation_field(request.scan_id)
    field = _load_or_create_scan_field(request.scan_id, revision_id)
    
    # Get deviation engine
    engine = get_deviation_engine(use_advanced=True)
    
    # Fit on baseline if not fitted (simplified - in production use historical data)
    if not engine._is_fitted:
        baseline_fields = [create_test_deviation_field(100, 0.1) for _ in range(10)]
        engine.fit(baseline_fields)
    
    # Analyze
    explanation = engine.explain(field, request.process_params)
    
    # Determine conformity status
    if explanation.scalar_error_score < 30:
        status = ConformityStatus.IN_TOLERANCE
    elif explanation.scalar_error_score < 70:
        status = ConformityStatus.OUT_OF_TOLERANCE
    else:
        status = ConformityStatus.CRITICAL
    
    # Create snapshot
    snapshot = ProductConformanceSnapshot(
        revision_id=revision_id,
        scan_id=request.scan_id,
        max_dev=field.max_deviation,
        mean_dev=field.mean_deviation,
        rms_dev=field.rms_deviation,
        scalar_error_score=explanation.scalar_error_score,
        conformity_status=status,
        explanation=json.dumps(explanation.to_dict()),
        machine_id=request.machine_id,
        operator_id=request.operator_id,
        process_params=json.dumps(request.process_params),
        created_at=datetime.utcnow(),
    )
    
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)
    
    logger.info(f"Scan analysis complete: score={explanation.scalar_error_score:.1f}, status={status.value}")
    
    return AnalyzeScanResponse(
        snapshot_id=snapshot.id,
        scan_id=request.scan_id,
        max_dev=field.max_deviation,
        mean_dev=field.mean_deviation,
        rms_dev=field.rms_deviation,
        scalar_error_score=explanation.scalar_error_score,
        conformity_status=status.value,
        probable_causes=[c.to_dict() for c in explanation.probable_causes],
        recommendations=explanation.recommendations,
        dominant_modes=explanation.dominant_modes,
    )


@router.get("/product/{revision_id}/conformance", response_model=List[ConformanceSnapshot])
def api_get_conformance(
    revision_id: int,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """Get conformance history for a revision."""
    from duplios.dpp_models import ProductConformanceSnapshot
    
    snapshots = db.query(ProductConformanceSnapshot).filter(
        ProductConformanceSnapshot.revision_id == revision_id
    ).order_by(
        ProductConformanceSnapshot.created_at.desc()
    ).limit(limit).all()
    
    return [
        ConformanceSnapshot(
            id=s.id,
            scan_id=s.scan_id,
            max_dev=s.max_dev,
            mean_dev=s.mean_dev,
            rms_dev=s.rms_dev,
            scalar_error_score=s.scalar_error_score,
            conformity_status=s.conformity_status.value,
            explanation=json.loads(s.explanation) if s.explanation else None,
            machine_id=s.machine_id,
            created_at=s.created_at.isoformat(),
        )
        for s in snapshots
    ]


@router.get("/product/{revision_id}/conformance/summary")
def api_get_conformance_summary(
    revision_id: int,
    db: Session = Depends(get_db),
):
    """Get conformance summary statistics."""
    from duplios.dpp_models import ProductConformanceSnapshot, ConformityStatus
    from sqlalchemy import func
    
    # Get counts by status
    status_counts = db.query(
        ProductConformanceSnapshot.conformity_status,
        func.count(ProductConformanceSnapshot.id)
    ).filter(
        ProductConformanceSnapshot.revision_id == revision_id
    ).group_by(
        ProductConformanceSnapshot.conformity_status
    ).all()
    
    counts = {status.value: 0 for status in ConformityStatus}
    for status, count in status_counts:
        counts[status.value] = count
    
    # Get average score
    avg_score = db.query(
        func.avg(ProductConformanceSnapshot.scalar_error_score)
    ).filter(
        ProductConformanceSnapshot.revision_id == revision_id
    ).scalar() or 0
    
    total = sum(counts.values())
    
    return {
        "revision_id": revision_id,
        "total_scans": total,
        "in_tolerance": counts["IN_TOLERANCE"],
        "out_of_tolerance": counts["OUT_OF_TOLERANCE"],
        "critical": counts["CRITICAL"],
        "avg_error_score": round(avg_score, 1),
        "compliance_rate": round(counts["IN_TOLERANCE"] / total * 100, 1) if total > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GOLDEN RUNS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/product/{revision_id}/golden-runs/compute")
def api_compute_golden_runs(
    revision_id: int,
    operation_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """Compute golden runs from historical data."""
    result = compute_golden_runs(revision_id, operation_id, db)
    return {
        "revision_id": revision_id,
        "operation_id": operation_id,
        "golden_runs_computed": result.get("count", 0),
        "best_score": result.get("best_score"),
    }


@router.get("/product/{revision_id}/golden-runs", response_model=List[GoldenRunResponse])
def api_get_golden_runs(
    revision_id: int,
    operation_id: Optional[int] = None,
    machine_id: Optional[str] = None,
    limit: int = 10,
    db: Session = Depends(get_db),
):
    """Get golden runs for a revision."""
    runs = get_golden_runs(revision_id, operation_id, machine_id, limit, db)
    
    return [
        GoldenRunResponse(
            id=r.id,
            revision_id=r.revision_id,
            operation_id=r.operation_id,
            machine_id=r.machine_id,
            process_params=json.loads(r.process_params) if r.process_params else {},
            kpis=json.loads(r.kpis) if r.kpis else None,
            score=r.score,
        )
        for r in runs
    ]


@router.get("/product/{revision_id}/suggest-params", response_model=ProcessParamSuggestion)
def api_suggest_params(
    revision_id: int,
    operation_id: Optional[int] = None,
    machine_id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Suggest optimal process parameters based on golden runs."""
    suggestion = suggest_process_params(revision_id, operation_id, machine_id, db)
    
    return ProcessParamSuggestion(
        revision_id=revision_id,
        operation_id=operation_id,
        machine_id=machine_id,
        suggested_params=suggestion.get("params", {}),
        based_on_golden_runs=suggestion.get("based_on", 0),
        expected_quality_score=suggestion.get("expected_score", 0),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _load_or_create_scan_field(scan_id: str, revision_id: int) -> DeviationField:
    """
    Load scan data as DeviationField.
    
    In production: load from scan database/files.
    For now: generate synthetic data based on scan_id hash.
    """
    import numpy as np
    
    # Use scan_id to seed random for reproducibility
    seed = hash(scan_id) % (2**32)
    np.random.seed(seed)
    
    # Generate synthetic deviation field
    n_points = 100
    magnitude = 0.1 + 0.3 * (seed % 10) / 10  # 0.1 to 0.4
    
    points = np.linspace(0, 1, n_points).reshape(-1, 1)
    deviations = magnitude * np.sin(2 * np.pi * points.flatten())
    deviations += 0.05 * np.random.randn(n_points)
    
    return DeviationField(
        points=points,
        deviations=deviations,
        field_name=f"scan_{scan_id}",
        metadata={"scan_id": scan_id, "revision_id": revision_id},
    )


def load_cad_reference(revision_id: int) -> Optional[str]:
    """
    Load CAD reference for a revision.
    
    Returns path to CAD file or None.
    """
    # Stub: in production, query database for CAD file path
    return f"/data/cad/revision_{revision_id}.step"


def load_scan_as_deviation_field(scan_id: str) -> DeviationField:
    """
    Load scan data and convert to DeviationField.
    
    In production: parse scan file (JSON/CSV/point cloud)
    """
    return _load_or_create_scan_field(scan_id, 0)



