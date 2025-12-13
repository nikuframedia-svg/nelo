"""
════════════════════════════════════════════════════════════════════════════════════════════════════
XAI-DT Product API - Explainable Digital Twin REST Endpoints
════════════════════════════════════════════════════════════════════════════════════════════════════

Endpoints for XAI-DT Product Analysis:
- POST /xai-dt/analyze - Analyze CAD vs Scan alignment and deviations
- GET /xai-dt/analyses - List analysis results
- GET /xai-dt/analyses/{id} - Get specific analysis
- POST /xai-dt/demo - Generate demo analysis
- GET /xai-dt/status - Service status

R&D / SIFIDE: WP1 - Digital Twin & Explainability
"""

from __future__ import annotations

import logging
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Body, File, UploadFile
from pydantic import BaseModel, Field

from .xai_dt_product import (
    XAIDTConfig,
    XAIDTProductAnalyzer,
    get_xai_dt_analyzer,
    PointCloud,
    DeviationField3D,
    PCAResult,
    RegionalAnalysis,
    IdentifiedPattern,
    RootCause,
    CorrectiveAction,
    XAIDTAnalysisResult,
    create_demo_cad_scan,
    DeviationPattern,
    RootCauseCategory,
)

import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/xai-dt", tags=["XAI-DT"])


# ═══════════════════════════════════════════════════════════════════════════════
# IN-MEMORY STORAGE
# ═══════════════════════════════════════════════════════════════════════════════

_analyses_store: Dict[str, XAIDTAnalysisResult] = {}


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class Point3DInput(BaseModel):
    """3D point input."""
    x: float
    y: float
    z: float


class PointCloudInput(BaseModel):
    """Point cloud input for analysis."""
    points: List[List[float]] = Field(..., description="List of [x, y, z] points")
    name: str = Field("pointcloud", description="Name of the point cloud")


class AnalysisRequest(BaseModel):
    """Request for CAD vs Scan analysis."""
    cad_points: List[List[float]] = Field(..., description="CAD nominal points [[x,y,z], ...]")
    scan_points: List[List[float]] = Field(..., description="Scanned measured points [[x,y,z], ...]")
    tolerance: float = Field(0.5, ge=0.01, le=10.0, description="Geometric tolerance (mm)")
    cad_name: str = Field("cad", description="Name of CAD model")
    scan_name: str = Field("scan", description="Name of scanned part")
    
    class Config:
        json_schema_extra = {
            "example": {
                "cad_points": [[0, 0, 0], [100, 0, 0], [0, 100, 0]],
                "scan_points": [[0.1, 0, 0], [100.2, 0, 0], [0, 100.1, 0]],
                "tolerance": 0.5,
                "cad_name": "bracket_v1",
                "scan_name": "bracket_scan_001"
            }
        }


class DemoAnalysisRequest(BaseModel):
    """Request for demo analysis."""
    n_points: int = Field(500, ge=100, le=10000, description="Number of points")
    deviation_type: str = Field("offset", description="Type: offset, scale, random, local")
    deviation_magnitude: float = Field(0.5, ge=0.1, le=5.0, description="Deviation magnitude (mm)")
    tolerance: float = Field(0.5, ge=0.01, le=10.0, description="Tolerance (mm)")


class DeviationFieldResponse(BaseModel):
    """Deviation field summary response."""
    n_points: int
    tolerance: float
    mean_deviation: float
    max_deviation: float
    rms_deviation: float
    pct_out_of_tolerance: float
    deviation_score: float
    alignment_rmse: float


class PatternResponse(BaseModel):
    """Identified pattern response."""
    pattern: str
    confidence: float
    parameters: Dict[str, float]
    affected_region: str
    evidence: List[str]


class RootCauseResponse(BaseModel):
    """Root cause response."""
    category: str
    description: str
    confidence: float
    evidence: List[str]
    patterns_linked: List[str]


class CorrectiveActionResponse(BaseModel):
    """Corrective action response."""
    action: str
    priority: str
    root_cause: str
    expected_impact: str


class AnalysisResultResponse(BaseModel):
    """Complete analysis result response."""
    analysis_id: str
    timestamp: str
    cad_name: str
    scan_name: str
    
    deviation_field: DeviationFieldResponse
    pca_result: Dict[str, Any]
    regional_analysis: Dict[str, Any]
    identified_patterns: List[PatternResponse]
    root_causes: List[RootCauseResponse]
    corrective_actions: List[CorrectiveActionResponse]
    
    overall_quality: str
    summary_text: str


class AnalysisListItem(BaseModel):
    """Item in analysis list."""
    analysis_id: str
    timestamp: str
    cad_name: str
    scan_name: str
    deviation_score: float
    overall_quality: str


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _convert_analysis_to_response(result: XAIDTAnalysisResult) -> AnalysisResultResponse:
    """Convert internal result to API response."""
    return AnalysisResultResponse(
        analysis_id=result.analysis_id,
        timestamp=result.timestamp.isoformat(),
        cad_name=result.cad_name,
        scan_name=result.scan_name,
        deviation_field=DeviationFieldResponse(
            n_points=result.deviation_field.n_points,
            tolerance=result.deviation_field.tolerance,
            mean_deviation=round(result.deviation_field.mean_deviation, 4),
            max_deviation=round(result.deviation_field.max_deviation, 4),
            rms_deviation=round(result.deviation_field.rms_deviation, 4),
            pct_out_of_tolerance=round(result.deviation_field.pct_out_of_tolerance, 2),
            deviation_score=round(result.deviation_field.deviation_score, 1),
            alignment_rmse=round(result.deviation_field.alignment_rmse, 4),
        ),
        pca_result=result.pca_result.to_dict(),
        regional_analysis=result.regional_analysis.to_dict(),
        identified_patterns=[
            PatternResponse(
                pattern=p.pattern.value,
                confidence=round(p.confidence, 3),
                parameters={k: round(v, 4) for k, v in p.parameters.items()},
                affected_region=p.affected_region,
                evidence=p.evidence,
            )
            for p in result.identified_patterns
        ],
        root_causes=[
            RootCauseResponse(
                category=rc.category.value,
                description=rc.description,
                confidence=round(rc.confidence, 3),
                evidence=rc.evidence,
                patterns_linked=[p.value for p in rc.patterns_linked],
            )
            for rc in result.root_causes
        ],
        corrective_actions=[
            CorrectiveActionResponse(
                action=ca.action,
                priority=ca.priority,
                root_cause=ca.root_cause.value,
                expected_impact=ca.expected_impact,
            )
            for ca in result.corrective_actions
        ],
        overall_quality=result.overall_quality,
        summary_text=result.summary_text,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/status")
async def get_status() -> Dict[str, Any]:
    """
    Get XAI-DT service status.
    
    Returns:
        Service status and configuration info
    """
    analyzer = get_xai_dt_analyzer()
    
    return {
        "service": "XAI-DT Product",
        "version": "1.0.0",
        "status": "operational",
        "analyses_stored": len(_analyses_store),
        "config": {
            "icp_max_iterations": analyzer.config.icp_max_iterations,
            "default_tolerance": analyzer.config.default_tolerance,
            "max_causes": analyzer.config.max_causes,
        },
        "supported_patterns": [p.value for p in DeviationPattern],
        "root_cause_categories": [c.value for c in RootCauseCategory],
    }


@router.post("/analyze")
async def analyze_cad_scan(
    request: AnalysisRequest = Body(...),
) -> AnalysisResultResponse:
    """
    Analyze CAD vs Scan geometric quality.
    
    Performs:
    1. ICP alignment of CAD to Scan
    2. Deviation field computation
    3. Pattern identification (PCA, regional)
    4. Root cause analysis
    5. Corrective action generation
    
    Args:
        request: CAD and Scan point clouds with parameters
    
    Returns:
        Complete analysis with patterns, causes, and recommendations
    """
    try:
        # Validate points
        if len(request.cad_points) < 10:
            raise HTTPException(status_code=400, detail="CAD must have at least 10 points")
        if len(request.scan_points) < 10:
            raise HTTPException(status_code=400, detail="Scan must have at least 10 points")
        
        # Convert to numpy arrays
        cad_points = np.array(request.cad_points, dtype=np.float64)
        scan_points = np.array(request.scan_points, dtype=np.float64)
        
        if cad_points.shape[1] != 3 or scan_points.shape[1] != 3:
            raise HTTPException(status_code=400, detail="Points must have 3 coordinates (x, y, z)")
        
        # Create point clouds
        cad = PointCloud(points=cad_points, name=request.cad_name)
        scan = PointCloud(points=scan_points, name=request.scan_name)
        
        # Run analysis
        analyzer = get_xai_dt_analyzer()
        result = analyzer.analyze(cad, scan, tolerance=request.tolerance)
        
        # Store result
        _analyses_store[result.analysis_id] = result
        
        logger.info(f"Analysis {result.analysis_id} completed: {result.overall_quality}")
        
        return _convert_analysis_to_response(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/demo")
async def run_demo_analysis(
    request: DemoAnalysisRequest = Body(...),
) -> AnalysisResultResponse:
    """
    Run demo analysis with generated data.
    
    Useful for testing and demonstration purposes.
    
    Args:
        request: Demo parameters (n_points, deviation_type, etc.)
    
    Returns:
        Complete analysis result
    """
    try:
        # Generate demo data
        cad, scan = create_demo_cad_scan(
            n_points=request.n_points,
            deviation_type=request.deviation_type,
            deviation_magnitude=request.deviation_magnitude,
        )
        
        # Run analysis
        analyzer = get_xai_dt_analyzer()
        result = analyzer.analyze(cad, scan, tolerance=request.tolerance)
        
        # Store result
        _analyses_store[result.analysis_id] = result
        
        logger.info(f"Demo analysis {result.analysis_id} completed")
        
        return _convert_analysis_to_response(result)
        
    except Exception as e:
        logger.error(f"Demo analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyses")
async def list_analyses(
    limit: int = Query(20, ge=1, le=100, description="Max results"),
) -> Dict[str, Any]:
    """
    List stored analyses.
    
    Returns:
        List of analysis summaries
    """
    items = []
    for analysis_id, result in sorted(
        _analyses_store.items(),
        key=lambda x: x[1].timestamp,
        reverse=True
    )[:limit]:
        items.append(AnalysisListItem(
            analysis_id=analysis_id,
            timestamp=result.timestamp.isoformat(),
            cad_name=result.cad_name,
            scan_name=result.scan_name,
            deviation_score=round(result.deviation_field.deviation_score, 1),
            overall_quality=result.overall_quality,
        ))
    
    return {
        "total": len(_analyses_store),
        "returned": len(items),
        "analyses": [item.dict() for item in items],
    }


@router.get("/analyses/{analysis_id}")
async def get_analysis(analysis_id: str) -> AnalysisResultResponse:
    """
    Get specific analysis result.
    
    Args:
        analysis_id: Analysis ID
    
    Returns:
        Complete analysis result
    """
    if analysis_id not in _analyses_store:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    
    return _convert_analysis_to_response(_analyses_store[analysis_id])


@router.get("/analyses/{analysis_id}/heatmap")
async def get_deviation_heatmap(
    analysis_id: str,
    axis: str = Query("z", description="Axis to project onto: x, y, z"),
) -> Dict[str, Any]:
    """
    Get deviation heatmap data for visualization.
    
    Projects 3D deviations onto a 2D plane for heatmap rendering.
    
    Args:
        analysis_id: Analysis ID
        axis: Axis perpendicular to projection plane
    
    Returns:
        2D grid with deviation values for heatmap
    """
    if analysis_id not in _analyses_store:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    
    result = _analyses_store[analysis_id]
    field = result.deviation_field
    
    # Determine projection axes
    axis_map = {"x": 0, "y": 1, "z": 2}
    proj_axis = axis_map.get(axis.lower(), 2)
    other_axes = [i for i in range(3) if i != proj_axis]
    
    # Project points onto 2D
    points_2d = field.points[:, other_axes]
    
    # Create grid
    n_bins = 50
    min_vals = np.min(points_2d, axis=0)
    max_vals = np.max(points_2d, axis=0)
    
    x_bins = np.linspace(min_vals[0], max_vals[0], n_bins + 1)
    y_bins = np.linspace(min_vals[1], max_vals[1], n_bins + 1)
    
    # Bin deviations
    heatmap = np.zeros((n_bins, n_bins))
    counts = np.zeros((n_bins, n_bins))
    
    for i, (pt, dist) in enumerate(zip(points_2d, field.distances)):
        x_idx = min(n_bins - 1, max(0, int((pt[0] - min_vals[0]) / (max_vals[0] - min_vals[0] + 1e-10) * n_bins)))
        y_idx = min(n_bins - 1, max(0, int((pt[1] - min_vals[1]) / (max_vals[1] - min_vals[1] + 1e-10) * n_bins)))
        heatmap[y_idx, x_idx] += dist
        counts[y_idx, x_idx] += 1
    
    # Average
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap = np.where(counts > 0, heatmap / counts, 0)
    
    return {
        "analysis_id": analysis_id,
        "projection_axis": axis,
        "grid_size": n_bins,
        "x_range": [float(min_vals[0]), float(max_vals[0])],
        "y_range": [float(min_vals[1]), float(max_vals[1])],
        "heatmap": heatmap.tolist(),
        "max_deviation": float(np.max(heatmap)),
        "tolerance": field.tolerance,
    }


@router.get("/patterns")
async def get_pattern_types() -> Dict[str, Any]:
    """
    Get available deviation pattern types.
    
    Returns:
        List of pattern types with descriptions
    """
    patterns = {
        DeviationPattern.UNIFORM_OFFSET: "Deslocamento uniforme em uma direção",
        DeviationPattern.UNIFORM_SCALE: "Contração/expansão uniforme",
        DeviationPattern.DIRECTIONAL_TREND: "Tendência direcional (gradiente)",
        DeviationPattern.LOCAL_HOTSPOT: "Região concentrada de desvio",
        DeviationPattern.PERIODIC: "Padrão periódico (vibração)",
        DeviationPattern.RANDOM: "Ruído aleatório/variabilidade",
        DeviationPattern.WARPING: "Deformação/empenamento",
        DeviationPattern.TAPER: "Afilamento",
        DeviationPattern.TWIST: "Torção",
    }
    
    return {
        "patterns": [
            {"id": p.value, "name": p.name, "description": desc}
            for p, desc in patterns.items()
        ]
    }


@router.get("/root-causes")
async def get_root_cause_categories() -> Dict[str, Any]:
    """
    Get root cause categories and their corrective actions.
    
    Returns:
        List of categories with associated actions
    """
    categories = {
        RootCauseCategory.FIXTURING: "Problemas de fixação ou posicionamento",
        RootCauseCategory.CALIBRATION: "Erros de calibração de máquina",
        RootCauseCategory.TOOL_WEAR: "Desgaste de ferramenta de corte",
        RootCauseCategory.THERMAL: "Efeitos térmicos (expansão/contração)",
        RootCauseCategory.MATERIAL: "Variações de material",
        RootCauseCategory.VIBRATION: "Vibrações mecânicas",
        RootCauseCategory.PROGRAMMING: "Erros de programação NC/CAM",
        RootCauseCategory.MACHINE: "Problemas mecânicos de máquina",
    }
    
    return {
        "categories": [
            {"id": c.value, "name": c.name, "description": desc}
            for c, desc in categories.items()
        ]
    }



