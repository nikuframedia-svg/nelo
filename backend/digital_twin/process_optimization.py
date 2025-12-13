"""
════════════════════════════════════════════════════════════════════════════════════════════════════
Process Optimization - Golden Runs & Parameter Suggestions
════════════════════════════════════════════════════════════════════════════════════════════════════

Contract 6 + Contract 9 Implementation: Golden Runs + Process Optimization

Features:
- Identify golden runs from historical production data
- Suggest optimal process parameters
- Track quality vs process parameter correlations
- Contract 9: Use OperationExecutionLog for golden run computation
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
from sqlalchemy.orm import Session
from digital_twin.sqlalchemy import and_

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# GOLDEN RUN MODEL (Already defined in dpp_models.py)
# ═══════════════════════════════════════════════════════════════════════════════

def get_golden_run_model():
    """Get GoldenRun model dynamically to avoid circular imports."""
    # Import both pdm_models and dpp_models to resolve relationships
    from duplios import pdm_models, dpp_models
    from duplios.dpp_models import GoldenRun
    return GoldenRun


# ═══════════════════════════════════════════════════════════════════════════════
# GOLDEN RUN IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_golden_runs(
    revision_id: int,
    operation_id: Optional[int] = None,
    db: Session = None,
    quality_threshold: float = 85.0,
) -> Dict[str, Any]:
    """
    Compute golden runs from historical conformance data.
    
    A golden run is a production run where:
    - Quality score is above threshold
    - Process parameters are recorded
    - Can be used as reference for future production
    
    Args:
        revision_id: Item revision
        operation_id: Optional specific operation
        db: Database session
        quality_threshold: Minimum quality score for golden run
    
    Returns:
        Dict with count, best_score, etc.
    """
    # Import both pdm_models and dpp_models to resolve relationships
    from duplios import pdm_models as _pdm, dpp_models as _dpp
    from duplios.dpp_models import ProductConformanceSnapshot, ConformityStatus, GoldenRun
    
    if not db:
        logger.warning("No database session provided for golden run computation")
        return {"count": 0, "best_score": 0}
    
    # Query conformance snapshots with good quality
    good_snapshots = db.query(ProductConformanceSnapshot).filter(
        and_(
            ProductConformanceSnapshot.revision_id == revision_id,
            ProductConformanceSnapshot.conformity_status == ConformityStatus.IN_TOLERANCE,
            ProductConformanceSnapshot.scalar_error_score < (100 - quality_threshold),
        )
    ).all()
    
    if not good_snapshots:
        logger.info(f"No good snapshots found for revision {revision_id}")
        # Create synthetic golden runs for demo
        return _create_demo_golden_runs(revision_id, operation_id, db)
    
    # Group by machine_id and find best parameters
    machine_runs: Dict[str, List] = {}
    for snapshot in good_snapshots:
        machine_id = snapshot.machine_id or "unknown"
        if machine_id not in machine_runs:
            machine_runs[machine_id] = []
        machine_runs[machine_id].append(snapshot)
    
    golden_runs_created = 0
    best_score = 0.0
    
    for machine_id, snapshots in machine_runs.items():
        # Sort by quality score (lower is better)
        snapshots.sort(key=lambda s: s.scalar_error_score)
        best = snapshots[0]
        
        # Create golden run record
        params = json.loads(best.process_params) if best.process_params else {}
        score = 100.0 - best.scalar_error_score
        
        if score > best_score:
            best_score = score
        
        # Check if golden run exists
        existing = db.query(GoldenRun).filter(
            and_(
                GoldenRun.revision_id == revision_id,
                GoldenRun.machine_id == machine_id,
            )
        ).first()
        
        if existing:
            # Update if better
            if score > existing.score:
                existing.process_params = json.dumps(params)
                existing.score = score
                existing.kpis = json.dumps({
                    "error_score": best.scalar_error_score,
                    "max_dev": best.max_dev,
                    "mean_dev": best.mean_dev,
                })
                existing.updated_at = datetime.utcnow()
        else:
            # Create new
            golden_run = GoldenRun(
                revision_id=revision_id,
                operation_id=operation_id,
                machine_id=machine_id,
                process_params=json.dumps(params),
                score=score,
                kpis=json.dumps({
                    "error_score": best.scalar_error_score,
                    "max_dev": best.max_dev,
                    "mean_dev": best.mean_dev,
                }),
            )
            db.add(golden_run)
            golden_runs_created += 1
    
    db.commit()
    
    return {
        "count": golden_runs_created,
        "best_score": round(best_score, 1),
        "machines_analyzed": len(machine_runs),
    }


def _create_demo_golden_runs(
    revision_id: int,
    operation_id: Optional[int],
    db: Session,
) -> Dict[str, Any]:
    """Create demo golden runs when no historical data exists."""
    from duplios import pdm_models as _pdm, dpp_models as _dpp
    from duplios.dpp_models import GoldenRun
    
    demo_machines = ["CNC-01", "CNC-02", "MILL-01"]
    
    for i, machine_id in enumerate(demo_machines):
        existing = db.query(GoldenRun).filter(
            and_(
                GoldenRun.revision_id == revision_id,
                GoldenRun.machine_id == machine_id,
            )
        ).first()
        
        if not existing:
            golden_run = GoldenRun(
                revision_id=revision_id,
                operation_id=operation_id,
                machine_id=machine_id,
                process_params=json.dumps({
                    "feed_rate": 100 + i * 10,
                    "spindle_speed": 3000 + i * 500,
                    "depth_of_cut": 0.5 + i * 0.1,
                    "coolant_flow": 10 + i,
                }),
                score=92.0 - i * 2,
                kpis=json.dumps({
                    "cycle_time": 120 + i * 10,
                    "tool_wear": 0.05 + i * 0.01,
                }),
            )
            db.add(golden_run)
    
    db.commit()
    
    return {
        "count": len(demo_machines),
        "best_score": 92.0,
        "demo": True,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GOLDEN RUN RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════════

def get_golden_runs(
    revision_id: int,
    operation_id: Optional[int] = None,
    machine_id: Optional[str] = None,
    limit: int = 10,
    db: Session = None,
) -> List:
    """Get golden runs for a revision."""
    from duplios import pdm_models as _pdm, dpp_models as _dpp
    from duplios.dpp_models import GoldenRun
    
    if not db:
        return []
    
    query = db.query(GoldenRun).filter(GoldenRun.revision_id == revision_id)
    
    if operation_id:
        query = query.filter(GoldenRun.operation_id == operation_id)
    
    if machine_id:
        query = query.filter(GoldenRun.machine_id == machine_id)
    
    return query.order_by(GoldenRun.score.desc()).limit(limit).all()


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER SUGGESTION
# ═══════════════════════════════════════════════════════════════════════════════

def suggest_process_params(
    revision_id: int,
    operation_id: Optional[int] = None,
    machine_id: Optional[str] = None,
    db: Session = None,
) -> Dict[str, Any]:
    """
    Suggest optimal process parameters based on golden runs.
    
    Strategy:
    1. Get golden runs for the revision/operation/machine
    2. Average parameters weighted by score
    3. Return suggested parameters
    """
    golden_runs = get_golden_runs(
        revision_id=revision_id,
        operation_id=operation_id,
        machine_id=machine_id,
        limit=5,
        db=db,
    )
    
    if not golden_runs:
        # Return default parameters
        return {
            "params": _get_default_params(),
            "based_on": 0,
            "expected_score": 75.0,
            "note": "No golden runs found, using defaults",
        }
    
    # Weighted average of parameters
    total_weight = 0.0
    param_sums: Dict[str, float] = {}
    expected_score = 0.0
    
    for run in golden_runs:
        params = json.loads(run.process_params) if run.process_params else {}
        weight = run.score / 100.0
        
        for key, value in params.items():
            if isinstance(value, (int, float)):
                if key not in param_sums:
                    param_sums[key] = 0.0
                param_sums[key] += value * weight
        
        total_weight += weight
        expected_score += run.score * weight
    
    if total_weight > 0:
        suggested_params = {k: round(v / total_weight, 2) for k, v in param_sums.items()}
        expected_score = expected_score / total_weight
    else:
        suggested_params = _get_default_params()
        expected_score = 75.0
    
    return {
        "params": suggested_params,
        "based_on": len(golden_runs),
        "expected_score": round(expected_score, 1),
    }


def _get_default_params() -> Dict[str, float]:
    """Get default process parameters."""
    return {
        "feed_rate": 100.0,
        "spindle_speed": 3000.0,
        "depth_of_cut": 0.5,
        "coolant_flow": 10.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESS PARAMETER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_parameter_impact(
    revision_id: int,
    parameter_name: str,
    db: Session = None,
) -> Dict[str, Any]:
    """
    Analyze impact of a process parameter on quality.
    
    Returns correlation and optimal range.
    """
    from duplios.dpp_models import ProductConformanceSnapshot
    
    if not db:
        return {"error": "No database session"}
    
    snapshots = db.query(ProductConformanceSnapshot).filter(
        ProductConformanceSnapshot.revision_id == revision_id,
        ProductConformanceSnapshot.process_params.isnot(None),
    ).limit(100).all()
    
    if len(snapshots) < 10:
        return {
            "parameter": parameter_name,
            "correlation": 0.0,
            "optimal_min": None,
            "optimal_max": None,
            "data_points": len(snapshots),
            "note": "Insufficient data for analysis",
        }
    
    # Extract parameter values and quality scores
    param_values = []
    quality_scores = []
    
    for s in snapshots:
        params = json.loads(s.process_params) if s.process_params else {}
        if parameter_name in params:
            param_values.append(params[parameter_name])
            quality_scores.append(100.0 - s.scalar_error_score)
    
    if len(param_values) < 10:
        return {
            "parameter": parameter_name,
            "correlation": 0.0,
            "data_points": len(param_values),
            "note": f"Parameter {parameter_name} not found in most records",
        }
    
    # Calculate correlation
    correlation = float(np.corrcoef(param_values, quality_scores)[0, 1])
    
    # Find optimal range (values associated with top 20% quality)
    quality_threshold = np.percentile(quality_scores, 80)
    good_indices = [i for i, q in enumerate(quality_scores) if q >= quality_threshold]
    good_values = [param_values[i] for i in good_indices]
    
    return {
        "parameter": parameter_name,
        "correlation": round(correlation, 3),
        "optimal_min": round(min(good_values), 2) if good_values else None,
        "optimal_max": round(max(good_values), 2) if good_values else None,
        "mean_value": round(np.mean(param_values), 2),
        "data_points": len(param_values),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def predict_quality(
    revision_id: int,
    process_params: Dict[str, float],
    machine_id: Optional[str] = None,
    db: Session = None,
) -> Dict[str, Any]:
    """
    Predict quality score for given process parameters.
    
    Uses a simple regression model based on golden runs.
    """
    golden_runs = get_golden_runs(
        revision_id=revision_id,
        machine_id=machine_id,
        limit=20,
        db=db,
    )
    
    if len(golden_runs) < 3:
        # Heuristic prediction
        default_params = _get_default_params()
        deviation = 0.0
        for key, value in process_params.items():
            if key in default_params:
                deviation += abs(value - default_params[key]) / (default_params[key] + 1)
        
        predicted_score = max(50.0, 90.0 - deviation * 10)
        
        return {
            "predicted_score": round(predicted_score, 1),
            "confidence": 0.3,
            "method": "heuristic",
            "note": "Limited historical data",
        }
    
    # Find most similar golden run
    min_distance = float("inf")
    best_score = 80.0
    
    for run in golden_runs:
        run_params = json.loads(run.process_params) if run.process_params else {}
        
        # Calculate Euclidean distance
        distance = 0.0
        for key, value in process_params.items():
            if key in run_params:
                normalized_diff = (value - run_params[key]) / (run_params[key] + 1)
                distance += normalized_diff ** 2
        
        distance = np.sqrt(distance)
        
        if distance < min_distance:
            min_distance = distance
            best_score = run.score
    
    # Adjust score based on distance from golden run
    predicted_score = best_score - min_distance * 5
    predicted_score = max(40.0, min(100.0, predicted_score))
    
    confidence = max(0.3, 1.0 - min_distance)
    
    return {
        "predicted_score": round(predicted_score, 1),
        "confidence": round(confidence, 2),
        "method": "similarity",
        "closest_golden_run_score": round(best_score, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH OPERATION EXECUTION LOGS (CONTRACT 9)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_golden_runs_from_logs(
    revision_id: int,
    operation_id: Optional[int] = None,
    machine_id: Optional[str] = None,
    min_qty: int = 5,
    max_scrap_rate: float = 0.05,
    top_n: int = 10,
    db: Session = None,
) -> List[GoldenRun]:
    """
    Compute Golden Runs from OperationExecutionLog data.
    
    Selects executions with low scrap, good cycle times, and reasonable energy.
    Returns top N as GoldenRun records.
    
    Args:
        revision_id: Product revision ID
        operation_id: Optional operation ID filter
        machine_id: Optional machine ID filter
        min_qty: Minimum quantity produced for consideration
        max_scrap_rate: Maximum scrap rate (qty_scrap / (qty_good + qty_scrap))
        top_n: Number of top runs to save
        db: Database session for GoldenRun (uses duplios.db)
        
    Returns:
        List of created GoldenRun records
    """
    from prodplan.execution_log_models import (
        query_execution_logs,
        ExecutionLogQuery,
        ExecutionLogStatus,
    )
    from duplios import pdm_models as _pdm, dpp_models as _dpp
    from duplios.dpp_models import GoldenRun
    
    # Get database session for GoldenRun records
    if not db:
        try:
            from duplios.models import SessionLocal
            db = SessionLocal()
        except ImportError:
            logger.error("Could not import SessionLocal")
            return []
        should_close = True
    else:
        should_close = False
    
    try:
        # Query completed executions from the execution_logs SQLite database
        query = ExecutionLogQuery(
            revision_id=revision_id,
            status=ExecutionLogStatus.COMPLETED,
            limit=500,
        )
        if operation_id:
            query.operation_id = str(operation_id)
        if machine_id:
            query.machine_id = machine_id
        
        executions = query_execution_logs(query)
        
        # Filter by min_qty
        executions = [e for e in executions if (e.qty_good + e.qty_scrap) >= min_qty]
        
        if not executions:
            logger.info(f"No executions found for revision {revision_id}")
            return []
        
        # Calculate scores for each execution
        scored_executions = []
        
        for exec_log in executions:
            # Calculate scrap rate
            total_qty = exec_log.qty_good + exec_log.qty_scrap
            if total_qty == 0:
                continue
            scrap_rate = exec_log.qty_scrap / total_qty
            
            if scrap_rate > max_scrap_rate:
                continue
            
            # Calculate score (higher is better)
            cycle_time_per_unit = (
                exec_log.cycle_time_s / exec_log.qty_good
                if exec_log.cycle_time_s and exec_log.qty_good > 0
                else float("inf")
            )
            energy_per_unit = (
                exec_log.energy_kwh / exec_log.qty_good
                if exec_log.energy_kwh and exec_log.qty_good > 0
                else 0.0
            )
            
            # Score calculation
            score = 100.0
            if cycle_time_per_unit < float("inf"):
                score -= max(0, (cycle_time_per_unit - 30) * 0.5)
            score -= scrap_rate * 100
            if energy_per_unit > 0:
                score -= max(0, (energy_per_unit - 0.5) * 10)
            score = max(0.0, min(100.0, score))
            
            # Convert params to JSON string
            params_json = json.dumps(exec_log.params.dict() if exec_log.params else {})
            
            scored_executions.append({
                "log": exec_log,
                "score": score,
                "scrap_rate": scrap_rate,
                "params_json": params_json,
            })
        
        # Sort by score descending
        scored_executions.sort(key=lambda x: x["score"], reverse=True)
        
        # Take top N and create GoldenRun records
        created_runs = []
        
        for item in scored_executions[:top_n]:
            exec_log = item["log"]
            op_id = int(exec_log.operation_id) if exec_log.operation_id.isdigit() else 0
            
            # Check if golden run already exists
            existing = db.query(GoldenRun).filter(
                GoldenRun.revision_id == revision_id,
                GoldenRun.operation_id == op_id,
                GoldenRun.machine_id == exec_log.machine_id,
            ).first()
            
            # Only update if new score is better
            if existing and existing.score >= item["score"]:
                continue
            
            if existing:
                # Update existing
                existing.process_params = item["params_json"]
                existing.score = item["score"]
                existing.cycle_time = exec_log.cycle_time_s
                existing.yield_rate = 1.0 - item["scrap_rate"]
                existing.energy_consumption = exec_log.energy_kwh
                existing.source_log_id = exec_log.id
                created_runs.append(existing)
            else:
                # Create new
                golden_run = GoldenRun(
                    revision_id=revision_id,
                    operation_id=op_id,
                    machine_id=exec_log.machine_id or "unknown",
                    process_params=item["params_json"],
                    score=item["score"],
                    cycle_time=exec_log.cycle_time_s,
                    yield_rate=1.0 - item["scrap_rate"],
                    energy_consumption=exec_log.energy_kwh,
                    source_log_id=exec_log.id,
                )
                db.add(golden_run)
                created_runs.append(golden_run)
        
        db.commit()
        
        logger.info(
            f"Computed {len(created_runs)} golden runs from {len(executions)} executions "
            f"for revision {revision_id}"
        )
        
        return created_runs
        
    except Exception as e:
        logger.error(f"Error computing golden runs from logs: {e}")
        db.rollback()
        return []
    finally:
        if should_close:
            db.close()


def suggest_process_params_from_logs(
    revision_id: int,
    operation_id: int,
    machine_id: Optional[str] = None,
    db: Session = None,
) -> Dict[str, Any]:
    """
    Suggest optimal process parameters based on execution logs and golden runs.
    
    First checks for existing golden runs, then computes from logs if needed.
    
    Args:
        revision_id: Product revision ID
        operation_id: Operation ID
        machine_id: Optional machine ID
        db: Database session
        
    Returns:
        Dict with suggested parameters and metadata
    """
    if not db:
        try:
            from duplios.models import SessionLocal
            db = SessionLocal()
        except ImportError:
            logger.error("Could not import SessionLocal")
            return {
                "params": _get_default_params(),
                "source": "default",
                "confidence": 0.3,
                "note": "No database available",
            }
        should_close = True
    else:
        should_close = False
    
    try:
        # First check for existing golden runs
        golden_runs = get_golden_runs(
            revision_id=revision_id,
            operation_id=operation_id,
            machine_id=machine_id,
            db=db,
        )
        
        if not golden_runs:
            # Try to compute from logs
            computed = compute_golden_runs_from_logs(
                revision_id=revision_id,
                operation_id=operation_id,
                machine_id=machine_id,
                db=db,
            )
            if computed:
                golden_runs = computed
        
        if not golden_runs:
            # Fall back to default parameters
            return {
                "params": _get_default_params(),
                "source": "default",
                "confidence": 0.3,
                "note": "No historical data available, using defaults",
            }
        
        # Use best golden run
        best_run = max(golden_runs, key=lambda r: r.score)
        params = json.loads(best_run.process_params) if best_run.process_params else {}
        
        return {
            "params": params,
            "source": "golden_run",
            "golden_run_id": best_run.id,
            "score": round(best_run.score, 1),
            "yield_rate": round(best_run.yield_rate * 100, 1) if best_run.yield_rate else None,
            "confidence": min(0.9, best_run.score / 100),
            "note": f"Based on golden run with score {best_run.score:.1f}",
        }
        
    finally:
        if should_close:
            db.close()
