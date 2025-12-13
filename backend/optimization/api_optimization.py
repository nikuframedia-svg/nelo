"""
════════════════════════════════════════════════════════════════════════════════════════════════════
OPTIMIZATION API - REST Endpoints for Mathematical Optimization
════════════════════════════════════════════════════════════════════════════════════════════════════

Endpoints for optimization operations:
- POST /optimization/predict-time - Predict processing time
- POST /optimization/golden-runs - Record/retrieve golden runs
- POST /optimization/parameters/optimize - Optimize parameters
- POST /optimization/schedule - Solve scheduling problem
- POST /optimization/pareto - Generate Pareto frontier

R&D / SIFIDE: WP4 - Learning Scheduler & Advanced Optimization
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

from .math_optimization import (
    MathOptimizationService,
    ProcessFeatures,
    TimePrediction,
    GoldenRun,
    ParameterBounds,
    OptimizationResult,
    Job,
    Machine,
    Schedule,
    ParetoSolution,
    OptimizationObjective,
    SchedulingPriority,
    get_optimization_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/optimization", tags=["Optimization"])


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessFeaturesInput(BaseModel):
    """Input for time prediction."""
    product_id: str
    operation_id: str
    machine_id: str
    material_type: str = ""
    batch_size: float = Field(1, ge=0)
    speed_setting: float = Field(1.0, ge=0.1, le=2.0)
    temperature: float = Field(20.0)
    pressure: float = Field(1.0)
    shift: int = Field(1, ge=1, le=3)
    operator_experience: float = Field(1.0, ge=0, le=1)
    machine_age_hours: float = Field(0, ge=0)
    last_setup_hours: float = Field(0, ge=0)
    consecutive_runs: int = Field(0, ge=0)


class RecordRunInput(BaseModel):
    """Input for recording a production run."""
    product_id: str
    operation_id: str
    machine_id: str
    cycle_time_minutes: float = Field(..., gt=0)
    defect_rate: float = Field(..., ge=0, le=1)
    oee: float = Field(..., ge=0, le=1)
    parameters: Dict[str, float]
    context: Dict[str, Any] = Field(default_factory=dict)


class GoldenRunGapInput(BaseModel):
    """Input for gap calculation."""
    product_id: str
    operation_id: str
    machine_id: str
    current_cycle_time: float = Field(..., gt=0)
    current_oee: float = Field(..., ge=0, le=1)


class ParameterBoundsInput(BaseModel):
    """Input for parameter bounds."""
    name: str
    min_value: float
    max_value: float
    default_value: float
    step: float = 0.1
    unit: str = ""


class OptimizeParametersInput(BaseModel):
    """Input for parameter optimization."""
    parameter_bounds: List[ParameterBoundsInput]
    objective: str = Field("minimize_time", description="minimize_time, minimize_defects, balanced")


class JobInput(BaseModel):
    """Input for a scheduling job."""
    job_id: str
    product_id: str
    quantity: float = Field(1, ge=0)
    processing_time_minutes: float = Field(..., gt=0)
    setup_time_minutes: float = Field(0, ge=0)
    due_date: str  # ISO format
    release_date: Optional[str] = None
    priority: int = Field(1, ge=1, le=10)
    weight: float = Field(1.0, ge=0)
    required_machine: Optional[str] = None
    allowed_machines: List[str] = Field(default_factory=list)
    predecessor_jobs: List[str] = Field(default_factory=list)


class MachineInput(BaseModel):
    """Input for a machine."""
    machine_id: str
    name: str
    available_hours_per_day: float = Field(8, ge=0, le=24)
    efficiency: float = Field(0.85, ge=0, le=1)


class SolveScheduleInput(BaseModel):
    """Input for schedule solving."""
    jobs: List[JobInput]
    machines: List[MachineInput]
    priority: str = Field("optimized", description="fifo, edd, spt, wspt, optimized")
    horizon_hours: float = Field(168, ge=1, le=720)


class ParetoInput(BaseModel):
    """Input for Pareto optimization."""
    parameter_bounds: List[ParameterBoundsInput]
    population_size: int = Field(50, ge=10, le=200)
    generations: int = Field(50, ge=10, le=200)


class DemoInput(BaseModel):
    """Demo input."""
    num_jobs: int = Field(10, ge=3, le=50)
    num_machines: int = Field(3, ge=1, le=10)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO datetime string."""
    if not dt_str:
        return None
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/status")
async def get_status():
    """Get optimization module status."""
    service = get_optimization_service()
    
    return {
        "service": "Mathematical Optimization",
        "version": "1.0.0",
        "status": "operational",
        "engines": {
            "time_prediction": "ml_pytorch" if service.time_predictor.trained else "base_heuristic",
            "golden_runs": len(service.golden_runs.golden_runs),
            "scheduler": "cp_sat_available",
            "multi_objective": "nsga2",
        },
        "objectives": [o.value for o in OptimizationObjective],
        "scheduling_priorities": [p.value for p in SchedulingPriority],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TIME PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/predict-time")
async def predict_time(features: ProcessFeaturesInput):
    """Predict processing time for given features."""
    service = get_optimization_service()
    
    process_features = ProcessFeatures(
        product_id=features.product_id,
        operation_id=features.operation_id,
        machine_id=features.machine_id,
        material_type=features.material_type,
        batch_size=features.batch_size,
        speed_setting=features.speed_setting,
        temperature=features.temperature,
        pressure=features.pressure,
        shift=features.shift,
        operator_experience=features.operator_experience,
        machine_age_hours=features.machine_age_hours,
        last_setup_hours=features.last_setup_hours,
        consecutive_runs=features.consecutive_runs,
    )
    
    prediction = service.predict_time(process_features)
    
    return prediction.to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# GOLDEN RUNS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/golden-runs/record")
async def record_golden_run(data: RecordRunInput):
    """Record a production run (may become golden run)."""
    service = get_optimization_service()
    
    golden = service.record_run(
        product_id=data.product_id,
        operation_id=data.operation_id,
        machine_id=data.machine_id,
        cycle_time_minutes=data.cycle_time_minutes,
        defect_rate=data.defect_rate,
        oee=data.oee,
        parameters=data.parameters,
        context=data.context,
    )
    
    return {
        "recorded": True,
        "is_golden_run": golden is not None,
        "golden_run": golden.to_dict() if golden else None,
    }


@router.get("/golden-runs/{product_id}/{operation_id}/{machine_id}")
async def get_golden_run(product_id: str, operation_id: str, machine_id: str):
    """Get golden run for a combination."""
    service = get_optimization_service()
    
    golden = service.golden_runs.get_golden_run(product_id, operation_id, machine_id)
    
    if not golden:
        raise HTTPException(status_code=404, detail="No golden run found")
    
    return golden.to_dict()


@router.post("/golden-runs/gap")
async def calculate_gap(data: GoldenRunGapInput):
    """Calculate gap from golden run."""
    service = get_optimization_service()
    
    gap = service.get_golden_run_gap(
        product_id=data.product_id,
        operation_id=data.operation_id,
        machine_id=data.machine_id,
        current_cycle_time=data.current_cycle_time,
        current_oee=data.current_oee,
    )
    
    if not gap:
        return {"available": False, "message": "No golden run recorded"}
    
    return {"available": True, **gap}


@router.get("/golden-runs/recommendations/{product_id}/{operation_id}/{machine_id}")
async def get_recommendations(product_id: str, operation_id: str, machine_id: str):
    """Get recommendations based on golden run."""
    service = get_optimization_service()
    
    return service.golden_runs.get_recommendations(product_id, operation_id, machine_id)


@router.get("/golden-runs")
async def list_golden_runs(limit: int = Query(50, ge=1, le=200)):
    """List all golden runs."""
    service = get_optimization_service()
    
    runs = list(service.golden_runs.golden_runs.values())
    runs.sort(key=lambda r: r.recorded_at, reverse=True)
    
    return {
        "total": len(runs),
        "golden_runs": [r.to_dict() for r in runs[:limit]],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/parameters/optimize")
async def optimize_parameters(data: OptimizeParametersInput):
    """Optimize process parameters."""
    service = get_optimization_service()
    
    bounds = [
        ParameterBounds(
            name=b.name,
            min_value=b.min_value,
            max_value=b.max_value,
            default_value=b.default_value,
            step=b.step,
            unit=b.unit,
        )
        for b in data.parameter_bounds
    ]
    
    objective = OptimizationObjective(data.objective)
    
    result = service.optimize_parameters(bounds, objective)
    
    return result.to_dict()


@router.post("/parameters/demo")
async def demo_parameter_optimization():
    """Run demo parameter optimization."""
    service = get_optimization_service()
    
    bounds = [
        ParameterBounds(
            name="speed",
            min_value=0.5,
            max_value=2.0,
            default_value=1.0,
            unit="factor",
        ),
        ParameterBounds(
            name="temperature",
            min_value=50,
            max_value=150,
            default_value=100,
            unit="°C",
        ),
        ParameterBounds(
            name="pressure",
            min_value=0.5,
            max_value=2.0,
            default_value=1.0,
            unit="bar",
        ),
    ]
    
    result = service.optimize_parameters(bounds, OptimizationObjective.BALANCED)
    
    return {
        "demo": True,
        "result": result.to_dict(),
        "bounds_used": [b.to_dict() for b in bounds],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULING
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/schedule/solve")
async def solve_schedule(data: SolveScheduleInput):
    """Solve scheduling problem."""
    service = get_optimization_service()
    
    jobs = []
    for j in data.jobs:
        jobs.append(Job(
            job_id=j.job_id,
            product_id=j.product_id,
            quantity=j.quantity,
            processing_time_minutes=j.processing_time_minutes,
            setup_time_minutes=j.setup_time_minutes,
            due_date=parse_datetime(j.due_date) or datetime.now(timezone.utc) + timedelta(days=7),
            release_date=parse_datetime(j.release_date),
            priority=j.priority,
            weight=j.weight,
            required_machine=j.required_machine,
            allowed_machines=j.allowed_machines,
            predecessor_jobs=j.predecessor_jobs,
        ))
    
    machines = []
    for m in data.machines:
        machines.append(Machine(
            machine_id=m.machine_id,
            name=m.name,
            available_hours_per_day=m.available_hours_per_day,
            efficiency=m.efficiency,
        ))
    
    priority = SchedulingPriority(data.priority)
    
    schedule = service.solve_schedule(jobs, machines, priority)
    
    return schedule.to_dict()


@router.post("/schedule/demo")
async def demo_scheduling(data: DemoInput = Body(...)):
    """Run demo scheduling."""
    service = get_optimization_service()
    
    # Generate demo jobs
    jobs = []
    base_time = datetime.now(timezone.utc)
    
    for i in range(data.num_jobs):
        jobs.append(Job(
            job_id=f"JOB-{i+1:03d}",
            product_id=f"PROD-{(i % 3) + 1}",
            quantity=10 + i * 5,
            processing_time_minutes=30 + (i % 5) * 10,
            setup_time_minutes=10 + (i % 3) * 5,
            due_date=base_time + timedelta(hours=24 + i * 8),
            priority=(i % 3) + 1,
            weight=1.0 + (i % 3) * 0.5,
        ))
    
    # Generate demo machines
    machines = []
    for i in range(data.num_machines):
        machines.append(Machine(
            machine_id=f"MC-{i+1:02d}",
            name=f"Machine {i+1}",
            available_hours_per_day=8,
            efficiency=0.80 + i * 0.05,
        ))
    
    # Solve with different methods
    results = {}
    
    # Heuristic
    heuristic_schedule = service.solve_schedule(jobs, machines, SchedulingPriority.EDD)
    results["edd"] = {
        "tardiness": heuristic_schedule.total_tardiness,
        "makespan": heuristic_schedule.total_makespan_minutes,
        "solve_time": heuristic_schedule.solve_time_seconds,
    }
    
    # Optimized
    try:
        optimized_schedule = service.solve_schedule(jobs, machines, SchedulingPriority.OPTIMIZED)
        results["optimized"] = {
            "tardiness": optimized_schedule.total_tardiness,
            "makespan": optimized_schedule.total_makespan_minutes,
            "solve_time": optimized_schedule.solve_time_seconds,
        }
        main_schedule = optimized_schedule
    except Exception as e:
        results["optimized"] = {"error": str(e)}
        main_schedule = heuristic_schedule
    
    return {
        "demo": True,
        "jobs_count": len(jobs),
        "machines_count": len(machines),
        "comparison": results,
        "schedule": main_schedule.to_dict(),
    }


@router.post("/schedule/compare")
async def compare_scheduling_methods(data: SolveScheduleInput):
    """Compare different scheduling methods."""
    service = get_optimization_service()
    
    jobs = []
    for j in data.jobs:
        jobs.append(Job(
            job_id=j.job_id,
            product_id=j.product_id,
            quantity=j.quantity,
            processing_time_minutes=j.processing_time_minutes,
            setup_time_minutes=j.setup_time_minutes,
            due_date=parse_datetime(j.due_date) or datetime.now(timezone.utc) + timedelta(days=7),
            release_date=parse_datetime(j.release_date),
            priority=j.priority,
            weight=j.weight,
        ))
    
    machines = []
    for m in data.machines:
        machines.append(Machine(
            machine_id=m.machine_id,
            name=m.name,
            available_hours_per_day=m.available_hours_per_day,
            efficiency=m.efficiency,
        ))
    
    results = {}
    for priority in [SchedulingPriority.FIFO, SchedulingPriority.EDD, SchedulingPriority.SPT]:
        schedule = service.solve_schedule(jobs, machines, priority)
        results[priority.value] = {
            "tardiness": schedule.total_tardiness,
            "makespan": schedule.total_makespan_minutes,
            "utilization": schedule.machine_utilization,
        }
    
    try:
        schedule = service.solve_schedule(jobs, machines, SchedulingPriority.OPTIMIZED)
        results["optimized"] = {
            "tardiness": schedule.total_tardiness,
            "makespan": schedule.total_makespan_minutes,
            "utilization": schedule.machine_utilization,
        }
    except Exception as e:
        results["optimized"] = {"error": str(e)}
    
    return {"comparison": results}


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-OBJECTIVE OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/pareto/optimize")
async def pareto_optimize(data: ParetoInput):
    """Generate Pareto frontier."""
    service = get_optimization_service()
    
    # Update config
    service.config.pareto_population_size = data.population_size
    service.config.pareto_generations = data.generations
    
    bounds = [
        ParameterBounds(
            name=b.name,
            min_value=b.min_value,
            max_value=b.max_value,
            default_value=b.default_value,
        )
        for b in data.parameter_bounds
    ]
    
    solutions = service.optimize_pareto(bounds)
    
    return {
        "pareto_solutions": len(solutions),
        "solutions": [s.to_dict() for s in solutions],
    }


@router.post("/pareto/demo")
async def demo_pareto():
    """Run demo Pareto optimization."""
    service = get_optimization_service()
    
    bounds = [
        ParameterBounds("speed", 0.5, 2.0, 1.0),
        ParameterBounds("temperature", 50, 150, 100),
    ]
    
    solutions = service.optimize_pareto(bounds)
    
    return {
        "demo": True,
        "pareto_solutions": len(solutions),
        "solutions": [s.to_dict() for s in solutions[:20]],  # Limit for demo
        "trade_offs": [
            {"description": "Fast but risky", "time": min(s.objectives.get("time", 999) for s in solutions)},
            {"description": "Slow but safe", "defect_rate": min(s.objectives.get("defect_rate", 999) for s in solutions)},
        ],
    }



