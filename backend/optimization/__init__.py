"""
Optimization Module - Mathematical & ML Optimization
=====================================================

Components:
- Time Prediction: ML-based processing time prediction
- Golden Runs: Optimal performance benchmarking
- Parameter Optimization: Bayesian/GA optimization
- Scheduling: CP-SAT advanced scheduling
- Multi-Objective: NSGA-II Pareto optimization

R&D / SIFIDE: WP4 - Learning Scheduler & Advanced Optimization
"""

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
    ScheduledJob,
    ParetoSolution,
    OptimizationObjective,
    SchedulingPriority,
    TimePredictionEngineML,
    GoldenRunsEngine,
    ProcessParameterOptimizer,
    SchedulingSolver,
    MultiObjectiveOptimizer,
    get_optimization_service,
)

from .api_optimization import router as optimization_router

__all__ = [
    "MathOptimizationService",
    "ProcessFeatures",
    "TimePrediction",
    "GoldenRun",
    "ParameterBounds",
    "OptimizationResult",
    "Job",
    "Machine",
    "Schedule",
    "ScheduledJob",
    "ParetoSolution",
    "OptimizationObjective",
    "SchedulingPriority",
    "TimePredictionEngineML",
    "GoldenRunsEngine",
    "ProcessParameterOptimizer",
    "SchedulingSolver",
    "MultiObjectiveOptimizer",
    "get_optimization_service",
    "optimization_router",
]
