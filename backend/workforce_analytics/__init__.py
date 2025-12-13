"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — WORKFORCE ANALYTICS MODULE
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Comprehensive workforce performance analysis, forecasting, and optimal assignment.

CAPABILITIES
════════════

1. PERFORMANCE ANALYSIS
   - Productivity metrics (units/time)
   - Efficiency relative to machine/team average
   - Saturation (utilization) analysis
   - Skill scoring based on historical performance
   - Learning curve modeling

2. FORECASTING
   - Short-term (7-30 days) productivity forecasting
   - ARIMA baseline
   - ML models (LSTM, Transformer) for advanced patterns
   - Confidence intervals with SNR

3. ASSIGNMENT OPTIMIZATION
   - MILP model for worker-operation assignment
   - Maximize skill-weighted productivity
   - Respect capacity, availability, qualifications

MATHEMATICAL FRAMEWORK
══════════════════════

Learning Curve (Wright's Law):
─────────────────────────────

    y(t) = a - b · exp(-c·t)

where:
    y(t) = productivity at time t
    a    = asymptotic (maximum) productivity
    b    = initial learning gap (a - y(0))
    c    = learning rate

SNR for Performance:
────────────────────

    SNR_perf = Var(μ_productivity) / Var(residual)

High SNR: Consistent, predictable performance
Low SNR: Highly variable, harder to forecast

R&D / SIFIDE ALIGNMENT
──────────────────────
Work Package 6: Workforce Intelligence
- Hypothesis H6.1: Learning curves explain 80%+ of productivity variance
- Hypothesis H6.2: MILP assignment improves throughput vs manual
- Experiment E6.1: Compare MILP vs heuristic worker assignment

REFERENCES
──────────
[1] Wright, T.P. (1936). Factors Affecting the Cost of Airplanes. Journal of Aeronautical Sciences.
[2] Argote, L. (2013). Organizational Learning: Creating, Retaining and Transferring Knowledge.
"""

from .workforce_performance_engine import (
    WorkerPerformance,
    WorkerMetrics,
    compute_worker_performance,
    compute_all_worker_performances,
    compute_learning_curve,
    fit_learning_curve,
    LearningCurveParams,
)
from .workforce_forecasting import (
    WorkforceForecast,
    ForecastConfig,
    forecast_worker_productivity,
    forecast_all_workers,
)
from .workforce_assignment_model import (
    Worker,
    Operation,
    WorkerAssignment,
    AssignmentPlan,
    optimize_worker_assignment,
    AssignmentConfig,
    build_operations_from_plan,
)

__all__ = [
    # Performance
    "WorkerPerformance",
    "WorkerMetrics",
    "compute_worker_performance",
    "compute_all_worker_performances",
    "compute_learning_curve",
    "fit_learning_curve",
    "LearningCurveParams",
    # Forecasting
    "WorkforceForecast",
    "ForecastConfig",
    "forecast_worker_productivity",
    "forecast_all_workers",
    # Assignment
    "Worker",
    "Operation",
    "WorkerAssignment",
    "AssignmentPlan",
    "optimize_worker_assignment",
    "AssignmentConfig",
    "build_operations_from_plan",
]

