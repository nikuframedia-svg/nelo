"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — REPORTING MODULE
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Reporting and comparison engine for planning scenarios.
"""

from .comparison_engine import (
    MachineMetrics,
    ComparisonMetrics,
    MetricDelta,
    ScenarioComparison,
    compute_scenario_metrics,
    compare_scenarios,
)

from .report_generator import (
    ExecutiveReport,
    TechnicalReport,
    ReportGenerator,
    generate_executive_report,
    generate_technical_explanation,
)

__all__ = [
    # Comparison
    "MachineMetrics",
    "ComparisonMetrics",
    "MetricDelta",
    "ScenarioComparison",
    "compute_scenario_metrics",
    "compare_scenarios",
    # Reports
    "ExecutiveReport",
    "TechnicalReport",
    "ReportGenerator",
    "generate_executive_report",
    "generate_technical_explanation",
]


