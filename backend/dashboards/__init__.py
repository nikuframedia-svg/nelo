"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — DASHBOARDS & VISUALIZATIONS
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Advanced visualization data generators:
- Comparative Gantt charts
- Hourly utilization heatmaps
- Operator dashboards
- Machine OEE dashboards
- Chained cell performance
- Annual capacity projections
"""

from .gantt_comparison import (
    generate_comparative_gantt_data,
    GanttComparisonData,
    GanttBar,
)

from .utilization_heatmap import (
    generate_utilization_heatmap,
    HeatmapData,
    HeatmapCell,
)

from .operator_dashboard import (
    generate_operator_dashboard,
    OperatorDashboard,
    OperatorMetrics,
)

from .machine_oee import (
    generate_machine_oee_dashboard,
    MachineDashboard,
    MachineOEE,
)

from .cell_performance import (
    generate_cell_performance,
    CellPerformance,
    CellMetrics,
)

from .capacity_projection import (
    generate_capacity_projection,
    CapacityProjection,
    MonthlyProjection,
)

__all__ = [
    "generate_comparative_gantt_data", "GanttComparisonData", "GanttBar",
    "generate_utilization_heatmap", "HeatmapData", "HeatmapCell",
    "generate_operator_dashboard", "OperatorDashboard", "OperatorMetrics",
    "generate_machine_oee_dashboard", "MachineDashboard", "MachineOEE",
    "generate_cell_performance", "CellPerformance", "CellMetrics",
    "generate_capacity_projection", "CapacityProjection", "MonthlyProjection",
]


