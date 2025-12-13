"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — PROJECT PLANNING MODULE
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Supervising layer for project-level production planning.

A PROJECT is a higher-level grouping of production orders:
- An obra (construction project)
- A customer order batch
- A product line campaign
- A contractual delivery

This module provides:
1. Project data model and aggregation
2. Project load and risk computation
3. Multi-project priority optimization (MILP)
4. Project-level KPIs

ARCHITECTURE
════════════

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    PROJECT PLANNING LAYER                                │
    │                                                                          │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
    │  │ project_model│  │ load_engine  │  │ priority_opt │  │ kpi_engine   │ │
    │  │              │  │              │  │              │  │              │ │
    │  │ • Project    │  │ • load_calc  │  │ • MILP       │  │ • lead_time  │ │
    │  │ • builder    │  │ • slack      │  │ • priority   │  │ • atraso     │ │
    │  │ • grouping   │  │ • risk       │  │ • feasibility│  │ • risk_score │ │
    │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │
    └────────────────────────────────┬────────────────────────────────────────┘
                                     │
    ┌────────────────────────────────▼────────────────────────────────────────┐
    │                           APS SCHEDULER                                  │
    │              (receives project-weighted priorities)                      │
    └─────────────────────────────────────────────────────────────────────────┘

R&D / SIFIDE ALIGNMENT
──────────────────────
Work Package 5: Project-Level Planning
- Hypothesis H5.1: Project-level optimization improves OTD vs order-level
- Hypothesis H5.2: Risk-based project ranking reduces late deliveries
- Experiment E5.1: Compare project-weighted vs equal-priority scheduling

REFERENCES
──────────
[1] PMI (2021). A Guide to the Project Management Body of Knowledge (PMBOK).
[2] Herroelen & Leus (2005). Project scheduling under uncertainty. EJOR.
"""

from .project_model import (
    Project,
    ProjectStatus,
    build_projects_from_orders,
    group_orders_by_project,
    AggregationMode,
)
from .project_load_engine import (
    ProjectLoad,
    compute_project_load,
    compute_project_slack,
    compute_project_risk,
    ProjectRiskScore,
    compute_all_project_loads,
    compute_all_project_risks,
)
from .project_priority_optimization import (
    ProjectPriorityOptimizer,
    PriorityPlan,
    PriorityPlanConfig,
    optimize_project_priorities,
)
from .project_kpi_engine import (
    ProjectKPIs,
    GlobalProjectKPIs,
    compute_project_kpis,
    compute_global_project_kpis,
    compute_all_project_kpis,
)

__all__ = [
    # Model
    "Project",
    "ProjectStatus",
    "AggregationMode",
    "build_projects_from_orders",
    "group_orders_by_project",
    # Load
    "ProjectLoad",
    "compute_project_load",
    "compute_project_slack",
    "compute_project_risk",
    "compute_all_project_loads",
    "compute_all_project_risks",
    "ProjectRiskScore",
    # Optimization
    "ProjectPriorityOptimizer",
    "PriorityPlan",
    "PriorityPlanConfig",
    "optimize_project_priorities",
    # KPIs
    "ProjectKPIs",
    "GlobalProjectKPIs",
    "compute_project_kpis",
    "compute_global_project_kpis",
    "compute_all_project_kpis",
]

