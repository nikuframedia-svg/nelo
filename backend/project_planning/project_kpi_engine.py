"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — PROJECT KPI ENGINE
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Computes Key Performance Indicators (KPIs) at the project level.

KPI DEFINITIONS
═══════════════

Per-Project KPIs:
─────────────────

1. Lead Time (LT_p):
   Time from first operation start to last operation end.
   
       LT_p = max(end_time) - min(start_time)   ∀ ops ∈ O(p)

2. Delay (D_p):
   Positive deviation from due date.
   
       D_p = max(0, max(end_time) - d_p)

3. Completion Rate (CR_p):
   Percentage of orders completed.
   
       CR_p = |{o ∈ O(p) : completed(o)}| / |O(p)|

4. Machine Utilization per Project (U_p^m):
   Hours used on machine m for project p.
   
       U_p^m = Σ_{op : machine(op)=m, order(op) ∈ O(p)} duration(op)

5. Bottleneck Contribution (BC_p):
   Percentage of bottleneck load from this project.
   
       BC_p = U_p^{bottleneck} / U_total^{bottleneck}

6. Value Density (VD_p):
   Priority per hour of load.
   
       VD_p = w_p / L_p

Global KPIs:
────────────

1. Project OTD (On-Time Delivery):
   Percentage of projects delivered on time.
   
       OTD = |{p : D_p = 0}| / |P|

2. Weighted OTD:
   Priority-weighted on-time rate.
   
       WOTD = Σ_{p : D_p = 0} w_p / Σ_p w_p

3. Average Project Lead Time:
   
       Avg_LT = (Σ_p LT_p) / |P|

4. Total Delay:
   
       Total_Delay = Σ_p D_p

5. Projects at Risk:
   Number of projects with risk_level ∈ {HIGH, CRITICAL}

R&D / SIFIDE: WP5 - Project Planning
────────────────────────────────────
- Experiment E5.4: Validate KPI correlation with business outcomes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .project_model import Project, ProjectStatus
from .project_load_engine import ProjectLoad, ProjectSlack, ProjectRiskScore

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class ProjectKPIs:
    """
    KPIs for a single project.
    """
    project_id: str
    project_name: str
    client: Optional[str] = None
    
    # Time KPIs
    lead_time_hours: float = 0.0
    lead_time_days: float = 0.0
    delay_hours: float = 0.0
    delay_days: float = 0.0
    slack_hours: float = 0.0
    
    # Completion KPIs
    total_orders: int = 0
    completed_orders: int = 0
    completion_rate_pct: float = 0.0
    
    # Load KPIs
    total_load_hours: float = 0.0
    machine_loads: Dict[str, float] = field(default_factory=dict)
    bottleneck_contribution_pct: float = 0.0
    
    # Value KPIs
    priority_weight: float = 1.0
    value_density: float = 0.0  # priority / load
    
    # Risk KPIs
    risk_score: float = 0.0
    risk_level: str = "LOW"
    probability_late_pct: float = 0.0
    
    # Status
    status: str = "planning"
    is_on_time: bool = True
    
    # Dates
    due_date: Optional[str] = None
    estimated_completion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'project_id': self.project_id,
            'project_name': self.project_name,
            'client': self.client,
            'lead_time_hours': round(self.lead_time_hours, 2),
            'lead_time_days': round(self.lead_time_days, 2),
            'delay_hours': round(self.delay_hours, 2),
            'delay_days': round(self.delay_days, 2),
            'slack_hours': round(self.slack_hours, 2),
            'total_orders': self.total_orders,
            'completed_orders': self.completed_orders,
            'completion_rate_pct': round(self.completion_rate_pct, 1),
            'total_load_hours': round(self.total_load_hours, 2),
            'machine_loads': {k: round(v, 2) for k, v in self.machine_loads.items()},
            'bottleneck_contribution_pct': round(self.bottleneck_contribution_pct, 1),
            'priority_weight': round(self.priority_weight, 2),
            'value_density': round(self.value_density, 3),
            'risk_score': round(self.risk_score, 1),
            'risk_level': self.risk_level,
            'probability_late_pct': round(self.probability_late_pct, 1),
            'status': self.status,
            'is_on_time': self.is_on_time,
            'due_date': self.due_date,
            'estimated_completion': self.estimated_completion,
        }


@dataclass
class GlobalProjectKPIs:
    """
    Global KPIs across all projects.
    """
    timestamp: str
    
    # Counts
    total_projects: int = 0
    projects_on_time: int = 0
    projects_delayed: int = 0
    projects_at_risk: int = 0
    projects_completed: int = 0
    
    # Rates
    otd_pct: float = 0.0
    weighted_otd_pct: float = 0.0
    at_risk_pct: float = 0.0
    
    # Averages
    avg_lead_time_hours: float = 0.0
    avg_lead_time_days: float = 0.0
    avg_delay_hours: float = 0.0
    avg_completion_rate_pct: float = 0.0
    
    # Totals
    total_delay_hours: float = 0.0
    total_load_hours: float = 0.0
    total_priority_weight: float = 0.0
    
    # Bottleneck
    bottleneck_machine: Optional[str] = None
    bottleneck_load_hours: float = 0.0
    
    # Top projects
    top_delayed_projects: List[str] = field(default_factory=list)
    top_at_risk_projects: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'total_projects': self.total_projects,
            'projects_on_time': self.projects_on_time,
            'projects_delayed': self.projects_delayed,
            'projects_at_risk': self.projects_at_risk,
            'projects_completed': self.projects_completed,
            'otd_pct': round(self.otd_pct, 1),
            'weighted_otd_pct': round(self.weighted_otd_pct, 1),
            'at_risk_pct': round(self.at_risk_pct, 1),
            'avg_lead_time_hours': round(self.avg_lead_time_hours, 2),
            'avg_lead_time_days': round(self.avg_lead_time_days, 2),
            'avg_delay_hours': round(self.avg_delay_hours, 2),
            'avg_completion_rate_pct': round(self.avg_completion_rate_pct, 1),
            'total_delay_hours': round(self.total_delay_hours, 2),
            'total_load_hours': round(self.total_load_hours, 2),
            'bottleneck_machine': self.bottleneck_machine,
            'bottleneck_load_hours': round(self.bottleneck_load_hours, 2),
            'top_delayed_projects': self.top_delayed_projects,
            'top_at_risk_projects': self.top_at_risk_projects,
        }


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# KPI COMPUTATION
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def compute_project_kpis(
    project: Project,
    load: ProjectLoad,
    slack: Optional[ProjectSlack] = None,
    risk: Optional[ProjectRiskScore] = None,
    completed_orders: Optional[List[str]] = None,
    bottleneck_machine: Optional[str] = None,
    total_bottleneck_load: float = 1.0
) -> ProjectKPIs:
    """
    Compute KPIs for a single project.
    
    Args:
        project: Project object
        load: Computed project load
        slack: Optional computed slack
        risk: Optional computed risk
        completed_orders: List of completed order IDs
        bottleneck_machine: ID of bottleneck machine
        total_bottleneck_load: Total load on bottleneck (for contribution %)
    
    Returns:
        ProjectKPIs with all metrics
    """
    kpis = ProjectKPIs(
        project_id=project.project_id,
        project_name=project.name,
        client=project.client,
    )
    
    # Time KPIs
    kpis.lead_time_hours = load.span_hours
    kpis.lead_time_days = load.span_hours / 24 if load.span_hours > 0 else 0
    
    if slack:
        kpis.slack_hours = slack.slack_hours
        if slack.slack_hours < 0:
            kpis.delay_hours = abs(slack.slack_hours)
            kpis.delay_days = kpis.delay_hours / 24
            kpis.is_on_time = False
        else:
            kpis.is_on_time = True
    
    # Completion KPIs
    kpis.total_orders = project.num_orders
    if completed_orders:
        kpis.completed_orders = len(set(completed_orders) & project.order_ids)
    kpis.completion_rate_pct = (kpis.completed_orders / kpis.total_orders * 100) if kpis.total_orders > 0 else 0
    
    # Load KPIs
    kpis.total_load_hours = load.total_load_hours
    kpis.machine_loads = load.machine_loads.copy()
    
    # Bottleneck contribution
    if bottleneck_machine and bottleneck_machine in load.machine_loads:
        bottleneck_load = load.machine_loads[bottleneck_machine]
        if total_bottleneck_load > 0:
            kpis.bottleneck_contribution_pct = (bottleneck_load / total_bottleneck_load) * 100
    
    # Value KPIs
    kpis.priority_weight = project.priority_weight
    if load.total_load_hours > 0:
        kpis.value_density = project.priority_weight / load.total_load_hours
    
    # Risk KPIs
    if risk:
        kpis.risk_score = risk.risk_score
        kpis.risk_level = risk.risk_level
        kpis.probability_late_pct = risk.probability_late * 100
    
    # Status
    kpis.status = project.status.value
    
    # Dates
    if project.due_date:
        kpis.due_date = project.due_date.isoformat()
    if load.max_end:
        kpis.estimated_completion = load.max_end.isoformat()
    
    return kpis


def compute_global_project_kpis(
    projects: List[Project],
    project_kpis: Dict[str, ProjectKPIs],
    loads: Dict[str, ProjectLoad]
) -> GlobalProjectKPIs:
    """
    Compute global KPIs across all projects.
    
    Args:
        projects: List of projects
        project_kpis: Dict of per-project KPIs
        loads: Dict of project loads
    
    Returns:
        GlobalProjectKPIs with aggregated metrics
    """
    global_kpis = GlobalProjectKPIs(
        timestamp=datetime.now().isoformat(),
        total_projects=len(projects),
    )
    
    if not projects:
        return global_kpis
    
    # Collect metrics
    on_time_count = 0
    delayed_count = 0
    at_risk_count = 0
    completed_count = 0
    
    on_time_weight = 0.0
    total_weight = 0.0
    
    lead_times = []
    delays = []
    completion_rates = []
    
    # Machine loads across all projects
    machine_totals: Dict[str, float] = {}
    
    delayed_projects: List[Tuple[str, float]] = []
    at_risk_projects: List[Tuple[str, float]] = []
    
    for project in projects:
        kpi = project_kpis.get(project.project_id)
        if not kpi:
            continue
        
        total_weight += kpi.priority_weight
        
        # Counts
        if kpi.is_on_time:
            on_time_count += 1
            on_time_weight += kpi.priority_weight
        else:
            delayed_count += 1
            delayed_projects.append((project.project_id, kpi.delay_hours))
        
        if kpi.risk_level in ["HIGH", "CRITICAL"]:
            at_risk_count += 1
            at_risk_projects.append((project.project_id, kpi.risk_score))
        
        if kpi.completion_rate_pct >= 100:
            completed_count += 1
        
        # Averages
        if kpi.lead_time_hours > 0:
            lead_times.append(kpi.lead_time_hours)
        delays.append(kpi.delay_hours)
        completion_rates.append(kpi.completion_rate_pct)
        
        # Machine totals
        for machine, load in kpi.machine_loads.items():
            machine_totals[machine] = machine_totals.get(machine, 0) + load
        
        global_kpis.total_load_hours += kpi.total_load_hours
    
    # Rates
    global_kpis.projects_on_time = on_time_count
    global_kpis.projects_delayed = delayed_count
    global_kpis.projects_at_risk = at_risk_count
    global_kpis.projects_completed = completed_count
    
    global_kpis.otd_pct = (on_time_count / len(projects) * 100) if projects else 0
    global_kpis.weighted_otd_pct = (on_time_weight / total_weight * 100) if total_weight > 0 else 0
    global_kpis.at_risk_pct = (at_risk_count / len(projects) * 100) if projects else 0
    
    # Averages
    global_kpis.avg_lead_time_hours = np.mean(lead_times) if lead_times else 0
    global_kpis.avg_lead_time_days = global_kpis.avg_lead_time_hours / 24
    global_kpis.avg_delay_hours = np.mean(delays) if delays else 0
    global_kpis.avg_completion_rate_pct = np.mean(completion_rates) if completion_rates else 0
    
    global_kpis.total_delay_hours = sum(delays)
    global_kpis.total_priority_weight = total_weight
    
    # Bottleneck
    if machine_totals:
        bottleneck = max(machine_totals, key=machine_totals.get)
        global_kpis.bottleneck_machine = bottleneck
        global_kpis.bottleneck_load_hours = machine_totals[bottleneck]
    
    # Top lists
    delayed_projects.sort(key=lambda x: x[1], reverse=True)
    at_risk_projects.sort(key=lambda x: x[1], reverse=True)
    
    global_kpis.top_delayed_projects = [p[0] for p in delayed_projects[:5]]
    global_kpis.top_at_risk_projects = [p[0] for p in at_risk_projects[:5]]
    
    return global_kpis


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def compute_all_project_kpis(
    projects: List[Project],
    plan_df: pd.DataFrame,
    completed_orders: Optional[List[str]] = None
) -> Tuple[Dict[str, ProjectKPIs], GlobalProjectKPIs]:
    """
    Compute all KPIs for all projects.
    
    Convenience function that runs the full KPI computation pipeline.
    
    Args:
        projects: List of projects
        plan_df: Production plan DataFrame
        completed_orders: List of completed order IDs
    
    Returns:
        (dict of ProjectKPIs, GlobalProjectKPIs)
    """
    from .project_load_engine import (
        compute_project_load,
        compute_project_slack,
        compute_project_risk,
    )
    
    # Compute loads
    loads = {}
    slacks = {}
    risks = {}
    
    for project in projects:
        load = compute_project_load(plan_df, project)
        loads[project.project_id] = load
        
        slack = compute_project_slack(project, load)
        slacks[project.project_id] = slack
        
        risk = compute_project_risk(project, load, slack)
        risks[project.project_id] = risk
    
    # Find bottleneck
    machine_totals: Dict[str, float] = {}
    for load in loads.values():
        for machine, hours in load.machine_loads.items():
            machine_totals[machine] = machine_totals.get(machine, 0) + hours
    
    bottleneck_machine = max(machine_totals, key=machine_totals.get) if machine_totals else None
    total_bottleneck_load = machine_totals.get(bottleneck_machine, 1.0) if bottleneck_machine else 1.0
    
    # Compute per-project KPIs
    project_kpis = {}
    for project in projects:
        kpi = compute_project_kpis(
            project=project,
            load=loads[project.project_id],
            slack=slacks.get(project.project_id),
            risk=risks.get(project.project_id),
            completed_orders=completed_orders,
            bottleneck_machine=bottleneck_machine,
            total_bottleneck_load=total_bottleneck_load,
        )
        project_kpis[project.project_id] = kpi
    
    # Compute global KPIs
    global_kpis = compute_global_project_kpis(projects, project_kpis, loads)
    
    return project_kpis, global_kpis


def get_project_summary_table(
    project_kpis: Dict[str, ProjectKPIs]
) -> pd.DataFrame:
    """
    Create a summary table of all project KPIs.
    
    Returns DataFrame suitable for display or export.
    """
    records = []
    for pid, kpi in project_kpis.items():
        records.append({
            'Projeto': kpi.project_name,
            'Cliente': kpi.client or '-',
            'Encomendas': kpi.total_orders,
            'Lead Time (dias)': round(kpi.lead_time_days, 1),
            'Atraso (h)': round(kpi.delay_hours, 1),
            '% Completo': round(kpi.completion_rate_pct, 0),
            'Carga (h)': round(kpi.total_load_hours, 1),
            'Risco': kpi.risk_level,
            'Status': kpi.status,
            'No Prazo': '✓' if kpi.is_on_time else '✗',
        })
    
    return pd.DataFrame(records)


