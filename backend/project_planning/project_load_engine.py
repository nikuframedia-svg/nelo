"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — PROJECT LOAD ENGINE
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Computes project load, slack, and risk scores.

DEFINITIONS
═══════════

Project Load:
─────────────
The total resource consumption of a project:

    L_p = Σ_{o ∈ O(p)} Σ_{op ∈ ops(o)} duration(op)

where:
    O(p) = orders belonging to project p
    ops(o) = operations of order o
    duration(op) = processing time of operation

Project Span:
─────────────
Time from first operation to last:

    Span_p = max_{op ∈ O(p)} end_time(op) - min_{op ∈ O(p)} start_time(op)

Project Slack:
──────────────
Time buffer before due date:

    Slack_p = d_p - max_{op ∈ O(p)} end_time(op)

where d_p is the project due date.

    Slack > 0 : On schedule
    Slack = 0 : At deadline
    Slack < 0 : Late by |Slack|

Project Risk:
─────────────
Probability-weighted score of delay:

    Risk_p = P(Late_p) × Impact_p

where:
    P(Late_p) = probability of being late (from forecast + SNR)
    Impact_p = w_p × L_p (priority × load = importance)

RISK MODEL
══════════

We model project completion time T_p as:

    T_p ~ N(μ_p, σ_p²)

where:
    μ_p = estimated completion time (from schedule)
    σ_p = uncertainty (derived from SNR of lead time)

Probability of delay:
    P(Late_p) = P(T_p > d_p) = Φ((d_p - μ_p) / σ_p)

where Φ is the standard normal CDF.

Risk score combines probability and impact:
    Risk_p ∈ [0, 100] scaled score

R&D / SIFIDE: WP5 - Project Planning
────────────────────────────────────
- Hypothesis H5.3: Risk scores predict actual delays
- Experiment E5.2: Validate risk model calibration
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .project_model import Project, ProjectStatus

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class ProjectLoad:
    """
    Load information for a project.
    
    Attributes:
        project_id: Project identifier
        total_load_min: Total processing time (minutes)
        total_load_hours: Total processing time (hours)
        min_start: Earliest operation start
        max_end: Latest operation end
        span_hours: Time from first to last operation
        machine_loads: Load per machine (hours)
        num_operations: Total operations
        num_orders: Number of orders
    """
    project_id: str
    total_load_min: float = 0.0
    total_load_hours: float = 0.0
    min_start: Optional[datetime] = None
    max_end: Optional[datetime] = None
    span_hours: float = 0.0
    machine_loads: Dict[str, float] = field(default_factory=dict)
    num_operations: int = 0
    num_orders: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'project_id': self.project_id,
            'total_load_hours': round(self.total_load_hours, 2),
            'total_load_min': round(self.total_load_min, 1),
            'min_start': self.min_start.isoformat() if self.min_start else None,
            'max_end': self.max_end.isoformat() if self.max_end else None,
            'span_hours': round(self.span_hours, 2),
            'num_operations': self.num_operations,
            'num_orders': self.num_orders,
            'machine_loads': {k: round(v, 2) for k, v in self.machine_loads.items()},
        }


@dataclass
class ProjectSlack:
    """
    Slack analysis for a project.
    
    Attributes:
        project_id: Project identifier
        due_date: Project due date
        completion_time: Estimated completion time
        slack_hours: Time buffer (positive = early, negative = late)
        slack_pct: Slack as percentage of span
        is_on_schedule: Whether slack >= 0
    """
    project_id: str
    due_date: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    slack_hours: float = 0.0
    slack_pct: float = 0.0
    is_on_schedule: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'project_id': self.project_id,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'completion_time': self.completion_time.isoformat() if self.completion_time else None,
            'slack_hours': round(self.slack_hours, 2),
            'slack_pct': round(self.slack_pct, 1),
            'is_on_schedule': self.is_on_schedule,
        }


@dataclass
class ProjectRiskScore:
    """
    Risk assessment for a project.
    
    Risk_p = P(Late) × Impact
    
    Attributes:
        project_id: Project identifier
        risk_score: Overall risk score (0-100)
        probability_late: Probability of missing deadline
        impact_score: Impact if late (based on priority and load)
        risk_level: Categorical risk level
        confidence: Confidence in risk estimate (from SNR)
        contributing_factors: What's driving the risk
    """
    project_id: str
    risk_score: float = 0.0
    probability_late: float = 0.0
    impact_score: float = 0.0
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float = 0.5
    contributing_factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'project_id': self.project_id,
            'risk_score': round(self.risk_score, 1),
            'probability_late_pct': round(self.probability_late * 100, 1),
            'impact_score': round(self.impact_score, 1),
            'risk_level': self.risk_level,
            'confidence': round(self.confidence, 2),
            'contributing_factors': self.contributing_factors,
        }


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# LOAD COMPUTATION
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def compute_project_load(
    plan_df: pd.DataFrame,
    project: Project,
    order_col: str = 'order_id',
    machine_col: str = 'machine_id',
    duration_col: str = 'duration_min',
    start_col: str = 'start_time',
    end_col: str = 'end_time'
) -> ProjectLoad:
    """
    Compute load for a single project.
    
    Mathematical Definition:
    ───────────────────────
    
    L_p = Σ_{o ∈ O(p)} Σ_{op ∈ ops(o)} duration(op)
    
    L_p^m = Σ_{op : machine(op)=m, order(op) ∈ O(p)} duration(op)  (per machine)
    
    Span_p = max(end_time) - min(start_time)
    
    Args:
        plan_df: Production plan with operations
        project: Project to compute load for
        order_col: Column name for order ID
        machine_col: Column name for machine ID
        duration_col: Column name for duration
        start_col: Column name for start time
        end_col: Column name for end time
    
    Returns:
        ProjectLoad with computed metrics
    """
    # Filter operations for this project's orders
    if order_col not in plan_df.columns:
        return ProjectLoad(project_id=project.project_id)
    
    project_ops = plan_df[plan_df[order_col].astype(str).isin(project.order_ids)]
    
    if project_ops.empty:
        return ProjectLoad(
            project_id=project.project_id,
            num_orders=project.num_orders,
        )
    
    # Total load
    total_load_min = 0.0
    if duration_col in project_ops.columns:
        total_load_min = float(project_ops[duration_col].sum())
    
    # Time span
    min_start = None
    max_end = None
    
    if start_col in project_ops.columns:
        starts = pd.to_datetime(project_ops[start_col], errors='coerce')
        valid_starts = starts.dropna()
        if not valid_starts.empty:
            min_start = valid_starts.min().to_pydatetime()
    
    if end_col in project_ops.columns:
        ends = pd.to_datetime(project_ops[end_col], errors='coerce')
        valid_ends = ends.dropna()
        if not valid_ends.empty:
            max_end = valid_ends.max().to_pydatetime()
    
    span_hours = 0.0
    if min_start and max_end:
        span_hours = (max_end - min_start).total_seconds() / 3600
    
    # Load per machine
    machine_loads = {}
    if machine_col in project_ops.columns and duration_col in project_ops.columns:
        for machine, group in project_ops.groupby(machine_col):
            machine_loads[str(machine)] = float(group[duration_col].sum()) / 60  # Convert to hours
    
    # Unique orders in this plan
    orders_in_plan = project_ops[order_col].astype(str).unique()
    
    return ProjectLoad(
        project_id=project.project_id,
        total_load_min=total_load_min,
        total_load_hours=total_load_min / 60,
        min_start=min_start,
        max_end=max_end,
        span_hours=span_hours,
        machine_loads=machine_loads,
        num_operations=len(project_ops),
        num_orders=len(orders_in_plan),
    )


def compute_all_project_loads(
    plan_df: pd.DataFrame,
    projects: List[Project],
    **kwargs
) -> Dict[str, ProjectLoad]:
    """
    Compute load for all projects.
    
    Returns:
        Dict mapping project_id -> ProjectLoad
    """
    loads = {}
    for project in projects:
        load = compute_project_load(plan_df, project, **kwargs)
        loads[project.project_id] = load
        
        # Update project with computed values
        project.min_start = load.min_start
        project.max_end = load.max_end
        project.total_load_hours = load.total_load_hours
    
    return loads


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# SLACK COMPUTATION
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def compute_project_slack(
    project: Project,
    load: ProjectLoad
) -> ProjectSlack:
    """
    Compute slack for a project.
    
    Mathematical Definition:
    ───────────────────────
    
    Slack_p = d_p - T_p
    
    where:
        d_p = project due date
        T_p = max(end_time) = completion time
    
    Args:
        project: Project with due date
        load: Computed project load
    
    Returns:
        ProjectSlack with slack analysis
    """
    slack = ProjectSlack(
        project_id=project.project_id,
        due_date=project.due_date,
        completion_time=load.max_end,
    )
    
    if project.due_date and load.max_end:
        slack_delta = project.due_date - load.max_end
        slack.slack_hours = slack_delta.total_seconds() / 3600
        slack.is_on_schedule = slack.slack_hours >= 0
        
        # Slack as percentage of span
        if load.span_hours > 0:
            slack.slack_pct = (slack.slack_hours / load.span_hours) * 100
    
    return slack


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# RISK COMPUTATION
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_project_risk(
    project: Project,
    load: ProjectLoad,
    slack: ProjectSlack,
    lead_time_snr: float = 1.0,
    lead_time_std_pct: float = 0.2
) -> ProjectRiskScore:
    """
    Compute risk score for a project.
    
    Mathematical Model:
    ──────────────────
    
    We model completion time T_p as:
    
        T_p ~ N(μ_p, σ_p²)
    
    where:
        μ_p = estimated completion (max_end from schedule)
        σ_p = uncertainty = span × cv / √(SNR)
        cv = coefficient of variation (default 0.2)
    
    Probability of delay:
        
        P(Late_p) = P(T_p > d_p) = 1 - Φ((d_p - μ_p) / σ_p)
    
    Impact (if late):
        
        Impact_p = w_p × L_p / L_total
    
    Risk score:
        
        Risk_p = 100 × P(Late_p) × Impact_p
    
    Confidence:
        
        Confidence = SNR / (1 + SNR)  (from R² relationship)
    
    Args:
        project: Project to assess
        load: Project load information
        slack: Project slack information
        lead_time_snr: SNR of lead time estimates
        lead_time_std_pct: Coefficient of variation for lead time
    
    Returns:
        ProjectRiskScore with risk assessment
    """
    risk = ProjectRiskScore(project_id=project.project_id)
    contributing_factors = []
    
    # Confidence from SNR
    risk.confidence = lead_time_snr / (1 + lead_time_snr)
    
    # If no due date, risk is uncertain
    if not project.due_date or not load.max_end:
        risk.risk_level = "UNKNOWN"
        contributing_factors.append("Sem data de entrega definida")
        risk.contributing_factors = contributing_factors
        return risk
    
    # Compute probability of delay
    # σ = span × cv / √(SNR)
    cv = lead_time_std_pct
    if lead_time_snr > 0:
        sigma_hours = load.span_hours * cv / math.sqrt(lead_time_snr)
    else:
        sigma_hours = load.span_hours * cv * 2  # High uncertainty
    
    # Avoid division by zero
    if sigma_hours < 0.1:
        sigma_hours = 0.1
    
    # P(Late) = P(T > d) = 1 - Φ((d - μ) / σ)
    # μ = max_end (estimated completion)
    # d = due_date
    z_score = slack.slack_hours / sigma_hours
    prob_on_time = _normal_cdf(z_score)
    prob_late = 1.0 - prob_on_time
    
    risk.probability_late = max(0.0, min(1.0, prob_late))
    
    # Impact score (0-100 scale)
    # Based on priority weight and load
    base_impact = project.priority_weight * 10  # Scale priority to 0-100
    load_factor = min(load.total_load_hours / 100, 1.0)  # Cap at 100h
    risk.impact_score = base_impact * (1 + load_factor)
    
    # Final risk score
    risk.risk_score = risk.probability_late * risk.impact_score
    
    # Determine risk level
    if risk.risk_score >= 50:
        risk.risk_level = "CRITICAL"
        contributing_factors.append("Risco muito elevado de atraso")
    elif risk.risk_score >= 25:
        risk.risk_level = "HIGH"
        contributing_factors.append("Risco elevado de atraso")
    elif risk.risk_score >= 10:
        risk.risk_level = "MEDIUM"
        contributing_factors.append("Risco moderado")
    else:
        risk.risk_level = "LOW"
        contributing_factors.append("Risco baixo")
    
    # Add specific contributing factors
    if slack.slack_hours < 0:
        contributing_factors.append(f"Já em atraso: {abs(slack.slack_hours):.1f}h")
    elif slack.slack_hours < 8:
        contributing_factors.append(f"Margem reduzida: {slack.slack_hours:.1f}h")
    
    if project.priority_weight > 1.5:
        contributing_factors.append(f"Alta prioridade: peso={project.priority_weight:.1f}")
    
    if load.total_load_hours > 50:
        contributing_factors.append(f"Carga elevada: {load.total_load_hours:.1f}h")
    
    if risk.confidence < 0.5:
        contributing_factors.append(f"Incerteza elevada (SNR={lead_time_snr:.1f})")
    
    risk.contributing_factors = contributing_factors
    
    return risk


def compute_all_project_risks(
    projects: List[Project],
    loads: Dict[str, ProjectLoad],
    slacks: Dict[str, ProjectSlack],
    snr_by_project: Optional[Dict[str, float]] = None,
    default_snr: float = 2.0
) -> Dict[str, ProjectRiskScore]:
    """
    Compute risk scores for all projects.
    
    Args:
        projects: List of projects
        loads: Dict of project loads
        slacks: Dict of project slacks
        snr_by_project: Optional SNR per project
        default_snr: Default SNR if not specified
    
    Returns:
        Dict mapping project_id -> ProjectRiskScore
    """
    risks = {}
    
    for project in projects:
        load = loads.get(project.project_id, ProjectLoad(project_id=project.project_id))
        slack = slacks.get(project.project_id, ProjectSlack(project_id=project.project_id))
        snr = default_snr
        
        if snr_by_project and project.project_id in snr_by_project:
            snr = snr_by_project[project.project_id]
        
        risk = compute_project_risk(project, load, slack, lead_time_snr=snr)
        risks[project.project_id] = risk
    
    return risks


def rank_projects_by_risk(
    risks: Dict[str, ProjectRiskScore],
    descending: bool = True
) -> List[Tuple[str, float]]:
    """
    Rank projects by risk score.
    
    Returns:
        List of (project_id, risk_score) tuples, sorted by risk
    """
    ranked = [(pid, r.risk_score) for pid, r in risks.items()]
    ranked.sort(key=lambda x: x[1], reverse=descending)
    return ranked


def update_project_status_from_risk(
    project: Project,
    risk: ProjectRiskScore,
    slack: ProjectSlack
) -> None:
    """
    Update project status based on risk and slack.
    """
    if slack.slack_hours < 0:
        if project.status != ProjectStatus.COMPLETED:
            project.status = ProjectStatus.DELAYED
    elif risk.risk_level in ["HIGH", "CRITICAL"]:
        project.status = ProjectStatus.AT_RISK
    elif project.total_load_hours > 0:
        project.status = ProjectStatus.IN_PROGRESS
    else:
        project.status = ProjectStatus.PLANNING



