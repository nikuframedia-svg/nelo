"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — PROJECT PRIORITY OPTIMIZATION
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Mathematical optimization for multi-project priority scheduling.

PROBLEM STATEMENT
═════════════════

Given:
- Set of projects P = {p₁, ..., pₙ}
- Each project has:
  - Load L_p (machine-hours)
  - Due date d_p
  - Priority weight w_p
  - Risk score r_p
- Machine capacity C_m (hours per horizon)
- Maximum parallel projects P_max

Decide:
- Which projects to prioritize (schedule earlier)
- What order to process projects

Objective:
- Maximize weighted on-time delivery
- Minimize total weighted delay

MATHEMATICAL FORMULATION
════════════════════════

MILP Model for Project Prioritization:

SETS
────
    P = {1, ..., n}         Projects
    M = {1, ..., m}         Machines
    T = {1, ..., H}         Time periods (days/weeks)

PARAMETERS
──────────
    L_p         Total load of project p (hours)
    L_{p,m}     Load of project p on machine m (hours)
    d_p         Due date of project p (period)
    w_p         Priority weight of project p
    C_m         Capacity of machine m per period
    P_max       Max projects in parallel

DECISION VARIABLES
──────────────────
    x_{p,t} ∈ {0,1}     1 if project p starts in period t
    y_p ∈ {0,1}         1 if project p is on-time
    s_p ≥ 0             Start period of project p
    e_p ≥ 0             End period of project p
    D_p ≥ 0             Delay of project p (periods)

CONSTRAINTS
───────────

(C1) Each project starts exactly once:
     Σ_t x_{p,t} = 1                              ∀ p ∈ P

(C2) Start period definition:
     s_p = Σ_t t · x_{p,t}                        ∀ p ∈ P

(C3) End period (simplified linear):
     e_p = s_p + ceiling(L_p / C_avg)             ∀ p ∈ P

(C4) Delay definition:
     D_p ≥ e_p - d_p                              ∀ p ∈ P
     D_p ≥ 0                                      ∀ p ∈ P

(C5) On-time indicator:
     y_p ≥ 1 - D_p / M                           ∀ p ∈ P (big-M)

(C6) Capacity constraint per period:
     Σ_p (load during t) ≤ C_m                    ∀ m ∈ M, t ∈ T

(C7) Parallel project limit:
     Σ_p (active in t) ≤ P_max                    ∀ t ∈ T

OBJECTIVE
─────────

Multi-objective (weighted sum):

    max  α · Σ_p w_p · y_p   (weighted OTD)
       - β · Σ_p w_p · D_p   (weighted delay)
       - γ · Σ_p r_p · (1-y_p) (risk penalty)

where α, β, γ are weights.

R&D / SIFIDE: WP5 - Project Planning
────────────────────────────────────
- Hypothesis H5.4: MILP prioritization improves OTD vs FCFS
- Experiment E5.3: Compare MILP vs heuristic project ordering
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .project_model import Project
from .project_load_engine import ProjectLoad, ProjectRiskScore

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class PriorityPlanConfig:
    """Configuration for priority optimization."""
    horizon_days: int = 30
    max_parallel_projects: int = 5
    capacity_hours_per_day: float = 8.0
    
    # Objective weights
    alpha_otd: float = 1.0      # Weight for on-time delivery
    beta_delay: float = 0.5     # Penalty for delay
    gamma_risk: float = 0.3     # Risk penalty
    
    # Solver settings
    time_limit_sec: float = 30.0
    mip_gap: float = 0.05


@dataclass
class ProjectPriority:
    """Priority assignment for a project."""
    project_id: str
    priority_rank: int          # 1 = highest priority
    suggested_start_day: int    # Suggested start (days from now)
    expected_end_day: int       # Expected completion
    expected_delay_days: float  # Expected delay (0 if on-time)
    priority_score: float       # Computed priority score (higher = more urgent)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'project_id': self.project_id,
            'priority_rank': self.priority_rank,
            'suggested_start_day': self.suggested_start_day,
            'expected_end_day': self.expected_end_day,
            'expected_delay_days': round(self.expected_delay_days, 1),
            'priority_score': round(self.priority_score, 2),
        }


@dataclass
class PriorityPlan:
    """
    Result of project priority optimization.
    
    Contains the suggested priority ordering and expected outcomes.
    """
    timestamp: str
    config: PriorityPlanConfig
    priorities: List[ProjectPriority]
    
    # Summary metrics
    total_projects: int = 0
    projects_on_time: int = 0
    projects_delayed: int = 0
    total_weighted_delay: float = 0.0
    expected_otd_pct: float = 0.0
    
    # Solver info
    solver_status: str = "NOT_SOLVED"
    objective_value: Optional[float] = None
    solve_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'total_projects': self.total_projects,
            'projects_on_time': self.projects_on_time,
            'projects_delayed': self.projects_delayed,
            'total_weighted_delay': round(self.total_weighted_delay, 2),
            'expected_otd_pct': round(self.expected_otd_pct, 1),
            'solver_status': self.solver_status,
            'objective_value': self.objective_value,
            'solve_time_sec': round(self.solve_time_sec, 3),
            'priorities': [p.to_dict() for p in self.priorities],
        }
    
    def get_order_priority_vector(self, projects: List[Project]) -> Dict[str, float]:
        """
        Generate order-level priority vector for APS integration.
        
        Maps order_id -> priority_score based on project priority.
        
        Returns:
            Dict mapping order_id -> priority_score
        """
        # Build project_id -> priority mapping
        proj_priority = {}
        for pp in self.priorities:
            proj_priority[pp.project_id] = pp.priority_score
        
        # Map to orders
        order_priority = {}
        for project in projects:
            score = proj_priority.get(project.project_id, 1.0)
            for order_id in project.order_ids:
                order_priority[order_id] = score
        
        return order_priority


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# HEURISTIC OPTIMIZER (Fast)
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def compute_priority_score(
    project: Project,
    load: ProjectLoad,
    risk: Optional[ProjectRiskScore] = None,
    reference_date: Optional[datetime] = None
) -> float:
    """
    Compute a priority score for heuristic ordering.
    
    Score combines:
    - Urgency (days until due)
    - Priority weight
    - Risk score
    
    Formula:
        Score = w_p × (1 + urgency_factor) × (1 + risk_factor)
    
    where:
        urgency_factor = max(0, 1 - days_to_due / 30)  (higher if closer to due)
        risk_factor = risk_score / 100
    
    Args:
        project: Project to score
        load: Project load
        risk: Optional risk score
        reference_date: Reference date for urgency (default: now)
    
    Returns:
        Priority score (higher = more urgent)
    """
    ref = reference_date or datetime.now()
    
    # Base priority from weight
    base = project.priority_weight
    
    # Urgency factor (days until due)
    urgency_factor = 0.0
    if project.due_date:
        days_to_due = (project.due_date - ref).days
        if days_to_due <= 0:
            urgency_factor = 2.0  # Already late!
        elif days_to_due <= 7:
            urgency_factor = 1.5  # Very urgent
        elif days_to_due <= 14:
            urgency_factor = 1.0  # Urgent
        elif days_to_due <= 30:
            urgency_factor = 0.5  # Moderate
        else:
            urgency_factor = 0.0  # Not urgent
    
    # Risk factor
    risk_factor = 0.0
    if risk:
        risk_factor = risk.risk_score / 100
    
    # Combined score
    score = base * (1 + urgency_factor) * (1 + risk_factor)
    
    return score


def optimize_project_priorities_heuristic(
    projects: List[Project],
    loads: Dict[str, ProjectLoad],
    risks: Optional[Dict[str, ProjectRiskScore]] = None,
    config: Optional[PriorityPlanConfig] = None
) -> PriorityPlan:
    """
    Optimize project priorities using heuristics.
    
    Fast algorithm suitable for real-time use.
    
    Algorithm:
    1. Compute priority score for each project
    2. Sort by score (descending)
    3. Assign start days respecting capacity
    
    Args:
        projects: List of projects
        loads: Dict of project loads
        risks: Optional dict of risk scores
        config: Optimization config
    
    Returns:
        PriorityPlan with suggested ordering
    """
    import time
    start_time = time.time()
    
    config = config or PriorityPlanConfig()
    reference_date = datetime.now()
    
    # Compute scores
    scored_projects = []
    for project in projects:
        load = loads.get(project.project_id, ProjectLoad(project_id=project.project_id))
        risk = risks.get(project.project_id) if risks else None
        
        score = compute_priority_score(project, load, risk, reference_date)
        scored_projects.append((project, load, risk, score))
    
    # Sort by score (descending)
    scored_projects.sort(key=lambda x: x[3], reverse=True)
    
    # Assign start days respecting capacity
    daily_capacity = config.capacity_hours_per_day * config.max_parallel_projects
    day_loads = {}  # day -> total load assigned
    
    priorities = []
    on_time = 0
    delayed = 0
    total_weighted_delay = 0.0
    
    for rank, (project, load, risk, score) in enumerate(scored_projects, 1):
        # Find first day with enough capacity
        start_day = 0
        duration_days = max(1, math.ceil(load.total_load_hours / config.capacity_hours_per_day))
        
        # Simple: just increment start day until we have capacity
        while True:
            can_schedule = True
            for d in range(start_day, start_day + duration_days):
                current = day_loads.get(d, 0)
                if current + load.total_load_hours / duration_days > daily_capacity:
                    can_schedule = False
                    break
            
            if can_schedule:
                break
            start_day += 1
            
            if start_day > config.horizon_days:
                break
        
        # Assign load to days
        for d in range(start_day, start_day + duration_days):
            day_loads[d] = day_loads.get(d, 0) + load.total_load_hours / duration_days
        
        end_day = start_day + duration_days
        
        # Check if on time
        delay_days = 0.0
        if project.due_date:
            due_day = (project.due_date - reference_date).days
            if end_day > due_day:
                delay_days = end_day - due_day
        
        if delay_days <= 0:
            on_time += 1
        else:
            delayed += 1
            total_weighted_delay += project.priority_weight * delay_days
        
        priorities.append(ProjectPriority(
            project_id=project.project_id,
            priority_rank=rank,
            suggested_start_day=start_day,
            expected_end_day=end_day,
            expected_delay_days=delay_days,
            priority_score=score,
        ))
    
    solve_time = time.time() - start_time
    
    return PriorityPlan(
        timestamp=datetime.now().isoformat(),
        config=config,
        priorities=priorities,
        total_projects=len(projects),
        projects_on_time=on_time,
        projects_delayed=delayed,
        total_weighted_delay=total_weighted_delay,
        expected_otd_pct=(on_time / len(projects) * 100) if projects else 0.0,
        solver_status="HEURISTIC",
        objective_value=None,
        solve_time_sec=solve_time,
    )


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# MILP OPTIMIZER (Optimal)
# ════════════════════════════════════════════════════════════════════════════════════════════════════

class ProjectPriorityOptimizer:
    """
    MILP-based optimizer for project prioritization.
    
    Uses OR-Tools to find optimal project ordering.
    
    Usage:
        optimizer = ProjectPriorityOptimizer(config)
        plan = optimizer.optimize(projects, loads, risks)
    """
    
    def __init__(self, config: Optional[PriorityPlanConfig] = None):
        self.config = config or PriorityPlanConfig()
    
    def optimize(
        self,
        projects: List[Project],
        loads: Dict[str, ProjectLoad],
        risks: Optional[Dict[str, ProjectRiskScore]] = None
    ) -> PriorityPlan:
        """
        Optimize project priorities using MILP.
        
        Falls back to heuristic if OR-Tools not available.
        """
        try:
            return self._optimize_milp(projects, loads, risks)
        except ImportError:
            logger.warning("OR-Tools not available. Using heuristic optimizer.")
            return optimize_project_priorities_heuristic(projects, loads, risks, self.config)
        except Exception as e:
            logger.error(f"MILP optimization failed: {e}. Using heuristic.")
            return optimize_project_priorities_heuristic(projects, loads, risks, self.config)
    
    def _optimize_milp(
        self,
        projects: List[Project],
        loads: Dict[str, ProjectLoad],
        risks: Optional[Dict[str, ProjectRiskScore]] = None
    ) -> PriorityPlan:
        """
        MILP implementation using OR-Tools.
        """
        from ortools.linear_solver import pywraplp
        import time
        
        start_time = time.time()
        reference_date = datetime.now()
        
        # Create solver
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            raise RuntimeError("Could not create MILP solver")
        
        H = self.config.horizon_days
        infinity = solver.infinity()
        big_M = H * 2
        
        n_projects = len(projects)
        if n_projects == 0:
            return PriorityPlan(
                timestamp=datetime.now().isoformat(),
                config=self.config,
                priorities=[],
                solver_status="NO_PROJECTS"
            )
        
        # Index mapping
        proj_idx = {p.project_id: i for i, p in enumerate(projects)}
        
        # ════════════════════════════════════════════════════════════════════
        # VARIABLES
        # ════════════════════════════════════════════════════════════════════
        
        # s[p] = start day of project p
        s = [solver.IntVar(0, H, f's_{i}') for i in range(n_projects)]
        
        # e[p] = end day of project p
        e = [solver.IntVar(0, H + 100, f'e_{i}') for i in range(n_projects)]
        
        # D[p] = delay of project p (days)
        D = [solver.NumVar(0, infinity, f'D_{i}') for i in range(n_projects)]
        
        # y[p] = 1 if project p is on-time
        y = [solver.NumVar(0, 1, f'y_{i}') for i in range(n_projects)]
        
        # ════════════════════════════════════════════════════════════════════
        # CONSTRAINTS
        # ════════════════════════════════════════════════════════════════════
        
        for i, project in enumerate(projects):
            load = loads.get(project.project_id, ProjectLoad(project_id=project.project_id))
            
            # Duration in days
            duration = max(1, math.ceil(load.total_load_hours / self.config.capacity_hours_per_day))
            
            # (C3) End = Start + Duration
            solver.Add(e[i] == s[i] + duration, f'C3_end_{i}')
            
            # Due date in days from reference
            if project.due_date:
                due_day = max(0, (project.due_date - reference_date).days)
            else:
                due_day = H  # If no due date, assume end of horizon
            
            # (C4) Delay definition
            solver.Add(D[i] >= e[i] - due_day, f'C4_delay_{i}')
            
            # (C5) On-time indicator (linearization)
            # y = 1 if D = 0, else y can be 0
            # D <= M * (1 - y)
            solver.Add(D[i] <= big_M * (1 - y[i]), f'C5_ontime_{i}')
        
        # ════════════════════════════════════════════════════════════════════
        # OBJECTIVE
        # ════════════════════════════════════════════════════════════════════
        
        alpha = self.config.alpha_otd
        beta = self.config.beta_delay
        gamma = self.config.gamma_risk
        
        # Maximize: α·Σw·y - β·Σw·D - γ·Σr·(1-y)
        objective = solver.Objective()
        
        for i, project in enumerate(projects):
            w = project.priority_weight
            r = risks[project.project_id].risk_score if risks and project.project_id in risks else 0
            
            # + α·w·y (on-time bonus)
            objective.SetCoefficient(y[i], alpha * w)
            
            # - β·w·D (delay penalty)
            objective.SetCoefficient(D[i], -beta * w)
            
            # - γ·r (risk penalty for late) = - γ·r + γ·r·y
            # Simplified: penalize late projects with high risk
            objective.SetCoefficient(y[i], gamma * r)  # Bonus for on-time
        
        objective.SetMaximization()
        
        # Solve
        solver.SetTimeLimit(int(self.config.time_limit_sec * 1000))
        status = solver.Solve()
        
        solve_time = time.time() - start_time
        
        # Extract solution
        priorities = []
        on_time = 0
        delayed = 0
        total_weighted_delay = 0.0
        
        status_name = {
            pywraplp.Solver.OPTIMAL: "OPTIMAL",
            pywraplp.Solver.FEASIBLE: "FEASIBLE",
            pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
            pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        }.get(status, "UNKNOWN")
        
        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            # Get solution values
            solutions = []
            for i, project in enumerate(projects):
                start_val = int(s[i].solution_value())
                end_val = int(e[i].solution_value())
                delay_val = D[i].solution_value()
                ontime_val = y[i].solution_value()
                
                score = compute_priority_score(
                    project,
                    loads.get(project.project_id, ProjectLoad(project_id=project.project_id)),
                    risks.get(project.project_id) if risks else None,
                    reference_date
                )
                
                solutions.append({
                    'project': project,
                    'start': start_val,
                    'end': end_val,
                    'delay': delay_val,
                    'ontime': ontime_val > 0.5,
                    'score': score,
                })
            
            # Sort by start time to determine rank
            solutions.sort(key=lambda x: (x['start'], -x['score']))
            
            for rank, sol in enumerate(solutions, 1):
                project = sol['project']
                
                if sol['ontime']:
                    on_time += 1
                else:
                    delayed += 1
                    total_weighted_delay += project.priority_weight * sol['delay']
                
                priorities.append(ProjectPriority(
                    project_id=project.project_id,
                    priority_rank=rank,
                    suggested_start_day=sol['start'],
                    expected_end_day=sol['end'],
                    expected_delay_days=sol['delay'],
                    priority_score=sol['score'],
                ))
        
        else:
            # Fallback to heuristic if MILP fails
            logger.warning(f"MILP status={status_name}. Falling back to heuristic.")
            return optimize_project_priorities_heuristic(projects, loads, risks, self.config)
        
        return PriorityPlan(
            timestamp=datetime.now().isoformat(),
            config=self.config,
            priorities=priorities,
            total_projects=n_projects,
            projects_on_time=on_time,
            projects_delayed=delayed,
            total_weighted_delay=total_weighted_delay,
            expected_otd_pct=(on_time / n_projects * 100) if n_projects else 0.0,
            solver_status=status_name,
            objective_value=solver.Objective().Value() if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE] else None,
            solve_time_sec=solve_time,
        )


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def optimize_project_priorities(
    projects: List[Project],
    loads: Dict[str, ProjectLoad],
    risks: Optional[Dict[str, ProjectRiskScore]] = None,
    config: Optional[PriorityPlanConfig] = None,
    use_milp: bool = False
) -> PriorityPlan:
    """
    Optimize project priorities.
    
    Args:
        projects: List of projects
        loads: Dict of project loads
        risks: Optional risk scores
        config: Optimization config
        use_milp: Whether to use MILP (slower but optimal)
    
    Returns:
        PriorityPlan with suggested ordering
    """
    config = config or PriorityPlanConfig()
    
    if use_milp:
        optimizer = ProjectPriorityOptimizer(config)
        return optimizer.optimize(projects, loads, risks)
    else:
        return optimize_project_priorities_heuristic(projects, loads, risks, config)


def apply_priority_plan_to_orders(
    plan: PriorityPlan,
    projects: List[Project],
    orders_df: pd.DataFrame,
    order_id_col: str = 'order_id',
    priority_col: str = 'priority'
) -> pd.DataFrame:
    """
    Apply priority plan to orders DataFrame.
    
    Updates the priority column based on project priorities.
    
    Args:
        plan: Priority plan from optimizer
        projects: List of projects
        orders_df: Orders DataFrame
        order_id_col: Order ID column name
        priority_col: Priority column name
    
    Returns:
        Updated DataFrame with new priorities
    """
    priority_vector = plan.get_order_priority_vector(projects)
    
    df = orders_df.copy()
    df[priority_col] = df[order_id_col].astype(str).map(
        lambda oid: priority_vector.get(oid, 1.0)
    )
    
    return df


