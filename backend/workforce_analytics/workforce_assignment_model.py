"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — WORKFORCE ASSIGNMENT MODEL
═══════════════════════════════════════════════════════════════════════════════════════════════════════

MILP optimization for optimal worker-to-operation assignment.

PROBLEM STATEMENT
═════════════════

Given:
- Set of workers W = {w₁, ..., wₙ}
- Set of operations O = {o₁, ..., oₘ}
- Set of machines M = {m₁, ..., mₖ}

For each worker w:
- skill_score[w]: Skill level (0-1)
- qualified_ops[w]: Set of qualified operation types
- qualified_machines[w]: Set of machines worker can operate
- availability[w]: Available hours

For each operation o:
- machine[o]: Required machine
- op_type[o]: Operation type
- duration[o]: Processing time
- priority[o]: Priority weight

Decide:
- Assignment z[w,o] ∈ {0,1}: Worker w assigned to operation o

MATHEMATICAL FORMULATION
════════════════════════

MILP Model:

SETS
────
    W = {1, ..., n}         Workers
    O = {1, ..., m}         Operations
    M = {1, ..., k}         Machines

PARAMETERS
──────────
    σ_w             Skill score of worker w
    q_w^op          1 if worker w qualified for operation type
    q_w^m           1 if worker w qualified for machine
    a_w             Available hours for worker w
    d_o             Duration of operation o (hours)
    p_o             Priority of operation o
    m_o             Machine required by operation o
    t_o             Operation type of o

DECISION VARIABLES
──────────────────
    z_{w,o} ∈ {0,1}     1 if worker w assigned to operation o

OBJECTIVE
─────────

Maximize weighted skill contribution:

    max Σ_{w,o} σ_w · p_o · z_{w,o}

Or multi-objective:
    
    max α · Σ skill_contribution - β · Σ workload_imbalance - γ · Σ qualification_gap

CONSTRAINTS
───────────

(C1) Each operation assigned to exactly one worker:
     Σ_w z_{w,o} = 1                          ∀ o ∈ O

(C2) Worker capacity:
     Σ_o d_o · z_{w,o} ≤ a_w                  ∀ w ∈ W

(C3) Qualification - operation type:
     z_{w,o} ≤ q_w^{t_o}                      ∀ w ∈ W, o ∈ O

(C4) Qualification - machine:
     z_{w,o} ≤ q_w^{m_o}                      ∀ w ∈ W, o ∈ O

(C5) Machine capacity (one worker per machine at a time):
     This requires time-indexed formulation or sequencing constraints
     (simplified for MVP)

R&D / SIFIDE: WP6 - Workforce Intelligence
──────────────────────────────────────────
- Hypothesis H6.2: MILP assignment improves throughput vs manual by >10%
- Experiment E6.3: Compare MILP vs greedy assignment
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class AssignmentConfig:
    """Configuration for assignment optimization."""
    
    # Objective weights
    weight_skill: float = 1.0           # Weight for skill contribution
    weight_balance: float = 0.3         # Weight for workload balance
    weight_efficiency: float = 0.2      # Weight for efficiency matching
    
    # Constraints
    max_hours_per_worker: float = 8.0   # Maximum hours per worker
    allow_overtime: bool = False
    overtime_penalty: float = 1.5       # Multiplier for overtime cost
    
    # Solver settings
    time_limit_sec: float = 30.0
    mip_gap: float = 0.05


@dataclass
class Worker:
    """Worker data for assignment."""
    worker_id: str
    name: Optional[str] = None
    skill_score: float = 0.5
    available_hours: float = 8.0
    qualified_operations: Set[str] = field(default_factory=set)
    qualified_machines: Set[str] = field(default_factory=set)
    efficiency_by_op: Dict[str, float] = field(default_factory=dict)  # op_type -> efficiency
    
    def is_qualified(self, op_type: str, machine: str) -> bool:
        """Check if worker is qualified for operation."""
        op_ok = not self.qualified_operations or op_type in self.qualified_operations
        machine_ok = not self.qualified_machines or machine in self.qualified_machines
        return op_ok and machine_ok


@dataclass
class Operation:
    """Operation to be assigned."""
    operation_id: str
    op_type: str
    machine_id: str
    duration_hours: float
    priority: float = 1.0
    order_id: Optional[str] = None
    article_id: Optional[str] = None


@dataclass
class WorkerAssignment:
    """Single assignment result."""
    worker_id: str
    operation_id: str
    expected_contribution: float  # skill_score * priority
    expected_duration: float
    efficiency_factor: float


@dataclass
class AssignmentPlan:
    """
    Complete assignment plan.
    """
    timestamp: str
    config: AssignmentConfig
    assignments: List[WorkerAssignment] = field(default_factory=list)
    
    # Aggregates
    total_workers: int = 0
    total_operations: int = 0
    assigned_operations: int = 0
    unassigned_operations: List[str] = field(default_factory=list)
    
    # Quality metrics
    total_skill_contribution: float = 0.0
    avg_efficiency: float = 1.0
    workload_std: float = 0.0  # Standard deviation of workload
    
    # By worker
    workload_by_worker: Dict[str, float] = field(default_factory=dict)
    
    # Solver info
    solver_status: str = "NOT_SOLVED"
    objective_value: Optional[float] = None
    solve_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'total_workers': self.total_workers,
            'total_operations': self.total_operations,
            'assigned_operations': self.assigned_operations,
            'unassigned_operations': self.unassigned_operations,
            'total_skill_contribution': round(self.total_skill_contribution, 2),
            'avg_efficiency': round(self.avg_efficiency, 3),
            'workload_std': round(self.workload_std, 2),
            'workload_by_worker': {k: round(v, 2) for k, v in self.workload_by_worker.items()},
            'solver_status': self.solver_status,
            'objective_value': round(self.objective_value, 2) if self.objective_value else None,
            'solve_time_sec': round(self.solve_time_sec, 3),
            'assignments': [
                {
                    'worker_id': a.worker_id,
                    'operation_id': a.operation_id,
                    'expected_contribution': round(a.expected_contribution, 3),
                    'expected_duration': round(a.expected_duration, 2),
                    'efficiency_factor': round(a.efficiency_factor, 3),
                }
                for a in self.assignments
            ],
        }


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# GREEDY HEURISTIC (Baseline)
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def _assign_greedy(
    workers: List[Worker],
    operations: List[Operation],
    config: AssignmentConfig
) -> AssignmentPlan:
    """
    Greedy assignment heuristic.
    
    Algorithm:
    1. Sort operations by priority (descending)
    2. For each operation:
       a. Find qualified workers with capacity
       b. Select worker with highest skill_score * efficiency
       c. Assign and update capacity
    """
    import time
    start_time = time.time()
    
    plan = AssignmentPlan(
        timestamp=datetime.now().isoformat(),
        config=config,
        total_workers=len(workers),
        total_operations=len(operations),
    )
    
    # Initialize worker capacities
    remaining_capacity = {w.worker_id: w.available_hours for w in workers}
    worker_map = {w.worker_id: w for w in workers}
    
    # Sort operations by priority
    sorted_ops = sorted(operations, key=lambda o: o.priority, reverse=True)
    
    efficiencies = []
    contributions = []
    
    for op in sorted_ops:
        # Find qualified workers with capacity
        candidates = []
        for w in workers:
            if remaining_capacity[w.worker_id] >= op.duration_hours:
                if w.is_qualified(op.op_type, op.machine_id):
                    # Get efficiency for this operation type
                    eff = w.efficiency_by_op.get(op.op_type, 1.0)
                    score = w.skill_score * eff * op.priority
                    candidates.append((w, score, eff))
        
        if not candidates:
            plan.unassigned_operations.append(op.operation_id)
            continue
        
        # Select best candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_worker, contribution, efficiency = candidates[0]
        
        # Create assignment
        assignment = WorkerAssignment(
            worker_id=best_worker.worker_id,
            operation_id=op.operation_id,
            expected_contribution=contribution,
            expected_duration=op.duration_hours,
            efficiency_factor=efficiency,
        )
        plan.assignments.append(assignment)
        
        # Update capacity
        remaining_capacity[best_worker.worker_id] -= op.duration_hours
        
        # Track metrics
        efficiencies.append(efficiency)
        contributions.append(contribution)
    
    # Compute aggregates
    plan.assigned_operations = len(plan.assignments)
    plan.total_skill_contribution = sum(contributions)
    plan.avg_efficiency = np.mean(efficiencies) if efficiencies else 1.0
    
    # Workload by worker
    for w in workers:
        assigned_hours = w.available_hours - remaining_capacity[w.worker_id]
        plan.workload_by_worker[w.worker_id] = assigned_hours
    
    workloads = list(plan.workload_by_worker.values())
    plan.workload_std = np.std(workloads) if workloads else 0.0
    
    plan.solver_status = "GREEDY"
    plan.objective_value = plan.total_skill_contribution
    plan.solve_time_sec = time.time() - start_time
    
    return plan


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# MILP OPTIMIZER
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def _assign_milp(
    workers: List[Worker],
    operations: List[Operation],
    config: AssignmentConfig
) -> AssignmentPlan:
    """
    MILP-based optimal assignment.
    
    Uses OR-Tools CBC solver.
    """
    try:
        from ortools.linear_solver import pywraplp
    except ImportError:
        logger.warning("OR-Tools not available. Using greedy heuristic.")
        return _assign_greedy(workers, operations, config)
    
    import time
    start_time = time.time()
    
    plan = AssignmentPlan(
        timestamp=datetime.now().isoformat(),
        config=config,
        total_workers=len(workers),
        total_operations=len(operations),
    )
    
    if not workers or not operations:
        plan.solver_status = "NO_DATA"
        return plan
    
    # Create solver
    solver = pywraplp.Solver.CreateSolver('CBC')
    if not solver:
        logger.warning("Could not create CBC solver. Using greedy.")
        return _assign_greedy(workers, operations, config)
    
    # Index mappings
    W = list(range(len(workers)))
    O = list(range(len(operations)))
    
    worker_map = {i: workers[i] for i in W}
    op_map = {j: operations[j] for j in O}
    
    # ════════════════════════════════════════════════════════════════════
    # VARIABLES
    # ════════════════════════════════════════════════════════════════════
    
    # z[w,o] = 1 if worker w assigned to operation o
    z = {}
    for i in W:
        for j in O:
            z[i, j] = solver.BoolVar(f'z_{i}_{j}')
    
    # ════════════════════════════════════════════════════════════════════
    # CONSTRAINTS
    # ════════════════════════════════════════════════════════════════════
    
    # (C1) Each operation assigned to at most one worker
    for j in O:
        solver.Add(
            sum(z[i, j] for i in W) <= 1,
            f'C1_assign_op_{j}'
        )
    
    # (C2) Worker capacity
    for i in W:
        w = worker_map[i]
        max_hours = w.available_hours
        if config.allow_overtime:
            max_hours *= 1.2  # Allow 20% overtime
        
        solver.Add(
            sum(op_map[j].duration_hours * z[i, j] for j in O) <= max_hours,
            f'C2_capacity_{i}'
        )
    
    # (C3, C4) Qualification constraints
    for i in W:
        w = worker_map[i]
        for j in O:
            op = op_map[j]
            if not w.is_qualified(op.op_type, op.machine_id):
                solver.Add(z[i, j] == 0, f'C3_qual_{i}_{j}')
    
    # ════════════════════════════════════════════════════════════════════
    # OBJECTIVE
    # ════════════════════════════════════════════════════════════════════
    
    # Maximize: Σ skill_score * priority * efficiency * z
    objective_terms = []
    
    for i in W:
        w = worker_map[i]
        for j in O:
            op = op_map[j]
            efficiency = w.efficiency_by_op.get(op.op_type, 1.0)
            contribution = w.skill_score * op.priority * efficiency
            objective_terms.append(contribution * z[i, j])
    
    # Add workload balance penalty (minimize variance approximation)
    # Using absolute deviation from mean as proxy
    avg_workload = sum(op.duration_hours for op in operations) / len(workers) if workers else 0
    
    # Auxiliary variables for deviation (simplified)
    # Full variance minimization would require quadratic or piecewise linear
    
    objective = solver.Objective()
    for i in W:
        w = worker_map[i]
        for j in O:
            op = op_map[j]
            efficiency = w.efficiency_by_op.get(op.op_type, 1.0)
            contribution = w.skill_score * op.priority * efficiency
            objective.SetCoefficient(z[i, j], config.weight_skill * contribution)
    
    objective.SetMaximization()
    
    # Solve
    solver.SetTimeLimit(int(config.time_limit_sec * 1000))
    status = solver.Solve()
    
    solve_time = time.time() - start_time
    
    # Extract solution
    status_name = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    }.get(status, "UNKNOWN")
    
    plan.solver_status = status_name
    plan.solve_time_sec = solve_time
    
    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        plan.objective_value = objective.Value()
        
        efficiencies = []
        contributions = []
        
        for i in W:
            w = worker_map[i]
            worker_hours = 0.0
            
            for j in O:
                if z[i, j].solution_value() > 0.5:
                    op = op_map[j]
                    efficiency = w.efficiency_by_op.get(op.op_type, 1.0)
                    contribution = w.skill_score * op.priority * efficiency
                    
                    assignment = WorkerAssignment(
                        worker_id=w.worker_id,
                        operation_id=op.operation_id,
                        expected_contribution=contribution,
                        expected_duration=op.duration_hours,
                        efficiency_factor=efficiency,
                    )
                    plan.assignments.append(assignment)
                    
                    worker_hours += op.duration_hours
                    efficiencies.append(efficiency)
                    contributions.append(contribution)
            
            plan.workload_by_worker[w.worker_id] = worker_hours
        
        # Check for unassigned operations
        assigned_ops = {a.operation_id for a in plan.assignments}
        for op in operations:
            if op.operation_id not in assigned_ops:
                plan.unassigned_operations.append(op.operation_id)
        
        plan.assigned_operations = len(plan.assignments)
        plan.total_skill_contribution = sum(contributions)
        plan.avg_efficiency = np.mean(efficiencies) if efficiencies else 1.0
        
        workloads = list(plan.workload_by_worker.values())
        plan.workload_std = np.std(workloads) if workloads else 0.0
    
    else:
        # Fallback to greedy
        logger.warning(f"MILP status={status_name}. Falling back to greedy.")
        return _assign_greedy(workers, operations, config)
    
    return plan


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def optimize_worker_assignment(
    workers: List[Worker],
    operations: List[Operation],
    config: Optional[AssignmentConfig] = None,
    use_milp: bool = True
) -> AssignmentPlan:
    """
    Optimize worker-to-operation assignment.
    
    Args:
        workers: List of Worker objects
        operations: List of Operation objects
        config: Assignment configuration
        use_milp: Whether to use MILP (slower but optimal)
    
    Returns:
        AssignmentPlan with optimal assignments
    """
    config = config or AssignmentConfig()
    
    if use_milp:
        return _assign_milp(workers, operations, config)
    else:
        return _assign_greedy(workers, operations, config)


def build_workers_from_performance(
    performances: Dict[str, 'WorkerPerformance'],
    default_available_hours: float = 8.0
) -> List[Worker]:
    """
    Build Worker objects from computed performances.
    """
    from .workforce_performance_engine import WorkerPerformance
    
    workers = []
    
    for wid, perf in performances.items():
        w = Worker(
            worker_id=wid,
            name=perf.metrics.worker_name,
            skill_score=perf.metrics.skill_score,
            available_hours=default_available_hours,
            qualified_operations=set(perf.qualified_operations),
            qualified_machines=set(perf.qualified_machines),
            efficiency_by_op=perf.metrics_by_operation.get('efficiency', {}),
        )
        workers.append(w)
    
    return workers


def build_operations_from_plan(
    plan_df: pd.DataFrame,
    op_id_col: str = 'operation_id',
    op_type_col: str = 'op_code',
    machine_col: str = 'machine_id',
    duration_col: str = 'duration_min',
    priority_col: str = 'priority',
) -> List[Operation]:
    """
    Build Operation objects from production plan.
    """
    operations = []
    
    for idx, row in plan_df.iterrows():
        op_id = str(row.get(op_id_col, f'OP-{idx}'))
        op_type = str(row.get(op_type_col, 'UNKNOWN'))
        machine = str(row.get(machine_col, 'M-000'))
        duration = float(row.get(duration_col, 60)) / 60  # Convert to hours
        priority = float(row.get(priority_col, 1.0))
        
        op = Operation(
            operation_id=op_id,
            op_type=op_type,
            machine_id=machine,
            duration_hours=duration,
            priority=priority,
            order_id=str(row.get('order_id', '')) if 'order_id' in row else None,
            article_id=str(row.get('article_id', '')) if 'article_id' in row else None,
        )
        operations.append(op)
    
    return operations



