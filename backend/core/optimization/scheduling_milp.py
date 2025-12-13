"""
═══════════════════════════════════════════════════════════════════════════════
              PRODPLAN 4.0 - MILP SCHEDULING MODEL FORMULATION
═══════════════════════════════════════════════════════════════════════════════

This module implements a Mixed-Integer Linear Programming (MILP) formulation
for the Flexible Job-Shop Scheduling Problem (FJSP).

Mathematical Formulation
════════════════════════

SETS
────
    O = {1, 2, ..., n}          Set of operations (indexed by o)
    M = {1, 2, ..., m}          Set of machines (indexed by m)
    J = {1, 2, ..., j}          Set of jobs/orders (indexed by j)
    F = {1, 2, ..., f}          Set of setup families (indexed by f)
    
    E_o ⊆ M                     Set of eligible machines for operation o
    Pred(o) ⊆ O                 Set of predecessor operations of o

PARAMETERS
──────────
    p_{o,m} ∈ ℝ⁺               Processing time of operation o on machine m
    s_{f,f'} ∈ ℝ⁺              Setup time when switching from family f to f'
    d_j ∈ ℝ⁺                   Due date of job j
    w_j ∈ ℝ⁺                   Priority weight of job j
    M ∈ ℝ⁺                     Big-M constant (upper bound on makespan)
    
    job(o) → j                  Maps operation o to its job j
    family(o) → f               Maps operation o to its setup family f

DECISION VARIABLES
──────────────────
    x_{o,m} ∈ {0, 1}           1 if operation o is assigned to machine m
    S_o ∈ ℝ⁺                   Start time of operation o
    C_o ∈ ℝ⁺                   Completion time of operation o
    y_{o,o'} ∈ {0, 1}          1 if operation o precedes o' on same machine
    T_j ∈ ℝ⁺                   Tardiness of job j
    C_max ∈ ℝ⁺                 Makespan (maximum completion time)

CONSTRAINTS
───────────

(C1) Machine Assignment - Each operation assigned to exactly one eligible machine:
     
     ∑_{m ∈ E_o} x_{o,m} = 1                              ∀ o ∈ O

(C2) Processing Time - Completion = Start + Processing:
     
     C_o = S_o + ∑_{m ∈ E_o} p_{o,m} · x_{o,m}            ∀ o ∈ O

(C3) Precedence - Operations within same job must respect sequence:
     
     S_{o'} ≥ C_o                                         ∀ o ∈ O, o' ∈ Succ(o)

(C4) No-Overlap - Operations on same machine cannot overlap (Big-M):
     For all operations o, o' that can share machine m:
     
     S_{o'} ≥ C_o + s_{family(o), family(o')} - M(1 - y_{o,o'}) - M(2 - x_{o,m} - x_{o',m})
     S_o ≥ C_{o'} + s_{family(o'), family(o)} - M·y_{o,o'} - M(2 - x_{o,m} - x_{o',m})

(C5) Makespan Definition:
     
     C_max ≥ C_o                                          ∀ o ∈ O

(C6) Tardiness Definition:
     
     T_j ≥ max_{o: job(o)=j} C_o - d_j                    ∀ j ∈ J
     T_j ≥ 0                                              ∀ j ∈ J

OBJECTIVE FUNCTION
──────────────────

Multi-objective (weighted sum):

    min  α · C_max + β · ∑_j w_j · T_j + γ · TotalSetup

where:
    α, β, γ ≥ 0 are weights (default: α=1, β=0.5, γ=0.1)
    TotalSetup = estimated total setup time

Complexity
──────────
- Variables: O(|O| × |M| + |O|²)
- Constraints: O(|O|² × |M|)
- NP-hard in general

Solver Compatibility
────────────────────
- OR-Tools CBC (open-source, default)
- OR-Tools SCIP (open-source, often faster)
- Gurobi (commercial, best performance)
- CPLEX (commercial)

R&D / SIFIDE: WP1 - Intelligent APS Core
Research Questions:
- Q1.1: MILP vs heuristic makespan improvement for |O| < 100
- Q1.2: Impact of setup time modeling on solution quality
- Q1.3: Warm-start from heuristic: computational speedup?

References:
[1] Manne, A. (1960). On the Job-Shop Scheduling Problem. Operations Research.
[2] Özgüven, C. et al. (2010). Mathematical models for job-shop scheduling.
[3] Graham, R.L. et al. (1979). Optimization and Approximation in Deterministic
    Sequencing and Scheduling: A Survey. Annals of Discrete Mathematics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..data_loader import DataBundle

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

class ObjectiveType(Enum):
    """Optimization objective type."""
    MINIMIZE_MAKESPAN = "min_makespan"
    MINIMIZE_TARDINESS = "min_tardiness"
    MINIMIZE_WEIGHTED_SUM = "min_weighted_sum"
    MINIMIZE_SETUP = "min_setup"


@dataclass(frozen=True)
class MILPConfig:
    """
    Configuration for MILP model.
    
    Attributes:
        objective: Type of objective function
        alpha: Weight for makespan (Cmax)
        beta: Weight for weighted tardiness (Σ wj·Tj)
        gamma: Weight for total setup time
        time_limit_sec: Solver time limit in seconds
        mip_gap: Acceptable optimality gap (0.05 = 5%)
        num_threads: Number of solver threads
        big_m_factor: Factor to compute Big-M constant
        include_setup: Whether to model sequence-dependent setup times
        verbose: Print solver output
    """
    objective: ObjectiveType = ObjectiveType.MINIMIZE_WEIGHTED_SUM
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.1
    time_limit_sec: float = 60.0
    mip_gap: float = 0.05
    num_threads: int = 4
    big_m_factor: float = 2.0
    include_setup: bool = True
    verbose: bool = False


@dataclass
class SolverStatistics:
    """
    Statistics from MILP solver.
    
    Used for R&D analysis and performance benchmarking.
    """
    status: str = "NOT_SOLVED"
    objective_value: Optional[float] = None
    best_bound: Optional[float] = None
    mip_gap: Optional[float] = None
    solve_time_sec: float = 0.0
    num_variables: int = 0
    num_constraints: int = 0
    num_iterations: int = 0
    num_nodes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status,
            'objective_value': self.objective_value,
            'best_bound': self.best_bound,
            'mip_gap': round(self.mip_gap, 4) if self.mip_gap else None,
            'solve_time_sec': round(self.solve_time_sec, 3),
            'num_variables': self.num_variables,
            'num_constraints': self.num_constraints,
            'num_iterations': self.num_iterations,
            'num_nodes': self.num_nodes,
        }


# ════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Operation:
    """
    Represents a single operation in the scheduling problem.
    
    Corresponds to index o ∈ O in the MILP formulation.
    """
    id: str                                    # Unique identifier
    job_id: str                                # Job/order this operation belongs to
    article_id: str                            # Article being processed
    sequence: int                              # Position in job sequence (op_seq)
    op_code: str                               # Operation type code
    setup_family: str                          # Setup family for changeover calculation
    eligible_machines: List[str]               # Set E_o of eligible machines
    processing_times: Dict[str, float]         # p_{o,m} for each eligible machine
    predecessors: List[str] = field(default_factory=list)  # Pred(o)
    successors: List[str] = field(default_factory=list)    # Succ(o)
    route_id: Optional[str] = None
    route_label: Optional[str] = None


@dataclass
class Job:
    """
    Represents a job (production order) containing multiple operations.
    
    Corresponds to index j ∈ J in the MILP formulation.
    """
    id: str                      # job_id
    article_id: str              # Article being produced
    quantity: int                # Quantity to produce
    due_date: Optional[datetime] # d_j (due date)
    priority: float = 1.0        # w_j (weight for tardiness)
    operations: List[str] = field(default_factory=list)  # Operation IDs in sequence


@dataclass
class Machine:
    """
    Represents a machine resource.
    
    Corresponds to index m ∈ M in the MILP formulation.
    """
    id: str
    name: str = ""
    work_center: str = ""
    speed_factor: float = 1.0
    available_from: Optional[datetime] = None
    available_until: Optional[datetime] = None


@dataclass
class ScheduleResult:
    """
    Result of solving the scheduling MILP.
    
    Contains the complete schedule and solver statistics.
    """
    success: bool
    schedule_df: Optional[pd.DataFrame] = None
    statistics: SolverStatistics = field(default_factory=SolverStatistics)
    objective_breakdown: Dict[str, float] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════════
# MILP MODEL
# ════════════════════════════════════════════════════════════════════════════

class SchedulingMILP:
    """
    MILP model for Flexible Job-Shop Scheduling.
    
    Implements the formulation described in the module docstring.
    Uses OR-Tools as the solver backend.
    
    Usage:
        model = SchedulingMILP(config)
        model.load_problem(operations, jobs, machines, setup_matrix)
        model.build_model()
        result = model.solve()
    
    Example:
        >>> config = MILPConfig(time_limit_sec=30, alpha=1.0, beta=0.5)
        >>> model = SchedulingMILP(config)
        >>> model.load_problem(ops, jobs, machines, setup_matrix)
        >>> model.build_model()
        >>> result = model.solve()
        >>> if result.success:
        ...     print(f"Makespan: {result.objective_breakdown['makespan']:.1f} min")
    """
    
    def __init__(self, config: Optional[MILPConfig] = None):
        self.config = config or MILPConfig()
        
        # Problem data
        self.operations: Dict[str, Operation] = {}
        self.jobs: Dict[str, Job] = {}
        self.machines: Dict[str, Machine] = {}
        self.setup_matrix: Dict[Tuple[str, str], float] = {}
        
        # Computed parameters
        self._big_m: float = 0.0
        self._horizon_start: datetime = datetime.now()
        
        # Solver state
        self._solver = None
        self._model_built = False
        
        # Decision variables (stored for solution extraction)
        self._x: Dict[Tuple[str, str], Any] = {}      # x_{o,m}
        self._S: Dict[str, Any] = {}                   # S_o
        self._C: Dict[str, Any] = {}                   # C_o
        self._y: Dict[Tuple[str, str, str], Any] = {}  # y_{o,o',m}
        self._T: Dict[str, Any] = {}                   # T_j
        self._Cmax: Any = None                         # C_max
    
    def load_problem(
        self,
        operations: List[Operation],
        jobs: List[Job],
        machines: List[Machine],
        setup_matrix: Optional[Dict[Tuple[str, str], float]] = None,
        horizon_start: Optional[datetime] = None
    ) -> None:
        """
        Load problem data into the model.
        
        Args:
            operations: List of Operation objects
            jobs: List of Job objects
            machines: List of Machine objects
            setup_matrix: Dict (from_family, to_family) -> setup_time_min
            horizon_start: Start of planning horizon
        """
        self.operations = {op.id: op for op in operations}
        self.jobs = {job.id: job for job in jobs}
        self.machines = {m.id: m for m in machines}
        self.setup_matrix = setup_matrix or {}
        self._horizon_start = horizon_start or datetime.now()
        
        # Compute Big-M constant
        # M = factor × (sum of all max processing times + setup times)
        total_proc = sum(
            max(op.processing_times.values()) if op.processing_times else 0
            for op in self.operations.values()
        )
        max_setup = max(self.setup_matrix.values()) if self.setup_matrix else 0
        self._big_m = self.config.big_m_factor * (total_proc + len(self.operations) * max_setup)
        
        self._model_built = False
        
        logger.info(
            f"Problem loaded: |O|={len(operations)}, |J|={len(jobs)}, "
            f"|M|={len(machines)}, Big-M={self._big_m:.0f}"
        )
    
    def build_model(self) -> None:
        """
        Build the MILP model using OR-Tools.
        
        Implements constraints (C1)-(C6) from the formulation.
        """
        try:
            from ortools.linear_solver import pywraplp
        except ImportError:
            raise ImportError(
                "OR-Tools not installed. Install with: pip install ortools"
            )
        
        # Create solver
        self._solver = pywraplp.Solver.CreateSolver('CBC')
        if not self._solver:
            self._solver = pywraplp.Solver.CreateSolver('SCIP')
        if not self._solver:
            raise RuntimeError("No MILP solver available in OR-Tools")
        
        solver = self._solver
        infinity = solver.infinity()
        M = self._big_m
        
        # ════════════════════════════════════════════════════════════════════
        # DECISION VARIABLES
        # ════════════════════════════════════════════════════════════════════
        
        # x_{o,m} ∈ {0, 1} - machine assignment
        for op_id, op in self.operations.items():
            for m_id in op.eligible_machines:
                self._x[(op_id, m_id)] = solver.BoolVar(f'x[{op_id},{m_id}]')
        
        # S_o, C_o ∈ ℝ⁺ - start and completion times
        for op_id in self.operations:
            self._S[op_id] = solver.NumVar(0, M, f'S[{op_id}]')
            self._C[op_id] = solver.NumVar(0, M, f'C[{op_id}]')
        
        # y_{o,o',m} ∈ {0, 1} - sequencing on same machine
        # Only needed for pairs that can share a machine
        for m_id in self.machines:
            ops_on_machine = [
                op_id for op_id, op in self.operations.items()
                if m_id in op.eligible_machines
            ]
            for i, o1 in enumerate(ops_on_machine):
                for o2 in ops_on_machine[i+1:]:
                    self._y[(o1, o2, m_id)] = solver.BoolVar(f'y[{o1},{o2},{m_id}]')
        
        # T_j ∈ ℝ⁺ - tardiness per job
        for job_id in self.jobs:
            self._T[job_id] = solver.NumVar(0, infinity, f'T[{job_id}]')
        
        # C_max ∈ ℝ⁺ - makespan
        self._Cmax = solver.NumVar(0, M, 'Cmax')
        
        # ════════════════════════════════════════════════════════════════════
        # CONSTRAINT (C1): Machine Assignment
        # ∑_{m ∈ E_o} x_{o,m} = 1  ∀ o ∈ O
        # ════════════════════════════════════════════════════════════════════
        
        for op_id, op in self.operations.items():
            solver.Add(
                sum(self._x[(op_id, m_id)] for m_id in op.eligible_machines) == 1,
                f'C1_assign[{op_id}]'
            )
        
        # ════════════════════════════════════════════════════════════════════
        # CONSTRAINT (C2): Processing Time
        # C_o = S_o + ∑_{m ∈ E_o} p_{o,m} · x_{o,m}  ∀ o ∈ O
        # ════════════════════════════════════════════════════════════════════
        
        for op_id, op in self.operations.items():
            proc_time_expr = sum(
                op.processing_times.get(m_id, 0) * self._x[(op_id, m_id)]
                for m_id in op.eligible_machines
            )
            solver.Add(
                self._C[op_id] == self._S[op_id] + proc_time_expr,
                f'C2_proc[{op_id}]'
            )
        
        # ════════════════════════════════════════════════════════════════════
        # CONSTRAINT (C3): Precedence
        # S_{o'} ≥ C_o  ∀ o ∈ O, o' ∈ Succ(o)
        # ════════════════════════════════════════════════════════════════════
        
        for op_id, op in self.operations.items():
            for succ_id in op.successors:
                if succ_id in self.operations:
                    solver.Add(
                        self._S[succ_id] >= self._C[op_id],
                        f'C3_prec[{op_id},{succ_id}]'
                    )
        
        # ════════════════════════════════════════════════════════════════════
        # CONSTRAINT (C4): No-Overlap (Disjunctive)
        # Big-M formulation for pairs on same machine
        # ════════════════════════════════════════════════════════════════════
        
        for (o1, o2, m_id), y_var in self._y.items():
            op1 = self.operations[o1]
            op2 = self.operations[o2]
            
            x1 = self._x[(o1, m_id)]
            x2 = self._x[(o2, m_id)]
            
            # Setup time from family(o1) to family(o2)
            setup_12 = self.setup_matrix.get(
                (op1.setup_family, op2.setup_family), 0
            ) if self.config.include_setup else 0
            
            # Setup time from family(o2) to family(o1)
            setup_21 = self.setup_matrix.get(
                (op2.setup_family, op1.setup_family), 0
            ) if self.config.include_setup else 0
            
            # If y=1: o1 before o2
            # S[o2] ≥ C[o1] + setup_12 - M(1-y) - M(2 - x1 - x2)
            solver.Add(
                self._S[o2] >= self._C[o1] + setup_12 - M * (1 - y_var) - M * (2 - x1 - x2),
                f'C4a_nooverlap[{o1},{o2},{m_id}]'
            )
            
            # If y=0: o2 before o1
            # S[o1] ≥ C[o2] + setup_21 - M·y - M(2 - x1 - x2)
            solver.Add(
                self._S[o1] >= self._C[o2] + setup_21 - M * y_var - M * (2 - x1 - x2),
                f'C4b_nooverlap[{o1},{o2},{m_id}]'
            )
        
        # ════════════════════════════════════════════════════════════════════
        # CONSTRAINT (C5): Makespan
        # C_max ≥ C_o  ∀ o ∈ O
        # ════════════════════════════════════════════════════════════════════
        
        for op_id in self.operations:
            solver.Add(
                self._Cmax >= self._C[op_id],
                f'C5_makespan[{op_id}]'
            )
        
        # ════════════════════════════════════════════════════════════════════
        # CONSTRAINT (C6): Tardiness
        # T_j ≥ C_{last_op_j} - d_j  ∀ j ∈ J with due date
        # ════════════════════════════════════════════════════════════════════
        
        for job_id, job in self.jobs.items():
            if job.due_date and job.operations:
                # Find last operation of this job
                last_op_id = job.operations[-1]
                if last_op_id in self._C:
                    # Convert due date to minutes from horizon start
                    due_minutes = (job.due_date - self._horizon_start).total_seconds() / 60
                    
                    solver.Add(
                        self._T[job_id] >= self._C[last_op_id] - due_minutes,
                        f'C6_tardiness[{job_id}]'
                    )
        
        # ════════════════════════════════════════════════════════════════════
        # OBJECTIVE FUNCTION
        # min α · C_max + β · ∑_j w_j · T_j + γ · TotalSetup
        # ════════════════════════════════════════════════════════════════════
        
        obj_type = self.config.objective
        alpha = self.config.alpha
        beta = self.config.beta
        gamma = self.config.gamma
        
        if obj_type == ObjectiveType.MINIMIZE_MAKESPAN:
            solver.Minimize(self._Cmax)
        
        elif obj_type == ObjectiveType.MINIMIZE_TARDINESS:
            tardiness_expr = sum(
                self.jobs[job_id].priority * self._T[job_id]
                for job_id in self.jobs
            )
            solver.Minimize(tardiness_expr)
        
        elif obj_type == ObjectiveType.MINIMIZE_WEIGHTED_SUM:
            # Weighted tardiness
            weighted_tardiness = sum(
                self.jobs[job_id].priority * self._T[job_id]
                for job_id in self.jobs
            )
            
            # Objective: α·Cmax + β·ΣwjTj
            # Note: Setup is implicitly minimized through sequencing
            objective = alpha * self._Cmax + beta * weighted_tardiness
            solver.Minimize(objective)
        
        self._model_built = True
        
        logger.info(
            f"MILP model built: {solver.NumVariables()} variables, "
            f"{solver.NumConstraints()} constraints"
        )
    
    def solve(self) -> ScheduleResult:
        """
        Solve the MILP model.
        
        Returns:
            ScheduleResult with schedule DataFrame and statistics
        """
        if not self._model_built:
            raise RuntimeError("Model not built. Call build_model() first.")
        
        from ortools.linear_solver import pywraplp
        
        solver = self._solver
        
        # Set solver parameters
        solver.SetTimeLimit(int(self.config.time_limit_sec * 1000))  # milliseconds
        if self.config.num_threads > 1:
            solver.SetNumThreads(self.config.num_threads)
        
        logger.info(
            f"Solving MILP (time_limit={self.config.time_limit_sec}s, "
            f"gap={self.config.mip_gap*100}%)..."
        )
        
        import time
        start_time = time.time()
        status = solver.Solve()
        solve_time = time.time() - start_time
        
        # Build statistics
        stats = SolverStatistics(
            status=self._status_name(status),
            solve_time_sec=solve_time,
            num_variables=solver.NumVariables(),
            num_constraints=solver.NumConstraints(),
            num_iterations=solver.iterations(),
        )
        
        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            stats.objective_value = solver.Objective().Value()
            
            # Extract solution
            schedule_df = self._extract_solution()
            objective_breakdown = self._compute_objective_breakdown()
            
            logger.info(
                f"Solution found: status={stats.status}, "
                f"objective={stats.objective_value:.2f}, "
                f"time={solve_time:.2f}s"
            )
            
            return ScheduleResult(
                success=True,
                schedule_df=schedule_df,
                statistics=stats,
                objective_breakdown=objective_breakdown
            )
        
        else:
            logger.warning(f"MILP solver failed: status={stats.status}")
            return ScheduleResult(success=False, statistics=stats)
    
    def _status_name(self, status: int) -> str:
        """Convert OR-Tools status to readable name."""
        from ortools.linear_solver import pywraplp
        names = {
            pywraplp.Solver.OPTIMAL: 'OPTIMAL',
            pywraplp.Solver.FEASIBLE: 'FEASIBLE',
            pywraplp.Solver.INFEASIBLE: 'INFEASIBLE',
            pywraplp.Solver.UNBOUNDED: 'UNBOUNDED',
            pywraplp.Solver.ABNORMAL: 'ABNORMAL',
            pywraplp.Solver.NOT_SOLVED: 'NOT_SOLVED',
        }
        return names.get(status, f'UNKNOWN({status})')
    
    def _extract_solution(self) -> pd.DataFrame:
        """Extract schedule from solved model."""
        records = []
        
        for op_id, op in self.operations.items():
            # Find assigned machine
            assigned_machine = None
            for m_id in op.eligible_machines:
                if self._x[(op_id, m_id)].solution_value() > 0.5:
                    assigned_machine = m_id
                    break
            
            start_min = self._S[op_id].solution_value()
            end_min = self._C[op_id].solution_value()
            
            # Convert to datetime
            start_time = self._horizon_start + timedelta(minutes=start_min)
            end_time = self._horizon_start + timedelta(minutes=end_min)
            
            records.append({
                'operation_id': op_id,
                'order_id': op.job_id,
                'article_id': op.article_id,
                'op_seq': op.sequence,
                'op_code': op.op_code,
                'machine_id': assigned_machine,
                'route_id': op.route_id,
                'route_label': op.route_label,
                'setup_family': op.setup_family,
                'start_time': start_time,
                'end_time': end_time,
                'start_min': start_min,
                'end_min': end_min,
                'duration_min': end_min - start_min,
            })
        
        return pd.DataFrame(records).sort_values(['machine_id', 'start_min'])
    
    def _compute_objective_breakdown(self) -> Dict[str, float]:
        """Compute individual objective components."""
        makespan = self._Cmax.solution_value()
        
        total_tardiness = sum(
            self._T[job_id].solution_value()
            for job_id in self.jobs
        )
        
        weighted_tardiness = sum(
            self.jobs[job_id].priority * self._T[job_id].solution_value()
            for job_id in self.jobs
        )
        
        return {
            'makespan_min': makespan,
            'makespan_hours': makespan / 60,
            'total_tardiness_min': total_tardiness,
            'weighted_tardiness_min': weighted_tardiness,
        }


# ════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def build_milp_from_data(
    orders_df: pd.DataFrame,
    routing_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    setup_matrix_df: Optional[pd.DataFrame] = None,
    config: Optional[MILPConfig] = None
) -> SchedulingMILP:
    """
    Build MILP model from DataFrames.
    
    This is the main entry point for creating a MILP model from
    raw data (as loaded from Excel).
    
    Args:
        orders_df: Orders with columns [order_id, article_id, qty, due_date, priority]
        routing_df: Routing with columns [article_id, route_id, route_label, op_seq, 
                                          op_code, primary_machine_id, base_time_per_unit_min,
                                          setup_family]
        machines_df: Machines with columns [machine_id, work_center_group, speed_factor]
        setup_matrix_df: Setup times with columns [from_setup_family, to_setup_family, setup_time_min]
        config: MILP configuration
    
    Returns:
        Configured SchedulingMILP ready to build and solve
    
    TODO[R&D]: Add instance size heuristics:
    - If |O| > 200, warn about computational time
    - Suggest CP-SAT for very large instances
    - Implement decomposition for extremely large problems
    """
    config = config or MILPConfig()
    
    # Parse setup matrix
    setup_matrix: Dict[Tuple[str, str], float] = {}
    if setup_matrix_df is not None:
        for _, row in setup_matrix_df.iterrows():
            key = (str(row['from_setup_family']), str(row['to_setup_family']))
            setup_matrix[key] = float(row['setup_time_min'])
    
    # Build machines
    machines = []
    for _, row in machines_df.iterrows():
        machines.append(Machine(
            id=str(row['machine_id']),
            name=str(row.get('description', '')),
            work_center=str(row.get('work_center_group', '')),
            speed_factor=float(row.get('speed_factor', 1.0)),
        ))
    machine_ids = {m.id for m in machines}
    
    # Build jobs and operations
    jobs = []
    operations = []
    
    for _, order in orders_df.iterrows():
        order_id = str(order['order_id'])
        article_id = str(order['article_id'])
        qty = int(order['qty'])
        due_date = pd.to_datetime(order.get('due_date')) if 'due_date' in order else None
        priority = float(order.get('priority', 1))
        
        # Get routing for this article
        article_routing = routing_df[routing_df['article_id'] == article_id]
        
        if article_routing.empty:
            logger.warning(f"No routing for article {article_id}, skipping order {order_id}")
            continue
        
        # Choose first route (TODO: route selection logic)
        route_id = str(article_routing.iloc[0]['route_id'])
        route_label = str(article_routing.iloc[0]['route_label'])
        route_ops = article_routing[article_routing['route_id'] == route_id].sort_values('op_seq')
        
        job_op_ids = []
        prev_op_id = None
        
        for _, op_row in route_ops.iterrows():
            op_id = f"{order_id}_{int(op_row['op_seq'])}"
            job_op_ids.append(op_id)
            
            # Processing time
            base_time = float(op_row['base_time_per_unit_min'])
            primary_machine = str(op_row['primary_machine_id'])
            
            # Get eligible machines
            eligible = [primary_machine]
            if 'alt_machine_ids' in op_row and pd.notna(op_row['alt_machine_ids']):
                alt_str = str(op_row['alt_machine_ids'])
                if alt_str:
                    eligible.extend([m.strip() for m in alt_str.split(',') if m.strip()])
            
            # Filter to existing machines
            eligible = [m for m in eligible if m in machine_ids]
            if not eligible:
                eligible = [primary_machine]  # Fallback
            
            # Processing times per machine (adjusted by speed factor and quantity)
            proc_times = {}
            for m_id in eligible:
                m = next((m for m in machines if m.id == m_id), None)
                speed = m.speed_factor if m else 1.0
                proc_times[m_id] = base_time * qty / speed
            
            operation = Operation(
                id=op_id,
                job_id=order_id,
                article_id=article_id,
                sequence=int(op_row['op_seq']),
                op_code=str(op_row['op_code']),
                setup_family=str(op_row.get('setup_family', 'default')),
                eligible_machines=eligible,
                processing_times=proc_times,
                predecessors=[prev_op_id] if prev_op_id else [],
                successors=[],
                route_id=route_id,
                route_label=route_label,
            )
            
            # Update predecessor's successors
            if prev_op_id:
                for op in operations:
                    if op.id == prev_op_id:
                        op.successors.append(op_id)
            
            operations.append(operation)
            prev_op_id = op_id
        
        job = Job(
            id=order_id,
            article_id=article_id,
            quantity=qty,
            due_date=due_date if isinstance(due_date, datetime) else None,
            priority=priority,
            operations=job_op_ids,
        )
        jobs.append(job)
    
    # Create and load model
    model = SchedulingMILP(config)
    model.load_problem(operations, jobs, machines, setup_matrix)
    
    return model



