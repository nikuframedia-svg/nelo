"""
ProdPlan 4.0 - Scheduling Optimization Models

This module defines mathematical programming models for production scheduling:
- MILP (Mixed-Integer Linear Programming)
- CP-SAT (Constraint Programming with SAT solver)

Design Philosophy:
- Models are defined independently of solvers
- Each model exposes: variables, constraints, objectives
- Solver interface handles instantiation and solving

R&D / SIFIDE: WP1 - Intelligent APS Core
Research Question Q1.1: Can MILP-based scheduling reduce makespan by ≥10% 
                        compared to heuristic methods for small instances?
Metrics: makespan, total tardiness, computation time, optimality gap.

References:
- Pinedo, M. (2016). Scheduling: Theory, Algorithms, and Systems.
- Job-shop and flexible job-shop formulations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS AND CONFIG
# ============================================================

class ModelType(Enum):
    """Type of mathematical model."""
    MILP = "milp"
    CPSAT = "cpsat"
    HEURISTIC = "heuristic"


class ObjectiveType(Enum):
    """Type of objective function."""
    MINIMIZE_MAKESPAN = "minimize_makespan"
    MINIMIZE_TARDINESS = "minimize_tardiness"
    MINIMIZE_SETUP = "minimize_setup"
    MINIMIZE_WEIGHTED_SUM = "minimize_weighted_sum"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class SchedulingConfig:
    """Configuration for scheduling model."""
    model_type: ModelType = ModelType.MILP
    objective_type: ObjectiveType = ObjectiveType.MINIMIZE_MAKESPAN
    
    # Time discretization
    time_horizon_hours: int = 168  # 1 week
    time_granularity_min: int = 15  # 15-minute slots
    
    # Solver parameters
    time_limit_sec: float = 60.0
    optimality_gap: float = 0.05  # 5% gap
    num_workers: int = 4
    
    # Objective weights (for weighted sum)
    weight_makespan: float = 1.0
    weight_tardiness: float = 0.5
    weight_setup: float = 0.2
    weight_load_balance: float = 0.1
    
    # Model options
    allow_preemption: bool = False
    allow_parallel_machines: bool = True
    include_setup_times: bool = True
    include_machine_downtime: bool = True
    
    # R&D flags
    log_model_statistics: bool = True
    export_model_file: bool = False


@dataclass
class Operation:
    """Represents a single operation to be scheduled."""
    id: str
    order_id: str
    article_id: str
    op_seq: int
    op_code: str
    processing_time_min: float
    setup_family: str
    
    # Machine compatibility
    eligible_machines: List[str] = field(default_factory=list)
    primary_machine: Optional[str] = None
    
    # Timing constraints
    earliest_start: Optional[datetime] = None
    due_date: Optional[datetime] = None
    
    # Predecessors (for precedence constraints)
    predecessors: List[str] = field(default_factory=list)
    
    # Route info
    route_id: Optional[str] = None
    route_label: Optional[str] = None


@dataclass 
class Machine:
    """Represents a machine resource."""
    id: str
    work_center: str
    speed_factor: float = 1.0
    
    # Availability windows
    available_from: Optional[datetime] = None
    available_until: Optional[datetime] = None
    
    # Downtime periods
    downtime_windows: List[Tuple[datetime, datetime]] = field(default_factory=list)
    
    # Capacity (for parallel processing)
    capacity: int = 1


# ============================================================
# ABSTRACT BASE CLASS
# ============================================================

class SchedulingModel(ABC):
    """
    Abstract base class for scheduling optimization models.
    
    Design Pattern: Template Method
    - Subclasses implement specific model formulations (MILP, CP-SAT)
    - Base class provides common interface and utilities
    
    TODO[R&D]: Benchmark different model formulations:
    - Position-based vs time-indexed MILP
    - Disjunctive vs cumulative constraints in CP
    - Impact on solution quality and computation time
    """
    
    def __init__(self, config: Optional[SchedulingConfig] = None):
        self.config = config or SchedulingConfig()
        self.operations: List[Operation] = []
        self.machines: List[Machine] = []
        self.setup_matrix: Dict[Tuple[str, str], float] = {}
        
        # Model state
        self._is_built = False
        self._solution: Optional[Dict[str, Any]] = None
        self._statistics: Dict[str, Any] = {}
    
    @abstractmethod
    def build_model(self) -> None:
        """Build the optimization model (variables, constraints, objective)."""
        pass
    
    @abstractmethod
    def solve(self) -> bool:
        """
        Solve the model.
        
        Returns:
            True if optimal or feasible solution found, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_solution(self) -> pd.DataFrame:
        """
        Extract solution as a DataFrame.
        
        Returns:
            DataFrame with columns: operation_id, machine_id, start_time, end_time
        """
        pass
    
    def set_operations(self, operations: List[Operation]) -> None:
        """Set operations to be scheduled."""
        self.operations = operations
        self._is_built = False
    
    def set_machines(self, machines: List[Machine]) -> None:
        """Set available machines."""
        self.machines = machines
        self._is_built = False
    
    def set_setup_matrix(self, setup_matrix: Dict[Tuple[str, str], float]) -> None:
        """Set setup time matrix (from_family, to_family) -> minutes."""
        self.setup_matrix = setup_matrix
        self._is_built = False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get solver statistics (for R&D logging)."""
        return self._statistics
    
    def _compute_precedence_graph(self) -> Dict[str, List[str]]:
        """
        Build precedence graph from operations.
        
        Returns dict: operation_id -> list of successor operation_ids
        """
        # Group operations by order
        ops_by_order: Dict[str, List[Operation]] = {}
        for op in self.operations:
            if op.order_id not in ops_by_order:
                ops_by_order[op.order_id] = []
            ops_by_order[op.order_id].append(op)
        
        # Sort by op_seq and build precedence
        successors: Dict[str, List[str]] = {op.id: [] for op in self.operations}
        for order_ops in ops_by_order.values():
            sorted_ops = sorted(order_ops, key=lambda x: x.op_seq)
            for i in range(len(sorted_ops) - 1):
                successors[sorted_ops[i].id].append(sorted_ops[i+1].id)
        
        return successors


# ============================================================
# MILP SCHEDULING MODEL
# ============================================================

class MILPSchedulingModel(SchedulingModel):
    """
    Mixed-Integer Linear Programming model for job-shop scheduling.
    
    Formulation:
    - Binary variables x_{op,machine,slot} = 1 if op starts on machine at slot
    - Continuous variables: start_time[op], end_time[op]
    - Constraints:
      * Each operation assigned to exactly one machine
      * Machine capacity: no overlap on same machine
      * Precedence: op_i finishes before op_j starts (if i precedes j)
      * Setup times between consecutive operations on same machine
      * Machine availability windows
    
    Objective: Configurable (makespan, tardiness, weighted sum)
    
    TODO[R&D]: Compare time-indexed vs big-M formulations:
    - Time-indexed: tighter LP relaxation but larger model
    - Big-M: smaller model but weaker relaxation
    Metrics: LP gap, solution time, memory usage
    
    Reference: Manne (1960) disjunctive formulation
    """
    
    def __init__(self, config: Optional[SchedulingConfig] = None):
        super().__init__(config)
        
        # OR-Tools will be imported lazily
        self._model = None
        self._solver = None
        
        # Decision variables (stored for solution extraction)
        self._start_vars: Dict[str, Any] = {}
        self._end_vars: Dict[str, Any] = {}
        self._assign_vars: Dict[Tuple[str, str], Any] = {}  # (op_id, machine_id)
        self._makespan_var = None
    
    def build_model(self) -> None:
        """
        Build MILP model using OR-Tools or PuLP.
        
        Model structure:
        1. Create variables
        2. Add assignment constraints
        3. Add precedence constraints
        4. Add machine capacity (no-overlap) constraints
        5. Add setup time constraints
        6. Set objective
        """
        try:
            # Try OR-Tools first (preferred)
            from ortools.linear_solver import pywraplp
            self._build_ortools_milp(pywraplp)
        except ImportError:
            logger.warning("OR-Tools not available, falling back to PuLP")
            try:
                import pulp
                self._build_pulp_milp(pulp)
            except ImportError:
                raise ImportError(
                    "No MILP solver available. Install ortools or pulp:\n"
                    "pip install ortools  # recommended\n"
                    "pip install pulp"
                )
        
        self._is_built = True
        
        if self.config.log_model_statistics:
            self._log_model_stats()
    
    def _build_ortools_milp(self, pywraplp) -> None:
        """Build model using OR-Tools MILP solver."""
        # Create solver (CBC is open-source, SCIP also good)
        self._solver = pywraplp.Solver.CreateSolver('CBC')
        if not self._solver:
            self._solver = pywraplp.Solver.CreateSolver('SCIP')
        if not self._solver:
            raise RuntimeError("No MILP solver available in OR-Tools")
        
        solver = self._solver
        infinity = solver.infinity()
        
        n_ops = len(self.operations)
        n_machines = len(self.machines)
        machine_ids = [m.id for m in self.machines]
        
        # Big-M constant (conservative upper bound)
        M = self.config.time_horizon_hours * 60  # in minutes
        
        # ========== VARIABLES ==========
        
        # start_time[op] - continuous, when operation starts
        # end_time[op] - continuous, when operation ends
        for op in self.operations:
            self._start_vars[op.id] = solver.NumVar(0, M, f's_{op.id}')
            self._end_vars[op.id] = solver.NumVar(0, M, f'e_{op.id}')
        
        # x[op, machine] - binary, 1 if op assigned to machine
        for op in self.operations:
            eligible = op.eligible_machines if op.eligible_machines else machine_ids
            for m_id in eligible:
                self._assign_vars[(op.id, m_id)] = solver.BoolVar(f'x_{op.id}_{m_id}')
        
        # makespan - continuous
        self._makespan_var = solver.NumVar(0, M, 'makespan')
        
        # ========== CONSTRAINTS ==========
        
        # 1. Each operation assigned to exactly one eligible machine
        for op in self.operations:
            eligible = op.eligible_machines if op.eligible_machines else machine_ids
            solver.Add(
                sum(self._assign_vars[(op.id, m)] for m in eligible) == 1,
                f'assign_{op.id}'
            )
        
        # 2. end_time = start_time + processing_time (adjusted by machine speed)
        for op in self.operations:
            eligible = op.eligible_machines if op.eligible_machines else machine_ids
            # Linear combination with machine speed factors
            machine_speed = {m.id: m.speed_factor for m in self.machines}
            
            # end[op] >= start[op] + proc_time / speed * x[op,m] for all m
            # Linearized using big-M
            for m_id in eligible:
                m_speed = machine_speed.get(m_id, 1.0)
                proc_time = op.processing_time_min / m_speed
                
                solver.Add(
                    self._end_vars[op.id] >= self._start_vars[op.id] + proc_time 
                    - M * (1 - self._assign_vars[(op.id, m_id)]),
                    f'proc_{op.id}_{m_id}'
                )
        
        # 3. Precedence constraints within same order
        precedence = self._compute_precedence_graph()
        for pred_id, successors in precedence.items():
            for succ_id in successors:
                solver.Add(
                    self._start_vars[succ_id] >= self._end_vars[pred_id],
                    f'prec_{pred_id}_{succ_id}'
                )
        
        # 4. Machine capacity: no overlap (disjunctive constraints)
        # For each pair of operations on same machine
        # TODO[R&D]: Implement lazy constraint generation for large instances
        # Current: direct big-M formulation O(n²) constraints per machine
        
        for m in self.machines:
            ops_on_machine = [
                op for op in self.operations 
                if (op.id, m.id) in self._assign_vars
            ]
            
            for i, op1 in enumerate(ops_on_machine):
                for op2 in ops_on_machine[i+1:]:
                    # Either op1 before op2, or op2 before op1
                    # Introduce auxiliary binary y: y=1 means op1 before op2
                    y = solver.BoolVar(f'seq_{op1.id}_{op2.id}_{m.id}')
                    
                    # If both assigned to this machine and y=1: end[op1] <= start[op2]
                    # If both assigned to this machine and y=0: end[op2] <= start[op1]
                    # Linearized with big-M
                    
                    x1 = self._assign_vars[(op1.id, m.id)]
                    x2 = self._assign_vars[(op2.id, m.id)]
                    
                    # end[op1] <= start[op2] + M*(1-y) + M*(1-x1) + M*(1-x2)
                    solver.Add(
                        self._end_vars[op1.id] <= self._start_vars[op2.id] 
                        + M * (1 - y) + M * (2 - x1 - x2),
                        f'nooverlap1_{op1.id}_{op2.id}_{m.id}'
                    )
                    
                    # end[op2] <= start[op1] + M*y + M*(1-x1) + M*(1-x2)
                    solver.Add(
                        self._end_vars[op2.id] <= self._start_vars[op1.id]
                        + M * y + M * (2 - x1 - x2),
                        f'nooverlap2_{op1.id}_{op2.id}_{m.id}'
                    )
        
        # 5. Makespan constraint
        for op in self.operations:
            solver.Add(
                self._makespan_var >= self._end_vars[op.id],
                f'makespan_{op.id}'
            )
        
        # ========== OBJECTIVE ==========
        self._set_objective_ortools(solver)
        
        logger.info(f"MILP model built: {solver.NumVariables()} vars, {solver.NumConstraints()} constraints")
    
    def _set_objective_ortools(self, solver) -> None:
        """Set objective function based on config."""
        obj_type = self.config.objective_type
        
        if obj_type == ObjectiveType.MINIMIZE_MAKESPAN:
            solver.Minimize(self._makespan_var)
        
        elif obj_type == ObjectiveType.MINIMIZE_WEIGHTED_SUM:
            # Weighted sum of multiple objectives
            obj_expr = (
                self.config.weight_makespan * self._makespan_var
            )
            
            # Add tardiness terms if due dates exist
            for op in self.operations:
                if op.due_date:
                    # Tardiness = max(0, end_time - due_date)
                    # Linearize: tardiness >= end_time - due_date_minutes
                    tardiness = solver.NumVar(0, solver.infinity(), f'tard_{op.id}')
                    due_min = (op.due_date - datetime.now()).total_seconds() / 60
                    solver.Add(tardiness >= self._end_vars[op.id] - due_min)
                    obj_expr += self.config.weight_tardiness * tardiness
            
            solver.Minimize(obj_expr)
        
        else:
            # Default to makespan
            solver.Minimize(self._makespan_var)
    
    def _build_pulp_milp(self, pulp) -> None:
        """Build model using PuLP (fallback solver)."""
        # TODO: Implement PuLP version for environments without OR-Tools
        raise NotImplementedError("PuLP backend not yet implemented")
    
    def solve(self) -> bool:
        """Solve the MILP model."""
        if not self._is_built:
            self.build_model()
        
        if self._solver is None:
            raise RuntimeError("Model not built")
        
        # Set solver parameters
        self._solver.SetTimeLimit(int(self.config.time_limit_sec * 1000))  # ms
        
        # Solve
        logger.info(f"Solving MILP with time limit {self.config.time_limit_sec}s...")
        status = self._solver.Solve()
        
        # Check status
        from ortools.linear_solver import pywraplp
        
        self._statistics = {
            'status': status,
            'status_name': self._status_name(status),
            'objective_value': self._solver.Objective().Value() if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE] else None,
            'wall_time_ms': self._solver.wall_time(),
            'iterations': self._solver.iterations(),
            'nodes': getattr(self._solver, 'nodes', lambda: 0)(),
        }
        
        if status == pywraplp.Solver.OPTIMAL:
            logger.info(f"Optimal solution found! Objective = {self._statistics['objective_value']:.2f}")
            return True
        elif status == pywraplp.Solver.FEASIBLE:
            logger.info(f"Feasible solution found. Objective = {self._statistics['objective_value']:.2f}")
            return True
        else:
            logger.warning(f"No solution found. Status: {self._statistics['status_name']}")
            return False
    
    def _status_name(self, status: int) -> str:
        """Convert OR-Tools status code to name."""
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
    
    def get_solution(self) -> pd.DataFrame:
        """Extract solution as DataFrame."""
        if self._solver is None:
            raise RuntimeError("Model not solved")
        
        records = []
        for op in self.operations:
            start = self._start_vars[op.id].solution_value()
            end = self._end_vars[op.id].solution_value()
            
            # Find assigned machine
            assigned_machine = None
            for m in self.machines:
                if (op.id, m.id) in self._assign_vars:
                    if self._assign_vars[(op.id, m.id)].solution_value() > 0.5:
                        assigned_machine = m.id
                        break
            
            records.append({
                'operation_id': op.id,
                'order_id': op.order_id,
                'article_id': op.article_id,
                'op_seq': op.op_seq,
                'op_code': op.op_code,
                'machine_id': assigned_machine,
                'route_label': op.route_label,
                'start_min': start,
                'end_min': end,
                'duration_min': end - start,
            })
        
        return pd.DataFrame(records)
    
    def _log_model_stats(self) -> None:
        """Log model statistics for R&D tracking."""
        stats = {
            'n_operations': len(self.operations),
            'n_machines': len(self.machines),
            'n_variables': self._solver.NumVariables() if self._solver else 0,
            'n_constraints': self._solver.NumConstraints() if self._solver else 0,
            'model_type': 'MILP',
            'config': {
                'time_limit': self.config.time_limit_sec,
                'gap': self.config.optimality_gap,
            }
        }
        logger.info(f"[R&D] Model statistics: {stats}")


# ============================================================
# CP-SAT SCHEDULING MODEL
# ============================================================

class CPSATSchedulingModel(SchedulingModel):
    """
    Constraint Programming model using OR-Tools CP-SAT solver.
    
    CP-SAT is often better than MILP for:
    - Highly combinatorial scheduling problems
    - Problems with many disjunctive constraints
    - Finding feasible solutions quickly
    
    Formulation:
    - Interval variables for each operation
    - Optional intervals for machine assignment
    - NoOverlap global constraint per machine
    - Cumulative constraint (if parallel capacity > 1)
    
    TODO[R&D]: Compare CP-SAT vs MILP:
    - Solution quality on different instance sizes
    - Computation time
    - Ability to prove optimality
    Metrics: makespan, gap to best bound, solve time
    
    Reference: Google OR-Tools CP-SAT documentation
    """
    
    def __init__(self, config: Optional[SchedulingConfig] = None):
        super().__init__(config)
        
        self._model = None
        self._solver = None
        
        # CP-SAT specific variables
        self._interval_vars: Dict[str, Any] = {}  # op_id -> interval
        self._optional_intervals: Dict[Tuple[str, str], Any] = {}  # (op_id, machine_id)
        self._start_vars: Dict[str, Any] = {}
        self._end_vars: Dict[str, Any] = {}
        self._machine_vars: Dict[str, Any] = {}
        self._makespan_var = None
    
    def build_model(self) -> None:
        """Build CP-SAT model using OR-Tools."""
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            raise ImportError(
                "OR-Tools CP-SAT not available. Install:\n"
                "pip install ortools"
            )
        
        model = cp_model.CpModel()
        self._model = model
        
        horizon = self.config.time_horizon_hours * 60  # minutes
        machine_ids = [m.id for m in self.machines]
        machine_idx = {m.id: i for i, m in enumerate(self.machines)}
        
        # ========== VARIABLES ==========
        
        for op in self.operations:
            duration = int(op.processing_time_min)
            
            # Start, end, interval for each operation
            start = model.NewIntVar(0, horizon, f's_{op.id}')
            end = model.NewIntVar(0, horizon, f'e_{op.id}')
            interval = model.NewIntervalVar(start, duration, end, f'i_{op.id}')
            
            self._start_vars[op.id] = start
            self._end_vars[op.id] = end
            self._interval_vars[op.id] = interval
            
            # Machine assignment variable
            eligible = op.eligible_machines if op.eligible_machines else machine_ids
            eligible_indices = [machine_idx[m] for m in eligible]
            
            if len(eligible_indices) > 1:
                self._machine_vars[op.id] = model.NewIntVarFromDomain(
                    cp_model.Domain.FromValues(eligible_indices),
                    f'm_{op.id}'
                )
            else:
                # Fixed machine
                self._machine_vars[op.id] = model.NewConstant(eligible_indices[0])
            
            # Optional intervals for each eligible machine
            for m_id in eligible:
                m_idx = machine_idx[m_id]
                is_present = model.NewBoolVar(f'pres_{op.id}_{m_id}')
                optional_interval = model.NewOptionalIntervalVar(
                    start, duration, end, is_present, f'oi_{op.id}_{m_id}'
                )
                self._optional_intervals[(op.id, m_id)] = (optional_interval, is_present)
                
                # Link: is_present == (machine_var == m_idx)
                model.Add(self._machine_vars[op.id] == m_idx).OnlyEnforceIf(is_present)
                model.Add(self._machine_vars[op.id] != m_idx).OnlyEnforceIf(is_present.Not())
        
        # Makespan variable
        self._makespan_var = model.NewIntVar(0, horizon, 'makespan')
        
        # ========== CONSTRAINTS ==========
        
        # 1. Each operation assigned to exactly one machine (implicit via optional intervals)
        for op in self.operations:
            eligible = op.eligible_machines if op.eligible_machines else machine_ids
            presence_vars = [self._optional_intervals[(op.id, m)][1] for m in eligible]
            model.AddExactlyOne(presence_vars)
        
        # 2. Precedence constraints
        precedence = self._compute_precedence_graph()
        for pred_id, successors in precedence.items():
            for succ_id in successors:
                model.Add(self._start_vars[succ_id] >= self._end_vars[pred_id])
        
        # 3. No-overlap constraint per machine
        for m in self.machines:
            machine_intervals = [
                self._optional_intervals[(op.id, m.id)][0]
                for op in self.operations
                if (op.id, m.id) in self._optional_intervals
            ]
            if machine_intervals:
                model.AddNoOverlap(machine_intervals)
        
        # 4. Makespan definition
        for op in self.operations:
            model.Add(self._makespan_var >= self._end_vars[op.id])
        
        # ========== OBJECTIVE ==========
        
        if self.config.objective_type == ObjectiveType.MINIMIZE_MAKESPAN:
            model.Minimize(self._makespan_var)
        
        elif self.config.objective_type == ObjectiveType.MINIMIZE_WEIGHTED_SUM:
            # Weighted objective
            objectives = [
                int(self.config.weight_makespan * 100) * self._makespan_var
            ]
            # Add tardiness terms
            for op in self.operations:
                if op.due_date:
                    due_min = int((op.due_date - datetime.now()).total_seconds() / 60)
                    tardiness = model.NewIntVar(0, horizon, f'tard_{op.id}')
                    model.AddMaxEquality(tardiness, [0, self._end_vars[op.id] - due_min])
                    objectives.append(int(self.config.weight_tardiness * 100) * tardiness)
            
            model.Minimize(sum(objectives))
        
        else:
            model.Minimize(self._makespan_var)
        
        self._is_built = True
        logger.info(f"CP-SAT model built: {len(self.operations)} ops, {len(self.machines)} machines")
    
    def solve(self) -> bool:
        """Solve the CP-SAT model."""
        if not self._is_built:
            self.build_model()
        
        from ortools.sat.python import cp_model
        
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.time_limit_sec
        solver.parameters.num_search_workers = self.config.num_workers
        
        logger.info(f"Solving CP-SAT with time limit {self.config.time_limit_sec}s...")
        status = solver.Solve(self._model)
        
        self._solver = solver
        self._statistics = {
            'status': status,
            'status_name': solver.StatusName(status),
            'objective_value': solver.ObjectiveValue() if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else None,
            'best_bound': solver.BestObjectiveBound() if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else None,
            'wall_time': solver.WallTime(),
            'branches': solver.NumBranches(),
            'conflicts': solver.NumConflicts(),
        }
        
        if status == cp_model.OPTIMAL:
            logger.info(f"Optimal solution found! Makespan = {self._statistics['objective_value']}")
            return True
        elif status == cp_model.FEASIBLE:
            logger.info(f"Feasible solution found. Makespan = {self._statistics['objective_value']}")
            return True
        else:
            logger.warning(f"No solution found. Status: {self._statistics['status_name']}")
            return False
    
    def get_solution(self) -> pd.DataFrame:
        """Extract solution as DataFrame."""
        if self._solver is None:
            raise RuntimeError("Model not solved")
        
        machine_ids = [m.id for m in self.machines]
        
        records = []
        for op in self.operations:
            start = self._solver.Value(self._start_vars[op.id])
            end = self._solver.Value(self._end_vars[op.id])
            machine_idx = self._solver.Value(self._machine_vars[op.id])
            machine_id = machine_ids[machine_idx]
            
            records.append({
                'operation_id': op.id,
                'order_id': op.order_id,
                'article_id': op.article_id,
                'op_seq': op.op_seq,
                'op_code': op.op_code,
                'machine_id': machine_id,
                'route_label': op.route_label,
                'start_min': start,
                'end_min': end,
                'duration_min': end - start,
            })
        
        return pd.DataFrame(records)


# ============================================================
# FACTORY FUNCTION
# ============================================================

def create_scheduling_model(
    model_type: ModelType = ModelType.CPSAT,
    config: Optional[SchedulingConfig] = None
) -> SchedulingModel:
    """
    Factory function to create scheduling models.
    
    Args:
        model_type: Type of model (MILP, CPSAT, HEURISTIC)
        config: Model configuration
    
    Returns:
        SchedulingModel instance
    
    TODO[R&D]: Add model selection heuristics:
    - Small instances (< 100 ops): try MILP first
    - Large instances: use CP-SAT or decomposition
    - Very large: heuristic with local search improvement
    """
    if model_type == ModelType.MILP:
        return MILPSchedulingModel(config)
    elif model_type == ModelType.CPSAT:
        return CPSATSchedulingModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")



