"""
ProdPlan 4.0 - Solver Interface

Abstract interface for optimization solvers, supporting:
- OR-Tools CP-SAT (default, recommended)
- OR-Tools MILP (CBC, SCIP)
- Gurobi (commercial, optional)
- HiGHS (open-source)

Design Pattern: Strategy + Factory
- Solvers implement common interface
- Factory selects best available solver

R&D / SIFIDE: WP1 - Solver comparison research
Research Question Q1.2: Which solver provides best trade-off between 
                        solution quality and computation time for 
                        industrial scheduling instances?
Metrics: optimality gap, solve time, memory usage.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS AND CONFIG
# ============================================================

class SolverType(Enum):
    """Available solver types."""
    ORTOOLS_CPSAT = "ortools_cpsat"
    ORTOOLS_CBC = "ortools_cbc"
    ORTOOLS_SCIP = "ortools_scip"
    GUROBI = "gurobi"
    HIGHS = "highs"
    HEURISTIC = "heuristic"


class SolverStatus(Enum):
    """Solver solution status."""
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    TIMEOUT = "timeout"
    ERROR = "error"
    NOT_SOLVED = "not_solved"


@dataclass
class SolverConfig:
    """Configuration for solver."""
    solver_type: SolverType = SolverType.ORTOOLS_CPSAT
    
    # Time limits
    time_limit_sec: float = 60.0
    
    # Quality parameters
    optimality_gap: float = 0.05  # 5%
    
    # Parallelism
    num_workers: int = 4
    
    # Memory limits (for large problems)
    memory_limit_mb: Optional[int] = None
    
    # Search strategy
    search_strategy: str = "default"  # "default", "first_solution", "best_solution"
    
    # Warm start from previous solution
    warm_start: bool = False
    warm_start_solution: Optional[pd.DataFrame] = None
    
    # Logging
    verbose: bool = False
    log_search: bool = False
    
    # R&D options
    export_model: bool = False
    export_path: Optional[str] = None
    log_statistics: bool = True


@dataclass
class SolverResult:
    """Result from solver."""
    status: SolverStatus
    objective_value: Optional[float] = None
    best_bound: Optional[float] = None
    gap: Optional[float] = None
    solution_df: Optional[pd.DataFrame] = None
    
    # Timing
    solve_time_sec: float = 0.0
    
    # Statistics (for R&D)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Messages
    message: str = ""
    
    def is_success(self) -> bool:
        """Check if a solution was found."""
        return self.status in [SolverStatus.OPTIMAL, SolverStatus.FEASIBLE]
    
    def compute_gap(self) -> Optional[float]:
        """Compute optimality gap."""
        if self.objective_value is None or self.best_bound is None:
            return None
        if self.best_bound == 0:
            return 0.0 if self.objective_value == 0 else float('inf')
        return abs(self.objective_value - self.best_bound) / abs(self.best_bound)


# ============================================================
# ABSTRACT SOLVER INTERFACE
# ============================================================

class SolverInterface(ABC):
    """
    Abstract interface for optimization solvers.
    
    Implementations handle:
    - Model building from DataBundle
    - Solving with configured parameters
    - Solution extraction
    
    TODO[R&D]: Implement callback interface for:
    - Progress reporting
    - Intermediate solution logging
    - Early termination on quality threshold
    """
    
    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        self._is_initialized = False
    
    @abstractmethod
    def build_model(
        self,
        operations: List[Dict[str, Any]],
        machines: List[Dict[str, Any]],
        setup_matrix: Optional[Dict] = None,
        orders: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Build optimization model from problem data.
        
        Args:
            operations: List of operation dicts
            machines: List of machine dicts
            setup_matrix: Setup time matrix
            orders: Orders DataFrame with due dates
        """
        pass
    
    @abstractmethod
    def solve(self) -> SolverResult:
        """
        Solve the model.
        
        Returns:
            SolverResult with solution and statistics
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get solver name for logging."""
        pass
    
    def set_warm_start(self, solution: pd.DataFrame) -> None:
        """Set warm start solution."""
        self.config.warm_start = True
        self.config.warm_start_solution = solution
    
    def is_available(self) -> bool:
        """Check if solver is available."""
        return True


# ============================================================
# OR-TOOLS CP-SAT SOLVER
# ============================================================

class ORToolsCPSATSolver(SolverInterface):
    """
    OR-Tools CP-SAT solver implementation.
    
    CP-SAT is a constraint programming solver with SAT solver backend.
    Generally excellent for scheduling problems with many disjunctive constraints.
    
    Strengths:
    - Fast on combinatorial problems
    - Good at finding feasible solutions quickly
    - Scales well with parallelism
    
    TODO[R&D]: Experiment with search strategies:
    - Default vs portfolio search
    - Impact of decision variables ordering
    - Symmetry breaking for machine assignment
    """
    
    def __init__(self, config: Optional[SolverConfig] = None):
        super().__init__(config)
        self._model = None
        self._solver = None
        self._variables = {}
    
    def build_model(
        self,
        operations: List[Dict[str, Any]],
        machines: List[Dict[str, Any]],
        setup_matrix: Optional[Dict] = None,
        orders: Optional[pd.DataFrame] = None
    ) -> None:
        """Build CP-SAT model."""
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            raise ImportError("OR-Tools not installed. Run: pip install ortools")
        
        model = cp_model.CpModel()
        self._model = model
        
        # Convert to internal structures
        from scheduling_models import Operation, Machine, CPSATSchedulingModel, SchedulingConfig
        
        ops = []
        for op_dict in operations:
            ops.append(Operation(
                id=op_dict.get('operation_id', op_dict.get('id', '')),
                order_id=op_dict.get('order_id', ''),
                article_id=op_dict.get('article_id', ''),
                op_seq=op_dict.get('op_seq', 0),
                op_code=op_dict.get('op_code', ''),
                processing_time_min=op_dict.get('duration_min', op_dict.get('processing_time_min', 60)),
                setup_family=op_dict.get('setup_family', 'default'),
                eligible_machines=op_dict.get('eligible_machines', [m['id'] for m in machines]),
                primary_machine=op_dict.get('primary_machine_id'),
                route_label=op_dict.get('route_label'),
            ))
        
        machs = []
        for m_dict in machines:
            machs.append(Machine(
                id=m_dict.get('machine_id', m_dict.get('id', '')),
                work_center=m_dict.get('work_center_group', ''),
                speed_factor=m_dict.get('speed_factor', 1.0),
            ))
        
        # Use the scheduling model
        config = SchedulingConfig(
            time_limit_sec=self.config.time_limit_sec,
            optimality_gap=self.config.optimality_gap,
            num_workers=self.config.num_workers,
        )
        
        self._scheduling_model = CPSATSchedulingModel(config)
        self._scheduling_model.set_operations(ops)
        self._scheduling_model.set_machines(machs)
        if setup_matrix:
            self._scheduling_model.set_setup_matrix(setup_matrix)
        
        self._scheduling_model.build_model()
        self._is_initialized = True
        
        logger.info(f"CP-SAT model built: {len(ops)} operations, {len(machs)} machines")
    
    def solve(self) -> SolverResult:
        """Solve using CP-SAT."""
        if not self._is_initialized:
            return SolverResult(
                status=SolverStatus.ERROR,
                message="Model not built"
            )
        
        import time
        start_time = time.time()
        
        success = self._scheduling_model.solve()
        
        solve_time = time.time() - start_time
        stats = self._scheduling_model.get_statistics()
        
        if success:
            solution_df = self._scheduling_model.get_solution()
            
            return SolverResult(
                status=SolverStatus.OPTIMAL if stats.get('status_name') == 'OPTIMAL' else SolverStatus.FEASIBLE,
                objective_value=stats.get('objective_value'),
                best_bound=stats.get('best_bound'),
                solution_df=solution_df,
                solve_time_sec=solve_time,
                statistics=stats,
                message=f"Solution found in {solve_time:.2f}s"
            )
        else:
            return SolverResult(
                status=SolverStatus.INFEASIBLE,
                solve_time_sec=solve_time,
                statistics=stats,
                message=f"No solution found. Status: {stats.get('status_name')}"
            )
    
    def get_name(self) -> str:
        return "OR-Tools CP-SAT"
    
    def is_available(self) -> bool:
        try:
            from ortools.sat.python import cp_model
            return True
        except ImportError:
            return False


# ============================================================
# OR-TOOLS MILP SOLVER
# ============================================================

class ORToolsMILPSolver(SolverInterface):
    """
    OR-Tools MILP solver implementation (CBC or SCIP backend).
    
    MILP can provide:
    - Proven optimal solutions
    - Tight bounds for quality guarantees
    
    Trade-offs:
    - May be slower than CP-SAT for large instances
    - Better at proving optimality
    
    TODO[R&D]: Implement cutting planes for scheduling:
    - Precedence-based cuts
    - Time-window cuts
    - Machine-capacity cuts
    """
    
    def __init__(self, config: Optional[SolverConfig] = None):
        super().__init__(config)
        self._scheduling_model = None
    
    def build_model(
        self,
        operations: List[Dict[str, Any]],
        machines: List[Dict[str, Any]],
        setup_matrix: Optional[Dict] = None,
        orders: Optional[pd.DataFrame] = None
    ) -> None:
        """Build MILP model."""
        from scheduling_models import Operation, Machine, MILPSchedulingModel, SchedulingConfig
        
        ops = []
        for op_dict in operations:
            ops.append(Operation(
                id=op_dict.get('operation_id', op_dict.get('id', '')),
                order_id=op_dict.get('order_id', ''),
                article_id=op_dict.get('article_id', ''),
                op_seq=op_dict.get('op_seq', 0),
                op_code=op_dict.get('op_code', ''),
                processing_time_min=op_dict.get('duration_min', op_dict.get('processing_time_min', 60)),
                setup_family=op_dict.get('setup_family', 'default'),
                eligible_machines=op_dict.get('eligible_machines', [m['id'] for m in machines]),
                primary_machine=op_dict.get('primary_machine_id'),
                route_label=op_dict.get('route_label'),
            ))
        
        machs = []
        for m_dict in machines:
            machs.append(Machine(
                id=m_dict.get('machine_id', m_dict.get('id', '')),
                work_center=m_dict.get('work_center_group', ''),
                speed_factor=m_dict.get('speed_factor', 1.0),
            ))
        
        config = SchedulingConfig(
            time_limit_sec=self.config.time_limit_sec,
            optimality_gap=self.config.optimality_gap,
            num_workers=self.config.num_workers,
        )
        
        self._scheduling_model = MILPSchedulingModel(config)
        self._scheduling_model.set_operations(ops)
        self._scheduling_model.set_machines(machs)
        if setup_matrix:
            self._scheduling_model.set_setup_matrix(setup_matrix)
        
        self._scheduling_model.build_model()
        self._is_initialized = True
        
        logger.info(f"MILP model built: {len(ops)} operations, {len(machs)} machines")
    
    def solve(self) -> SolverResult:
        """Solve using MILP."""
        if not self._is_initialized:
            return SolverResult(
                status=SolverStatus.ERROR,
                message="Model not built"
            )
        
        import time
        start_time = time.time()
        
        success = self._scheduling_model.solve()
        
        solve_time = time.time() - start_time
        stats = self._scheduling_model.get_statistics()
        
        if success:
            solution_df = self._scheduling_model.get_solution()
            
            return SolverResult(
                status=SolverStatus.OPTIMAL if stats.get('status_name') == 'OPTIMAL' else SolverStatus.FEASIBLE,
                objective_value=stats.get('objective_value'),
                solution_df=solution_df,
                solve_time_sec=solve_time,
                statistics=stats,
                message=f"Solution found in {solve_time:.2f}s"
            )
        else:
            return SolverResult(
                status=SolverStatus.INFEASIBLE,
                solve_time_sec=solve_time,
                statistics=stats,
                message=f"No solution found. Status: {stats.get('status_name')}"
            )
    
    def get_name(self) -> str:
        return "OR-Tools MILP (CBC)"
    
    def is_available(self) -> bool:
        try:
            from ortools.linear_solver import pywraplp
            solver = pywraplp.Solver.CreateSolver('CBC')
            return solver is not None
        except ImportError:
            return False


# ============================================================
# HEURISTIC SOLVER (Fallback)
# ============================================================

class HeuristicSolver(SolverInterface):
    """
    Heuristic solver using dispatching rules.
    
    Fast but no optimality guarantee.
    Used as:
    - Fallback when no optimization solver available
    - Warm start generator for MILP/CP-SAT
    - Baseline for R&D comparison
    
    Dispatching rules implemented:
    - SPT (Shortest Processing Time)
    - EDD (Earliest Due Date)
    - FIFO (First In First Out)
    - Priority-based
    
    TODO[R&D]: Implement meta-heuristics:
    - Genetic Algorithm
    - Simulated Annealing
    - Tabu Search
    - GRASP
    """
    
    def __init__(self, config: Optional[SolverConfig] = None):
        super().__init__(config)
        self._operations = []
        self._machines = []
        self._setup_matrix = {}
        self._orders = None
        self._rule = "SPT"  # Default dispatching rule
    
    def set_dispatching_rule(self, rule: str) -> None:
        """Set dispatching rule: SPT, EDD, FIFO, PRIORITY."""
        self._rule = rule.upper()
    
    def build_model(
        self,
        operations: List[Dict[str, Any]],
        machines: List[Dict[str, Any]],
        setup_matrix: Optional[Dict] = None,
        orders: Optional[pd.DataFrame] = None
    ) -> None:
        """Store problem data."""
        self._operations = operations
        self._machines = machines
        self._setup_matrix = setup_matrix or {}
        self._orders = orders
        self._is_initialized = True
    
    def solve(self) -> SolverResult:
        """Solve using dispatching rule heuristic."""
        if not self._is_initialized:
            return SolverResult(status=SolverStatus.ERROR, message="No data")
        
        import time
        start_time = time.time()
        
        # Sort operations by dispatching rule
        sorted_ops = self._sort_operations()
        
        # Track machine availability
        machine_available = {m['machine_id'] if 'machine_id' in m else m['id']: 0.0 for m in self._machines}
        
        # Schedule operations
        schedule = []
        for op in sorted_ops:
            op_id = op.get('operation_id', op.get('id', ''))
            duration = op.get('duration_min', op.get('processing_time_min', 60))
            eligible = op.get('eligible_machines', list(machine_available.keys()))
            
            if not eligible:
                eligible = list(machine_available.keys())
            
            # Choose machine with earliest availability
            best_machine = min(eligible, key=lambda m: machine_available.get(m, 0))
            
            start = machine_available.get(best_machine, 0)
            end = start + duration
            
            schedule.append({
                'operation_id': op_id,
                'order_id': op.get('order_id', ''),
                'article_id': op.get('article_id', ''),
                'op_seq': op.get('op_seq', 0),
                'op_code': op.get('op_code', ''),
                'machine_id': best_machine,
                'route_label': op.get('route_label', 'A'),
                'start_min': start,
                'end_min': end,
                'duration_min': duration,
            })
            
            machine_available[best_machine] = end
        
        solve_time = time.time() - start_time
        solution_df = pd.DataFrame(schedule)
        
        # Compute objective (makespan)
        makespan = solution_df['end_min'].max() if not solution_df.empty else 0
        
        return SolverResult(
            status=SolverStatus.FEASIBLE,
            objective_value=makespan,
            solution_df=solution_df,
            solve_time_sec=solve_time,
            statistics={'rule': self._rule, 'n_ops': len(schedule)},
            message=f"Heuristic ({self._rule}) completed in {solve_time:.3f}s"
        )
    
    def _sort_operations(self) -> List[Dict]:
        """Sort operations according to dispatching rule."""
        ops = list(self._operations)
        
        if self._rule == "SPT":
            # Shortest Processing Time first
            ops.sort(key=lambda x: x.get('duration_min', x.get('processing_time_min', 0)))
        
        elif self._rule == "EDD":
            # Earliest Due Date first
            ops.sort(key=lambda x: x.get('due_date', float('inf')))
        
        elif self._rule == "PRIORITY":
            # Priority first (higher = more urgent)
            ops.sort(key=lambda x: -x.get('priority', 0))
        
        # FIFO: keep original order
        return ops
    
    def get_name(self) -> str:
        return f"Heuristic ({self._rule})"
    
    def is_available(self) -> bool:
        return True  # Always available


# ============================================================
# SOLVER FACTORY
# ============================================================

def get_solver(solver_type: SolverType = SolverType.ORTOOLS_CPSAT, config: Optional[SolverConfig] = None) -> SolverInterface:
    """
    Factory function to get appropriate solver.
    
    Args:
        solver_type: Desired solver type
        config: Solver configuration
    
    Returns:
        SolverInterface instance
    
    Falls back to heuristic if requested solver unavailable.
    """
    if config is None:
        config = SolverConfig(solver_type=solver_type)
    
    if solver_type == SolverType.ORTOOLS_CPSAT:
        solver = ORToolsCPSATSolver(config)
        if solver.is_available():
            return solver
        logger.warning("CP-SAT not available, falling back to MILP")
        solver_type = SolverType.ORTOOLS_CBC
    
    if solver_type in [SolverType.ORTOOLS_CBC, SolverType.ORTOOLS_SCIP]:
        solver = ORToolsMILPSolver(config)
        if solver.is_available():
            return solver
        logger.warning("MILP not available, falling back to heuristic")
        solver_type = SolverType.HEURISTIC
    
    if solver_type == SolverType.GUROBI:
        # TODO: Implement Gurobi interface
        logger.warning("Gurobi not implemented, falling back to heuristic")
        solver_type = SolverType.HEURISTIC
    
    if solver_type == SolverType.HIGHS:
        # TODO: Implement HiGHS interface
        logger.warning("HiGHS not implemented, falling back to heuristic")
        solver_type = SolverType.HEURISTIC
    
    # Default: heuristic
    return HeuristicSolver(config)


def list_available_solvers() -> List[str]:
    """List all available solvers."""
    available = []
    
    for solver_type in SolverType:
        try:
            solver = get_solver(solver_type)
            if solver.is_available():
                available.append(solver.get_name())
        except Exception:
            pass
    
    return available


# ============================================================
# SOLVER COMPARISON (R&D)
# ============================================================

def compare_solvers(
    operations: List[Dict[str, Any]],
    machines: List[Dict[str, Any]],
    solver_types: List[SolverType],
    config: Optional[SolverConfig] = None
) -> pd.DataFrame:
    """
    Compare multiple solvers on the same problem.
    
    Returns DataFrame with:
    - solver_name
    - status
    - objective_value
    - solve_time_sec
    - gap (if available)
    
    TODO[R&D]: Use this for systematic benchmarking:
    - Different instance sizes
    - Different problem structures
    - Statistical analysis of performance
    """
    results = []
    
    for solver_type in solver_types:
        try:
            solver = get_solver(solver_type, config)
            solver.build_model(operations, machines)
            result = solver.solve()
            
            results.append({
                'solver': solver.get_name(),
                'status': result.status.value,
                'objective': result.objective_value,
                'solve_time_sec': result.solve_time_sec,
                'gap': result.compute_gap(),
            })
        except Exception as e:
            results.append({
                'solver': solver_type.value,
                'status': 'error',
                'objective': None,
                'solve_time_sec': None,
                'gap': None,
                'error': str(e),
            })
    
    return pd.DataFrame(results)



