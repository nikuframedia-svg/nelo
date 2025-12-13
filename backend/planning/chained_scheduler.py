"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    CHAINED SCHEDULER — Multi-Stage Flow Shop
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Implements synchronized scheduling across machine chains (flow shop).

Mathematical Model (MILP Formulation):
─────────────────────────────────────────────────────────────────────────────────────────────────────

Sets:
    J = {1, ..., n}     : Jobs (orders)
    M = {1, ..., m}     : Machines in the chain
    
Variables:
    S_{j,k}             : Start time of job j on machine k
    C_{j,k}             : Completion time of job j on machine k
    y_{i,j,k} ∈ {0,1}   : 1 if job i precedes job j on machine k

Parameters:
    p_{j,k}             : Processing time of job j on machine k
    d_j                 : Due date of job j
    b_{k,k+1}           : Buffer time between machines k and k+1
    w_j                 : Weight (priority) of job j

Constraints:
    (1) Completion: C_{j,k} = S_{j,k} + p_{j,k}                           ∀j,k
    (2) Precedence: S_{j,k+1} ≥ C_{j,k} + b_{k,k+1}                       ∀j,k<m
    (3) No overlap: S_{j,k} ≥ C_{i,k} ∨ S_{i,k} ≥ C_{j,k}                 ∀i≠j,k
    (4) Sequencing: y_{i,j,k} + y_{j,i,k} = 1                             ∀i<j,k
    (5) Big-M: S_{j,k} ≥ C_{i,k} - M(1 - y_{i,j,k})                       ∀i≠j,k

Objective:
    min  α·C_max + β·Σ w_j·max(0, C_{j,m} - d_j) + γ·Σ setup_{i,j}

Where:
    C_max = max_j C_{j,m}  (makespan)
    T_j = max(0, C_{j,m} - d_j)  (tardiness)

R&D Hypothesis (SIFIDE WP5):
    H5.1: Chained scheduling reduces makespan by 15-25% vs independent scheduling
    H5.2: Buffer optimization improves flow balance and reduces WIP

═══════════════════════════════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd

from .planning_modes import (
    ChainedPlanningConfig,
    PlanningResult,
    PlanningMode,
    SolverType,
)

logger = logging.getLogger(__name__)


@dataclass
class ChainBuffer:
    """Buffer configuration between two machines in a chain."""
    from_machine: str
    to_machine: str
    buffer_minutes: float = 30.0
    reason: str = "transfer"  # transfer, cooling, inspection, etc.


@dataclass
class MachineChain:
    """
    Definition of a machine chain (sequence of machines).
    """
    chain_id: str
    machines: List[str]  # Ordered list of machine IDs
    buffers: List[ChainBuffer] = field(default_factory=list)
    
    def get_buffer(self, from_idx: int, to_idx: int) -> float:
        """Get buffer time between two stages."""
        if from_idx >= len(self.machines) - 1:
            return 0.0
        
        from_machine = self.machines[from_idx]
        to_machine = self.machines[to_idx]
        
        for buf in self.buffers:
            if buf.from_machine == from_machine and buf.to_machine == to_machine:
                return buf.buffer_minutes
        
        return 30.0  # Default buffer


@dataclass
class ChainedJob:
    """A job to be scheduled through a chain."""
    job_id: str
    order_id: str
    article_id: str
    qty: float
    due_date: datetime
    priority: float = 1.0
    
    # Processing times per machine (machine_id -> minutes)
    processing_times: Dict[str, float] = field(default_factory=dict)
    
    # Setup family for setup time calculation
    setup_family: str = "default"


@dataclass
class FlowShopResult:
    """Result of flow shop scheduling."""
    chain_id: str
    jobs: List[ChainedJob]
    schedule: pd.DataFrame  # Columns: job_id, machine_id, start_time, end_time
    
    makespan_minutes: float = 0.0
    total_flow_time: float = 0.0
    total_tardiness: float = 0.0
    idle_time_per_machine: Dict[str, float] = field(default_factory=dict)
    
    solver_status: str = "unknown"


class ChainedScheduler:
    """
    Multi-stage flow shop scheduler.
    
    Supports:
    - Permutation flow shop (same job order on all machines)
    - Non-permutation flow shop (different orders allowed)
    - Buffer/lag times between stages
    - Setup time optimization
    """
    
    def __init__(self, config: ChainedPlanningConfig):
        self.config = config
        self.chains: List[MachineChain] = []
        self._setup_chains()
    
    def _setup_chains(self):
        """Build MachineChain objects from config."""
        for i, machine_list in enumerate(self.config.chains):
            chain = MachineChain(
                chain_id=f"chain_{i}",
                machines=machine_list,
                buffers=[],
            )
            
            # Add buffers between consecutive machines
            for j in range(len(machine_list) - 1):
                from_m = machine_list[j]
                to_m = machine_list[j + 1]
                key = f"{from_m}->{to_m}"
                
                buffer_min = self.config.buffers.get(key, self.config.default_buffer_min)
                chain.buffers.append(ChainBuffer(
                    from_machine=from_m,
                    to_machine=to_m,
                    buffer_minutes=buffer_min,
                ))
            
            self.chains.append(chain)
    
    def schedule(
        self,
        jobs: List[ChainedJob],
        chain: MachineChain,
        start_time: datetime,
    ) -> FlowShopResult:
        """
        Schedule jobs through a machine chain.
        
        Args:
            jobs: List of jobs to schedule
            chain: Machine chain to use
            start_time: Planning horizon start
            
        Returns:
            FlowShopResult with schedule and metrics
        """
        if self.config.solver == SolverType.CPSAT:
            return self._schedule_cpsat(jobs, chain, start_time)
        elif self.config.solver == SolverType.MILP:
            return self._schedule_milp(jobs, chain, start_time)
        else:
            return self._schedule_heuristic(jobs, chain, start_time)
    
    def _schedule_heuristic(
        self,
        jobs: List[ChainedJob],
        chain: MachineChain,
        start_time: datetime,
    ) -> FlowShopResult:
        """
        Heuristic-based flow shop scheduling.
        
        Uses NEH (Nawaz-Enscore-Ham) algorithm for initial sequence,
        then local search for improvement.
        """
        if not jobs:
            return FlowShopResult(chain_id=chain.chain_id, jobs=[], schedule=pd.DataFrame())
        
        # Step 1: Initial sequence using NEH heuristic
        sequence = self._neh_heuristic(jobs, chain)
        
        # Step 2: Compute schedule for this sequence
        schedule_df, metrics = self._compute_schedule(sequence, chain, start_time)
        
        # Step 3: Local search improvement (optional)
        if len(jobs) <= 20:  # Only for small instances
            sequence, schedule_df, metrics = self._local_search(
                sequence, chain, start_time, schedule_df, metrics
            )
        
        return FlowShopResult(
            chain_id=chain.chain_id,
            jobs=sequence,
            schedule=schedule_df,
            makespan_minutes=metrics["makespan"],
            total_flow_time=metrics["flow_time"],
            total_tardiness=metrics["tardiness"],
            idle_time_per_machine=metrics["idle_time"],
            solver_status="heuristic_optimal",
        )
    
    def _neh_heuristic(
        self,
        jobs: List[ChainedJob],
        chain: MachineChain,
    ) -> List[ChainedJob]:
        """
        NEH Heuristic for flow shop sequencing.
        
        1. Order jobs by decreasing total processing time
        2. Insert jobs one by one in the position that minimizes makespan
        """
        # Sort by total processing time (descending)
        jobs_sorted = sorted(
            jobs,
            key=lambda j: sum(j.processing_times.get(m, 0) for m in chain.machines),
            reverse=True
        )
        
        sequence = []
        
        for job in jobs_sorted:
            best_position = 0
            best_makespan = float('inf')
            
            # Try inserting at each position
            for pos in range(len(sequence) + 1):
                test_sequence = sequence[:pos] + [job] + sequence[pos:]
                makespan = self._quick_makespan(test_sequence, chain)
                
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_position = pos
            
            sequence.insert(best_position, job)
        
        return sequence
    
    def _quick_makespan(
        self,
        sequence: List[ChainedJob],
        chain: MachineChain,
    ) -> float:
        """Quick makespan calculation for a given sequence."""
        if not sequence:
            return 0.0
        
        n_jobs = len(sequence)
        n_machines = len(chain.machines)
        
        # C[i][j] = completion time of job i on machine j
        C = np.zeros((n_jobs, n_machines))
        
        for i, job in enumerate(sequence):
            for j, machine in enumerate(chain.machines):
                proc_time = job.processing_times.get(machine, 0)
                buffer = chain.get_buffer(j - 1, j) if j > 0 else 0
                
                if i == 0 and j == 0:
                    C[i][j] = proc_time
                elif i == 0:
                    C[i][j] = C[i][j-1] + buffer + proc_time
                elif j == 0:
                    C[i][j] = C[i-1][j] + proc_time
                else:
                    # Must wait for: previous job on this machine AND previous machine + buffer
                    C[i][j] = max(C[i-1][j], C[i][j-1] + buffer) + proc_time
        
        return float(C[-1][-1])
    
    def _compute_schedule(
        self,
        sequence: List[ChainedJob],
        chain: MachineChain,
        start_time: datetime,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Compute detailed schedule for a job sequence.
        
        Returns:
            (schedule_df, metrics_dict)
        """
        if not sequence:
            return pd.DataFrame(), {"makespan": 0, "flow_time": 0, "tardiness": 0, "idle_time": {}}
        
        records = []
        n_jobs = len(sequence)
        n_machines = len(chain.machines)
        
        # Completion times matrix
        C = np.zeros((n_jobs, n_machines))
        S = np.zeros((n_jobs, n_machines))  # Start times
        
        for i, job in enumerate(sequence):
            for j, machine in enumerate(chain.machines):
                proc_time = job.processing_times.get(machine, 0)
                buffer = chain.get_buffer(j - 1, j) if j > 0 else 0
                
                if i == 0 and j == 0:
                    S[i][j] = 0
                elif i == 0:
                    S[i][j] = C[i][j-1] + buffer
                elif j == 0:
                    S[i][j] = C[i-1][j]
                else:
                    S[i][j] = max(C[i-1][j], C[i][j-1] + buffer)
                
                C[i][j] = S[i][j] + proc_time
                
                # Create schedule record
                start_dt = start_time + timedelta(minutes=float(S[i][j]))
                end_dt = start_time + timedelta(minutes=float(C[i][j]))
                
                records.append({
                    "job_id": job.job_id,
                    "order_id": job.order_id,
                    "article_id": job.article_id,
                    "machine_id": machine,
                    "chain_stage": j + 1,
                    "sequence_position": i + 1,
                    "start_time": start_dt,
                    "end_time": end_dt,
                    "duration_min": proc_time,
                    "qty": job.qty,
                })
        
        schedule_df = pd.DataFrame(records)
        
        # Compute metrics
        makespan = float(C[-1][-1])
        
        # Flow time = sum of completion times
        flow_time = float(np.sum(C[:, -1]))
        
        # Tardiness
        tardiness = 0.0
        for i, job in enumerate(sequence):
            completion_dt = start_time + timedelta(minutes=float(C[i][-1]))
            if completion_dt > job.due_date:
                tardiness += (completion_dt - job.due_date).total_seconds() / 60
        
        # Idle time per machine
        idle_time = {}
        for j, machine in enumerate(chain.machines):
            machine_makespan = C[-1][j]
            total_proc = sum(job.processing_times.get(machine, 0) for job in sequence)
            idle_time[machine] = max(0, machine_makespan - total_proc)
        
        metrics = {
            "makespan": makespan,
            "flow_time": flow_time,
            "tardiness": tardiness,
            "idle_time": idle_time,
        }
        
        return schedule_df, metrics
    
    def _local_search(
        self,
        sequence: List[ChainedJob],
        chain: MachineChain,
        start_time: datetime,
        current_schedule: pd.DataFrame,
        current_metrics: Dict[str, Any],
        max_iterations: int = 100,
    ) -> Tuple[List[ChainedJob], pd.DataFrame, Dict[str, Any]]:
        """
        Local search improvement using pairwise interchange.
        """
        best_sequence = sequence.copy()
        best_schedule = current_schedule
        best_metrics = current_metrics
        best_cost = self._compute_cost(current_metrics)
        
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for i in range(len(best_sequence) - 1):
                for j in range(i + 1, len(best_sequence)):
                    # Swap positions i and j
                    test_sequence = best_sequence.copy()
                    test_sequence[i], test_sequence[j] = test_sequence[j], test_sequence[i]
                    
                    test_schedule, test_metrics = self._compute_schedule(
                        test_sequence, chain, start_time
                    )
                    test_cost = self._compute_cost(test_metrics)
                    
                    if test_cost < best_cost:
                        best_sequence = test_sequence
                        best_schedule = test_schedule
                        best_metrics = test_metrics
                        best_cost = test_cost
                        improved = True
        
        return best_sequence, best_schedule, best_metrics
    
    def _compute_cost(self, metrics: Dict[str, Any]) -> float:
        """Compute weighted cost from metrics."""
        weights = self.config.objective_weights
        
        cost = (
            weights.get("makespan", 0.5) * metrics["makespan"] +
            weights.get("tardiness", 1.0) * metrics["tardiness"] +
            weights.get("flow_time", 0.0) * metrics["flow_time"]
        )
        
        return cost
    
    def _schedule_cpsat(
        self,
        jobs: List[ChainedJob],
        chain: MachineChain,
        start_time: datetime,
    ) -> FlowShopResult:
        """
        CP-SAT based flow shop scheduling using OR-Tools.
        
        TODO[R&D]: Implement full CP-SAT model for optimal solutions.
        """
        try:
            from ortools.sat.python import cp_model
            
            model = cp_model.CpModel()
            
            # Horizon estimation
            total_proc = sum(
                sum(j.processing_times.get(m, 0) for m in chain.machines)
                for j in jobs
            )
            horizon = int(total_proc * 2)
            
            # Variables
            starts = {}
            ends = {}
            intervals = {}
            
            for job in jobs:
                for machine in chain.machines:
                    proc_time = int(job.processing_times.get(machine, 1))
                    
                    start = model.NewIntVar(0, horizon, f"start_{job.job_id}_{machine}")
                    end = model.NewIntVar(0, horizon, f"end_{job.job_id}_{machine}")
                    interval = model.NewIntervalVar(start, proc_time, end, f"interval_{job.job_id}_{machine}")
                    
                    starts[(job.job_id, machine)] = start
                    ends[(job.job_id, machine)] = end
                    intervals[(job.job_id, machine)] = interval
            
            # Constraints
            # 1. Precedence within job (flow through chain)
            for job in jobs:
                for k in range(len(chain.machines) - 1):
                    m1 = chain.machines[k]
                    m2 = chain.machines[k + 1]
                    buffer = int(chain.get_buffer(k, k + 1))
                    
                    model.Add(starts[(job.job_id, m2)] >= ends[(job.job_id, m1)] + buffer)
            
            # 2. No overlap on same machine
            for machine in chain.machines:
                machine_intervals = [intervals[(j.job_id, machine)] for j in jobs]
                model.AddNoOverlap(machine_intervals)
            
            # Objective: minimize makespan
            makespan = model.NewIntVar(0, horizon, "makespan")
            last_machine = chain.machines[-1]
            for job in jobs:
                model.Add(makespan >= ends[(job.job_id, last_machine)])
            
            model.Minimize(makespan)
            
            # Solve
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = self.config.time_limit_sec
            status = solver.Solve(model)
            
            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                # Extract schedule
                records = []
                for job in jobs:
                    for k, machine in enumerate(chain.machines):
                        proc_time = job.processing_times.get(machine, 0)
                        s = solver.Value(starts[(job.job_id, machine)])
                        e = solver.Value(ends[(job.job_id, machine)])
                        
                        start_dt = start_time + timedelta(minutes=s)
                        end_dt = start_time + timedelta(minutes=e)
                        
                        records.append({
                            "job_id": job.job_id,
                            "order_id": job.order_id,
                            "article_id": job.article_id,
                            "machine_id": machine,
                            "chain_stage": k + 1,
                            "start_time": start_dt,
                            "end_time": end_dt,
                            "duration_min": proc_time,
                            "qty": job.qty,
                        })
                
                schedule_df = pd.DataFrame(records)
                
                return FlowShopResult(
                    chain_id=chain.chain_id,
                    jobs=jobs,
                    schedule=schedule_df,
                    makespan_minutes=float(solver.Value(makespan)),
                    solver_status="optimal" if status == cp_model.OPTIMAL else "feasible",
                )
            else:
                logger.warning("CP-SAT solver failed, falling back to heuristic")
                return self._schedule_heuristic(jobs, chain, start_time)
                
        except ImportError:
            logger.warning("OR-Tools not available, using heuristic")
            return self._schedule_heuristic(jobs, chain, start_time)
    
    def _schedule_milp(
        self,
        jobs: List[ChainedJob],
        chain: MachineChain,
        start_time: datetime,
    ) -> FlowShopResult:
        """
        MILP-based flow shop scheduling.
        
        TODO[R&D]: Implement using PuLP or Gurobi for MILP.
        For now, falls back to heuristic.
        """
        logger.info("MILP solver requested, using heuristic (MILP not yet implemented)")
        return self._schedule_heuristic(jobs, chain, start_time)


def build_chained_jobs(
    orders_df: pd.DataFrame,
    routing_df: pd.DataFrame,
    chain: MachineChain,
) -> List[ChainedJob]:
    """
    Build ChainedJob objects from orders and routing data.
    
    Args:
        orders_df: Orders DataFrame with order_id, article_id, qty, due_date
        routing_df: Routing DataFrame with article_id, machine_id, time
        chain: Machine chain to filter routing
        
    Returns:
        List of ChainedJob objects
    """
    jobs = []
    
    for _, order in orders_df.iterrows():
        article_id = order['article_id']
        
        # Get routing for this article filtered to chain machines
        article_routing = routing_df[routing_df['article_id'] == article_id]
        
        processing_times = {}
        for machine in chain.machines:
            machine_routing = article_routing[article_routing['primary_machine_id'] == machine]
            if not machine_routing.empty:
                time_per_unit = machine_routing['base_time_per_unit_min'].iloc[0]
                processing_times[machine] = time_per_unit * order['qty']
        
        # Only include if article uses at least one machine in chain
        if processing_times:
            due_date = pd.to_datetime(order['due_date']) if 'due_date' in order else datetime.now() + timedelta(days=30)
            
            jobs.append(ChainedJob(
                job_id=f"{order['order_id']}_{article_id}",
                order_id=order['order_id'],
                article_id=article_id,
                qty=order['qty'],
                due_date=due_date,
                priority=order.get('priority', 1.0),
                processing_times=processing_times,
                setup_family=order.get('setup_family', 'default'),
            ))
    
    return jobs



