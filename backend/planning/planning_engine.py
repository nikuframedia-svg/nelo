"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PLANNING ENGINE — Unified APS Controller
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Main orchestrator for all planning modes:
- Conventional (independent) scheduling
- Chained (flow shop) scheduling
- Short-term detailed planning
- Long-term strategic planning

Provides unified interface for:
- Switching between planning modes
- Comparing results
- Generating recommendations

═══════════════════════════════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import logging
import pandas as pd

from .planning_modes import (
    PlanningMode,
    PlanningConfig,
    PlanningResult,
    PlanningComparison,
    ChainedPlanningConfig,
    ConventionalPlanningConfig,
    ShortTermPlanningConfig,
    LongTermPlanningConfig,
)
from .chained_scheduler import ChainedScheduler, MachineChain, ChainedJob, build_chained_jobs
from .conventional_scheduler import ConventionalScheduler, Operation, build_operations_from_plan
from .setup_optimizer import SetupOptimizer, SetupMatrix
from .operator_allocator import OperatorAllocator, OperatorSkill
from .capacity_planner import CapacityPlanner, CapacityScenario, create_baseline_vs_expansion_scenarios

logger = logging.getLogger(__name__)


class PlanningEngine:
    """
    Unified planning engine for all scheduling modes.
    
    Usage:
        engine = PlanningEngine(data_bundle)
        
        # Conventional planning
        result = engine.plan(mode="conventional")
        
        # Chained planning
        result = engine.plan(
            mode="chained",
            chains=[["M-101", "M-102", "M-103"]],
            buffers={"M-101->M-102": 30, "M-102->M-103": 15}
        )
        
        # Compare modes
        comparison = engine.compare(["conventional", "chained"])
    """
    
    def __init__(
        self,
        orders_df: pd.DataFrame,
        routing_df: pd.DataFrame,
        machines_df: pd.DataFrame,
        setup_matrix_df: Optional[pd.DataFrame] = None,
        operators_df: Optional[pd.DataFrame] = None,
        shifts_df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize planning engine with data.
        
        Args:
            orders_df: Orders with order_id, article_id, qty, due_date, priority
            routing_df: Routing with article_id, op_code, primary_machine_id, base_time_per_unit_min
            machines_df: Machines with machine_id
            setup_matrix_df: Setup times (optional)
            operators_df: Operator skills (optional)
            shifts_df: Shift calendar (optional)
        """
        self.orders_df = orders_df
        self.routing_df = routing_df
        self.machines_df = machines_df
        self.setup_matrix_df = setup_matrix_df
        self.operators_df = operators_df
        self.shifts_df = shifts_df
        
        self.machines = machines_df['machine_id'].tolist()
        
        # Build setup matrix if available
        self.setup_matrix = None
        if setup_matrix_df is not None and not setup_matrix_df.empty:
            self.setup_matrix = SetupMatrix.from_dataframe(setup_matrix_df)
        
        # Cache for results
        self._results_cache: Dict[str, PlanningResult] = {}
    
    def plan(
        self,
        mode: str = "conventional",
        config: Optional[PlanningConfig] = None,
        **kwargs
    ) -> PlanningResult:
        """
        Execute planning with specified mode.
        
        Args:
            mode: Planning mode ("conventional", "chained", "short_term", "long_term")
            config: Configuration object (or use kwargs)
            **kwargs: Additional configuration parameters
            
        Returns:
            PlanningResult with schedule and metrics
        """
        mode_enum = PlanningMode(mode.lower())
        
        if mode_enum == PlanningMode.CONVENTIONAL:
            return self._plan_conventional(config or ConventionalPlanningConfig(**kwargs))
        
        elif mode_enum == PlanningMode.CHAINED:
            return self._plan_chained(config or ChainedPlanningConfig(**kwargs))
        
        elif mode_enum == PlanningMode.SHORT_TERM:
            return self._plan_short_term(config or ShortTermPlanningConfig(**kwargs))
        
        elif mode_enum == PlanningMode.LONG_TERM:
            return self._plan_long_term(config or LongTermPlanningConfig(**kwargs))
        
        else:
            raise ValueError(f"Unknown planning mode: {mode}")
    
    def _plan_conventional(self, config: ConventionalPlanningConfig) -> PlanningResult:
        """Execute conventional independent scheduling."""
        logger.info(f"Planning with CONVENTIONAL mode, rule={config.dispatching_rule}")
        
        start_time = config.horizon_start or datetime.now()
        
        # Build operations from orders + routing
        operations = build_operations_from_plan(self.orders_df, self.routing_df)
        
        # Create scheduler
        scheduler = ConventionalScheduler(config)
        
        # Schedule
        setup_dict = None
        if self.setup_matrix and config.group_by_setup_family:
            setup_dict = self.setup_matrix.matrix
            schedule_df, machine_schedules = scheduler.schedule_with_setup_optimization(
                operations, self.machines, start_time, setup_dict
            )
        else:
            schedule_df, machine_schedules = scheduler.schedule(
                operations, self.machines, start_time
            )
        
        # Compute result metrics
        result = self._build_result(PlanningMode.CONVENTIONAL, config, schedule_df, machine_schedules)
        
        self._results_cache["conventional"] = result
        return result
    
    def _plan_chained(self, config: ChainedPlanningConfig) -> PlanningResult:
        """Execute chained flow shop scheduling."""
        logger.info(f"Planning with CHAINED mode, chains={len(config.chains)}")
        
        start_time = config.horizon_start or datetime.now()
        
        # Create scheduler
        scheduler = ChainedScheduler(config)
        
        all_schedules = []
        all_metrics = {}
        
        for chain in scheduler.chains:
            # Build jobs for this chain
            jobs = build_chained_jobs(self.orders_df, self.routing_df, chain)
            
            if not jobs:
                logger.warning(f"No jobs for chain {chain.chain_id}")
                continue
            
            # Schedule
            result = scheduler.schedule(jobs, chain, start_time)
            
            if not result.schedule.empty:
                all_schedules.append(result.schedule)
            
            all_metrics[chain.chain_id] = {
                "makespan_min": result.makespan_minutes,
                "flow_time_min": result.total_flow_time,
                "tardiness_min": result.total_tardiness,
                "idle_time": result.idle_time_per_machine,
            }
        
        # Combine schedules
        if all_schedules:
            schedule_df = pd.concat(all_schedules, ignore_index=True)
        else:
            schedule_df = pd.DataFrame()
        
        # Build result
        result = self._build_result_chained(config, schedule_df, all_metrics)
        
        self._results_cache["chained"] = result
        return result
    
    def _plan_short_term(self, config: ShortTermPlanningConfig) -> PlanningResult:
        """Execute short-term detailed planning."""
        logger.info(f"Planning with SHORT_TERM mode, horizon={config.horizon_days} days")
        
        # For short-term, use conventional with additional constraints
        conv_config = ConventionalPlanningConfig(
            horizon_start=config.horizon_start or datetime.now(),
            horizon_end=(config.horizon_start or datetime.now()) + timedelta(days=config.horizon_days),
            respect_due_dates=True,
            respect_shifts=config.shift_calendar is not None,
            respect_maintenance=bool(config.maintenance_windows),
            dispatching_rule="EDD",
            solver=config.solver,
        )
        
        result = self._plan_conventional(conv_config)
        result.mode = PlanningMode.SHORT_TERM
        
        # Add operator allocation if required
        if config.require_operators and self.operators_df is not None:
            # TODO: Integrate operator allocation
            result.warnings.append("Alocação de operadores: funcionalidade em desenvolvimento.")
        
        self._results_cache["short_term"] = result
        return result
    
    def _plan_long_term(self, config: LongTermPlanningConfig) -> PlanningResult:
        """Execute long-term strategic planning."""
        logger.info(f"Planning with LONG_TERM mode, horizon={config.horizon_months} months")
        
        # Create capacity planner
        planner = CapacityPlanner(self.machines_df)
        
        # Estimate base demand from orders
        total_order_hours = sum(
            self.routing_df[self.routing_df['article_id'] == row['article_id']]['base_time_per_unit_min'].sum() * row['qty'] / 60
            for _, row in self.orders_df.iterrows()
        )
        annual_demand = total_order_hours * 12  # Extrapolate
        
        # Create scenarios
        scenarios = []
        
        baseline, expansion = create_baseline_vs_expansion_scenarios(
            base_demand=annual_demand,
            growth_rate=config.demand_growth_quarterly,
        )
        
        scenarios.append(baseline)
        scenarios.append(expansion)
        
        # Apply custom capacity changes
        for change in config.capacity_changes:
            from .capacity_planner import CapacityChange
            expansion.capacity_changes.append(CapacityChange(
                change_id=f"custom_{len(expansion.capacity_changes)}",
                effective_date=datetime.fromisoformat(change['date']) if isinstance(change['date'], str) else change['date'],
                change_type=change.get('action', 'add_machine'),
                machine_id=change.get('machine_id', 'M-NEW'),
                capacity_delta_hours_per_day=change.get('capacity_delta', 16.0),
                description=change.get('description', ''),
            ))
        
        # Run capacity planning
        plan = planner.plan(
            scenarios=scenarios,
            horizon_months=config.horizon_months,
            start_date=config.horizon_start or datetime.now(),
        )
        
        # Convert to PlanningResult
        result = PlanningResult(
            mode=PlanningMode.LONG_TERM,
            config=config,
            plan_df=pd.DataFrame(),  # Long-term doesn't have detailed schedule
            solver_status="completed",
        )
        
        # Add capacity planning info to warnings/recommendations
        result.warnings.extend(plan.recommendations)
        
        for inv in plan.investment_decisions:
            result.warnings.append(
                f"Investimento recomendado: {inv['type']} em {inv['recommended_date']} - {inv['reason']}"
            )
        
        self._results_cache["long_term"] = result
        return result
    
    def _build_result(
        self,
        mode: PlanningMode,
        config: PlanningConfig,
        schedule_df: pd.DataFrame,
        machine_schedules: Dict,
    ) -> PlanningResult:
        """Build PlanningResult from schedule data."""
        
        # Compute aggregate metrics
        makespan = 0.0
        total_setup = 0.0
        total_proc = 0.0
        total_tardiness = 0.0
        
        machine_metrics = {}
        
        for machine_id, sched in machine_schedules.items():
            makespan = max(makespan, sched.makespan_min / 60)
            total_setup += sched.total_setup_min / 60
            total_proc += sched.total_processing_min / 60
            
            machine_metrics[machine_id] = {
                "processing_hours": sched.total_processing_min / 60,
                "setup_hours": sched.total_setup_min / 60,
                "idle_hours": sched.total_idle_min / 60,
                "utilization_pct": sched.utilization_pct,
                "makespan_hours": sched.makespan_min / 60,
            }
        
        # Calculate tardiness from schedule
        if not schedule_df.empty and 'end_time' in schedule_df.columns:
            # Group by order and check due dates
            pass  # TODO: Implement tardiness calculation
        
        # Find bottleneck
        bottleneck_machine = None
        bottleneck_util = 0.0
        for m_id, metrics in machine_metrics.items():
            if metrics.get('utilization_pct', 0) > bottleneck_util:
                bottleneck_util = metrics['utilization_pct']
                bottleneck_machine = m_id
        
        # Total throughput (units)
        throughput = float(schedule_df['qty'].sum()) if 'qty' in schedule_df.columns else 0.0
        
        # Average utilization
        avg_util = float(sum(m.get('utilization_pct', 0) for m in machine_metrics.values()) / len(machine_metrics)) if machine_metrics else 0.0
        
        # Convert all metrics to native Python types
        for m_id in machine_metrics:
            machine_metrics[m_id] = {k: float(v) for k, v in machine_metrics[m_id].items()}
        
        return PlanningResult(
            mode=mode,
            config=config,
            plan_df=schedule_df,
            makespan_hours=float(makespan),
            total_tardiness_hours=float(total_tardiness),
            total_setup_hours=float(total_setup),
            throughput_units=float(throughput),
            utilization_pct=float(avg_util),
            machine_metrics=machine_metrics,
            bottleneck_machine=bottleneck_machine,
            bottleneck_utilization=float(bottleneck_util),
            solver_status="completed",
        )
    
    def _build_result_chained(
        self,
        config: ChainedPlanningConfig,
        schedule_df: pd.DataFrame,
        chain_metrics: Dict,
    ) -> PlanningResult:
        """Build PlanningResult for chained planning."""
        
        # Aggregate metrics across chains
        total_makespan = 0.0
        total_tardiness = 0.0
        
        machine_metrics = {}
        
        for chain_id, metrics in chain_metrics.items():
            total_makespan = max(total_makespan, metrics.get('makespan_min', 0) / 60)
            total_tardiness += metrics.get('tardiness_min', 0) / 60
            
            # Aggregate idle time by machine
            for m_id, idle in metrics.get('idle_time', {}).items():
                if m_id not in machine_metrics:
                    machine_metrics[m_id] = {"idle_hours": 0.0}
                machine_metrics[m_id]["idle_hours"] += float(idle) / 60
        
        # Convert all metrics to native Python types
        for m_id in machine_metrics:
            machine_metrics[m_id] = {k: float(v) for k, v in machine_metrics[m_id].items()}
        
        throughput = float(schedule_df['qty'].sum()) if 'qty' in schedule_df.columns else 0.0
        
        return PlanningResult(
            mode=PlanningMode.CHAINED,
            config=config,
            plan_df=schedule_df,
            makespan_hours=total_makespan,
            total_tardiness_hours=total_tardiness,
            total_setup_hours=0,  # Setups are part of flow in chained
            throughput_units=throughput,
            machine_metrics=machine_metrics,
            solver_status="completed",
        )
    
    def compare(
        self,
        modes: List[str] = None,
    ) -> PlanningComparison:
        """
        Compare results between planning modes.
        
        Args:
            modes: List of modes to compare (default: ["conventional", "chained"])
            
        Returns:
            PlanningComparison with delta metrics
        """
        if modes is None:
            modes = ["conventional", "chained"]
        
        # Ensure both modes are planned
        results = []
        for mode in modes:
            if mode not in self._results_cache:
                self.plan(mode=mode)
            results.append(self._results_cache[mode])
        
        if len(results) < 2:
            raise ValueError("Need at least 2 modes to compare")
        
        comparison = PlanningComparison(
            baseline=results[0],
            scenario=results[1],
        )
        comparison.compute_deltas()
        
        return comparison


def execute_planning(
    data_bundle,
    mode: str = "conventional",
    **kwargs
) -> PlanningResult:
    """
    Convenience function to execute planning.
    
    Args:
        data_bundle: DataBundle with orders, routing, machines, etc.
        mode: Planning mode
        **kwargs: Additional configuration
        
    Returns:
        PlanningResult
    """
    engine = PlanningEngine(
        orders_df=data_bundle.orders,
        routing_df=data_bundle.routing,
        machines_df=data_bundle.machines,
        setup_matrix_df=getattr(data_bundle, 'setup_matrix', None),
        operators_df=getattr(data_bundle, 'operators', None),
        shifts_df=getattr(data_bundle, 'shifts', None),
    )
    
    return engine.plan(mode=mode, **kwargs)


def compare_planning_modes(
    data_bundle,
    modes: List[str] = None,
    **kwargs
) -> PlanningComparison:
    """
    Convenience function to compare planning modes.
    
    Args:
        data_bundle: DataBundle with orders, routing, machines, etc.
        modes: List of modes to compare
        **kwargs: Additional configuration
        
    Returns:
        PlanningComparison
    """
    engine = PlanningEngine(
        orders_df=data_bundle.orders,
        routing_df=data_bundle.routing,
        machines_df=data_bundle.machines,
        setup_matrix_df=getattr(data_bundle, 'setup_matrix', None),
    )
    
    return engine.compare(modes=modes)

