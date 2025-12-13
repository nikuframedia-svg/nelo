"""
ProdPlan 4.0 - Scheduling Module
================================

Módulo central de scheduling com múltiplos engines:
- Heurísticas (FIFO, SPT, EDD, etc.)
- MILP (Mixed-Integer Linear Programming)
- CP-SAT (Constraint Programming)
- DRL (Deep Reinforcement Learning) - stub

Feature flags controlam qual engine está ativo.

API de alto nível:
- HeuristicScheduler.build_schedule(instance) -> result
- solve_milp(instance) -> result
- solve_cpsat(instance) -> result

Todos retornam formato unificado com scheduled_operations, kpis, etc.
"""

# Types
from scheduling.types import (
    SchedulerEngine,
    DispatchRule,
    ObjectiveType,
    Operation,
    Machine,
    SchedulingInstance,
    ScheduledOperation,
    SchedulingKPIs,
    SchedulingResult,
    PlanRequest,
    PlanResponse,
    create_instance_from_dataframes,
)

# Heuristics
from scheduling.heuristics import (
    dispatch_fifo,
    dispatch_spt,
    dispatch_edd,
    dispatch_cr,
    dispatch_wspt,
    HeuristicDispatcher,
    HeuristicScheduler,
    DispatchingRule,
    ReadyOperation,
)

# MILP
from scheduling.milp_models import (
    MILPJobShopModel,
    MILPFlowShopModel,
    solve_milp,
)

# CP-SAT
from scheduling.cpsat_models import (
    CPSATJobShopModel,
    CPSATFlexibleJobShopModel,
    solve_cpsat,
)

# DRL
from scheduling.drl_policy_stub import (
    DRLPolicyStub,
    DRLSchedulerConfig,
)

# Data-driven durations
from scheduling.data_driven_durations import (
    DataDrivenDurations,
    DurationEstimate,
    get_data_driven_durations,
    estimate_operation_duration,
)

__all__ = [
    # Types
    "SchedulerEngine",
    "DispatchRule",
    "ObjectiveType",
    "Operation",
    "Machine",
    "SchedulingInstance",
    "ScheduledOperation",
    "SchedulingKPIs",
    "SchedulingResult",
    "PlanRequest",
    "PlanResponse",
    "create_instance_from_dataframes",
    # Heuristics
    "dispatch_fifo",
    "dispatch_spt",
    "dispatch_edd",
    "dispatch_cr",
    "dispatch_wspt",
    "HeuristicDispatcher",
    "HeuristicScheduler",
    "DispatchingRule",
    "ReadyOperation",
    # MILP
    "MILPJobShopModel",
    "MILPFlowShopModel",
    "solve_milp",
    # CP-SAT
    "CPSATJobShopModel",
    "CPSATFlexibleJobShopModel",
    "solve_cpsat",
    # DRL
    "DRLPolicyStub",
    "DRLSchedulerConfig",
    # Data-driven
    "DataDrivenDurations",
    "DurationEstimate",
    "get_data_driven_durations",
    "estimate_operation_duration",
]

