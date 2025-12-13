"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — ADVANCED PLANNING ENGINE
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Comprehensive production planning system supporting:
- Conventional independent machine scheduling
- Chained multi-stage flow shop scheduling
- Short-term detailed planning (2 weeks)
- Long-term strategic planning (12 months)
- Setup time optimization
- Operator skill-based allocation
- Multi-resource synchronization
- Buffer/lag time management

Mathematical Foundation:
- MILP/CP-SAT for optimal scheduling
- Heuristics for large-scale problems
- Multi-objective optimization (makespan, tardiness, setups)

R&D Context (SIFIDE):
- WP5: Advanced APS Algorithms
- Hypothesis: Chained planning reduces makespan by 15-25% vs independent scheduling
"""

from .planning_modes import (
    PlanningMode,
    PlanningConfig,
    PlanningResult,
    ChainedPlanningConfig,
    ConventionalPlanningConfig,
    ShortTermPlanningConfig,
    LongTermPlanningConfig,
)

from .chained_scheduler import (
    ChainedScheduler,
    MachineChain,
    ChainBuffer,
    FlowShopResult,
)

from .conventional_scheduler import (
    ConventionalScheduler,
    MachineSchedule,
)

from .setup_optimizer import (
    SetupOptimizer,
    SetupMatrix,
    SequenceResult,
)

from .operator_allocator import (
    OperatorAllocator,
    OperatorSkill,
    AllocationResult,
)

from .capacity_planner import (
    CapacityPlanner,
    CapacityScenario,
    LongTermPlan,
)

from .planning_engine import (
    PlanningEngine,
    execute_planning,
    compare_planning_modes,
)

__all__ = [
    # Modes
    "PlanningMode", "PlanningConfig", "PlanningResult",
    "ChainedPlanningConfig", "ConventionalPlanningConfig",
    "ShortTermPlanningConfig", "LongTermPlanningConfig",
    # Chained
    "ChainedScheduler", "MachineChain", "ChainBuffer", "FlowShopResult",
    # Conventional
    "ConventionalScheduler", "MachineSchedule",
    # Setup
    "SetupOptimizer", "SetupMatrix", "SequenceResult",
    # Operators
    "OperatorAllocator", "OperatorSkill", "AllocationResult",
    # Capacity
    "CapacityPlanner", "CapacityScenario", "LongTermPlan",
    # Engine
    "PlanningEngine", "execute_planning", "compare_planning_modes",
]


