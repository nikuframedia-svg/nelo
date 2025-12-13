"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — DEEP REINFORCEMENT LEARNING SCHEDULER
═══════════════════════════════════════════════════════════════════════════════════════════════════════

R&D Module: WP4 - Learning-Based Scheduling

This module implements a Deep Reinforcement Learning (DRL) approach to production scheduling,
as an alternative to heuristic and MILP-based methods. The DRL agent learns scheduling policies
from experience, potentially discovering novel strategies that outperform hand-crafted rules.

SIFIDE R&D Classification:
- Technical Uncertainty: Can DRL learn competitive scheduling policies for real-world APS problems?
- Scientific Novelty: Application of modern DRL algorithms (PPO, A2C) to multi-resource job shop scheduling
- Experimental Nature: Comparative evaluation against baseline heuristics and MILP

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     DRL Scheduler Module                        │
    ├─────────────────────────────────────────────────────────────────┤
    │  SchedulingEnv (Gymnasium)                                      │
    │    ├─ State: machine status, operation status, time             │
    │    ├─ Action: dispatch operation to machine                     │
    │    ├─ Reward: -tardiness - makespan_factor - setup_penalty      │
    │    └─ Done: all operations scheduled                            │
    ├─────────────────────────────────────────────────────────────────┤
    │  DRL Trainer (Stable-Baselines3)                                │
    │    ├─ PPO (default): robust policy gradient                     │
    │    ├─ A2C: advantage actor-critic                               │
    │    └─ DQN: value-based (discrete actions)                       │
    ├─────────────────────────────────────────────────────────────────┤
    │  Scheduler Interface                                            │
    │    └─ build_plan_drl() → pd.DataFrame                           │
    └─────────────────────────────────────────────────────────────────┘

Mathematical Formulation (MDP):
    
    State Space S:
        s_t = (
            machine_state: [available_time, current_setup, queue_length] for each machine,
            operation_state: [status, remaining_time, due_date_slack] for each operation,
            global_time: t
        )
    
    Action Space A:
        a_t = operation_id ∈ {1, ..., n_ops} | operation is ready and unscheduled
        
    Transition P(s', r | s, a):
        - Dispatch selected operation to its designated machine
        - Update machine availability
        - Advance time if no operation is immediately dispatchable
        
    Reward R(s, a, s'):
        R = -α·tardiness(a) - β·makespan_contribution(a) - γ·setup_cost(a) + δ·flow_bonus(a)
        
        where:
            - tardiness(a) = max(0, completion_time - due_date)
            - makespan_contribution = increase in overall makespan
            - setup_cost = setup time if product family changes
            - flow_bonus = reward for continuous flow without idle

TODO[R&D]: Future enhancements:
    - Multi-objective DRL with Pareto-based rewards
    - State enrichment with ML predictions (lead time, RUL)
    - Attention-based policy networks for variable-size problems
    - Curriculum learning for progressive problem complexity
    - Transfer learning between similar production scenarios

Dependencies:
    - gymnasium: Environment definition
    - stable-baselines3: DRL algorithms
    - numpy: Numerical operations
    - pandas: Data handling

Author: ProdPlan R&D Team
Version: 0.1.0 (Experimental)
"""

# Conditional imports based on available dependencies
try:
    from env_scheduling import (
        SchedulingEnv,
        SchedulingEnvConfig,
        MachineState,
        OperationState,
        EnvState,
    )
    _HAS_ENV = True
except ImportError:
    _HAS_ENV = False
    SchedulingEnv = None
    SchedulingEnvConfig = None
    MachineState = None
    OperationState = None
    EnvState = None

try:
    from drl_trainer import (
        DRLTrainer,
        TrainingConfig,
        TrainingResult,
        train_policy,
        evaluate_policy,
        AlgorithmType,
    )
    _HAS_TRAINER = True
except ImportError:
    _HAS_TRAINER = False
    DRLTrainer = None
    TrainingConfig = None
    TrainingResult = None
    train_policy = None
    evaluate_policy = None
    AlgorithmType = None

try:
    from drl_scheduler_interface import (
        build_plan_drl,
        DRLSchedulerConfig,
        load_trained_policy,
        get_drl_scheduler_info,
    )
    _HAS_INTERFACE = True
except ImportError:
    _HAS_INTERFACE = False
    build_plan_drl = None
    DRLSchedulerConfig = None
    load_trained_policy = None
    get_drl_scheduler_info = None


def is_available() -> bool:
    """Check if DRL scheduling is available (dependencies installed)."""
    return _HAS_ENV and _HAS_TRAINER and _HAS_INTERFACE


def get_missing_dependencies() -> list:
    """Get list of missing dependencies."""
    missing = []
    try:
        import gymnasium
    except ImportError:
        missing.append("gymnasium")
    try:
        import stable_baselines3
    except ImportError:
        missing.append("stable-baselines3")
    return missing

__all__ = [
    # Availability check
    "is_available",
    "get_missing_dependencies",
    # Environment
    "SchedulingEnv",
    "SchedulingEnvConfig",
    "MachineState",
    "OperationState",
    "EnvState",
    # Trainer
    "DRLTrainer",
    "TrainingConfig",
    "TrainingResult",
    "train_policy",
    "evaluate_policy",
    "AlgorithmType",
    # Interface
    "build_plan_drl",
    "DRLSchedulerConfig",
    "load_trained_policy",
    "get_drl_scheduler_info",
]

__version__ = "0.1.0"
__author__ = "ProdPlan R&D Team"

