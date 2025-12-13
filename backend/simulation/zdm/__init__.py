"""
════════════════════════════════════════════════════════════════════════════════════════════════════
ZDM - Zero Disruption Manufacturing Simulation
════════════════════════════════════════════════════════════════════════════════════════════════════

Submódulo para simulação de cenários de falha e análise de resiliência.
"""

from .failure_scenario_generator import (
    FailureScenario,
    FailureType,
    FailureConfig,
    generate_failure_scenarios,
    generate_single_failure,
)

from .zdm_simulator import (
    ZDMSimulator,
    SimulationResult,
    ResilienceReport,
    SimulationConfig,
    ImpactMetrics,
)

from .zdm_simulator import (
    quick_resilience_check,
)

from .recovery_strategy_engine import (
    RecoveryStrategy,
    RecoveryAction,
    RecoveryPlan,
    RecoveryConfig,
    suggest_best_recovery,
    apply_recovery_strategy,
    get_recovery_recommendations,
)

__all__ = [
    # Failure Generation
    'FailureScenario',
    'FailureType',
    'FailureConfig',
    'generate_failure_scenarios',
    'generate_single_failure',
    # Simulation
    'ZDMSimulator',
    'SimulationResult',
    'ResilienceReport',
    'SimulationConfig',
    'ImpactMetrics',
    'quick_resilience_check',
    # Recovery
    'RecoveryStrategy',
    'RecoveryAction',
    'RecoveryPlan',
    'RecoveryConfig',
    'suggest_best_recovery',
    'apply_recovery_strategy',
    'get_recovery_recommendations',
]

