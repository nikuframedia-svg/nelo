"""
════════════════════════════════════════════════════════════════════════════════════════════════════
SIMULATION MODULE - Zero Disruption Manufacturing (ZDM)
════════════════════════════════════════════════════════════════════════════════════════════════════

Módulo de simulação preditiva para análise de resiliência do plano de produção.

Features:
- Geração de cenários de falha baseados em RUL e histórico
- Simulação de perturbações e recuperação
- Estratégias de recuperação automática
- Cálculo de resilience score
"""

from .zdm import (
    FailureScenario,
    FailureType,
    generate_failure_scenarios,
    ZDMSimulator,
    SimulationResult,
    ResilienceReport,
    RecoveryStrategy,
    RecoveryAction,
    suggest_best_recovery,
)

__all__ = [
    'FailureScenario',
    'FailureType',
    'generate_failure_scenarios',
    'ZDMSimulator',
    'SimulationResult',
    'ResilienceReport',
    'RecoveryStrategy',
    'RecoveryAction',
    'suggest_best_recovery',
]


