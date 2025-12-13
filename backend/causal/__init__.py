"""
════════════════════════════════════════════════════════════════════════════════════════════════════
CAUSAL CONTEXT MODELS (CCM) - Módulo de Análise Causal
════════════════════════════════════════════════════════════════════════════════════════════════════

Integra análise causal para compreender o impacto de decisões de curto prazo em objetivos de longo prazo.

Decisões (X):
- Policy de scheduling, sequência, carga, setups, turnos

Outcomes (Y):
- Makespan, tardiness, energia, desgaste, acidentes, stress operadores, OTD

Contexto (Z):
- Sazonalidade, procura, mix de produto

Features:
- Aprendizagem de grafos causais (DAG)
- Estimação de efeitos causais (ATE, CATE)
- Explicações em linguagem natural
- Dashboard de complexidade e trade-offs
"""

from .causal_graph_builder import (
    CausalVariable,
    VariableType,
    CausalRelation,
    CausalGraph,
    CausalGraphBuilder,
    learn_causal_graph,
)

from .causal_effect_estimator import (
    CausalEffect,
    EffectType,
    CausalEffectEstimator,
    estimate_effect,
    estimate_intervention,
    get_all_effects_for_outcome,
    get_all_effects_from_treatment,
)

from .complexity_dashboard_engine import (
    ComplexityMetrics,
    CausalInsight,
    InsightType,
    ComplexityDashboard,
    compute_complexity_metrics,
    generate_causal_insights,
    generate_tradeoff_analysis,
)

__all__ = [
    # Graph Builder
    'CausalVariable',
    'VariableType',
    'CausalRelation',
    'CausalGraph',
    'CausalGraphBuilder',
    'learn_causal_graph',
    # Effect Estimator
    'CausalEffect',
    'EffectType',
    'CausalEffectEstimator',
    'estimate_effect',
    'estimate_intervention',
    'get_all_effects_for_outcome',
    'get_all_effects_from_treatment',
    # Complexity Dashboard
    'ComplexityMetrics',
    'CausalInsight',
    'InsightType',
    'ComplexityDashboard',
    'compute_complexity_metrics',
    'generate_causal_insights',
    'generate_tradeoff_analysis',
]

