"""
ProdPlan 4.0 - R&D Module
=========================

Módulo de Research & Development para experimentação e validação.

Work Packages:
- WP1: Routing Intelligence (heurísticas, MILP, routing dinâmico)
- WP2: Suggestion Evaluation (avaliação de qualidade de sugestões)
- WP3: Inventory + Capacity Optimization (políticas de inventário)
- WP4: Learning-Based Scheduler (Multi-Armed Bandit, adaptive policies)

Estrutura:
- experiments_core.py: Núcleo de logging e gestão de experiências
- wp1_routing_experiments.py: Experiências de routing
- wp2_suggestions_eval.py: Avaliação de sugestões
- wp3_inventory_capacity.py: Otimização de inventário
- wp4_learning_scheduler.py: Learning scheduler (MAB)
- api.py: Endpoints REST para R&D
"""

from .experiments_core import (
    WorkPackage,
    ExperimentStatus,
    RDExperiment,
    RDExperimentCreate,
    RDExperimentUpdate,
    ExperimentLogger,
    create_experiment,
    get_experiment,
    list_experiments,
    update_experiment_status,
    get_experiments_summary,
    delete_experiment,
)

from .wp1_routing_experiments import (
    WP1RoutingRequest,
    WP1RoutingExperiment,
    WP1PolicyResult,
    run_routing_experiment,
    run_routing_comparison,
    get_routing_strategies,
)

from .wp2_suggestions_eval import (
    SuggestionLabel,
    SuggestionType,
    SuggestionOrigin,
    SuggestionRecord,
    WP2EvaluationRequest,
    WP2EvaluationResult,
    WP2BatchEvaluationRequest,
    WP2BatchEvaluationResult,
    log_suggestion,
    get_suggestion,
    get_all_suggestions,
    evaluate_suggestion,
    evaluate_suggestions_batch,
)

from .wp3_inventory_capacity import (
    DateRange,
    InventoryPolicy,
    InventoryKPIs,
    SchedulingKPIs,
    WP3ScenarioRequest,
    WP3ScenarioResult,
    WP3Experiment,
    WP3ComparisonRequest,
    run_inventory_capacity_scenario,
    compare_inventory_policies,
    get_default_policies,
)

from .wp4_learning_scheduler import (
    BanditType,
    RewardType,
    PolicyStats,
    EpisodeResult,
    WP4RunRequest,
    WP4ExperimentResult,
    BanditScheduler,
    compute_reward,
    run_scheduler_episode,
    run_learning_experiment,
)

from .causal_deep_experiments import (
    CevaeConfig,
    CevaeEstimate,
    CevaeEstimator,
    TarnetEstimator,
    DragonnetEstimator,
    CausalDeepExperiment,
    run_cevae_experiment,
    compare_deep_causal_models,
    get_research_notes,
)

__all__ = [
    # Core
    "WorkPackage",
    "ExperimentStatus",
    "RDExperiment",
    "RDExperimentCreate",
    "RDExperimentUpdate",
    "ExperimentLogger",
    "create_experiment",
    "get_experiment",
    "list_experiments",
    "update_experiment_status",
    "get_experiments_summary",
    "delete_experiment",
    # WP1
    "WP1RoutingRequest",
    "WP1RoutingExperiment",
    "WP1PolicyResult",
    "run_routing_experiment",
    "run_routing_comparison",
    "get_routing_strategies",
    # WP2
    "SuggestionLabel",
    "SuggestionType",
    "SuggestionOrigin",
    "SuggestionRecord",
    "WP2EvaluationRequest",
    "WP2EvaluationResult",
    "WP2BatchEvaluationRequest",
    "WP2BatchEvaluationResult",
    "log_suggestion",
    "get_suggestion",
    "get_all_suggestions",
    "evaluate_suggestion",
    "evaluate_suggestions_batch",
    # WP3
    "DateRange",
    "InventoryPolicy",
    "InventoryKPIs",
    "SchedulingKPIs",
    "WP3ScenarioRequest",
    "WP3ScenarioResult",
    "WP3Experiment",
    "WP3ComparisonRequest",
    "run_inventory_capacity_scenario",
    "compare_inventory_policies",
    "get_default_policies",
    # WP4
    "BanditType",
    "RewardType",
    "PolicyStats",
    "EpisodeResult",
    "WP4RunRequest",
    "WP4ExperimentResult",
    "BanditScheduler",
    "compute_reward",
    "run_scheduler_episode",
    "run_learning_experiment",
    # Causal Deep (R&D)
    "CevaeConfig",
    "CevaeEstimate",
    "CevaeEstimator",
    "TarnetEstimator",
    "DragonnetEstimator",
    "CausalDeepExperiment",
    "run_cevae_experiment",
    "compare_deep_causal_models",
    "get_research_notes",
]
