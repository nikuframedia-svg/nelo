"""
ProdPlan 4.0 Research Modules

This package contains experimental modules for the SIFIDE R&D programme.
Each module addresses specific research questions with pluggable algorithms
and comprehensive experiment logging.

Work Packages:
- WP1: APS Core + Routing Intelligence (routing_engine, setup_engine)
- WP2: What-If + Explainable AI (explainability_engine)
- WP3: Inventory-Production Coupling (inventory_optimization)
- WP4: Learning Scheduler (learning_scheduler, experiment_logger)

Research Questions:
- Q1: Can hybrid heuristic+ML scheduling outperform fixed-route APS?
- Q2: Can we generate credible, useful industrial suggestions?
- Q3: Can coupled inventory+production optimization reduce risk?
- Q4: Can we build an explainable AI co-pilot for factories?
"""

from experiment_logger import ExperimentLogger, log_experiment, load_experiment_logs
from routing_engine import RoutingEngine, ScoringStrategy, RouteOption, RoutingContext
from setup_engine import SetupEngine, SetupPredictor, SetupPrediction
from explainability_engine import ExplainabilityEngine, Explanation, AudienceLevel
from learning_scheduler import LearningScheduler, PolicyType
from inventory_optimization import InventoryOptimizer, OptimizationMode, SKUProfile, InventoryPolicy

__all__ = [
    # Experiment Infrastructure
    "ExperimentLogger",
    "log_experiment",
    "load_experiment_logs",
    
    # Routing Engine (WP1)
    "RoutingEngine",
    "ScoringStrategy",
    "RouteOption",
    "RoutingContext",
    
    # Setup Engine (WP1)
    "SetupEngine",
    "SetupPredictor",
    "SetupPrediction",
    
    # Explainability Engine (WP2)
    "ExplainabilityEngine",
    "Explanation",
    "AudienceLevel",
    
    # Learning Scheduler (WP4)
    "LearningScheduler",
    "PolicyType",
    
    # Inventory Optimization (WP3)
    "InventoryOptimizer",
    "OptimizationMode",
    "SKUProfile",
    "InventoryPolicy",
]

