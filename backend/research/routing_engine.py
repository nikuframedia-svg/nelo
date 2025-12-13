"""
Routing Engine — Dynamic Route Selection per Operation

R&D Module for WP1: APS Core + Routing Intelligence

Research Question (Q1):
    Can we design a scheduling engine that mixes classical APS heuristics
    with dynamic per-operation routing decisions and perform better than
    fixed-route baseline in makespan + stability?

Hypotheses:
    H1.1: Dynamic routing with setup-aware scoring reduces makespan by ≥8%
    H1.2: Multi-criteria scoring (load + setup + due date) improves OTD by ≥5%

Technical Uncertainty:
    - Optimal scoring function weights are unknown
    - Trade-off between local optimization and global makespan
    - Computational cost of dynamic decisions at scale

Usage:
    from backend.research.routing_engine import RoutingEngine, ScoringStrategy
    
    engine = RoutingEngine(strategy=ScoringStrategy.SETUP_AWARE)
    route = engine.select_route(operation, available_routes, context)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import random

import pandas as pd


class ScoringStrategy(Enum):
    """Available routing strategies for experimentation."""
    FIXED_PRIMARY = "fixed_primary"          # Baseline: always use primary route
    SHORTEST_QUEUE = "shortest_queue"        # Route with least queued work
    SETUP_AWARE = "setup_aware"              # Minimize setup time
    LOAD_BALANCED = "load_balanced"          # Balance load across machines
    MULTI_OBJECTIVE = "multi_objective"      # Weighted combination
    ML_PREDICTED = "ml_predicted"            # Use ML model for scoring
    RANDOM = "random"                        # Random selection (control)


@dataclass
class RouteOption:
    """A candidate route for an operation."""
    route_id: str
    route_label: str
    machine_id: str
    base_time_min: float
    setup_family: str
    alt_machines: List[str]
    priority: int = 0  # Lower = higher priority
    
    # Computed at scheduling time
    estimated_setup_min: float = 0.0
    estimated_queue_min: float = 0.0
    estimated_start: Optional[pd.Timestamp] = None
    score: float = 0.0


@dataclass
class RoutingContext:
    """Context information for routing decisions."""
    article_id: str
    op_code: str
    op_seq: int
    qty: int
    due_date: Optional[pd.Timestamp]
    previous_machine: Optional[str]
    previous_setup_family: Optional[str]
    machine_loads: Dict[str, float]  # machine_id -> total_min currently assigned
    machine_last_family: Dict[str, str]  # machine_id -> last setup family
    current_time: pd.Timestamp


class RoutingScorer(ABC):
    """Abstract base class for routing scorers."""
    
    @abstractmethod
    def score(self, route: RouteOption, context: RoutingContext) -> float:
        """
        Compute a score for a route option.
        Lower score = better (will be selected).
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the scoring strategy."""
        pass


class FixedPrimaryScorer(RoutingScorer):
    """Baseline: always prefer primary route (lowest priority number)."""
    
    def score(self, route: RouteOption, context: RoutingContext) -> float:
        return float(route.priority)
    
    @property
    def name(self) -> str:
        return "fixed_primary"


class ShortestQueueScorer(RoutingScorer):
    """Select route with shortest queue (least work assigned)."""
    
    def score(self, route: RouteOption, context: RoutingContext) -> float:
        return context.machine_loads.get(route.machine_id, 0.0)
    
    @property
    def name(self) -> str:
        return "shortest_queue"


class SetupAwareScorer(RoutingScorer):
    """
    Minimize setup time by preferring machines already set up for this family.
    
    TODO[R&D]: Integrate with setup_engine.py for ML-predicted setup times.
    """
    
    def __init__(self, setup_matrix: Optional[Dict[tuple, float]] = None):
        self.setup_matrix = setup_matrix or {}
    
    def score(self, route: RouteOption, context: RoutingContext) -> float:
        # Get last family on this machine
        last_family = context.machine_last_family.get(route.machine_id, "")
        current_family = route.setup_family
        
        # Lookup setup time
        if last_family == current_family:
            setup_time = 0.0
        else:
            setup_time = self.setup_matrix.get(
                (last_family, current_family),
                30.0  # Default setup time if not in matrix
            )
        
        # Score = base time + setup time
        return route.base_time_min + setup_time
    
    @property
    def name(self) -> str:
        return "setup_aware"


class LoadBalancedScorer(RoutingScorer):
    """Balance load across all machines to avoid bottlenecks."""
    
    def score(self, route: RouteOption, context: RoutingContext) -> float:
        current_load = context.machine_loads.get(route.machine_id, 0.0)
        avg_load = sum(context.machine_loads.values()) / max(len(context.machine_loads), 1)
        
        # Penalize machines with above-average load
        load_penalty = max(0, current_load - avg_load)
        
        return route.base_time_min + load_penalty * 0.5
    
    @property
    def name(self) -> str:
        return "load_balanced"


class MultiObjectiveScorer(RoutingScorer):
    """
    Weighted combination of multiple objectives.
    
    TODO[R&D]: Experiment with different weight configurations.
    TODO[R&D]: Implement Pareto-based selection for true multi-objective.
    
    Experiment E1.3: Weight sensitivity analysis
    """
    
    def __init__(
        self,
        w_time: float = 0.4,
        w_setup: float = 0.3,
        w_load: float = 0.2,
        w_due: float = 0.1,
        setup_matrix: Optional[Dict[tuple, float]] = None,
    ):
        self.w_time = w_time
        self.w_setup = w_setup
        self.w_load = w_load
        self.w_due = w_due
        self.setup_matrix = setup_matrix or {}
    
    def score(self, route: RouteOption, context: RoutingContext) -> float:
        # Time component (normalized)
        time_score = route.base_time_min / 60.0  # Normalize to hours
        
        # Setup component
        last_family = context.machine_last_family.get(route.machine_id, "")
        if last_family == route.setup_family:
            setup_score = 0.0
        else:
            setup_score = self.setup_matrix.get(
                (last_family, route.setup_family), 30.0
            ) / 60.0
        
        # Load component
        current_load = context.machine_loads.get(route.machine_id, 0.0)
        max_load = max(context.machine_loads.values()) if context.machine_loads else 1
        load_score = current_load / max(max_load, 1)
        
        # Due date urgency component
        due_score = 0.0
        if context.due_date and route.estimated_start:
            slack = (context.due_date - route.estimated_start).total_seconds() / 3600
            due_score = max(0, -slack)  # Penalize if we'd be late
        
        return (
            self.w_time * time_score +
            self.w_setup * setup_score +
            self.w_load * load_score +
            self.w_due * due_score
        )
    
    @property
    def name(self) -> str:
        return f"multi_objective(t={self.w_time},s={self.w_setup},l={self.w_load},d={self.w_due})"


class MLPredictedScorer(RoutingScorer):
    """
    Use ML model to predict best route.
    
    TODO[R&D]: Train model on historical routing decisions and outcomes.
    TODO[R&D]: Features: operation type, machine state, queue length, time of day, etc.
    
    Experiment E1.4: ML vs heuristic scoring comparison
    """
    
    def __init__(self, model: Optional[Any] = None):
        self.model = model
        # Fallback to multi-objective if no model
        self._fallback = MultiObjectiveScorer()
    
    def score(self, route: RouteOption, context: RoutingContext) -> float:
        if self.model is None:
            # TODO[R&D]: Load trained model here
            return self._fallback.score(route, context)
        
        # TODO[R&D]: Implement feature extraction and prediction
        # features = self._extract_features(route, context)
        # return self.model.predict([features])[0]
        return self._fallback.score(route, context)
    
    @property
    def name(self) -> str:
        return "ml_predicted"


class RandomScorer(RoutingScorer):
    """Random selection (control group for experiments)."""
    
    def score(self, route: RouteOption, context: RoutingContext) -> float:
        return random.random()
    
    @property
    def name(self) -> str:
        return "random"


class RoutingEngine:
    """
    Main routing engine that selects optimal routes for operations.
    
    Supports pluggable scoring strategies for experimentation.
    """
    
    SCORERS: Dict[ScoringStrategy, type] = {
        ScoringStrategy.FIXED_PRIMARY: FixedPrimaryScorer,
        ScoringStrategy.SHORTEST_QUEUE: ShortestQueueScorer,
        ScoringStrategy.SETUP_AWARE: SetupAwareScorer,
        ScoringStrategy.LOAD_BALANCED: LoadBalancedScorer,
        ScoringStrategy.MULTI_OBJECTIVE: MultiObjectiveScorer,
        ScoringStrategy.ML_PREDICTED: MLPredictedScorer,
        ScoringStrategy.RANDOM: RandomScorer,
    }
    
    def __init__(
        self,
        strategy: ScoringStrategy = ScoringStrategy.FIXED_PRIMARY,
        setup_matrix: Optional[Dict[tuple, float]] = None,
        scorer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.strategy = strategy
        self.setup_matrix = setup_matrix or {}
        
        # Initialize scorer
        scorer_class = self.SCORERS[strategy]
        kwargs = scorer_kwargs or {}
        
        if strategy in (ScoringStrategy.SETUP_AWARE, ScoringStrategy.MULTI_OBJECTIVE):
            kwargs.setdefault("setup_matrix", self.setup_matrix)
        
        self.scorer = scorer_class(**kwargs)
        
        # Logging for experiment analysis
        self._decision_log: List[Dict[str, Any]] = []
    
    def select_route(
        self,
        routes: List[RouteOption],
        context: RoutingContext,
        log_decision: bool = True,
    ) -> RouteOption:
        """
        Select the best route from available options.
        
        Args:
            routes: List of candidate routes
            context: Scheduling context
            log_decision: Whether to log this decision for analysis
        
        Returns:
            Selected RouteOption with score computed
        """
        if not routes:
            raise ValueError("No routes available for selection")
        
        # Score all routes
        for route in routes:
            route.score = self.scorer.score(route, context)
        
        # Select best (lowest score)
        best_route = min(routes, key=lambda r: r.score)
        
        # Log decision
        if log_decision:
            self._decision_log.append({
                "article_id": context.article_id,
                "op_code": context.op_code,
                "strategy": self.scorer.name,
                "num_candidates": len(routes),
                "selected_route": best_route.route_id,
                "selected_machine": best_route.machine_id,
                "selected_score": best_route.score,
                "all_scores": {r.route_id: r.score for r in routes},
            })
        
        return best_route
    
    def get_decision_log(self) -> List[Dict[str, Any]]:
        """Return logged decisions for experiment analysis."""
        return self._decision_log
    
    def clear_decision_log(self) -> None:
        """Clear decision log."""
        self._decision_log = []


def build_route_options_from_routing_df(
    routing_df: pd.DataFrame,
    article_id: str,
    op_seq: int,
) -> List[RouteOption]:
    """
    Build RouteOption objects from routing DataFrame.
    
    Args:
        routing_df: DataFrame with routing data
        article_id: Article to get routes for
        op_seq: Operation sequence number
    
    Returns:
        List of RouteOption objects
    """
    mask = (routing_df["article_id"] == article_id) & (routing_df["op_seq"] == op_seq)
    filtered = routing_df[mask]
    
    options = []
    for idx, row in filtered.iterrows():
        options.append(RouteOption(
            route_id=str(row.get("route_id", f"R-{idx}")),
            route_label=str(row.get("route_label", "A")),
            machine_id=str(row.get("primary_machine_id", "")),
            base_time_min=float(row.get("base_time_per_unit_min", 1.0)),
            setup_family=str(row.get("setup_family", "")),
            alt_machines=str(row.get("alt_machine_ids", "")).split(",") if row.get("alt_machine_ids") else [],
            priority=ord(str(row.get("route_label", "A"))) - ord("A"),  # A=0, B=1, C=2
        ))
    
    return options


# ============================================================
# EXPERIMENT SUPPORT
# ============================================================

def run_routing_experiment(
    routing_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    strategy: ScoringStrategy,
    setup_matrix: Optional[Dict[tuple, float]] = None,
) -> Dict[str, Any]:
    """
    Run a routing experiment with given strategy.
    
    TODO[R&D]: This is the entry point for experiment E1.1 and E1.2.
    
    Returns:
        Dict with experiment results (decisions, metrics)
    """
    engine = RoutingEngine(strategy=strategy, setup_matrix=setup_matrix)
    
    # TODO[R&D]: Integrate with full scheduler
    # For now, return placeholder
    return {
        "strategy": strategy.value,
        "num_decisions": 0,
        "decision_log": engine.get_decision_log(),
    }



