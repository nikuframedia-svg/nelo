"""
ProdPlan 4.0 - Objective Functions for Optimization

This module defines objective functions for production scheduling:
- Makespan minimization
- Total tardiness minimization
- Setup time minimization
- Machine load balancing

Supports:
- Single-objective optimization
- Weighted sum multi-objective
- ε-constraint method (for Pareto-optimal solutions)
- Lexicographic optimization

R&D / SIFIDE: WP1 - Multi-objective optimization research
Research Question Q1.3: Can multi-objective optimization improve OTD 
                        without significantly increasing makespan?
Metrics: Pareto front coverage, hypervolume indicator.

References:
- Deb (2001). Multi-Objective Optimization using Evolutionary Algorithms
- Ehrgott (2005). Multicriteria Optimization
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# OBJECTIVE FUNCTION COMPUTATIONS
# ============================================================

def compute_makespan(plan_df: pd.DataFrame) -> float:
    """
    Compute makespan: time from first operation start to last operation end.
    
    Args:
        plan_df: DataFrame with columns [start_time/start_min, end_time/end_min]
    
    Returns:
        Makespan in minutes
    
    Note: Lower is better. Key metric for production efficiency.
    """
    if plan_df.empty:
        return 0.0
    
    # Support both datetime and numeric columns
    if 'end_time' in plan_df.columns and pd.api.types.is_datetime64_any_dtype(plan_df['end_time']):
        start = plan_df['start_time'].min()
        end = plan_df['end_time'].max()
        return (end - start).total_seconds() / 60
    
    # Numeric minutes
    start_col = 'start_min' if 'start_min' in plan_df.columns else 'start_time'
    end_col = 'end_min' if 'end_min' in plan_df.columns else 'end_time'
    
    return float(plan_df[end_col].max() - plan_df[start_col].min())


def compute_total_tardiness(
    plan_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    due_date_col: str = 'due_date'
) -> float:
    """
    Compute total tardiness: sum of max(0, completion - due_date) for each order.
    
    Args:
        plan_df: DataFrame with operation schedule
        orders_df: DataFrame with order due dates
    
    Returns:
        Total tardiness in minutes
    
    Note: Lower is better. Critical for OTD (On-Time Delivery).
    
    TODO[R&D]: Analyze tardiness distribution across orders:
    - Mean vs max tardiness
    - Tardiness by priority class
    - Early detection of high-risk orders
    """
    if plan_df.empty or orders_df.empty:
        return 0.0
    
    if due_date_col not in orders_df.columns:
        return 0.0
    
    # Get completion time per order
    order_completion = plan_df.groupby('order_id').agg(
        completion_time=('end_time' if 'end_time' in plan_df.columns else 'end_min', 'max')
    ).reset_index()
    
    # Merge with due dates
    merged = order_completion.merge(
        orders_df[['order_id', due_date_col]],
        on='order_id',
        how='left'
    )
    
    total_tardiness = 0.0
    for _, row in merged.iterrows():
        if pd.isna(row[due_date_col]):
            continue
        
        completion = row['completion_time']
        due = row[due_date_col]
        
        # Handle different types
        if isinstance(completion, datetime) and isinstance(due, datetime):
            tardiness = max(0, (completion - due).total_seconds() / 60)
        else:
            # Assume numeric (minutes)
            tardiness = max(0, float(completion) - float(due))
        
        total_tardiness += tardiness
    
    return total_tardiness


def compute_total_setup_time(
    plan_df: pd.DataFrame,
    setup_matrix: Optional[Dict[Tuple[str, str], float]] = None
) -> float:
    """
    Compute total setup time across all machines.
    
    Setup time occurs when consecutive operations on the same machine
    belong to different setup families.
    
    Args:
        plan_df: DataFrame with columns [machine_id, setup_family, start_time]
        setup_matrix: Dict (from_family, to_family) -> setup_time_minutes
    
    Returns:
        Total setup time in minutes
    
    TODO[R&D]: Setup optimization research:
    - Sequence-dependent setup times
    - Setup avoidance through batch scheduling
    - Learning effects in setups
    """
    if plan_df.empty:
        return 0.0
    
    if setup_matrix is None:
        # Default setup time
        default_setup = 15  # minutes
    else:
        default_setup = 10
    
    total_setup = 0.0
    
    # Sort by machine and start time
    sorted_df = plan_df.sort_values(['machine_id', 'start_time' if 'start_time' in plan_df.columns else 'start_min'])
    
    # Group by machine
    for machine_id, machine_ops in sorted_df.groupby('machine_id'):
        ops_list = machine_ops.to_dict('records')
        
        for i in range(1, len(ops_list)):
            prev_family = ops_list[i-1].get('setup_family', 'default')
            curr_family = ops_list[i].get('setup_family', 'default')
            
            if prev_family != curr_family:
                if setup_matrix and (prev_family, curr_family) in setup_matrix:
                    total_setup += setup_matrix[(prev_family, curr_family)]
                else:
                    total_setup += default_setup
    
    return total_setup


def compute_load_imbalance(plan_df: pd.DataFrame) -> float:
    """
    Compute machine load imbalance: standard deviation of machine utilization.
    
    Args:
        plan_df: DataFrame with [machine_id, duration_min or equivalent]
    
    Returns:
        Load imbalance score (std dev of utilization percentages)
    
    Note: Lower is better. Balanced loads reduce bottlenecks.
    
    TODO[R&D]: Consider weighted load balancing:
    - Weight by machine cost/value
    - Account for maintenance schedules
    - Energy consumption optimization
    """
    if plan_df.empty:
        return 0.0
    
    # Compute load per machine
    duration_col = 'duration_min' if 'duration_min' in plan_df.columns else None
    
    if duration_col is None:
        # Try to compute from start/end
        if 'start_time' in plan_df.columns and 'end_time' in plan_df.columns:
            if pd.api.types.is_datetime64_any_dtype(plan_df['end_time']):
                plan_df = plan_df.copy()
                plan_df['duration_min'] = (plan_df['end_time'] - plan_df['start_time']).dt.total_seconds() / 60
            else:
                plan_df = plan_df.copy()
                plan_df['duration_min'] = plan_df['end_time'] - plan_df['start_time']
            duration_col = 'duration_min'
        else:
            return 0.0
    
    machine_loads = plan_df.groupby('machine_id')[duration_col].sum()
    
    if len(machine_loads) < 2:
        return 0.0
    
    return float(machine_loads.std())


def compute_weighted_flowtime(plan_df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> float:
    """
    Compute weighted flow time (sum of weighted completion times).
    
    Args:
        plan_df: DataFrame with [order_id, end_time]
        weights: Dict order_id -> weight (priority)
    
    Returns:
        Weighted flow time
    
    Note: Useful for prioritized scheduling (VIP orders).
    """
    if plan_df.empty:
        return 0.0
    
    # Get completion per order
    order_completion = plan_df.groupby('order_id').agg(
        completion=('end_time' if 'end_time' in plan_df.columns else 'end_min', 'max')
    ).reset_index()
    
    total = 0.0
    for _, row in order_completion.iterrows():
        weight = weights.get(row['order_id'], 1.0) if weights else 1.0
        total += weight * float(row['completion'])
    
    return total


def compute_machine_utilization(plan_df: pd.DataFrame, horizon_minutes: float) -> Dict[str, float]:
    """
    Compute utilization percentage per machine.
    
    Args:
        plan_df: DataFrame with [machine_id, duration_min]
        horizon_minutes: Total time horizon
    
    Returns:
        Dict machine_id -> utilization percentage
    """
    if plan_df.empty or horizon_minutes <= 0:
        return {}
    
    duration_col = 'duration_min' if 'duration_min' in plan_df.columns else 'duration'
    
    if duration_col not in plan_df.columns:
        return {}
    
    machine_loads = plan_df.groupby('machine_id')[duration_col].sum()
    
    return {
        m: (load / horizon_minutes) * 100 
        for m, load in machine_loads.items()
    }


# ============================================================
# OBJECTIVE FUNCTION CLASSES
# ============================================================

class ObjectiveFunction(ABC):
    """Abstract base class for objective functions."""
    
    @abstractmethod
    def evaluate(self, plan_df: pd.DataFrame, context: Dict[str, Any]) -> float:
        """
        Evaluate objective function value.
        
        Args:
            plan_df: Current schedule DataFrame
            context: Additional data (orders_df, setup_matrix, etc.)
        
        Returns:
            Objective value (lower is better by convention)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return objective name for logging/display."""
        pass


class MakespanObjective(ObjectiveFunction):
    """Makespan minimization objective."""
    
    def evaluate(self, plan_df: pd.DataFrame, context: Dict[str, Any]) -> float:
        return compute_makespan(plan_df)
    
    def get_name(self) -> str:
        return "Makespan"


class TardinessObjective(ObjectiveFunction):
    """Total tardiness minimization objective."""
    
    def evaluate(self, plan_df: pd.DataFrame, context: Dict[str, Any]) -> float:
        orders_df = context.get('orders_df', pd.DataFrame())
        return compute_total_tardiness(plan_df, orders_df)
    
    def get_name(self) -> str:
        return "Tardiness"


class SetupTimeObjective(ObjectiveFunction):
    """Total setup time minimization objective."""
    
    def evaluate(self, plan_df: pd.DataFrame, context: Dict[str, Any]) -> float:
        setup_matrix = context.get('setup_matrix')
        return compute_total_setup_time(plan_df, setup_matrix)
    
    def get_name(self) -> str:
        return "SetupTime"


class LoadBalanceObjective(ObjectiveFunction):
    """Machine load balancing objective."""
    
    def evaluate(self, plan_df: pd.DataFrame, context: Dict[str, Any]) -> float:
        return compute_load_imbalance(plan_df)
    
    def get_name(self) -> str:
        return "LoadImbalance"


# ============================================================
# MULTI-OBJECTIVE AGGREGATION
# ============================================================

@dataclass
class WeightedSumObjective(ObjectiveFunction):
    """
    Weighted sum of multiple objectives.
    
    f(x) = Σ w_i * f_i(x)
    
    Pros:
    - Simple to implement
    - Single scalar optimization
    
    Cons:
    - Cannot find all Pareto-optimal solutions
    - Sensitive to weight selection
    
    TODO[R&D]: Implement adaptive weight adjustment:
    - Start with equal weights
    - Adjust based on solution quality on each objective
    - Track Pareto front approximation
    """
    objectives: List[ObjectiveFunction] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.weights:
            # Equal weights by default
            self.weights = [1.0 / len(self.objectives)] * len(self.objectives)
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def evaluate(self, plan_df: pd.DataFrame, context: Dict[str, Any]) -> float:
        total = 0.0
        for obj, weight in zip(self.objectives, self.weights):
            value = obj.evaluate(plan_df, context)
            total += weight * value
        return total
    
    def get_name(self) -> str:
        return "WeightedSum"
    
    def evaluate_components(self, plan_df: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate each objective component separately."""
        return {
            obj.get_name(): obj.evaluate(plan_df, context)
            for obj in self.objectives
        }


@dataclass
class EpsilonConstraintObjective(ObjectiveFunction):
    """
    ε-constraint method for multi-objective optimization.
    
    Optimize one objective while constraining others:
    min f_1(x) subject to f_i(x) <= ε_i for i = 2, ..., k
    
    Pros:
    - Can find all Pareto-optimal solutions
    - Supports non-convex Pareto fronts
    
    Cons:
    - Requires solving multiple problems
    - Need to choose epsilon values
    
    TODO[R&D]: Implement augmented ε-constraint (AUGMECON):
    - Better handling of redundant constraints
    - Guaranteed Pareto-optimality
    Reference: Mavrotas (2009)
    """
    primary_objective: ObjectiveFunction = None
    constrained_objectives: List[ObjectiveFunction] = field(default_factory=list)
    epsilon_bounds: List[float] = field(default_factory=list)
    
    def evaluate(self, plan_df: pd.DataFrame, context: Dict[str, Any]) -> float:
        """Evaluate primary objective (constraints checked separately)."""
        return self.primary_objective.evaluate(plan_df, context)
    
    def check_constraints(self, plan_df: pd.DataFrame, context: Dict[str, Any]) -> bool:
        """Check if epsilon constraints are satisfied."""
        for obj, eps in zip(self.constrained_objectives, self.epsilon_bounds):
            if obj.evaluate(plan_df, context) > eps:
                return False
        return True
    
    def get_constraint_violations(self, plan_df: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, float]:
        """Get violation amount for each constraint."""
        violations = {}
        for obj, eps in zip(self.constrained_objectives, self.epsilon_bounds):
            value = obj.evaluate(plan_df, context)
            violations[obj.get_name()] = max(0, value - eps)
        return violations
    
    def get_name(self) -> str:
        return "EpsilonConstraint"


# ============================================================
# PARETO FRONT UTILITIES
# ============================================================

@dataclass
class ParetoPoint:
    """A point in objective space."""
    objectives: Dict[str, float]
    solution_id: str
    is_dominated: bool = False


def is_dominated(point_a: Dict[str, float], point_b: Dict[str, float]) -> bool:
    """
    Check if point_a is dominated by point_b.
    
    point_a is dominated if point_b is at least as good on all objectives
    and strictly better on at least one.
    
    Assumes minimization for all objectives.
    """
    at_least_as_good = all(point_b[k] <= point_a[k] for k in point_a)
    strictly_better = any(point_b[k] < point_a[k] for k in point_a)
    return at_least_as_good and strictly_better


def compute_pareto_front(points: List[Dict[str, float]]) -> List[int]:
    """
    Compute Pareto front from a set of points.
    
    Args:
        points: List of objective dictionaries
    
    Returns:
        Indices of non-dominated points
    
    TODO[R&D]: Implement efficient Pareto front algorithms:
    - Fast non-dominated sorting (NSGA-II style)
    - Divide and conquer for large sets
    - Online Pareto front maintenance
    """
    n = len(points)
    is_pareto = [True] * n
    
    for i in range(n):
        for j in range(n):
            if i != j and is_pareto[i]:
                if is_dominated(points[i], points[j]):
                    is_pareto[i] = False
                    break
    
    return [i for i in range(n) if is_pareto[i]]


def compute_hypervolume(pareto_front: List[Dict[str, float]], reference_point: Dict[str, float]) -> float:
    """
    Compute hypervolume indicator for Pareto front quality.
    
    Hypervolume = volume of objective space dominated by the front
    and bounded by the reference point.
    
    Higher is better.
    
    TODO[R&D]: Implement efficient hypervolume calculation:
    - WFG algorithm for high dimensions
    - Incremental updates for dynamic fronts
    Reference: While et al. (2012)
    """
    # Simple 2D implementation
    if not pareto_front:
        return 0.0
    
    keys = list(pareto_front[0].keys())
    if len(keys) != 2:
        logger.warning("Hypervolume only implemented for 2D. Returning 0.")
        return 0.0
    
    k1, k2 = keys
    
    # Sort by first objective
    sorted_front = sorted(pareto_front, key=lambda p: p[k1])
    
    hv = 0.0
    prev_k2 = reference_point[k2]
    
    for point in sorted_front:
        if point[k1] < reference_point[k1] and point[k2] < prev_k2:
            hv += (reference_point[k1] - point[k1]) * (prev_k2 - point[k2])
            prev_k2 = point[k2]
    
    return hv


# ============================================================
# OBJECTIVE FACTORY
# ============================================================

def create_objective(
    objective_type: str,
    weights: Optional[Dict[str, float]] = None,
    epsilon_bounds: Optional[Dict[str, float]] = None
) -> ObjectiveFunction:
    """
    Factory function to create objective functions.
    
    Args:
        objective_type: One of 'makespan', 'tardiness', 'setup', 'load_balance', 
                       'weighted_sum', 'epsilon_constraint'
        weights: Weights for weighted sum (objective_name -> weight)
        epsilon_bounds: Bounds for epsilon constraint (objective_name -> bound)
    
    Returns:
        ObjectiveFunction instance
    """
    if objective_type == 'makespan':
        return MakespanObjective()
    
    elif objective_type == 'tardiness':
        return TardinessObjective()
    
    elif objective_type == 'setup':
        return SetupTimeObjective()
    
    elif objective_type == 'load_balance':
        return LoadBalanceObjective()
    
    elif objective_type == 'weighted_sum':
        objectives = [
            MakespanObjective(),
            TardinessObjective(),
            SetupTimeObjective(),
            LoadBalanceObjective(),
        ]
        weight_list = [
            weights.get('Makespan', 1.0) if weights else 1.0,
            weights.get('Tardiness', 0.5) if weights else 0.5,
            weights.get('SetupTime', 0.2) if weights else 0.2,
            weights.get('LoadImbalance', 0.1) if weights else 0.1,
        ]
        return WeightedSumObjective(objectives=objectives, weights=weight_list)
    
    elif objective_type == 'epsilon_constraint':
        primary = MakespanObjective()
        constrained = [TardinessObjective(), SetupTimeObjective()]
        bounds = [
            epsilon_bounds.get('Tardiness', float('inf')) if epsilon_bounds else float('inf'),
            epsilon_bounds.get('SetupTime', float('inf')) if epsilon_bounds else float('inf'),
        ]
        return EpsilonConstraintObjective(
            primary_objective=primary,
            constrained_objectives=constrained,
            epsilon_bounds=bounds
        )
    
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")



