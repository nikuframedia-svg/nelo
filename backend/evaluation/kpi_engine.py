"""
ProdPlan 4.0 - KPI Engine

Rigorous Key Performance Indicator computation for production scheduling.

Mathematical Definitions
========================

Makespan (Cmax):
---------------
    Cmax = max{Cⱼ : j ∈ Jobs}
    
    where Cⱼ is the completion time of job j.
    
    Minimizing makespan maximizes throughput.

Total Tardiness (∑Tⱼ):
---------------------
    Tⱼ = max(0, Cⱼ - dⱼ)
    
    where dⱼ is the due date of job j.
    
    Total tardiness: T = Σⱼ Tⱼ

Weighted Tardiness (∑wⱼTⱼ):
--------------------------
    Weighted by priority: T_w = Σⱼ wⱼ × max(0, Cⱼ - dⱼ)

Total Flow Time (∑Fⱼ):
---------------------
    Fⱼ = Cⱼ - rⱼ
    
    where rⱼ is the release/arrival time of job j.
    
    Minimizing flow time reduces WIP.

Machine Utilization (Uₘ):
------------------------
    Uₘ = (Σ processing time on m) / (Cmax - start_time)
    
    where m is a machine.

Setup Time Ratio:
----------------
    Setup_ratio = (Total setup time) / (Total processing time)

On-Time Delivery (OTD):
----------------------
    OTD = |{j : Cⱼ ≤ dⱼ}| / |Jobs| × 100%

R&D / SIFIDE: WP4 - Evaluation Framework
Research Question Q4.2: Which KPIs best correlate with customer satisfaction?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from data_quality import compute_snr, interpret_snr

logger = logging.getLogger(__name__)


# ============================================================
# KPI DATA CLASSES
# ============================================================

@dataclass
class PlanKPIs:
    """
    Complete set of KPIs for a production plan.
    
    All timing KPIs are in hours unless otherwise noted.
    """
    # ===== TIMING KPIs =====
    makespan_hours: float = 0.0  # Cmax in hours
    total_processing_hours: float = 0.0
    total_idle_hours: float = 0.0
    
    # ===== DELIVERY KPIs =====
    total_tardiness_hours: float = 0.0  # ΣTⱼ
    max_tardiness_hours: float = 0.0    # max{Tⱼ}
    weighted_tardiness: float = 0.0     # ΣwⱼTⱼ
    total_earliness_hours: float = 0.0  # Σmax(0, dⱼ - Cⱼ)
    
    # OTD metrics
    otd_percent: float = 0.0
    orders_on_time: int = 0
    orders_late: int = 0
    orders_early: int = 0
    
    # ===== FLOW TIME KPIs =====
    total_flow_time_hours: float = 0.0   # ΣFⱼ
    avg_flow_time_hours: float = 0.0     # (ΣFⱼ)/n
    max_flow_time_hours: float = 0.0     # max{Fⱼ}
    
    # ===== SETUP KPIs =====
    total_setup_hours: float = 0.0
    setup_count: int = 0
    avg_setup_min: float = 0.0
    setup_ratio: float = 0.0  # setup_time / processing_time
    
    # ===== RESOURCE KPIs =====
    avg_utilization_pct: float = 0.0
    min_utilization_pct: float = 0.0
    max_utilization_pct: float = 0.0
    utilization_std: float = 0.0  # Measures load imbalance
    bottleneck_machine: str = ""
    bottleneck_utilization: float = 0.0
    machines_used: int = 0
    
    # ===== VOLUME KPIs =====
    total_operations: int = 0
    total_orders: int = 0
    total_articles: int = 0
    
    # ===== QUALITY INDICATORS =====
    overlaps_count: int = 0  # Constraint violations
    precedence_violations: int = 0
    
    # ===== SNR/CONFIDENCE =====
    plan_snr: float = 0.0  # Signal-to-noise ratio of plan execution
    confidence: float = 0.0
    
    # Route distribution
    route_distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-friendly dictionary with proper rounding."""
        return {
            # Timing
            'makespan_hours': round(self.makespan_hours, 2),
            'total_processing_hours': round(self.total_processing_hours, 2),
            'total_idle_hours': round(self.total_idle_hours, 2),
            
            # Delivery
            'total_tardiness_hours': round(self.total_tardiness_hours, 2),
            'max_tardiness_hours': round(self.max_tardiness_hours, 2),
            'weighted_tardiness': round(self.weighted_tardiness, 2),
            'otd_percent': round(self.otd_percent, 1),
            'orders_on_time': self.orders_on_time,
            'orders_late': self.orders_late,
            'orders_early': self.orders_early,
            
            # Flow time
            'total_flow_time_hours': round(self.total_flow_time_hours, 2),
            'avg_flow_time_hours': round(self.avg_flow_time_hours, 2),
            'max_flow_time_hours': round(self.max_flow_time_hours, 2),
            
            # Setup
            'total_setup_hours': round(self.total_setup_hours, 2),
            'setup_count': self.setup_count,
            'avg_setup_min': round(self.avg_setup_min, 1),
            'setup_ratio': round(self.setup_ratio, 4),
            
            # Resources
            'avg_utilization_pct': round(self.avg_utilization_pct, 1),
            'min_utilization_pct': round(self.min_utilization_pct, 1),
            'max_utilization_pct': round(self.max_utilization_pct, 1),
            'utilization_std': round(self.utilization_std, 2),
            'bottleneck_machine': self.bottleneck_machine,
            'bottleneck_utilization': round(self.bottleneck_utilization, 1),
            'machines_used': self.machines_used,
            
            # Volume
            'total_operations': self.total_operations,
            'total_orders': self.total_orders,
            'total_articles': self.total_articles,
            
            # Quality
            'overlaps_count': self.overlaps_count,
            'precedence_violations': self.precedence_violations,
            'plan_snr': round(self.plan_snr, 2),
            'confidence': round(self.confidence, 3),
            
            # Route
            'route_distribution': self.route_distribution,
        }


@dataclass
class PlanComparison:
    """
    Statistical comparison between two plans.
    
    Follows the format: Plan B - Plan A (positive = B better for minimization objectives)
    """
    # ===== ABSOLUTE DELTAS =====
    makespan_delta_hours: float = 0.0
    tardiness_delta_hours: float = 0.0
    setup_delta_hours: float = 0.0
    flow_time_delta_hours: float = 0.0
    otd_delta_pct: float = 0.0
    utilization_delta_pct: float = 0.0
    
    # ===== PERCENTAGE IMPROVEMENTS =====
    # Positive = Plan B is better (for minimization: smaller value)
    makespan_improvement_pct: float = 0.0
    tardiness_improvement_pct: float = 0.0
    setup_improvement_pct: float = 0.0
    flow_time_improvement_pct: float = 0.0
    
    # ===== STATISTICAL SIGNIFICANCE =====
    # TODO[R&D]: Implement proper statistical tests
    is_significant: bool = False
    p_value: Optional[float] = None
    effect_size: Optional[float] = None  # Cohen's d or similar
    
    # ===== DOMINANCE =====
    plan_a_dominates: bool = False
    plan_b_dominates: bool = False
    pareto_equivalent: bool = False
    
    # ===== RECOMMENDATION =====
    recommendation: str = ""
    confidence: float = 0.0
    
    # Raw KPIs
    kpis_a: Optional[PlanKPIs] = None
    kpis_b: Optional[PlanKPIs] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-friendly dictionary."""
        return {
            # Deltas
            'makespan_delta_hours': round(self.makespan_delta_hours, 2),
            'tardiness_delta_hours': round(self.tardiness_delta_hours, 2),
            'setup_delta_hours': round(self.setup_delta_hours, 2),
            'flow_time_delta_hours': round(self.flow_time_delta_hours, 2),
            'otd_delta_pct': round(self.otd_delta_pct, 1),
            'utilization_delta_pct': round(self.utilization_delta_pct, 1),
            
            # Improvements
            'makespan_improvement_pct': round(self.makespan_improvement_pct, 1),
            'tardiness_improvement_pct': round(self.tardiness_improvement_pct, 1),
            'setup_improvement_pct': round(self.setup_improvement_pct, 1),
            'flow_time_improvement_pct': round(self.flow_time_improvement_pct, 1),
            
            # Dominance
            'plan_a_dominates': self.plan_a_dominates,
            'plan_b_dominates': self.plan_b_dominates,
            'pareto_equivalent': self.pareto_equivalent,
            
            # Recommendation
            'recommendation': self.recommendation,
            'confidence': round(self.confidence, 2),
            
            # Raw KPIs
            'kpis_a': self.kpis_a.to_dict() if self.kpis_a else None,
            'kpis_b': self.kpis_b.to_dict() if self.kpis_b else None,
        }


# ============================================================
# KPI COMPUTATION
# ============================================================

def compute_plan_kpis(
    plan_df: pd.DataFrame,
    orders_df: Optional[pd.DataFrame] = None,
    setup_matrix: Optional[Dict] = None,
    horizon_hours: float = 168.0
) -> PlanKPIs:
    """
    Compute comprehensive KPIs for a production plan.
    
    Mathematical Formulations:
    -------------------------
    
    Makespan: Cmax = max{end_time} - min{start_time}
    
    Tardiness: Tⱼ = max(0, Cⱼ - dⱼ) for each order j
    
    Utilization: Uₘ = Σ(duration on m) / Cmax for each machine m
    
    Flow Time: Fⱼ = Cⱼ - rⱼ where rⱼ is release time
    
    Args:
        plan_df: Schedule with columns [operation_id, machine_id, start_time, end_time, ...]
        orders_df: Orders with columns [order_id, due_date, priority, ...]
        setup_matrix: Setup times Dict[(from_family, to_family)] -> minutes
        horizon_hours: Planning horizon for utilization calculation
    
    Returns:
        PlanKPIs with all computed metrics
    """
    kpis = PlanKPIs()
    
    if plan_df is None or plan_df.empty:
        return kpis
    
    # Standardize column names
    plan_df = _standardize_columns(plan_df)
    
    # ===== BASIC COUNTS =====
    kpis.total_operations = len(plan_df)
    kpis.total_orders = plan_df['order_id'].nunique() if 'order_id' in plan_df.columns else 0
    kpis.total_articles = plan_df['article_id'].nunique() if 'article_id' in plan_df.columns else 0
    kpis.machines_used = plan_df['machine_id'].nunique() if 'machine_id' in plan_df.columns else 0
    
    # ===== TIMING KPIs =====
    if 'start_time' in plan_df.columns and 'end_time' in plan_df.columns:
        kpis.makespan_hours = _compute_makespan(plan_df) / 60
        kpis.total_processing_hours = _compute_total_processing(plan_df) / 60
    
    # ===== DELIVERY KPIs =====
    if orders_df is not None and 'due_date' in orders_df.columns:
        delivery = _compute_delivery_kpis(plan_df, orders_df)
        kpis.total_tardiness_hours = delivery['total_tardiness'] / 60
        kpis.max_tardiness_hours = delivery['max_tardiness'] / 60
        kpis.total_earliness_hours = delivery['total_earliness'] / 60
        kpis.weighted_tardiness = delivery['weighted_tardiness'] / 60
        kpis.otd_percent = delivery['otd_percent']
        kpis.orders_on_time = delivery['on_time']
        kpis.orders_late = delivery['late']
        kpis.orders_early = delivery['early']
        kpis.total_orders = delivery['total_orders']
    
    # ===== FLOW TIME KPIs =====
    flow = _compute_flow_time_kpis(plan_df, orders_df)
    kpis.total_flow_time_hours = flow['total'] / 60
    kpis.avg_flow_time_hours = flow['avg'] / 60
    kpis.max_flow_time_hours = flow['max'] / 60
    
    # ===== SETUP KPIs =====
    setup = _compute_setup_kpis(plan_df, setup_matrix)
    kpis.total_setup_hours = setup['total'] / 60
    kpis.setup_count = setup['count']
    kpis.avg_setup_min = setup['avg']
    if kpis.total_processing_hours > 0:
        kpis.setup_ratio = kpis.total_setup_hours / kpis.total_processing_hours
    
    # ===== RESOURCE KPIs =====
    util = _compute_utilization_kpis(plan_df, horizon_hours * 60)
    kpis.avg_utilization_pct = util['avg']
    kpis.min_utilization_pct = util['min']
    kpis.max_utilization_pct = util['max']
    kpis.utilization_std = util['std']
    kpis.bottleneck_machine = util['bottleneck']
    kpis.bottleneck_utilization = util['bottleneck_util']
    
    # ===== QUALITY INDICATORS =====
    kpis.overlaps_count = _count_overlaps(plan_df)
    kpis.precedence_violations = _count_precedence_violations(plan_df)
    
    # ===== ROUTE DISTRIBUTION =====
    if 'route_label' in plan_df.columns:
        kpis.route_distribution = plan_df['route_label'].value_counts().to_dict()
    
    # ===== SNR =====
    if 'duration_min' in plan_df.columns:
        durations = plan_df['duration_min'].dropna().values
        if len(durations) >= 3:
            kpis.plan_snr = compute_snr(durations)
            _, _, kpis.confidence = interpret_snr(kpis.plan_snr)
    
    return kpis


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for consistent processing."""
    df = df.copy()
    
    # Map common variations
    mappings = {
        'start_min': 'start_time',
        'end_min': 'end_time',
        'op_id': 'operation_id',
    }
    
    for old, new in mappings.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    
    # Compute duration if missing
    if 'duration_min' not in df.columns:
        if 'start_time' in df.columns and 'end_time' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['end_time']):
                df['duration_min'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
            else:
                df['duration_min'] = df['end_time'] - df['start_time']
    
    return df


def _compute_makespan(plan_df: pd.DataFrame) -> float:
    """
    Compute makespan: Cmax = max{Cⱼ} - min{Sⱼ}
    
    Returns: Makespan in minutes
    """
    if pd.api.types.is_datetime64_any_dtype(plan_df['end_time']):
        start = plan_df['start_time'].min()
        end = plan_df['end_time'].max()
        return (end - start).total_seconds() / 60
    else:
        return float(plan_df['end_time'].max() - plan_df['start_time'].min())


def _compute_total_processing(plan_df: pd.DataFrame) -> float:
    """Compute total processing time in minutes."""
    if 'duration_min' in plan_df.columns:
        return plan_df['duration_min'].sum()
    return 0.0


def _compute_delivery_kpis(plan_df: pd.DataFrame, orders_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute delivery KPIs: tardiness, OTD, earliness.
    
    Tardiness: Tⱼ = max(0, Cⱼ - dⱼ)
    Earliness: Eⱼ = max(0, dⱼ - Cⱼ)
    Weighted Tardiness: Σwⱼ × Tⱼ
    """
    result = {
        'total_tardiness': 0.0,
        'max_tardiness': 0.0,
        'total_earliness': 0.0,
        'weighted_tardiness': 0.0,
        'otd_percent': 100.0,
        'on_time': 0,
        'late': 0,
        'early': 0,
        'total_orders': 0,
    }
    
    if 'order_id' not in plan_df.columns:
        return result
    
    # Get completion time per order
    order_completion = plan_df.groupby('order_id').agg(
        completion=('end_time', 'max')
    ).reset_index()
    
    # Merge with orders
    merged = order_completion.merge(
        orders_df[['order_id', 'due_date'] + (['priority'] if 'priority' in orders_df.columns else [])],
        on='order_id',
        how='left'
    )
    
    total_tardiness = 0.0
    max_tardiness = 0.0
    total_earliness = 0.0
    weighted_tardiness = 0.0
    on_time = 0
    late = 0
    early = 0
    
    for _, row in merged.iterrows():
        if pd.isna(row['due_date']):
            continue
        
        completion = row['completion']
        due = row['due_date']
        priority = row.get('priority', 1) if 'priority' in row.index else 1
        
        # Convert to comparable format
        if isinstance(completion, datetime) and isinstance(due, datetime):
            tardiness = max(0, (completion - due).total_seconds() / 60)
            earliness = max(0, (due - completion).total_seconds() / 60)
        else:
            tardiness = max(0, float(completion) - float(due))
            earliness = max(0, float(due) - float(completion))
        
        total_tardiness += tardiness
        max_tardiness = max(max_tardiness, tardiness)
        total_earliness += earliness
        weighted_tardiness += priority * tardiness
        
        if tardiness == 0:
            on_time += 1
            if earliness > 0:
                early += 1
        else:
            late += 1
    
    total = on_time + late
    
    result['total_tardiness'] = total_tardiness
    result['max_tardiness'] = max_tardiness
    result['total_earliness'] = total_earliness
    result['weighted_tardiness'] = weighted_tardiness
    result['otd_percent'] = (on_time / total * 100) if total > 0 else 100.0
    result['on_time'] = on_time
    result['late'] = late
    result['early'] = early
    result['total_orders'] = total
    
    return result


def _compute_flow_time_kpis(plan_df: pd.DataFrame, orders_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """
    Compute flow time KPIs.
    
    Flow Time: Fⱼ = Cⱼ - rⱼ (completion - release time)
    
    If release times not available, use plan start as release time.
    """
    result = {'total': 0.0, 'avg': 0.0, 'max': 0.0}
    
    if 'order_id' not in plan_df.columns:
        return result
    
    # Get start and completion per order
    order_times = plan_df.groupby('order_id').agg(
        start=('start_time', 'min'),
        completion=('end_time', 'max')
    ).reset_index()
    
    flow_times = []
    
    for _, row in order_times.iterrows():
        start = row['start']
        completion = row['completion']
        
        if isinstance(completion, datetime) and isinstance(start, datetime):
            flow = (completion - start).total_seconds() / 60
        else:
            flow = float(completion) - float(start)
        
        flow_times.append(flow)
    
    if flow_times:
        result['total'] = sum(flow_times)
        result['avg'] = np.mean(flow_times)
        result['max'] = max(flow_times)
    
    return result


def _compute_setup_kpis(plan_df: pd.DataFrame, setup_matrix: Optional[Dict]) -> Dict[str, float]:
    """
    Compute setup time KPIs.
    
    Setup occurs when consecutive operations on same machine have different families.
    """
    result = {'total': 0.0, 'count': 0, 'avg': 0.0}
    
    if 'machine_id' not in plan_df.columns:
        return result
    
    default_setup = 15.0  # minutes
    total_setup = 0.0
    setup_count = 0
    
    for machine_id, machine_ops in plan_df.groupby('machine_id'):
        sorted_ops = machine_ops.sort_values('start_time')
        
        families = sorted_ops['setup_family'].tolist() if 'setup_family' in sorted_ops.columns else []
        
        for i in range(1, len(families)):
            prev_family = families[i-1]
            curr_family = families[i]
            
            if prev_family != curr_family:
                setup_count += 1
                
                if setup_matrix and (prev_family, curr_family) in setup_matrix:
                    total_setup += setup_matrix[(prev_family, curr_family)]
                else:
                    total_setup += default_setup
    
    result['total'] = total_setup
    result['count'] = setup_count
    result['avg'] = total_setup / setup_count if setup_count > 0 else 0.0
    
    return result


def _compute_utilization_kpis(plan_df: pd.DataFrame, horizon_minutes: float) -> Dict[str, Any]:
    """
    Compute machine utilization KPIs.
    
    Utilization: Uₘ = (processing time on m) / horizon × 100%
    """
    result = {
        'avg': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0,
        'bottleneck': '', 'bottleneck_util': 0.0
    }
    
    if 'machine_id' not in plan_df.columns or 'duration_min' not in plan_df.columns:
        return result
    
    if horizon_minutes <= 0:
        return result
    
    machine_loads = plan_df.groupby('machine_id')['duration_min'].sum()
    utilizations = (machine_loads / horizon_minutes * 100).clip(upper=100)
    
    if utilizations.empty:
        return result
    
    result['avg'] = utilizations.mean()
    result['min'] = utilizations.min()
    result['max'] = utilizations.max()
    result['std'] = utilizations.std()
    result['bottleneck'] = utilizations.idxmax()
    result['bottleneck_util'] = utilizations.max()
    
    return result


def _count_overlaps(plan_df: pd.DataFrame) -> int:
    """Count overlapping operations on same machine."""
    overlaps = 0
    
    if 'machine_id' not in plan_df.columns:
        return 0
    
    for machine_id, machine_ops in plan_df.groupby('machine_id'):
        sorted_ops = machine_ops.sort_values('start_time')
        times = sorted_ops[['start_time', 'end_time']].values
        
        for i in range(len(times) - 1):
            # Check if end[i] > start[i+1]
            if pd.api.types.is_datetime64_any_dtype(plan_df['end_time']):
                if times[i][1] > times[i+1][0]:
                    overlaps += 1
            else:
                if float(times[i][1]) > float(times[i+1][0]):
                    overlaps += 1
    
    return overlaps


def _count_precedence_violations(plan_df: pd.DataFrame) -> int:
    """Count precedence violations within orders."""
    violations = 0
    
    if 'order_id' not in plan_df.columns or 'op_seq' not in plan_df.columns:
        return 0
    
    for order_id, order_ops in plan_df.groupby('order_id'):
        sorted_ops = order_ops.sort_values('op_seq')
        
        for i in range(len(sorted_ops) - 1):
            curr_end = sorted_ops.iloc[i]['end_time']
            next_start = sorted_ops.iloc[i+1]['start_time']
            
            if pd.api.types.is_datetime64_any_dtype(plan_df['end_time']):
                if curr_end > next_start:
                    violations += 1
            else:
                if float(curr_end) > float(next_start):
                    violations += 1
    
    return violations


# ============================================================
# PLAN COMPARISON
# ============================================================

def compare_plans(
    plan_a: pd.DataFrame,
    plan_b: pd.DataFrame,
    orders_df: Optional[pd.DataFrame] = None,
    setup_matrix: Optional[Dict] = None,
    weights: Optional[Dict[str, float]] = None
) -> PlanComparison:
    """
    Statistically compare two production plans.
    
    Convention:
    - Negative delta = Plan B is better (for minimization objectives)
    - Positive improvement_pct = Plan B is better
    
    Pareto Dominance:
    - Plan A dominates B if A ≤ B on all objectives and A < B on at least one
    - Plans are Pareto-equivalent if neither dominates
    
    Args:
        plan_a: Baseline plan
        plan_b: Alternative plan
        orders_df: Orders with due dates
        setup_matrix: Setup time matrix
        weights: Weights for overall comparison
    
    Returns:
        PlanComparison with detailed metrics
    """
    if weights is None:
        weights = {
            'makespan': 0.3,
            'tardiness': 0.3,
            'setup': 0.2,
            'flow_time': 0.2,
        }
    
    # Compute KPIs
    kpis_a = compute_plan_kpis(plan_a, orders_df, setup_matrix)
    kpis_b = compute_plan_kpis(plan_b, orders_df, setup_matrix)
    
    comparison = PlanComparison(kpis_a=kpis_a, kpis_b=kpis_b)
    
    # ===== DELTAS (B - A) =====
    comparison.makespan_delta_hours = kpis_b.makespan_hours - kpis_a.makespan_hours
    comparison.tardiness_delta_hours = kpis_b.total_tardiness_hours - kpis_a.total_tardiness_hours
    comparison.setup_delta_hours = kpis_b.total_setup_hours - kpis_a.total_setup_hours
    comparison.flow_time_delta_hours = kpis_b.avg_flow_time_hours - kpis_a.avg_flow_time_hours
    comparison.otd_delta_pct = kpis_b.otd_percent - kpis_a.otd_percent
    comparison.utilization_delta_pct = kpis_b.avg_utilization_pct - kpis_a.avg_utilization_pct
    
    # ===== IMPROVEMENTS (positive = B better for min objectives) =====
    if kpis_a.makespan_hours > 0:
        comparison.makespan_improvement_pct = (kpis_a.makespan_hours - kpis_b.makespan_hours) / kpis_a.makespan_hours * 100
    
    if kpis_a.total_tardiness_hours > 0:
        comparison.tardiness_improvement_pct = (kpis_a.total_tardiness_hours - kpis_b.total_tardiness_hours) / kpis_a.total_tardiness_hours * 100
    
    if kpis_a.total_setup_hours > 0:
        comparison.setup_improvement_pct = (kpis_a.total_setup_hours - kpis_b.total_setup_hours) / kpis_a.total_setup_hours * 100
    
    if kpis_a.avg_flow_time_hours > 0:
        comparison.flow_time_improvement_pct = (kpis_a.avg_flow_time_hours - kpis_b.avg_flow_time_hours) / kpis_a.avg_flow_time_hours * 100
    
    # ===== PARETO DOMINANCE =====
    obj_a = [kpis_a.makespan_hours, kpis_a.total_tardiness_hours, kpis_a.total_setup_hours]
    obj_b = [kpis_b.makespan_hours, kpis_b.total_tardiness_hours, kpis_b.total_setup_hours]
    
    a_dominates = all(a <= b for a, b in zip(obj_a, obj_b)) and any(a < b for a, b in zip(obj_a, obj_b))
    b_dominates = all(b <= a for a, b in zip(obj_a, obj_b)) and any(b < a for a, b in zip(obj_a, obj_b))
    
    comparison.plan_a_dominates = a_dominates
    comparison.plan_b_dominates = b_dominates
    comparison.pareto_equivalent = not a_dominates and not b_dominates
    
    # ===== RECOMMENDATION =====
    improvements = [
        comparison.makespan_improvement_pct,
        comparison.tardiness_improvement_pct,
        comparison.setup_improvement_pct,
    ]
    
    avg_improvement = np.mean([i for i in improvements if not np.isnan(i)])
    
    if b_dominates:
        comparison.recommendation = "Plano B é estritamente melhor (domina Plano A)"
        comparison.confidence = 0.95
    elif a_dominates:
        comparison.recommendation = "Plano A é estritamente melhor (domina Plano B)"
        comparison.confidence = 0.95
    elif avg_improvement > 5:
        comparison.recommendation = f"Plano B é melhor (melhoria média: {avg_improvement:.1f}%)"
        comparison.confidence = 0.8
    elif avg_improvement < -5:
        comparison.recommendation = f"Plano A é melhor (Plano B piora {-avg_improvement:.1f}%)"
        comparison.confidence = 0.8
    else:
        comparison.recommendation = "Planos são aproximadamente equivalentes"
        comparison.confidence = 0.6
    
    return comparison


# ============================================================
# KPI ENGINE CLASS
# ============================================================

class KPIEngine:
    """
    High-level KPI computation and tracking engine.
    
    Provides:
    - Plan evaluation
    - Multi-plan comparison
    - KPI history tracking
    - Benchmark comparisons
    """
    
    def __init__(
        self,
        orders_df: Optional[pd.DataFrame] = None,
        setup_matrix: Optional[Dict] = None,
        horizon_hours: float = 168.0
    ):
        self.orders_df = orders_df
        self.setup_matrix = setup_matrix
        self.horizon_hours = horizon_hours
        self._history: List[Tuple[str, PlanKPIs, datetime]] = []
    
    def evaluate(self, plan_df: pd.DataFrame, plan_name: str = "plan") -> PlanKPIs:
        """Evaluate a plan and store in history."""
        kpis = compute_plan_kpis(
            plan_df, 
            self.orders_df, 
            self.setup_matrix, 
            self.horizon_hours
        )
        self._history.append((plan_name, kpis, datetime.now()))
        return kpis
    
    def compare(self, plan_a: pd.DataFrame, plan_b: pd.DataFrame) -> PlanComparison:
        """Compare two plans."""
        return compare_plans(plan_a, plan_b, self.orders_df, self.setup_matrix)
    
    def get_history_df(self) -> pd.DataFrame:
        """Get KPI history as DataFrame."""
        records = []
        for name, kpis, ts in self._history:
            record = kpis.to_dict()
            record['plan_name'] = name
            record['timestamp'] = ts
            records.append(record)
        return pd.DataFrame(records)
    
    def generate_report(self, plan_df: pd.DataFrame, plan_name: str = "Plano") -> str:
        """Generate human-readable KPI report."""
        kpis = compute_plan_kpis(plan_df, self.orders_df, self.setup_matrix, self.horizon_hours)
        
        return f"""
════════════════════════════════════════════════════════════════
                 RELATÓRIO KPI - {plan_name}
════════════════════════════════════════════════════════════════

📊 MÉTRICAS DE TEMPO
────────────────────
  Makespan (Cmax):           {kpis.makespan_hours:.1f} h
  Tempo Total Processamento: {kpis.total_processing_hours:.1f} h
  Flow Time Médio:           {kpis.avg_flow_time_hours:.1f} h

📦 MÉTRICAS DE ENTREGA
─────────────────────
  OTD (On-Time Delivery):    {kpis.otd_percent:.1f}%
  Encomendas a tempo:        {kpis.orders_on_time}
  Encomendas atrasadas:      {kpis.orders_late}
  Atraso Total (ΣTⱼ):        {kpis.total_tardiness_hours:.1f} h
  Atraso Máximo:             {kpis.max_tardiness_hours:.1f} h

🔧 MÉTRICAS DE SETUP
────────────────────
  Tempo Total Setup:         {kpis.total_setup_hours:.1f} h
  Número de Setups:          {kpis.setup_count}
  Setup Ratio:               {kpis.setup_ratio:.2%}

🏭 MÉTRICAS DE RECURSOS
──────────────────────
  Utilização Média:          {kpis.avg_utilization_pct:.1f}%
  Utilização Máxima:         {kpis.max_utilization_pct:.1f}%
  Desvio Padrão Carga:       {kpis.utilization_std:.1f}
  Máquina Gargalo:           {kpis.bottleneck_machine}
  Utilização Gargalo:        {kpis.bottleneck_utilization:.1f}%

📈 VOLUME
─────────
  Total Operações:           {kpis.total_operations}
  Total Encomendas:          {kpis.total_orders}
  Total Artigos:             {kpis.total_articles}
  Máquinas Utilizadas:       {kpis.machines_used}

⚠️ QUALIDADE
────────────
  Sobreposições:             {kpis.overlaps_count}
  Violações Precedência:     {kpis.precedence_violations}
  SNR do Plano:              {kpis.plan_snr:.2f}
  Confiança:                 {kpis.confidence:.1%}

════════════════════════════════════════════════════════════════
"""



