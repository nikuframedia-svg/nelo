"""
ProdPlan 4.0 - Plan Evaluator

This module provides tools for evaluating and comparing production plans:
- KPI computation (makespan, OTD, setup time, utilization)
- Plan comparison (baseline vs optimized)
- Sensitivity analysis
- Quality metrics for R&D

R&D / SIFIDE: WP1 - Plan quality research
Research Question Q1.4: What combination of KPIs best captures 
                        industrial plan quality?
Metrics: correlation with actual performance, stakeholder satisfaction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from objectives import (
    compute_makespan,
    compute_total_tardiness,
    compute_total_setup_time,
    compute_load_imbalance,
    compute_machine_utilization,
)

logger = logging.getLogger(__name__)


# ============================================================
# KPI COMPUTATION
# ============================================================

@dataclass
class PlanKPIs:
    """Comprehensive KPIs for a production plan."""
    
    # Timing KPIs
    makespan_hours: float = 0.0
    total_processing_hours: float = 0.0
    total_idle_hours: float = 0.0
    
    # Delivery KPIs
    total_tardiness_hours: float = 0.0
    max_tardiness_hours: float = 0.0
    otd_percent: float = 0.0  # On-Time Delivery percentage
    early_orders: int = 0
    late_orders: int = 0
    
    # Setup KPIs
    total_setup_hours: float = 0.0
    avg_setup_min: float = 0.0
    setup_count: int = 0
    
    # Resource KPIs
    avg_utilization_pct: float = 0.0
    max_utilization_pct: float = 0.0
    min_utilization_pct: float = 0.0
    load_imbalance: float = 0.0
    bottleneck_machine: str = ""
    
    # Volume KPIs
    total_operations: int = 0
    total_orders: int = 0
    total_articles: int = 0
    machines_used: int = 0
    
    # Route KPIs
    route_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Quality indicators
    overlaps: int = 0  # Constraint violations
    precedence_violations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            'makespan_hours': round(self.makespan_hours, 2),
            'total_processing_hours': round(self.total_processing_hours, 2),
            'total_idle_hours': round(self.total_idle_hours, 2),
            'total_tardiness_hours': round(self.total_tardiness_hours, 2),
            'max_tardiness_hours': round(self.max_tardiness_hours, 2),
            'otd_percent': round(self.otd_percent, 1),
            'early_orders': self.early_orders,
            'late_orders': self.late_orders,
            'total_setup_hours': round(self.total_setup_hours, 2),
            'avg_setup_min': round(self.avg_setup_min, 1),
            'setup_count': self.setup_count,
            'avg_utilization_pct': round(self.avg_utilization_pct, 1),
            'max_utilization_pct': round(self.max_utilization_pct, 1),
            'min_utilization_pct': round(self.min_utilization_pct, 1),
            'load_imbalance': round(self.load_imbalance, 2),
            'bottleneck_machine': self.bottleneck_machine,
            'total_operations': self.total_operations,
            'total_orders': self.total_orders,
            'total_articles': self.total_articles,
            'machines_used': self.machines_used,
            'route_distribution': self.route_distribution,
            'overlaps': self.overlaps,
            'precedence_violations': self.precedence_violations,
        }


def compute_plan_kpis(
    plan_df: pd.DataFrame,
    orders_df: Optional[pd.DataFrame] = None,
    setup_matrix: Optional[Dict] = None,
    horizon_hours: float = 168.0  # 1 week default
) -> PlanKPIs:
    """
    Compute comprehensive KPIs for a production plan.
    
    Args:
        plan_df: DataFrame with schedule (operation_id, machine_id, start_time, end_time, ...)
        orders_df: DataFrame with orders (order_id, due_date, ...)
        setup_matrix: Setup time matrix
        horizon_hours: Planning horizon for utilization calculation
    
    Returns:
        PlanKPIs dataclass
    
    TODO[R&D]: Add probabilistic KPIs:
    - Expected tardiness under uncertainty
    - Robustness measures
    - Schedule stability metrics
    """
    kpis = PlanKPIs()
    
    if plan_df is None or plan_df.empty:
        return kpis
    
    # Standardize column names
    plan_df = _standardize_plan_columns(plan_df)
    
    # ========== TIMING KPIs ==========
    
    kpis.makespan_hours = compute_makespan(plan_df) / 60
    
    if 'duration_min' in plan_df.columns:
        kpis.total_processing_hours = plan_df['duration_min'].sum() / 60
    
    # ========== DELIVERY KPIs ==========
    
    if orders_df is not None and 'due_date' in orders_df.columns:
        kpis.total_tardiness_hours = compute_total_tardiness(plan_df, orders_df) / 60
        
        # OTD calculation
        order_completion = plan_df.groupby('order_id')['end_time'].max().reset_index()
        merged = order_completion.merge(orders_df[['order_id', 'due_date']], on='order_id', how='left')
        
        on_time = 0
        late = 0
        early = 0
        max_tard = 0.0
        
        for _, row in merged.iterrows():
            if pd.isna(row['due_date']):
                continue
            
            completion = row['end_time']
            due = row['due_date']
            
            # Handle different types
            if isinstance(completion, datetime) and isinstance(due, datetime):
                diff = (completion - due).total_seconds() / 3600
            else:
                diff = float(completion) / 60 - float(due) / 60 if not pd.isna(due) else 0
            
            if diff <= 0:
                on_time += 1
                if diff < 0:
                    early += 1
            else:
                late += 1
                max_tard = max(max_tard, diff)
        
        total = on_time + late
        kpis.otd_percent = (on_time / total * 100) if total > 0 else 100.0
        kpis.early_orders = early
        kpis.late_orders = late
        kpis.max_tardiness_hours = max_tard
        kpis.total_orders = total
    
    # ========== SETUP KPIs ==========
    
    kpis.total_setup_hours = compute_total_setup_time(plan_df, setup_matrix) / 60
    
    # Count setups
    setup_count = 0
    for machine_id, machine_ops in plan_df.groupby('machine_id'):
        sorted_ops = machine_ops.sort_values('start_time')
        families = sorted_ops['setup_family'].tolist() if 'setup_family' in sorted_ops.columns else []
        for i in range(1, len(families)):
            if families[i] != families[i-1]:
                setup_count += 1
    
    kpis.setup_count = setup_count
    kpis.avg_setup_min = (kpis.total_setup_hours * 60 / setup_count) if setup_count > 0 else 0
    
    # ========== RESOURCE KPIs ==========
    
    utilization = compute_machine_utilization(plan_df, horizon_hours * 60)
    
    if utilization:
        kpis.avg_utilization_pct = np.mean(list(utilization.values()))
        kpis.max_utilization_pct = max(utilization.values())
        kpis.min_utilization_pct = min(utilization.values())
        kpis.bottleneck_machine = max(utilization, key=utilization.get)
    
    kpis.load_imbalance = compute_load_imbalance(plan_df)
    kpis.machines_used = plan_df['machine_id'].nunique()
    
    # ========== VOLUME KPIs ==========
    
    kpis.total_operations = len(plan_df)
    kpis.total_orders = plan_df['order_id'].nunique() if 'order_id' in plan_df.columns else 0
    kpis.total_articles = plan_df['article_id'].nunique() if 'article_id' in plan_df.columns else 0
    
    # Route distribution
    if 'route_label' in plan_df.columns:
        kpis.route_distribution = plan_df['route_label'].value_counts().to_dict()
    
    # ========== QUALITY CHECKS ==========
    
    kpis.overlaps = _count_overlaps(plan_df)
    kpis.precedence_violations = _count_precedence_violations(plan_df)
    
    return kpis


def _standardize_plan_columns(plan_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for consistent processing."""
    df = plan_df.copy()
    
    # Map common column variations
    column_map = {
        'start_min': 'start_time',
        'end_min': 'end_time',
        'operation_id': 'operation_id',
        'op_id': 'operation_id',
    }
    
    for old, new in column_map.items():
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


def _count_overlaps(plan_df: pd.DataFrame) -> int:
    """Count overlapping operations on same machine."""
    overlaps = 0
    
    for machine_id, machine_ops in plan_df.groupby('machine_id'):
        sorted_ops = machine_ops.sort_values('start_time')
        times = sorted_ops[['start_time', 'end_time']].values
        
        for i in range(len(times) - 1):
            if times[i][1] > times[i+1][0]:  # end > next start
                overlaps += 1
    
    return overlaps


def _count_precedence_violations(plan_df: pd.DataFrame) -> int:
    """Count precedence constraint violations within orders."""
    violations = 0
    
    if 'op_seq' not in plan_df.columns:
        return 0
    
    for order_id, order_ops in plan_df.groupby('order_id'):
        sorted_ops = order_ops.sort_values('op_seq')
        times = sorted_ops[['start_time', 'end_time']].values
        
        for i in range(len(times) - 1):
            if times[i][1] > times[i+1][0]:  # predecessor ends after successor starts
                violations += 1
    
    return violations


# ============================================================
# PLAN COMPARISON
# ============================================================

@dataclass
class PlanComparison:
    """Comparison between two plans."""
    
    # Absolute differences
    makespan_delta_hours: float = 0.0
    tardiness_delta_hours: float = 0.0
    setup_delta_hours: float = 0.0
    otd_delta_pct: float = 0.0
    utilization_delta_pct: float = 0.0
    
    # Percentage improvements (positive = plan_b better)
    makespan_improvement_pct: float = 0.0
    tardiness_improvement_pct: float = 0.0
    setup_improvement_pct: float = 0.0
    
    # Which plan is better for each metric
    better_makespan: str = ""  # "A" or "B"
    better_tardiness: str = ""
    better_setup: str = ""
    better_otd: str = ""
    
    # Overall recommendation
    recommendation: str = ""
    confidence: float = 0.0
    
    # Raw KPIs
    kpis_a: Optional[PlanKPIs] = None
    kpis_b: Optional[PlanKPIs] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            'makespan_delta_hours': round(self.makespan_delta_hours, 2),
            'tardiness_delta_hours': round(self.tardiness_delta_hours, 2),
            'setup_delta_hours': round(self.setup_delta_hours, 2),
            'otd_delta_pct': round(self.otd_delta_pct, 1),
            'utilization_delta_pct': round(self.utilization_delta_pct, 1),
            'makespan_improvement_pct': round(self.makespan_improvement_pct, 1),
            'tardiness_improvement_pct': round(self.tardiness_improvement_pct, 1),
            'setup_improvement_pct': round(self.setup_improvement_pct, 1),
            'better_makespan': self.better_makespan,
            'better_tardiness': self.better_tardiness,
            'better_setup': self.better_setup,
            'better_otd': self.better_otd,
            'recommendation': self.recommendation,
            'confidence': round(self.confidence, 2),
            'kpis_a': self.kpis_a.to_dict() if self.kpis_a else None,
            'kpis_b': self.kpis_b.to_dict() if self.kpis_b else None,
        }


def compare_plans(
    plan_a: pd.DataFrame,
    plan_b: pd.DataFrame,
    orders_df: Optional[pd.DataFrame] = None,
    setup_matrix: Optional[Dict] = None,
    weights: Optional[Dict[str, float]] = None
) -> PlanComparison:
    """
    Compare two production plans.
    
    Args:
        plan_a: Baseline plan (usually heuristic)
        plan_b: Alternative plan (usually optimized)
        orders_df: Orders with due dates
        setup_matrix: Setup time matrix
        weights: Weights for overall comparison (makespan, tardiness, setup, otd)
    
    Returns:
        PlanComparison with detailed metrics
    
    TODO[R&D]: Add statistical significance tests:
    - Permutation test for small differences
    - Bootstrap confidence intervals
    - Multiple comparison correction
    """
    # Default weights (can be tuned based on factory priorities)
    if weights is None:
        weights = {
            'makespan': 0.3,
            'tardiness': 0.3,
            'setup': 0.2,
            'otd': 0.2,
        }
    
    # Compute KPIs for both plans
    kpis_a = compute_plan_kpis(plan_a, orders_df, setup_matrix)
    kpis_b = compute_plan_kpis(plan_b, orders_df, setup_matrix)
    
    comparison = PlanComparison(kpis_a=kpis_a, kpis_b=kpis_b)
    
    # ========== ABSOLUTE DIFFERENCES ==========
    
    comparison.makespan_delta_hours = kpis_b.makespan_hours - kpis_a.makespan_hours
    comparison.tardiness_delta_hours = kpis_b.total_tardiness_hours - kpis_a.total_tardiness_hours
    comparison.setup_delta_hours = kpis_b.total_setup_hours - kpis_a.total_setup_hours
    comparison.otd_delta_pct = kpis_b.otd_percent - kpis_a.otd_percent
    comparison.utilization_delta_pct = kpis_b.avg_utilization_pct - kpis_a.avg_utilization_pct
    
    # ========== PERCENTAGE IMPROVEMENTS ==========
    # Positive = B is better (for minimization objectives: A > B means B is better)
    
    if kpis_a.makespan_hours > 0:
        comparison.makespan_improvement_pct = (
            (kpis_a.makespan_hours - kpis_b.makespan_hours) / kpis_a.makespan_hours * 100
        )
    
    if kpis_a.total_tardiness_hours > 0:
        comparison.tardiness_improvement_pct = (
            (kpis_a.total_tardiness_hours - kpis_b.total_tardiness_hours) / kpis_a.total_tardiness_hours * 100
        )
    
    if kpis_a.total_setup_hours > 0:
        comparison.setup_improvement_pct = (
            (kpis_a.total_setup_hours - kpis_b.total_setup_hours) / kpis_a.total_setup_hours * 100
        )
    
    # ========== WHICH IS BETTER ==========
    
    comparison.better_makespan = "B" if kpis_b.makespan_hours < kpis_a.makespan_hours else "A"
    comparison.better_tardiness = "B" if kpis_b.total_tardiness_hours < kpis_a.total_tardiness_hours else "A"
    comparison.better_setup = "B" if kpis_b.total_setup_hours < kpis_a.total_setup_hours else "A"
    comparison.better_otd = "B" if kpis_b.otd_percent > kpis_a.otd_percent else "A"  # Higher is better
    
    # ========== OVERALL RECOMMENDATION ==========
    
    # Compute weighted score (normalize to 0-1 scale where 1 = B is better)
    scores = []
    
    # Makespan: lower is better, so positive improvement = B better
    makespan_score = 0.5 + (comparison.makespan_improvement_pct / 200)  # Map -100% to 100% -> 0 to 1
    makespan_score = max(0, min(1, makespan_score))
    scores.append(('makespan', weights['makespan'], makespan_score))
    
    # Tardiness
    tardiness_score = 0.5 + (comparison.tardiness_improvement_pct / 200)
    tardiness_score = max(0, min(1, tardiness_score))
    scores.append(('tardiness', weights['tardiness'], tardiness_score))
    
    # Setup
    setup_score = 0.5 + (comparison.setup_improvement_pct / 200)
    setup_score = max(0, min(1, setup_score))
    scores.append(('setup', weights['setup'], setup_score))
    
    # OTD: higher is better
    otd_score = 0.5 + (comparison.otd_delta_pct / 200)
    otd_score = max(0, min(1, otd_score))
    scores.append(('otd', weights['otd'], otd_score))
    
    # Weighted average
    total_weight = sum(w for _, w, _ in scores)
    weighted_score = sum(w * s for _, w, s in scores) / total_weight if total_weight > 0 else 0.5
    
    if weighted_score > 0.55:
        comparison.recommendation = "Plano B (otimizado) é melhor"
        comparison.confidence = (weighted_score - 0.5) * 2
    elif weighted_score < 0.45:
        comparison.recommendation = "Plano A (baseline) é melhor"
        comparison.confidence = (0.5 - weighted_score) * 2
    else:
        comparison.recommendation = "Planos são equivalentes"
        comparison.confidence = 1 - abs(weighted_score - 0.5) * 4
    
    return comparison


# ============================================================
# PLAN EVALUATOR CLASS
# ============================================================

class PlanEvaluator:
    """
    High-level plan evaluation and comparison utility.
    
    Provides:
    - Multi-plan comparison
    - Sensitivity analysis
    - Quality reports
    
    TODO[R&D]: Implement:
    - Robustness analysis (how sensitive to parameter changes)
    - Scenario analysis (performance under demand uncertainty)
    - Learning from historical actual vs planned
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
        self._history: List[Tuple[str, PlanKPIs]] = []
    
    def evaluate(self, plan_df: pd.DataFrame, plan_name: str = "plan") -> PlanKPIs:
        """Evaluate a single plan and store in history."""
        kpis = compute_plan_kpis(
            plan_df, 
            self.orders_df, 
            self.setup_matrix,
            self.horizon_hours
        )
        self._history.append((plan_name, kpis))
        return kpis
    
    def compare_with_baseline(
        self, 
        plan_df: pd.DataFrame,
        baseline_df: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> PlanComparison:
        """Compare plan against baseline."""
        return compare_plans(
            baseline_df,
            plan_df,
            self.orders_df,
            self.setup_matrix,
            weights
        )
    
    def rank_plans(self, plans: Dict[str, pd.DataFrame], metric: str = "makespan") -> List[Tuple[str, float]]:
        """
        Rank multiple plans by a specific metric.
        
        Args:
            plans: Dict plan_name -> plan_df
            metric: Metric to rank by (makespan, tardiness, setup, otd)
        
        Returns:
            List of (plan_name, metric_value) sorted by metric
        """
        results = []
        
        for name, plan_df in plans.items():
            kpis = compute_plan_kpis(plan_df, self.orders_df, self.setup_matrix)
            
            if metric == "makespan":
                value = kpis.makespan_hours
            elif metric == "tardiness":
                value = kpis.total_tardiness_hours
            elif metric == "setup":
                value = kpis.total_setup_hours
            elif metric == "otd":
                value = -kpis.otd_percent  # Negate for consistent sorting (lower = better)
            else:
                value = kpis.makespan_hours
            
            results.append((name, value))
        
        # Sort ascending (lower is better for all except OTD which we negated)
        results.sort(key=lambda x: x[1])
        
        # Restore OTD sign
        if metric == "otd":
            results = [(name, -value) for name, value in results]
        
        return results
    
    def generate_report(self, plan_df: pd.DataFrame, plan_name: str = "Plan") -> str:
        """Generate human-readable evaluation report."""
        kpis = compute_plan_kpis(plan_df, self.orders_df, self.setup_matrix, self.horizon_hours)
        
        report = f"""
═══════════════════════════════════════════════════════════════
                    RELATÓRIO DE AVALIAÇÃO DO PLANO
                           {plan_name}
═══════════════════════════════════════════════════════════════

📊 MÉTRICAS DE TEMPO
────────────────────
  Makespan:              {kpis.makespan_hours:.1f} horas
  Tempo de Processamento: {kpis.total_processing_hours:.1f} horas
  
📦 MÉTRICAS DE ENTREGA
─────────────────────
  OTD (On-Time Delivery): {kpis.otd_percent:.1f}%
  Encomendas a tempo:     {kpis.early_orders + (kpis.total_orders - kpis.late_orders)}
  Encomendas atrasadas:   {kpis.late_orders}
  Atraso máximo:          {kpis.max_tardiness_hours:.1f} horas
  Atraso total:           {kpis.total_tardiness_hours:.1f} horas

🔧 MÉTRICAS DE SETUP
────────────────────
  Tempo total de setup:   {kpis.total_setup_hours:.1f} horas
  Número de setups:       {kpis.setup_count}
  Setup médio:            {kpis.avg_setup_min:.1f} minutos

🏭 MÉTRICAS DE RECURSOS
──────────────────────
  Utilização média:       {kpis.avg_utilization_pct:.1f}%
  Utilização máxima:      {kpis.max_utilization_pct:.1f}%
  Desequilíbrio de carga: {kpis.load_imbalance:.2f}
  Máquina gargalo:        {kpis.bottleneck_machine}
  Máquinas utilizadas:    {kpis.machines_used}

📈 VOLUME
─────────
  Total de operações:     {kpis.total_operations}
  Total de encomendas:    {kpis.total_orders}
  Total de artigos:       {kpis.total_articles}
  Distribuição de rotas:  {kpis.route_distribution}

⚠️ QUALIDADE
────────────
  Sobreposições:          {kpis.overlaps}
  Violações precedência:  {kpis.precedence_violations}

═══════════════════════════════════════════════════════════════
"""
        return report
    
    def get_history(self) -> pd.DataFrame:
        """Get evaluation history as DataFrame."""
        if not self._history:
            return pd.DataFrame()
        
        records = []
        for name, kpis in self._history:
            record = kpis.to_dict()
            record['plan_name'] = name
            records.append(record)
        
        return pd.DataFrame(records)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def quick_evaluate(plan_df: pd.DataFrame) -> Dict[str, Any]:
    """Quick evaluation with minimal parameters."""
    kpis = compute_plan_kpis(plan_df)
    return kpis.to_dict()


def quick_compare(plan_a: pd.DataFrame, plan_b: pd.DataFrame) -> Dict[str, Any]:
    """Quick comparison with minimal parameters."""
    comparison = compare_plans(plan_a, plan_b)
    return comparison.to_dict()



