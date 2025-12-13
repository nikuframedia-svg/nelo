"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    COMPARISON ENGINE — Scenario Metrics & Analysis
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Computes and compares metrics between planning scenarios.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class MachineMetrics:
    """Metrics for a single machine."""
    machine_id: str
    utilization_pct: float = 0.0
    total_processing_hours: float = 0.0
    total_setup_hours: float = 0.0
    total_idle_hours: float = 0.0
    num_operations: int = 0
    is_bottleneck: bool = False
    queue_time_avg_hours: float = 0.0


@dataclass
class ComparisonMetrics:
    """
    Comprehensive metrics for a planning scenario.
    """
    scenario_name: str
    scenario_description: str = ""
    
    # Time metrics
    makespan_hours: float = 0.0
    lead_time_avg_days: float = 0.0
    lead_time_median_days: float = 0.0
    lead_time_p90_days: float = 0.0
    
    # Throughput
    throughput_units_total: float = 0.0
    throughput_units_per_week: float = 0.0
    throughput_orders_total: int = 0
    throughput_orders_per_week: float = 0.0
    
    # On-time delivery
    orders_on_time: int = 0
    orders_late: int = 0
    otd_pct: float = 0.0
    total_tardiness_hours: float = 0.0
    max_tardiness_hours: float = 0.0
    
    # Setup
    total_setup_hours: float = 0.0
    num_setups: int = 0
    avg_setup_time_min: float = 0.0
    
    # Machine utilization
    avg_utilization_pct: float = 0.0
    max_utilization_pct: float = 0.0
    min_utilization_pct: float = 0.0
    bottleneck_machine: Optional[str] = None
    bottleneck_utilization: float = 0.0
    
    # Per-machine metrics
    machine_metrics: Dict[str, MachineMetrics] = field(default_factory=dict)
    
    # Metadata
    num_machines: int = 0
    num_orders: int = 0
    horizon_days: int = 0
    computed_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "scenario_description": self.scenario_description,
            "makespan_hours": round(self.makespan_hours, 1),
            "lead_time_avg_days": round(self.lead_time_avg_days, 2),
            "lead_time_median_days": round(self.lead_time_median_days, 2),
            "lead_time_p90_days": round(self.lead_time_p90_days, 2),
            "throughput_units_total": round(self.throughput_units_total, 0),
            "throughput_units_per_week": round(self.throughput_units_per_week, 1),
            "throughput_orders_total": self.throughput_orders_total,
            "throughput_orders_per_week": round(self.throughput_orders_per_week, 2),
            "orders_on_time": self.orders_on_time,
            "orders_late": self.orders_late,
            "otd_pct": round(self.otd_pct, 1),
            "total_tardiness_hours": round(self.total_tardiness_hours, 1),
            "max_tardiness_hours": round(self.max_tardiness_hours, 1),
            "total_setup_hours": round(self.total_setup_hours, 1),
            "num_setups": self.num_setups,
            "avg_setup_time_min": round(self.avg_setup_time_min, 1),
            "avg_utilization_pct": round(self.avg_utilization_pct, 1),
            "max_utilization_pct": round(self.max_utilization_pct, 1),
            "min_utilization_pct": round(self.min_utilization_pct, 1),
            "bottleneck_machine": self.bottleneck_machine,
            "bottleneck_utilization": round(self.bottleneck_utilization, 1),
            "num_machines": self.num_machines,
            "num_orders": self.num_orders,
            "horizon_days": self.horizon_days,
            "machine_metrics": {
                m_id: {
                    "utilization_pct": round(m.utilization_pct, 1),
                    "processing_hours": round(m.total_processing_hours, 1),
                    "setup_hours": round(m.total_setup_hours, 1),
                    "idle_hours": round(m.total_idle_hours, 1),
                    "operations": m.num_operations,
                    "is_bottleneck": m.is_bottleneck,
                }
                for m_id, m in self.machine_metrics.items()
            },
        }


@dataclass
class MetricDelta:
    """Delta between two metric values."""
    metric_name: str
    baseline_value: float
    scenario_value: float
    absolute_delta: float = 0.0
    percent_delta: float = 0.0
    is_improvement: bool = False
    significance: str = "low"  # low, medium, high
    
    def compute(self, higher_is_better: bool = False):
        self.absolute_delta = self.scenario_value - self.baseline_value
        
        if self.baseline_value != 0:
            self.percent_delta = (self.absolute_delta / abs(self.baseline_value)) * 100
        else:
            self.percent_delta = 100.0 if self.scenario_value != 0 else 0.0
        
        if higher_is_better:
            self.is_improvement = self.absolute_delta > 0
        else:
            self.is_improvement = self.absolute_delta < 0
        
        # Significance based on percent change
        abs_pct = abs(self.percent_delta)
        if abs_pct >= 20:
            self.significance = "high"
        elif abs_pct >= 10:
            self.significance = "medium"
        else:
            self.significance = "low"


@dataclass
class ScenarioComparison:
    """
    Complete comparison between two scenarios.
    """
    baseline: ComparisonMetrics
    scenario: ComparisonMetrics
    
    # Computed deltas
    deltas: Dict[str, MetricDelta] = field(default_factory=dict)
    
    # Analysis results
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Machine-level changes
    machine_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Overall assessment
    overall_improvement: bool = False
    improvement_score: float = 0.0  # -100 to +100
    
    def compute_deltas(self):
        """Compute all metric deltas."""
        b = self.baseline
        s = self.scenario
        
        # Define metrics and whether higher is better
        metrics_config = [
            ("makespan_hours", b.makespan_hours, s.makespan_hours, False),
            ("lead_time_avg_days", b.lead_time_avg_days, s.lead_time_avg_days, False),
            ("throughput_units_per_week", b.throughput_units_per_week, s.throughput_units_per_week, True),
            ("orders_late", b.orders_late, s.orders_late, False),
            ("otd_pct", b.otd_pct, s.otd_pct, True),
            ("total_tardiness_hours", b.total_tardiness_hours, s.total_tardiness_hours, False),
            ("total_setup_hours", b.total_setup_hours, s.total_setup_hours, False),
            ("avg_utilization_pct", b.avg_utilization_pct, s.avg_utilization_pct, True),
            ("bottleneck_utilization", b.bottleneck_utilization, s.bottleneck_utilization, False),
        ]
        
        for name, base_val, scen_val, higher_better in metrics_config:
            delta = MetricDelta(
                metric_name=name,
                baseline_value=base_val,
                scenario_value=scen_val,
            )
            delta.compute(higher_is_better=higher_better)
            self.deltas[name] = delta
        
        # Compute machine-level changes
        self._compute_machine_changes()
        
        # Analyze and generate insights
        self._analyze()
    
    def _compute_machine_changes(self):
        """Compare machine metrics between scenarios."""
        baseline_machines = set(self.baseline.machine_metrics.keys())
        scenario_machines = set(self.scenario.machine_metrics.keys())
        
        all_machines = baseline_machines | scenario_machines
        
        for m_id in all_machines:
            b_metrics = self.baseline.machine_metrics.get(m_id)
            s_metrics = self.scenario.machine_metrics.get(m_id)
            
            change = {
                "machine_id": m_id,
                "in_baseline": m_id in baseline_machines,
                "in_scenario": m_id in scenario_machines,
                "is_new": m_id not in baseline_machines and m_id in scenario_machines,
                "is_removed": m_id in baseline_machines and m_id not in scenario_machines,
            }
            
            if b_metrics and s_metrics:
                util_delta = s_metrics.utilization_pct - b_metrics.utilization_pct
                change.update({
                    "baseline_utilization": b_metrics.utilization_pct,
                    "scenario_utilization": s_metrics.utilization_pct,
                    "utilization_delta": util_delta,
                    "was_bottleneck": b_metrics.is_bottleneck,
                    "is_bottleneck": s_metrics.is_bottleneck,
                    "bottleneck_eliminated": b_metrics.is_bottleneck and not s_metrics.is_bottleneck,
                    "became_bottleneck": not b_metrics.is_bottleneck and s_metrics.is_bottleneck,
                })
            elif s_metrics:  # New machine
                change.update({
                    "baseline_utilization": 0,
                    "scenario_utilization": s_metrics.utilization_pct,
                    "utilization_delta": s_metrics.utilization_pct,
                    "is_bottleneck": s_metrics.is_bottleneck,
                })
            
            self.machine_changes.append(change)
    
    def _analyze(self):
        """Generate strengths, weaknesses, and recommendations."""
        self.strengths = []
        self.weaknesses = []
        self.recommendations = []
        
        # Analyze each delta
        for name, delta in self.deltas.items():
            if delta.is_improvement and delta.significance in ["medium", "high"]:
                self.strengths.append(self._format_strength(name, delta))
            elif not delta.is_improvement and delta.significance in ["medium", "high"]:
                self.weaknesses.append(self._format_weakness(name, delta))
        
        # Machine-specific insights
        for mc in self.machine_changes:
            if mc.get("bottleneck_eliminated"):
                self.strengths.append(
                    f"Gargalo eliminado na máquina {mc['machine_id']} "
                    f"(utilização reduziu de {mc['baseline_utilization']:.0f}% para {mc['scenario_utilization']:.0f}%)"
                )
            
            if mc.get("became_bottleneck"):
                self.weaknesses.append(
                    f"Máquina {mc['machine_id']} tornou-se novo gargalo "
                    f"(utilização aumentou para {mc['scenario_utilization']:.0f}%)"
                )
            
            if mc.get("is_new"):
                self.strengths.append(
                    f"Nova máquina {mc['machine_id']} adicionada "
                    f"(utilização: {mc.get('scenario_utilization', 0):.0f}%)"
                )
            
            # High utilization warning
            if mc.get("scenario_utilization", 0) > 90 and not mc.get("is_bottleneck"):
                self.weaknesses.append(
                    f"Máquina {mc['machine_id']} com utilização muito alta "
                    f"({mc['scenario_utilization']:.0f}%) - risco de se tornar gargalo"
                )
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Overall assessment
        improvement_count = sum(1 for d in self.deltas.values() if d.is_improvement)
        degradation_count = len(self.deltas) - improvement_count
        
        self.overall_improvement = improvement_count > degradation_count
        
        # Score: weighted average of improvements
        weights = {
            "orders_late": 3.0,
            "otd_pct": 2.5,
            "lead_time_avg_days": 2.0,
            "makespan_hours": 1.5,
            "throughput_units_per_week": 2.0,
            "bottleneck_utilization": 1.0,
            "total_setup_hours": 0.5,
            "avg_utilization_pct": 1.0,
            "total_tardiness_hours": 2.0,
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for name, delta in self.deltas.items():
            w = weights.get(name, 1.0)
            # Normalize percent delta to -100 to +100 range
            normalized = max(-100, min(100, delta.percent_delta))
            if not delta.is_improvement:
                normalized = -abs(normalized)
            else:
                normalized = abs(normalized)
            
            weighted_score += w * normalized
            total_weight += w
        
        self.improvement_score = weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _format_strength(self, metric_name: str, delta: MetricDelta) -> str:
        labels = {
            "makespan_hours": "Makespan",
            "lead_time_avg_days": "Lead time médio",
            "throughput_units_per_week": "Throughput semanal",
            "orders_late": "Ordens atrasadas",
            "otd_pct": "OTD",
            "total_tardiness_hours": "Atrasos totais",
            "total_setup_hours": "Tempo de setup",
            "avg_utilization_pct": "Utilização média",
            "bottleneck_utilization": "Utilização do gargalo",
        }
        label = labels.get(metric_name, metric_name)
        
        direction = "aumentou" if delta.absolute_delta > 0 else "reduziu"
        
        return f"{label} {direction} {abs(delta.percent_delta):.1f}% ({delta.baseline_value:.1f} → {delta.scenario_value:.1f})"
    
    def _format_weakness(self, metric_name: str, delta: MetricDelta) -> str:
        return self._format_strength(metric_name, delta)  # Same format, context determines meaning
    
    def _generate_recommendations(self):
        """Generate actionable recommendations."""
        # Based on weaknesses
        for mc in self.machine_changes:
            if mc.get("scenario_utilization", 0) > 85:
                self.recommendations.append(
                    f"Considere aumentar capacidade da máquina {mc['machine_id']} "
                    f"(turno extra ou máquina adicional)"
                )
        
        if self.deltas.get("orders_late") and self.deltas["orders_late"].scenario_value > 0:
            self.recommendations.append(
                "Existem encomendas atrasadas. Reveja prioridades ou "
                "considere horas extra para recuperar atrasos."
            )
        
        if self.deltas.get("total_setup_hours") and not self.deltas["total_setup_hours"].is_improvement:
            self.recommendations.append(
                "Tempo de setup aumentou. Considere agrupar ordens por família "
                "de produto para minimizar trocas."
            )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "scenario": self.scenario.to_dict(),
            "deltas": {
                name: {
                    "baseline": d.baseline_value,
                    "scenario": d.scenario_value,
                    "absolute": round(d.absolute_delta, 2),
                    "percent": round(d.percent_delta, 1),
                    "is_improvement": d.is_improvement,
                    "significance": d.significance,
                }
                for name, d in self.deltas.items()
            },
            "machine_changes": self.machine_changes,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendations": self.recommendations,
            "overall_improvement": self.overall_improvement,
            "improvement_score": round(self.improvement_score, 1),
        }


def compute_scenario_metrics(
    plan_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    scenario_name: str = "Cenário",
    scenario_description: str = "",
) -> ComparisonMetrics:
    """
    Compute comprehensive metrics for a planning scenario.
    
    Args:
        plan_df: Production plan DataFrame
        orders_df: Orders DataFrame with due dates
        machines_df: Machines DataFrame
        scenario_name: Name for this scenario
        scenario_description: Description of scenario changes
        
    Returns:
        ComparisonMetrics with all computed values
    """
    metrics = ComparisonMetrics(
        scenario_name=scenario_name,
        scenario_description=scenario_description,
    )
    
    if plan_df.empty:
        return metrics
    
    # Parse dates if needed
    if 'start_time' in plan_df.columns and not pd.api.types.is_datetime64_any_dtype(plan_df['start_time']):
        plan_df = plan_df.copy()
        plan_df['start_time'] = pd.to_datetime(plan_df['start_time'])
        plan_df['end_time'] = pd.to_datetime(plan_df['end_time'])
    
    # Time horizon
    min_time = plan_df['start_time'].min()
    max_time = plan_df['end_time'].max()
    horizon_hours = (max_time - min_time).total_seconds() / 3600
    horizon_days = horizon_hours / 24
    
    metrics.horizon_days = int(horizon_days)
    metrics.makespan_hours = float(horizon_hours)
    
    # Throughput
    metrics.throughput_units_total = float(plan_df['qty'].sum()) if 'qty' in plan_df.columns else 0
    metrics.throughput_orders_total = plan_df['order_id'].nunique() if 'order_id' in plan_df.columns else 0
    
    weeks = max(1, horizon_days / 7)
    metrics.throughput_units_per_week = metrics.throughput_units_total / weeks
    metrics.throughput_orders_per_week = metrics.throughput_orders_total / weeks
    
    # Machine metrics
    machines = plan_df['machine_id'].unique().tolist()
    metrics.num_machines = len(machines)
    
    total_util = 0.0
    max_util = 0.0
    min_util = 100.0
    bottleneck = None
    
    for m_id in machines:
        m_ops = plan_df[plan_df['machine_id'] == m_id]
        
        total_proc = m_ops['duration_min'].sum() / 60 if 'duration_min' in m_ops.columns else 0
        
        # Calculate utilization based on horizon
        util = (total_proc / horizon_hours * 100) if horizon_hours > 0 else 0
        
        mm = MachineMetrics(
            machine_id=m_id,
            utilization_pct=util,
            total_processing_hours=total_proc,
            num_operations=len(m_ops),
        )
        
        metrics.machine_metrics[m_id] = mm
        
        total_util += util
        max_util = max(max_util, util)
        min_util = min(min_util, util)
        
        if util > (bottleneck_utilization := metrics.bottleneck_utilization):
            metrics.bottleneck_machine = m_id
            metrics.bottleneck_utilization = util
            mm.is_bottleneck = True
    
    metrics.avg_utilization_pct = total_util / len(machines) if machines else 0
    metrics.max_utilization_pct = max_util
    metrics.min_utilization_pct = min_util
    
    # Order metrics (on-time delivery)
    if 'order_id' in plan_df.columns and not orders_df.empty:
        orders_df = orders_df.copy()
        if 'due_date' in orders_df.columns:
            orders_df['due_date'] = pd.to_datetime(orders_df['due_date'])
        
        metrics.num_orders = len(orders_df)
        
        # Get completion time per order
        order_completion = plan_df.groupby('order_id')['end_time'].max().reset_index()
        order_completion.columns = ['order_id', 'completion_time']
        
        # Merge with due dates
        merged = orders_df.merge(order_completion, on='order_id', how='left')
        
        if 'due_date' in merged.columns and 'completion_time' in merged.columns:
            merged['is_late'] = merged['completion_time'] > merged['due_date']
            merged['tardiness'] = (merged['completion_time'] - merged['due_date']).apply(
                lambda x: max(0, x.total_seconds() / 3600) if pd.notna(x) else 0
            )
            
            metrics.orders_late = int(merged['is_late'].sum())
            metrics.orders_on_time = metrics.num_orders - metrics.orders_late
            metrics.otd_pct = (metrics.orders_on_time / metrics.num_orders * 100) if metrics.num_orders > 0 else 100
            
            metrics.total_tardiness_hours = float(merged['tardiness'].sum())
            metrics.max_tardiness_hours = float(merged['tardiness'].max())
            
            # Lead time calculation
            if 'release_date' in merged.columns or 'created_at' in merged.columns:
                release_col = 'release_date' if 'release_date' in merged.columns else 'created_at'
                merged[release_col] = pd.to_datetime(merged[release_col])
                merged['lead_time_days'] = (merged['completion_time'] - merged[release_col]).dt.total_seconds() / 86400
            else:
                # Use time from plan start as proxy
                merged['lead_time_days'] = (merged['completion_time'] - min_time).dt.total_seconds() / 86400
            
            metrics.lead_time_avg_days = float(merged['lead_time_days'].mean())
            metrics.lead_time_median_days = float(merged['lead_time_days'].median())
            metrics.lead_time_p90_days = float(merged['lead_time_days'].quantile(0.90))
    
    # Setup metrics
    if 'setup_min' in plan_df.columns:
        metrics.total_setup_hours = float(plan_df['setup_min'].sum() / 60)
        metrics.num_setups = int((plan_df['setup_min'] > 0).sum())
        metrics.avg_setup_time_min = float(plan_df[plan_df['setup_min'] > 0]['setup_min'].mean()) if metrics.num_setups > 0 else 0
    
    return metrics


def compare_scenarios(
    baseline_plan: pd.DataFrame,
    scenario_plan: pd.DataFrame,
    orders_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    baseline_name: str = "Plano Base",
    scenario_name: str = "Novo Cenário",
    scenario_description: str = "",
) -> ScenarioComparison:
    """
    Compare two planning scenarios.
    
    Args:
        baseline_plan: Baseline production plan
        scenario_plan: New scenario production plan
        orders_df: Orders DataFrame
        machines_df: Machines DataFrame
        baseline_name: Name for baseline
        scenario_name: Name for scenario
        scenario_description: Description of changes in scenario
        
    Returns:
        ScenarioComparison with full analysis
    """
    baseline_metrics = compute_scenario_metrics(
        baseline_plan, orders_df, machines_df,
        scenario_name=baseline_name,
    )
    
    scenario_metrics = compute_scenario_metrics(
        scenario_plan, orders_df, machines_df,
        scenario_name=scenario_name,
        scenario_description=scenario_description,
    )
    
    comparison = ScenarioComparison(
        baseline=baseline_metrics,
        scenario=scenario_metrics,
    )
    comparison.compute_deltas()
    
    return comparison



