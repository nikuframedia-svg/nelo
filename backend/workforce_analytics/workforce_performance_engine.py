"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — WORKFORCE PERFORMANCE ENGINE
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Comprehensive worker performance metrics with mathematical rigor and SNR integration.

PERFORMANCE METRICS
═══════════════════

1. PRODUCTIVITY (P)
   ────────────────
   
       P = Σ units_processed / Σ time_worked
   
   Units depend on operation type (pieces, kg, m², etc.)

2. EFFICIENCY (E)
   ────────────────
   
       E = P_worker / P_reference
   
   where P_reference can be:
   - Machine average
   - Team average
   - Standard time (from routing)

3. SATURATION (S)
   ────────────────
   
       S = time_occupied / time_available
   
   - S < 0.7: Underutilized
   - S ∈ [0.7, 0.9]: Optimal range
   - S > 0.9: Potentially overloaded

4. SKILL SCORE (σ)
   ────────────────
   
       σ = Σ w_op · success_rate_op / Σ w_op
   
   where:
   - w_op = weight for operation type (complexity)
   - success_rate_op = successful_completions / total_completions

5. LEARNING CURVE
   ────────────────
   
   Wright's Model (exponential):
       
       y(t) = a - b · exp(-c·t)
   
   where:
       a = asymptotic productivity (maximum achievable)
       b = initial gap = a - y(0)
       c = learning rate (higher = faster learning)
   
   Alternative: Power Law (Crawford)
       
       y(n) = y₁ · n^(log₂(r))
   
   where r = learning rate (typically 0.7-0.9)

6. SNR PERFORMANCE
   ────────────────
   
       SNR = Var(trend_component) / Var(residual)
   
   High SNR indicates consistent, predictable worker.
   Low SNR indicates high variability (may need investigation).

R&D / SIFIDE: WP6 - Workforce Intelligence
──────────────────────────────────────────
- Hypothesis H6.1: Learning curves explain >80% productivity variance
- Experiment E6.1: Fit learning curves to historical data
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional scipy for curve fitting
try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    curve_fit = None

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

# Saturation thresholds
SATURATION_LOW = 0.5
SATURATION_OPTIMAL_MIN = 0.7
SATURATION_OPTIMAL_MAX = 0.9
SATURATION_HIGH = 0.95

# Skill score weights by operation complexity
DEFAULT_COMPLEXITY_WEIGHTS = {
    "simple": 1.0,
    "medium": 1.5,
    "complex": 2.5,
    "critical": 4.0,
}

# SNR thresholds
SNR_EXCELLENT = 10.0
SNR_GOOD = 5.0
SNR_FAIR = 2.0
SNR_POOR = 1.0


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════════════════════════════

class PerformanceLevel(str, Enum):
    """Performance classification."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    BELOW_AVERAGE = "below_average"
    NEEDS_IMPROVEMENT = "needs_improvement"


class SaturationLevel(str, Enum):
    """Saturation classification."""
    UNDERUTILIZED = "underutilized"
    LOW = "low"
    OPTIMAL = "optimal"
    HIGH = "high"
    OVERLOADED = "overloaded"


@dataclass
class LearningCurveParams:
    """
    Parameters for the learning curve model.
    
    Model: y(t) = a - b * exp(-c * t)
    
    Attributes:
        a: Asymptotic productivity (maximum achievable)
        b: Initial gap (a - y(0))
        c: Learning rate (higher = faster learning)
        r_squared: Goodness of fit
        time_to_90pct: Days to reach 90% of asymptote
    """
    a: float  # Asymptotic productivity
    b: float  # Initial gap
    c: float  # Learning rate
    r_squared: float = 0.0
    time_to_90pct: Optional[float] = None
    
    def predict(self, t: float) -> float:
        """Predict productivity at time t."""
        return self.a - self.b * math.exp(-self.c * t)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'a_asymptote': round(self.a, 4),
            'b_initial_gap': round(self.b, 4),
            'c_learning_rate': round(self.c, 6),
            'r_squared': round(self.r_squared, 4),
            'time_to_90pct_days': round(self.time_to_90pct, 1) if self.time_to_90pct else None,
            'model': 'y(t) = a - b × exp(-c × t)',
        }


@dataclass
class WorkerMetrics:
    """
    Computed metrics for a worker.
    """
    worker_id: str
    worker_name: Optional[str] = None
    
    # Core metrics
    productivity: float = 0.0  # units/hour
    efficiency: float = 1.0    # ratio vs reference
    saturation: float = 0.0    # occupied/available time
    skill_score: float = 0.0   # 0-1 skill rating
    
    # Time metrics
    total_time_hours: float = 0.0
    occupied_time_hours: float = 0.0
    available_time_hours: float = 0.0
    
    # Production metrics
    total_units: float = 0.0
    total_operations: int = 0
    successful_operations: int = 0
    
    # SNR and quality
    snr_performance: float = 1.0
    snr_level: str = "FAIR"
    consistency_score: float = 0.5
    
    # Classifications
    performance_level: str = "average"
    saturation_level: str = "optimal"
    
    # Learning curve
    learning_curve: Optional[LearningCurveParams] = None
    
    # Metadata
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'worker_id': self.worker_id,
            'worker_name': self.worker_name,
            'productivity': float(round(self.productivity, 2)),
            'efficiency': float(round(self.efficiency, 3)),
            'saturation': float(round(self.saturation, 3)),
            'skill_score': float(round(self.skill_score, 3)),
            'total_time_hours': float(round(self.total_time_hours, 2)),
            'occupied_time_hours': float(round(self.occupied_time_hours, 2)),
            'total_units': float(round(self.total_units, 1)),
            'total_operations': int(self.total_operations),
            'successful_operations': int(self.successful_operations),
            'snr_performance': float(round(self.snr_performance, 2)),
            'snr_level': self.snr_level,
            'consistency_score': float(round(self.consistency_score, 3)),
            'performance_level': self.performance_level,
            'saturation_level': self.saturation_level,
            'learning_curve': self.learning_curve.to_dict() if self.learning_curve else None,
            'period_start': self.period_start,
            'period_end': self.period_end,
        }


@dataclass
class WorkerPerformance:
    """
    Complete performance profile for a worker.
    """
    worker_id: str
    metrics: WorkerMetrics
    
    # Historical data
    productivity_history: List[float] = field(default_factory=list)
    efficiency_history: List[float] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    
    # By operation type
    metrics_by_operation: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Qualifications
    qualified_operations: List[str] = field(default_factory=list)
    qualified_machines: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'worker_id': self.worker_id,
            'metrics': self.metrics.to_dict(),
            'productivity_history': [float(round(p, 2)) for p in self.productivity_history[-30:]],
            'efficiency_history': [float(round(e, 3)) for e in self.efficiency_history[-30:]],
            'dates': [str(d) for d in self.dates[-30:]],
            'metrics_by_operation': self.metrics_by_operation,
            'qualified_operations': list(self.qualified_operations),
            'qualified_machines': list(self.qualified_machines),
            'recommendations': list(self.recommendations),
        }


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# LEARNING CURVE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def _exponential_learning_curve(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Exponential learning curve model.
    
    y(t) = a - b * exp(-c * t)
    
    Args:
        t: Time (days since start)
        a: Asymptotic productivity
        b: Initial gap
        c: Learning rate
    
    Returns:
        Predicted productivity
    """
    return a - b * np.exp(-c * t)


def compute_learning_curve(
    t: float,
    params: LearningCurveParams
) -> float:
    """
    Compute productivity at time t using learning curve.
    
    Args:
        t: Time (days since start)
        params: Learning curve parameters
    
    Returns:
        Predicted productivity
    """
    return params.predict(t)


def fit_learning_curve(
    times: np.ndarray,
    productivities: np.ndarray,
    initial_guess: Optional[Tuple[float, float, float]] = None
) -> LearningCurveParams:
    """
    Fit learning curve to historical data.
    
    Uses non-linear least squares to fit:
        y(t) = a - b * exp(-c * t)
    
    Args:
        times: Array of times (days since start)
        productivities: Array of productivity values
    
    Returns:
        Fitted LearningCurveParams
    """
    times = np.asarray(times, dtype=np.float64)
    productivities = np.asarray(productivities, dtype=np.float64)
    
    # Remove NaN
    mask = ~(np.isnan(times) | np.isnan(productivities))
    times = times[mask]
    productivities = productivities[mask]
    
    if len(times) < 5:
        # Not enough data - return simple average
        avg_prod = np.mean(productivities) if len(productivities) > 0 else 1.0
        return LearningCurveParams(a=avg_prod, b=0.0, c=0.1, r_squared=0.0)
    
    # If scipy not available, use simple approximation
    if not HAS_SCIPY or curve_fit is None:
        return _fit_learning_curve_simple(times, productivities)
    
    # Initial guess
    if initial_guess is None:
        a_init = np.max(productivities) * 1.1
        b_init = a_init - np.min(productivities)
        c_init = 0.05
        initial_guess = (a_init, b_init, c_init)
    
    try:
        # Fit curve
        popt, _ = curve_fit(
            _exponential_learning_curve,
            times,
            productivities,
            p0=initial_guess,
            bounds=(
                [0, 0, 0.001],  # Lower bounds
                [np.inf, np.inf, 1.0]  # Upper bounds
            ),
            maxfev=5000
        )
        
        a, b, c = popt
        
        # Compute R²
        y_pred = _exponential_learning_curve(times, a, b, c)
        ss_res = np.sum((productivities - y_pred) ** 2)
        ss_tot = np.sum((productivities - np.mean(productivities)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Time to 90% of asymptote
        if c > 0 and b > 0:
            # y(t) = a - b * exp(-c*t) = 0.9 * a
            # => exp(-c*t) = (a - 0.9*a) / b = 0.1 * a / b
            # => t = -ln(0.1 * a / b) / c
            ratio = 0.1 * a / b
            if ratio > 0:
                time_to_90 = -math.log(ratio) / c
            else:
                time_to_90 = None
        else:
            time_to_90 = None
        
        return LearningCurveParams(
            a=a,
            b=b,
            c=c,
            r_squared=max(0, r_squared),
            time_to_90pct=time_to_90
        )
    
    except Exception as e:
        logger.warning(f"Learning curve fitting failed: {e}")
        avg_prod = np.mean(productivities)
        return LearningCurveParams(a=avg_prod, b=0.0, c=0.1, r_squared=0.0)


def _fit_learning_curve_simple(
    times: np.ndarray,
    productivities: np.ndarray
) -> LearningCurveParams:
    """
    Simple learning curve approximation without scipy.
    
    Uses linear regression on log-transformed data.
    """
    # Estimate asymptote as max + 10%
    a = np.max(productivities) * 1.1
    
    # Initial value
    y0 = productivities[0] if len(productivities) > 0 else a * 0.5
    b = a - y0
    
    # Estimate c from trend
    if len(productivities) > 1 and b > 0:
        # Use simple exponential fit approximation
        # y(t) = a - b*exp(-ct) => (a - y) = b*exp(-ct) => ln((a-y)/b) = -ct
        y_transformed = np.clip((a - productivities) / b, 0.01, 10)
        try:
            slope = np.polyfit(times, np.log(y_transformed), 1)[0]
            c = -slope
        except:
            c = 0.05
    else:
        c = 0.05
    
    c = max(0.001, min(c, 1.0))
    
    # Compute R²
    y_pred = a - b * np.exp(-c * times)
    ss_res = np.sum((productivities - y_pred) ** 2)
    ss_tot = np.sum((productivities - np.mean(productivities)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Time to 90%
    if c > 0 and b > 0:
        ratio = 0.1 * a / b
        if ratio > 0 and ratio < 10:
            time_to_90 = -math.log(ratio) / c
        else:
            time_to_90 = None
    else:
        time_to_90 = None
    
    return LearningCurveParams(
        a=a,
        b=b,
        c=c,
        r_squared=max(0, r_squared),
        time_to_90pct=time_to_90
    )


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# SNR COMPUTATION
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def compute_performance_snr(
    productivities: np.ndarray,
    window: int = 5
) -> Tuple[float, str]:
    """
    Compute Signal-to-Noise Ratio for performance data.
    
    SNR = Var(trend) / Var(residual)
    
    where trend is estimated via moving average.
    
    Args:
        productivities: Array of productivity values
        window: Window size for moving average
    
    Returns:
        (snr_value, snr_level)
    """
    productivities = np.asarray(productivities, dtype=np.float64)
    productivities = productivities[~np.isnan(productivities)]
    
    if len(productivities) < window + 2:
        return 1.0, "FAIR"
    
    # Compute moving average as signal estimate
    kernel = np.ones(window) / window
    signal = np.convolve(productivities, kernel, mode='valid')
    
    # Align and compute residual
    aligned = productivities[window // 2:window // 2 + len(signal)]
    if len(aligned) != len(signal):
        aligned = productivities[:len(signal)]
    
    residual = aligned - signal
    
    var_signal = np.var(signal)
    var_residual = np.var(residual)
    
    if var_residual < 1e-10:
        snr = 100.0 if var_signal > 0 else 1.0
    else:
        snr = var_signal / var_residual
    
    snr = min(max(snr, 0.1), 100.0)
    
    # Classify
    if snr >= SNR_EXCELLENT:
        level = "EXCELLENT"
    elif snr >= SNR_GOOD:
        level = "GOOD"
    elif snr >= SNR_FAIR:
        level = "FAIR"
    else:
        level = "POOR"
    
    return snr, level


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# PERFORMANCE COMPUTATION
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def classify_performance(efficiency: float) -> str:
    """Classify performance level based on efficiency."""
    if efficiency >= 1.3:
        return PerformanceLevel.EXCELLENT.value
    elif efficiency >= 1.1:
        return PerformanceLevel.GOOD.value
    elif efficiency >= 0.9:
        return PerformanceLevel.AVERAGE.value
    elif efficiency >= 0.7:
        return PerformanceLevel.BELOW_AVERAGE.value
    else:
        return PerformanceLevel.NEEDS_IMPROVEMENT.value


def classify_saturation(saturation: float) -> str:
    """Classify saturation level."""
    if saturation < SATURATION_LOW:
        return SaturationLevel.UNDERUTILIZED.value
    elif saturation < SATURATION_OPTIMAL_MIN:
        return SaturationLevel.LOW.value
    elif saturation <= SATURATION_OPTIMAL_MAX:
        return SaturationLevel.OPTIMAL.value
    elif saturation <= SATURATION_HIGH:
        return SaturationLevel.HIGH.value
    else:
        return SaturationLevel.OVERLOADED.value


def compute_skill_score(
    operations_by_type: Dict[str, Dict[str, int]],
    complexity_weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute weighted skill score.
    
    Formula:
        σ = Σ w_op * success_rate_op / Σ w_op
    
    Args:
        operations_by_type: Dict[op_type -> {'total': n, 'success': m}]
        complexity_weights: Dict[complexity_level -> weight]
    
    Returns:
        Skill score in [0, 1]
    """
    weights = complexity_weights or DEFAULT_COMPLEXITY_WEIGHTS
    
    total_weighted = 0.0
    total_weight = 0.0
    
    for op_type, counts in operations_by_type.items():
        total = counts.get('total', 0)
        success = counts.get('success', 0)
        
        if total == 0:
            continue
        
        # Determine complexity (default to medium)
        complexity = counts.get('complexity', 'medium')
        weight = weights.get(complexity, 1.5)
        
        success_rate = success / total
        total_weighted += weight * success_rate
        total_weight += weight
    
    if total_weight == 0:
        return 0.5  # Default neutral score
    
    return total_weighted / total_weight


def compute_worker_performance(
    worker_id: str,
    operations_df: pd.DataFrame,
    reference_productivity: Optional[float] = None,
    available_hours: Optional[float] = None,
    worker_col: str = 'worker_id',
    time_col: str = 'duration_min',
    units_col: str = 'qty',
    date_col: str = 'date',
    op_type_col: str = 'op_code',
    success_col: str = 'success',
) -> WorkerMetrics:
    """
    Compute performance metrics for a single worker.
    
    Args:
        worker_id: Worker identifier
        operations_df: DataFrame with operation records
        reference_productivity: Reference productivity for efficiency calc
        available_hours: Total available hours for saturation calc
        worker_col: Column name for worker ID
        time_col: Column name for time (minutes)
        units_col: Column name for units processed
        date_col: Column name for date
        op_type_col: Column name for operation type
        success_col: Column name for success indicator (1/0)
    
    Returns:
        WorkerMetrics with computed values
    """
    metrics = WorkerMetrics(worker_id=worker_id)
    
    # Filter for this worker
    if worker_col not in operations_df.columns:
        return metrics
    
    worker_ops = operations_df[operations_df[worker_col] == worker_id]
    
    if worker_ops.empty:
        return metrics
    
    # Total time and units
    if time_col in worker_ops.columns:
        total_time_min = worker_ops[time_col].sum()
        metrics.total_time_hours = total_time_min / 60
        metrics.occupied_time_hours = total_time_min / 60
    
    if units_col in worker_ops.columns:
        metrics.total_units = worker_ops[units_col].sum()
    
    metrics.total_operations = len(worker_ops)
    
    # Productivity
    if metrics.total_time_hours > 0:
        metrics.productivity = metrics.total_units / metrics.total_time_hours
    
    # Efficiency
    if reference_productivity and reference_productivity > 0:
        metrics.efficiency = metrics.productivity / reference_productivity
    else:
        # Compute machine/team average as reference
        if time_col in operations_df.columns and units_col in operations_df.columns:
            total_all_time = operations_df[time_col].sum() / 60
            total_all_units = operations_df[units_col].sum()
            if total_all_time > 0:
                avg_productivity = total_all_units / total_all_time
                if avg_productivity > 0:
                    metrics.efficiency = metrics.productivity / avg_productivity
    
    # Saturation
    if available_hours and available_hours > 0:
        metrics.available_time_hours = available_hours
        metrics.saturation = metrics.occupied_time_hours / available_hours
    else:
        # Estimate available hours (8h/day * unique days)
        if date_col in worker_ops.columns:
            unique_days = worker_ops[date_col].nunique()
            metrics.available_time_hours = unique_days * 8
            if metrics.available_time_hours > 0:
                metrics.saturation = metrics.occupied_time_hours / metrics.available_time_hours
    
    # Skill score
    if op_type_col in worker_ops.columns:
        ops_by_type = {}
        for op_type in worker_ops[op_type_col].unique():
            type_ops = worker_ops[worker_ops[op_type_col] == op_type]
            total = len(type_ops)
            success = total  # Default: all successful
            if success_col in type_ops.columns:
                success = type_ops[success_col].sum()
            ops_by_type[str(op_type)] = {'total': total, 'success': int(success)}
        
        metrics.skill_score = compute_skill_score(ops_by_type)
        metrics.successful_operations = sum(v['success'] for v in ops_by_type.values())
    
    # Classifications
    metrics.performance_level = classify_performance(metrics.efficiency)
    metrics.saturation_level = classify_saturation(metrics.saturation)
    
    # Period
    if date_col in worker_ops.columns:
        dates = pd.to_datetime(worker_ops[date_col], errors='coerce')
        valid_dates = dates.dropna()
        if not valid_dates.empty:
            metrics.period_start = valid_dates.min().isoformat()
            metrics.period_end = valid_dates.max().isoformat()
    
    return metrics


def compute_all_worker_performances(
    operations_df: pd.DataFrame,
    workers_df: Optional[pd.DataFrame] = None,
    worker_col: str = 'worker_id',
    **kwargs
) -> Dict[str, WorkerPerformance]:
    """
    Compute performance for all workers.
    
    Args:
        operations_df: DataFrame with operation records
        workers_df: Optional DataFrame with worker info
        worker_col: Column name for worker ID
        **kwargs: Additional arguments for compute_worker_performance
    
    Returns:
        Dict mapping worker_id -> WorkerPerformance
    """
    performances = {}
    
    if worker_col not in operations_df.columns:
        return performances
    
    # Get unique workers
    workers = operations_df[worker_col].unique()
    
    for worker_id in workers:
        if pd.isna(worker_id):
            continue
        
        worker_id = str(worker_id)
        
        # Compute metrics
        metrics = compute_worker_performance(
            worker_id, operations_df, worker_col=worker_col, **kwargs
        )
        
        # Build performance profile
        perf = WorkerPerformance(
            worker_id=worker_id,
            metrics=metrics,
        )
        
        # Compute daily productivity history for SNR and learning curve
        date_col = kwargs.get('date_col', 'date')
        time_col = kwargs.get('time_col', 'duration_min')
        units_col = kwargs.get('units_col', 'qty')
        
        worker_ops = operations_df[operations_df[worker_col] == worker_id]
        
        if date_col in worker_ops.columns:
            daily = worker_ops.groupby(date_col).agg({
                time_col: 'sum',
                units_col: 'sum'
            }).reset_index()
            
            daily['productivity'] = daily[units_col] / (daily[time_col] / 60)
            daily = daily.dropna(subset=['productivity'])
            daily = daily.sort_values(date_col)
            
            if len(daily) >= 3:
                perf.productivity_history = daily['productivity'].tolist()
                perf.dates = daily[date_col].astype(str).tolist()
                
                # Compute SNR
                snr, snr_level = compute_performance_snr(np.array(perf.productivity_history))
                metrics.snr_performance = snr
                metrics.snr_level = snr_level
                metrics.consistency_score = snr / (1 + snr)
                
                # Fit learning curve
                times = np.arange(len(perf.productivity_history))
                prods = np.array(perf.productivity_history)
                metrics.learning_curve = fit_learning_curve(times, prods)
        
        # Get worker name if available
        if workers_df is not None and worker_col in workers_df.columns:
            worker_info = workers_df[workers_df[worker_col] == worker_id]
            if not worker_info.empty:
                name_cols = ['name', 'worker_name', 'nome']
                for nc in name_cols:
                    if nc in worker_info.columns:
                        metrics.worker_name = str(worker_info.iloc[0][nc])
                        break
        
        # Generate recommendations
        perf.recommendations = _generate_recommendations(metrics)
        
        performances[worker_id] = perf
    
    return performances


def _generate_recommendations(metrics: WorkerMetrics) -> List[str]:
    """Generate improvement recommendations based on metrics."""
    recommendations = []
    
    # Performance-based
    if metrics.performance_level == PerformanceLevel.NEEDS_IMPROVEMENT.value:
        recommendations.append(
            f"Performance abaixo do esperado (eficiência: {metrics.efficiency:.0%}). "
            f"Considerar formação adicional ou reatribuição de tarefas."
        )
    elif metrics.performance_level == PerformanceLevel.EXCELLENT.value:
        recommendations.append(
            f"Performance excelente (eficiência: {metrics.efficiency:.0%}). "
            f"Candidato a formador ou tarefas mais complexas."
        )
    
    # Saturation-based
    if metrics.saturation_level == SaturationLevel.UNDERUTILIZED.value:
        recommendations.append(
            f"Colaborador subutilizado (saturação: {metrics.saturation:.0%}). "
            f"Considerar atribuir mais tarefas ou redistribuir recursos."
        )
    elif metrics.saturation_level == SaturationLevel.OVERLOADED.value:
        recommendations.append(
            f"Colaborador sobrecarregado (saturação: {metrics.saturation:.0%}). "
            f"Risco de burnout. Redistribuir carga ou contratar reforços."
        )
    
    # Consistency-based
    if metrics.snr_level == "POOR":
        recommendations.append(
            f"Alta variabilidade de performance (SNR: {metrics.snr_performance:.1f}). "
            f"Investigar causas: formação, equipamento, condições de trabalho."
        )
    
    # Learning curve
    if metrics.learning_curve:
        if metrics.learning_curve.r_squared < 0.5:
            recommendations.append(
                "Curva de aprendizagem não ajusta bem aos dados. "
                "Performance pode ser afetada por factores externos."
            )
        elif metrics.learning_curve.time_to_90pct and metrics.learning_curve.time_to_90pct > 60:
            recommendations.append(
                f"Tempo estimado para atingir 90% da produtividade máxima: "
                f"{metrics.learning_curve.time_to_90pct:.0f} dias. "
                f"Considerar formação acelerada."
            )
    
    if not recommendations:
        recommendations.append("Performance dentro dos parâmetros esperados.")
    
    return recommendations

