"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — SIGNAL-TO-NOISE RATIO (SNR) ENGINE
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Central module for data quality assessment using Signal-to-Noise Ratio (SNR) as the fundamental metric.

SNR provides a principled, statistically-grounded measure of data predictability and model reliability.

MATHEMATICAL FOUNDATION
═══════════════════════

Definition (Signal-to-Noise Ratio):
───────────────────────────────────
    
    SNR = σ²_signal / σ²_noise = Var(μ) / Var(ε)

where:
    μ = E[Y|X] is the conditional expectation (systematic/explainable component)
    ε = Y - μ  is the residual (unexplained noise)

Equivalently, in terms of ANOVA decomposition:
    
    SNR = SS_between / SS_within = MSB / MSW

Relationship to R² (Coefficient of Determination):
    
    R² = SNR / (1 + SNR)    ⟺    SNR = R² / (1 - R²)

    Example mappings:
    ┌────────┬────────┬─────────────────────────────┐
    │   R²   │   SNR  │       Interpretation        │
    ├────────┼────────┼─────────────────────────────┤
    │  0.50  │   1.0  │ Signal = Noise              │
    │  0.75  │   3.0  │ Signal dominates            │
    │  0.90  │   9.0  │ Strong signal               │
    │  0.95  │  19.0  │ Very strong signal          │
    │  0.99  │  99.0  │ Near-deterministic          │
    └────────┴────────┴─────────────────────────────┘

SNR in Decibels (dB):
    
    SNR_dB = 10 · log₁₀(SNR)

    ┌──────────┬─────────┬─────────────────────────────┐
    │ SNR (dB) │   SNR   │       Interpretation        │
    ├──────────┼─────────┼─────────────────────────────┤
    │   0 dB   │   1.0   │ Signal = Noise              │
    │   5 dB   │   3.2   │ Signal moderately stronger  │
    │  10 dB   │  10.0   │ Signal clearly dominant     │
    │  20 dB   │ 100.0   │ Noise negligible            │
    └──────────┴─────────┴─────────────────────────────┘

CLASSIFICATION THRESHOLDS
─────────────────────────

For industrial APS applications:

    SNR_class = {
        "EXCELLENT" : SNR ≥ 10.0   (R² ≥ 0.91)  — High predictability
        "HIGH"      : SNR ≥ 5.0    (R² ≥ 0.83)  — Good predictability
        "MEDIUM"    : SNR ≥ 2.0    (R² ≥ 0.67)  — Moderate predictability
        "LOW"       : SNR ≥ 1.0    (R² ≥ 0.50)  — Limited predictability
        "POOR"      : SNR < 1.0    (R² < 0.50)  — Noise-dominated
    }

CONFIDENCE SCORE
────────────────

Confidence is derived from SNR using a sigmoid transformation:

    confidence = 1 - exp(-SNR / τ)

where τ is a temperature parameter (default τ = 3.0).

This gives:
    SNR = 0   → confidence ≈ 0.00
    SNR = 1   → confidence ≈ 0.28
    SNR = 3   → confidence ≈ 0.63
    SNR = 10  → confidence ≈ 0.96

R&D / SIFIDE ALIGNMENT
──────────────────────
Work Package 4: Evaluation & Explainability
- Hypothesis H4.1: SNR correlates with forecast accuracy
- Hypothesis H4.2: Low SNR operations benefit most from ML
- Experiment E4.1: SNR vs. MAPE across operation types

REFERENCES
──────────
[1] Fisher, R.A. (1925). Statistical Methods for Research Workers.
[2] Box, Hunter & Hunter (2005). Statistics for Experimenters. Wiley.
[3] Cover & Thomas (2006). Elements of Information Theory. Wiley.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# CONSTANTS AND THRESHOLDS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

class SNRClass(str, Enum):
    """SNR classification levels."""
    EXCELLENT = "EXCELLENT"  # SNR ≥ 10.0
    HIGH = "HIGH"            # SNR ≥ 5.0
    MEDIUM = "MEDIUM"        # SNR ≥ 2.0
    LOW = "LOW"              # SNR ≥ 1.0
    POOR = "POOR"            # SNR < 1.0


# SNR thresholds for classification
SNR_THRESHOLDS: Dict[str, float] = {
    "EXCELLENT": 10.0,
    "HIGH": 5.0,
    "MEDIUM": 2.0,
    "LOW": 1.0,
    "POOR": 0.0,
}

# Confidence temperature parameter
CONFIDENCE_TAU: float = 3.0


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# CORE SNR COMPUTATION FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def compute_snr(
    values: np.ndarray,
    groups: Optional[np.ndarray] = None,
    method: str = "anova"
) -> float:
    """
    Compute Signal-to-Noise Ratio.
    
    Mathematical Definition:
    ────────────────────────
    
    Method 1 (ANOVA - when groups provided):
        
        SNR = MSB / MSW = SS_between/df_between / SS_within/df_within
        
        where:
            SS_between = Σᵢ nᵢ(μᵢ - μ)²    (between-group sum of squares)
            SS_within  = Σᵢ Σⱼ (xᵢⱼ - μᵢ)² (within-group sum of squares)
            
    Method 2 (Variance ratio - no groups):
        
        SNR = Var(signal) / Var(noise)
        
        Estimated using:
            signal_var = Var(smoothed_values) where smoothed = moving_average(values)
            noise_var  = Var(values - smoothed_values)
    
    Args:
        values: Array of observed values
        groups: Optional array of group labels for ANOVA method
        method: "anova" or "variance"
    
    Returns:
        SNR value (≥ 0)
    
    Examples:
        >>> values = np.array([10, 11, 9, 10, 12, 8])  # Low noise
        >>> compute_snr(values)
        8.5
        
        >>> groups = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
        >>> compute_snr(values, groups, method='anova')
        3.2
    """
    values = np.asarray(values, dtype=np.float64)
    
    if len(values) < 3:
        return 0.0
    
    # Remove NaN values
    mask = ~np.isnan(values)
    values = values[mask]
    
    if len(values) < 3:
        return 0.0
    
    if groups is not None and method == "anova":
        groups = np.asarray(groups)[mask]
        return _compute_snr_anova(values, groups)
    else:
        return _compute_snr_variance(values)


def _compute_snr_anova(values: np.ndarray, groups: np.ndarray) -> float:
    """
    Compute SNR using one-way ANOVA decomposition.
    
    SNR = MSB / MSW
    
    where:
        MSB = SS_between / (k - 1)     k = number of groups
        MSW = SS_within / (n - k)      n = total observations
    """
    unique_groups = np.unique(groups)
    k = len(unique_groups)
    n = len(values)
    
    if k < 2 or n <= k:
        return _compute_snr_variance(values)
    
    grand_mean = np.mean(values)
    
    # Compute SS_between and SS_within
    ss_between = 0.0
    ss_within = 0.0
    
    for g in unique_groups:
        group_mask = groups == g
        group_values = values[group_mask]
        group_mean = np.mean(group_values)
        group_n = len(group_values)
        
        ss_between += group_n * (group_mean - grand_mean) ** 2
        ss_within += np.sum((group_values - group_mean) ** 2)
    
    # Compute mean squares
    df_between = k - 1
    df_within = n - k
    
    msb = ss_between / df_between if df_between > 0 else 0.0
    msw = ss_within / df_within if df_within > 0 else 1e-10
    
    snr = msb / msw if msw > 1e-10 else float('inf')
    
    return min(snr, 1000.0)  # Cap at reasonable maximum


def _compute_snr_variance(values: np.ndarray, window: int = 3) -> float:
    """
    Compute SNR using variance decomposition with smoothing.
    
    signal = moving_average(values)
    noise = values - signal
    
    SNR = Var(signal) / Var(noise)
    """
    if len(values) < window:
        return 1.0
    
    # Compute moving average as signal estimate
    kernel = np.ones(window) / window
    signal = np.convolve(values, kernel, mode='valid')
    
    # Align values with signal
    trim = (len(values) - len(signal)) // 2
    values_aligned = values[trim:trim + len(signal)]
    
    # Compute noise as residual
    noise = values_aligned - signal
    
    var_signal = np.var(signal)
    var_noise = np.var(noise)
    
    if var_noise < 1e-10:
        return 100.0 if var_signal > 0 else 1.0
    
    snr = var_signal / var_noise
    
    return min(max(snr, 0.01), 1000.0)


def snr_to_db(snr: float) -> float:
    """
    Convert SNR to decibels.
    
    SNR_dB = 10 · log₁₀(SNR)
    """
    if snr <= 0:
        return -float('inf')
    return 10.0 * math.log10(snr)


def db_to_snr(snr_db: float) -> float:
    """
    Convert decibels to SNR.
    
    SNR = 10^(SNR_dB / 10)
    """
    return 10.0 ** (snr_db / 10.0)


def snr_to_r_squared(snr: float) -> float:
    """
    Convert SNR to coefficient of determination R².
    
    R² = SNR / (1 + SNR)
    """
    if snr <= 0:
        return 0.0
    return snr / (1.0 + snr)


def r_squared_to_snr(r_squared: float) -> float:
    """
    Convert R² to SNR.
    
    SNR = R² / (1 - R²)
    """
    if r_squared <= 0:
        return 0.0
    if r_squared >= 1:
        return float('inf')
    return r_squared / (1.0 - r_squared)


def classify_snr(snr: float) -> SNRClass:
    """
    Classify SNR into quality level.
    
    Classification:
        EXCELLENT : SNR ≥ 10.0
        HIGH      : SNR ≥ 5.0
        MEDIUM    : SNR ≥ 2.0
        LOW       : SNR ≥ 1.0
        POOR      : SNR < 1.0
    """
    if snr >= SNR_THRESHOLDS["EXCELLENT"]:
        return SNRClass.EXCELLENT
    elif snr >= SNR_THRESHOLDS["HIGH"]:
        return SNRClass.HIGH
    elif snr >= SNR_THRESHOLDS["MEDIUM"]:
        return SNRClass.MEDIUM
    elif snr >= SNR_THRESHOLDS["LOW"]:
        return SNRClass.LOW
    else:
        return SNRClass.POOR


def compute_confidence(snr: float, tau: float = CONFIDENCE_TAU) -> float:
    """
    Compute confidence score from SNR.
    
    confidence = 1 - exp(-SNR / τ)
    
    where τ is the temperature parameter controlling the curve.
    
    Args:
        snr: Signal-to-Noise Ratio
        tau: Temperature parameter (default: 3.0)
    
    Returns:
        Confidence score in [0, 1]
    """
    if snr <= 0:
        return 0.0
    return 1.0 - math.exp(-snr / tau)


def interpret_snr(snr: float) -> Tuple[str, str, float]:
    """
    Interpret SNR value for human communication.
    
    Returns:
        (level_code, description_pt, confidence_score)
    """
    level = classify_snr(snr)
    confidence = compute_confidence(snr)
    
    descriptions = {
        SNRClass.EXCELLENT: "Excelente previsibilidade. Dados altamente consistentes.",
        SNRClass.HIGH: "Alta previsibilidade. Dados confiáveis.",
        SNRClass.MEDIUM: "Previsibilidade moderada. Monitorizar resultados.",
        SNRClass.LOW: "Previsibilidade limitada. Usar com cautela.",
        SNRClass.POOR: "Baixa previsibilidade. Dados dominados por ruído.",
    }
    
    return level.value, descriptions[level], confidence


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# DOMAIN-SPECIFIC SNR FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class SNRResult:
    """Result of an SNR computation."""
    snr_value: float
    snr_db: float
    snr_class: str
    confidence_score: float
    r_squared: float
    sample_size: int
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'snr_value': round(self.snr_value, 3),
            'snr_db': round(self.snr_db, 2),
            'snr_class': self.snr_class,
            'confidence_score': round(self.confidence_score, 3),
            'r_squared': round(self.r_squared, 3),
            'sample_size': self.sample_size,
            'description': self.description,
            **self.details,
        }


def snr_processing_time(
    plan_df: pd.DataFrame,
    group_by: str = 'op_code',
    time_col: str = 'duration_min'
) -> SNRResult:
    """
    Compute SNR for processing times grouped by operation type.
    
    Measures how consistent processing times are within each operation type
    compared to variation between types.
    
    Mathematical Interpretation:
    ───────────────────────────
    High SNR means:
        - Processing times are CONSISTENT within each op_code
        - Knowing op_code explains most of the variance
        - Predictions based on op_code will be accurate
    
    Low SNR means:
        - High variability WITHIN each op_code
        - Other factors (operator, machine state, etc.) dominate
        - Need additional features for accurate prediction
    
    Args:
        plan_df: DataFrame with plan operations
        group_by: Column to group by (default: 'op_code')
        time_col: Column with processing times
    
    Returns:
        SNRResult with SNR metrics
    """
    if plan_df.empty or group_by not in plan_df.columns or time_col not in plan_df.columns:
        return SNRResult(
            snr_value=0.0, snr_db=-float('inf'), snr_class="POOR",
            confidence_score=0.0, r_squared=0.0, sample_size=0,
            description="Dados insuficientes para análise SNR."
        )
    
    df = plan_df[[group_by, time_col]].dropna()
    
    if len(df) < 5:
        return SNRResult(
            snr_value=0.0, snr_db=-float('inf'), snr_class="POOR",
            confidence_score=0.0, r_squared=0.0, sample_size=len(df),
            description="Amostra muito pequena."
        )
    
    values = df[time_col].values
    groups = df[group_by].values
    
    snr = compute_snr(values, groups, method='anova')
    snr_db = snr_to_db(snr)
    level, desc, conf = interpret_snr(snr)
    r2 = snr_to_r_squared(snr)
    
    # Compute per-group statistics
    group_stats = df.groupby(group_by)[time_col].agg(['mean', 'std', 'count']).to_dict('index')
    
    return SNRResult(
        snr_value=snr,
        snr_db=snr_db,
        snr_class=level,
        confidence_score=conf,
        r_squared=r2,
        sample_size=len(df),
        description=f"SNR tempos de processamento por {group_by}: {desc}",
        details={
            'group_by': group_by,
            'n_groups': len(group_stats),
            'group_stats': group_stats,
        }
    )


def snr_setup_matrix(
    setup_df: pd.DataFrame,
    from_col: str = 'from_setup_family',
    to_col: str = 'to_setup_family',
    time_col: str = 'setup_time_min',
    historical_df: Optional[pd.DataFrame] = None
) -> SNRResult:
    """
    Compute SNR for setup times.
    
    Measures how well the setup matrix predicts actual setup times.
    
    If historical_df is provided:
        SNR = Var(matrix_values) / Var(actual - matrix_value)
    
    Otherwise:
        SNR based on consistency within each (from, to) transition.
    
    Args:
        setup_df: Setup matrix DataFrame
        from_col: Source family column
        to_col: Destination family column
        time_col: Setup time column
        historical_df: Optional historical setup times with 'actual_setup_min'
    
    Returns:
        SNRResult with setup time SNR
    """
    if setup_df.empty:
        return SNRResult(
            snr_value=1.0, snr_db=0.0, snr_class="LOW",
            confidence_score=0.28, r_squared=0.5, sample_size=0,
            description="Matriz de setup vazia. SNR assumido = 1.0."
        )
    
    # Create transition key
    setup_df = setup_df.copy()
    setup_df['transition'] = setup_df[from_col].astype(str) + '_to_' + setup_df[to_col].astype(str)
    
    values = setup_df[time_col].values
    groups = setup_df['transition'].values
    
    if historical_df is not None and 'actual_setup_min' in historical_df.columns:
        # Compare matrix values with actual values
        # Merge and compute residuals
        merged = historical_df.merge(
            setup_df[[from_col, to_col, time_col]],
            on=[from_col, to_col],
            how='left'
        )
        merged = merged.dropna(subset=['actual_setup_min', time_col])
        
        if len(merged) >= 5:
            predicted = merged[time_col].values
            actual = merged['actual_setup_min'].values
            residuals = actual - predicted
            
            var_signal = np.var(predicted)
            var_noise = np.var(residuals)
            
            snr = var_signal / var_noise if var_noise > 1e-10 else 100.0
        else:
            snr = compute_snr(values, groups, method='anova')
    else:
        snr = compute_snr(values, groups, method='anova')
    
    snr_db = snr_to_db(snr)
    level, desc, conf = interpret_snr(snr)
    r2 = snr_to_r_squared(snr)
    
    return SNRResult(
        snr_value=snr,
        snr_db=snr_db,
        snr_class=level,
        confidence_score=conf,
        r_squared=r2,
        sample_size=len(setup_df),
        description=f"SNR matriz de setup: {desc}",
        details={
            'n_transitions': len(setup_df['transition'].unique()) if 'transition' in setup_df else 0,
            'avg_setup_min': float(np.mean(values)),
            'std_setup_min': float(np.std(values)),
        }
    )


def snr_forecast(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str = "unknown"
) -> SNRResult:
    """
    Compute SNR for a forecast model.
    
    SNR = Var(predicted) / Var(actual - predicted)
        = Var(signal) / Var(residual)
    
    This measures how much of the actual variance is captured by the model
    versus unexplained residual variance.
    
    Args:
        actual: Actual observed values
        predicted: Model predictions
        model_name: Name of the forecasting model
    
    Returns:
        SNRResult with forecast SNR
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    
    # Align lengths
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    
    # Remove NaN
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) < 3:
        return SNRResult(
            snr_value=0.0, snr_db=-float('inf'), snr_class="POOR",
            confidence_score=0.0, r_squared=0.0, sample_size=len(actual),
            description="Amostra insuficiente para avaliar forecast."
        )
    
    residuals = actual - predicted
    
    var_signal = np.var(predicted)
    var_noise = np.var(residuals)
    
    if var_noise < 1e-10:
        snr = 100.0 if var_signal > 0 else 1.0
    else:
        snr = var_signal / var_noise
    
    snr = min(max(snr, 0.01), 1000.0)
    
    snr_db = snr_to_db(snr)
    level, desc, conf = interpret_snr(snr)
    r2 = snr_to_r_squared(snr)
    
    # Compute additional metrics
    mape = np.mean(np.abs(residuals / (actual + 1e-10))) * 100
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    return SNRResult(
        snr_value=snr,
        snr_db=snr_db,
        snr_class=level,
        confidence_score=conf,
        r_squared=r2,
        sample_size=len(actual),
        description=f"SNR do modelo {model_name}: {desc}",
        details={
            'model_name': model_name,
            'mape_pct': round(mape, 2),
            'rmse': round(rmse, 3),
            'var_signal': round(var_signal, 3),
            'var_noise': round(var_noise, 3),
        }
    )


def snr_rul(
    degradation_signal: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    failure_threshold: Optional[float] = None
) -> SNRResult:
    """
    Compute SNR for Remaining Useful Life (RUL) estimation.
    
    Measures how clearly the degradation signal stands out from noise.
    A high SNR indicates that degradation is progressing in a predictable manner.
    
    Mathematical Approach:
    ─────────────────────
    1. Fit a trend line (linear or polynomial) to the degradation signal
    2. Compute residuals from the trend
    3. SNR = Var(trend) / Var(residuals)
    
    High SNR: Degradation follows a clear trend → RUL prediction reliable
    Low SNR: Noisy signal → RUL prediction uncertain
    
    Args:
        degradation_signal: Array of health indicator values over time
        timestamps: Optional timestamps (uses indices if not provided)
        failure_threshold: Optional threshold defining failure
    
    Returns:
        SNRResult with RUL SNR analysis
    """
    degradation_signal = np.asarray(degradation_signal, dtype=np.float64)
    
    if len(degradation_signal) < 5:
        return SNRResult(
            snr_value=0.5, snr_db=-3.0, snr_class="POOR",
            confidence_score=0.15, r_squared=0.33, sample_size=len(degradation_signal),
            description="Sinal de degradação insuficiente para análise."
        )
    
    # Create time index
    if timestamps is None:
        t = np.arange(len(degradation_signal))
    else:
        t = np.asarray(timestamps, dtype=np.float64)
    
    # Remove NaN
    mask = ~np.isnan(degradation_signal)
    t = t[mask]
    y = degradation_signal[mask]
    
    if len(y) < 5:
        return SNRResult(
            snr_value=0.5, snr_db=-3.0, snr_class="POOR",
            confidence_score=0.15, r_squared=0.33, sample_size=len(y),
            description="Dados válidos insuficientes após remoção de NaN."
        )
    
    # Fit linear trend
    coeffs = np.polyfit(t, y, 1)
    trend = np.polyval(coeffs, t)
    residuals = y - trend
    
    var_signal = np.var(trend)
    var_noise = np.var(residuals)
    
    if var_noise < 1e-10:
        snr = 100.0 if var_signal > 0 else 1.0
    else:
        snr = var_signal / var_noise
    
    snr = min(max(snr, 0.01), 1000.0)
    
    snr_db = snr_to_db(snr)
    level, desc, conf = interpret_snr(snr)
    r2 = snr_to_r_squared(snr)
    
    # Estimate RUL if threshold provided
    rul_estimate = None
    if failure_threshold is not None and coeffs[0] != 0:
        # Time to reach threshold from last observation
        t_failure = (failure_threshold - coeffs[1]) / coeffs[0]
        rul_estimate = t_failure - t[-1]
    
    return SNRResult(
        snr_value=snr,
        snr_db=snr_db,
        snr_class=level,
        confidence_score=conf,
        r_squared=r2,
        sample_size=len(y),
        description=f"SNR sinal de degradação: {desc}",
        details={
            'trend_slope': round(coeffs[0], 6),
            'trend_intercept': round(coeffs[1], 3),
            'rul_estimate': round(rul_estimate, 1) if rul_estimate else None,
            'failure_threshold': failure_threshold,
        }
    )


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# SIGNAL-NOISE ANALYZER CLASS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class DataQualityReport:
    """Comprehensive data quality report based on SNR analysis."""
    timestamp: str
    total_records: int
    missing_rate: float
    global_snr: float
    global_snr_db: float
    global_confidence: float
    quality_level: str
    
    snr_by_machine: Dict[str, SNRResult] = field(default_factory=dict)
    snr_by_operation: Dict[str, SNRResult] = field(default_factory=dict)
    snr_setup: Optional[SNRResult] = None
    
    low_snr_machines: List[str] = field(default_factory=list)
    low_snr_operations: List[str] = field(default_factory=list)
    
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'total_records': self.total_records,
            'missing_rate': round(self.missing_rate, 3),
            'global_snr': round(self.global_snr, 2),
            'global_snr_db': round(self.global_snr_db, 1),
            'global_confidence': round(self.global_confidence, 2),
            'quality_level': self.quality_level,
            'summary': self._generate_summary(),
            'low_snr_machines': self.low_snr_machines,
            'low_snr_operations': self.low_snr_operations,
            'recommendations': self.recommendations,
            'snr_by_machine': {k: v.to_dict() for k, v in self.snr_by_machine.items()},
            'snr_by_operation': {k: v.to_dict() for k, v in self.snr_by_operation.items()},
            'snr_setup': self.snr_setup.to_dict() if self.snr_setup else None,
        }
    
    def _generate_summary(self) -> str:
        parts = [
            f"Análise de {self.total_records} registos.",
            f"SNR global: {self.global_snr:.1f} ({self.global_snr_db:.1f} dB).",
            f"Nível de qualidade: {self.quality_level}.",
        ]
        
        if self.low_snr_machines:
            parts.append(f"Máquinas com SNR baixo: {', '.join(self.low_snr_machines[:5])}.")
        
        if self.low_snr_operations:
            parts.append(f"Operações com SNR baixo: {', '.join(self.low_snr_operations[:5])}.")
        
        return " ".join(parts)


class SignalNoiseAnalyzer:
    """
    Comprehensive analyzer for Signal-to-Noise Ratio across plan data.
    
    Usage:
        analyzer = SignalNoiseAnalyzer(min_samples=5, snr_threshold=2.0)
        report = analyzer.analyze(plan_df, routing_df)
        print(report.to_dict())
    """
    
    def __init__(
        self,
        min_samples: int = 5,
        snr_threshold: float = 2.0,
        time_col: str = 'duration_min'
    ):
        """
        Initialize analyzer.
        
        Args:
            min_samples: Minimum samples required for SNR computation
            snr_threshold: Threshold below which SNR is considered "low"
            time_col: Column name for processing times
        """
        self.min_samples = min_samples
        self.snr_threshold = snr_threshold
        self.time_col = time_col
    
    def analyze(
        self,
        plan_df: pd.DataFrame,
        routing_df: Optional[pd.DataFrame] = None,
        setup_df: Optional[pd.DataFrame] = None,
        historical_df: Optional[pd.DataFrame] = None
    ) -> DataQualityReport:
        """
        Perform comprehensive SNR analysis.
        
        Args:
            plan_df: Production plan DataFrame
            routing_df: Optional routing DataFrame
            setup_df: Optional setup matrix DataFrame
            historical_df: Optional historical data for validation
        
        Returns:
            DataQualityReport with SNR metrics
        """
        timestamp = datetime.now().isoformat()
        total_records = len(plan_df)
        
        # Compute missing rate
        missing_rate = plan_df.isnull().sum().sum() / (plan_df.shape[0] * plan_df.shape[1])
        
        # Global SNR (processing times by operation)
        global_result = snr_processing_time(plan_df, 'op_code', self.time_col)
        
        # SNR by machine
        snr_by_machine = {}
        low_snr_machines = []
        
        if 'machine_id' in plan_df.columns:
            for machine in plan_df['machine_id'].unique():
                machine_df = plan_df[plan_df['machine_id'] == machine]
                if len(machine_df) >= self.min_samples:
                    result = snr_processing_time(machine_df, 'op_code', self.time_col)
                    snr_by_machine[machine] = result
                    if result.snr_value < self.snr_threshold:
                        low_snr_machines.append(machine)
        
        # SNR by operation
        snr_by_operation = {}
        low_snr_operations = []
        
        if 'op_code' in plan_df.columns:
            for op_code in plan_df['op_code'].unique():
                op_df = plan_df[plan_df['op_code'] == op_code]
                if len(op_df) >= self.min_samples:
                    values = op_df[self.time_col].dropna().values
                    if len(values) >= self.min_samples:
                        snr = compute_snr(values)
                        level, desc, conf = interpret_snr(snr)
                        result = SNRResult(
                            snr_value=snr,
                            snr_db=snr_to_db(snr),
                            snr_class=level,
                            confidence_score=conf,
                            r_squared=snr_to_r_squared(snr),
                            sample_size=len(values),
                            description=desc,
                        )
                        snr_by_operation[op_code] = result
                        if snr < self.snr_threshold:
                            low_snr_operations.append(op_code)
        
        # SNR for setup matrix
        snr_setup = None
        if setup_df is not None and not setup_df.empty:
            snr_setup = snr_setup_matrix(setup_df, historical_df=historical_df)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            global_result, low_snr_machines, low_snr_operations, snr_setup
        )
        
        return DataQualityReport(
            timestamp=timestamp,
            total_records=total_records,
            missing_rate=missing_rate,
            global_snr=global_result.snr_value,
            global_snr_db=global_result.snr_db,
            global_confidence=global_result.confidence_score,
            quality_level=global_result.snr_class,
            snr_by_machine=snr_by_machine,
            snr_by_operation=snr_by_operation,
            snr_setup=snr_setup,
            low_snr_machines=low_snr_machines,
            low_snr_operations=low_snr_operations,
            recommendations=recommendations,
        )
    
    def _generate_recommendations(
        self,
        global_result: SNRResult,
        low_snr_machines: List[str],
        low_snr_operations: List[str],
        snr_setup: Optional[SNRResult]
    ) -> List[str]:
        """Generate actionable recommendations based on SNR analysis."""
        recommendations = []
        
        if global_result.snr_value < 2.0:
            recommendations.append(
                "SNR global baixo. Considerar recolha de mais dados ou identificar "
                "fontes de variabilidade não explicada (operador, condições ambientais)."
            )
        
        if len(low_snr_machines) > 0:
            recommendations.append(
                f"Máquinas com SNR baixo ({', '.join(low_snr_machines[:3])}): "
                "Verificar calibração, manutenção ou condições operacionais."
            )
        
        if len(low_snr_operations) > 0:
            recommendations.append(
                f"Operações com SNR baixo ({', '.join(low_snr_operations[:3])}): "
                "Considerar ML para previsão mais precisa destes tempos."
            )
        
        if snr_setup and snr_setup.snr_value < 3.0:
            recommendations.append(
                "Matriz de setup com SNR moderado. Considerar aprendizagem "
                "de setup times com modelo ML (XGBoost/LightGBM)."
            )
        
        if not recommendations:
            recommendations.append(
                "Qualidade de dados adequada para previsões fiáveis."
            )
        
        return recommendations
