"""
ProdPlan 4.0 - Model Metrics & SNR for ML

Rigorous evaluation metrics for ML models, with Signal-to-Noise Ratio
as a core measure of model quality and prediction reliability.

Mathematical Framework
======================

Signal-to-Noise Ratio for Forecasts:
-----------------------------------
For predictions ŷ vs actuals y:

    SNR_model = Var(ŷ) / Var(y - ŷ)
             = Var(signal explained) / Var(residual)

This is related to R²:
    R² = 1 - Var(residual)/Var(y) = SNR/(1+SNR)

Confidence from SNR:
-------------------
The SNR provides a principled confidence measure:
    confidence = min(1, SNR / (1 + SNR)) = R²

For probabilistic predictions with uncertainty σ:
    SNR_bayesian = (predicted_mean)² / σ²

Standard Metrics:
----------------
MAPE (Mean Absolute Percentage Error):
    MAPE = (100/n) × Σ|yᵢ - ŷᵢ|/|yᵢ|

RMSE (Root Mean Square Error):
    RMSE = √((1/n) × Σ(yᵢ - ŷᵢ)²)

MAE (Mean Absolute Error):
    MAE = (1/n) × Σ|yᵢ - ŷᵢ|

SMAPE (Symmetric MAPE):
    SMAPE = (200/n) × Σ|yᵢ - ŷᵢ|/(|yᵢ| + |ŷᵢ|)

R&D / SIFIDE: WP4 - Model Evaluation
Research Question Q4.3: How does SNR correlate with forecast reliability?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from data_quality import compute_snr, compute_snr_db, interpret_snr, snr_to_r_squared

logger = logging.getLogger(__name__)


# ============================================================
# FORECAST METRICS
# ============================================================

@dataclass
class ForecastMetrics:
    """
    Comprehensive metrics for forecast/prediction evaluation.
    
    Includes both standard metrics (MAPE, RMSE) and SNR-based measures.
    """
    # ===== STANDARD METRICS =====
    mape: float = float('inf')          # Mean Absolute Percentage Error (%)
    smape: float = float('inf')         # Symmetric MAPE (%)
    rmse: float = float('inf')          # Root Mean Square Error
    mae: float = float('inf')           # Mean Absolute Error
    mse: float = float('inf')           # Mean Square Error
    r_squared: float = 0.0              # Coefficient of Determination
    
    # ===== SNR-BASED METRICS =====
    snr: float = 0.0                    # Signal-to-Noise Ratio
    snr_db: float = float('-inf')       # SNR in decibels
    model_confidence: float = 0.0       # Derived from SNR
    
    # ===== BIAS METRICS =====
    mean_bias: float = 0.0              # Mean(actual - predicted)
    bias_ratio: float = 0.0             # Systematic over/under-estimation
    
    # ===== DIRECTIONAL ACCURACY =====
    direction_accuracy: float = 0.0     # % of correct direction predictions
    
    # ===== COVERAGE (for prediction intervals) =====
    coverage_90: Optional[float] = None  # % of actuals within 90% CI
    coverage_95: Optional[float] = None  # % of actuals within 95% CI
    
    # ===== SAMPLE INFO =====
    n_samples: int = 0
    
    # ===== INTERPRETATION =====
    quality_level: str = "UNKNOWN"
    interpretation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-friendly dictionary."""
        return {
            # Standard
            'mape': round(self.mape, 2) if self.mape < float('inf') else None,
            'smape': round(self.smape, 2) if self.smape < float('inf') else None,
            'rmse': round(self.rmse, 4),
            'mae': round(self.mae, 4),
            'mse': round(self.mse, 4),
            'r_squared': round(self.r_squared, 4),
            
            # SNR
            'snr': round(self.snr, 2),
            'snr_db': round(self.snr_db, 1) if self.snr_db > float('-inf') else None,
            'model_confidence': round(self.model_confidence, 3),
            
            # Bias
            'mean_bias': round(self.mean_bias, 4),
            'bias_ratio': round(self.bias_ratio, 4),
            
            # Directional
            'direction_accuracy': round(self.direction_accuracy, 2),
            
            # Coverage
            'coverage_90': round(self.coverage_90, 2) if self.coverage_90 is not None else None,
            'coverage_95': round(self.coverage_95, 2) if self.coverage_95 is not None else None,
            
            # Sample
            'n_samples': self.n_samples,
            
            # Interpretation
            'quality_level': self.quality_level,
            'interpretation': self.interpretation,
        }


def compute_forecast_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None
) -> ForecastMetrics:
    """
    Compute comprehensive forecast metrics including SNR.
    
    Mathematical Definitions:
    ------------------------
    
    MAPE = (100/n) × Σ|yᵢ - ŷᵢ|/|yᵢ|  (only where yᵢ ≠ 0)
    
    SMAPE = (200/n) × Σ|yᵢ - ŷᵢ|/(|yᵢ| + |ŷᵢ|)
    
    RMSE = √(MSE) = √((1/n) × Σ(yᵢ - ŷᵢ)²)
    
    R² = 1 - SS_res/SS_tot = 1 - Σ(yᵢ - ŷᵢ)²/Σ(yᵢ - ȳ)²
    
    SNR = Var(ŷ) / Var(y - ŷ) = Var(explained) / Var(residual)
    
    Args:
        actual: Actual values (y)
        predicted: Predicted values (ŷ)
        lower_bound: Lower prediction interval (optional)
        upper_bound: Upper prediction interval (optional)
    
    Returns:
        ForecastMetrics with all computed values
    """
    metrics = ForecastMetrics()
    
    actual = np.asarray(actual, dtype=float).flatten()
    predicted = np.asarray(predicted, dtype=float).flatten()
    
    # Remove NaN pairs
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    n = len(actual)
    metrics.n_samples = n
    
    if n < 2:
        metrics.interpretation = "Dados insuficientes para avaliação"
        return metrics
    
    residuals = actual - predicted
    
    # ===== BASIC METRICS =====
    
    # MSE, RMSE, MAE
    metrics.mse = np.mean(residuals ** 2)
    metrics.rmse = np.sqrt(metrics.mse)
    metrics.mae = np.mean(np.abs(residuals))
    
    # MAPE (excluding zeros in actual)
    nonzero_mask = actual != 0
    if nonzero_mask.sum() > 0:
        metrics.mape = 100 * np.mean(np.abs(residuals[nonzero_mask]) / np.abs(actual[nonzero_mask]))
    
    # SMAPE
    denominator = np.abs(actual) + np.abs(predicted)
    nonzero_denom = denominator > 0
    if nonzero_denom.sum() > 0:
        metrics.smape = 200 * np.mean(np.abs(residuals[nonzero_denom]) / denominator[nonzero_denom])
    
    # R² (Coefficient of Determination)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot > 0:
        metrics.r_squared = 1 - ss_res / ss_tot
    
    # ===== SNR METRICS =====
    
    metrics.snr = compute_snr(actual, predicted)
    metrics.snr_db = compute_snr_db(metrics.snr)
    metrics.model_confidence = snr_to_r_squared(metrics.snr)
    
    # ===== BIAS METRICS =====
    
    metrics.mean_bias = np.mean(residuals)
    
    mean_actual = np.mean(np.abs(actual))
    if mean_actual > 0:
        metrics.bias_ratio = metrics.mean_bias / mean_actual
    
    # ===== DIRECTIONAL ACCURACY =====
    # (for time series: did we predict the direction of change correctly?)
    
    if n > 1:
        actual_diff = np.diff(actual)
        pred_diff = np.diff(predicted)
        
        # Ignore zero changes
        nonzero_changes = (actual_diff != 0)
        if nonzero_changes.sum() > 0:
            correct_direction = np.sign(actual_diff[nonzero_changes]) == np.sign(pred_diff[nonzero_changes])
            metrics.direction_accuracy = 100 * np.mean(correct_direction)
    
    # ===== COVERAGE (if prediction intervals provided) =====
    
    if lower_bound is not None and upper_bound is not None:
        lower_bound = np.asarray(lower_bound)[mask]
        upper_bound = np.asarray(upper_bound)[mask]
        
        in_interval = (actual >= lower_bound) & (actual <= upper_bound)
        coverage = np.mean(in_interval) * 100
        
        # Assume bounds are 90% CI (common case)
        metrics.coverage_90 = coverage
    
    # ===== INTERPRETATION =====
    
    level, desc, _ = interpret_snr(metrics.snr)
    metrics.quality_level = level
    
    # Build interpretation
    parts = []
    
    if metrics.mape < float('inf'):
        if metrics.mape < 10:
            parts.append(f"MAPE excelente ({metrics.mape:.1f}%)")
        elif metrics.mape < 20:
            parts.append(f"MAPE bom ({metrics.mape:.1f}%)")
        elif metrics.mape < 50:
            parts.append(f"MAPE moderado ({metrics.mape:.1f}%)")
        else:
            parts.append(f"MAPE elevado ({metrics.mape:.1f}%)")
    
    parts.append(f"SNR={metrics.snr:.1f} ({metrics.snr_db:.1f}dB)")
    parts.append(f"R²={metrics.r_squared:.3f}")
    
    if abs(metrics.bias_ratio) > 0.1:
        if metrics.bias_ratio > 0:
            parts.append(f"Viés: subestima {abs(metrics.bias_ratio)*100:.1f}%")
        else:
            parts.append(f"Viés: sobrestima {abs(metrics.bias_ratio)*100:.1f}%")
    
    metrics.interpretation = ". ".join(parts) + "."
    
    return metrics


def compute_model_snr(
    predictions: np.ndarray,
    uncertainties: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute SNR for model predictions with uncertainty.
    
    For Bayesian models or ensemble predictions:
    
        SNR_bayesian = E[ŷ]² / Var(ŷ)
        
    This measures how confident the model is in its predictions.
    High SNR = narrow uncertainty bands relative to prediction magnitude.
    
    Args:
        predictions: Point predictions (or mean of posterior)
        uncertainties: Standard deviations (if available)
    
    Returns:
        Dict with SNR metrics
    """
    predictions = np.asarray(predictions).flatten()
    predictions = predictions[~np.isnan(predictions)]
    
    if len(predictions) < 2:
        return {'snr': 0, 'confidence': 0}
    
    # Simple SNR from predictions alone
    pred_mean = np.mean(predictions)
    pred_var = np.var(predictions)
    
    simple_snr = (pred_mean ** 2) / pred_var if pred_var > 0 else float('inf')
    
    result = {
        'simple_snr': simple_snr,
        'simple_snr_db': compute_snr_db(simple_snr),
    }
    
    # Bayesian SNR (if uncertainties provided)
    if uncertainties is not None:
        uncertainties = np.asarray(uncertainties).flatten()
        uncertainties = uncertainties[~np.isnan(uncertainties)]
        
        if len(uncertainties) == len(predictions):
            # SNR per prediction
            snr_per_pred = predictions ** 2 / (uncertainties ** 2 + 1e-10)
            
            result['bayesian_snr_mean'] = np.mean(snr_per_pred)
            result['bayesian_snr_median'] = np.median(snr_per_pred)
            result['bayesian_snr_min'] = np.min(snr_per_pred)
            
            # Overall confidence
            avg_uncertainty = np.mean(uncertainties)
            avg_prediction = np.mean(np.abs(predictions))
            result['uncertainty_ratio'] = avg_uncertainty / avg_prediction if avg_prediction > 0 else float('inf')
            result['bayesian_confidence'] = 1 / (1 + result['uncertainty_ratio'])
    
    # Interpret
    primary_snr = result.get('bayesian_snr_mean', simple_snr)
    level, desc, conf = interpret_snr(primary_snr)
    result['quality_level'] = level
    result['confidence'] = conf
    result['description'] = desc
    
    return result


# ============================================================
# SPECIALIZED METRICS
# ============================================================

def compute_setup_prediction_metrics(
    actual_setup: np.ndarray,
    predicted_setup: np.ndarray
) -> ForecastMetrics:
    """
    Compute metrics for setup time predictions.
    
    Setup times have specific characteristics:
    - Often zero (same family)
    - High variability
    - Asymmetric cost of errors (under-prediction worse)
    
    Additional metric: Under-prediction rate
    """
    metrics = compute_forecast_metrics(actual_setup, predicted_setup)
    
    # Additional: under-prediction rate (predicted < actual)
    under_predictions = predicted_setup < actual_setup
    metrics.interpretation += f" Taxa de subestimação: {100*np.mean(under_predictions):.1f}%."
    
    return metrics


def compute_rul_prediction_metrics(
    actual_rul: np.ndarray,
    predicted_rul: np.ndarray,
    predicted_std: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute metrics for Remaining Useful Life (RUL) predictions.
    
    RUL prediction has specific requirements:
    - Early predictions (overestimate) can lead to unexpected failures
    - Late predictions (underestimate) waste maintenance resources
    - Uncertainty quantification is critical
    
    Special Metrics:
    - α-λ accuracy: % of predictions within α% of actual at λ% of life
    - Early/Late prediction asymmetry
    - Probability of Detection (PoD) at warning threshold
    """
    base_metrics = compute_forecast_metrics(actual_rul, predicted_rul)
    
    actual = np.asarray(actual_rul).flatten()
    predicted = np.asarray(predicted_rul).flatten()
    
    # Remove NaN
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    result = base_metrics.to_dict()
    
    if len(actual) < 2:
        return result
    
    # Early/Late classification
    early = predicted > actual  # Overestimate remaining life
    late = predicted < actual   # Underestimate remaining life
    
    result['early_prediction_rate'] = 100 * np.mean(early)  # Dangerous: unexpected failures
    result['late_prediction_rate'] = 100 * np.mean(late)    # Wasteful: premature maintenance
    
    # α-λ accuracy (e.g., within 20% of actual RUL)
    alpha = 0.20
    relative_error = np.abs(predicted - actual) / np.maximum(actual, 1e-10)
    result['alpha_lambda_accuracy_20'] = 100 * np.mean(relative_error <= alpha)
    
    # Uncertainty calibration (if provided)
    if predicted_std is not None:
        predicted_std = np.asarray(predicted_std)[mask]
        
        # Standardized residuals
        std_residuals = (actual - predicted) / np.maximum(predicted_std, 1e-10)
        
        # Should be ~N(0,1) if well-calibrated
        result['std_residual_mean'] = np.mean(std_residuals)
        result['std_residual_std'] = np.std(std_residuals)
        
        # Coverage
        in_1sigma = np.abs(std_residuals) <= 1
        in_2sigma = np.abs(std_residuals) <= 2
        result['coverage_1sigma'] = 100 * np.mean(in_1sigma)  # Should be ~68%
        result['coverage_2sigma'] = 100 * np.mean(in_2sigma)  # Should be ~95%
        
        # Calibration error
        result['calibration_error_1sigma'] = abs(result['coverage_1sigma'] - 68.27)
        result['calibration_error_2sigma'] = abs(result['coverage_2sigma'] - 95.45)
    
    return result


# ============================================================
# BENCHMARK COMPARISON
# ============================================================

def compare_forecasters(
    actual: np.ndarray,
    predictions: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Compare multiple forecasting methods.
    
    Args:
        actual: Actual values
        predictions: Dict model_name -> predictions
    
    Returns:
        DataFrame ranking models by various metrics
    """
    results = []
    
    for name, pred in predictions.items():
        metrics = compute_forecast_metrics(actual, pred)
        results.append({
            'model': name,
            **metrics.to_dict()
        })
    
    df = pd.DataFrame(results)
    
    # Add ranks
    for col in ['mape', 'rmse', 'mae']:
        if col in df.columns:
            df[f'{col}_rank'] = df[col].rank()
    
    df['snr_rank'] = df['snr'].rank(ascending=False)  # Higher SNR is better
    
    # Overall rank (average of individual ranks)
    rank_cols = [c for c in df.columns if c.endswith('_rank')]
    df['overall_rank'] = df[rank_cols].mean(axis=1)
    
    return df.sort_values('overall_rank')


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def quick_evaluate(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Quick evaluation with key metrics."""
    metrics = compute_forecast_metrics(actual, predicted)
    return {
        'mape': metrics.mape,
        'rmse': metrics.rmse,
        'r_squared': metrics.r_squared,
        'snr': metrics.snr,
        'confidence': metrics.model_confidence,
        'quality': metrics.quality_level,
    }



