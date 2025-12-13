"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — WORKFORCE FORECASTING
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Forecast worker productivity using time series models.

FORECASTING MODELS
══════════════════

1. ARIMA (Baseline)
   ────────────────
   
   ARIMA(p, d, q):
   - p: Autoregressive order (past values)
   - d: Differencing order (stationarity)
   - q: Moving average order (past errors)
   
   Suitable for:
   - Stationary or near-stationary series
   - Short-term forecasts (7-14 days)
   - Limited historical data

2. EXPONENTIAL SMOOTHING
   ──────────────────────
   
   Holt-Winters with damping:
   - Level component (α)
   - Trend component (β)
   - Seasonality (γ)
   
   Good for:
   - Weekly/daily patterns
   - Trend detection

3. TODO[R&D]: LSTM / Transformer
   ──────────────────────────────
   
   Deep learning for complex patterns:
   - Non-linear dependencies
   - Long-term patterns
   - Multi-variate inputs (weather, shifts, etc.)
   
   Requires:
   - Larger datasets (100+ samples)
   - GPU for training
   - Feature engineering

CONFIDENCE & SNR
════════════════

Forecast confidence based on:
- Historical SNR
- Model fit (R², MAPE)
- Prediction variance

    Confidence = f(SNR, model_accuracy)
    
where:
    SNR_forecast = Var(predicted) / Var(residual)

R&D / SIFIDE: WP6 - Workforce Intelligence
──────────────────────────────────────────
- Hypothesis H6.3: ARIMA achieves MAPE < 15% for stable workers
- Hypothesis H6.4: LSTM improves forecast for high-variability workers
- Experiment E6.2: Compare ARIMA vs LSTM on 30-day horizon
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class ForecastConfig:
    """Configuration for forecasting."""
    horizon_days: int = 14
    model_type: str = "ARIMA"  # ARIMA, ETS, LEARNING_CURVE
    confidence_level: float = 0.95
    min_history_days: int = 7
    
    # ARIMA parameters (auto-selected if None)
    arima_order: Optional[Tuple[int, int, int]] = None
    
    # Exponential smoothing
    seasonal_period: Optional[int] = None  # e.g., 7 for weekly


@dataclass
class ForecastPoint:
    """Single forecast point."""
    date: str
    value: float
    lower_bound: float
    upper_bound: float


@dataclass
class WorkforceForecast:
    """
    Forecast result for a worker.
    """
    worker_id: str
    model_type: str
    
    # Forecast values
    forecast_values: List[ForecastPoint] = field(default_factory=list)
    forecast_mean: float = 0.0
    forecast_trend: str = "stable"  # increasing, decreasing, stable
    
    # Model quality
    mape: float = 0.0  # Mean Absolute Percentage Error
    rmse: float = 0.0  # Root Mean Square Error
    r_squared: float = 0.0
    
    # SNR
    snr_forecast: float = 1.0
    snr_level: str = "FAIR"
    confidence_score: float = 0.5
    
    # Metadata
    history_length: int = 0
    created_at: str = ""
    horizon_days: int = 14
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'worker_id': self.worker_id,
            'model_type': self.model_type,
            'forecast_values': [
                {
                    'date': fp.date,
                    'value': round(fp.value, 2),
                    'lower_bound': round(fp.lower_bound, 2),
                    'upper_bound': round(fp.upper_bound, 2),
                }
                for fp in self.forecast_values
            ],
            'forecast_mean': round(self.forecast_mean, 2),
            'forecast_trend': self.forecast_trend,
            'mape': round(self.mape, 2),
            'rmse': round(self.rmse, 2),
            'r_squared': round(self.r_squared, 3),
            'snr_forecast': round(self.snr_forecast, 2),
            'snr_level': self.snr_level,
            'confidence_score': round(self.confidence_score, 3),
            'history_length': self.history_length,
            'horizon_days': self.horizon_days,
            'created_at': self.created_at,
        }


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# ARIMA FORECASTING
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def _forecast_arima(
    history: np.ndarray,
    horizon: int,
    order: Optional[Tuple[int, int, int]] = None,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    ARIMA forecast implementation.
    
    Falls back to simple methods if statsmodels unavailable.
    
    Args:
        history: Historical productivity values
        horizon: Number of periods to forecast
        order: ARIMA(p, d, q) order, auto-selected if None
        confidence: Confidence level for intervals
    
    Returns:
        (forecast, lower_bound, upper_bound, metrics)
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
        
        # Auto-select order if not provided
        if order is None:
            # Test stationarity
            try:
                adf_result = adfuller(history, autolag='AIC')
                p_value = adf_result[1]
                d = 0 if p_value < 0.05 else 1
            except:
                d = 1
            
            # Simple order selection
            order = (2, d, 1)
        
        # Fit ARIMA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(history, order=order)
            fitted = model.fit()
        
        # Forecast
        forecast_result = fitted.get_forecast(steps=horizon)
        forecast = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=1 - confidence)
        
        lower = conf_int.iloc[:, 0].values
        upper = conf_int.iloc[:, 1].values
        
        # Compute metrics on in-sample
        residuals = fitted.resid
        mape = np.mean(np.abs(residuals / history)) * 100 if np.all(history != 0) else 0
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        # R² approximation
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((history - np.mean(history)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        metrics = {
            'mape': mape,
            'rmse': rmse,
            'r_squared': max(0, r_squared),
            'aic': fitted.aic if hasattr(fitted, 'aic') else 0,
        }
        
        return forecast, lower, upper, metrics
    
    except ImportError:
        logger.warning("statsmodels not available. Using simple forecast.")
        return _forecast_simple(history, horizon, confidence)
    except Exception as e:
        logger.warning(f"ARIMA failed: {e}. Using simple forecast.")
        return _forecast_simple(history, horizon, confidence)


def _forecast_simple(
    history: np.ndarray,
    horizon: int,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Simple forecast using exponential weighted average.
    
    Fallback when statsmodels unavailable.
    """
    # Exponential weighted mean
    weights = np.exp(np.linspace(-1, 0, len(history)))
    weights /= weights.sum()
    
    mean = np.average(history, weights=weights)
    std = np.std(history)
    
    # Simple trend (linear regression slope)
    x = np.arange(len(history))
    if len(history) > 1:
        slope = np.polyfit(x, history, 1)[0]
    else:
        slope = 0
    
    # Forecast with trend
    forecast = np.array([mean + slope * i for i in range(1, horizon + 1)])
    
    # Confidence intervals (widen over time)
    # Use z-score approximation without scipy
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)
    
    try:
        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)
    except ImportError:
        pass
    
    margin = z * std * np.sqrt(1 + np.arange(1, horizon + 1) / len(history))
    
    lower = forecast - margin
    upper = forecast + margin
    
    # Metrics
    fitted = mean + slope * (x - len(history) // 2)
    residuals = history - fitted
    mape = np.mean(np.abs(residuals / history)) * 100 if np.all(history != 0) else 0
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    metrics = {
        'mape': min(mape, 100),
        'rmse': rmse,
        'r_squared': max(0, 1 - np.var(residuals) / np.var(history)) if np.var(history) > 0 else 0,
        'aic': 0,
    }
    
    return forecast, lower, upper, metrics


def _forecast_learning_curve(
    history: np.ndarray,
    horizon: int,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Forecast using fitted learning curve.
    
    Good for workers still in learning phase.
    """
    from .workforce_performance_engine import fit_learning_curve
    
    times = np.arange(len(history))
    params = fit_learning_curve(times, history)
    
    # Forecast
    future_times = np.arange(len(history), len(history) + horizon)
    forecast = np.array([params.predict(t) for t in future_times])
    
    # Confidence based on residuals
    fitted = np.array([params.predict(t) for t in times])
    residuals = history - fitted
    std = np.std(residuals)
    
    from scipy import stats
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * std * 1.2  # Slightly wider for extrapolation
    
    lower = forecast - margin
    upper = forecast + margin
    
    metrics = {
        'mape': np.mean(np.abs(residuals / history)) * 100 if np.all(history != 0) else 0,
        'rmse': np.sqrt(np.mean(residuals ** 2)),
        'r_squared': params.r_squared,
        'aic': 0,
    }
    
    return forecast, lower, upper, metrics


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# SNR FOR FORECAST
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def compute_forecast_snr(
    forecast: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray
) -> Tuple[float, str]:
    """
    Compute SNR for forecast.
    
    SNR = Var(forecast) / Var(uncertainty)
    
    where uncertainty = (upper - lower) / 2
    """
    uncertainty = (upper - lower) / 2
    
    var_forecast = np.var(forecast)
    var_uncertainty = np.var(uncertainty)
    
    if var_uncertainty < 1e-10:
        snr = 10.0 if var_forecast > 0 else 1.0
    else:
        snr = var_forecast / var_uncertainty
    
    snr = min(max(snr, 0.1), 100.0)
    
    if snr >= 10:
        level = "EXCELLENT"
    elif snr >= 5:
        level = "GOOD"
    elif snr >= 2:
        level = "FAIR"
    else:
        level = "POOR"
    
    return snr, level


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# MAIN FORECAST FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def forecast_worker_productivity(
    worker_id: str,
    productivity_history: List[float],
    dates: Optional[List[str]] = None,
    config: Optional[ForecastConfig] = None
) -> WorkforceForecast:
    """
    Forecast productivity for a single worker.
    
    Args:
        worker_id: Worker identifier
        productivity_history: Historical productivity values (daily)
        dates: Optional date labels for history
        config: Forecast configuration
    
    Returns:
        WorkforceForecast with predictions and metrics
    """
    config = config or ForecastConfig()
    
    result = WorkforceForecast(
        worker_id=worker_id,
        model_type=config.model_type,
        history_length=len(productivity_history),
        horizon_days=config.horizon_days,
        created_at=datetime.now().isoformat(),
    )
    
    # Validate history
    history = np.array(productivity_history, dtype=np.float64)
    history = history[~np.isnan(history)]
    
    if len(history) < config.min_history_days:
        logger.warning(f"Insufficient history for worker {worker_id}: {len(history)} < {config.min_history_days}")
        # Return simple average forecast
        avg = np.mean(history) if len(history) > 0 else 0
        std = np.std(history) if len(history) > 1 else avg * 0.1
        
        last_date = datetime.now()
        if dates and len(dates) > 0:
            try:
                last_date = pd.to_datetime(dates[-1])
            except:
                pass
        
        for i in range(config.horizon_days):
            d = last_date + timedelta(days=i + 1)
            result.forecast_values.append(ForecastPoint(
                date=d.strftime('%Y-%m-%d'),
                value=avg,
                lower_bound=avg - 1.96 * std,
                upper_bound=avg + 1.96 * std,
            ))
        
        result.forecast_mean = avg
        result.confidence_score = 0.3
        return result
    
    # Select forecasting method
    if config.model_type == "LEARNING_CURVE":
        forecast, lower, upper, metrics = _forecast_learning_curve(
            history, config.horizon_days, config.confidence_level
        )
    elif config.model_type == "ARIMA":
        forecast, lower, upper, metrics = _forecast_arima(
            history, config.horizon_days, config.arima_order, config.confidence_level
        )
    else:  # Fallback
        forecast, lower, upper, metrics = _forecast_simple(
            history, config.horizon_days, config.confidence_level
        )
    
    # Determine dates
    last_date = datetime.now()
    if dates and len(dates) > 0:
        try:
            last_date = pd.to_datetime(dates[-1])
        except:
            pass
    
    # Build forecast points
    for i in range(len(forecast)):
        d = last_date + timedelta(days=i + 1)
        result.forecast_values.append(ForecastPoint(
            date=d.strftime('%Y-%m-%d'),
            value=max(0, float(forecast[i])),
            lower_bound=max(0, float(lower[i])),
            upper_bound=float(upper[i]),
        ))
    
    # Aggregate metrics
    result.forecast_mean = float(np.mean(forecast))
    result.mape = metrics['mape']
    result.rmse = metrics['rmse']
    result.r_squared = metrics['r_squared']
    
    # Trend
    if len(forecast) >= 2:
        slope = (forecast[-1] - forecast[0]) / len(forecast)
        if slope > 0.05 * result.forecast_mean:
            result.forecast_trend = "increasing"
        elif slope < -0.05 * result.forecast_mean:
            result.forecast_trend = "decreasing"
        else:
            result.forecast_trend = "stable"
    
    # SNR
    snr, level = compute_forecast_snr(forecast, lower, upper)
    result.snr_forecast = snr
    result.snr_level = level
    
    # Confidence score (combining model fit and SNR)
    model_confidence = max(0, min(1, result.r_squared))
    snr_confidence = snr / (1 + snr)
    mape_confidence = max(0, 1 - result.mape / 100)
    
    result.confidence_score = 0.4 * model_confidence + 0.3 * snr_confidence + 0.3 * mape_confidence
    
    return result


def forecast_all_workers(
    performances: Dict[str, 'WorkerPerformance'],
    config: Optional[ForecastConfig] = None
) -> Dict[str, WorkforceForecast]:
    """
    Forecast productivity for all workers.
    
    Args:
        performances: Dict of WorkerPerformance objects
        config: Forecast configuration
    
    Returns:
        Dict mapping worker_id -> WorkforceForecast
    """
    from .workforce_performance_engine import WorkerPerformance
    
    forecasts = {}
    
    for worker_id, perf in performances.items():
        if not perf.productivity_history:
            continue
        
        forecast = forecast_worker_productivity(
            worker_id=worker_id,
            productivity_history=perf.productivity_history,
            dates=perf.dates,
            config=config,
        )
        
        forecasts[worker_id] = forecast
    
    return forecasts


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# TODO[R&D]: ADVANCED ML FORECASTING
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def forecast_lstm(
    history: np.ndarray,
    horizon: int,
    sequence_length: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    TODO[R&D]: LSTM-based forecasting for complex patterns.
    
    Requirements:
    - TensorFlow or PyTorch
    - Sufficient historical data (100+ samples)
    - Feature engineering (time features, external factors)
    
    Architecture:
    - Input: [batch, sequence_length, features]
    - LSTM layers: 2x64 units
    - Dense output: horizon predictions
    - Uncertainty: Monte Carlo dropout or ensemble
    
    Expected to outperform ARIMA when:
    - High variability (low SNR in ARIMA)
    - Strong weekly/monthly patterns
    - External factors available
    """
    logger.warning("LSTM forecasting not implemented. Using ARIMA fallback.")
    return _forecast_arima(history, horizon)


def forecast_transformer(
    history: np.ndarray,
    horizon: int,
    d_model: int = 64,
    n_heads: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    TODO[R&D]: Transformer-based forecasting.
    
    Potential advantages:
    - Better long-range dependencies
    - Attention mechanism reveals important time steps
    - Parallel training
    
    Architecture:
    - Positional encoding
    - Multi-head self-attention
    - Feed-forward layers
    - Probabilistic output (mean + variance)
    
    References:
    - Vaswani et al. (2017). Attention Is All You Need.
    - Lim et al. (2021). Temporal Fusion Transformers.
    """
    logger.warning("Transformer forecasting not implemented. Using ARIMA fallback.")
    return _forecast_arima(history, horizon)

