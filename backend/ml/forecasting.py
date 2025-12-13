"""
ProdPlan 4.0 - Time Series Forecasting

This module provides demand and lead time forecasting capabilities:
- Statistical models: ARIMA, ETS, Holt-Winters
- ML models: XGBoost, LightGBM
- Deep Learning: LSTM, Transformer-based (future)

Design Philosophy:
- Start with simple, interpretable models
- Add complexity only when needed
- Always provide uncertainty estimates

R&D / SIFIDE: WP2 - Predictive Intelligence
Research Questions:
- Q2.1: Can transformer models outperform ARIMA for non-stationary industrial demand?
- Q2.2: What lead time prediction accuracy is achievable with ML?
Metrics: MAPE, RMSE, prediction interval coverage.

References:
- Hyndman & Athanasopoulos (2021). Forecasting: Principles and Practice
- Lim & Zohren (2021). Time-series forecasting with deep learning: A survey
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS AND CONFIG
# ============================================================

class ForecastModel(Enum):
    """Available forecasting models."""
    NAIVE = "naive"  # Last value or seasonal naive
    MOVING_AVERAGE = "moving_average"
    ETS = "ets"  # Exponential smoothing
    ARIMA = "arima"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LSTM = "lstm"
    TRANSFORMER = "transformer"


@dataclass
class ForecastConfig:
    """Configuration for forecasting."""
    model: ForecastModel = ForecastModel.MOVING_AVERAGE
    
    # Forecast horizon
    horizon_periods: int = 7  # Number of periods to forecast
    frequency: str = "D"  # D=daily, W=weekly, M=monthly
    
    # Model-specific parameters
    window_size: int = 7  # For moving average
    arima_order: Tuple[int, int, int] = (1, 1, 1)
    seasonal_period: int = 7
    
    # ML parameters
    n_lags: int = 14  # Number of lag features
    n_estimators: int = 100
    
    # Uncertainty
    confidence_level: float = 0.95
    n_simulations: int = 100  # For bootstrap intervals
    
    # Training
    train_test_split: float = 0.8
    cross_validation_folds: int = 5


@dataclass
class ForecastResult:
    """Result of a forecast."""
    # Point forecasts
    point_forecast: pd.Series
    
    # Prediction intervals
    lower_bound: Optional[pd.Series] = None
    upper_bound: Optional[pd.Series] = None
    confidence_level: float = 0.95
    
    # Metadata
    model_name: str = ""
    fit_time_sec: float = 0.0
    predict_time_sec: float = 0.0
    
    # Quality metrics (from backtesting)
    mape: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        df = pd.DataFrame({
            'forecast': self.point_forecast,
        })
        if self.lower_bound is not None:
            df['lower'] = self.lower_bound
        if self.upper_bound is not None:
            df['upper'] = self.upper_bound
        return df


# ============================================================
# ABSTRACT BASE CLASS
# ============================================================

class BaseForecaster(ABC):
    """
    Abstract base class for forecasters.
    
    Implements common interface for all forecasting models.
    """
    
    def __init__(self, config: Optional[ForecastConfig] = None):
        self.config = config or ForecastConfig()
        self._is_fitted = False
        self._model = None
    
    @abstractmethod
    def fit(self, series: pd.Series) -> None:
        """Fit model to historical data."""
        pass
    
    @abstractmethod
    def predict(self, horizon: int) -> ForecastResult:
        """Generate forecast for given horizon."""
        pass
    
    def fit_predict(self, series: pd.Series, horizon: Optional[int] = None) -> ForecastResult:
        """Fit and predict in one step."""
        self.fit(series)
        h = horizon or self.config.horizon_periods
        return self.predict(h)
    
    @abstractmethod
    def get_name(self) -> str:
        """Get model name."""
        pass


# ============================================================
# SIMPLE FORECASTERS (Always available)
# ============================================================

class NaiveForecaster(BaseForecaster):
    """
    Naive forecaster: last value repeated.
    
    Useful as baseline for comparison.
    """
    
    def __init__(self, config: Optional[ForecastConfig] = None):
        super().__init__(config)
        self._last_value = None
        self._std = None
    
    def fit(self, series: pd.Series) -> None:
        self._last_value = series.iloc[-1]
        self._std = series.std()
        self._is_fitted = True
    
    def predict(self, horizon: int) -> ForecastResult:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        # Point forecast: repeat last value
        index = pd.date_range(start=datetime.now(), periods=horizon, freq=self.config.frequency)
        point = pd.Series([self._last_value] * horizon, index=index)
        
        # Prediction intervals (assuming normal errors)
        z = 1.96  # 95% CI
        lower = pd.Series([self._last_value - z * self._std] * horizon, index=index)
        upper = pd.Series([self._last_value + z * self._std] * horizon, index=index)
        
        return ForecastResult(
            point_forecast=point,
            lower_bound=lower,
            upper_bound=upper,
            model_name="Naive",
        )
    
    def get_name(self) -> str:
        return "Naive"


class MovingAverageForecaster(BaseForecaster):
    """
    Moving average forecaster.
    
    Simple but robust for stable series.
    """
    
    def __init__(self, config: Optional[ForecastConfig] = None):
        super().__init__(config)
        self._ma_value = None
        self._std = None
    
    def fit(self, series: pd.Series) -> None:
        window = min(self.config.window_size, len(series))
        self._ma_value = series.iloc[-window:].mean()
        self._std = series.iloc[-window:].std()
        self._is_fitted = True
    
    def predict(self, horizon: int) -> ForecastResult:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        index = pd.date_range(start=datetime.now(), periods=horizon, freq=self.config.frequency)
        point = pd.Series([self._ma_value] * horizon, index=index)
        
        z = 1.96
        lower = pd.Series([self._ma_value - z * self._std] * horizon, index=index)
        upper = pd.Series([self._ma_value + z * self._std] * horizon, index=index)
        
        return ForecastResult(
            point_forecast=point,
            lower_bound=lower,
            upper_bound=upper,
            model_name="MovingAverage",
        )
    
    def get_name(self) -> str:
        return "MovingAverage"


class ExponentialSmoothingForecaster(BaseForecaster):
    """
    Exponential smoothing (ETS) forecaster.
    
    Good for series with trend and/or seasonality.
    Uses statsmodels if available, falls back to simple implementation.
    """
    
    def __init__(self, config: Optional[ForecastConfig] = None):
        super().__init__(config)
        self._level = None
        self._trend = None
        self._alpha = 0.3  # Smoothing parameter
        self._beta = 0.1   # Trend smoothing
    
    def fit(self, series: pd.Series) -> None:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            self._model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal=None,  # TODO: Add seasonal support
            ).fit()
            self._use_statsmodels = True
            
        except ImportError:
            logger.warning("statsmodels not available, using simple ETS")
            # Simple Holt's linear method
            self._level = series.iloc[-1]
            if len(series) > 1:
                self._trend = (series.iloc[-1] - series.iloc[-2])
            else:
                self._trend = 0
            self._use_statsmodels = False
        
        self._is_fitted = True
    
    def predict(self, horizon: int) -> ForecastResult:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        if hasattr(self, '_use_statsmodels') and self._use_statsmodels:
            forecast = self._model.forecast(horizon)
            index = forecast.index
            point = forecast
            
            # Confidence intervals from model
            # TODO: Implement proper intervals
            std = forecast.std() if len(forecast) > 1 else forecast.iloc[0] * 0.1
            lower = point - 1.96 * std
            upper = point + 1.96 * std
        else:
            index = pd.date_range(start=datetime.now(), periods=horizon, freq=self.config.frequency)
            point_values = [self._level + (i + 1) * self._trend for i in range(horizon)]
            point = pd.Series(point_values, index=index)
            
            std = abs(self._trend) * 2 + 1  # Simple uncertainty estimate
            lower = point - 1.96 * std
            upper = point + 1.96 * std
        
        return ForecastResult(
            point_forecast=point,
            lower_bound=lower,
            upper_bound=upper,
            model_name="ETS",
        )
    
    def get_name(self) -> str:
        return "ExponentialSmoothing"


# ============================================================
# ADVANCED FORECASTERS (Require additional dependencies)
# ============================================================

class ARIMAForecaster(BaseForecaster):
    """
    ARIMA forecaster.
    
    Good for stationary series with autocorrelation.
    Requires statsmodels.
    
    TODO[R&D]: Implement auto ARIMA:
    - Use pmdarima for automatic order selection
    - Compare with manual order selection
    - Measure computation time vs accuracy tradeoff
    """
    
    def __init__(self, config: Optional[ForecastConfig] = None):
        super().__init__(config)
    
    def fit(self, series: pd.Series) -> None:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            order = self.config.arima_order
            self._model = ARIMA(series, order=order).fit()
            self._is_fitted = True
            
        except ImportError:
            raise ImportError("statsmodels required for ARIMA. Run: pip install statsmodels")
    
    def predict(self, horizon: int) -> ForecastResult:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        forecast_result = self._model.get_forecast(steps=horizon)
        point = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=1 - self.config.confidence_level)
        
        return ForecastResult(
            point_forecast=point,
            lower_bound=conf_int.iloc[:, 0],
            upper_bound=conf_int.iloc[:, 1],
            confidence_level=self.config.confidence_level,
            model_name="ARIMA",
        )
    
    def get_name(self) -> str:
        return f"ARIMA{self.config.arima_order}"


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost-based forecaster.
    
    Uses lagged features for ML-based forecasting.
    Requires xgboost.
    
    TODO[R&D]: Feature engineering research:
    - Calendar features (day of week, month, holidays)
    - External regressors (orders, promotions)
    - Feature selection impact on accuracy
    """
    
    def __init__(self, config: Optional[ForecastConfig] = None):
        super().__init__(config)
        self._scaler = None
        self._last_values = None
    
    def _create_features(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Create lag features for ML model."""
        n_lags = self.config.n_lags
        values = series.values
        
        X, y = [], []
        for i in range(n_lags, len(values)):
            X.append(values[i-n_lags:i])
            y.append(values[i])
        
        return np.array(X), np.array(y)
    
    def fit(self, series: pd.Series) -> None:
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost required. Run: pip install xgboost")
        
        X, y = self._create_features(series)
        
        self._model = xgb.XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        self._model.fit(X, y)
        
        # Store last values for prediction
        self._last_values = series.values[-self.config.n_lags:]
        self._std = series.std()
        self._is_fitted = True
    
    def predict(self, horizon: int) -> ForecastResult:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        predictions = []
        current_input = self._last_values.copy()
        
        for _ in range(horizon):
            pred = self._model.predict(current_input.reshape(1, -1))[0]
            predictions.append(pred)
            current_input = np.append(current_input[1:], pred)
        
        index = pd.date_range(start=datetime.now(), periods=horizon, freq=self.config.frequency)
        point = pd.Series(predictions, index=index)
        
        # Simple prediction intervals
        lower = point - 1.96 * self._std
        upper = point + 1.96 * self._std
        
        return ForecastResult(
            point_forecast=point,
            lower_bound=lower,
            upper_bound=upper,
            model_name="XGBoost",
        )
    
    def get_name(self) -> str:
        return "XGBoost"


class TransformerForecaster(BaseForecaster):
    """
    Transformer-based forecaster (stub for future implementation).
    
    TODO[R&D]: Implement transformer models:
    - Temporal Fusion Transformer (TFT)
    - Pyraformer for long-range dependencies
    - Non-stationary transformers
    
    Reference: Wu et al. (2022). Non-stationary Transformers
    
    Research Questions:
    - Q2.1: Performance vs ARIMA on non-stationary industrial demand
    - Training data requirements for good performance
    - Computational requirements for inference
    """
    
    def __init__(self, config: Optional[ForecastConfig] = None):
        super().__init__(config)
        logger.warning("TransformerForecaster is a stub - falling back to ETS")
        self._fallback = ExponentialSmoothingForecaster(config)
    
    def fit(self, series: pd.Series) -> None:
        # TODO[R&D]: Implement transformer training
        # - Requires PyTorch or TensorFlow
        # - Need significant training data (1000+ samples)
        # - GPU acceleration recommended
        self._fallback.fit(series)
        self._is_fitted = True
    
    def predict(self, horizon: int) -> ForecastResult:
        result = self._fallback.predict(horizon)
        result.model_name = "Transformer (fallback to ETS)"
        return result
    
    def get_name(self) -> str:
        return "Transformer (stub)"


# ============================================================
# DEMAND FORECASTER (High-level interface)
# ============================================================

class DemandForecaster:
    """
    High-level demand forecasting interface.
    
    Automatically selects best model based on data characteristics.
    
    Usage:
        forecaster = DemandForecaster()
        result = forecaster.forecast(demand_series, horizon=7)
    """
    
    def __init__(self, config: Optional[ForecastConfig] = None):
        self.config = config or ForecastConfig()
        self._model: Optional[BaseForecaster] = None
    
    def forecast(
        self,
        series: pd.Series,
        horizon: Optional[int] = None,
        model: Optional[ForecastModel] = None
    ) -> ForecastResult:
        """
        Generate demand forecast.
        
        Args:
            series: Historical demand series
            horizon: Forecast horizon (default from config)
            model: Specific model to use (default: auto-select)
        
        Returns:
            ForecastResult with predictions and intervals
        """
        h = horizon or self.config.horizon_periods
        model_type = model or self.config.model
        
        # Select forecaster
        self._model = self._get_forecaster(model_type)
        
        # Fit and predict
        return self._model.fit_predict(series, h)
    
    def _get_forecaster(self, model_type: ForecastModel) -> BaseForecaster:
        """Get forecaster for given model type."""
        forecasters = {
            ForecastModel.NAIVE: NaiveForecaster,
            ForecastModel.MOVING_AVERAGE: MovingAverageForecaster,
            ForecastModel.ETS: ExponentialSmoothingForecaster,
            ForecastModel.ARIMA: ARIMAForecaster,
            ForecastModel.XGBOOST: XGBoostForecaster,
            ForecastModel.TRANSFORMER: TransformerForecaster,
        }
        
        forecaster_class = forecasters.get(model_type, MovingAverageForecaster)
        return forecaster_class(self.config)
    
    def auto_select_model(self, series: pd.Series) -> ForecastModel:
        """
        Automatically select best model based on series characteristics.
        
        TODO[R&D]: Implement intelligent model selection:
        - Use series characteristics (length, stationarity, seasonality)
        - Cross-validation for model comparison
        - Meta-learning from historical performance
        """
        n = len(series)
        
        if n < 10:
            return ForecastModel.NAIVE
        elif n < 30:
            return ForecastModel.MOVING_AVERAGE
        elif n < 100:
            return ForecastModel.ETS
        else:
            return ForecastModel.XGBOOST


class LeadTimeForecaster:
    """
    Lead time forecasting for production orders.
    
    Predicts how long orders will take to complete based on:
    - Article characteristics
    - Current factory load
    - Historical lead times
    
    TODO[R&D]: Lead time prediction research:
    - Feature importance analysis
    - Impact of factory state on accuracy
    - Uncertainty quantification for planning
    """
    
    def __init__(self, config: Optional[ForecastConfig] = None):
        self.config = config or ForecastConfig()
        self._model = None
    
    def fit(
        self,
        historical_orders: pd.DataFrame,
        features: List[str] = None
    ) -> None:
        """
        Fit lead time model.
        
        Args:
            historical_orders: DataFrame with completed orders including
                              actual lead times
            features: Feature columns to use (default: auto-detect)
        """
        # TODO: Implement ML-based lead time prediction
        # For now, use simple average by article
        
        if 'article_id' in historical_orders.columns and 'lead_time_hours' in historical_orders.columns:
            self._lead_times_by_article = historical_orders.groupby('article_id')['lead_time_hours'].mean().to_dict()
        else:
            self._lead_times_by_article = {}
        
        self._global_avg = historical_orders['lead_time_hours'].mean() if 'lead_time_hours' in historical_orders.columns else 24.0
        self._is_fitted = True
    
    def predict(self, article_id: str, features: Optional[Dict] = None) -> Tuple[float, float]:
        """
        Predict lead time for an article.
        
        Args:
            article_id: Article identifier
            features: Additional features (load, priority, etc.)
        
        Returns:
            Tuple of (predicted_lead_time_hours, uncertainty)
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            return (24.0, 12.0)  # Default fallback
        
        if article_id in self._lead_times_by_article:
            mean = self._lead_times_by_article[article_id]
        else:
            mean = self._global_avg
        
        # Simple uncertainty estimate
        uncertainty = mean * 0.3
        
        return (mean, uncertainty)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def forecast_demand(
    series: pd.Series,
    horizon: int = 7,
    model: ForecastModel = ForecastModel.MOVING_AVERAGE
) -> ForecastResult:
    """
    Convenience function for demand forecasting.
    
    Args:
        series: Historical demand data
        horizon: Number of periods to forecast
        model: Forecasting model to use
    
    Returns:
        ForecastResult
    """
    config = ForecastConfig(model=model, horizon_periods=horizon)
    forecaster = DemandForecaster(config)
    return forecaster.forecast(series, horizon, model)


def forecast_lead_time(
    article_id: str,
    historical_orders: pd.DataFrame
) -> Tuple[float, float]:
    """
    Convenience function for lead time forecasting.
    
    Returns:
        (predicted_hours, uncertainty_hours)
    """
    forecaster = LeadTimeForecaster()
    forecaster.fit(historical_orders)
    return forecaster.predict(article_id)


def evaluate_forecaster(
    series: pd.Series,
    model: ForecastModel,
    test_size: int = 7
) -> Dict[str, float]:
    """
    Evaluate forecaster on held-out test data.
    
    Returns:
        Dict with MAPE, RMSE, MAE
    """
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    
    config = ForecastConfig(model=model, horizon_periods=test_size)
    forecaster = DemandForecaster(config)
    result = forecaster.forecast(train, test_size, model)
    
    # Align indices for comparison
    predictions = result.point_forecast.values[:len(test)]
    actuals = test.values
    
    # Compute metrics
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    mae = np.mean(np.abs(actuals - predictions))
    
    return {
        'mape': mape,
        'rmse': rmse,
        'mae': mae,
        'model': model.value,
    }



