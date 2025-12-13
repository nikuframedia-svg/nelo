"""
ProdPlan 4.0 - Unified Forecasting Engine
=========================================

Motor de forecasting unificado para SmartInventory.

Arquitetura:
- ForecastEngineBase: Interface abstrata
- ClassicalForecastEngine: ETS/ARIMA/XGBoost (produção, BASE)
- AdvancedForecastEngine: N-HiTS/TFT (R&D, ADVANCED)

Feature Flags:
- ForecastEngine.BASE → ClassicalForecastEngine
- ForecastEngine.ADVANCED → AdvancedForecastEngine (com fallback)

Integração com rop_engine.py para cálculo de ROP dinâmico.

R&D / SIFIDE: WP3 - Inventory & Capacity Optimization
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class ForecastModelType(str, Enum):
    """Tipos de modelo de forecast."""
    ETS = "ets"           # Exponential Smoothing
    ARIMA = "arima"       # ARIMA
    XGBOOST = "xgboost"   # XGBoost
    PROPHET = "prophet"   # Prophet
    NHITS = "nhits"       # N-HiTS
    NBEATS = "nbeats"     # N-BEATS
    TFT = "tft"           # Temporal Fusion Transformer
    ENSEMBLE = "ensemble" # Ensemble de modelos


class SNRClass(str, Enum):
    """Classificação de Signal-to-Noise Ratio."""
    HIGH = "HIGH"       # SNR > 8
    MEDIUM = "MEDIUM"   # 3 < SNR <= 8
    LOW = "LOW"         # SNR <= 3


@dataclass
class ForecastConfig:
    """Configuração de forecast."""
    horizon_days: int = 30
    confidence_level: float = 0.95
    use_external_regressors: bool = False
    seasonal_period: Optional[int] = None
    model_type: ForecastModelType = ForecastModelType.ETS
    # Advanced settings
    min_history_for_advanced: int = 100  # Mínimo de pontos para usar modelo avançado
    use_quantile_regression: bool = False


@dataclass
class ForecastResult:
    """
    Resultado de um forecast.
    
    Attributes:
        sku_id: SKU previsto
        forecast_values: Valores previstos por período
        forecast_dates: Datas correspondentes
        lower_bound: Limite inferior do intervalo de confiança
        upper_bound: Limite superior do intervalo de confiança
        mean_daily: Média diária prevista
        std_daily: Desvio padrão diário
        snr: Signal-to-Noise Ratio
        snr_class: Classificação do SNR
        confidence_score: Score de confiança (0-1)
        model_used: Modelo utilizado
        metrics: Métricas de avaliação (MAPE, RMSE, etc.)
        is_advanced: Se usou modelo avançado
    """
    sku_id: str
    forecast_values: List[float]
    forecast_dates: List[datetime]
    lower_bound: List[float]
    upper_bound: List[float]
    mean_daily: float
    std_daily: float
    snr: float
    snr_class: SNRClass
    confidence_score: float
    model_used: ForecastModelType
    metrics: Dict[str, float] = field(default_factory=dict)
    is_advanced: bool = False
    
    def get_series(self) -> pd.Series:
        """Retorna forecast como Series."""
        return pd.Series(self.forecast_values, index=self.forecast_dates)
    
    def get_total(self) -> float:
        """Retorna soma total do forecast."""
        return sum(self.forecast_values)
    
    @property
    def mape(self) -> Optional[float]:
        """MAPE se disponível."""
        return self.metrics.get("mape")
    
    @property
    def rmse(self) -> Optional[float]:
        """RMSE se disponível."""
        return self.metrics.get("rmse")


# ═══════════════════════════════════════════════════════════════════════════════
# BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class ForecastEngineBase(ABC):
    """
    Interface abstrata para motores de forecast.
    
    Todas as implementações devem seguir esta interface.
    """
    
    @abstractmethod
    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> ForecastResult:
        """
        Gera forecast para uma série temporal.
        
        Args:
            series: Série histórica (index: datetime, values: quantidade)
            horizon: Número de períodos a prever
            context: Contexto adicional (sazonalidade, regressores, etc.)
        
        Returns:
            ForecastResult
        """
        pass
    
    @abstractmethod
    def fit(self, series: pd.Series) -> None:
        """Ajusta modelo aos dados históricos."""
        pass
    
    def compute_snr(self, signal: np.ndarray, residuals: np.ndarray) -> Tuple[float, SNRClass]:
        """
        Calcula Signal-to-Noise Ratio.
        
        SNR = Var(signal) / Var(noise)
        """
        signal_var = np.var(signal) if len(signal) > 1 else 0
        noise_var = np.var(residuals) if len(residuals) > 1 else 1
        
        snr = signal_var / noise_var if noise_var > 0 else 0
        
        if snr > 8:
            snr_class = SNRClass.HIGH
        elif snr > 3:
            snr_class = SNRClass.MEDIUM
        else:
            snr_class = SNRClass.LOW
        
        return snr, snr_class
    
    def _compute_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calcula MAPE."""
        mask = actual != 0
        if not np.any(mask):
            return 0.0
        return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
    
    def _compute_rmse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calcula RMSE."""
        return float(np.sqrt(np.mean((actual - predicted) ** 2)))
    
    def _snr_to_confidence(self, snr: float) -> float:
        """Converte SNR para score de confiança."""
        if snr > 10:
            return 0.95
        elif snr > 5:
            return 0.8
        elif snr > 2:
            return 0.6
        else:
            return 0.4


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSICAL FORECAST ENGINE (BASE)
# ═══════════════════════════════════════════════════════════════════════════════

class ClassicalForecastEngine(ForecastEngineBase):
    """
    Motor de forecast com modelos clássicos.
    
    Implementa:
    - ETS (Exponential Smoothing)
    - ARIMA
    - XGBoost (regressão com features temporais)
    
    Fallback automático se modelo principal falhar.
    """
    
    def __init__(
        self,
        model_type: ForecastModelType = ForecastModelType.ETS,
        confidence_level: float = 0.95,
    ):
        self.model_type = model_type
        self.confidence_level = confidence_level
        self._fitted_model = None
        self._history: Optional[pd.Series] = None
    
    def fit(self, series: pd.Series) -> None:
        """Ajusta modelo aos dados históricos."""
        self._history = series.copy()
    
    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> ForecastResult:
        """Gera forecast."""
        context = context or {}
        sku_id = context.get("sku_id", "unknown")
        
        # Garantir que série tem dados suficientes
        if len(series) < 3:
            return self._naive_forecast(series, horizon, sku_id)
        
        # Tentar modelo principal
        try:
            if self.model_type == ForecastModelType.ARIMA:
                return self._forecast_arima(series, horizon, sku_id)
            elif self.model_type == ForecastModelType.XGBOOST:
                return self._forecast_xgboost(series, horizon, sku_id)
            else:
                return self._forecast_ets(series, horizon, sku_id)
        except Exception as e:
            logger.warning(f"Model {self.model_type} failed: {e}. Falling back to naive.")
            return self._naive_forecast(series, horizon, sku_id)
    
    def _forecast_ets(
        self,
        series: pd.Series,
        horizon: int,
        sku_id: str,
    ) -> ForecastResult:
        """Forecast com Exponential Smoothing."""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            # Configuração adaptativa
            if len(series) >= 24:
                model = ExponentialSmoothing(
                    series,
                    trend='add',
                    seasonal='add' if len(series) >= 24 else None,
                    seasonal_periods=12 if len(series) >= 24 else None,
                )
            else:
                model = ExponentialSmoothing(series, trend='add')
            
            fitted = model.fit()
            forecast = fitted.forecast(horizon)
            
            # Calcular intervalos de confiança
            residuals = fitted.resid
            std_resid = np.std(residuals) if len(residuals) > 0 else 1
            
            z = 1.96  # 95% confidence
            lower = forecast - z * std_resid
            upper = forecast + z * std_resid
            
            # Calcular SNR
            snr, snr_class = self.compute_snr(series.values, residuals.values)
            
            # Calcular métricas
            mape = self._compute_mape(series.values, fitted.fittedvalues.values)
            rmse = self._compute_rmse(series.values, fitted.fittedvalues.values)
            
            return ForecastResult(
                sku_id=sku_id,
                forecast_values=forecast.tolist(),
                forecast_dates=list(forecast.index) if hasattr(forecast, 'index') else self._generate_dates(series, horizon),
                lower_bound=lower.tolist(),
                upper_bound=upper.tolist(),
                mean_daily=float(np.mean(forecast)),
                std_daily=float(np.std(forecast)),
                snr=float(snr),
                snr_class=snr_class,
                confidence_score=self._snr_to_confidence(snr),
                model_used=ForecastModelType.ETS,
                metrics={"mape": mape, "rmse": rmse},
                is_advanced=False,
            )
            
        except ImportError:
            logger.warning("statsmodels not available, using naive forecast")
            return self._naive_forecast(series, horizon, sku_id)
    
    def _forecast_arima(
        self,
        series: pd.Series,
        horizon: int,
        sku_id: str,
    ) -> ForecastResult:
        """Forecast com ARIMA."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            # Auto-order simples
            model = ARIMA(series, order=(1, 1, 1))
            fitted = model.fit()
            
            forecast_result = fitted.get_forecast(steps=horizon)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
            
            residuals = fitted.resid
            snr, snr_class = self.compute_snr(series.values, residuals.values)
            
            return ForecastResult(
                sku_id=sku_id,
                forecast_values=forecast.tolist(),
                forecast_dates=self._generate_dates(series, horizon),
                lower_bound=conf_int.iloc[:, 0].tolist(),
                upper_bound=conf_int.iloc[:, 1].tolist(),
                mean_daily=float(np.mean(forecast)),
                std_daily=float(np.std(forecast)),
                snr=float(snr),
                snr_class=snr_class,
                confidence_score=self._snr_to_confidence(snr),
                model_used=ForecastModelType.ARIMA,
                metrics={"aic": fitted.aic},
                is_advanced=False,
            )
            
        except Exception as e:
            logger.warning(f"ARIMA failed: {e}")
            return self._forecast_ets(series, horizon, sku_id)
    
    def _forecast_xgboost(
        self,
        series: pd.Series,
        horizon: int,
        sku_id: str,
    ) -> ForecastResult:
        """Forecast com XGBoost (usando features temporais)."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            
            # Criar features temporais
            X, y = self._create_temporal_features(series, lag=7)
            
            if len(X) < 10:
                return self._naive_forecast(series, horizon, sku_id)
            
            model = GradientBoostingRegressor(n_estimators=50, max_depth=3)
            model.fit(X, y)
            
            # Prever
            forecast_values = []
            last_values = series.values[-7:].tolist()
            
            for _ in range(horizon):
                X_pred = np.array(last_values[-7:]).reshape(1, -1)
                pred = model.predict(X_pred)[0]
                forecast_values.append(max(0, pred))
                last_values.append(pred)
            
            # Calcular residuais
            y_pred = model.predict(X)
            residuals = y - y_pred
            
            snr, snr_class = self.compute_snr(series.values, residuals)
            std_resid = np.std(residuals)
            
            return ForecastResult(
                sku_id=sku_id,
                forecast_values=forecast_values,
                forecast_dates=self._generate_dates(series, horizon),
                lower_bound=[v - 1.96 * std_resid for v in forecast_values],
                upper_bound=[v + 1.96 * std_resid for v in forecast_values],
                mean_daily=float(np.mean(forecast_values)),
                std_daily=float(np.std(forecast_values)),
                snr=float(snr),
                snr_class=snr_class,
                confidence_score=self._snr_to_confidence(snr),
                model_used=ForecastModelType.XGBOOST,
                metrics={"r2": float(model.score(X, y))},
                is_advanced=False,
            )
            
        except ImportError:
            logger.warning("sklearn not available")
            return self._forecast_ets(series, horizon, sku_id)
    
    def _naive_forecast(
        self,
        series: pd.Series,
        horizon: int,
        sku_id: str,
    ) -> ForecastResult:
        """Forecast naive (média)."""
        mean_val = float(series.mean()) if len(series) > 0 else 0
        std_val = float(series.std()) if len(series) > 1 else mean_val * 0.2
        
        forecast_values = [mean_val] * horizon
        dates = self._generate_dates(series, horizon)
        
        return ForecastResult(
            sku_id=sku_id,
            forecast_values=forecast_values,
            forecast_dates=dates,
            lower_bound=[max(0, mean_val - 1.96 * std_val)] * horizon,
            upper_bound=[mean_val + 1.96 * std_val] * horizon,
            mean_daily=mean_val,
            std_daily=std_val,
            snr=1.0,
            snr_class=SNRClass.LOW,
            confidence_score=0.3,
            model_used=ForecastModelType.ETS,
            metrics={"method": "naive_mean"},
            is_advanced=False,
        )
    
    def _generate_dates(self, series: pd.Series, horizon: int) -> List[datetime]:
        """Gera datas para o forecast."""
        if len(series) == 0:
            start = datetime.now()
        elif hasattr(series.index[-1], 'to_pydatetime'):
            start = series.index[-1].to_pydatetime()
        else:
            start = datetime.now()
        
        return [start + timedelta(days=i+1) for i in range(horizon)]
    
    def _create_temporal_features(
        self,
        series: pd.Series,
        lag: int = 7,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cria features temporais para regressão."""
        X, y = [], []
        values = series.values
        
        for i in range(lag, len(values)):
            X.append(values[i-lag:i])
            y.append(values[i])
        
        return np.array(X), np.array(y)


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED FORECAST ENGINE (N-HiTS / TFT)
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedForecastEngine(ForecastEngineBase):
    """
    Motor de forecast avançado com modelos deep learning.
    
    Implementa (com fallback para Classical):
    - N-HiTS (Neural Hierarchical Interpolation for Time Series)
    - N-BEATS (Neural Basis Expansion Analysis)
    - TFT (Temporal Fusion Transformer) - stub
    
    Requer:
    - Mínimo de N pontos de histórico (default: 100)
    - pytorch e darts/neuralforecast opcionais
    
    Se não disponível ou erro: fallback automático para ClassicalForecastEngine
    """
    
    # Mínimo de pontos para usar modelo avançado
    MIN_HISTORY_POINTS = 100
    
    def __init__(
        self,
        model_type: ForecastModelType = ForecastModelType.NHITS,
        min_history: int = 100,
    ):
        self.model_type = model_type
        self.min_history = min_history
        self._classical_engine = ClassicalForecastEngine()
        self._model = None
        self._is_fitted = False
        
        # Verificar disponibilidade de libs
        self._has_neuralforecast = self._check_neuralforecast()
        self._has_darts = self._check_darts()
        
        if not (self._has_neuralforecast or self._has_darts):
            logger.info("AdvancedForecastEngine: No deep learning libs available, will use classical fallback")
    
    def _check_neuralforecast(self) -> bool:
        """Verifica se neuralforecast está disponível."""
        try:
            import neuralforecast
            return True
        except ImportError:
            return False
    
    def _check_darts(self) -> bool:
        """Verifica se darts está disponível."""
        try:
            import darts
            return True
        except ImportError:
            return False
    
    def fit(self, series: pd.Series) -> None:
        """Ajusta modelo aos dados históricos."""
        self._classical_engine.fit(series)
        
        # Tentar ajustar modelo avançado se houver dados suficientes
        if len(series) >= self.min_history:
            try:
                self._fit_advanced(series)
                self._is_fitted = True
            except Exception as e:
                logger.warning(f"Failed to fit advanced model: {e}")
                self._is_fitted = False
    
    def _fit_advanced(self, series: pd.Series) -> None:
        """Ajusta modelo avançado (N-HiTS/N-BEATS)."""
        if self._has_neuralforecast:
            self._fit_neuralforecast(series)
        elif self._has_darts:
            self._fit_darts(series)
    
    def _fit_neuralforecast(self, series: pd.Series) -> None:
        """Ajusta usando neuralforecast."""
        try:
            from neuralforecast import NeuralForecast
            from neuralforecast.models import NHITS, NBEATS
            
            # Preparar dados no formato esperado
            df = pd.DataFrame({
                'unique_id': 'series',
                'ds': series.index,
                'y': series.values,
            })
            
            # Escolher modelo
            horizon = 30  # Default horizon para treino
            if self.model_type == ForecastModelType.NBEATS:
                model = NBEATS(h=horizon, input_size=2*horizon, max_steps=100)
            else:
                model = NHITS(h=horizon, input_size=2*horizon, max_steps=100)
            
            nf = NeuralForecast(models=[model], freq='D')
            nf.fit(df)
            
            self._model = nf
            logger.info(f"Fitted {self.model_type.value} model with neuralforecast")
            
        except Exception as e:
            logger.error(f"neuralforecast fit failed: {e}")
            raise
    
    def _fit_darts(self, series: pd.Series) -> None:
        """Ajusta usando darts."""
        try:
            from darts import TimeSeries
            from darts.models import NBEATSModel
            
            ts = TimeSeries.from_series(series)
            
            model = NBEATSModel(
                input_chunk_length=30,
                output_chunk_length=30,
                n_epochs=50,
                random_state=42,
            )
            model.fit(ts)
            
            self._model = model
            logger.info("Fitted N-BEATS model with darts")
            
        except Exception as e:
            logger.error(f"darts fit failed: {e}")
            raise
    
    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> ForecastResult:
        """
        Gera forecast usando modelo avançado ou fallback.
        
        Lógica:
        1. Se len(series) < min_history → fallback imediato
        2. Caso contrário, tentar modelo avançado
        3. Se erro → fallback para Classical
        """
        context = context or {}
        sku_id = context.get("sku_id", "unknown")
        
        # Verificar se temos dados suficientes
        if len(series) < self.min_history:
            logger.debug(f"Insufficient history ({len(series)} < {self.min_history}), using classical")
            result = self._classical_engine.forecast(series, horizon, context)
            result.is_advanced = False
            return result
        
        # Tentar modelo avançado
        try:
            if self._has_neuralforecast and self._model is not None:
                return self._forecast_neuralforecast(series, horizon, sku_id)
            elif self._has_darts and self._model is not None:
                return self._forecast_darts(series, horizon, sku_id)
            else:
                # Tentar fit on-the-fly
                self.fit(series)
                if self._is_fitted and self._model is not None:
                    if self._has_neuralforecast:
                        return self._forecast_neuralforecast(series, horizon, sku_id)
                    elif self._has_darts:
                        return self._forecast_darts(series, horizon, sku_id)
                
                # Fallback
                logger.debug("No advanced model available, using classical")
                result = self._classical_engine.forecast(series, horizon, context)
                result.is_advanced = False
                return result
                
        except Exception as e:
            logger.warning(f"Advanced forecast failed: {e}. Falling back to classical.")
            result = self._classical_engine.forecast(series, horizon, context)
            result.is_advanced = False
            return result
    
    def _forecast_neuralforecast(
        self,
        series: pd.Series,
        horizon: int,
        sku_id: str,
    ) -> ForecastResult:
        """Forecast usando neuralforecast."""
        from neuralforecast import NeuralForecast
        
        # Preparar dados
        df = pd.DataFrame({
            'unique_id': 'series',
            'ds': series.index,
            'y': series.values,
        })
        
        # Prever
        predictions = self._model.predict(df)
        
        forecast_values = predictions['NHITS' if self.model_type == ForecastModelType.NHITS else 'NBEATS'].values[:horizon].tolist()
        
        # Calcular métricas básicas
        mean_val = float(np.mean(forecast_values))
        std_val = float(np.std(forecast_values)) if len(forecast_values) > 1 else mean_val * 0.2
        
        # Intervalos de confiança (estimativa simples)
        lower = [max(0, v - 1.96 * std_val) for v in forecast_values]
        upper = [v + 1.96 * std_val for v in forecast_values]
        
        return ForecastResult(
            sku_id=sku_id,
            forecast_values=forecast_values,
            forecast_dates=self._generate_dates(series, horizon),
            lower_bound=lower,
            upper_bound=upper,
            mean_daily=mean_val,
            std_daily=std_val,
            snr=5.0,  # Assume medium SNR for deep models
            snr_class=SNRClass.MEDIUM,
            confidence_score=0.8,
            model_used=self.model_type,
            metrics={"model": "neuralforecast"},
            is_advanced=True,
        )
    
    def _forecast_darts(
        self,
        series: pd.Series,
        horizon: int,
        sku_id: str,
    ) -> ForecastResult:
        """Forecast usando darts."""
        from darts import TimeSeries
        
        ts = TimeSeries.from_series(series)
        predictions = self._model.predict(n=horizon)
        
        forecast_values = predictions.values().flatten()[:horizon].tolist()
        
        mean_val = float(np.mean(forecast_values))
        std_val = float(np.std(forecast_values)) if len(forecast_values) > 1 else mean_val * 0.2
        
        lower = [max(0, v - 1.96 * std_val) for v in forecast_values]
        upper = [v + 1.96 * std_val for v in forecast_values]
        
        return ForecastResult(
            sku_id=sku_id,
            forecast_values=forecast_values,
            forecast_dates=self._generate_dates(series, horizon),
            lower_bound=lower,
            upper_bound=upper,
            mean_daily=mean_val,
            std_daily=std_val,
            snr=5.0,
            snr_class=SNRClass.MEDIUM,
            confidence_score=0.8,
            model_used=ForecastModelType.NBEATS,
            metrics={"model": "darts_nbeats"},
            is_advanced=True,
        )
    
    def _generate_dates(self, series: pd.Series, horizon: int) -> List[datetime]:
        """Gera datas para o forecast."""
        if len(series) == 0:
            start = datetime.now()
        elif hasattr(series.index[-1], 'to_pydatetime'):
            start = series.index[-1].to_pydatetime()
        else:
            start = datetime.now()
        
        return [start + timedelta(days=i+1) for i in range(horizon)]


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION (with FeatureFlags integration)
# ═══════════════════════════════════════════════════════════════════════════════

def get_forecast_engine(
    model_type: str = "ets",
    use_advanced: Optional[bool] = None,
) -> ForecastEngineBase:
    """
    Factory function para obter forecast engine baseado em FeatureFlags.
    
    Args:
        model_type: Tipo de modelo ("ets", "arima", "xgboost", "nhits", "tft")
        use_advanced: Forçar modo avançado (se None, usa FeatureFlags)
    
    Returns:
        ForecastEngineBase
    """
    # Importar FeatureFlags
    try:
        from ..feature_flags import FeatureFlags, ForecastEngine as FE
        
        if use_advanced is None:
            use_advanced = FeatureFlags.get_forecast_engine() == FE.ADVANCED
    except ImportError:
        if use_advanced is None:
            use_advanced = False
    
    # Converter model_type
    try:
        model_type_enum = ForecastModelType(model_type.lower())
    except ValueError:
        model_type_enum = ForecastModelType.ETS
    
    # Retornar engine apropriado
    if use_advanced or model_type_enum in (ForecastModelType.NHITS, ForecastModelType.TFT, ForecastModelType.NBEATS):
        return AdvancedForecastEngine(model_type_enum)
    else:
        return ClassicalForecastEngine(model_type_enum)


def forecast_demand(
    sku_id: str,
    historical_data: pd.Series,
    horizon_days: int = 30,
    model: str = "ets",
) -> ForecastResult:
    """
    Função de alto nível para forecast de demanda.
    
    Args:
        sku_id: ID do SKU
        historical_data: Série histórica de consumo
        horizon_days: Dias a prever
        model: Modelo a usar
    
    Returns:
        ForecastResult
    """
    engine = get_forecast_engine(model)
    return engine.forecast(historical_data, horizon_days, context={"sku_id": sku_id})


def create_sample_series(n_days: int = 90, trend: float = 0.1, noise: float = 0.2) -> pd.Series:
    """Cria série temporal de exemplo para testes."""
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Trend + sazonalidade + ruído
    t = np.arange(n_days)
    values = 100 + trend * t + 20 * np.sin(2 * np.pi * t / 7) + noise * 100 * np.random.randn(n_days)
    values = np.maximum(0, values)  # Não negativo
    
    return pd.Series(values, index=dates)
