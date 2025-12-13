"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    DEMAND FORECASTING (ML Avançado)
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Este módulo implementa forecasting de consumo/vendas usando modelos de ML avançados.

Modelos Implementados:
──────────────────────
    MVP (Produção):
        - ARIMA: AutoRegressive Integrated Moving Average (statsmodels)
        - Prophet: Facebook Prophet (decomposição aditiva)
    
    TODO[R&D] (State-of-the-Art):
        - N-BEATS: Neural Basis Expansion Analysis for Time Series
        - Non-Stationary Transformer (NST): Transformer adaptado para séries temporais
        - D-Linear: Linear model com decomposição aprendida

Mathematical Foundations:
─────────────────────────
    Decomposição de série temporal:
        y(t) = Trend(t) + Seasonality(t) + Residual(t)
    
    ARIMA(p, d, q):
        (1 - φ₁B - ... - φₚBᵖ)(1 - B)ᵈ y(t) = (1 + θ₁B + ... + θₑBᵉ) ε(t)
        onde B é o operador de lag, d é diferenciação, p e q são ordens
    
    SNR (Signal-to-Noise Ratio):
        SNR = Var(signal) / Var(noise)
        onde signal = forecasted trend, noise = residuals
    
    Confidence Classification:
        - HIGH: SNR > 8
        - MEDIUM: 3 < SNR ≤ 8
        - LOW: SNR ≤ 3

TODO[R&D]: Research questions:
    - H5.1: N-BEATS reduz MAPE em ≥15% vs ARIMA para séries sazonais?
    - H5.2: External regressors (preços, notícias) melhoram accuracy em ≥10%?
    - H5.3: Ensemble de modelos (ARIMA + N-BEATS + NST) supera modelos individuais?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class ForecastModel(str, Enum):
    """Modelos de forecasting disponíveis."""
    ARIMA = "ARIMA"
    PROPHET = "PROPHET"
    NBEATS = "N-BEATS"  # TODO[R&D]
    NST = "NST"  # TODO[R&D] Non-Stationary Transformer
    DLINEAR = "D-LINEAR"  # TODO[R&D]
    ENSEMBLE = "ENSEMBLE"  # TODO[R&D] Combinação de múltiplos modelos


@dataclass
class ForecastResult:
    """
    Resultado de um forecast de demanda.
    
    Attributes:
        sku: SKU previsto
        forecast_series: Série temporal prevista (date -> quantity)
        lower_ci: Intervalo de confiança inferior (95%)
        upper_ci: Intervalo de confiança superior (95%)
        model_used: Modelo utilizado
        snr: Signal-to-Noise Ratio
        snr_class: Classificação de SNR (HIGH, MEDIUM, LOW)
        confidence_score: Score de confiança (0-1)
        metrics: Métricas de validação (MAPE, RMSE, etc.)
        residuals: Resíduos do modelo (para análise)
        external_signals_used: Sinais externos utilizados
    """
    sku: str
    forecast_series: pd.Series  # index: date, values: quantity
    lower_ci: pd.Series
    upper_ci: pd.Series
    model_used: ForecastModel
    snr: float
    snr_class: str  # "HIGH", "MEDIUM", "LOW"
    confidence_score: float  # 0-1
    metrics: Dict[str, float] = None  # MAPE, RMSE, MAE, etc.
    residuals: Optional[pd.Series] = None
    external_signals_used: List[str] = None
    
    def __post_init__(self):
        """Validação e inicialização."""
        if self.metrics is None:
            self.metrics = {}
        if self.external_signals_used is None:
            self.external_signals_used = []
    
    def to_dict(self) -> Dict:
        """Converte para dicionário (serialização)."""
        return {
            "sku": self.sku,
            "forecast_series": self.forecast_series.to_dict(),
            "lower_ci": self.lower_ci.to_dict(),
            "upper_ci": self.upper_ci.to_dict(),
            "model_used": self.model_used.value,
            "snr": self.snr,
            "snr_class": self.snr_class,
            "confidence_score": self.confidence_score,
            "metrics": self.metrics,
            "external_signals_used": self.external_signals_used,
        }


@dataclass
class ForecastConfig:
    """
    Configuração para forecasting.
    
    Attributes:
        model: Modelo a usar
        horizon_days: Horizonte de previsão (dias)
        confidence_level: Nível de confiança para intervalos (0.95 = 95%)
        use_external_signals: Usar sinais externos como regressores
        seasonality_mode: Modo de sazonalidade ('additive', 'multiplicative')
        auto_arima: Auto-selecionar parâmetros ARIMA
    """
    model: ForecastModel = ForecastModel.ARIMA
    horizon_days: int = 90
    confidence_level: float = 0.95
    use_external_signals: bool = False
    seasonality_mode: str = "additive"
    auto_arima: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# SNR CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_snr_forecast(
    forecast_series: pd.Series,
    residuals: Optional[pd.Series] = None,
    actual_series: Optional[pd.Series] = None,
) -> Tuple[float, str, float]:
    """
    Calcula Signal-to-Noise Ratio para um forecast.
    
    SNR = Var(signal) / Var(noise)
    
    onde:
        signal = forecasted trend (ou média móvel do forecast)
        noise = residuals (forecast - actual) ou variância do forecast
    
    Args:
        forecast_series: Série prevista
        residuals: Resíduos (forecast - actual)
        actual_series: Série real (para calcular resíduos se não fornecidos)
    
    Returns:
        (snr_value, snr_class, confidence_score)
    """
    if residuals is None and actual_series is not None:
        # Calcular resíduos
        aligned = pd.concat([forecast_series, actual_series], axis=1).dropna()
        if len(aligned) > 0:
            residuals = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    
    if residuals is not None and len(residuals) > 1:
        # SNR baseado em resíduos
        var_signal = np.var(forecast_series.values)
        var_noise = np.var(residuals.values)
        
        if var_noise > 1e-9:
            snr_value = var_signal / var_noise
        else:
            snr_value = np.inf  # Perfeito (sem ruído)
    else:
        # SNR simplificado: usar variância do forecast como proxy
        if len(forecast_series) > 1:
            mean_forecast = forecast_series.mean()
            std_forecast = forecast_series.std()
            if std_forecast > 1e-9:
                snr_value = (mean_forecast / std_forecast) ** 2
            else:
                snr_value = np.inf
        else:
            snr_value = 0.0
    
    # Classificar SNR
    if snr_value >= 8.0:
        snr_class = "HIGH"
        confidence = min(1.0, snr_value / 20.0)  # Normalizar para 0-1
    elif snr_value >= 3.0:
        snr_class = "MEDIUM"
        confidence = snr_value / 8.0
    else:
        snr_class = "LOW"
        confidence = snr_value / 3.0
    
    return float(snr_value), snr_class, float(confidence)


# ═══════════════════════════════════════════════════════════════════════════════
# ARIMA FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════

def _forecast_arima(
    history: pd.Series,
    horizon_days: int,
    confidence_level: float = 0.95,
    auto_arima: bool = True,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Forecast usando ARIMA.
    
    Args:
        history: Série histórica (date -> quantity)
        horizon_days: Horizonte de previsão
        confidence_level: Nível de confiança
        auto_arima: Auto-selecionar parâmetros
    
    Returns:
        (forecast, lower_ci, upper_ci, residuals)
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        logger.warning("statsmodels não disponível, usando método simples")
        return _forecast_simple(history, horizon_days, confidence_level)
    
    # Preparar dados
    if not isinstance(history.index, pd.DatetimeIndex):
        history.index = pd.to_datetime(history.index)
    
    history_clean = history.dropna()
    if len(history_clean) < 10:
        logger.warning(f"Histórico muito curto ({len(history_clean)} pontos), usando método simples")
        return _forecast_simple(history, horizon_days, confidence_level)
    
    # Teste de estacionaridade (Dickey-Fuller)
    try:
        adf_result = adfuller(history_clean.values)
        is_stationary = adf_result[1] < 0.05  # p-value < 0.05
    except:
        is_stationary = False
    
    # Auto-ARIMA ou parâmetros fixos
    if auto_arima:
        # Tentar diferentes ordens
        best_aic = np.inf
        best_model = None
        best_order = (1, 1, 1)
        
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(history_clean, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_model = fitted
                            best_order = (p, d, q)
                    except:
                        continue
        
        if best_model is None:
            # Fallback para ordem simples
            best_order = (1, 1, 1)
            model = ARIMA(history_clean, order=best_order)
            best_model = model.fit()
    else:
        # Ordem fixa
        order = (1, 1, 1) if not is_stationary else (1, 0, 1)
        model = ARIMA(history_clean, order=order)
        best_model = model.fit()
    
    # Forecast
    forecast = best_model.forecast(steps=horizon_days)
    conf_int = best_model.get_forecast(steps=horizon_days).conf_int(alpha=1 - confidence_level)
    
    # Criar índices de data
    last_date = history_clean.index[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon_days, freq='D')
    
    forecast_series = pd.Series(forecast.values, index=forecast_dates)
    lower_ci = pd.Series(conf_int.iloc[:, 0].values, index=forecast_dates)
    upper_ci = pd.Series(conf_int.iloc[:, 1].values, index=forecast_dates)
    
    # Resíduos
    residuals = best_model.resid
    
    return forecast_series, lower_ci, upper_ci, residuals


# ═══════════════════════════════════════════════════════════════════════════════
# PROPHET FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════

def _forecast_prophet(
    history: pd.Series,
    horizon_days: int,
    confidence_level: float = 0.95,
    external_signals: Optional[Dict[str, pd.Series]] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Forecast usando Facebook Prophet.
    
    Args:
        history: Série histórica
        horizon_days: Horizonte
        confidence_level: Nível de confiança
        external_signals: Sinais externos (regressores)
    
    Returns:
        (forecast, lower_ci, upper_ci, residuals)
    """
    try:
        from prophet import Prophet
    except ImportError:
        logger.warning("Prophet não disponível, usando ARIMA")
        return _forecast_arima(history, horizon_days, confidence_level)
    
    # Preparar dados para Prophet (ds, y)
    if not isinstance(history.index, pd.DatetimeIndex):
        history.index = pd.to_datetime(history.index)
    
    df = pd.DataFrame({
        'ds': history.index,
        'y': history.values,
    }).dropna()
    
    if len(df) < 10:
        return _forecast_simple(history, horizon_days, confidence_level)
    
    # Criar modelo
    model = Prophet(
        interval_width=confidence_level,
        seasonality_mode='additive',
    )
    
    # Adicionar sinais externos
    if external_signals:
        for signal_name, signal_series in external_signals.items():
            # Alinhar com histórico
            aligned = signal_series.reindex(history.index, method='ffill').fillna(0)
            df[signal_name] = aligned.values
            model.add_regressor(signal_name)
    
    # Treinar
    model.fit(df)
    
    # Forecast
    future = model.make_future_dataframe(periods=horizon_days)
    
    # Adicionar sinais externos ao futuro (usar último valor ou forecast)
    if external_signals:
        for signal_name, signal_series in external_signals.items():
            last_value = signal_series.iloc[-1] if len(signal_series) > 0 else 0
            future[signal_name] = last_value  # Simplificado: usar último valor
    
    forecast_df = model.predict(future)
    
    # Extrair séries
    forecast_series = pd.Series(
        forecast_df['yhat'].values,
        index=pd.to_datetime(forecast_df['ds'])
    ).tail(horizon_days)
    
    lower_ci = pd.Series(
        forecast_df['yhat_lower'].values,
        index=pd.to_datetime(forecast_df['ds'])
    ).tail(horizon_days)
    
    upper_ci = pd.Series(
        forecast_df['yhat_upper'].values,
        index=pd.to_datetime(forecast_df['ds'])
    ).tail(horizon_days)
    
    # Resíduos (forecast vs actual no período histórico)
    historical_forecast = forecast_df['yhat'].head(len(df))
    residuals = pd.Series(df['y'].values - historical_forecast.values, index=df['ds'])
    
    return forecast_series, lower_ci, upper_ci, residuals


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

def _forecast_simple(
    history: pd.Series,
    horizon_days: int,
    confidence_level: float = 0.95,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Forecast simples (média móvel exponencial) como fallback.
    
    Args:
        history: Série histórica
        horizon_days: Horizonte
        confidence_level: Nível de confiança
    
    Returns:
        (forecast, lower_ci, upper_ci, residuals)
    """
    if not isinstance(history.index, pd.DatetimeIndex):
        history.index = pd.to_datetime(history.index)
    
    history_clean = history.dropna()
    if len(history_clean) == 0:
        # Sem dados: forecast zero
        last_date = datetime.now()
        forecast_dates = pd.date_range(start=last_date, periods=horizon_days, freq='D')
        zeros = pd.Series(0.0, index=forecast_dates)
        return zeros, zeros, zeros, pd.Series()
    
    # Média móvel exponencial
    alpha = 0.3
    forecast_value = history_clean.ewm(alpha=alpha).mean().iloc[-1]
    std_value = history_clean.std()
    
    # Criar forecast (constante)
    last_date = history_clean.index[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon_days, freq='D')
    forecast_series = pd.Series(forecast_value, index=forecast_dates)
    
    # Intervalos de confiança (assumindo distribuição normal)
    z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% ou 99%
    lower_ci = forecast_series - z_score * std_value
    upper_ci = forecast_series + z_score * std_value
    
    # Resíduos (simplificado)
    residuals = history_clean - history_clean.ewm(alpha=alpha).mean()
    
    return forecast_series, lower_ci, upper_ci, residuals


# ═══════════════════════════════════════════════════════════════════════════════
# N-BEATS (TODO[R&D])
# ═══════════════════════════════════════════════════════════════════════════════

def _forecast_nbeats(
    history: pd.Series,
    horizon_days: int,
    external_signals: Optional[Dict[str, pd.Series]] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Forecast usando N-BEATS (Neural Basis Expansion Analysis).
    
    TODO[R&D]: Implementar N-BEATS usando PyTorch ou TensorFlow.
    
    Referências:
        - Paper: "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"
        - GitHub: https://github.com/ElementAI/N-BEATS
    
    Hipótese H5.1: N-BEATS reduz MAPE em ≥15% vs ARIMA para séries sazonais.
    
    Args:
        history: Série histórica
        horizon_days: Horizonte
        external_signals: Sinais externos
    
    Returns:
        (forecast, lower_ci, upper_ci, residuals)
    """
    logger.warning("N-BEATS não implementado, usando ARIMA como fallback")
    return _forecast_arima(history, horizon_days)


# ═══════════════════════════════════════════════════════════════════════════════
# NON-STATIONARY TRANSFORMER (TODO[R&D])
# ═══════════════════════════════════════════════════════════════════════════════

def _forecast_nst(
    history: pd.Series,
    horizon_days: int,
    external_signals: Optional[Dict[str, pd.Series]] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Forecast usando Non-Stationary Transformer (NST).
    
    TODO[R&D]: Implementar NST usando PyTorch.
    
    Referências:
        - Paper: "Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting"
        - GitHub: https://github.com/thuml/Nonstationary_Transformers
    
    Hipótese H5.2: NST supera N-BEATS em séries não-estacionárias.
    
    Args:
        history: Série histórica
        horizon_days: Horizonte
        external_signals: Sinais externos
    
    Returns:
        (forecast, lower_ci, upper_ci, residuals)
    """
    logger.warning("NST não implementado, usando ARIMA como fallback")
    return _forecast_arima(history, horizon_days)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FORECAST FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def forecast_demand(
    sku: str,
    history: pd.Series,
    external_signals: Optional[Dict[str, pd.Series]] = None,
    config: Optional[ForecastConfig] = None,
) -> ForecastResult:
    """
    Forecast de demanda para um SKU.
    
    Função principal para forecasting, suporta múltiplos modelos.
    
    Args:
        sku: SKU a prever
        history: Série histórica (date -> quantity)
        external_signals: Sinais externos (preços, notícias, etc.)
        config: Configuração do forecast
    
    Returns:
        ForecastResult com forecast, intervalos, SNR, métricas
    
    Example:
        >>> history = pd.Series([10, 12, 11, 13, ...], index=pd.date_range('2025-01-01', periods=30))
        >>> result = forecast_demand('SKU-123', history, config=ForecastConfig(model=ForecastModel.ARIMA))
        >>> print(f"Forecast: {result.forecast_series.mean():.1f}, SNR: {result.snr:.2f}")
    """
    config = config or ForecastConfig()
    
    # Selecionar modelo
    if config.model == ForecastModel.ARIMA:
        forecast, lower_ci, upper_ci, residuals = _forecast_arima(
            history,
            config.horizon_days,
            config.confidence_level,
            config.auto_arima,
        )
    elif config.model == ForecastModel.PROPHET:
        forecast, lower_ci, upper_ci, residuals = _forecast_prophet(
            history,
            config.horizon_days,
            config.confidence_level,
            external_signals if config.use_external_signals else None,
        )
    elif config.model == ForecastModel.NBEATS:
        forecast, lower_ci, upper_ci, residuals = _forecast_nbeats(
            history,
            config.horizon_days,
            external_signals if config.use_external_signals else None,
        )
    elif config.model == ForecastModel.NST:
        forecast, lower_ci, upper_ci, residuals = _forecast_nst(
            history,
            config.horizon_days,
            external_signals if config.use_external_signals else None,
        )
    else:
        # Fallback para ARIMA
        forecast, lower_ci, upper_ci, residuals = _forecast_arima(
            history,
            config.horizon_days,
            config.confidence_level,
        )
    
    # Calcular SNR
    snr, snr_class, confidence = compute_snr_forecast(forecast, residuals, history)
    
    # Calcular métricas (se houver dados históricos suficientes)
    metrics = {}
    if len(history) > 10 and residuals is not None and len(residuals) > 0:
        # MAPE (Mean Absolute Percentage Error)
        if history.mean() > 0:
            mape = np.mean(np.abs(residuals / history)) * 100
            metrics["MAPE"] = float(mape)
        
        # RMSE
        rmse = np.sqrt(np.mean(residuals ** 2))
        metrics["RMSE"] = float(rmse)
        
        # MAE
        mae = np.mean(np.abs(residuals))
        metrics["MAE"] = float(mae)
    
    # Sinais externos usados
    external_signals_used = list(external_signals.keys()) if external_signals else []
    
    return ForecastResult(
        sku=sku,
        forecast_series=forecast,
        lower_ci=lower_ci,
        upper_ci=upper_ci,
        model_used=config.model,
        snr=snr,
        snr_class=snr_class,
        confidence_score=confidence,
        metrics=metrics,
        residuals=residuals,
        external_signals_used=external_signals_used,
    )



