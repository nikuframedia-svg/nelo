"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    ROP ENGINE (Ponto de Encomenda Dinâmico)
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Este módulo calcula o Reorder Point (ROP) dinâmico e avalia o risco de ruptura.

Mathematical Formulation:
─────────────────────────
    ROP Clássico:
        ROP = μ_d * L + z * σ_d * sqrt(L)
    
    onde:
        μ_d = consumo médio diário (do forecast)
        σ_d = desvio padrão do consumo diário
        L = lead time (dias)
        z = quantil do nível de serviço (ex: 1.96 para 95%)
    
    Safety Stock:
        SS = z * σ_d * sqrt(L)
    
    ROP Dinâmico (ajustado por sazonalidade):
        ROP = μ_d(t) * L + z * σ_d(t) * sqrt(L) + seasonal_adjustment(t)
    
    Risco de Ruptura (30 dias):
        P(stock < 0 em 30 dias) = Monte Carlo simulation
        ou aproximação analítica usando distribuição normal

TODO[R&D]: Future enhancements:
    - Machine learning para ajuste dinâmico de z (service level)
    - Integração com sinais externos (preços, notícias) para ajuste de ROP
    - Multi-echelon ROP (cadeia de fornecimento)
    - Stochastic lead time (L variável)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from smart_inventory.demand_forecasting import ForecastResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ROPConfig:
    """
    Configuração para cálculo de ROP.
    
    Attributes:
        service_level: Nível de serviço desejado (0.95 = 95%)
        lead_time_days: Lead time médio (dias)
        lead_time_std_days: Desvio padrão do lead time (para stochastic)
        use_seasonality: Ajustar ROP por sazonalidade
        use_external_signals: Ajustar ROP com sinais externos
        risk_simulation_samples: Número de amostras para simulação Monte Carlo
    """
    service_level: float = 0.95
    lead_time_days: float = 7.0
    lead_time_std_days: float = 1.0
    use_seasonality: bool = True
    use_external_signals: bool = False
    risk_simulation_samples: int = 10000


@dataclass
class ROPResult:
    """
    Resultado do cálculo de ROP.
    
    Attributes:
        sku: SKU analisado
        rop: Reorder Point (quantidade)
        safety_stock: Safety Stock (quantidade)
        reorder_quantity: Quantidade sugerida para encomenda
        risk_30d: Probabilidade de ruptura nos próximos 30 dias (%)
        coverage_days: Dias de cobertura estimado (stock atual / consumo médio)
        confidence: Confiança do cálculo (baseado no SNR do forecast)
        current_stock: Stock atual (se fornecido)
        days_until_rop: Dias até atingir ROP (se stock atual fornecido)
        explanation: Explicação textual do cálculo
    """
    sku: str
    rop: float
    safety_stock: float
    reorder_quantity: float
    risk_30d: float
    coverage_days: float
    confidence: float
    current_stock: Optional[float] = None
    days_until_rop: Optional[float] = None
    explanation: str = ""
    
    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "sku": self.sku,
            "rop": self.rop,
            "safety_stock": self.safety_stock,
            "reorder_quantity": self.reorder_quantity,
            "risk_30d": self.risk_30d,
            "coverage_days": self.coverage_days,
            "confidence": self.confidence,
            "current_stock": self.current_stock,
            "days_until_rop": self.days_until_rop,
            "explanation": self.explanation,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ROP CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def _get_service_level_z(service_level: float) -> float:
    """
    Obtém z-score para um nível de serviço.
    
    Args:
        service_level: Nível de serviço (0.95 = 95%)
    
    Returns:
        z-score (quantil da distribuição normal)
    """
    from scipy import stats
    try:
        z = stats.norm.ppf(service_level)
        return float(z)
    except ImportError:
        # Fallback: valores comuns
        z_map = {
            0.90: 1.28,
            0.95: 1.96,
            0.99: 2.58,
        }
        return z_map.get(service_level, 1.96)


def compute_dynamic_rop(
    sku: str,
    forecast_result: ForecastResult,
    config: Optional[ROPConfig] = None,
    current_stock: Optional[float] = None,
) -> ROPResult:
    """
    Calcula ROP dinâmico baseado no forecast de demanda.
    
    Args:
        sku: SKU a analisar
        forecast_result: Resultado do forecast de demanda
        config: Configuração do ROP
        current_stock: Stock atual (opcional, para calcular days_until_rop)
    
    Returns:
        ROPResult com ROP, safety stock, risco, etc.
    """
    config = config or ROPConfig()
    
    # Extrair parâmetros do forecast
    forecast_series = forecast_result.forecast_series
    
    # Consumo médio diário (média do forecast)
    mu_d = float(forecast_series.mean())
    
    # Desvio padrão do consumo diário
    # Usar desvio do forecast ou dos resíduos
    if forecast_result.residuals is not None and len(forecast_result.residuals) > 0:
        sigma_d = float(forecast_result.residuals.std())
    else:
        # Fallback: usar variabilidade do forecast
        sigma_d = float(forecast_series.std())
    
    # Se sigma_d for muito pequeno, usar 10% de mu_d como proxy
    if sigma_d < 0.01 * mu_d:
        sigma_d = 0.1 * mu_d
        logger.debug(f"Sigma_d muito pequeno, usando 10% de mu_d como proxy")
    
    # Lead time
    L = config.lead_time_days
    L_std = config.lead_time_std_days
    
    # Z-score para nível de serviço
    z = _get_service_level_z(config.service_level)
    
    # Cálculo clássico de ROP
    # ROP = μ_d * L + z * σ_d * sqrt(L)
    rop_base = mu_d * L + z * sigma_d * np.sqrt(L)
    
    # Safety Stock
    safety_stock = z * sigma_d * np.sqrt(L)
    
    # Ajuste por sazonalidade (se configurado)
    seasonal_adjustment = 0.0
    if config.use_seasonality and len(forecast_series) > 0:
        # Detectar tendência no forecast (últimos 7 dias vs média)
        recent_mean = forecast_series.tail(7).mean() if len(forecast_series) >= 7 else forecast_series.mean()
        if recent_mean > mu_d * 1.1:  # Aumento de 10%+
            seasonal_adjustment = (recent_mean - mu_d) * L * 0.5  # Ajuste conservador
            logger.debug(f"Ajuste sazonal positivo: +{seasonal_adjustment:.1f}")
        elif recent_mean < mu_d * 0.9:  # Redução de 10%+
            seasonal_adjustment = (recent_mean - mu_d) * L * 0.3  # Ajuste menor para redução
            logger.debug(f"Ajuste sazonal negativo: {seasonal_adjustment:.1f}")
    
    # ROP final
    rop = rop_base + seasonal_adjustment
    
    # Reorder Quantity (EOQ simplificado ou baseado em lead time)
    # Por agora, usar 2x o consumo durante lead time
    reorder_quantity = max(rop, mu_d * L * 2)
    
    # Calcular risco de ruptura em 30 dias
    risk_30d = compute_risk_30d(
        forecast_result,
        current_stock if current_stock is not None else rop,
        config,
    )
    
    # Coverage days (dias de cobertura)
    if current_stock is not None:
        coverage_days = current_stock / mu_d if mu_d > 0 else float('inf')
    else:
        coverage_days = rop / mu_d if mu_d > 0 else float('inf')
    
    # Days until ROP
    days_until_rop = None
    if current_stock is not None and mu_d > 0:
        if current_stock > rop:
            days_until_rop = (current_stock - rop) / mu_d
        else:
            days_until_rop = 0.0  # Já abaixo do ROP
    
    # Confidence (baseado no SNR do forecast)
    confidence = forecast_result.confidence_score
    
    # Explicação
    explanation = (
        f"ROP calculado: {rop:.1f} unidades. "
        f"Consumo médio diário: {mu_d:.1f}, Lead time: {L:.1f} dias, "
        f"Nível de serviço: {config.service_level*100:.0f}% (z={z:.2f}). "
        f"Safety stock: {safety_stock:.1f}. "
        f"Risco de ruptura (30d): {risk_30d:.1f}%. "
        f"Confiança do forecast: {forecast_result.snr_class} (SNR={forecast_result.snr:.2f})."
    )
    
    return ROPResult(
        sku=sku,
        rop=rop,
        safety_stock=safety_stock,
        reorder_quantity=reorder_quantity,
        risk_30d=risk_30d,
        coverage_days=coverage_days,
        confidence=confidence,
        current_stock=current_stock,
        days_until_rop=days_until_rop,
        explanation=explanation,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RISK SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_risk_30d(
    forecast_result: ForecastResult,
    current_stock: float,
    config: Optional[ROPConfig] = None,
) -> float:
    """
    Calcula probabilidade de ruptura nos próximos 30 dias usando simulação Monte Carlo.
    
    Método:
        1. Simular consumo diário usando distribuição normal (μ_d, σ_d)
        2. Para cada simulação, calcular stock ao fim de 30 dias
        3. Contar quantas simulações resultam em stock < 0
        4. Risco = (simulações com ruptura) / (total simulações)
    
    Args:
        forecast_result: Resultado do forecast
        current_stock: Stock atual
        config: Configuração (número de amostras)
    
    Returns:
        Probabilidade de ruptura (0-100%)
    """
    config = config or ROPConfig()
    
    forecast_series = forecast_result.forecast_series
    
    # Parâmetros da distribuição
    mu_d = float(forecast_series.mean())
    if forecast_result.residuals is not None and len(forecast_result.residuals) > 0:
        sigma_d = float(forecast_result.residuals.std())
    else:
        sigma_d = float(forecast_series.std())
    
    # Se sigma_d muito pequeno, usar 10% de mu_d
    if sigma_d < 0.01 * mu_d:
        sigma_d = 0.1 * mu_d
    
    if mu_d <= 0:
        # Sem consumo previsto: risco zero
        return 0.0
    
    # Simulação Monte Carlo
    n_samples = config.risk_simulation_samples
    days = 30
    
    # Gerar amostras de consumo diário
    daily_consumption_samples = np.random.normal(
        loc=mu_d,
        scale=sigma_d,
        size=(n_samples, days)
    )
    
    # Garantir valores não-negativos
    daily_consumption_samples = np.maximum(daily_consumption_samples, 0)
    
    # Calcular stock final para cada simulação
    stock_final = current_stock - daily_consumption_samples.sum(axis=1)
    
    # Contar rupturas (stock < 0)
    ruptures = np.sum(stock_final < 0)
    
    # Risco em percentagem
    risk_pct = (ruptures / n_samples) * 100
    
    return float(risk_pct)


def compute_risk_analytical(
    forecast_result: ForecastResult,
    current_stock: float,
    days: int = 30,
) -> float:
    """
    Calcula risco de ruptura usando aproximação analítica (distribuição normal).
    
    Método mais rápido que Monte Carlo, mas assume normalidade.
    
    Args:
        forecast_result: Resultado do forecast
        current_stock: Stock atual
        days: Horizonte de análise (dias)
    
    Returns:
        Probabilidade de ruptura (0-100%)
    """
    forecast_series = forecast_result.forecast_series
    
    mu_d = float(forecast_series.mean())
    if forecast_result.residuals is not None and len(forecast_result.residuals) > 0:
        sigma_d = float(forecast_result.residuals.std())
    else:
        sigma_d = float(forecast_series.std())
    
    if mu_d <= 0:
        return 0.0
    
    # Consumo total esperado em 'days' dias
    mu_total = mu_d * days
    sigma_total = sigma_d * np.sqrt(days)  # Variância aditiva
    
    # Stock final esperado
    stock_final_mean = current_stock - mu_total
    
    # Probabilidade de stock < 0
    if sigma_total > 0:
        from scipy import stats
        try:
            # P(stock_final < 0) = P(N(stock_final_mean, sigma_total) < 0)
            risk = stats.norm.cdf(0, loc=stock_final_mean, scale=sigma_total)
            return float(risk * 100)
        except ImportError:
            # Fallback: aproximação simples
            if stock_final_mean < 0:
                return 100.0
            elif stock_final_mean < sigma_total:
                return 50.0
            else:
                return 0.0
    else:
        # Sem variabilidade: risco binário
        return 100.0 if stock_final_mean < 0 else 0.0



