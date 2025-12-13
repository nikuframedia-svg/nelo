"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    EXTERNAL SIGNALS (APIs de Preços, Notícias, Macro)
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Este módulo integra sinais externos que podem influenciar a demanda e decisões de inventário:
- Preços de matérias-primas (commodities)
- Notícias de setor (sentiment analysis)
- Indicadores macroeconómicos (inflação, PMI, etc.)

TODO[R&D]: Future enhancements:
    - NLP para análise de sentiment de notícias
    - Integração com APIs reais (Bloomberg, Reuters, etc.)
    - Machine learning para correlacionar sinais com demanda
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class ExternalSignalType(str, Enum):
    """Tipos de sinais externos."""
    COMMODITY_PRICE = "COMMODITY_PRICE"
    NEWS_SENTIMENT = "NEWS_SENTIMENT"
    MACRO_INDICATOR = "MACRO_INDICATOR"
    WEATHER = "WEATHER"  # TODO
    SOCIAL_MEDIA = "SOCIAL_MEDIA"  # TODO


@dataclass
class ExternalSignal:
    """
    Sinal externo processado.
    
    Attributes:
        signal_type: Tipo de sinal
        series_id: Identificador da série
        time_series: Série temporal (date -> value)
        metadata: Metadados adicionais
    """
    signal_type: ExternalSignalType
    series_id: str
    time_series: pd.Series
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExternalSignalConfig:
    """
    Configuração para obtenção de sinais externos.
    
    Attributes:
        enable_mock: Usar dados simulados (para ambiente offline)
        api_timeout: Timeout para chamadas API (segundos)
        cache_duration_hours: Duração do cache (horas)
    """
    enable_mock: bool = True
    api_timeout: float = 5.0
    cache_duration_hours: int = 24


# ═══════════════════════════════════════════════════════════════════════════════
# COMMODITY PRICES
# ═══════════════════════════════════════════════════════════════════════════════

def get_commodity_price(
    series_id: str,
    start_date: datetime,
    end_date: datetime,
    config: Optional[ExternalSignalConfig] = None,
) -> pd.Series:
    """
    Obtém preços de matérias-primas (commodities).
    
    Series IDs comuns:
        - "METAL_COPPER": Cobre
        - "METAL_ALUMINUM": Alumínio
        - "GLASS_RAW": Matéria-prima para vidro
        - "CHEMICAL_PETRO": Produtos petroquímicos
    
    Args:
        series_id: Identificador da série
        start_date: Data inicial
        end_date: Data final
        config: Configuração
    
    Returns:
        Série temporal de preços (date -> price)
    """
    config = config or ExternalSignalConfig()
    
    if config.enable_mock:
        # Simular dados de preços
        dates = pd.date_range(start_date, end_date, freq='D')
        # Simular tendência + ruído
        base_price = 100.0
        trend = np.linspace(0, 10, len(dates))
        noise = np.random.normal(0, 2, len(dates))
        prices = base_price + trend + noise
        
        return pd.Series(prices, index=dates)
    
    # TODO: Integração com API real (ex: Alpha Vantage, Quandl)
    logger.warning(f"API real não implementada para {series_id}, usando mock")
    return get_commodity_price(series_id, start_date, end_date, ExternalSignalConfig(enable_mock=True))


# ═══════════════════════════════════════════════════════════════════════════════
# NEWS HEADLINES
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_news_headlines(
    keywords: List[str],
    days: int = 7,
    config: Optional[ExternalSignalConfig] = None,
) -> List[Dict]:
    """
    Obtém manchetes de notícias relevantes.
    
    Args:
        keywords: Palavras-chave para pesquisa
        days: Número de dias de histórico
        config: Configuração
    
    Returns:
        Lista de notícias com: title, date, url, sentiment_score
    """
    config = config or ExternalSignalConfig()
    
    if config.enable_mock:
        # Simular notícias
        news = []
        for i in range(min(10, days * 2)):
            news.append({
                "title": f"Notícia simulada sobre {keywords[0] if keywords else 'setor'}",
                "date": datetime.now() - timedelta(days=i),
                "url": f"https://example.com/news/{i}",
                "sentiment_score": np.random.uniform(-1, 1),  # -1 (negativo) a +1 (positivo)
            })
        return news
    
    # TODO: Integração com API real (ex: NewsAPI, Google News)
    logger.warning("API real de notícias não implementada, usando mock")
    return fetch_news_headlines(keywords, days, ExternalSignalConfig(enable_mock=True))


def compute_news_sentiment(news_list: List[Dict]) -> float:
    """
    Calcula score de sentiment agregado de notícias.
    
    Args:
        news_list: Lista de notícias
    
    Returns:
        Score de sentiment (-1 a +1)
    """
    if not news_list:
        return 0.0
    
    scores = [n.get("sentiment_score", 0) for n in news_list]
    return float(np.mean(scores))


# ═══════════════════════════════════════════════════════════════════════════════
# MACRO INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def get_macro_indicator(
    indicator_id: str,
    start_date: datetime,
    end_date: datetime,
    config: Optional[ExternalSignalConfig] = None,
) -> pd.Series:
    """
    Obtém indicador macroeconómico.
    
    Indicator IDs:
        - "INFLATION": Taxa de inflação
        - "INTEREST_RATE": Taxa de juro
        - "PMI_MANUFACTURING": PMI Industrial
        - "GDP_GROWTH": Crescimento do PIB
    
    Args:
        indicator_id: ID do indicador
        start_date: Data inicial
        end_date: Data final
        config: Configuração
    
    Returns:
        Série temporal do indicador
    """
    config = config or ExternalSignalConfig()
    
    if config.enable_mock:
        # Simular indicador
        dates = pd.date_range(start_date, end_date, freq='D')
        # Valores constantes com pequeno ruído
        base_value = {
            "INFLATION": 2.5,
            "INTEREST_RATE": 3.0,
            "PMI_MANUFACTURING": 50.0,
            "GDP_GROWTH": 2.0,
        }.get(indicator_id, 0.0)
        
        values = base_value + np.random.normal(0, 0.1, len(dates))
        return pd.Series(values, index=dates)
    
    # TODO: Integração com API real (ex: FRED, World Bank)
    logger.warning(f"API real não implementada para {indicator_id}, usando mock")
    return get_macro_indicator(indicator_id, start_date, end_date, ExternalSignalConfig(enable_mock=True))



