"""
ProdPlan 4.0 - Feature Flags System
====================================

Sistema de flags para seleção de engines e modelos.
Permite alternar entre implementações BASE (estáveis) e ADVANCED (R&D).

Uso:
    from backend.feature_flags import FeatureFlags, ForecastEngine
    
    if FeatureFlags.forecast_engine == ForecastEngine.ADVANCED:
        # Usar N-HiTS/TFT
    else:
        # Usar ETS/ARIMA/XGBoost

Configuração via variáveis de ambiente:
    PRODPLAN_FORECAST_ENGINE=ADVANCED
    PRODPLAN_SCHEDULER_ENGINE=MILP
"""

from __future__ import annotations

import os
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class ForecastEngine(str, Enum):
    """
    Engines de forecasting disponíveis.
    
    BASE: Modelos clássicos estáveis (ETS, ARIMA, XGBoost)
    ADVANCED: Modelos avançados R&D (N-HiTS, TFT, NST) - stubs por agora
    """
    BASE = "base"           # ETS, ARIMA, XGBoost
    ADVANCED = "advanced"   # N-HiTS, TFT, Non-Stationary Transformer (stub)


class RulEngine(str, Enum):
    """
    Engines de RUL (Remaining Useful Life) disponíveis.
    
    BASE: Modelo exponencial/linear com Monte Carlo
    DEEPSURV: Deep Learning survival analysis (stub)
    """
    BASE = "base"           # Exponencial/Linear + Monte Carlo
    DEEPSURV = "deepsurv"   # Deep Survival Analysis (stub)


class DeviationEngine(str, Enum):
    """
    Engines de deteção de desvios.
    
    BASE: Métricas simples (diferença absoluta, percentual)
    POD: Proper Orthogonal Decomposition / PCA (stub)
    """
    BASE = "base"           # Métricas simples
    POD = "pod"             # PCA/POD para redução dimensional (stub)


class SchedulerEngine(str, Enum):
    """
    Engines de scheduling disponíveis.
    
    HEURISTIC: Regras de dispatching (FIFO, SPT, EDD, etc.)
    MILP: Mixed-Integer Linear Programming
    CPSAT: Constraint Programming with SAT
    DRL: Deep Reinforcement Learning (stub)
    """
    HEURISTIC = "heuristic"
    MILP = "milp"
    CPSAT = "cpsat"
    DRL = "drl"             # Stub


class InventoryPolicyEngine(str, Enum):
    """
    Engines de política de inventário.
    
    CLASSIC: ROP/SS clássico (fórmulas analíticas)
    BANDIT: Multi-Armed Bandit para otimização adaptativa (stub)
    """
    CLASSIC = "classic"     # ROP, Safety Stock clássico
    BANDIT = "bandit"       # Multi-Armed Bandit (stub)


class CausalEngine(str, Enum):
    """
    Engines de inferência causal.
    
    OLS: Regressão OLS simples
    DOWHY: DoWhy/EconML para CATE (stub)
    """
    OLS = "ols"             # OLS regression adjustment
    DOWHY = "dowhy"         # DoWhy/EconML (stub)


class XAIEngine(str, Enum):
    """
    Engines de explainability.
    
    BASE: Feature importance simples
    SHAP: SHAP values (stub)
    """
    BASE = "base"
    SHAP = "shap"           # Stub


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE FLAGS CLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureFlagsConfig:
    """
    Configuração de feature flags.
    
    Valores default são os mais conservadores (BASE/CLASSIC/HEURISTIC).
    """
    forecast_engine: ForecastEngine = ForecastEngine.BASE
    rul_engine: RulEngine = RulEngine.BASE
    deviation_engine: DeviationEngine = DeviationEngine.BASE
    scheduler_engine: SchedulerEngine = SchedulerEngine.HEURISTIC
    inventory_policy_engine: InventoryPolicyEngine = InventoryPolicyEngine.CLASSIC
    causal_engine: CausalEngine = CausalEngine.OLS
    xai_engine: XAIEngine = XAIEngine.BASE
    
    # Feature toggles
    enable_digital_twin: bool = True
    enable_zdm_simulation: bool = True
    enable_causal_analysis: bool = True
    enable_rd_experiments: bool = True
    
    # R&D mode (habilita logs detalhados e métricas experimentais)
    rd_mode: bool = False


class FeatureFlags:
    """
    Singleton para gestão de feature flags.
    
    Carrega configuração de variáveis de ambiente ou usa defaults.
    
    Uso:
        # Obter engine atual
        engine = FeatureFlags.get_scheduler_engine()
        
        # Verificar se feature está ativa
        if FeatureFlags.is_enabled("digital_twin"):
            ...
        
        # Obter config completa
        config = FeatureFlags.get_config()
    """
    
    _instance: Optional[FeatureFlagsConfig] = None
    
    @classmethod
    def _load_from_env(cls) -> FeatureFlagsConfig:
        """Carrega configuração de variáveis de ambiente."""
        config = FeatureFlagsConfig()
        
        # Mapear env vars para enums
        env_mapping = {
            "PRODPLAN_FORECAST_ENGINE": ("forecast_engine", ForecastEngine),
            "PRODPLAN_RUL_ENGINE": ("rul_engine", RulEngine),
            "PRODPLAN_DEVIATION_ENGINE": ("deviation_engine", DeviationEngine),
            "PRODPLAN_SCHEDULER_ENGINE": ("scheduler_engine", SchedulerEngine),
            "PRODPLAN_INVENTORY_ENGINE": ("inventory_policy_engine", InventoryPolicyEngine),
            "PRODPLAN_CAUSAL_ENGINE": ("causal_engine", CausalEngine),
            "PRODPLAN_XAI_ENGINE": ("xai_engine", XAIEngine),
        }
        
        for env_var, (attr_name, enum_class) in env_mapping.items():
            value = os.environ.get(env_var)
            if value:
                try:
                    setattr(config, attr_name, enum_class(value.lower()))
                    logger.info(f"Feature flag {attr_name} = {value}")
                except ValueError:
                    logger.warning(f"Invalid value for {env_var}: {value}")
        
        # Boolean flags
        bool_mapping = {
            "PRODPLAN_ENABLE_DIGITAL_TWIN": "enable_digital_twin",
            "PRODPLAN_ENABLE_ZDM": "enable_zdm_simulation",
            "PRODPLAN_ENABLE_CAUSAL": "enable_causal_analysis",
            "PRODPLAN_ENABLE_RD": "enable_rd_experiments",
            "PRODPLAN_RD_MODE": "rd_mode",
        }
        
        for env_var, attr_name in bool_mapping.items():
            value = os.environ.get(env_var)
            if value:
                setattr(config, attr_name, value.lower() in ("true", "1", "yes"))
        
        return config
    
    @classmethod
    def get_config(cls) -> FeatureFlagsConfig:
        """Obtém configuração atual."""
        if cls._instance is None:
            cls._instance = cls._load_from_env()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset para recarregar config."""
        cls._instance = None
    
    @classmethod
    def get_forecast_engine(cls) -> ForecastEngine:
        """Obtém engine de forecasting."""
        return cls.get_config().forecast_engine
    
    @classmethod
    def get_rul_engine(cls) -> RulEngine:
        """Obtém engine de RUL."""
        return cls.get_config().rul_engine
    
    @classmethod
    def get_deviation_engine(cls) -> DeviationEngine:
        """Obtém engine de desvios."""
        return cls.get_config().deviation_engine
    
    @classmethod
    def get_scheduler_engine(cls) -> SchedulerEngine:
        """Obtém engine de scheduling."""
        return cls.get_config().scheduler_engine
    
    @classmethod
    def get_inventory_policy_engine(cls) -> InventoryPolicyEngine:
        """Obtém engine de política de inventário."""
        return cls.get_config().inventory_policy_engine
    
    @classmethod
    def get_causal_engine(cls) -> CausalEngine:
        """Obtém engine causal."""
        return cls.get_config().causal_engine
    
    @classmethod
    def is_enabled(cls, feature: str) -> bool:
        """
        Verifica se feature está ativa.
        
        Args:
            feature: Nome da feature (digital_twin, zdm, causal, rd)
        """
        config = cls.get_config()
        feature_map = {
            "digital_twin": config.enable_digital_twin,
            "zdm": config.enable_zdm_simulation,
            "causal": config.enable_causal_analysis,
            "rd": config.enable_rd_experiments,
        }
        return feature_map.get(feature, False)
    
    @classmethod
    def is_rd_mode(cls) -> bool:
        """Verifica se modo R&D está ativo."""
        return cls.get_config().rd_mode
    
    @classmethod
    def set_engine(cls, engine_type: str, value: str) -> bool:
        """
        Define engine em runtime (para testes/R&D).
        
        Args:
            engine_type: forecast, rul, scheduler, etc.
            value: Nome do engine
        
        Returns:
            True se sucesso
        """
        config = cls.get_config()
        
        engine_map = {
            "forecast": (ForecastEngine, "forecast_engine"),
            "rul": (RulEngine, "rul_engine"),
            "deviation": (DeviationEngine, "deviation_engine"),
            "scheduler": (SchedulerEngine, "scheduler_engine"),
            "inventory": (InventoryPolicyEngine, "inventory_policy_engine"),
            "causal": (CausalEngine, "causal_engine"),
        }
        
        if engine_type not in engine_map:
            logger.warning(f"Unknown engine type: {engine_type}")
            return False
        
        enum_class, attr_name = engine_map[engine_type]
        try:
            setattr(config, attr_name, enum_class(value.lower()))
            logger.info(f"Engine {engine_type} set to {value}")
            return True
        except ValueError:
            logger.warning(f"Invalid value {value} for {engine_type}")
            return False
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Exporta configuração como dict."""
        config = cls.get_config()
        return {
            "engines": {
                "forecast": config.forecast_engine.value,
                "rul": config.rul_engine.value,
                "deviation": config.deviation_engine.value,
                "scheduler": config.scheduler_engine.value,
                "inventory_policy": config.inventory_policy_engine.value,
                "causal": config.causal_engine.value,
                "xai": config.xai_engine.value,
            },
            "features": {
                "digital_twin": config.enable_digital_twin,
                "zdm_simulation": config.enable_zdm_simulation,
                "causal_analysis": config.enable_causal_analysis,
                "rd_experiments": config.enable_rd_experiments,
            },
            "rd_mode": config.rd_mode,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_active_engines() -> Dict[str, str]:
    """Retorna dict com engines ativos."""
    config = FeatureFlags.get_config()
    return {
        "forecast": config.forecast_engine.value,
        "rul": config.rul_engine.value,
        "deviation": config.deviation_engine.value,
        "scheduler": config.scheduler_engine.value,
        "inventory": config.inventory_policy_engine.value,
        "causal": config.causal_engine.value,
    }


def is_advanced_mode(engine_type: str) -> bool:
    """
    Verifica se engine está em modo avançado.
    
    Útil para decidir se usar implementação R&D ou estável.
    """
    config = FeatureFlags.get_config()
    
    advanced_values = {
        "forecast": config.forecast_engine == ForecastEngine.ADVANCED,
        "rul": config.rul_engine == RulEngine.DEEPSURV,
        "deviation": config.deviation_engine == DeviationEngine.POD,
        "scheduler": config.scheduler_engine in (SchedulerEngine.DRL,),
        "inventory": config.inventory_policy_engine == InventoryPolicyEngine.BANDIT,
        "causal": config.causal_engine == CausalEngine.DOWHY,
    }
    
    return advanced_values.get(engine_type, False)



