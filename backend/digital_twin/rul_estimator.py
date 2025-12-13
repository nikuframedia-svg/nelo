"""
════════════════════════════════════════════════════════════════════════════════════════════════════
RUL ESTIMATOR - Remaining Useful Life Estimation with Uncertainty
════════════════════════════════════════════════════════════════════════════════════════════════════

Utiliza Health Indicators (HI) ao longo do tempo para estimar a Vida Útil Remanescente (RUL)
de cada máquina.

Abordagens implementadas:
1. BaseRulEstimator (BASE): Exponential/Linear degradation with Monte Carlo
2. DeepSurvRulEstimator (ADVANCED): Deep Survival Analysis

Feature Flags:
- RulEngine.BASE → BaseRulEstimator
- RulEngine.DEEPSURV → DeepSurvRulEstimator (com fallback)

Output:
- RUL médio (em horas)
- Desvio padrão do RUL (incerteza)
- Intervalo de confiança

TODO[R&D]:
- Deep Bayesian Networks (MC Dropout, HMC/VI) para melhor estimativa de incerteza
- LSTM/Transformer para capturar padrões temporais complexos
- Ensemble methods para robustez
- Conformal prediction para intervalos de confiança garantidos
- Physics-informed neural networks para incorporar conhecimento do domínio
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class HealthStatus(str, Enum):
    """Estado de saúde da máquina."""
    HEALTHY = "HEALTHY"  # HI > 0.7
    DEGRADED = "DEGRADED"  # 0.5 < HI ≤ 0.7
    WARNING = "WARNING"  # 0.3 < HI ≤ 0.5
    CRITICAL = "CRITICAL"  # HI ≤ 0.3


@dataclass
class RULEstimatorConfig:
    """Configuração do estimador de RUL."""
    # Thresholds de HI
    hi_threshold_failure: float = 0.2  # HI abaixo disto = falha
    hi_threshold_critical: float = 0.3
    hi_threshold_warning: float = 0.5
    hi_threshold_degraded: float = 0.7
    
    # Modelo de degradação
    degradation_model: str = "exponential"  # "exponential", "linear", "gp"
    
    # Gaussian Process settings
    gp_length_scale: float = 100.0  # Horas
    gp_noise_variance: float = 0.01
    
    # RUL estimation
    min_history_points: int = 5  # Mínimo de pontos para estimar RUL
    max_rul_hours: float = 2000.0  # RUL máximo (cap)
    confidence_level: float = 0.95  # Para intervalos de confiança
    
    # Monte Carlo settings (para incerteza)
    num_mc_samples: int = 1000


@dataclass
class RULEstimate:
    """Resultado da estimativa de RUL."""
    machine_id: str
    timestamp: datetime
    
    # RUL estimates
    rul_mean_hours: float
    rul_std_hours: float
    rul_lower_hours: float  # Lower bound do IC
    rul_upper_hours: float  # Upper bound do IC
    
    # Current state
    current_hi: float
    health_status: HealthStatus
    
    # Degradation rate
    degradation_rate_per_hour: float  # Taxa de degradação (negativa)
    
    # Confidence
    confidence: float  # Confiança na estimativa (0-1)
    
    # Metadata
    history_points_used: int
    model_used: str
    is_advanced: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário serializável."""
        return {
            "machine_id": self.machine_id,
            "timestamp": self.timestamp.isoformat(),
            "rul_mean_hours": round(self.rul_mean_hours, 1),
            "rul_std_hours": round(self.rul_std_hours, 1),
            "rul_lower_hours": round(self.rul_lower_hours, 1),
            "rul_upper_hours": round(self.rul_upper_hours, 1),
            "current_hi": round(self.current_hi, 4),
            "health_status": self.health_status.value,
            "degradation_rate_per_hour": round(self.degradation_rate_per_hour, 6),
            "confidence": round(self.confidence, 3),
            "history_points_used": self.history_points_used,
            "model_used": self.model_used,
            "is_advanced": self.is_advanced,
        }
    
    @property
    def is_critical(self) -> bool:
        """Verifica se a máquina está em estado crítico."""
        return self.health_status in (HealthStatus.CRITICAL, HealthStatus.WARNING)
    
    @property
    def days_until_failure(self) -> float:
        """RUL em dias."""
        return self.rul_mean_hours / 24.0
    
    def format_with_ci(self) -> str:
        """Formata RUL com intervalo de confiança."""
        return f"{self.rul_mean_hours:.0f}h (95%: {self.rul_lower_hours:.0f}–{self.rul_upper_hours:.0f}h)"


# ═══════════════════════════════════════════════════════════════════════════════
# ABSTRACT BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class RulEstimatorBase(ABC):
    """
    Classe base abstrata para estimadores de RUL.
    
    Todas as implementações devem seguir esta interface.
    """
    
    def __init__(self, config: Optional[RULEstimatorConfig] = None):
        self.config = config or RULEstimatorConfig()
        self._hi_history: Dict[str, List[Tuple[datetime, float]]] = {}
    
    def add_hi_observation(
        self,
        machine_id: str,
        timestamp: datetime,
        hi: float,
    ) -> None:
        """Adiciona observação de HI ao histórico."""
        if machine_id not in self._hi_history:
            self._hi_history[machine_id] = []
        
        self._hi_history[machine_id].append((timestamp, hi))
        
        # Ordenar por timestamp
        self._hi_history[machine_id].sort(key=lambda x: x[0])
    
    def get_hi_history(self, machine_id: str) -> List[Tuple[datetime, float]]:
        """Retorna histórico de HI para uma máquina."""
        return self._hi_history.get(machine_id, [])
    
    @abstractmethod
    def predict_rul(self, machine_id: str, current_state: Optional[Dict] = None) -> Optional[RULEstimate]:
        """
        Estima RUL para uma máquina.
        
        Args:
            machine_id: ID da máquina
            current_state: Estado atual opcional (features adicionais)
        
        Returns:
            RULEstimate ou None se não houver histórico suficiente
        """
        pass
    
    def _get_health_status(self, hi: float) -> HealthStatus:
        """Determina o estado de saúde baseado no HI."""
        if hi > self.config.hi_threshold_degraded:
            return HealthStatus.HEALTHY
        elif hi > self.config.hi_threshold_warning:
            return HealthStatus.DEGRADED
        elif hi > self.config.hi_threshold_critical:
            return HealthStatus.WARNING
        else:
            return HealthStatus.CRITICAL
    
    def _compute_confidence(
        self,
        num_points: int,
        rmse: float,
        current_hi: float,
    ) -> float:
        """Calcula confiança da estimativa."""
        # Mais pontos = mais confiança
        point_factor = min(1.0, num_points / 20)
        
        # Menor erro = mais confiança
        error_factor = max(0.2, 1.0 - rmse * 5)
        
        # HI mais alto = mais tempo para degradar, menos urgente
        hi_factor = 0.5 + 0.5 * current_hi
        
        confidence = point_factor * error_factor * hi_factor
        return np.clip(confidence, 0.1, 0.99)


# ═══════════════════════════════════════════════════════════════════════════════
# BASE RUL ESTIMATOR (Exponential/Linear + Monte Carlo)
# ═══════════════════════════════════════════════════════════════════════════════

class BaseRulEstimator(RulEstimatorBase):
    """
    Estimador de RUL base com modelos de degradação clássicos.
    
    Modelos disponíveis:
    - exponential: HI(t) = HI_0 * exp(-λt) → RUL = ln(HI/HI_fail) / λ
    - linear: HI(t) = HI_0 - λt → RUL = (HI - HI_fail) / λ
    - gp: Gaussian Process Regression com Monte Carlo
    """
    
    def predict_rul(self, machine_id: str, current_state: Optional[Dict] = None) -> Optional[RULEstimate]:
        """Estima RUL usando modelo de degradação."""
        history = self.get_hi_history(machine_id)
        
        if len(history) < self.config.min_history_points:
            logger.warning(
                f"Histórico insuficiente para {machine_id}: "
                f"{len(history)} < {self.config.min_history_points}"
            )
            return None
        
        # Escolher modelo
        if self.config.degradation_model == "exponential":
            return self._estimate_exponential(machine_id, history)
        elif self.config.degradation_model == "linear":
            return self._estimate_linear(machine_id, history)
        elif self.config.degradation_model == "gp":
            return self._estimate_gp(machine_id, history)
        else:
            return self._estimate_exponential(machine_id, history)
    
    def _estimate_exponential(
        self,
        machine_id: str,
        history: List[Tuple[datetime, float]],
    ) -> RULEstimate:
        """
        Modelo de degradação exponencial: HI(t) = HI_0 * exp(-λt)
        
        RUL = ln(HI_current / HI_fail) / λ
        """
        # Converter timestamps para horas desde o início
        t0 = history[0][0]
        times = np.array([(t - t0).total_seconds() / 3600 for t, _ in history])
        his = np.array([hi for _, hi in history])
        
        # Clip HI to avoid log(0)
        his = np.clip(his, 0.01, 1.0)
        
        # Fit exponential: log(HI) = log(HI_0) - λt
        log_his = np.log(his)
        
        # Least squares fit
        n = len(times)
        if n > 1 and np.std(times) > 0:
            slope = (n * np.sum(times * log_his) - np.sum(times) * np.sum(log_his)) / \
                    (n * np.sum(times**2) - np.sum(times)**2)
            intercept = (np.sum(log_his) - slope * np.sum(times)) / n
            
            lambda_rate = -slope
            hi_0 = np.exp(intercept)
        else:
            lambda_rate = 0.001
            hi_0 = his[0]
        
        # Garantir lambda positivo
        lambda_rate = max(1e-6, lambda_rate)
        
        # HI atual
        current_hi = his[-1]
        hi_fail = self.config.hi_threshold_failure
        
        if current_hi > hi_fail and lambda_rate > 0:
            rul_mean = np.log(current_hi / hi_fail) / lambda_rate
        else:
            rul_mean = 0.0
        
        # Cap RUL
        rul_mean = min(rul_mean, self.config.max_rul_hours)
        
        # Incerteza via residuais
        predicted_his = hi_0 * np.exp(-lambda_rate * times)
        residuals = his - predicted_his
        rmse = np.sqrt(np.mean(residuals**2))
        
        # Propagar incerteza para RUL
        if lambda_rate > 0 and current_hi > 0:
            rul_std = rmse / (lambda_rate * current_hi) * rul_mean * 0.3
        else:
            rul_std = rul_mean * 0.3
        
        rul_std = min(rul_std, rul_mean * 0.5)
        
        # Intervalo de confiança
        z = 1.96
        rul_lower = max(0, rul_mean - z * rul_std)
        rul_upper = min(self.config.max_rul_hours, rul_mean + z * rul_std)
        
        status = self._get_health_status(current_hi)
        confidence = self._compute_confidence(len(history), rmse, current_hi)
        
        return RULEstimate(
            machine_id=machine_id,
            timestamp=datetime.now(timezone.utc),
            rul_mean_hours=rul_mean,
            rul_std_hours=rul_std,
            rul_lower_hours=rul_lower,
            rul_upper_hours=rul_upper,
            current_hi=current_hi,
            health_status=status,
            degradation_rate_per_hour=-lambda_rate,
            confidence=confidence,
            history_points_used=len(history),
            model_used="exponential",
            is_advanced=False,
        )
    
    def _estimate_linear(
        self,
        machine_id: str,
        history: List[Tuple[datetime, float]],
    ) -> RULEstimate:
        """Modelo de degradação linear."""
        t0 = history[0][0]
        times = np.array([(t - t0).total_seconds() / 3600 for t, _ in history])
        his = np.array([hi for _, hi in history])
        
        n = len(times)
        if n > 1 and np.std(times) > 0:
            slope = (n * np.sum(times * his) - np.sum(times) * np.sum(his)) / \
                    (n * np.sum(times**2) - np.sum(times)**2)
            intercept = (np.sum(his) - slope * np.sum(times)) / n
            lambda_rate = -slope
        else:
            lambda_rate = 0.0001
            intercept = his[0]
        
        lambda_rate = max(1e-6, lambda_rate)
        
        current_hi = his[-1]
        hi_fail = self.config.hi_threshold_failure
        
        if current_hi > hi_fail and lambda_rate > 0:
            rul_mean = (current_hi - hi_fail) / lambda_rate
        else:
            rul_mean = 0.0
        
        rul_mean = min(rul_mean, self.config.max_rul_hours)
        
        predicted_his = intercept + slope * times
        residuals = his - predicted_his
        rmse = np.sqrt(np.mean(residuals**2))
        
        rul_std = rmse / lambda_rate if lambda_rate > 0 else rul_mean * 0.3
        rul_std = min(rul_std, rul_mean * 0.5)
        
        z = 1.96
        rul_lower = max(0, rul_mean - z * rul_std)
        rul_upper = min(self.config.max_rul_hours, rul_mean + z * rul_std)
        
        status = self._get_health_status(current_hi)
        confidence = self._compute_confidence(len(history), rmse, current_hi)
        
        return RULEstimate(
            machine_id=machine_id,
            timestamp=datetime.now(timezone.utc),
            rul_mean_hours=rul_mean,
            rul_std_hours=rul_std,
            rul_lower_hours=rul_lower,
            rul_upper_hours=rul_upper,
            current_hi=current_hi,
            health_status=status,
            degradation_rate_per_hour=-lambda_rate,
            confidence=confidence,
            history_points_used=len(history),
            model_used="linear",
            is_advanced=False,
        )
    
    def _estimate_gp(
        self,
        machine_id: str,
        history: List[Tuple[datetime, float]],
    ) -> RULEstimate:
        """GP com Monte Carlo para incerteza."""
        base_estimate = self._estimate_exponential(machine_id, history)
        
        rul_samples = []
        current_hi = base_estimate.current_hi
        
        for _ in range(self.config.num_mc_samples):
            rate_perturbation = np.random.normal(1.0, 0.2)
            perturbed_rate = abs(base_estimate.degradation_rate_per_hour * rate_perturbation)
            
            hi_perturbation = np.random.normal(0, 0.05)
            perturbed_hi = np.clip(current_hi + hi_perturbation, 0.01, 1.0)
            
            if perturbed_rate > 1e-6 and perturbed_hi > self.config.hi_threshold_failure:
                rul = np.log(perturbed_hi / self.config.hi_threshold_failure) / perturbed_rate
                rul = min(rul, self.config.max_rul_hours)
                rul_samples.append(rul)
        
        if rul_samples:
            rul_mean = np.mean(rul_samples)
            rul_std = np.std(rul_samples)
            rul_lower = np.percentile(rul_samples, 2.5)
            rul_upper = np.percentile(rul_samples, 97.5)
        else:
            rul_mean = base_estimate.rul_mean_hours
            rul_std = base_estimate.rul_std_hours
            rul_lower = base_estimate.rul_lower_hours
            rul_upper = base_estimate.rul_upper_hours
        
        return RULEstimate(
            machine_id=machine_id,
            timestamp=datetime.now(timezone.utc),
            rul_mean_hours=rul_mean,
            rul_std_hours=rul_std,
            rul_lower_hours=rul_lower,
            rul_upper_hours=rul_upper,
            current_hi=current_hi,
            health_status=base_estimate.health_status,
            degradation_rate_per_hour=base_estimate.degradation_rate_per_hour,
            confidence=base_estimate.confidence * 1.1,
            history_points_used=len(history),
            model_used="gp_monte_carlo",
            is_advanced=False,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DEEP SURVIVAL ESTIMATOR (ADVANCED)
# ═══════════════════════════════════════════════════════════════════════════════

class DeepSurvRulEstimator(RulEstimatorBase):
    """
    Estimador de RUL avançado usando Deep Survival Analysis.
    
    Implementa:
    - DeepSurv (Cox Proportional Hazards com Deep Learning)
    - Pycox/lifelines integration
    
    Se não disponível ou erro: fallback para BaseRulEstimator
    
    Requer:
    - pycox ou lifelines instalado
    - Modelo treinado ou dados suficientes para treinar
    
    TODO[R&D]:
    - Implementar treino completo com pycox
    - Adicionar transfer learning para novos tipos de máquinas
    - Integrar com CVAE para features de HI
    """
    
    # Mínimo de pontos para usar modelo avançado
    MIN_HISTORY_ADVANCED = 50
    
    def __init__(self, config: Optional[RULEstimatorConfig] = None):
        super().__init__(config)
        self._base_estimator = BaseRulEstimator(config)
        self._model = None
        self._is_trained = False
        
        # Verificar disponibilidade de libs
        self._has_pycox = self._check_pycox()
        self._has_lifelines = self._check_lifelines()
        
        if not (self._has_pycox or self._has_lifelines):
            logger.info("DeepSurvRulEstimator: No survival libs available, will use base fallback")
    
    def add_hi_observation(self, machine_id: str, timestamp: datetime, hi: float) -> None:
        """Adiciona observação ao histórico próprio e do base estimator."""
        super().add_hi_observation(machine_id, timestamp, hi)
        # Também adicionar ao base estimator para fallback
        self._base_estimator.add_hi_observation(machine_id, timestamp, hi)
    
    def _check_pycox(self) -> bool:
        """Verifica se pycox está disponível."""
        try:
            import pycox
            import torch
            return True
        except ImportError:
            return False
    
    def _check_lifelines(self) -> bool:
        """Verifica se lifelines está disponível."""
        try:
            import lifelines
            return True
        except ImportError:
            return False
    
    @classmethod
    def load_or_init(cls, model_path: Optional[str] = None) -> "DeepSurvRulEstimator":
        """
        Carrega modelo treinado ou inicializa novo.
        
        Args:
            model_path: Caminho para modelo treinado
        
        Returns:
            DeepSurvRulEstimator
        """
        estimator = cls()
        
        if model_path:
            try:
                estimator._load_model(model_path)
                estimator._is_trained = True
                logger.info(f"Loaded DeepSurv model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {e}")
        
        return estimator
    
    def _load_model(self, path: str) -> None:
        """Carrega modelo do disco."""
        # TODO[R&D]: Implementar carregamento real do modelo
        logger.info(f"Loading model from {path} - STUB")
    
    def fit(self, training_data: Dict[str, Any]) -> None:
        """
        Treina modelo de sobrevivência.
        
        Args:
            training_data: Dict com:
                - features: np.ndarray de features
                - times: np.ndarray de tempos de sobrevivência
                - events: np.ndarray de eventos (1=falha, 0=censurado)
        
        TODO[R&D]: Implementar treino completo
        """
        logger.info("Training DeepSurv model - STUB")
        
        if self._has_lifelines:
            try:
                self._fit_lifelines(training_data)
                self._is_trained = True
            except Exception as e:
                logger.error(f"Lifelines training failed: {e}")
        elif self._has_pycox:
            try:
                self._fit_pycox(training_data)
                self._is_trained = True
            except Exception as e:
                logger.error(f"Pycox training failed: {e}")
    
    def _fit_lifelines(self, training_data: Dict) -> None:
        """Treina usando lifelines CoxPH."""
        from lifelines import CoxPHFitter
        import pandas as pd
        
        # Preparar dados
        df = pd.DataFrame({
            'duration': training_data.get('times', []),
            'event': training_data.get('events', []),
        })
        
        # Adicionar features
        features = training_data.get('features', np.array([]))
        if len(features) > 0:
            for i in range(features.shape[1]):
                df[f'feature_{i}'] = features[:, i]
        
        # Treinar
        self._model = CoxPHFitter()
        self._model.fit(df, duration_col='duration', event_col='event')
        
        logger.info("Lifelines CoxPH model trained")
    
    def _fit_pycox(self, training_data: Dict) -> None:
        """Treina usando pycox DeepSurv."""
        # TODO[R&D]: Implementar treino com pycox
        logger.info("Pycox DeepSurv training - NOT IMPLEMENTED")
    
    def predict_rul(self, machine_id: str, current_state: Optional[Dict] = None) -> Optional[RULEstimate]:
        """
        Estima RUL usando modelo de sobrevivência.
        
        Se modelo não treinado ou histórico insuficiente: fallback para base
        """
        history = self.get_hi_history(machine_id)
        
        # Verificar se temos histórico suficiente
        if len(history) < self.config.min_history_points:
            logger.warning(f"Insufficient history for {machine_id}")
            return None
        
        # Se não temos modelo avançado ou histórico curto, usar base
        if not self._is_trained or len(history) < self.MIN_HISTORY_ADVANCED:
            return self._base_estimator.predict_rul(machine_id, current_state)
        
        # Tentar modelo avançado
        try:
            return self._predict_advanced(machine_id, history, current_state)
        except Exception as e:
            logger.warning(f"Advanced prediction failed: {e}. Falling back to base.")
            return self._base_estimator.predict_rul(machine_id, current_state)
    
    def _predict_advanced(
        self,
        machine_id: str,
        history: List[Tuple[datetime, float]],
        current_state: Optional[Dict],
    ) -> RULEstimate:
        """Predição usando modelo avançado."""
        current_hi = history[-1][1]
        
        if self._has_lifelines and self._model is not None:
            return self._predict_lifelines(machine_id, history, current_state)
        elif self._has_pycox and self._model is not None:
            return self._predict_pycox(machine_id, history, current_state)
        else:
            # Fallback
            return self._base_estimator.predict_rul(machine_id, current_state)
    
    def _predict_lifelines(
        self,
        machine_id: str,
        history: List[Tuple[datetime, float]],
        current_state: Optional[Dict],
    ) -> RULEstimate:
        """Predição usando lifelines."""
        import pandas as pd
        
        current_hi = history[-1][1]
        
        # Preparar features
        features = self._extract_features(history, current_state)
        df = pd.DataFrame([features])
        
        # Obter função de sobrevivência
        survival_function = self._model.predict_survival_function(df)
        
        # Calcular RUL como tempo mediano de sobrevivência
        try:
            median_survival = self._model.predict_median(df).values[0]
            rul_mean = float(median_survival) if not np.isnan(median_survival) else 500.0
        except Exception:
            rul_mean = 500.0
        
        # Intervalo de confiança via bootstrap ou quantis
        rul_std = rul_mean * 0.25  # Estimativa conservadora
        rul_lower = max(0, rul_mean - 1.96 * rul_std)
        rul_upper = min(self.config.max_rul_hours, rul_mean + 1.96 * rul_std)
        
        return RULEstimate(
            machine_id=machine_id,
            timestamp=datetime.now(timezone.utc),
            rul_mean_hours=rul_mean,
            rul_std_hours=rul_std,
            rul_lower_hours=rul_lower,
            rul_upper_hours=rul_upper,
            current_hi=current_hi,
            health_status=self._get_health_status(current_hi),
            degradation_rate_per_hour=0.0,  # Não aplicável para survival
            confidence=0.85,
            history_points_used=len(history),
            model_used="deepsurv_lifelines",
            is_advanced=True,
        )
    
    def _predict_pycox(
        self,
        machine_id: str,
        history: List[Tuple[datetime, float]],
        current_state: Optional[Dict],
    ) -> RULEstimate:
        """Predição usando pycox."""
        # TODO[R&D]: Implementar predição com pycox
        current_hi = history[-1][1]
        
        return RULEstimate(
            machine_id=machine_id,
            timestamp=datetime.now(timezone.utc),
            rul_mean_hours=500.0,  # Placeholder
            rul_std_hours=125.0,
            rul_lower_hours=255.0,
            rul_upper_hours=745.0,
            current_hi=current_hi,
            health_status=self._get_health_status(current_hi),
            degradation_rate_per_hour=0.0,
            confidence=0.75,
            history_points_used=len(history),
            model_used="deepsurv_pycox",
            is_advanced=True,
        )
    
    def _extract_features(
        self,
        history: List[Tuple[datetime, float]],
        current_state: Optional[Dict],
    ) -> Dict[str, float]:
        """Extrai features para modelo de sobrevivência."""
        his = [h for _, h in history]
        
        features = {
            "current_hi": his[-1],
            "mean_hi": np.mean(his),
            "std_hi": np.std(his) if len(his) > 1 else 0,
            "min_hi": np.min(his),
            "trend": (his[-1] - his[0]) / len(his) if len(his) > 1 else 0,
            "history_length": len(his),
        }
        
        # Adicionar features do current_state se disponível
        if current_state:
            for k, v in current_state.items():
                if isinstance(v, (int, float)):
                    features[f"state_{k}"] = v
        
        return features


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION (with FeatureFlags integration)
# ═══════════════════════════════════════════════════════════════════════════════

def get_rul_estimator(
    config: Optional[RULEstimatorConfig] = None,
    use_advanced: Optional[bool] = None,
) -> RulEstimatorBase:
    """
    Factory function para obter estimador de RUL baseado em FeatureFlags.
    
    Args:
        config: Configuração do estimador
        use_advanced: Forçar modo avançado (se None, usa FeatureFlags)
    
    Returns:
        RulEstimatorBase (BaseRulEstimator ou DeepSurvRulEstimator)
    """
    # Importar FeatureFlags
    try:
        from ..feature_flags import FeatureFlags, RulEngine as RE
        
        if use_advanced is None:
            use_advanced = FeatureFlags.get_rul_engine() == RE.DEEPSURV
    except ImportError:
        if use_advanced is None:
            use_advanced = False
    
    if use_advanced:
        try:
            return DeepSurvRulEstimator.load_or_init()
        except Exception as e:
            logger.warning(f"Failed to create DeepSurvRulEstimator: {e}. Using base.")
            return BaseRulEstimator(config)
    else:
        return BaseRulEstimator(config)


# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY (RULEstimator alias)
# ═══════════════════════════════════════════════════════════════════════════════

class RULEstimator(BaseRulEstimator):
    """Alias para compatibilidade com código existente."""
    
    def estimate(self, machine_id: str) -> Optional[RULEstimate]:
        """Alias para predict_rul."""
        return self.predict_rul(machine_id)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_default_estimator: Optional[RulEstimatorBase] = None


def get_default_estimator() -> RulEstimatorBase:
    """Retorna o estimador global."""
    global _default_estimator
    if _default_estimator is None:
        _default_estimator = get_rul_estimator()
    return _default_estimator


def estimate_rul(
    machine_id: str,
    hi_history: List[Tuple[datetime, float]],
    config: Optional[RULEstimatorConfig] = None,
) -> Optional[RULEstimate]:
    """
    Estimar RUL para uma máquina dado o histórico de HI.
    
    Args:
        machine_id: ID da máquina
        hi_history: Lista de (timestamp, HI) ordenada cronologicamente
        config: Configuração do estimador
    
    Returns:
        RULEstimate ou None se histórico insuficiente
    """
    estimator = get_rul_estimator(config)
    
    # Adicionar histórico
    for timestamp, hi in hi_history:
        estimator.add_hi_observation(machine_id, timestamp, hi)
    
    return estimator.predict_rul(machine_id)


def get_machine_health_status(hi: float) -> HealthStatus:
    """Determina o estado de saúde baseado no HI."""
    config = RULEstimatorConfig()
    if hi > config.hi_threshold_degraded:
        return HealthStatus.HEALTHY
    elif hi > config.hi_threshold_warning:
        return HealthStatus.DEGRADED
    elif hi > config.hi_threshold_critical:
        return HealthStatus.WARNING
    else:
        return HealthStatus.CRITICAL


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO / TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def create_demo_hi_history(
    machine_id: str,
    num_points: int = 50,
    degradation_type: str = "exponential",
    initial_hi: float = 0.95,
    final_hi: float = 0.4,
) -> List[Tuple[datetime, float]]:
    """
    Gerar histórico de HI de demonstração para uma máquina.
    """
    now = datetime.now(timezone.utc)
    history = []
    
    for i in range(num_points):
        t = now - timedelta(hours=num_points - i - 1)
        progress = i / (num_points - 1) if num_points > 1 else 1.0
        
        if degradation_type == "exponential":
            lambda_rate = -np.log(final_hi / initial_hi)
            hi = initial_hi * np.exp(-lambda_rate * progress)
        elif degradation_type == "linear":
            hi = initial_hi - (initial_hi - final_hi) * progress
        elif degradation_type == "sudden":
            if progress < 0.8:
                hi = initial_hi - (initial_hi - 0.7) * (progress / 0.8)
            else:
                hi = 0.7 - (0.7 - final_hi) * ((progress - 0.8) / 0.2)
        else:
            hi = initial_hi - (initial_hi - final_hi) * progress
        
        noise = np.random.normal(0, 0.02)
        hi = np.clip(hi + noise, 0.01, 1.0)
        
        history.append((t, hi))
    
    return history


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH EXECUTION LOGS (CONTRACT 9)
# ═══════════════════════════════════════════════════════════════════════════════

def get_machine_operational_features(
    machine_id: str,
    days_lookback: int = 30,
) -> Dict[str, Any]:
    """
    Collect operational features from execution logs for a machine.
    
    These features can be used to improve RUL estimation by considering:
    - Load intensity (average cycle times, utilization)
    - Stress indicators (scrap rates, downtime frequency)
    - Operating conditions
    
    Args:
        machine_id: Machine ID
        days_lookback: Number of days to look back
    
    Returns:
        Dict with operational features
    """
    try:
        from prodplan.execution_log_models import (
            query_execution_logs,
            ExecutionLogQuery,
            ExecutionLogStatus,
            get_execution_stats,
        )
        from datetime import timedelta
        
        # Get aggregated stats
        stats = get_execution_stats(
            operation_id="*",  # All operations
            machine_id=machine_id,
            days=days_lookback,
        )
        
        # Get recent logs for more detailed analysis
        query = ExecutionLogQuery(
            machine_id=machine_id,
            status=ExecutionLogStatus.COMPLETED,
            from_date=datetime.now(timezone.utc) - timedelta(days=days_lookback),
            limit=500,
        )
        logs = query_execution_logs(query)
        
        # Calculate features
        total_run_time_hours = sum(
            log.cycle_time_s * log.qty_good / 3600 
            for log in logs 
            if log.cycle_time_s and log.qty_good
        )
        
        total_downtime_hours = sum(
            log.downtime_minutes / 60 
            for log in logs 
            if log.downtime_minutes
        )
        
        avg_energy_per_hour = np.mean([
            log.energy_kwh / (log.cycle_time_s * log.qty_good / 3600)
            for log in logs
            if log.energy_kwh and log.cycle_time_s and log.qty_good > 0
        ]) if logs else 0.0
        
        downtime_events = sum(1 for log in logs if log.downtime_minutes and log.downtime_minutes > 0)
        
        # Calculate utilization
        available_hours = days_lookback * 16  # 16 hours/day assumed
        utilization = total_run_time_hours / available_hours if available_hours > 0 else 0.0
        
        return {
            "machine_id": machine_id,
            "n_executions": stats.n_executions,
            "total_run_time_hours": round(total_run_time_hours, 1),
            "total_downtime_hours": round(total_downtime_hours, 1),
            "downtime_events": downtime_events,
            "avg_scrap_rate": round(stats.avg_scrap_rate, 2),
            "avg_cycle_time_s": round(stats.avg_cycle_time_s, 1),
            "utilization": round(utilization, 3),
            "avg_energy_per_hour": round(avg_energy_per_hour, 2) if not np.isnan(avg_energy_per_hour) else 0.0,
            "load_intensity": "high" if utilization > 0.7 else "medium" if utilization > 0.4 else "low",
            "stress_indicator": "elevated" if stats.avg_scrap_rate > 3.0 or downtime_events > 10 else "normal",
        }
    except ImportError:
        logger.warning("Execution log models not available")
        return {"machine_id": machine_id, "error": "Execution log models not available"}
    except Exception as e:
        logger.error(f"Error getting machine operational features: {e}")
        return {"machine_id": machine_id, "error": str(e)}


def adjust_rul_for_load(
    base_rul_hours: float,
    operational_features: Dict[str, Any],
) -> float:
    """
    Adjust RUL estimate based on operational load.
    
    Higher utilization and stress reduce RUL.
    
    Args:
        base_rul_hours: Base RUL estimate
        operational_features: Features from get_machine_operational_features
    
    Returns:
        Adjusted RUL in hours
    """
    if operational_features.get("error"):
        return base_rul_hours
    
    # Load factor: high utilization reduces RUL
    utilization = operational_features.get("utilization", 0.5)
    load_factor = 1.0 - (utilization - 0.5) * 0.3  # 50% util = 1.0, 100% util = 0.85
    load_factor = max(0.7, min(1.1, load_factor))  # Clamp
    
    # Stress factor: high scrap/downtime reduces RUL
    stress = operational_features.get("stress_indicator", "normal")
    stress_factor = 0.85 if stress == "elevated" else 1.0
    
    # Combine factors
    adjusted_rul = base_rul_hours * load_factor * stress_factor
    
    return max(10.0, adjusted_rul)  # Minimum 10 hours


def estimate_rul_with_context(
    machine_id: str,
    hi_history: Optional[List[Tuple[datetime, float]]] = None,
    include_operational_features: bool = True,
    estimator: Optional[BaseRulEstimator] = None,
) -> RULEstimate:
    """
    Estimate RUL with operational context from execution logs (Contract 9).
    
    This is the recommended function to use for RUL estimation as it
    incorporates both HI history and operational load data.
    
    Args:
        machine_id: Machine ID
        hi_history: Optional HI history; if None, demo data is generated
        include_operational_features: Whether to include execution log features
        estimator: Optional estimator to use; if None, uses get_rul_estimator()
    
    Returns:
        RULEstimate with context-adjusted values
    """
    if estimator is None:
        estimator = get_rul_estimator()
    
    # Generate demo history if not provided
    if hi_history is None:
        hi_history = create_demo_hi_history(
            machine_id=machine_id,
            num_points=50,
            degradation_type="exponential",
            initial_hi=0.9,
            final_hi=0.6,
        )
    
    # Get base RUL estimate
    # Build current_state from latest HI in history
    if hi_history:
        latest_time, latest_hi = hi_history[-1]
        current_state = {
            "hi": latest_hi,
            "hi_history": hi_history,
        }
    else:
        current_state = None
    
    base_estimate = estimator.predict_rul(machine_id, current_state)
    
    if base_estimate is None:
        # Create a default estimate if prediction failed
        logger.warning(f"RUL prediction failed for {machine_id}, using defaults")
        base_estimate = RULEstimate(
            machine_id=machine_id,
            timestamp=datetime.now(timezone.utc),
            rul_mean_hours=500.0,  # Default
            rul_std_hours=100.0,
            rul_lower_hours=300.0,
            rul_upper_hours=700.0,
            current_hi=hi_history[-1][1] if hi_history else 0.7,
            health_status=HealthStatus.DEGRADED,
            degradation_rate_per_hour=-0.001,
            confidence=0.3,
            history_points_used=len(hi_history) if hi_history else 0,
            model_used="default",
            is_advanced=False,
        )
    
    if not include_operational_features:
        return base_estimate
    
    # Get operational features and adjust
    features = get_machine_operational_features(machine_id)
    
    if features.get("error"):
        # No operational data, return base estimate
        base_estimate.confidence *= 0.9  # Slightly reduce confidence
        return base_estimate
    
    # Adjust RUL based on operational load
    adjusted_rul_mean = adjust_rul_for_load(base_estimate.rul_mean_hours, features)
    
    # Calculate adjustment ratio
    ratio = adjusted_rul_mean / base_estimate.rul_mean_hours if base_estimate.rul_mean_hours > 0 else 1.0
    
    # Apply ratio to all RUL values
    adjusted_estimate = RULEstimate(
        machine_id=base_estimate.machine_id,
        timestamp=base_estimate.timestamp,
        rul_mean_hours=adjusted_rul_mean,
        rul_std_hours=base_estimate.rul_std_hours * ratio,
        rul_lower_hours=base_estimate.rul_lower_hours * ratio,
        rul_upper_hours=base_estimate.rul_upper_hours * ratio,
        current_hi=base_estimate.current_hi,
        health_status=base_estimate.health_status,
        degradation_rate_per_hour=base_estimate.degradation_rate_per_hour / ratio if ratio > 0 else base_estimate.degradation_rate_per_hour,
        confidence=base_estimate.confidence,
        history_points_used=base_estimate.history_points_used,
        model_used=f"{base_estimate.model_used}+context",
        is_advanced=base_estimate.is_advanced,
    )
    
    # Store additional metadata in a way that doesn't break the dataclass
    adjusted_estimate._operational_features = features
    adjusted_estimate._load_adjustment_ratio = ratio
    
    return adjusted_estimate
