"""
════════════════════════════════════════════════════════════════════════════════════════════════════
SHI-DT (Smart Health Index Digital Twin)
════════════════════════════════════════════════════════════════════════════════════════════════════

Gêmeo Digital de Saúde de Máquina para monitorização de condição e previsão de falhas.

Features:
- Índice de Saúde (HI) em tempo real: 0-100% por máquina
- Adaptação a perfis operacionais (regimes diferentes têm baselines diferentes)
- Estimativa de RUL (Remaining Useful Life) com modelos de degradação
- Alertas explicáveis com contribuição de cada sensor
- Aprendizado contínuo (online learning)

Modelo Matemático:
- CVAE (Conditional Variational Autoencoder) para modelar estado saudável
  - Loss: L = E_{q_φ(z|x)}[-log p_θ(x|z)] + β * KL(q_φ(z|x) || p(z))
  - Onde x são sequências de dados de sensor e z é a representação latente
- Health Index: H(t) = 100 * exp(-α * E_rec(t))
  - Onde E_rec(t) é o erro de reconstrução do CVAE no tempo t
  - α (alpha) é um fator de escala calibrado empiricamente
  - Ajustado por perfil operacional: α_adjusted = α * threshold_factor(profile)
- Degradação de parâmetros: P(t) = P(0) - Δ_d * f(uso_acumulado, regime)
- RUL: Estimado como tempo τ tal que H(τ) < threshold_critical (ex: 20)
  - Ou através de modelo auxiliar que prediz RUL a partir da trajetória de H(t)

Referências:
- Li et al., "Remaining Useful Life Prediction Using CVAE", IEEE Trans. Industrial Informatics
- Malhotra et al., "Multi-Sensor Prognostics using VAE", ICLR Workshop
- Paris' Law para modelagem de fadiga mecânica

R&D / SIFIDE: WP1 - Digital Twin para manutenção preditiva
"""

from __future__ import annotations

import logging
import math
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from collections import deque

import numpy as np

from .health_indicator_cvae import (
    CVAEConfig,
    CVAE,
    SensorSnapshot,
    OperationContext,
    HealthIndicatorResult,
    infer_hi,
    train_cvae,
    create_demo_dataset,
    TORCH_AVAILABLE,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SHIDTConfig:
    """Configuração do Smart Health Index Digital Twin."""
    
    # CVAE Configuration
    cvae_config: CVAEConfig = field(default_factory=CVAEConfig)
    
    # Health Index Settings
    reconstruction_threshold: float = 0.5  # T para normalização do HI (legacy)
    hi_alpha: float = 0.1  # α (alpha) para fórmula H(t) = 100 * exp(-α * E_rec(t))
    hi_smoothing_window: int = 10  # Janela para suavização exponencial
    hi_ema_alpha: float = 0.3  # Alpha para EMA do HI (após cálculo exponencial)
    
    # Alert Thresholds (HI percentages)
    threshold_healthy: float = 80.0  # > 80% = saudável
    threshold_warning: float = 50.0  # 50-80% = aviso
    threshold_critical: float = 30.0  # < 30% = crítico
    
    # RUL Settings
    rul_failure_threshold: float = 20.0  # HI abaixo do qual considera-se falha
    rul_min_history_points: int = 5  # Mínimo de pontos para estimar RUL
    rul_extrapolation_method: str = "exponential"  # "linear" ou "exponential"
    rul_confidence_level: float = 0.95  # Para intervalos de confiança
    
    # Operational Profiles
    profile_detection_enabled: bool = True
    profile_window_seconds: int = 300  # 5 minutos para detectar regime
    
    # Online Learning / Periodic Re-training
    online_learning_enabled: bool = True  # Habilitado por padrão
    online_learning_buffer_size: int = 1000
    online_learning_update_interval: int = 100  # Re-treinar após N amostras
    periodic_retrain_interval_hours: float = 168.0  # Re-treinar semanalmente (7 dias)
    periodic_retrain_min_samples: int = 500  # Mínimo de amostras para re-treino
    
    # Explainability
    top_k_contributors: int = 5  # Top K sensores que contribuem para degradação
    
    # Model Paths
    model_dir: Path = field(default_factory=lambda: Path("models/shi_dt"))


class OperationalProfile(str, Enum):
    """Perfis operacionais da máquina."""
    IDLE = "idle"  # Parada/standby
    LOW_LOAD = "low_load"  # Carga baixa (<30%)
    NORMAL = "normal"  # Operação normal (30-70%)
    HIGH_LOAD = "high_load"  # Carga alta (70-90%)
    PEAK = "peak"  # Carga máxima (>90%)
    STARTUP = "startup"  # Arranque
    SHUTDOWN = "shutdown"  # Paragem


class AlertSeverity(str, Enum):
    """Severidade de alertas."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HealthIndexReading:
    """Leitura de Health Index num instante."""
    timestamp: datetime
    hi_raw: float  # HI direto do modelo
    hi_smoothed: float  # HI suavizado (EMA)
    reconstruction_error: float
    profile: OperationalProfile
    latent_vector: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "hi_raw": round(self.hi_raw, 2),
            "hi_smoothed": round(self.hi_smoothed, 2),
            "reconstruction_error": round(self.reconstruction_error, 4),
            "profile": self.profile.value,
        }


@dataclass
class RULEstimate:
    """Estimativa de Remaining Useful Life."""
    timestamp: datetime
    rul_hours: float  # RUL médio estimado
    rul_lower: float  # Limite inferior (CI)
    rul_upper: float  # Limite superior (CI)
    confidence: float  # Nível de confiança da estimativa
    degradation_rate: float  # Taxa de degradação (HI/hora)
    extrapolation_method: str
    data_points_used: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "rul_hours": float(round(self.rul_hours, 1)),
            "rul_lower": float(round(self.rul_lower, 1)),
            "rul_upper": float(round(self.rul_upper, 1)),
            "confidence": float(round(self.confidence, 3)),
            "degradation_rate_per_hour": float(round(self.degradation_rate, 4)),
            "method": self.extrapolation_method,
            "data_points": int(self.data_points_used),
        }


@dataclass
class SensorContribution:
    """Contribuição de um sensor para a degradação."""
    sensor_name: str
    contribution_pct: float  # Percentagem de contribuição
    current_value: float
    baseline_value: float  # Valor esperado em estado saudável
    deviation: float  # Desvio do baseline
    trend: str  # "increasing", "stable", "decreasing"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor": str(self.sensor_name),
            "contribution_pct": float(round(self.contribution_pct, 1)),
            "current": float(round(self.current_value, 3)),
            "baseline": float(round(self.baseline_value, 3)),
            "deviation": float(round(self.deviation, 3)),
            "trend": str(self.trend),
        }


@dataclass
class HealthAlert:
    """Alerta de saúde da máquina."""
    alert_id: str
    machine_id: str
    timestamp: datetime
    severity: AlertSeverity
    title: str
    message: str
    hi_current: float
    rul_estimate: Optional[RULEstimate] = None
    contributing_sensors: List[SensorContribution] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "machine_id": self.machine_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "hi_current": round(self.hi_current, 1),
            "rul": self.rul_estimate.to_dict() if self.rul_estimate else None,
            "contributing_sensors": [s.to_dict() for s in self.contributing_sensors],
            "recommended_actions": self.recommended_actions,
            "acknowledged": self.acknowledged,
        }


@dataclass
class MachineHealthStatus:
    """Estado de saúde completo de uma máquina."""
    machine_id: str
    timestamp: datetime
    
    # Health Index
    health_index: float  # 0-100%
    health_index_std: float  # Incerteza
    status: str  # "HEALTHY", "WARNING", "CRITICAL"
    
    # RUL
    rul_estimate: Optional[RULEstimate]
    
    # Operational Profile
    current_profile: OperationalProfile
    
    # Degradation
    degradation_trend: str  # "stable", "degrading", "improving"
    degradation_rate: float  # HI/hora
    
    # Contributing Factors
    top_contributors: List[SensorContribution]
    
    # Active Alerts
    active_alerts: List[HealthAlert]
    
    # Last Update
    last_sensor_update: datetime
    model_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine_id": str(self.machine_id),
            "timestamp": self.timestamp.isoformat(),
            "health_index": float(round(self.health_index, 1)),
            "health_index_std": float(round(self.health_index_std, 2)),
            "status": str(self.status),
            "rul": self.rul_estimate.to_dict() if self.rul_estimate else None,
            "profile": str(self.current_profile.value),
            "degradation": {
                "trend": str(self.degradation_trend),
                "rate_per_hour": float(round(self.degradation_rate, 4)),
            },
            "top_contributors": [c.to_dict() for c in self.top_contributors],
            "active_alerts": [a.to_dict() for a in self.active_alerts],
            "last_sensor_update": self.last_sensor_update.isoformat(),
            "model_version": str(self.model_version),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# OPERATIONAL PROFILE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class ProfileDetector:
    """Detecta o perfil operacional atual da máquina."""
    
    def __init__(self, config: SHIDTConfig):
        self.config = config
        self._load_history: Dict[str, deque] = {}  # machine_id -> deque de loads
    
    def detect_profile(
        self,
        machine_id: str,
        snapshot: SensorSnapshot,
    ) -> OperationalProfile:
        """
        Detecta o perfil operacional com base nos sensores.
        
        Usa load_percent e speed_rpm como indicadores principais.
        """
        if not self.config.profile_detection_enabled:
            return OperationalProfile.NORMAL
        
        load = snapshot.load_percent
        speed = snapshot.speed_rpm
        
        # Manter histórico para detectar transições
        if machine_id not in self._load_history:
            self._load_history[machine_id] = deque(maxlen=10)
        self._load_history[machine_id].append(load)
        
        # Detectar startup/shutdown
        if len(self._load_history[machine_id]) >= 3:
            recent_loads = list(self._load_history[machine_id])[-3:]
            load_change = recent_loads[-1] - recent_loads[0]
            
            if load_change > 0.3 and recent_loads[0] < 0.1:
                return OperationalProfile.STARTUP
            if load_change < -0.3 and recent_loads[-1] < 0.1:
                return OperationalProfile.SHUTDOWN
        
        # Classificação por load
        if load < 0.05 and speed < 0.1:
            return OperationalProfile.IDLE
        elif load < 0.30:
            return OperationalProfile.LOW_LOAD
        elif load < 0.70:
            return OperationalProfile.NORMAL
        elif load < 0.90:
            return OperationalProfile.HIGH_LOAD
        else:
            return OperationalProfile.PEAK
    
    def get_profile_baseline(
        self,
        profile: OperationalProfile,
    ) -> Dict[str, float]:
        """
        Retorna valores baseline esperados para cada perfil.
        
        Usado para ajustar thresholds do HI por regime.
        """
        baselines = {
            OperationalProfile.IDLE: {
                "vibration_rms": 0.02,
                "temperature_motor": 0.15,
                "current_rms": 0.05,
                "reconstruction_threshold_factor": 0.5,
            },
            OperationalProfile.LOW_LOAD: {
                "vibration_rms": 0.10,
                "temperature_motor": 0.25,
                "current_rms": 0.25,
                "reconstruction_threshold_factor": 0.8,
            },
            OperationalProfile.NORMAL: {
                "vibration_rms": 0.20,
                "temperature_motor": 0.35,
                "current_rms": 0.45,
                "reconstruction_threshold_factor": 1.0,
            },
            OperationalProfile.HIGH_LOAD: {
                "vibration_rms": 0.35,
                "temperature_motor": 0.50,
                "current_rms": 0.65,
                "reconstruction_threshold_factor": 1.2,
            },
            OperationalProfile.PEAK: {
                "vibration_rms": 0.50,
                "temperature_motor": 0.65,
                "current_rms": 0.80,
                "reconstruction_threshold_factor": 1.5,
            },
            OperationalProfile.STARTUP: {
                "vibration_rms": 0.40,
                "temperature_motor": 0.30,
                "current_rms": 0.70,
                "reconstruction_threshold_factor": 2.0,
            },
            OperationalProfile.SHUTDOWN: {
                "vibration_rms": 0.15,
                "temperature_motor": 0.40,
                "current_rms": 0.30,
                "reconstruction_threshold_factor": 1.5,
            },
        }
        return baselines.get(profile, baselines[OperationalProfile.NORMAL])


# ═══════════════════════════════════════════════════════════════════════════════
# RUL ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════════

class RULCalculator:
    """
    Calcula Remaining Useful Life baseado na tendência de degradação.
    
    Métodos:
    - Linear: RUL = (HI_current - HI_failure) / degradation_rate
    - Exponential: Ajusta curva exponencial para modelar aceleração
    - Paris' Law: Para fadiga mecânica (crack propagation)
    - Kalman Filter: Para tracking adaptativo
    """
    
    def __init__(self, config: SHIDTConfig):
        self.config = config
        self._history: Dict[str, deque] = {}  # machine_id -> deque de (timestamp, hi)
        self._kalman_state: Dict[str, Dict] = {}  # Estado do filtro de Kalman
    
    def add_observation(
        self,
        machine_id: str,
        timestamp: datetime,
        hi: float,
    ) -> None:
        """Adiciona observação ao histórico."""
        if machine_id not in self._history:
            self._history[machine_id] = deque(maxlen=1000)
        self._history[machine_id].append((timestamp, hi))
    
    def estimate_rul(
        self,
        machine_id: str,
        current_hi: float,
    ) -> Optional[RULEstimate]:
        """
        Estima RUL baseado no histórico de HI.
        
        Retorna None se não houver dados suficientes.
        """
        if machine_id not in self._history:
            return None
        
        history = list(self._history[machine_id])
        
        if len(history) < self.config.rul_min_history_points:
            return self._fallback_rul(machine_id, current_hi, len(history))
        
        # Extrair séries temporais
        timestamps = [h[0] for h in history]
        his = [h[1] for h in history]
        
        # Converter timestamps para horas desde o primeiro ponto
        t0 = timestamps[0]
        hours = [(t - t0).total_seconds() / 3600 for t in timestamps]
        
        method = self.config.rul_extrapolation_method
        
        try:
            if method == "linear":
                return self._estimate_linear(machine_id, hours, his, current_hi)
            elif method == "exponential":
                return self._estimate_exponential(machine_id, hours, his, current_hi)
            else:
                return self._estimate_linear(machine_id, hours, his, current_hi)
        except Exception as e:
            logger.warning(f"RUL estimation failed for {machine_id}: {e}")
            return self._fallback_rul(machine_id, current_hi, len(history))
    
    def _estimate_linear(
        self,
        machine_id: str,
        hours: List[float],
        his: List[float],
        current_hi: float,
    ) -> RULEstimate:
        """
        Estimativa linear: HI(t) = HI_0 - rate * t
        
        RUL = (HI_current - HI_failure) / rate
        """
        hours_arr = np.array(hours)
        his_arr = np.array(his)
        
        # Regressão linear
        n = len(hours)
        slope, intercept = np.polyfit(hours_arr, his_arr, 1)
        
        # Rate de degradação (negativo = degradando)
        degradation_rate = -slope  # HI/hora
        
        # Calcular RUL
        hi_failure = self.config.rul_failure_threshold
        
        if degradation_rate <= 0:
            # Não está degradando, RUL infinito
            rul_hours = 10000.0  # Cap a 10000 horas
        else:
            rul_hours = max(0, (current_hi - hi_failure) / degradation_rate)
        
        # Intervalo de confiança
        # Usar erro padrão da regressão
        residuals = his_arr - (intercept + slope * hours_arr)
        se = np.sqrt(np.sum(residuals**2) / (n - 2)) if n > 2 else 0.1
        
        # CI para RUL (propagação de incerteza simplificada)
        t_value = 1.96  # ~95% CI
        rul_std = rul_hours * (se / abs(current_hi - hi_failure + 0.01))
        rul_lower = max(0, rul_hours - t_value * rul_std)
        rul_upper = rul_hours + t_value * rul_std
        
        # Confidence baseada no R²
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((his_arr - np.mean(his_arr))**2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        confidence = max(0.5, min(0.99, r2))
        
        return RULEstimate(
            timestamp=datetime.now(timezone.utc),
            rul_hours=min(rul_hours, 10000),
            rul_lower=min(rul_lower, 10000),
            rul_upper=min(rul_upper, 15000),
            confidence=confidence,
            degradation_rate=degradation_rate,
            extrapolation_method="linear",
            data_points_used=n,
        )
    
    def _estimate_exponential(
        self,
        machine_id: str,
        hours: List[float],
        his: List[float],
        current_hi: float,
    ) -> RULEstimate:
        """
        Estimativa exponencial: HI(t) = HI_0 * exp(-λ * t)
        
        Modela degradação acelerada típica de fadiga.
        """
        hours_arr = np.array(hours)
        his_arr = np.array(his)
        
        # Transformar para log para regressão linear
        his_positive = np.maximum(his_arr, 0.01)  # Evitar log(0)
        log_his = np.log(his_positive)
        
        # Regressão no domínio log
        slope, intercept = np.polyfit(hours_arr, log_his, 1)
        
        # Taxa de degradação exponencial
        lambda_rate = -slope
        hi_0 = np.exp(intercept)
        
        # RUL: resolver HI_failure = HI_0 * exp(-λ * t_rul)
        hi_failure = self.config.rul_failure_threshold / 100  # Normalizado
        
        if lambda_rate <= 0 or current_hi <= hi_failure:
            rul_hours = 0 if current_hi <= hi_failure else 10000
        else:
            # t_rul = -ln(HI_failure / HI_current) / λ
            ratio = max(0.01, hi_failure / (current_hi / 100))
            rul_hours = max(0, -np.log(ratio) / lambda_rate)
        
        # Intervalo de confiança (simplificado)
        n = len(hours)
        residuals = log_his - (intercept + slope * hours_arr)
        se = np.sqrt(np.sum(residuals**2) / (n - 2)) if n > 2 else 0.2
        
        rul_std = rul_hours * se * 2  # Factor de escala heurístico
        t_value = 1.96
        rul_lower = max(0, rul_hours - t_value * rul_std)
        rul_upper = rul_hours + t_value * rul_std
        
        # Confidence
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_his - np.mean(log_his))**2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        confidence = max(0.5, min(0.99, r2))
        
        return RULEstimate(
            timestamp=datetime.now(timezone.utc),
            rul_hours=min(rul_hours, 10000),
            rul_lower=min(rul_lower, 10000),
            rul_upper=min(rul_upper, 15000),
            confidence=confidence,
            degradation_rate=lambda_rate * (current_hi / 100),  # Aproximação linear local
            extrapolation_method="exponential",
            data_points_used=n,
        )
    
    def _fallback_rul(
        self,
        machine_id: str,
        current_hi: float,
        data_points: int,
    ) -> RULEstimate:
        """RUL fallback quando não há dados suficientes."""
        # Assumir taxa de degradação típica de 0.5% por hora
        default_rate = 0.5
        hi_failure = self.config.rul_failure_threshold
        
        rul_hours = max(0, (current_hi - hi_failure) / default_rate)
        
        return RULEstimate(
            timestamp=datetime.now(timezone.utc),
            rul_hours=min(rul_hours, 5000),
            rul_lower=rul_hours * 0.5,
            rul_upper=rul_hours * 2.0,
            confidence=0.3,  # Baixa confiança
            degradation_rate=default_rate,
            extrapolation_method="fallback",
            data_points_used=data_points,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPLAINABILITY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ExplainabilityEngine:
    """
    Gera explicações para a degradação observada.
    
    Analisa quais sensores mais contribuem para o desvio do estado saudável.
    """
    
    SENSOR_NAMES = [
        "vibration_x", "vibration_y", "vibration_z", "vibration_rms",
        "current_a", "current_b", "current_c", "current_rms",
        "temp_bearing", "temp_motor", "temp_ambient",
        "acoustic", "oil_particles", "pressure",
        "speed", "load", "power_factor",
    ]
    
    SENSOR_BASELINES = {
        "vibration_x": 0.1, "vibration_y": 0.1, "vibration_z": 0.1, "vibration_rms": 0.17,
        "current_a": 0.3, "current_b": 0.3, "current_c": 0.3, "current_rms": 0.52,
        "temp_bearing": 0.25, "temp_motor": 0.20, "temp_ambient": 0.25,
        "acoustic": 0.15, "oil_particles": 0.10, "pressure": 0.50,
        "speed": 0.70, "load": 0.50, "power_factor": 0.85,
    }
    
    def __init__(self, config: SHIDTConfig):
        self.config = config
        self._history: Dict[str, deque] = {}  # Para calcular trends
    
    def explain_degradation(
        self,
        machine_id: str,
        snapshot: SensorSnapshot,
        reconstruction_error: float,
        profile: OperationalProfile,
    ) -> List[SensorContribution]:
        """
        Identifica os sensores que mais contribuem para a degradação.
        
        Retorna os top K sensores ordenados por contribuição.
        """
        # Obter vector de sensores
        sensor_values = snapshot.to_vector()[:17]  # Só os 17 principais
        
        # Ajustar baselines pelo perfil
        profile_baseline = ProfileDetector(self.config).get_profile_baseline(profile)
        baseline_factor = profile_baseline.get("reconstruction_threshold_factor", 1.0)
        
        # Calcular desvios
        contributions = []
        total_deviation = 0.0
        
        for i, (name, value) in enumerate(zip(self.SENSOR_NAMES, sensor_values)):
            baseline = self.SENSOR_BASELINES.get(name, 0.5) * baseline_factor
            deviation = abs(value - baseline)
            total_deviation += deviation
            
            # Calcular trend
            trend = self._calculate_trend(machine_id, name, value)
            
            contributions.append({
                "name": name,
                "value": value,
                "baseline": baseline,
                "deviation": deviation,
                "trend": trend,
            })
        
        # Normalizar para percentagens
        if total_deviation > 0:
            for c in contributions:
                c["contribution_pct"] = (c["deviation"] / total_deviation) * 100
        else:
            for c in contributions:
                c["contribution_pct"] = 100 / len(contributions)
        
        # Ordenar por contribuição
        contributions.sort(key=lambda x: x["contribution_pct"], reverse=True)
        
        # Converter para objetos
        result = []
        for c in contributions[:self.config.top_k_contributors]:
            result.append(SensorContribution(
                sensor_name=c["name"],
                contribution_pct=c["contribution_pct"],
                current_value=c["value"],
                baseline_value=c["baseline"],
                deviation=c["deviation"],
                trend=c["trend"],
            ))
        
        return result
    
    def _calculate_trend(
        self,
        machine_id: str,
        sensor_name: str,
        current_value: float,
    ) -> str:
        """Calcula tendência do sensor (increasing/stable/decreasing)."""
        key = f"{machine_id}_{sensor_name}"
        
        if key not in self._history:
            self._history[key] = deque(maxlen=20)
        
        self._history[key].append(current_value)
        
        if len(self._history[key]) < 5:
            return "stable"
        
        values = list(self._history[key])
        recent_avg = np.mean(values[-5:])
        older_avg = np.mean(values[:-5]) if len(values) > 5 else recent_avg
        
        diff = recent_avg - older_avg
        threshold = 0.05  # 5% change threshold
        
        if diff > threshold:
            return "increasing"
        elif diff < -threshold:
            return "decreasing"
        else:
            return "stable"


# ═══════════════════════════════════════════════════════════════════════════════
# ALERT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class AlertGenerator:
    """Gera alertas baseados no estado de saúde."""
    
    def __init__(self, config: SHIDTConfig):
        self.config = config
        self._alert_counter = 0
        self._active_alerts: Dict[str, HealthAlert] = {}
    
    def generate_alerts(
        self,
        machine_id: str,
        hi: float,
        rul: Optional[RULEstimate],
        contributors: List[SensorContribution],
    ) -> List[HealthAlert]:
        """Gera alertas se necessário."""
        alerts = []
        now = datetime.now(timezone.utc)
        
        # Alert por HI baixo
        if hi < self.config.threshold_critical:
            alert = self._create_hi_alert(
                machine_id, hi, AlertSeverity.CRITICAL,
                "Estado Crítico",
                f"Índice de saúde em {hi:.1f}% - intervenção imediata recomendada",
                contributors,
            )
            if rul:
                alert.rul_estimate = rul
            alert.recommended_actions = [
                "Parar máquina e inspecionar componentes críticos",
                "Verificar sensores com maior desvio",
                "Contactar manutenção urgente",
            ]
            alerts.append(alert)
            
        elif hi < self.config.threshold_warning:
            alert = self._create_hi_alert(
                machine_id, hi, AlertSeverity.WARNING,
                "Degradação Detectada",
                f"Índice de saúde em {hi:.1f}% - monitorização atenta recomendada",
                contributors,
            )
            if rul:
                alert.rul_estimate = rul
            alert.recommended_actions = [
                "Agendar inspeção preventiva",
                "Monitorar tendência dos sensores destacados",
                "Preparar peças de substituição",
            ]
            alerts.append(alert)
        
        # Alert por RUL baixo
        if rul and rul.rul_hours < 100:
            severity = AlertSeverity.CRITICAL if rul.rul_hours < 24 else AlertSeverity.WARNING
            alert = self._create_rul_alert(
                machine_id, rul, severity,
                f"RUL Crítico: {rul.rul_hours:.0f}h",
                f"Vida útil estimada de {rul.rul_hours:.0f} horas (CI: {rul.rul_lower:.0f}-{rul.rul_upper:.0f}h)",
            )
            alerts.append(alert)
        
        return alerts
    
    def _create_hi_alert(
        self,
        machine_id: str,
        hi: float,
        severity: AlertSeverity,
        title: str,
        message: str,
        contributors: List[SensorContribution],
    ) -> HealthAlert:
        self._alert_counter += 1
        return HealthAlert(
            alert_id=f"HI-{machine_id}-{self._alert_counter}",
            machine_id=machine_id,
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            title=title,
            message=message,
            hi_current=hi,
            contributing_sensors=contributors,
        )
    
    def _create_rul_alert(
        self,
        machine_id: str,
        rul: RULEstimate,
        severity: AlertSeverity,
        title: str,
        message: str,
    ) -> HealthAlert:
        self._alert_counter += 1
        return HealthAlert(
            alert_id=f"RUL-{machine_id}-{self._alert_counter}",
            machine_id=machine_id,
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            title=title,
            message=message,
            hi_current=0,
            rul_estimate=rul,
            recommended_actions=[
                f"Planear manutenção dentro de {rul.rul_hours:.0f} horas",
                "Verificar disponibilidade de peças",
                "Considerar rerouting de produção",
            ],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SHI-DT SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class SHIDT:
    """
    Smart Health Index Digital Twin - Serviço Principal.
    
    Métodos públicos:
    - get_health_index(machine_id) -> float
    - get_rul(machine_id) -> float
    - get_full_status(machine_id) -> MachineHealthStatus
    - ingest_sensor_data(machine_id, snapshot)
    - update_model(machine_id, data) - para aprendizado online
    
    Uso:
        shi_dt = SHIDT()
        shi_dt.initialize()
        
        # Ingerir dados de sensores
        shi_dt.ingest_sensor_data("M-001", sensor_snapshot, context)
        
        # Obter índice de saúde
        hi = shi_dt.get_health_index("M-001")  # 0-100%
        rul = shi_dt.get_rul("M-001")  # horas
        
        # Obter status completo
        status = shi_dt.get_full_status("M-001")
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, config: Optional[SHIDTConfig] = None):
        self.config = config or SHIDTConfig()
        
        # Components
        self._models: Dict[str, CVAE] = {}  # Por máquina ou global
        self._global_model: Optional[CVAE] = None
        self._profile_detector = ProfileDetector(self.config)
        self._rul_calculator = RULCalculator(self.config)
        self._explainability = ExplainabilityEngine(self.config)
        self._alert_generator = AlertGenerator(self.config)
        
        # State
        self._machine_states: Dict[str, MachineHealthStatus] = {}
        self._hi_ema: Dict[str, float] = {}  # EMA de HI por máquina
        self._last_readings: Dict[str, HealthIndexReading] = {}
        self._online_buffer: Dict[str, List] = {}
        
        # Periodic re-training tracking
        self._last_retrain_time: Dict[str, datetime] = {}  # Por máquina
        self._retrain_data_buffer: Dict[str, List] = {}  # Buffer para re-treino periódico
        
        # Performance optimization: cache
        self._inference_cache: Dict[str, Tuple[datetime, float]] = {}  # Cache de inferências recentes
        self._cache_ttl_seconds: float = 1.0  # Cache válido por 1 segundo (inferência < 1s)
        
        self._initialized = False
    
    def initialize(
        self,
        model_path: Optional[Path] = None,
        train_demo: bool = True,
    ) -> bool:
        """
        Inicializa o sistema SHI-DT.
        
        Args:
            model_path: Caminho para modelo pré-treinado
            train_demo: Se True, treina com dados de demonstração
        
        Returns:
            True se inicialização bem sucedida
        """
        try:
            # Tentar carregar modelo existente
            if model_path and model_path.exists():
                self._global_model = self._load_model(model_path)
                logger.info(f"Modelo carregado de {model_path}")
            elif train_demo:
                # Treinar com dados de demonstração
                logger.info("Treinando modelo com dados de demonstração...")
                demo_data = create_demo_dataset(num_samples=500, num_machines=10)
                self._global_model = train_cvae(
                    demo_data,
                    self.config.cvae_config,
                )
                logger.info("Modelo treinado com sucesso")
            else:
                # Criar modelo não treinado (modo simulado)
                self._global_model = CVAE(self.config.cvae_config)
                logger.warning("Modelo não treinado - usando modo simulado")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Erro ao inicializar SHI-DT: {e}")
            # Fallback: criar modelo simulado
            self._global_model = CVAE(self.config.cvae_config)
            self._initialized = True
            return True
    
    def _load_model(self, path: Path) -> CVAE:
        """Carrega modelo de ficheiro."""
        if not TORCH_AVAILABLE:
            return CVAE(self.config.cvae_config)
        
        import torch
        checkpoint = torch.load(path)
        config = checkpoint.get("config", self.config.cvae_config)
        model = CVAE(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
    
    def get_health_index(self, machine_id: str) -> float:
        """
        Retorna o Health Index atual da máquina (0-100%).
        
        Método principal para consulta rápida.
        """
        if machine_id in self._hi_ema:
            return self._hi_ema[machine_id]
        return 100.0  # Default saudável se não há dados
    
    def get_rul(self, machine_id: str) -> float:
        """
        Retorna a Remaining Useful Life estimada (em horas).
        
        Retorna -1 se não há dados suficientes.
        """
        hi = self.get_health_index(machine_id)
        rul = self._rul_calculator.estimate_rul(machine_id, hi)
        return rul.rul_hours if rul else -1
    
    def get_full_status(self, machine_id: str) -> MachineHealthStatus:
        """
        Retorna o estado completo de saúde da máquina.
        
        Inclui HI, RUL, alertas, e contribuição de sensores.
        """
        if machine_id in self._machine_states:
            return self._machine_states[machine_id]
        
        # Estado default
        return MachineHealthStatus(
            machine_id=machine_id,
            timestamp=datetime.now(timezone.utc),
            health_index=100.0,
            health_index_std=0.0,
            status="HEALTHY",
            rul_estimate=None,
            current_profile=OperationalProfile.IDLE,
            degradation_trend="stable",
            degradation_rate=0.0,
            top_contributors=[],
            active_alerts=[],
            last_sensor_update=datetime.now(timezone.utc),
            model_version=self.VERSION,
        )
    
    def ingest_sensor_data(
        self,
        machine_id: str,
        snapshot: SensorSnapshot,
        context: Optional[OperationContext] = None,
    ) -> HealthIndexReading:
        """
        Ingere dados de sensores e atualiza o estado da máquina.
        
        Este é o método principal para alimentar o sistema com dados em tempo real.
        
        Args:
            machine_id: ID da máquina
            snapshot: Leitura dos sensores
            context: Contexto operacional (opcional)
        
        Returns:
            HealthIndexReading com o resultado processado
        """
        if not self._initialized:
            self.initialize()
        
        # Detectar perfil operacional
        profile = self._profile_detector.detect_profile(machine_id, snapshot)
        
        # Preparar contexto
        if context is None:
            context = OperationContext(
                machine_id=machine_id,
                op_code="UNKNOWN",
                product_type="UNKNOWN",
            )
        
        # Verificar cache para otimização de performance (< 1 segundo por inferência)
        cache_key = f"{machine_id}_{snapshot.timestamp.isoformat()}"
        now = datetime.now(timezone.utc)
        
        if cache_key in self._inference_cache:
            cache_time, cached_hi = self._inference_cache[cache_key]
            if (now - cache_time).total_seconds() < self._cache_ttl_seconds:
                # Usar resultado em cache (para múltiplas chamadas rápidas)
                hi_result = cached_hi
            else:
                # Cache expirado, recalcular
                del self._inference_cache[cache_key]
                model = self._models.get(machine_id, self._global_model)
                hi_result = infer_hi(model, snapshot, context, self.config.cvae_config)
                self._inference_cache[cache_key] = (now, hi_result)
        else:
            # Inferir Health Index
            model = self._models.get(machine_id, self._global_model)
            hi_result = infer_hi(model, snapshot, context, self.config.cvae_config)
            self._inference_cache[cache_key] = (now, hi_result)
        
        # Limpar cache antigo (manter apenas últimos 100)
        if len(self._inference_cache) > 100:
            # Remover entradas mais antigas
            sorted_cache = sorted(
                self._inference_cache.items(),
                key=lambda x: x[1][0]  # Ordenar por timestamp
            )
            self._inference_cache = dict(sorted_cache[-100:])
        
        # Verificar re-treino periódico
        self._check_periodic_retrain(machine_id)
        
        # Ajustar HI pelo perfil (diferentes regimes têm diferentes baselines)
        profile_baseline = self._profile_detector.get_profile_baseline(profile)
        threshold_factor = profile_baseline.get("reconstruction_threshold_factor", 1.0)
        
        # Calcular HI usando fórmula especificada: H(t) = 100 * exp(-α * E_rec(t))
        # Onde E_rec(t) é o erro de reconstrução do CVAE
        E_rec = hi_result.reconstruction_error
        
        # Ajustar α pelo perfil operacional (perfis mais pesados têm maior degradação)
        alpha_adjusted = self.config.hi_alpha * threshold_factor
        
        # Aplicar fórmula: H(t) = 100 * exp(-α * E_rec(t))
        hi_raw = 100.0 * math.exp(-alpha_adjusted * E_rec)
        
        # Garantir que está no intervalo [0, 100]
        hi_raw = max(0.0, min(100.0, hi_raw))
        
        # Aplicar EMA para suavização
        alpha = self.config.hi_ema_alpha
        if machine_id in self._hi_ema:
            hi_smoothed = alpha * hi_raw + (1 - alpha) * self._hi_ema[machine_id]
        else:
            hi_smoothed = hi_raw
        
        self._hi_ema[machine_id] = hi_smoothed
        
        # Criar reading
        reading = HealthIndexReading(
            timestamp=snapshot.timestamp,
            hi_raw=hi_raw,
            hi_smoothed=hi_smoothed,
            reconstruction_error=hi_result.reconstruction_error,
            profile=profile,
            latent_vector=hi_result.latent_mean,
        )
        
        self._last_readings[machine_id] = reading
        
        # Atualizar histórico para RUL
        self._rul_calculator.add_observation(machine_id, snapshot.timestamp, hi_smoothed)
        
        # Calcular RUL
        rul = self._rul_calculator.estimate_rul(machine_id, hi_smoothed)
        
        # Explicabilidade
        contributors = self._explainability.explain_degradation(
            machine_id, snapshot, hi_result.reconstruction_error, profile
        )
        
        # Determinar status
        if hi_smoothed >= self.config.threshold_healthy:
            status = "HEALTHY"
        elif hi_smoothed >= self.config.threshold_warning:
            status = "WARNING"
        else:
            status = "CRITICAL"
        
        # Calcular trend de degradação
        degradation_trend = "stable"
        degradation_rate = 0.0
        if rul:
            degradation_rate = rul.degradation_rate
            if degradation_rate > 0.1:
                degradation_trend = "degrading"
            elif degradation_rate < -0.1:
                degradation_trend = "improving"
        
        # Gerar alertas
        alerts = self._alert_generator.generate_alerts(
            machine_id, hi_smoothed, rul, contributors
        )
        
        # Atualizar estado completo
        self._machine_states[machine_id] = MachineHealthStatus(
            machine_id=machine_id,
            timestamp=snapshot.timestamp,
            health_index=hi_smoothed,
            health_index_std=hi_result.hi_std * 100,
            status=status,
            rul_estimate=rul,
            current_profile=profile,
            degradation_trend=degradation_trend,
            degradation_rate=degradation_rate,
            top_contributors=contributors,
            active_alerts=alerts,
            last_sensor_update=snapshot.timestamp,
            model_version=self.VERSION,
        )
        
        # Online learning buffer
        if self.config.online_learning_enabled:
            if machine_id not in self._online_buffer:
                self._online_buffer[machine_id] = []
            self._online_buffer[machine_id].append((snapshot, context, hi_smoothed / 100))
            
            # Trigger update se buffer cheio
            if len(self._online_buffer[machine_id]) >= self.config.online_learning_update_interval:
                self._update_model_online(machine_id)
        
        return reading
    
    def _update_model_online(self, machine_id: str) -> None:
        """
        Atualiza modelo com dados recentes (online learning).
        
        Implementa re-treino incremental para melhorar precisão ao longo do tempo.
        """
        if machine_id not in self._online_buffer:
            return
        
        data = self._online_buffer[machine_id]
        if len(data) < 50:
            return
        
        logger.info(f"Online update para máquina {machine_id} com {len(data)} amostras")
        
        try:
            # Preparar dados para re-treino
            training_data = []
            for snapshot, context, hi_target in data:
                training_data.append((snapshot, context, hi_target))
            
            # Usar modelo global ou específico da máquina
            base_model = self._models.get(machine_id, self._global_model)
            if base_model is None:
                logger.warning(f"Sem modelo base para {machine_id}, usando dados demo")
                demo_data = create_demo_dataset(num_samples=100, num_machines=1)
                base_model = train_cvae(demo_data, self.config.cvae_config)
            
            # Re-treinar com dados recentes (fine-tuning)
            if TORCH_AVAILABLE and hasattr(base_model, 'train'):
                # Combinar dados antigos (se disponíveis) com novos
                if machine_id in self._retrain_data_buffer:
                    training_data.extend(self._retrain_data_buffer[machine_id])
                
                # Limitar tamanho do dataset para performance
                if len(training_data) > 1000:
                    training_data = training_data[-1000:]  # Manter apenas últimos 1000
                
                # Re-treinar modelo
                new_model = train_cvae(
                    training_data,
                    self.config.cvae_config,
                )
                
                # Atualizar modelo
                self._models[machine_id] = new_model
                self._last_retrain_time[machine_id] = datetime.now(timezone.utc)
                
                # Guardar dados para próximo re-treino (amostragem)
                self._retrain_data_buffer[machine_id] = training_data[::10]  # Guardar 10% dos dados
                
                logger.info(f"Modelo re-treinado para {machine_id} com {len(training_data)} amostras")
        except Exception as e:
            logger.warning(f"Online update falhou para {machine_id}: {e}")
        
        # Limpar buffer
        self._online_buffer[machine_id] = []
    
    def _check_periodic_retrain(self, machine_id: str) -> None:
        """
        Verifica se é necessário re-treino periódico baseado em tempo.
        
        Re-treina automaticamente após intervalo configurado (ex: semanalmente).
        """
        if not self.config.online_learning_enabled:
            return
        
        now = datetime.now(timezone.utc)
        last_retrain = self._last_retrain_time.get(machine_id)
        
        # Verificar se passou tempo suficiente desde último re-treino
        if last_retrain is None:
            # Primeira vez, marcar como agora
            self._last_retrain_time[machine_id] = now
            return
        
        hours_since_retrain = (now - last_retrain).total_seconds() / 3600.0
        
        if hours_since_retrain >= self.config.periodic_retrain_interval_hours:
            # Verificar se há dados suficientes
            buffer_size = len(self._online_buffer.get(machine_id, []))
            retrain_buffer_size = len(self._retrain_data_buffer.get(machine_id, []))
            total_samples = buffer_size + retrain_buffer_size
            
            if total_samples >= self.config.periodic_retrain_min_samples:
                logger.info(
                    f"Re-treino periódico para {machine_id} "
                    f"({hours_since_retrain:.1f}h desde último, {total_samples} amostras)"
                )
                self._update_model_online(machine_id)
    
    def update_model(
        self,
        machine_id: str,
        training_data: List[Tuple[SensorSnapshot, OperationContext, float]],
    ) -> bool:
        """
        Atualiza modelo com novos dados de treino.
        
        Permite refinar o modelo para uma máquina específica.
        """
        try:
            logger.info(f"Atualizando modelo para {machine_id} com {len(training_data)} amostras")
            
            new_model = train_cvae(training_data, self.config.cvae_config)
            self._models[machine_id] = new_model
            
            logger.info(f"Modelo atualizado para {machine_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao atualizar modelo: {e}")
            return False
    
    def get_all_machines_status(self) -> Dict[str, MachineHealthStatus]:
        """Retorna status de todas as máquinas monitorizadas."""
        return self._machine_states.copy()
    
    def export_metrics(self) -> Dict[str, Any]:
        """Exporta métricas para integração com sistemas externos."""
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": self.VERSION,
            "machines_monitored": len(self._machine_states),
            "machines": {},
        }
        
        for machine_id, status in self._machine_states.items():
            metrics["machines"][machine_id] = {
                "health_index": status.health_index,
                "status": status.status,
                "rul_hours": status.rul_estimate.rul_hours if status.rul_estimate else None,
                "profile": status.current_profile.value,
                "alerts_count": len(status.active_alerts),
            }
        
        return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

_shidt_instance: Optional[SHIDT] = None


def get_shidt() -> SHIDT:
    """
    Obtém instância singleton do SHI-DT.
    
    Uso:
        shi_dt = get_shidt()
        hi = shi_dt.get_health_index("M-001")
    """
    global _shidt_instance
    
    if _shidt_instance is None:
        _shidt_instance = SHIDT()
        _shidt_instance.initialize(train_demo=True)
    
    return _shidt_instance


def reset_shidt() -> None:
    """Reset da instância (para testes)."""
    global _shidt_instance
    _shidt_instance = None

