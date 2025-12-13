"""
════════════════════════════════════════════════════════════════════════════════════════════════════
PREDICTIVE CARE SERVICE - Manutenção Preditiva Proativa
════════════════════════════════════════════════════════════════════════════════════════════════════

Serviço central de PredictiveCare que integra:
- SHI-DT (Health Index via CVAE)
- RUL Estimator (Remaining Useful Life)
- Anomaly Detection
- Risk Assessment

Fornece uma visão unificada do estado de saúde das máquinas para:
- Dashboard da aba Máquinas (ProdPlan)
- Geração automática de ordens de manutenção
- Integração com agendamento PdM-IPS

R&D / SIFIDE: WP1 - Digital Twin para manutenção preditiva
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class MachineHealthState(str, Enum):
    """Estado de saúde da máquina."""
    HEALTHY = "HEALTHY"      # SHI > 80%, RUL > 720h
    WARNING = "WARNING"      # SHI 50-80% ou RUL 168-720h
    CRITICAL = "CRITICAL"    # SHI 30-50% ou RUL 24-168h
    EMERGENCY = "EMERGENCY"  # SHI < 30% ou RUL < 24h


class RiskLevel(str, Enum):
    """Nível de risco de falha."""
    LOW = "LOW"           # < 10%
    MEDIUM = "MEDIUM"     # 10-30%
    HIGH = "HIGH"         # 30-60%
    CRITICAL = "CRITICAL" # > 60%


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SensorContribution:
    """Contribuição de um sensor para o estado de saúde."""
    sensor_type: str
    contribution_percent: float  # % de contribuição para degradação
    current_value: float
    baseline_value: float
    deviation_percent: float
    is_anomalous: bool = False


@dataclass
class MachinePredictiveState:
    """
    Estado preditivo completo de uma máquina.
    
    Agrega todas as informações relevantes para PredictiveCare:
    - Health Index (SHI)
    - Remaining Useful Life (RUL)
    - Anomaly Score
    - Risk Probabilities
    - Top Contributors
    """
    machine_id: str
    timestamp: datetime
    
    # Health Index (0-100)
    shi_percent: float
    shi_trend: str = "stable"  # "improving", "stable", "degrading"
    
    # Anomaly Detection
    anomaly_score: float = 0.0  # 0-1, higher = more anomalous
    is_anomalous: bool = False
    
    # RUL (Remaining Useful Life)
    rul_hours: float = float('inf')
    rul_lower_hours: float = float('inf')
    rul_upper_hours: float = float('inf')
    rul_confidence: float = 0.0
    
    # Failure Risk
    risk_next_7d: float = 0.0  # Probability of failure in next 7 days
    risk_next_30d: float = 0.0  # Probability of failure in next 30 days
    risk_level: RiskLevel = RiskLevel.LOW
    
    # Overall State
    state: MachineHealthState = MachineHealthState.HEALTHY
    
    # Top Contributors to degradation
    top_contributors: List[SensorContribution] = field(default_factory=list)
    
    # Metadata
    last_maintenance: Optional[datetime] = None
    hours_since_maintenance: float = 0.0
    total_operation_hours: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "machine_id": self.machine_id,
            "timestamp": self.timestamp.isoformat(),
            "shi_percent": round(self.shi_percent, 1),
            "shi_trend": self.shi_trend,
            "anomaly_score": round(self.anomaly_score, 3),
            "is_anomalous": self.is_anomalous,
            "rul_hours": round(self.rul_hours, 1) if self.rul_hours != float('inf') else None,
            "rul_lower_hours": round(self.rul_lower_hours, 1) if self.rul_lower_hours != float('inf') else None,
            "rul_upper_hours": round(self.rul_upper_hours, 1) if self.rul_upper_hours != float('inf') else None,
            "rul_confidence": round(self.rul_confidence, 2),
            "risk_next_7d": round(self.risk_next_7d, 3),
            "risk_next_30d": round(self.risk_next_30d, 3),
            "risk_level": self.risk_level.value,
            "state": self.state.value,
            "top_contributors": [
                {
                    "sensor_type": c.sensor_type,
                    "contribution_percent": round(c.contribution_percent, 1),
                    "current_value": round(c.current_value, 3),
                    "deviation_percent": round(c.deviation_percent, 1),
                    "is_anomalous": c.is_anomalous,
                }
                for c in self.top_contributors
            ],
            "last_maintenance": self.last_maintenance.isoformat() if self.last_maintenance else None,
            "hours_since_maintenance": round(self.hours_since_maintenance, 1),
        }


@dataclass
class PredictiveCareConfig:
    """Configuration for PredictiveCare service."""
    # Health Index thresholds
    shi_healthy_threshold: float = 80.0
    shi_warning_threshold: float = 50.0
    shi_critical_threshold: float = 30.0
    
    # RUL thresholds (hours)
    rul_healthy_threshold: float = 720.0  # 30 days
    rul_warning_threshold: float = 168.0  # 7 days
    rul_critical_threshold: float = 24.0  # 1 day
    
    # Anomaly detection
    anomaly_threshold: float = 0.7  # Score above this = anomalous
    
    # Risk calculation
    risk_low_threshold: float = 0.10
    risk_medium_threshold: float = 0.30
    risk_high_threshold: float = 0.60
    
    # Work order creation
    auto_create_workorder_risk_threshold: float = 0.30


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTIVE CARE SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class PredictiveCareService:
    """
    Serviço central de PredictiveCare.
    
    Integra SHI-DT, RUL e análise de risco para fornecer
    uma visão completa do estado das máquinas.
    """
    
    def __init__(
        self,
        config: Optional[PredictiveCareConfig] = None,
        shi_dt_service=None,
        rul_estimator=None,
        iot_service=None,
    ):
        """
        Initialize PredictiveCare service.
        
        Args:
            config: Configuration options
            shi_dt_service: SHI-DT service instance (optional, will import if None)
            rul_estimator: RUL estimator instance (optional)
            iot_service: IoT ingestion service (optional)
        """
        self.config = config or PredictiveCareConfig()
        self._shi_dt_service = shi_dt_service
        self._rul_estimator = rul_estimator
        self._iot_service = iot_service
        
        # Cache for recent states
        self._state_cache: Dict[str, MachinePredictiveState] = {}
        self._cache_ttl_seconds = 60  # 1 minute cache
        
        logger.info("PredictiveCareService initialized")
    
    def get_machine_state(self, machine_id: str) -> MachinePredictiveState:
        """
        Get complete predictive state for a machine.
        
        Aggregates:
        - Current Health Index (SHI)
        - Remaining Useful Life (RUL)
        - Anomaly Score
        - Failure Risk
        - Top Contributors
        
        Args:
            machine_id: Machine identifier
            
        Returns:
            MachinePredictiveState with all metrics
        """
        now = datetime.now(timezone.utc)
        
        # Check cache
        if machine_id in self._state_cache:
            cached = self._state_cache[machine_id]
            age = (now - cached.timestamp).total_seconds()
            if age < self._cache_ttl_seconds:
                return cached
        
        # Build state
        state = self._build_machine_state(machine_id, now)
        
        # Cache it
        self._state_cache[machine_id] = state
        
        return state
    
    def get_all_machines_state(
        self,
        machine_ids: Optional[List[str]] = None,
    ) -> List[MachinePredictiveState]:
        """
        Get predictive state for all machines.
        
        Args:
            machine_ids: Optional list of specific machines (default: all known)
            
        Returns:
            List of MachinePredictiveState
        """
        if machine_ids is None:
            # Get from SHI-DT or default demo machines
            machine_ids = self._get_known_machines()
        
        states = []
        for mid in machine_ids:
            try:
                state = self.get_machine_state(mid)
                states.append(state)
            except Exception as e:
                logger.warning(f"Failed to get state for {mid}: {e}")
        
        # Sort by risk (highest first)
        states.sort(key=lambda s: s.risk_next_7d, reverse=True)
        
        return states
    
    def compute_failure_risk(
        self,
        machine_id: str,
        horizon_days: int = 7,
    ) -> float:
        """
        Compute failure probability within a time horizon.
        
        Uses:
        - Current SHI and degradation rate
        - RUL estimate and confidence
        - Historical failure patterns
        
        Args:
            machine_id: Machine identifier
            horizon_days: Time horizon in days
            
        Returns:
            Probability of failure (0-1)
        """
        state = self.get_machine_state(machine_id)
        
        horizon_hours = horizon_days * 24
        
        # Method 1: From RUL
        if state.rul_hours != float('inf') and state.rul_hours > 0:
            # Probability that failure occurs before horizon
            # Using exponential distribution approximation
            rate = 1 / state.rul_hours
            risk_from_rul = 1 - math.exp(-rate * horizon_hours)
        else:
            risk_from_rul = 0.0
        
        # Method 2: From SHI degradation
        shi_normalized = state.shi_percent / 100.0
        # Lower SHI = higher risk
        risk_from_shi = max(0, 1 - shi_normalized) ** 2
        
        # Method 3: Anomaly score contribution
        risk_from_anomaly = state.anomaly_score * 0.3  # 30% weight max
        
        # Combine (weighted average)
        combined_risk = (
            0.5 * risk_from_rul +
            0.3 * risk_from_shi +
            0.2 * risk_from_anomaly
        )
        
        return min(1.0, max(0.0, combined_risk))
    
    def get_machines_requiring_attention(
        self,
        risk_threshold: float = 0.3,
    ) -> List[MachinePredictiveState]:
        """
        Get machines that require attention based on risk.
        
        Args:
            risk_threshold: Minimum risk to include
            
        Returns:
            List of machines above threshold, sorted by risk
        """
        all_states = self.get_all_machines_state()
        
        requiring_attention = [
            s for s in all_states
            if s.risk_next_7d >= risk_threshold or s.state in (
                MachineHealthState.CRITICAL,
                MachineHealthState.EMERGENCY,
            )
        ]
        
        return requiring_attention
    
    def invalidate_cache(self, machine_id: Optional[str] = None):
        """Invalidate state cache."""
        if machine_id:
            self._state_cache.pop(machine_id, None)
        else:
            self._state_cache.clear()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRIVATE METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _build_machine_state(
        self,
        machine_id: str,
        timestamp: datetime,
    ) -> MachinePredictiveState:
        """Build complete machine state."""
        
        # Get SHI from SHI-DT
        shi_result = self._get_shi(machine_id)
        shi_percent = shi_result.get("hi_percent", 100.0)
        reconstruction_error = shi_result.get("reconstruction_error", 0.0)
        
        # Get RUL
        rul_result = self._get_rul(machine_id)
        rul_hours = rul_result.get("rul_mean_hours", float('inf'))
        rul_lower = rul_result.get("rul_lower_hours", float('inf'))
        rul_upper = rul_result.get("rul_upper_hours", float('inf'))
        rul_confidence = rul_result.get("confidence", 0.0)
        
        # Calculate anomaly score from reconstruction error
        # Higher error = more anomalous
        anomaly_score = min(1.0, reconstruction_error * 2)
        is_anomalous = anomaly_score >= self.config.anomaly_threshold
        
        # Calculate risk
        risk_7d = self._calculate_risk(shi_percent, rul_hours, anomaly_score, 7)
        risk_30d = self._calculate_risk(shi_percent, rul_hours, anomaly_score, 30)
        
        # Determine state
        state = self._determine_state(shi_percent, rul_hours)
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_7d)
        
        # Determine trend
        trend = self._get_shi_trend(machine_id)
        
        # Get top contributors
        contributors = shi_result.get("top_contributors", [])
        sensor_contributions = [
            SensorContribution(
                sensor_type=c.get("sensor", "unknown"),
                contribution_percent=c.get("contribution", 0) * 100,
                current_value=c.get("value", 0),
                baseline_value=c.get("baseline", 0),
                deviation_percent=c.get("deviation", 0) * 100,
                is_anomalous=c.get("is_anomalous", False),
            )
            for c in contributors[:5]  # Top 5
        ]
        
        return MachinePredictiveState(
            machine_id=machine_id,
            timestamp=timestamp,
            shi_percent=shi_percent,
            shi_trend=trend,
            anomaly_score=anomaly_score,
            is_anomalous=is_anomalous,
            rul_hours=rul_hours,
            rul_lower_hours=rul_lower,
            rul_upper_hours=rul_upper,
            rul_confidence=rul_confidence,
            risk_next_7d=risk_7d,
            risk_next_30d=risk_30d,
            risk_level=risk_level,
            state=state,
            top_contributors=sensor_contributions,
        )
    
    def _get_shi(self, machine_id: str) -> Dict[str, Any]:
        """Get SHI from SHI-DT service or fallback."""
        if self._shi_dt_service is not None:
            try:
                result = self._shi_dt_service.get_health_index(machine_id)
                return {
                    "hi_percent": result.hi_smoothed * 100 if hasattr(result, 'hi_smoothed') else result.get("hi_percent", 100),
                    "reconstruction_error": result.reconstruction_error if hasattr(result, 'reconstruction_error') else result.get("reconstruction_error", 0),
                    "top_contributors": result.top_contributors if hasattr(result, 'top_contributors') else [],
                }
            except Exception as e:
                logger.warning(f"Failed to get SHI for {machine_id}: {e}")
        
        # Fallback: simulate based on machine_id hash
        import hashlib
        h = int(hashlib.md5(machine_id.encode()).hexdigest(), 16)
        base_hi = 60 + (h % 40)  # 60-100
        noise = (h % 100) / 100 * 10  # 0-10
        
        return {
            "hi_percent": base_hi - noise,
            "reconstruction_error": noise / 50,
            "top_contributors": [
                {"sensor": "vibration", "contribution": 0.4, "deviation": 0.15},
                {"sensor": "temperature", "contribution": 0.3, "deviation": 0.10},
                {"sensor": "current", "contribution": 0.2, "deviation": 0.05},
            ],
        }
    
    def _get_rul(self, machine_id: str) -> Dict[str, Any]:
        """Get RUL from estimator or fallback."""
        if self._rul_estimator is not None:
            try:
                result = self._rul_estimator.estimate(machine_id)
                return {
                    "rul_mean_hours": result.rul_mean_hours,
                    "rul_lower_hours": result.rul_lower_hours,
                    "rul_upper_hours": result.rul_upper_hours,
                    "confidence": result.confidence,
                }
            except Exception as e:
                logger.warning(f"Failed to get RUL for {machine_id}: {e}")
        
        # Fallback: estimate from SHI
        shi_result = self._get_shi(machine_id)
        shi = shi_result.get("hi_percent", 100)
        
        # Simple linear model: RUL ≈ k * SHI
        # At SHI=100, RUL~1000h; at SHI=20, RUL~0
        k = 12.5  # hours per % SHI
        rul_mean = max(0, (shi - 20) * k)
        
        return {
            "rul_mean_hours": rul_mean,
            "rul_lower_hours": rul_mean * 0.7,
            "rul_upper_hours": rul_mean * 1.5,
            "confidence": 0.6,
        }
    
    def _calculate_risk(
        self,
        shi: float,
        rul_hours: float,
        anomaly_score: float,
        horizon_days: int,
    ) -> float:
        """Calculate failure risk probability."""
        horizon_hours = horizon_days * 24
        
        # Risk from RUL
        if rul_hours != float('inf') and rul_hours > 0:
            risk_rul = 1 - math.exp(-horizon_hours / rul_hours)
        else:
            risk_rul = 0.0
        
        # Risk from SHI (exponential increase as SHI drops)
        shi_norm = shi / 100
        risk_shi = (1 - shi_norm) ** 1.5
        
        # Combine
        risk = 0.6 * risk_rul + 0.3 * risk_shi + 0.1 * anomaly_score
        
        return min(1.0, max(0.0, risk))
    
    def _determine_state(self, shi: float, rul_hours: float) -> MachineHealthState:
        """Determine overall health state."""
        cfg = self.config
        
        if shi < cfg.shi_critical_threshold or (rul_hours != float('inf') and rul_hours < cfg.rul_critical_threshold):
            return MachineHealthState.EMERGENCY
        elif shi < cfg.shi_warning_threshold or (rul_hours != float('inf') and rul_hours < cfg.rul_warning_threshold):
            return MachineHealthState.CRITICAL
        elif shi < cfg.shi_healthy_threshold or (rul_hours != float('inf') and rul_hours < cfg.rul_healthy_threshold):
            return MachineHealthState.WARNING
        else:
            return MachineHealthState.HEALTHY
    
    def _determine_risk_level(self, risk: float) -> RiskLevel:
        """Determine risk level from probability."""
        cfg = self.config
        
        if risk >= cfg.risk_high_threshold:
            return RiskLevel.CRITICAL
        elif risk >= cfg.risk_medium_threshold:
            return RiskLevel.HIGH
        elif risk >= cfg.risk_low_threshold:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _get_shi_trend(self, machine_id: str) -> str:
        """Get SHI trend from historical data."""
        # TODO: Implement actual trend calculation from historical data
        # For now, return stable
        return "stable"
    
    def _get_known_machines(self) -> List[str]:
        """Get list of known machines."""
        # Try to get from SHI-DT or IoT service
        if self._shi_dt_service is not None:
            try:
                return list(self._shi_dt_service.get_all_machines())
            except:
                pass
        
        # Fallback: return demo machines
        return [
            "MACH-001", "MACH-002", "MACH-003", "MACH-004", "MACH-005",
            "CNC-001", "CNC-002", "ROBOT-001", "ROBOT-002", "PRESS-001",
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

_service_instance: Optional[PredictiveCareService] = None


def get_predictive_care_service() -> PredictiveCareService:
    """Get or create the PredictiveCare service singleton."""
    global _service_instance
    if _service_instance is None:
        _service_instance = PredictiveCareService()
    return _service_instance


