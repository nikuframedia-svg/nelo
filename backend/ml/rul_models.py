"""
ProdPlan 4.0 - Remaining Useful Life (RUL) Estimation

This module provides predictive maintenance capabilities:
- RUL estimation (time until failure/maintenance needed)
- Degradation modeling
- Failure probability estimation

Models support:
- Statistical (Weibull, Exponential)
- Bayesian (with uncertainty quantification)
- ML (survival analysis, deep learning)

R&D / SIFIDE: WP2 - Predictive Intelligence
Research Questions:
- Q2.5: Can Bayesian RUL models provide accurate uncertainty estimates?
- Q2.6: What sensor features are most predictive of machine degradation?
Metrics: MAE, coverage of prediction intervals, early warning accuracy.

References:
- Si et al. (2011). Remaining useful life estimation - A review
- Nectoux et al. (2012). PRONOSTIA: An experimental platform for bearings
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS AND CONFIG
# ============================================================

class RULModel(Enum):
    """Available RUL estimation models."""
    STATISTICAL = "statistical"  # Weibull/Exponential
    BAYESIAN = "bayesian"        # Bayesian with uncertainty
    ML = "ml"                    # Machine learning based
    DEEP = "deep"                # Deep learning (LSTM, etc.)


class DegradationModel(Enum):
    """Degradation trajectory models."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    WIENER = "wiener"  # Wiener process
    GAMMA = "gamma"    # Gamma process


@dataclass
class RULConfig:
    """Configuration for RUL estimation."""
    model: RULModel = RULModel.STATISTICAL
    degradation_model: DegradationModel = DegradationModel.LINEAR
    
    # Thresholds
    failure_threshold: float = 1.0  # Normalized degradation level
    warning_threshold: float = 0.8  # When to trigger warning
    
    # Statistical parameters
    weibull_shape: float = 2.0  # Shape parameter (β)
    weibull_scale: float = 1000.0  # Scale parameter (η) in hours
    
    # Bayesian parameters
    prior_mean_hours: float = 500.0
    prior_std_hours: float = 200.0
    n_samples: int = 1000
    
    # ML parameters
    n_estimators: int = 100
    sequence_length: int = 50  # For deep learning


@dataclass
class RULEstimate:
    """Result of RUL estimation."""
    # Point estimate
    mean_rul_hours: float
    
    # Uncertainty
    std_rul_hours: float
    lower_bound_hours: float  # e.g., 5th percentile
    upper_bound_hours: float  # e.g., 95th percentile
    confidence_level: float = 0.90
    
    # Probability estimates
    failure_probability_24h: float = 0.0
    failure_probability_7d: float = 0.0
    failure_probability_30d: float = 0.0
    
    # Health indicator
    health_index: float = 1.0  # 1 = healthy, 0 = failed
    
    # Recommendation
    maintenance_urgency: str = "low"  # low, medium, high, critical
    recommended_action: str = ""
    
    # Metadata
    model_used: str = ""
    estimation_time: datetime = field(default_factory=datetime.now)
    features_used: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mean_rul_hours': round(self.mean_rul_hours, 1),
            'std_rul_hours': round(self.std_rul_hours, 1),
            'lower_bound_hours': round(self.lower_bound_hours, 1),
            'upper_bound_hours': round(self.upper_bound_hours, 1),
            'confidence_level': self.confidence_level,
            'failure_probability_24h': round(self.failure_probability_24h, 4),
            'failure_probability_7d': round(self.failure_probability_7d, 4),
            'failure_probability_30d': round(self.failure_probability_30d, 4),
            'health_index': round(self.health_index, 3),
            'maintenance_urgency': self.maintenance_urgency,
            'recommended_action': self.recommended_action,
            'model_used': self.model_used,
        }


# ============================================================
# ABSTRACT BASE CLASS
# ============================================================

class RULEstimator(ABC):
    """
    Abstract base class for RUL estimation.
    
    RUL = Remaining Useful Life = time until maintenance/failure
    
    Input: sensor data, operational data, age
    Output: RUL estimate with uncertainty
    """
    
    def __init__(self, config: Optional[RULConfig] = None):
        self.config = config or RULConfig()
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, historical_data: pd.DataFrame) -> None:
        """
        Fit model to historical failure/maintenance data.
        
        Args:
            historical_data: DataFrame with columns:
                - machine_id
                - time_to_failure (or censoring time)
                - event (1=failure, 0=censored)
                - features (sensor readings, operating conditions)
        """
        pass
    
    @abstractmethod
    def estimate(
        self,
        machine_id: str,
        current_age_hours: float,
        sensor_features: Optional[Dict[str, float]] = None
    ) -> RULEstimate:
        """
        Estimate RUL for a specific machine.
        
        Args:
            machine_id: Machine identifier
            current_age_hours: Current age since last maintenance
            sensor_features: Current sensor readings
        
        Returns:
            RULEstimate with prediction and uncertainty
        """
        pass
    
    def _compute_urgency(self, rul_hours: float, health_index: float) -> str:
        """Compute maintenance urgency level."""
        if rul_hours < 24 or health_index < 0.2:
            return "critical"
        elif rul_hours < 72 or health_index < 0.4:
            return "high"
        elif rul_hours < 168 or health_index < 0.6:  # 1 week
            return "medium"
        else:
            return "low"
    
    def _get_recommendation(self, urgency: str, rul_hours: float) -> str:
        """Generate maintenance recommendation."""
        if urgency == "critical":
            return f"URGENTE: Agendar manutenção imediata. RUL estimado: {rul_hours:.0f}h"
        elif urgency == "high":
            return f"Agendar manutenção nas próximas 48h. RUL estimado: {rul_hours:.0f}h"
        elif urgency == "medium":
            return f"Planear manutenção para próxima semana. RUL estimado: {rul_hours:.0f}h"
        else:
            return f"Máquina saudável. Próxima manutenção em ~{rul_hours:.0f}h"


# ============================================================
# STATISTICAL RUL ESTIMATOR
# ============================================================

class StatisticalRULEstimator(RULEstimator):
    """
    Statistical RUL estimation using Weibull distribution.
    
    Weibull is commonly used because:
    - Flexible shape (captures infant mortality, random, wear-out failures)
    - Closed-form expressions for RUL
    - Interpretable parameters
    
    RUL at time t: R(t) = exp(-(t/η)^β)
    where β = shape, η = scale
    """
    
    def __init__(self, config: Optional[RULConfig] = None):
        super().__init__(config)
        self._machine_params: Dict[str, Tuple[float, float]] = {}  # machine_id -> (shape, scale)
        self._default_params = (self.config.weibull_shape, self.config.weibull_scale)
    
    def fit(self, historical_data: pd.DataFrame) -> None:
        """
        Fit Weibull parameters from historical failure data.
        
        Uses Maximum Likelihood Estimation (MLE).
        """
        if historical_data.empty:
            logger.warning("No historical data, using default parameters")
            self._is_fitted = True
            return
        
        try:
            from scipy import stats
            from scipy.optimize import minimize
            
            # Fit per machine if enough data
            for machine_id in historical_data['machine_id'].unique():
                machine_data = historical_data[historical_data['machine_id'] == machine_id]
                
                if len(machine_data) >= 5:
                    # Fit Weibull to time-to-failure data
                    ttf = machine_data['time_to_failure'].values
                    
                    # MLE for Weibull
                    shape, loc, scale = stats.weibull_min.fit(ttf, floc=0)
                    self._machine_params[machine_id] = (shape, scale)
                    
            self._is_fitted = True
            logger.info(f"Fitted Weibull parameters for {len(self._machine_params)} machines")
            
        except ImportError:
            logger.warning("scipy not available, using default parameters")
            self._is_fitted = True
    
    def estimate(
        self,
        machine_id: str,
        current_age_hours: float,
        sensor_features: Optional[Dict[str, float]] = None
    ) -> RULEstimate:
        """Estimate RUL using Weibull distribution."""
        
        # Get parameters for this machine
        shape, scale = self._machine_params.get(machine_id, self._default_params)
        
        # Conditional RUL given survival to current age
        # E[T - t | T > t] for Weibull
        
        try:
            from scipy import stats
            from scipy.integrate import quad
            
            # Survival function at current age
            S_t = np.exp(-(current_age_hours / scale) ** shape)
            
            if S_t < 0.001:
                # Already past expected life
                mean_rul = 0.0
                std_rul = 0.0
            else:
                # Conditional expectation of remaining life
                def integrand(x):
                    return np.exp(-((current_age_hours + x) / scale) ** shape)
                
                mean_rul, _ = quad(integrand, 0, scale * 10)
                mean_rul = mean_rul / S_t
                
                # Approximate std using Weibull properties
                std_rul = scale * np.sqrt(
                    np.math.gamma(1 + 2/shape) - np.math.gamma(1 + 1/shape)**2
                ) / S_t
            
            # Confidence interval
            z = 1.645  # 90% CI
            lower = max(0, mean_rul - z * std_rul)
            upper = mean_rul + z * std_rul
            
            # Failure probabilities
            prob_24h = 1 - np.exp(-((current_age_hours + 24) / scale) ** shape) / S_t if S_t > 0 else 1.0
            prob_7d = 1 - np.exp(-((current_age_hours + 168) / scale) ** shape) / S_t if S_t > 0 else 1.0
            prob_30d = 1 - np.exp(-((current_age_hours + 720) / scale) ** shape) / S_t if S_t > 0 else 1.0
            
            # Health index (inverse of failure probability)
            health = S_t
            
        except Exception as e:
            logger.warning(f"Scipy calculation failed: {e}, using simple estimate")
            # Simple fallback
            expected_life = scale * np.math.gamma(1 + 1/shape)
            mean_rul = max(0, expected_life - current_age_hours)
            std_rul = mean_rul * 0.3
            lower = max(0, mean_rul * 0.5)
            upper = mean_rul * 1.5
            prob_24h = min(1, current_age_hours / expected_life)
            prob_7d = min(1, (current_age_hours + 168) / expected_life)
            prob_30d = min(1, (current_age_hours + 720) / expected_life)
            health = max(0, 1 - current_age_hours / expected_life)
        
        urgency = self._compute_urgency(mean_rul, health)
        recommendation = self._get_recommendation(urgency, mean_rul)
        
        return RULEstimate(
            mean_rul_hours=mean_rul,
            std_rul_hours=std_rul,
            lower_bound_hours=lower,
            upper_bound_hours=upper,
            confidence_level=0.90,
            failure_probability_24h=min(1, max(0, prob_24h)),
            failure_probability_7d=min(1, max(0, prob_7d)),
            failure_probability_30d=min(1, max(0, prob_30d)),
            health_index=min(1, max(0, health)),
            maintenance_urgency=urgency,
            recommended_action=recommendation,
            model_used="Weibull",
            features_used=["current_age_hours"],
        )


# ============================================================
# BAYESIAN RUL ESTIMATOR
# ============================================================

class BayesianRULEstimator(RULEstimator):
    """
    Bayesian RUL estimation with uncertainty quantification.
    
    Advantages:
    - Principled uncertainty estimates
    - Incorporates prior knowledge
    - Updates with new observations
    
    TODO[R&D]: Bayesian RUL research:
    - Compare parametric vs non-parametric priors
    - Online updating with sensor streams
    - Multi-output models for multiple failure modes
    
    Reference: Gebraeel et al. (2005). Residual-life distributions
    """
    
    def __init__(self, config: Optional[RULConfig] = None):
        super().__init__(config)
        self._prior_samples = None
        self._posterior_samples: Dict[str, np.ndarray] = {}
    
    def fit(self, historical_data: pd.DataFrame) -> None:
        """
        Fit Bayesian model using historical data to update prior.
        
        Uses conjugate priors where possible for efficiency.
        """
        # Generate prior samples
        np.random.seed(42)
        self._prior_samples = np.random.normal(
            self.config.prior_mean_hours,
            self.config.prior_std_hours,
            self.config.n_samples
        )
        self._prior_samples = np.maximum(self._prior_samples, 0)  # RUL >= 0
        
        if historical_data.empty:
            self._is_fitted = True
            return
        
        # Update posterior for each machine with data
        for machine_id in historical_data['machine_id'].unique():
            machine_data = historical_data[historical_data['machine_id'] == machine_id]
            
            if len(machine_data) >= 3:
                # Bayesian update (conjugate Normal-Normal)
                observed_ttf = machine_data['time_to_failure'].values
                n = len(observed_ttf)
                sample_mean = observed_ttf.mean()
                sample_var = observed_ttf.var() if n > 1 else self.config.prior_std_hours ** 2
                
                prior_mean = self.config.prior_mean_hours
                prior_var = self.config.prior_std_hours ** 2
                
                # Posterior parameters
                posterior_var = 1 / (1/prior_var + n/sample_var)
                posterior_mean = posterior_var * (prior_mean/prior_var + n*sample_mean/sample_var)
                
                # Generate posterior samples
                self._posterior_samples[machine_id] = np.random.normal(
                    posterior_mean,
                    np.sqrt(posterior_var),
                    self.config.n_samples
                )
                self._posterior_samples[machine_id] = np.maximum(
                    self._posterior_samples[machine_id], 0
                )
        
        self._is_fitted = True
        logger.info(f"Bayesian model fitted for {len(self._posterior_samples)} machines")
    
    def estimate(
        self,
        machine_id: str,
        current_age_hours: float,
        sensor_features: Optional[Dict[str, float]] = None
    ) -> RULEstimate:
        """Estimate RUL using Bayesian posterior."""
        
        # Get samples (posterior if available, prior otherwise)
        if machine_id in self._posterior_samples:
            ttf_samples = self._posterior_samples[machine_id]
        else:
            ttf_samples = self._prior_samples if self._prior_samples is not None else \
                          np.random.normal(self.config.prior_mean_hours, 
                                         self.config.prior_std_hours, 
                                         self.config.n_samples)
        
        # Conditional RUL samples given survival to current age
        # RUL = TTF - current_age | TTF > current_age
        rul_samples = ttf_samples - current_age_hours
        rul_samples = rul_samples[rul_samples > 0]  # Condition on survival
        
        if len(rul_samples) == 0:
            # All samples indicate failure already occurred
            return RULEstimate(
                mean_rul_hours=0.0,
                std_rul_hours=0.0,
                lower_bound_hours=0.0,
                upper_bound_hours=0.0,
                health_index=0.0,
                maintenance_urgency="critical",
                recommended_action="URGENTE: Máquina além da vida útil esperada",
                model_used="Bayesian",
            )
        
        # Posterior statistics
        mean_rul = np.mean(rul_samples)
        std_rul = np.std(rul_samples)
        lower = np.percentile(rul_samples, 5)
        upper = np.percentile(rul_samples, 95)
        
        # Failure probabilities from samples
        prob_24h = np.mean(rul_samples < 24)
        prob_7d = np.mean(rul_samples < 168)
        prob_30d = np.mean(rul_samples < 720)
        
        # Health index
        health = len(rul_samples) / len(ttf_samples)  # Survival probability
        
        # Adjust for sensor features if provided
        if sensor_features:
            # TODO[R&D]: Integrate sensor features into Bayesian model
            # For now, simple adjustment based on anomaly score
            anomaly_score = sensor_features.get('anomaly_score', 0)
            if anomaly_score > 0.5:
                mean_rul *= (1 - anomaly_score)
                health *= (1 - anomaly_score)
        
        urgency = self._compute_urgency(mean_rul, health)
        recommendation = self._get_recommendation(urgency, mean_rul)
        
        return RULEstimate(
            mean_rul_hours=mean_rul,
            std_rul_hours=std_rul,
            lower_bound_hours=lower,
            upper_bound_hours=upper,
            confidence_level=0.90,
            failure_probability_24h=prob_24h,
            failure_probability_7d=prob_7d,
            failure_probability_30d=prob_30d,
            health_index=health,
            maintenance_urgency=urgency,
            recommended_action=recommendation,
            model_used="Bayesian",
            features_used=["current_age_hours"] + (list(sensor_features.keys()) if sensor_features else []),
        )


# ============================================================
# ML-BASED RUL ESTIMATOR (Stub)
# ============================================================

class MLRULEstimator(RULEstimator):
    """
    Machine learning based RUL estimation.
    
    Uses survival analysis or regression approaches:
    - Random Survival Forests
    - Gradient Boosting Survival Analysis
    - Deep Survival models (future)
    
    TODO[R&D]: ML RUL research:
    - Compare survival vs regression formulations
    - Feature engineering for sensor data
    - Handling censored data properly
    
    Reference: Ishwaran et al. (2008). Random Survival Forests
    """
    
    def __init__(self, config: Optional[RULConfig] = None):
        super().__init__(config)
        self._model = None
        self._fallback = StatisticalRULEstimator(config)
        logger.warning("MLRULEstimator is experimental - using statistical fallback")
    
    def fit(self, historical_data: pd.DataFrame) -> None:
        """Fit ML model (stub - uses fallback)."""
        # TODO: Implement proper ML-based survival analysis
        # Options:
        # - scikit-survival: Random Survival Forests, Gradient Boosting
        # - lifelines: Cox regression, Weibull AFT
        # - pycox: Deep survival models
        
        self._fallback.fit(historical_data)
        self._is_fitted = True
    
    def estimate(
        self,
        machine_id: str,
        current_age_hours: float,
        sensor_features: Optional[Dict[str, float]] = None
    ) -> RULEstimate:
        """Estimate RUL (uses fallback)."""
        result = self._fallback.estimate(machine_id, current_age_hours, sensor_features)
        result.model_used = "ML (fallback to Statistical)"
        return result


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def estimate_rul(
    machine_id: str,
    current_age_hours: float,
    sensor_features: Optional[Dict[str, float]] = None,
    model: RULModel = RULModel.STATISTICAL
) -> RULEstimate:
    """
    Convenience function for RUL estimation.
    
    Args:
        machine_id: Machine identifier
        current_age_hours: Hours since last maintenance
        sensor_features: Optional sensor readings
        model: Model type to use
    
    Returns:
        RULEstimate
    """
    config = RULConfig(model=model)
    
    if model == RULModel.STATISTICAL:
        estimator = StatisticalRULEstimator(config)
    elif model == RULModel.BAYESIAN:
        estimator = BayesianRULEstimator(config)
    else:
        estimator = StatisticalRULEstimator(config)
    
    # Fit with empty data (uses defaults)
    estimator.fit(pd.DataFrame())
    
    return estimator.estimate(machine_id, current_age_hours, sensor_features)


def create_rul_estimator(
    model: RULModel = RULModel.BAYESIAN,
    config: Optional[RULConfig] = None
) -> RULEstimator:
    """Factory function for RUL estimators."""
    if model == RULModel.STATISTICAL:
        return StatisticalRULEstimator(config)
    elif model == RULModel.BAYESIAN:
        return BayesianRULEstimator(config)
    elif model == RULModel.ML:
        return MLRULEstimator(config)
    else:
        return BayesianRULEstimator(config)


def get_maintenance_schedule(
    machines: List[str],
    ages: Dict[str, float],
    estimator: Optional[RULEstimator] = None
) -> pd.DataFrame:
    """
    Generate maintenance schedule based on RUL estimates.
    
    Args:
        machines: List of machine IDs
        ages: Dict machine_id -> current age in hours
        estimator: RUL estimator to use
    
    Returns:
        DataFrame with maintenance recommendations sorted by urgency
    """
    if estimator is None:
        estimator = BayesianRULEstimator()
        estimator.fit(pd.DataFrame())
    
    records = []
    for machine_id in machines:
        age = ages.get(machine_id, 0)
        estimate = estimator.estimate(machine_id, age)
        
        records.append({
            'machine_id': machine_id,
            'current_age_hours': age,
            'rul_mean_hours': estimate.mean_rul_hours,
            'rul_lower_hours': estimate.lower_bound_hours,
            'rul_upper_hours': estimate.upper_bound_hours,
            'health_index': estimate.health_index,
            'urgency': estimate.maintenance_urgency,
            'failure_prob_7d': estimate.failure_probability_7d,
            'recommendation': estimate.recommended_action,
        })
    
    df = pd.DataFrame(records)
    
    # Sort by urgency
    urgency_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
    df['urgency_rank'] = df['urgency'].map(urgency_order)
    df = df.sort_values('urgency_rank').drop('urgency_rank', axis=1)
    
    return df
