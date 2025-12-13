"""
════════════════════════════════════════════════════════════════════════════════════════════════════
CAUSAL EFFECT ESTIMATOR - Estimação de Efeitos Causais
════════════════════════════════════════════════════════════════════════════════════════════════════

Estima efeitos causais de intervenções usando métodos de inferência causal.

Contract 4 Implementation:
- CausalEstimatorBase: Abstract base class
- OlsCausalEstimator: Simple OLS regression adjustment (BASE)
- DmlCausalEstimator: Double Machine Learning (ADVANCED)
- get_causal_estimator(): Factory function with FeatureFlags

Feature Flags Integration:
- CausalEngine.OLS → OlsCausalEstimator
- CausalEngine.DOWHY → DmlCausalEstimator (with fallback)

Métodos suportados:
- Regression Adjustment (OLS)
- Double Machine Learning (DML) via EconML
- Propensity Score Matching (future)
- DoWhy CATE estimation (future)

Output:
- CausalEstimate with ATE/ATT/CATE
- Confidence intervals
- Natural language explanation
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class EffectType(str, Enum):
    """Tipos de efeito causal."""
    ATE = "ate"           # Average Treatment Effect
    ATT = "att"           # Average Treatment Effect on Treated
    CATE = "cate"         # Conditional Average Treatment Effect
    MARGINAL = "marginal" # Marginal effect (per unit change)


class EstimationMethod(str, Enum):
    """Método de estimação."""
    OLS = "ols"                 # Simple OLS regression
    DML = "dml"                 # Double Machine Learning
    PSM = "psm"                 # Propensity Score Matching
    IPW = "ipw"                 # Inverse Probability Weighting
    GRAPH = "graph"             # Graph-based (structural equations)


@dataclass
class CausalEstimate:
    """
    Resultado padronizado de estimação de efeito causal.
    
    Seguindo Contract 4: inclui treatment, outcome, ate, ci_lower, ci_upper, p_value, method
    """
    treatment: str
    outcome: str
    ate: float  # Average Treatment Effect
    ci_lower: float
    ci_upper: float
    p_value: float
    method: EstimationMethod
    
    # Additional fields
    std_error: float = 0.0
    n_observations: int = 0
    confounders_adjusted: List[str] = field(default_factory=list)
    effect_type: EffectType = EffectType.ATE
    confidence: float = 0.5  # Overall confidence in estimate
    is_advanced: bool = False  # True if using advanced method (DML)
    
    # Interpretation
    direction: str = ""
    magnitude: str = ""
    significance: str = ""
    explanation: str = ""
    
    def compute_interpretation(self) -> None:
        """Compute automatic interpretation."""
        # Direction
        if self.ate > 0:
            self.direction = "positive"
        elif self.ate < 0:
            self.direction = "negative"
        else:
            self.direction = "neutral"
        
        # Magnitude (in terms of std errors)
        if self.std_error > 0:
            z = abs(self.ate) / self.std_error
            if z > 2.5:
                self.magnitude = "large"
            elif z > 1.5:
                self.magnitude = "medium"
            else:
                self.magnitude = "small"
        else:
            self.magnitude = "unknown"
        
        # Significance
        if self.p_value < 0.01:
            self.significance = "significant"
        elif self.p_value < 0.05:
            self.significance = "marginal"
        else:
            self.significance = "not_significant"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "treatment": self.treatment,
            "outcome": self.outcome,
            "ate": round(self.ate, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "p_value": round(self.p_value, 4),
            "method": self.method.value,
            "std_error": round(self.std_error, 4),
            "n_observations": self.n_observations,
            "confounders_adjusted": self.confounders_adjusted,
            "effect_type": self.effect_type.value,
            "confidence": round(self.confidence, 3),
            "is_advanced": self.is_advanced,
            "direction": self.direction,
            "magnitude": self.magnitude,
            "significance": self.significance,
            "explanation": self.explanation,
        }


# Legacy alias for backward compatibility
@dataclass
class CausalEffect:
    """Legacy class - use CausalEstimate instead."""
    treatment: str
    outcome: str
    effect_type: EffectType
    estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float
    method_used: str
    n_observations: int
    confounders_adjusted: List[str]
    direction: str = ""
    magnitude: str = ""
    significance: str = ""
    explanation: str = ""
    
    def compute_interpretation(self):
        if self.estimate > 0:
            self.direction = "positive"
        elif self.estimate < 0:
            self.direction = "negative"
        else:
            self.direction = "neutral"
        
        if abs(self.estimate) > 2 * self.std_error:
            self.magnitude = "large"
        elif abs(self.estimate) > self.std_error:
            self.magnitude = "medium"
        else:
            self.magnitude = "small"
        
        if self.p_value < 0.01:
            self.significance = "significant"
        elif self.p_value < 0.05:
            self.significance = "marginal"
        else:
            self.significance = "not_significant"
    
    def to_dict(self) -> Dict:
        return {
            "treatment": self.treatment,
            "outcome": self.outcome,
            "effect_type": self.effect_type.value,
            "estimate": round(self.estimate, 4),
            "std_error": round(self.std_error, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "p_value": round(self.p_value, 4),
            "method_used": self.method_used,
            "n_observations": self.n_observations,
            "confounders_adjusted": self.confounders_adjusted,
            "direction": self.direction,
            "magnitude": self.magnitude,
            "significance": self.significance,
            "explanation": self.explanation,
        }
    
    def to_causal_estimate(self) -> CausalEstimate:
        """Convert to new CausalEstimate format."""
        est = CausalEstimate(
            treatment=self.treatment,
            outcome=self.outcome,
            ate=self.estimate,
            ci_lower=self.ci_lower,
            ci_upper=self.ci_upper,
            p_value=self.p_value,
            method=EstimationMethod.OLS,
            std_error=self.std_error,
            n_observations=self.n_observations,
            confounders_adjusted=self.confounders_adjusted,
            effect_type=self.effect_type,
            direction=self.direction,
            magnitude=self.magnitude,
            significance=self.significance,
            explanation=self.explanation,
        )
        return est


# ═══════════════════════════════════════════════════════════════════════════════
# ABSTRACT BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class CausalEstimatorBase(ABC):
    """
    Abstract base class for causal effect estimators.
    
    All implementations must follow this interface.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        self.data = data
    
    @abstractmethod
    def estimate_ate(
        self,
        treatment: str,
        outcome: str,
        controls: Optional[List[str]] = None,
    ) -> CausalEstimate:
        """
        Estimate Average Treatment Effect.
        
        Args:
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            controls: List of control/confounder variables
        
        Returns:
            CausalEstimate with ATE and confidence intervals
        """
        pass
    
    def set_data(self, data: pd.DataFrame) -> None:
        """Set data for estimation."""
        self.data = data
    
    def _validate_columns(
        self,
        treatment: str,
        outcome: str,
        controls: List[str],
    ) -> Tuple[bool, str]:
        """Validate that required columns exist."""
        if self.data is None:
            return False, "No data provided"
        
        if treatment not in self.data.columns:
            return False, f"Treatment '{treatment}' not in data"
        
        if outcome not in self.data.columns:
            return False, f"Outcome '{outcome}' not in data"
        
        missing_controls = [c for c in controls if c not in self.data.columns]
        if missing_controls:
            logger.warning(f"Controls not in data: {missing_controls}")
        
        return True, ""


# ═══════════════════════════════════════════════════════════════════════════════
# OLS CAUSAL ESTIMATOR (BASE)
# ═══════════════════════════════════════════════════════════════════════════════

class OlsCausalEstimator(CausalEstimatorBase):
    """
    Simple OLS regression adjustment for causal effect estimation.
    
    This is the BASE implementation for production stability.
    
    Method:
    Y = β₀ + β₁*T + β₂*Z₁ + ... + βₙ*Zₙ + ε
    
    Where:
    - Y = outcome
    - T = treatment
    - Z = controls/confounders
    - β₁ = ATE estimate
    """
    
    def estimate_ate(
        self,
        treatment: str,
        outcome: str,
        controls: Optional[List[str]] = None,
    ) -> CausalEstimate:
        """Estimate ATE using OLS regression adjustment."""
        controls = controls or []
        
        # Validate
        is_valid, error = self._validate_columns(treatment, outcome, controls)
        if not is_valid:
            logger.warning(f"Validation failed: {error}")
            return self._default_estimate(treatment, outcome, controls, error)
        
        # Filter available controls
        available_controls = [c for c in controls if c in self.data.columns]
        
        # Prepare data
        cols = [treatment, outcome] + available_controls
        df = self.data[cols].dropna()
        
        if len(df) < 30:
            return self._default_estimate(treatment, outcome, controls, "Insufficient data")
        
        # Build design matrix
        X = df[[treatment] + available_controls].values
        y = df[outcome].values
        
        # Add intercept
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        try:
            # OLS estimation: β = (X'X)^(-1) X'y
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            
            # Treatment effect is β₁
            ate = float(beta[1])
            
            # Calculate residuals and standard error
            y_pred = X_with_const @ beta
            residuals = y - y_pred
            n = len(y)
            k = X_with_const.shape[1]
            mse = np.sum(residuals**2) / (n - k)
            
            # Variance of coefficients
            try:
                var_beta = mse * np.linalg.inv(X_with_const.T @ X_with_const)
                std_error = float(np.sqrt(var_beta[1, 1]))
            except np.linalg.LinAlgError:
                std_error = abs(ate) * 0.2 + 0.1
            
            # 95% CI
            t_crit = stats.t.ppf(0.975, max(n - k, 1))
            ci_lower = ate - t_crit * std_error
            ci_upper = ate + t_crit * std_error
            
            # P-value
            if std_error > 0:
                t_stat = ate / std_error
                p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), max(n - k, 1))))
            else:
                p_value = 0.5
            
        except Exception as e:
            logger.warning(f"OLS estimation failed: {e}")
            return self._default_estimate(treatment, outcome, controls, str(e))
        
        estimate = CausalEstimate(
            treatment=treatment,
            outcome=outcome,
            ate=ate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            method=EstimationMethod.OLS,
            std_error=std_error,
            n_observations=n,
            confounders_adjusted=available_controls,
            effect_type=EffectType.ATE,
            confidence=0.6,
            is_advanced=False,
        )
        
        estimate.compute_interpretation()
        estimate.explanation = self._generate_explanation(estimate)
        
        return estimate
    
    def _default_estimate(
        self,
        treatment: str,
        outcome: str,
        controls: List[str],
        reason: str,
    ) -> CausalEstimate:
        """Return default estimate when estimation fails."""
        return CausalEstimate(
            treatment=treatment,
            outcome=outcome,
            ate=0.0,
            ci_lower=-0.5,
            ci_upper=0.5,
            p_value=1.0,
            method=EstimationMethod.OLS,
            std_error=0.5,
            n_observations=0,
            confounders_adjusted=controls,
            effect_type=EffectType.ATE,
            confidence=0.1,
            is_advanced=False,
            significance="not_significant",
            explanation=f"Estimation failed: {reason}",
        )
    
    def _generate_explanation(self, estimate: CausalEstimate) -> str:
        """Generate natural language explanation."""
        direction_text = "aumenta" if estimate.direction == "positive" else "diminui"
        magnitude_text = {
            "large": "significativamente",
            "medium": "moderadamente",
            "small": "ligeiramente",
        }.get(estimate.magnitude, "")
        
        significance_text = {
            "significant": "com alta confiança estatística",
            "marginal": "com confiança moderada",
            "not_significant": "mas sem significância estatística clara",
        }.get(estimate.significance, "")
        
        effect_value = abs(estimate.ate)
        
        explanation = (
            f"Um aumento unitário em '{estimate.treatment}' {direction_text} {magnitude_text} "
            f"'{estimate.outcome}' em {effect_value:.2f} unidades, {significance_text}. "
            f"(IC 95%: [{estimate.ci_lower:.2f}, {estimate.ci_upper:.2f}])"
        )
        
        return explanation


# ═══════════════════════════════════════════════════════════════════════════════
# DML CAUSAL ESTIMATOR (ADVANCED)
# ═══════════════════════════════════════════════════════════════════════════════

class DmlCausalEstimator(CausalEstimatorBase):
    """
    Double Machine Learning estimator for causal effects.
    
    This is the ADVANCED implementation for R&D.
    
    Uses EconML/DoWhy libraries if available, otherwise falls back to OLS.
    
    Method (Double ML):
    1. Predict Y from Z (outcome model): Ŷ = f(Z)
    2. Predict T from Z (treatment model): T̂ = g(Z)
    3. Residuals: ε_Y = Y - Ŷ, ε_T = T - T̂
    4. Final model: ε_Y = θ * ε_T + noise
    5. θ is the debiased ATE estimate
    
    Benefits:
    - Handles non-linear confounding
    - Asymptotically normal estimates
    - Better than naive OLS in presence of many controls
    
    TODO[R&D]:
    - Implement full EconML integration
    - Add CATE estimation
    - Support for heterogeneous effects
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        super().__init__(data)
        self._has_econml = self._check_econml()
        self._has_sklearn = self._check_sklearn()
        self._base_estimator = OlsCausalEstimator(data)
    
    def _check_econml(self) -> bool:
        """Check if EconML is available."""
        try:
            import econml
            return True
        except ImportError:
            return False
    
    def _check_sklearn(self) -> bool:
        """Check if sklearn is available."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            return True
        except ImportError:
            return False
    
    def set_data(self, data: pd.DataFrame) -> None:
        """Set data for both this and base estimator."""
        super().set_data(data)
        self._base_estimator.set_data(data)
    
    def estimate_ate(
        self,
        treatment: str,
        outcome: str,
        controls: Optional[List[str]] = None,
    ) -> CausalEstimate:
        """
        Estimate ATE using Double Machine Learning.
        
        Falls back to OLS if libraries unavailable or data insufficient.
        """
        controls = controls or []
        
        # Validate
        is_valid, error = self._validate_columns(treatment, outcome, controls)
        if not is_valid:
            logger.warning(f"DML validation failed: {error}, falling back to OLS")
            return self._base_estimator.estimate_ate(treatment, outcome, controls)
        
        # Need enough controls for DML to be useful
        available_controls = [c for c in controls if c in self.data.columns]
        if len(available_controls) < 2:
            logger.info("Too few controls for DML, using OLS")
            return self._base_estimator.estimate_ate(treatment, outcome, controls)
        
        # Try EconML first
        if self._has_econml:
            try:
                return self._estimate_econml(treatment, outcome, available_controls)
            except Exception as e:
                logger.warning(f"EconML estimation failed: {e}")
        
        # Try manual DML with sklearn
        if self._has_sklearn:
            try:
                return self._estimate_sklearn_dml(treatment, outcome, available_controls)
            except Exception as e:
                logger.warning(f"Sklearn DML failed: {e}")
        
        # Fallback to OLS
        logger.info("Using OLS fallback")
        return self._base_estimator.estimate_ate(treatment, outcome, controls)
    
    def _estimate_econml(
        self,
        treatment: str,
        outcome: str,
        controls: List[str],
    ) -> CausalEstimate:
        """Estimate using EconML LinearDML."""
        from econml.dml import LinearDML
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        
        # Prepare data
        df = self.data[[treatment, outcome] + controls].dropna()
        
        Y = df[outcome].values
        T = df[treatment].values.reshape(-1, 1)
        X = df[controls].values
        
        # Fit DML
        model = LinearDML(
            model_y=GradientBoostingRegressor(n_estimators=50, max_depth=3),
            model_t=GradientBoostingRegressor(n_estimators=50, max_depth=3),
            discrete_treatment=False,
        )
        model.fit(Y, T, X=X)
        
        # Get effect
        ate = float(model.const_marginal_effect(X).mean())
        
        # Confidence intervals
        effects = model.const_marginal_effect(X)
        std_error = float(np.std(effects) / np.sqrt(len(effects)))
        ci_lower = ate - 1.96 * std_error
        ci_upper = ate + 1.96 * std_error
        
        # P-value
        if std_error > 0:
            z_stat = ate / std_error
            p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
        else:
            p_value = 0.5
        
        estimate = CausalEstimate(
            treatment=treatment,
            outcome=outcome,
            ate=ate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            method=EstimationMethod.DML,
            std_error=std_error,
            n_observations=len(df),
            confounders_adjusted=controls,
            effect_type=EffectType.ATE,
            confidence=0.75,
            is_advanced=True,
        )
        
        estimate.compute_interpretation()
        estimate.explanation = self._generate_explanation(estimate)
        
        return estimate
    
    def _estimate_sklearn_dml(
        self,
        treatment: str,
        outcome: str,
        controls: List[str],
    ) -> CausalEstimate:
        """Manual DML implementation using sklearn."""
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_predict
        
        # Prepare data
        df = self.data[[treatment, outcome] + controls].dropna()
        n = len(df)
        
        if n < 100:
            raise ValueError("Insufficient data for DML")
        
        Y = df[outcome].values
        T = df[treatment].values
        Z = df[controls].values
        
        # Step 1: Predict Y from Z (cross-fitting)
        model_y = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        Y_hat = cross_val_predict(model_y, Z, Y, cv=5)
        
        # Step 2: Predict T from Z (cross-fitting)
        model_t = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        T_hat = cross_val_predict(model_t, Z, T, cv=5)
        
        # Step 3: Residuals
        Y_res = Y - Y_hat
        T_res = T - T_hat
        
        # Step 4: OLS on residuals
        # θ = (T_res' T_res)^(-1) T_res' Y_res
        denom = np.sum(T_res ** 2)
        if denom < 1e-10:
            raise ValueError("Treatment has no variation after controlling")
        
        ate = float(np.sum(T_res * Y_res) / denom)
        
        # Standard error (heteroskedasticity-robust)
        eps = Y_res - ate * T_res
        var_ate = np.sum(eps**2 * T_res**2) / (denom**2)
        std_error = float(np.sqrt(var_ate))
        
        # CI and p-value
        ci_lower = ate - 1.96 * std_error
        ci_upper = ate + 1.96 * std_error
        
        if std_error > 0:
            z_stat = ate / std_error
            p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
        else:
            p_value = 0.5
        
        estimate = CausalEstimate(
            treatment=treatment,
            outcome=outcome,
            ate=ate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            method=EstimationMethod.DML,
            std_error=std_error,
            n_observations=n,
            confounders_adjusted=controls,
            effect_type=EffectType.ATE,
            confidence=0.7,
            is_advanced=True,
        )
        
        estimate.compute_interpretation()
        estimate.explanation = self._generate_explanation(estimate)
        
        return estimate
    
    def _generate_explanation(self, estimate: CausalEstimate) -> str:
        """Generate explanation with DML context."""
        direction_text = "aumenta" if estimate.direction == "positive" else "diminui"
        effect_value = abs(estimate.ate)
        
        explanation = (
            f"Usando Double Machine Learning (DML), estimamos que "
            f"'{estimate.treatment}' {direction_text} '{estimate.outcome}' "
            f"em {effect_value:.2f} unidades (IC 95%: [{estimate.ci_lower:.2f}, {estimate.ci_upper:.2f}]). "
            f"Esta estimativa controla para {len(estimate.confounders_adjusted)} confounders usando ML."
        )
        
        return explanation


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_causal_estimator(
    method: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    use_advanced: Optional[bool] = None,
) -> CausalEstimatorBase:
    """
    Factory function to get causal estimator based on FeatureFlags.
    
    Args:
        method: "ols" or "dml" (overrides FeatureFlags if provided)
        data: Data for estimation
        use_advanced: Force advanced mode (if None, uses FeatureFlags)
    
    Returns:
        CausalEstimatorBase (OlsCausalEstimator or DmlCausalEstimator)
    """
    # Import FeatureFlags
    try:
        from ..feature_flags import FeatureFlags, CausalEngine as CE
        
        if use_advanced is None and method is None:
            use_advanced = FeatureFlags.get_causal_engine() == CE.DOWHY
    except ImportError:
        if use_advanced is None:
            use_advanced = False
    
    # Method override
    if method is not None:
        use_advanced = method.lower() == "dml"
    
    if use_advanced:
        try:
            logger.info("Using DmlCausalEstimator (ADVANCED)")
            return DmlCausalEstimator(data)
        except Exception as e:
            logger.warning(f"Failed to create DML estimator: {e}. Using OLS.")
            return OlsCausalEstimator(data)
    else:
        logger.info("Using OlsCausalEstimator (BASE)")
        return OlsCausalEstimator(data)


# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY - CausalEffectEstimator class
# ═══════════════════════════════════════════════════════════════════════════════

class CausalEffectEstimator:
    """
    Legacy estimator class - delegates to new architecture.
    
    Use get_causal_estimator() for new code.
    """
    
    def __init__(self, causal_graph: Any, data: Optional[pd.DataFrame] = None):
        from .causal_graph_builder import generate_synthetic_data
        
        self.graph = causal_graph
        self.data = data if data is not None else generate_synthetic_data()
        self._estimator = get_causal_estimator(data=self.data)
    
    def identify_confounders(self, treatment: str, outcome: str) -> List[str]:
        """Identify confounders from graph."""
        from .causal_graph_builder import VariableType
        
        confounders = []
        treatment_ancestors = self.graph.get_ancestors(treatment)
        outcome_parents = set(self.graph.get_parents(outcome)) - {treatment}
        
        for var_name, var in self.graph.variables.items():
            if var.var_type == VariableType.CONFOUNDER:
                affects_treatment = var_name in treatment_ancestors or var_name in self.graph.get_parents(treatment)
                affects_outcome = var_name in outcome_parents or var_name in self.graph.get_ancestors(outcome)
                
                if affects_treatment or affects_outcome:
                    confounders.append(var_name)
        
        return confounders
    
    def estimate_effect_regression(
        self,
        treatment: str,
        outcome: str,
        confounders: Optional[List[str]] = None,
    ) -> CausalEffect:
        """Estimate effect using regression (legacy interface)."""
        if confounders is None:
            confounders = self.identify_confounders(treatment, outcome)
        
        # Use new estimator
        estimate = self._estimator.estimate_ate(treatment, outcome, confounders)
        
        # Convert to legacy format
        effect = CausalEffect(
            treatment=estimate.treatment,
            outcome=estimate.outcome,
            effect_type=estimate.effect_type,
            estimate=estimate.ate,
            std_error=estimate.std_error,
            ci_lower=estimate.ci_lower,
            ci_upper=estimate.ci_upper,
            p_value=estimate.p_value,
            method_used=estimate.method.value,
            n_observations=estimate.n_observations,
            confounders_adjusted=estimate.confounders_adjusted,
            direction=estimate.direction,
            magnitude=estimate.magnitude,
            significance=estimate.significance,
            explanation=estimate.explanation,
        )
        
        return effect


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_effect(
    treatment: str,
    outcome: str,
    causal_graph: Optional[Any] = None,
    data: Optional[pd.DataFrame] = None,
) -> CausalEffect:
    """
    Main function to estimate a causal effect (legacy interface).
    """
    from .causal_graph_builder import learn_causal_graph
    
    if causal_graph is None:
        causal_graph = learn_causal_graph(data)
    
    estimator = CausalEffectEstimator(causal_graph, data)
    return estimator.estimate_effect_regression(treatment, outcome)


def estimate_intervention(
    intervention: str,
    outcome: str,
    intervention_value: Optional[float] = None,
    causal_graph: Optional[Any] = None,
    data: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Estimate effect of a specific intervention."""
    from .causal_graph_builder import learn_causal_graph
    
    if causal_graph is None:
        causal_graph = learn_causal_graph(data)
    
    treatment = _parse_intervention(intervention, causal_graph)
    
    if treatment is None:
        return {
            "success": False,
            "error": f"Could not identify treatment variable for '{intervention}'",
        }
    
    effect = estimate_effect(treatment, outcome, causal_graph, data)
    
    if intervention_value is not None:
        absolute_impact = effect.estimate * intervention_value
    else:
        absolute_impact = effect.estimate
    
    return {
        "success": True,
        "treatment": treatment,
        "outcome": outcome,
        "effect": effect.to_dict(),
        "absolute_impact": round(absolute_impact, 4),
        "interpretation": effect.explanation,
    }


def _parse_intervention(intervention: str, graph: Any) -> Optional[str]:
    """Map textual intervention description to treatment variable."""
    intervention_lower = intervention.lower()
    
    keyword_mapping = {
        "setup": "setup_frequency",
        "changeover": "setup_frequency",
        "batch": "batch_size",
        "lote": "batch_size",
        "carga": "machine_load",
        "load": "machine_load",
        "utilização": "machine_load",
        "noturno": "night_shifts",
        "night": "night_shifts",
        "turno": "night_shifts",
        "hora extra": "overtime_hours",
        "overtime": "overtime_hours",
        "manutenção": "maintenance_delay",
        "maintenance": "maintenance_delay",
        "prioridade": "priority_changes",
        "priority": "priority_changes",
    }
    
    for keyword, var_name in keyword_mapping.items():
        if keyword in intervention_lower:
            if var_name in graph.variables:
                return var_name
    
    for var_name in graph.get_treatments():
        if var_name in intervention_lower or var_name.replace("_", " ") in intervention_lower:
            return var_name
    
    return None


def get_all_effects_for_outcome(
    outcome: str,
    causal_graph: Optional[Any] = None,
    data: Optional[pd.DataFrame] = None,
) -> List[CausalEffect]:
    """Estimate effects of all treatments on a specific outcome."""
    from .causal_graph_builder import learn_causal_graph
    
    if causal_graph is None:
        causal_graph = learn_causal_graph(data)
    
    estimator = CausalEffectEstimator(causal_graph, data)
    effects = []
    
    for treatment in causal_graph.get_treatments():
        try:
            effect = estimator.estimate_effect_regression(treatment, outcome)
            effects.append(effect)
        except Exception:
            continue
    
    effects.sort(key=lambda e: -abs(e.estimate))
    return effects


def get_all_effects_from_treatment(
    treatment: str,
    causal_graph: Optional[Any] = None,
    data: Optional[pd.DataFrame] = None,
) -> List[CausalEffect]:
    """Estimate effects of a treatment on all outcomes."""
    from .causal_graph_builder import learn_causal_graph
    
    if causal_graph is None:
        causal_graph = learn_causal_graph(data)
    
    estimator = CausalEffectEstimator(causal_graph, data)
    effects = []
    
    for outcome in causal_graph.get_outcomes():
        try:
            effect = estimator.estimate_effect_regression(treatment, outcome)
            effects.append(effect)
        except Exception:
            continue
    
    effects.sort(key=lambda e: -abs(e.estimate))
    return effects
