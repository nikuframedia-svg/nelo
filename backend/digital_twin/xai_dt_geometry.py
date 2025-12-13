"""
════════════════════════════════════════════════════════════════════════════════════════════════════
XAI DIGITAL TWIN GEOMETRY - Explainable AI for Digital Twin
════════════════════════════════════════════════════════════════════════════════════════════════════

Explainable AI for Digital Twin with geometric/topological methods.

Contract 4 Implementation:
- PodDeviationEngine: PCA/POD for dimensionality reduction and deviation analysis
- DeviationSurrogateModel: Fast approximation of deviation behavior  
- explain_deviation(): XAI explanations with probable causes

Feature Flags Integration:
- DeviationEngine.BASE → SimpleDeviationEngine (max/mean/rms)
- DeviationEngine.POD → PodDeviationEngine (advanced)

Features:
- PCA/POD (Proper Orthogonal Decomposition) for dimension reduction
- Surrogate models for fast simulation
- Sensitivity analysis for parameter importance
- Anomaly detection via reconstruction error
- XAI explanations for deviations

R&D / SIFIDE: WP1 - Digital Twin & Explainability

TODO[R&D]:
- Implement full POD decomposition
- Add UMAP/t-SNE for visualization  
- Integrate with real sensor data
- Physics-informed surrogate models
- Uncertainty quantification
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class ReductionMethod(str, Enum):
    """Dimensionality reduction method."""
    PCA = "pca"
    POD = "pod"  # Proper Orthogonal Decomposition
    AUTOENCODER = "autoencoder"  # Neural network based


@dataclass
class PODConfig:
    """Configuration for POD analysis."""
    n_components: int = 10
    energy_threshold: float = 0.95  # Keep modes capturing this much variance
    center_data: bool = True
    scale_data: bool = False


@dataclass
class XAIConfig:
    """Configuration for XAI analysis."""
    max_causes: int = 5  # Maximum number of probable causes
    significance_threshold: float = 0.05  # Minimum impact to report
    include_recommendations: bool = True
    language: str = "pt"  # pt or en


@dataclass
class DeviationField:
    """
    Represents a deviation field (e.g., temperature/pressure distribution).
    
    Used for comparing actual vs expected behavior in Digital Twin.
    """
    points: np.ndarray  # Shape: (n_points, n_dims) - spatial coordinates
    deviations: np.ndarray  # Shape: (n_points,) - deviation values at each point
    field_name: str = "deviation"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_points(self) -> int:
        return len(self.deviations)
    
    @property
    def max_deviation(self) -> float:
        return float(np.max(np.abs(self.deviations)))
    
    @property
    def mean_deviation(self) -> float:
        return float(np.mean(np.abs(self.deviations)))
    
    @property
    def rms_deviation(self) -> float:
        return float(np.sqrt(np.mean(self.deviations ** 2)))
    
    def to_flat(self) -> np.ndarray:
        """Flatten to 1D array for POD."""
        return self.deviations.flatten()


@dataclass
class ProvableCause:
    """A probable cause for a deviation with its impact."""
    parameter_name: str
    impact: float  # Importance score (0-1)
    direction: str  # "increase", "decrease", "both"
    suggested_action: str
    confidence: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter": self.parameter_name,
            "impact": round(self.impact, 3),
            "direction": self.direction,
            "action": self.suggested_action,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class XaiDtExplanation:
    """
    XAI explanation for Digital Twin deviations.
    
    This is the main output structure for explainability.
    """
    timestamp: datetime
    machine_id: str
    scalar_error_score: float  # Overall deviation score (0-100)
    probable_causes: List[ProvableCause]
    
    # Detailed analysis
    dominant_modes: List[int] = field(default_factory=list)  # Most important POD modes
    reconstruction_error: float = 0.0
    confidence: float = 0.5
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "machine_id": self.machine_id,
            "scalar_error_score": round(self.scalar_error_score, 2),
            "probable_causes": [c.to_dict() for c in self.probable_causes],
            "dominant_modes": self.dominant_modes,
            "reconstruction_error": round(self.reconstruction_error, 4),
            "confidence": round(self.confidence, 3),
            "recommendations": self.recommendations,
        }
    
    def summary(self, lang: str = "pt") -> str:
        """Generate human-readable summary."""
        if lang == "pt":
            if self.scalar_error_score < 20:
                status = "Normal"
            elif self.scalar_error_score < 50:
                status = "Desvio Moderado"
            elif self.scalar_error_score < 80:
                status = "Desvio Significativo"
            else:
                status = "Desvio Crítico"
            
            causes_text = ", ".join([c.parameter_name for c in self.probable_causes[:3]])
            return f"{status} (score: {self.scalar_error_score:.1f}). Causas prováveis: {causes_text}"
        else:
            if self.scalar_error_score < 20:
                status = "Normal"
            elif self.scalar_error_score < 50:
                status = "Moderate Deviation"
            elif self.scalar_error_score < 80:
                status = "Significant Deviation"
            else:
                status = "Critical Deviation"
            
            causes_text = ", ".join([c.parameter_name for c in self.probable_causes[:3]])
            return f"{status} (score: {self.scalar_error_score:.1f}). Probable causes: {causes_text}"


@dataclass
class PODResult:
    """Result of POD analysis."""
    n_components: int
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    modes: np.ndarray  # Principal modes (eigenvectors)
    singular_values: np.ndarray
    mean_field: np.ndarray
    reconstruction_error: float


@dataclass 
class SensitivityResult:
    """Result of sensitivity analysis."""
    variable_names: List[str]
    sensitivities: Dict[str, float]  # Variable -> sensitivity index
    total_sensitivity: float
    interaction_indices: Optional[Dict[Tuple[str, str], float]] = None


@dataclass
class AnomalyScore:
    """Anomaly detection result."""
    timestamp: datetime
    machine_id: str
    reconstruction_error: float
    threshold: float
    is_anomaly: bool
    contributing_features: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class XAIResult:
    """Legacy result structure for compatibility."""
    factors: List[Tuple[str, float]]
    explanation: str
    confidence: float
    dominant_modes: List[int] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# ABSTRACT BASE CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class DeviationEngineBase(ABC):
    """
    Base class for deviation analysis engines.
    
    Implementations:
    - SimpleDeviationEngine (BASE): Simple statistical measures
    - PodDeviationEngine (ADVANCED): PCA/POD based analysis
    """
    
    @abstractmethod
    def fit(self, baseline_fields: List[DeviationField]) -> None:
        """Fit the engine on baseline (normal) data."""
        pass
    
    @abstractmethod
    def analyze(self, field: DeviationField) -> float:
        """Analyze a deviation field and return scalar error score."""
        pass
    
    @abstractmethod
    def explain(
        self,
        field: DeviationField,
        process_params: Optional[Dict[str, float]] = None,
    ) -> XaiDtExplanation:
        """Generate XAI explanation for deviation."""
        pass


class SurrogateModelBase(ABC):
    """
    Base class for surrogate models.
    
    Surrogate models provide fast approximations of expensive simulations.
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit surrogate model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def get_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Get prediction uncertainty."""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE DEVIATION ENGINE (BASE)
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleDeviationEngine(DeviationEngineBase):
    """
    Simple deviation analysis using basic statistical measures.
    
    This is the BASE implementation for production stability.
    Uses max/mean/RMS deviation metrics.
    """
    
    def __init__(self, config: Optional[XAIConfig] = None):
        self.config = config or XAIConfig()
        self._baseline_mean: float = 0.0
        self._baseline_std: float = 1.0
        self._is_fitted = False
    
    def fit(self, baseline_fields: List[DeviationField]) -> None:
        """Fit on baseline data to establish normal deviation levels."""
        if not baseline_fields:
            logger.warning("No baseline fields provided, using defaults")
            self._is_fitted = True
            return
        
        # Calculate baseline statistics
        all_deviations = []
        for field in baseline_fields:
            all_deviations.append(field.rms_deviation)
        
        self._baseline_mean = float(np.mean(all_deviations))
        self._baseline_std = float(np.std(all_deviations)) if len(all_deviations) > 1 else 1.0
        self._baseline_std = max(self._baseline_std, 0.001)  # Avoid zero
        
        self._is_fitted = True
        logger.info(f"SimpleDeviationEngine fitted: mean={self._baseline_mean:.4f}, std={self._baseline_std:.4f}")
    
    def analyze(self, field: DeviationField) -> float:
        """Return scalar error score (0-100) based on deviation magnitude."""
        if not self._is_fitted:
            logger.warning("Engine not fitted, returning raw RMS")
            return min(100, field.rms_deviation * 10)
        
        # Z-score based scoring
        z_score = (field.rms_deviation - self._baseline_mean) / self._baseline_std
        
        # Convert to 0-100 scale using sigmoid-like transformation
        score = 100 * (1 / (1 + np.exp(-z_score)))
        
        return float(np.clip(score, 0, 100))
    
    def explain(
        self,
        field: DeviationField,
        process_params: Optional[Dict[str, float]] = None,
    ) -> XaiDtExplanation:
        """Generate basic explanation based on deviation magnitude."""
        score = self.analyze(field)
        
        # Basic probable causes based on metadata
        causes = []
        
        # If process params provided, rank by correlation with deviation
        if process_params:
            param_impacts = []
            for param, value in process_params.items():
                # Simple heuristic: higher values = more impact
                impact = min(1.0, abs(value) / 100) if value != 0 else 0.1
                param_impacts.append((param, impact))
            
            param_impacts.sort(key=lambda x: x[1], reverse=True)
            
            for param, impact in param_impacts[:self.config.max_causes]:
                causes.append(ProvableCause(
                    parameter_name=param,
                    impact=impact,
                    direction="increase" if impact > 0.5 else "decrease",
                    suggested_action=f"Verificar parâmetro {param}",
                    confidence=0.4,
                ))
        else:
            # Default generic causes
            if score > 50:
                causes.append(ProvableCause(
                    parameter_name="temperatura",
                    impact=0.6,
                    direction="increase",
                    suggested_action="Verificar sistema de arrefecimento",
                    confidence=0.3,
                ))
            if score > 30:
                causes.append(ProvableCause(
                    parameter_name="desgaste",
                    impact=0.4,
                    direction="increase",
                    suggested_action="Inspecionar componentes",
                    confidence=0.3,
                ))
        
        recommendations = []
        if score > 80:
            recommendations.append("Manutenção urgente recomendada")
        elif score > 50:
            recommendations.append("Monitorização intensiva")
        elif score > 30:
            recommendations.append("Monitorização regular")
        
        return XaiDtExplanation(
            timestamp=field.timestamp,
            machine_id=field.metadata.get("machine_id", "unknown"),
            scalar_error_score=score,
            probable_causes=causes,
            dominant_modes=[],
            reconstruction_error=field.rms_deviation,
            confidence=0.4,  # Low confidence for simple method
            recommendations=recommendations,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# POD DEVIATION ENGINE (ADVANCED)
# ═══════════════════════════════════════════════════════════════════════════════

class PodDeviationEngine(DeviationEngineBase):
    """
    Advanced deviation analysis using Proper Orthogonal Decomposition (POD/PCA).
    
    This is the ADVANCED implementation for R&D.
    
    Features:
    - Extracts dominant modes of variation
    - Projects new data onto learned modes
    - Identifies which modes contribute to deviation
    - Provides richer explanations
    
    Mathematical basis:
    - SVD: X = U * S * V^T
    - Modes: V (eigenvectors)
    - Projection: coefficients = (X - mean) @ V
    - Reconstruction: X_recon = mean + coefficients @ V^T
    - Error: ||X - X_recon||
    """
    
    def __init__(
        self,
        config: Optional[PODConfig] = None,
        xai_config: Optional[XAIConfig] = None,
    ):
        self.pod_config = config or PODConfig()
        self.xai_config = xai_config or XAIConfig()
        
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._modes: Optional[np.ndarray] = None
        self._singular_values: Optional[np.ndarray] = None
        self._explained_variance: Optional[np.ndarray] = None
        self._baseline_error: float = 0.0
        self._error_threshold: float = 0.0
        self._is_fitted = False
    
    def fit(self, baseline_fields: List[DeviationField]) -> None:
        """
        Fit POD model on baseline deviation fields.
        
        Build data matrix from baseline fields and apply SVD.
        """
        if not baseline_fields:
            logger.warning("No baseline fields, cannot fit POD")
            return
        
        logger.info(f"Fitting PodDeviationEngine on {len(baseline_fields)} baseline fields")
        
        # Build data matrix: each row is a flattened deviation field
        data_matrix = np.array([f.to_flat() for f in baseline_fields])
        n_samples, n_features = data_matrix.shape
        
        # Center data
        if self.pod_config.center_data:
            self._mean = np.mean(data_matrix, axis=0)
            centered = data_matrix - self._mean
        else:
            self._mean = np.zeros(n_features)
            centered = data_matrix
        
        # Scale data
        if self.pod_config.scale_data:
            self._std = np.std(centered, axis=0)
            self._std[self._std == 0] = 1
            centered = centered / self._std
        else:
            self._std = np.ones(n_features)
        
        # SVD decomposition
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError as e:
            logger.error(f"SVD failed: {e}")
            self._is_fitted = False
            return
        
        # Calculate explained variance
        total_var = np.sum(S ** 2)
        self._explained_variance = (S ** 2) / total_var if total_var > 0 else np.zeros_like(S)
        cumulative_var = np.cumsum(self._explained_variance)
        
        # Determine number of components to keep
        if self.pod_config.energy_threshold < 1.0:
            n_comp = np.searchsorted(cumulative_var, self.pod_config.energy_threshold) + 1
            n_comp = min(n_comp, self.pod_config.n_components, len(S))
        else:
            n_comp = min(self.pod_config.n_components, len(S))
        
        # Store modes and singular values
        self._modes = Vt[:n_comp].T  # Shape: (n_features, n_components)
        self._singular_values = S[:n_comp]
        self._explained_variance = self._explained_variance[:n_comp]
        
        # Calculate baseline reconstruction error
        coeffs = centered @ self._modes
        reconstructed = coeffs @ self._modes.T
        errors = np.sqrt(np.mean((centered - reconstructed) ** 2, axis=1))
        self._baseline_error = float(np.mean(errors))
        self._error_threshold = float(np.percentile(errors, 95))
        
        self._is_fitted = True
        
        logger.info(
            f"POD fitted: {n_comp} components, "
            f"{cumulative_var[n_comp-1]*100:.1f}% variance explained, "
            f"baseline_error={self._baseline_error:.4f}"
        )
    
    def project(self, field: DeviationField) -> np.ndarray:
        """Project a deviation field onto POD modes."""
        if not self._is_fitted:
            raise ValueError("Engine not fitted")
        
        flat = field.to_flat()
        centered = flat - self._mean
        
        if self.pod_config.scale_data:
            centered = centered / self._std
        
        coefficients = centered @ self._modes
        return coefficients
    
    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        """Reconstruct deviation field from POD coefficients."""
        if not self._is_fitted:
            raise ValueError("Engine not fitted")
        
        reconstructed = coefficients @ self._modes.T
        
        if self.pod_config.scale_data:
            reconstructed = reconstructed * self._std
        
        return reconstructed + self._mean
    
    def compute_reconstruction_error(self, field: DeviationField) -> float:
        """Compute reconstruction error for a deviation field."""
        if not self._is_fitted:
            return field.rms_deviation
        
        coeffs = self.project(field)
        reconstructed = self.reconstruct(coeffs)
        original = field.to_flat()
        
        error = np.sqrt(np.mean((original - reconstructed) ** 2))
        return float(error)
    
    def analyze(self, field: DeviationField) -> float:
        """Return scalar error score (0-100) based on POD analysis."""
        if not self._is_fitted:
            logger.warning("POD not fitted, using RMS fallback")
            return min(100, field.rms_deviation * 10)
        
        # Compute reconstruction error
        recon_error = self.compute_reconstruction_error(field)
        
        # Normalize by threshold
        if self._error_threshold > 0:
            normalized = recon_error / self._error_threshold
        else:
            normalized = recon_error
        
        # Convert to 0-100 scale
        # 0 = normal, 100 = very anomalous
        score = 100 * (1 - np.exp(-normalized))
        
        return float(np.clip(score, 0, 100))
    
    def explain(
        self,
        field: DeviationField,
        process_params: Optional[Dict[str, float]] = None,
    ) -> XaiDtExplanation:
        """
        Generate detailed XAI explanation using POD analysis.
        
        1. Project field onto modes
        2. Identify dominant modes contributing to deviation
        3. If surrogate trained, link modes to parameters
        4. Generate probable causes with impacts
        """
        score = self.analyze(field)
        
        # Get POD coefficients
        if self._is_fitted:
            coeffs = self.project(field)
            
            # Identify dominant modes (those with largest |coefficient|)
            mode_contributions = np.abs(coeffs) * self._explained_variance
            dominant_idx = np.argsort(mode_contributions)[::-1][:3]
            dominant_modes = dominant_idx.tolist()
        else:
            coeffs = np.array([])
            dominant_modes = []
        
        # Build probable causes
        causes = []
        
        if process_params and self._is_fitted:
            # Analyze sensitivity of coefficients to parameters
            sensitivities = self._analyze_param_sensitivity(coeffs, process_params)
            
            for param, sensitivity in sensitivities[:self.xai_config.max_causes]:
                if sensitivity > self.xai_config.significance_threshold:
                    causes.append(ProvableCause(
                        parameter_name=param,
                        impact=sensitivity,
                        direction=self._estimate_direction(param, process_params),
                        suggested_action=self._generate_action(param, sensitivity),
                        confidence=0.6,  # Higher confidence with POD
                    ))
        else:
            # Fallback causes based on modes
            for i, mode_idx in enumerate(dominant_modes):
                causes.append(ProvableCause(
                    parameter_name=f"modo_{mode_idx}",
                    impact=float(self._explained_variance[mode_idx]) if mode_idx < len(self._explained_variance) else 0.1,
                    direction="both",
                    suggested_action=f"Investigar variação no modo principal {mode_idx}",
                    confidence=0.5,
                ))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(score, dominant_modes)
        
        recon_error = self.compute_reconstruction_error(field) if self._is_fitted else field.rms_deviation
        
        return XaiDtExplanation(
            timestamp=field.timestamp,
            machine_id=field.metadata.get("machine_id", "unknown"),
            scalar_error_score=score,
            probable_causes=causes,
            dominant_modes=dominant_modes,
            reconstruction_error=recon_error,
            confidence=0.6 if self._is_fitted else 0.3,
            recommendations=recommendations,
        )
    
    def _analyze_param_sensitivity(
        self,
        coeffs: np.ndarray,
        params: Dict[str, float],
    ) -> List[Tuple[str, float]]:
        """Analyze which parameters most influence the current deviation."""
        # Simple heuristic: parameters further from normal range
        # In full implementation: use trained surrogate model
        
        sensitivities = []
        for param, value in params.items():
            # Assume normal range is 0-100
            # Higher absolute value = more influence
            normalized = abs(value - 50) / 50 if value != 50 else 0
            sensitivity = normalized * 0.8  # Scale to [0, 0.8]
            sensitivities.append((param, sensitivity))
        
        sensitivities.sort(key=lambda x: x[1], reverse=True)
        return sensitivities
    
    def _estimate_direction(self, param: str, params: Dict[str, float]) -> str:
        """Estimate if parameter should increase or decrease."""
        value = params.get(param, 50)
        if value > 70:
            return "decrease"
        elif value < 30:
            return "increase"
        return "both"
    
    def _generate_action(self, param: str, sensitivity: float) -> str:
        """Generate suggested action for a parameter."""
        if sensitivity > 0.7:
            return f"Ajustar {param} urgentemente"
        elif sensitivity > 0.5:
            return f"Verificar e ajustar {param}"
        else:
            return f"Monitorar {param}"
    
    def _generate_recommendations(
        self,
        score: float,
        dominant_modes: List[int],
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if score > 80:
            recommendations.append("⚠️ Desvio crítico - intervenção imediata necessária")
            recommendations.append("Verificar todos os parâmetros de processo")
        elif score > 60:
            recommendations.append("⚡ Desvio significativo - monitorização intensiva")
            if dominant_modes:
                recommendations.append(f"Foco nos modos de variação: {dominant_modes}")
        elif score > 40:
            recommendations.append("📊 Desvio moderado - monitorização regular")
        elif score > 20:
            recommendations.append("✅ Desvio ligeiro - operação normal")
        else:
            recommendations.append("✅ Operação dentro dos parâmetros normais")
        
        return recommendations


# ═══════════════════════════════════════════════════════════════════════════════
# DEVIATION SURROGATE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class DeviationSurrogateModel(SurrogateModelBase):
    """
    Surrogate model for predicting deviation patterns from process parameters.
    
    Uses POD for dimensionality reduction + regression for interpolation.
    
    Pipeline:
    1. Train POD on deviation fields to get modes
    2. Train regression model: process_params → POD_coefficients
    3. Predict: new_params → predicted_coefficients → reconstructed_field
    
    TODO[R&D]:
    - Implement with Gaussian Process regression
    - Add kriging/RBF interpolation
    - Neural network surrogate
    """
    
    def __init__(self, n_modes: int = 5):
        self.n_modes = n_modes
        self.pod_engine = PodDeviationEngine(PODConfig(n_components=n_modes))
        self._X_train: Optional[np.ndarray] = None
        self._coefficients: Optional[np.ndarray] = None
        self._regression_model: Optional[Any] = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit surrogate model.
        
        Args:
            X: Input parameters (n_samples, n_params)
            y: Output deviation fields (n_samples, n_features) - flattened
        """
        logger.info(f"Fitting DeviationSurrogateModel: X={X.shape}, y={y.shape}")
        
        self._X_train = X
        
        # Convert y to deviation fields
        fields = []
        for i, row in enumerate(y):
            fields.append(DeviationField(
                points=np.arange(len(row)).reshape(-1, 1),
                deviations=row,
                field_name=f"sample_{i}",
            ))
        
        # Fit POD on deviation fields
        self.pod_engine.fit(fields)
        
        # Get coefficients for all training samples
        if self.pod_engine._is_fitted:
            self._coefficients = np.array([self.pod_engine.project(f) for f in fields])
        else:
            self._coefficients = y  # Fallback
        
        # Fit simple regression (linear for now)
        # TODO[R&D]: Use GP or neural network
        try:
            from sklearn.linear_model import Ridge
            self._regression_model = Ridge(alpha=1.0)
            self._regression_model.fit(X, self._coefficients)
            logger.info("Ridge regression fitted for surrogate")
        except ImportError:
            logger.warning("sklearn not available, using nearest neighbor fallback")
            self._regression_model = None
        
        self._is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict deviation fields for new parameters."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        
        if self._regression_model is not None:
            coeffs = self._regression_model.predict(X)
        else:
            # Nearest neighbor fallback
            coeffs = np.mean(self._coefficients, axis=0).reshape(1, -1)
            coeffs = np.tile(coeffs, (X.shape[0], 1))
        
        # Reconstruct from coefficients
        if self.pod_engine._is_fitted:
            predictions = np.array([self.pod_engine.reconstruct(c) for c in coeffs])
        else:
            predictions = coeffs
        
        return predictions
    
    def get_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction uncertainty.
        
        TODO[R&D]: Implement GP/kriging uncertainty
        """
        if not self._is_fitted:
            return np.ones(X.shape[0])
        
        # Simple distance-based uncertainty
        if self._X_train is not None:
            # Distance to nearest training point
            distances = np.min(np.linalg.norm(X[:, np.newaxis] - self._X_train, axis=2), axis=1)
            uncertainty = 1 - np.exp(-distances)  # Higher distance = higher uncertainty
        else:
            uncertainty = np.ones(X.shape[0]) * 0.5
        
        return uncertainty


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_deviation_engine(
    pod_config: Optional[PODConfig] = None,
    xai_config: Optional[XAIConfig] = None,
    use_advanced: Optional[bool] = None,
) -> DeviationEngineBase:
    """
    Factory function to get deviation engine based on FeatureFlags.
    
    Args:
        pod_config: Configuration for POD (if advanced)
        xai_config: Configuration for XAI
        use_advanced: Force mode (if None, uses FeatureFlags)
    
    Returns:
        DeviationEngineBase (SimpleDeviationEngine or PodDeviationEngine)
    """
    # Import FeatureFlags
    try:
        from ..feature_flags import FeatureFlags, DeviationEngine as DE
        
        if use_advanced is None:
            use_advanced = FeatureFlags.get_deviation_engine() == DE.POD
    except ImportError:
        if use_advanced is None:
            use_advanced = False
    
    if use_advanced:
        logger.info("Using PodDeviationEngine (ADVANCED)")
        return PodDeviationEngine(pod_config, xai_config)
    else:
        logger.info("Using SimpleDeviationEngine (BASE)")
        return SimpleDeviationEngine(xai_config)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def explain_deviation(
    field: DeviationField,
    process_params: Optional[Dict[str, float]] = None,
    use_advanced: bool = False,
) -> XaiDtExplanation:
    """
    Convenience function to explain a deviation.
    
    Args:
        field: The deviation field to explain
        process_params: Process parameters at time of measurement
        use_advanced: Whether to use POD-based analysis
    
    Returns:
        XaiDtExplanation with probable causes and recommendations
    """
    engine = get_deviation_engine(use_advanced=use_advanced)
    return engine.explain(field, process_params)


def create_test_deviation_field(
    n_points: int = 100,
    deviation_magnitude: float = 0.5,
    machine_id: str = "M1",
) -> DeviationField:
    """Create a test deviation field for demos."""
    points = np.linspace(0, 1, n_points).reshape(-1, 1)
    
    # Create deviation pattern with some structure
    deviations = deviation_magnitude * np.sin(2 * np.pi * points.flatten())
    deviations += 0.1 * np.random.randn(n_points)
    
    return DeviationField(
        points=points,
        deviations=deviations,
        field_name="test_deviation",
        metadata={"machine_id": machine_id},
    )


def analyze_machine_data(
    data: np.ndarray,
    n_components: int = 5,
    anomaly_threshold: float = 95,
) -> Dict[str, Any]:
    """
    Complete analysis pipeline for machine data.
    
    Returns:
        Dict with POD results, anomaly scores, etc.
    """
    # Create deviation fields from data
    fields = []
    for i, row in enumerate(data):
        fields.append(DeviationField(
            points=np.arange(len(row)).reshape(-1, 1),
            deviations=row,
            field_name=f"sample_{i}",
        ))
    
    # POD analysis
    engine = PodDeviationEngine(PODConfig(n_components=n_components))
    engine.fit(fields)
    
    # Analyze each sample
    scores = [engine.analyze(f) for f in fields]
    
    # Count anomalies
    threshold_score = 100 - anomaly_threshold
    n_anomalies = sum(1 for s in scores if s > threshold_score)
    
    return {
        "n_components": n_components,
        "n_anomalies": n_anomalies,
        "anomaly_rate": n_anomalies / len(fields) if fields else 0,
        "variance_explained": float(np.sum(engine._explained_variance)) if engine._is_fitted else 0,
        "mean_score": float(np.mean(scores)),
        "max_score": float(np.max(scores)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

# PODEngine alias for backward compatibility
PODEngine = PodDeviationEngine


def compute_pca_pod(data: np.ndarray, n_components: int = 5) -> PODResult:
    """Legacy function for POD computation."""
    fields = [DeviationField(
        points=np.arange(len(row)).reshape(-1, 1),
        deviations=row,
    ) for row in data]
    
    engine = PodDeviationEngine(PODConfig(n_components=n_components))
    engine.fit(fields)
    
    if engine._is_fitted:
        return PODResult(
            n_components=len(engine._explained_variance),
            explained_variance_ratio=engine._explained_variance,
            cumulative_variance=np.cumsum(engine._explained_variance),
            modes=engine._modes,
            singular_values=engine._singular_values,
            mean_field=engine._mean,
            reconstruction_error=engine._baseline_error,
        )
    else:
        return PODResult(
            n_components=0,
            explained_variance_ratio=np.array([]),
            cumulative_variance=np.array([]),
            modes=np.array([]),
            singular_values=np.array([]),
            mean_field=np.zeros(data.shape[1] if len(data.shape) > 1 else 1),
            reconstruction_error=float('inf'),
        )


def build_surrogate_model(X: np.ndarray, y: np.ndarray, n_modes: int = 5) -> DeviationSurrogateModel:
    """Legacy function to build surrogate model."""
    model = DeviationSurrogateModel(n_modes=n_modes)
    model.fit(X, y)
    return model


def explain_rul_factors(
    rul_estimate: Any,
    sensor_history: Optional[np.ndarray] = None,
) -> XAIResult:
    """Legacy function for RUL factor explanation."""
    factors = []
    
    if hasattr(rul_estimate, 'current_hi'):
        hi = rul_estimate.current_hi
        if hi < 0.5:
            factors.append(("health_indicator", 0.8))
            factors.append(("degradation_rate", 0.6))
        else:
            factors.append(("health_indicator", 0.4))
    
    if sensor_history is not None and len(sensor_history) > 0:
        trend = np.mean(np.diff(sensor_history[-10:])) if len(sensor_history) > 10 else 0
        if abs(trend) > 0.01:
            factors.append(("recent_trend", abs(trend) * 10))
    
    explanation = f"RUL baseado em {len(factors)} factores principais"
    
    return XAIResult(
        factors=factors,
        explanation=explanation,
        confidence=0.6,
        dominant_modes=[0, 1] if factors else [],
    )
