"""
ProdPlan 4.0 - Setup Time and Process Time Prediction

This module provides ML models for predicting:
- Setup times (changeover between products)
- Process times (operation duration)

Models support:
- Rule-based (from setup matrix)
- Statistical (historical averages)
- ML (XGBoost, LightGBM)
- Neural networks (future)

R&D / SIFIDE: WP2 - Predictive Intelligence
Research Questions:
- Q2.3: Can ML models predict setup times with <10% MAPE?
- Q2.4: What features are most predictive for process times?
Metrics: MAPE, RMSE, feature importance.

References:
- Allahverdi et al. (2008). A survey of scheduling problems with setup times
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS AND CONFIG
# ============================================================

class PredictionModel(Enum):
    """Available prediction models."""
    RULE_BASED = "rule_based"
    STATISTICAL = "statistical"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    NEURAL = "neural"


@dataclass
class SetupPredictionConfig:
    """Configuration for setup time prediction."""
    model: PredictionModel = PredictionModel.RULE_BASED
    
    # Default values (when no data available)
    default_setup_min: float = 15.0
    same_family_setup_min: float = 5.0
    
    # ML parameters
    n_estimators: int = 100
    max_depth: int = 5
    
    # Feature engineering
    use_machine_features: bool = True
    use_temporal_features: bool = True
    use_sequence_features: bool = True
    
    # Training
    min_samples_for_ml: int = 100


@dataclass
class ProcessPredictionConfig:
    """Configuration for process time prediction."""
    model: PredictionModel = PredictionModel.RULE_BASED
    
    # Default values
    default_time_per_unit_min: float = 1.0
    
    # Speed factor bounds
    min_speed_factor: float = 0.5
    max_speed_factor: float = 2.0
    
    # ML parameters
    n_estimators: int = 100
    
    # Learning factors
    learning_rate: float = 0.0  # 0 = no learning effect


@dataclass
class PredictionResult:
    """Result of a prediction."""
    predicted_value: float
    confidence_lower: float
    confidence_upper: float
    confidence_level: float = 0.95
    model_used: str = ""
    features_used: List[str] = field(default_factory=list)


# ============================================================
# SETUP TIME PREDICTION
# ============================================================

class SetupTimePredictor(ABC):
    """Abstract base class for setup time prediction."""
    
    @abstractmethod
    def predict(
        self,
        from_family: str,
        to_family: str,
        machine_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> PredictionResult:
        """
        Predict setup time.
        
        Args:
            from_family: Previous setup family
            to_family: Next setup family
            machine_id: Machine identifier
            context: Additional context (operator, time, etc.)
        
        Returns:
            PredictionResult with predicted setup time in minutes
        """
        pass
    
    @abstractmethod
    def fit(self, historical_data: pd.DataFrame) -> None:
        """Fit model to historical setup data."""
        pass


class RuleBasedSetupPredictor(SetupTimePredictor):
    """
    Rule-based setup time prediction using setup matrix.
    
    Simple but interpretable. Uses:
    - Setup matrix (from_family, to_family) -> time
    - Default values for missing entries
    """
    
    def __init__(self, config: Optional[SetupPredictionConfig] = None):
        self.config = config or SetupPredictionConfig()
        self._setup_matrix: Dict[Tuple[str, str], float] = {}
        self._machine_factors: Dict[str, float] = {}
    
    def fit(self, historical_data: pd.DataFrame) -> None:
        """
        Fit from historical data or setup matrix.
        
        Expected columns: from_family, to_family, setup_time_min, machine_id (optional)
        """
        if 'from_setup_family' in historical_data.columns:
            # Setup matrix format
            for _, row in historical_data.iterrows():
                key = (row['from_setup_family'], row['to_setup_family'])
                self._setup_matrix[key] = row['setup_time_min']
        
        elif 'from_family' in historical_data.columns:
            # Historical data format
            grouped = historical_data.groupby(['from_family', 'to_family'])['setup_time_min'].mean()
            for (f, t), v in grouped.items():
                self._setup_matrix[(f, t)] = v
            
            # Machine factors
            if 'machine_id' in historical_data.columns:
                overall_avg = historical_data['setup_time_min'].mean()
                machine_avg = historical_data.groupby('machine_id')['setup_time_min'].mean()
                self._machine_factors = (machine_avg / overall_avg).to_dict()
    
    def set_setup_matrix(self, matrix: Dict[Tuple[str, str], float]) -> None:
        """Directly set setup matrix."""
        self._setup_matrix = matrix
    
    def predict(
        self,
        from_family: str,
        to_family: str,
        machine_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> PredictionResult:
        """Predict setup time using rules."""
        
        # Same family = minimal setup
        if from_family == to_family:
            base_time = self.config.same_family_setup_min
        else:
            # Look up in matrix
            key = (from_family, to_family)
            if key in self._setup_matrix:
                base_time = self._setup_matrix[key]
            else:
                # Try reverse lookup (some matrices are symmetric)
                rev_key = (to_family, from_family)
                if rev_key in self._setup_matrix:
                    base_time = self._setup_matrix[rev_key]
                else:
                    base_time = self.config.default_setup_min
        
        # Apply machine factor
        if machine_id and machine_id in self._machine_factors:
            base_time *= self._machine_factors[machine_id]
        
        # Simple confidence interval
        uncertainty = base_time * 0.2  # 20% uncertainty
        
        return PredictionResult(
            predicted_value=base_time,
            confidence_lower=max(0, base_time - uncertainty),
            confidence_upper=base_time + uncertainty,
            model_used="RuleBased",
            features_used=["from_family", "to_family"],
        )


class MLSetupPredictor(SetupTimePredictor):
    """
    ML-based setup time prediction.
    
    Uses gradient boosting with features:
    - From/to family (one-hot encoded)
    - Machine ID
    - Time of day/week
    - Operator (if available)
    
    TODO[R&D]: Setup time prediction research:
    - Feature engineering for sequence effects
    - Transfer learning across machines
    - Online learning for adaptation
    """
    
    def __init__(self, config: Optional[SetupPredictionConfig] = None):
        self.config = config or SetupPredictionConfig()
        self._model = None
        self._encoder = None
        self._feature_names = []
        self._is_fitted = False
        
        # Fallback predictor
        self._fallback = RuleBasedSetupPredictor(config)
    
    def fit(self, historical_data: pd.DataFrame) -> None:
        """Fit ML model to historical setup data."""
        
        # Check if enough data for ML
        if len(historical_data) < self.config.min_samples_for_ml:
            logger.warning(f"Not enough data for ML ({len(historical_data)} < {self.config.min_samples_for_ml}), using fallback")
            self._fallback.fit(historical_data)
            return
        
        try:
            import xgboost as xgb
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            logger.warning("xgboost/sklearn not available, using fallback")
            self._fallback.fit(historical_data)
            return
        
        # Prepare features
        required_cols = ['from_family', 'to_family', 'setup_time_min']
        if not all(c in historical_data.columns for c in required_cols):
            logger.warning("Missing required columns, using fallback")
            self._fallback.fit(historical_data)
            return
        
        # Encode categorical features
        self._encoder = {}
        X_parts = []
        
        for col in ['from_family', 'to_family']:
            le = LabelEncoder()
            encoded = le.fit_transform(historical_data[col])
            self._encoder[col] = le
            X_parts.append(encoded.reshape(-1, 1))
        
        if 'machine_id' in historical_data.columns and self.config.use_machine_features:
            le = LabelEncoder()
            encoded = le.fit_transform(historical_data['machine_id'])
            self._encoder['machine_id'] = le
            X_parts.append(encoded.reshape(-1, 1))
        
        X = np.hstack(X_parts)
        y = historical_data['setup_time_min'].values
        
        # Train model
        self._model = xgb.XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=42,
        )
        self._model.fit(X, y)
        
        self._is_fitted = True
        logger.info(f"ML setup predictor trained on {len(historical_data)} samples")
    
    def predict(
        self,
        from_family: str,
        to_family: str,
        machine_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> PredictionResult:
        """Predict using ML model or fallback."""
        
        if not self._is_fitted or self._model is None:
            return self._fallback.predict(from_family, to_family, machine_id, context)
        
        try:
            # Encode features
            X_parts = []
            
            for col, val in [('from_family', from_family), ('to_family', to_family)]:
                le = self._encoder.get(col)
                if le and val in le.classes_:
                    encoded = le.transform([val])[0]
                else:
                    # Unknown category - use fallback
                    return self._fallback.predict(from_family, to_family, machine_id, context)
                X_parts.append(encoded)
            
            if 'machine_id' in self._encoder and machine_id:
                le = self._encoder['machine_id']
                if machine_id in le.classes_:
                    X_parts.append(le.transform([machine_id])[0])
                else:
                    X_parts.append(0)  # Default
            
            X = np.array(X_parts).reshape(1, -1)
            pred = self._model.predict(X)[0]
            
            # Estimate uncertainty using prediction interval
            # TODO: Implement proper prediction intervals
            uncertainty = pred * 0.15
            
            return PredictionResult(
                predicted_value=max(0, pred),
                confidence_lower=max(0, pred - 2 * uncertainty),
                confidence_upper=pred + 2 * uncertainty,
                model_used="XGBoost",
                features_used=list(self._encoder.keys()),
            )
            
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}, using fallback")
            return self._fallback.predict(from_family, to_family, machine_id, context)


# ============================================================
# PROCESS TIME PREDICTION
# ============================================================

class ProcessTimePredictor(ABC):
    """Abstract base class for process time prediction."""
    
    @abstractmethod
    def predict(
        self,
        op_code: str,
        machine_id: str,
        quantity: int,
        context: Optional[Dict[str, Any]] = None
    ) -> PredictionResult:
        """
        Predict process time.
        
        Args:
            op_code: Operation code
            machine_id: Machine identifier
            quantity: Quantity to process
            context: Additional context (article, batch size, etc.)
        
        Returns:
            PredictionResult with predicted process time in minutes
        """
        pass
    
    @abstractmethod
    def fit(self, historical_data: pd.DataFrame) -> None:
        """Fit model to historical process data."""
        pass


class RuleBasedProcessPredictor(ProcessTimePredictor):
    """
    Rule-based process time prediction.
    
    Uses: base_time_per_unit × quantity × machine_speed_factor
    
    With optional learning effect:
    time = base_time × quantity^learning_rate
    """
    
    def __init__(self, config: Optional[ProcessPredictionConfig] = None):
        self.config = config or ProcessPredictionConfig()
        self._base_times: Dict[str, float] = {}  # op_code -> time_per_unit
        self._speed_factors: Dict[str, float] = {}  # machine_id -> factor
    
    def fit(self, historical_data: pd.DataFrame) -> None:
        """
        Fit from historical data.
        
        Expected columns: op_code, machine_id, quantity, actual_time_min
        """
        if 'op_code' in historical_data.columns:
            # Compute average time per unit by operation
            if 'actual_time_min' in historical_data.columns and 'quantity' in historical_data.columns:
                historical_data = historical_data.copy()
                historical_data['time_per_unit'] = historical_data['actual_time_min'] / historical_data['quantity']
                self._base_times = historical_data.groupby('op_code')['time_per_unit'].mean().to_dict()
            elif 'base_time_per_unit_min' in historical_data.columns:
                # Use provided base times
                self._base_times = historical_data.set_index('op_code')['base_time_per_unit_min'].to_dict()
        
        if 'machine_id' in historical_data.columns:
            if 'speed_factor' in historical_data.columns:
                self._speed_factors = historical_data.set_index('machine_id')['speed_factor'].to_dict()
            elif 'actual_time_min' in historical_data.columns:
                # Compute speed factors from actual times
                overall_avg = historical_data['actual_time_min'].mean()
                machine_avg = historical_data.groupby('machine_id')['actual_time_min'].mean()
                self._speed_factors = (machine_avg / overall_avg).to_dict()
    
    def set_base_times(self, base_times: Dict[str, float]) -> None:
        """Directly set base times."""
        self._base_times = base_times
    
    def set_speed_factors(self, speed_factors: Dict[str, float]) -> None:
        """Directly set machine speed factors."""
        self._speed_factors = speed_factors
    
    def predict(
        self,
        op_code: str,
        machine_id: str,
        quantity: int,
        context: Optional[Dict[str, Any]] = None
    ) -> PredictionResult:
        """Predict process time using rules."""
        
        # Get base time per unit
        base_time = self._base_times.get(op_code, self.config.default_time_per_unit_min)
        
        # Get speed factor
        speed_factor = self._speed_factors.get(machine_id, 1.0)
        speed_factor = max(self.config.min_speed_factor, min(self.config.max_speed_factor, speed_factor))
        
        # Compute total time
        if self.config.learning_rate > 0 and quantity > 1:
            # Learning curve: time decreases with quantity
            # T = base_time × quantity^(learning_rate)
            # where learning_rate ≈ -0.2 for typical learning curves
            total_time = base_time * (quantity ** (1 + self.config.learning_rate)) / speed_factor
        else:
            total_time = base_time * quantity / speed_factor
        
        # Uncertainty
        uncertainty = total_time * 0.1
        
        return PredictionResult(
            predicted_value=total_time,
            confidence_lower=max(0, total_time - uncertainty),
            confidence_upper=total_time + uncertainty,
            model_used="RuleBased",
            features_used=["op_code", "machine_id", "quantity"],
        )


class MLProcessPredictor(ProcessTimePredictor):
    """
    ML-based process time prediction.
    
    Uses gradient boosting with features:
    - Operation code
    - Machine ID
    - Quantity
    - Article characteristics (if available)
    - Operator experience (if available)
    
    TODO[R&D]: Process time prediction research:
    - Impact of material variations on accuracy
    - Online learning for drift adaptation
    - Anomaly detection for quality issues
    """
    
    def __init__(self, config: Optional[ProcessPredictionConfig] = None):
        self.config = config or ProcessPredictionConfig()
        self._model = None
        self._encoder = None
        self._is_fitted = False
        self._fallback = RuleBasedProcessPredictor(config)
    
    def fit(self, historical_data: pd.DataFrame) -> None:
        """Fit ML model."""
        
        required_cols = ['op_code', 'machine_id', 'quantity', 'actual_time_min']
        if not all(c in historical_data.columns for c in required_cols):
            logger.warning("Missing required columns for ML, using fallback")
            self._fallback.fit(historical_data)
            return
        
        if len(historical_data) < 100:
            logger.warning("Not enough data for ML, using fallback")
            self._fallback.fit(historical_data)
            return
        
        try:
            import xgboost as xgb
            from sklearn.preprocessing import LabelEncoder
            
            self._encoder = {}
            X_parts = []
            
            for col in ['op_code', 'machine_id']:
                le = LabelEncoder()
                encoded = le.fit_transform(historical_data[col])
                self._encoder[col] = le
                X_parts.append(encoded.reshape(-1, 1))
            
            X_parts.append(historical_data['quantity'].values.reshape(-1, 1))
            
            X = np.hstack(X_parts)
            y = historical_data['actual_time_min'].values
            
            self._model = xgb.XGBRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=5,
                random_state=42,
            )
            self._model.fit(X, y)
            self._is_fitted = True
            
        except ImportError:
            logger.warning("xgboost not available, using fallback")
            self._fallback.fit(historical_data)
    
    def predict(
        self,
        op_code: str,
        machine_id: str,
        quantity: int,
        context: Optional[Dict[str, Any]] = None
    ) -> PredictionResult:
        """Predict using ML model or fallback."""
        
        if not self._is_fitted or self._model is None:
            return self._fallback.predict(op_code, machine_id, quantity, context)
        
        try:
            X_parts = []
            
            for col, val in [('op_code', op_code), ('machine_id', machine_id)]:
                le = self._encoder.get(col)
                if le and val in le.classes_:
                    X_parts.append(le.transform([val])[0])
                else:
                    return self._fallback.predict(op_code, machine_id, quantity, context)
            
            X_parts.append(quantity)
            X = np.array(X_parts).reshape(1, -1)
            
            pred = self._model.predict(X)[0]
            uncertainty = pred * 0.1
            
            return PredictionResult(
                predicted_value=max(0, pred),
                confidence_lower=max(0, pred - 2 * uncertainty),
                confidence_upper=pred + 2 * uncertainty,
                model_used="XGBoost",
                features_used=['op_code', 'machine_id', 'quantity'],
            )
            
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return self._fallback.predict(op_code, machine_id, quantity, context)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def predict_setup_time(
    from_family: str,
    to_family: str,
    machine_id: Optional[str] = None,
    setup_matrix: Optional[Dict[Tuple[str, str], float]] = None
) -> float:
    """
    Convenience function for setup time prediction.
    
    Returns:
        Predicted setup time in minutes
    """
    predictor = RuleBasedSetupPredictor()
    if setup_matrix:
        predictor.set_setup_matrix(setup_matrix)
    result = predictor.predict(from_family, to_family, machine_id)
    return result.predicted_value


def predict_process_time(
    op_code: str,
    machine_id: str,
    quantity: int,
    base_time_per_unit: Optional[float] = None,
    speed_factor: Optional[float] = None
) -> float:
    """
    Convenience function for process time prediction.
    
    Returns:
        Predicted process time in minutes
    """
    predictor = RuleBasedProcessPredictor()
    if base_time_per_unit:
        predictor.set_base_times({op_code: base_time_per_unit})
    if speed_factor:
        predictor.set_speed_factors({machine_id: speed_factor})
    
    result = predictor.predict(op_code, machine_id, quantity)
    return result.predicted_value


def create_setup_predictor(
    model_type: PredictionModel = PredictionModel.RULE_BASED,
    config: Optional[SetupPredictionConfig] = None
) -> SetupTimePredictor:
    """Factory function for setup predictors."""
    if model_type == PredictionModel.RULE_BASED:
        return RuleBasedSetupPredictor(config)
    elif model_type in [PredictionModel.XGBOOST, PredictionModel.LIGHTGBM]:
        return MLSetupPredictor(config)
    else:
        return RuleBasedSetupPredictor(config)


def create_process_predictor(
    model_type: PredictionModel = PredictionModel.RULE_BASED,
    config: Optional[ProcessPredictionConfig] = None
) -> ProcessTimePredictor:
    """Factory function for process predictors."""
    if model_type == PredictionModel.RULE_BASED:
        return RuleBasedProcessPredictor(config)
    elif model_type in [PredictionModel.XGBOOST, PredictionModel.LIGHTGBM]:
        return MLProcessPredictor(config)
    else:
        return RuleBasedProcessPredictor(config)


