"""
Setup Engine — Setup Time Prediction (Rules + ML)

R&D Module for WP1: APS Core + Routing Intelligence

Research Question (Q1):
    Can we use ML to predict setup times more accurately than rule-based
    family matrices, and does this improve scheduling quality?

Hypotheses:
    H1.2: ML-predicted setup times (XGBoost on historical) reduce total setup
          by ≥15% vs rule-based family matrix
    H1.3: Learned setup patterns reveal hidden dependencies not captured
          in manual family definitions

Technical Uncertainty:
    - Quality and quantity of historical setup data
    - Feature engineering for setup prediction
    - Handling of new machine/product combinations (cold start)
    - Trade-off between prediction accuracy and computational cost

Experiment Design:
    E1.2: Setup Prediction Accuracy
    - Dataset: 6 months of actual setup times (if available) or synthetic
    - Train/test split: 80/20 temporal
    - Models: Baseline (family matrix), XGBoost, LightGBM
    - Metrics: MAE, RMSE, Setup Hours Saved

Usage:
    from backend.research.setup_engine import SetupEngine, SetupPredictor
    
    engine = SetupEngine(predictor=SetupPredictor.ML_XGBOOST)
    setup_time = engine.predict_setup(
        machine_id="M-301",
        from_family="corte_fino",
        to_family="corte_grosso",
        features={"operator_id": "OP-01", "shift": "morning"}
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

import pandas as pd
import numpy as np


class SetupPredictor(Enum):
    """Available setup prediction methods."""
    RULE_BASED = "rule_based"           # Family matrix lookup (baseline)
    HISTORICAL_MEAN = "historical_mean"  # Mean from historical data
    ML_XGBOOST = "ml_xgboost"           # XGBoost regressor
    ML_LIGHTGBM = "ml_lightgbm"         # LightGBM regressor
    HYBRID = "hybrid"                    # Rule + ML correction


@dataclass
class SetupPrediction:
    """Result of a setup time prediction."""
    predicted_min: float
    confidence: float  # 0-1, how confident the prediction is
    method: str
    features_used: Dict[str, Any]
    explanation: str


class BaseSetupPredictor(ABC):
    """Abstract base class for setup predictors."""
    
    @abstractmethod
    def predict(
        self,
        machine_id: str,
        from_family: str,
        to_family: str,
        features: Optional[Dict[str, Any]] = None,
    ) -> SetupPrediction:
        """Predict setup time for a family transition."""
        pass
    
    @abstractmethod
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the predictor on historical data."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the predictor."""
        pass


class RuleBasedPredictor(BaseSetupPredictor):
    """
    Baseline predictor using a family transition matrix.
    
    This is the current production method.
    """
    
    def __init__(self, setup_matrix: Optional[Dict[Tuple[str, str], float]] = None):
        self.setup_matrix = setup_matrix or {}
        self.default_setup_min = 30.0
    
    def predict(
        self,
        machine_id: str,
        from_family: str,
        to_family: str,
        features: Optional[Dict[str, Any]] = None,
    ) -> SetupPrediction:
        if from_family == to_family:
            return SetupPrediction(
                predicted_min=0.0,
                confidence=1.0,
                method="rule_based",
                features_used={"from_family": from_family, "to_family": to_family},
                explanation="Mesma família de setup, sem troca necessária.",
            )
        
        key = (from_family, to_family)
        if key in self.setup_matrix:
            setup_time = self.setup_matrix[key]
            confidence = 0.9  # High confidence if in matrix
            explanation = f"Setup de {from_family} → {to_family} definido na matriz."
        else:
            setup_time = self.default_setup_min
            confidence = 0.5  # Lower confidence for default
            explanation = f"Setup {from_family} → {to_family} não definido, usando default ({self.default_setup_min} min)."
        
        return SetupPrediction(
            predicted_min=setup_time,
            confidence=confidence,
            method="rule_based",
            features_used={"from_family": from_family, "to_family": to_family},
            explanation=explanation,
        )
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Rule-based doesn't need training, but can extract matrix from data."""
        if data.empty:
            return {"status": "no_data"}
        
        # TODO[R&D]: Extract setup matrix from historical data
        # Group by (from_family, to_family) and compute mean
        return {"status": "rule_based_no_training"}
    
    @property
    def name(self) -> str:
        return "rule_based"
    
    def load_matrix_from_df(self, setup_df: pd.DataFrame) -> None:
        """Load setup matrix from DataFrame."""
        for _, row in setup_df.iterrows():
            from_fam = str(row.get("from_setup_family", ""))
            to_fam = str(row.get("to_setup_family", ""))
            time_min = float(row.get("setup_time_min", 30.0))
            self.setup_matrix[(from_fam, to_fam)] = time_min


class HistoricalMeanPredictor(BaseSetupPredictor):
    """
    Predict based on historical mean setup times.
    
    Simple but can capture patterns not in manual matrix.
    """
    
    def __init__(self):
        self.historical_means: Dict[Tuple[str, str], float] = {}
        self.global_mean: float = 30.0
        self.sample_counts: Dict[Tuple[str, str], int] = {}
    
    def predict(
        self,
        machine_id: str,
        from_family: str,
        to_family: str,
        features: Optional[Dict[str, Any]] = None,
    ) -> SetupPrediction:
        if from_family == to_family:
            return SetupPrediction(
                predicted_min=0.0,
                confidence=1.0,
                method="historical_mean",
                features_used={"from_family": from_family, "to_family": to_family},
                explanation="Mesma família, sem setup.",
            )
        
        key = (from_family, to_family)
        if key in self.historical_means:
            setup_time = self.historical_means[key]
            n_samples = self.sample_counts.get(key, 0)
            confidence = min(0.95, 0.5 + 0.05 * n_samples)  # More samples = more confidence
            explanation = f"Média histórica baseada em {n_samples} observações."
        else:
            setup_time = self.global_mean
            confidence = 0.3
            explanation = "Sem dados históricos para esta transição, usando média global."
        
        return SetupPrediction(
            predicted_min=setup_time,
            confidence=confidence,
            method="historical_mean",
            features_used={"from_family": from_family, "to_family": to_family},
            explanation=explanation,
        )
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train on historical setup data.
        
        Expected columns: from_family, to_family, setup_time_min
        """
        if data.empty:
            return {"status": "no_data", "n_samples": 0}
        
        # Compute global mean
        self.global_mean = data["setup_time_min"].mean()
        
        # Compute per-transition means
        grouped = data.groupby(["from_family", "to_family"])["setup_time_min"]
        self.historical_means = grouped.mean().to_dict()
        self.sample_counts = grouped.count().to_dict()
        
        return {
            "status": "trained",
            "n_samples": len(data),
            "n_transitions": len(self.historical_means),
            "global_mean": self.global_mean,
        }
    
    @property
    def name(self) -> str:
        return "historical_mean"


class MLXGBoostPredictor(BaseSetupPredictor):
    """
    ML-based setup prediction using XGBoost.
    
    TODO[R&D]: Implement full feature engineering and training pipeline.
    
    Features to consider:
    - from_family (encoded)
    - to_family (encoded)
    - machine_id (encoded)
    - hour_of_day
    - day_of_week
    - operator_experience (if available)
    - previous_setup_duration (lag feature)
    
    Experiment E1.2: Compare MAE against rule-based baseline.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.model_path = model_path
        self.feature_columns: List[str] = []
        self.family_encoder: Dict[str, int] = {}
        self.machine_encoder: Dict[str, int] = {}
        self._fallback = RuleBasedPredictor()
    
    def predict(
        self,
        machine_id: str,
        from_family: str,
        to_family: str,
        features: Optional[Dict[str, Any]] = None,
    ) -> SetupPrediction:
        if from_family == to_family:
            return SetupPrediction(
                predicted_min=0.0,
                confidence=1.0,
                method="ml_xgboost",
                features_used={"from_family": from_family, "to_family": to_family},
                explanation="Mesma família, sem setup.",
            )
        
        if self.model is None:
            # Fallback to rule-based if no model trained
            fallback_pred = self._fallback.predict(machine_id, from_family, to_family, features)
            fallback_pred.explanation = "Modelo ML não treinado, usando fallback rule-based. " + fallback_pred.explanation
            return fallback_pred
        
        # TODO[R&D]: Implement actual prediction
        # 1. Encode features
        # 2. Build feature vector
        # 3. Predict with model
        # 4. Return prediction with confidence interval
        
        return self._fallback.predict(machine_id, from_family, to_family, features)
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train XGBoost model on historical setup data.
        
        TODO[R&D]: Implement full training pipeline.
        
        Expected columns:
        - from_family, to_family, machine_id
        - setup_time_min (target)
        - Optional: timestamp, operator_id, etc.
        """
        if data.empty or len(data) < 100:
            return {"status": "insufficient_data", "n_samples": len(data)}
        
        # TODO[R&D]: Implement training
        # 1. Feature engineering
        # 2. Train/test split (temporal)
        # 3. Hyperparameter tuning
        # 4. Model training
        # 5. Evaluation metrics
        
        return {
            "status": "not_implemented",
            "note": "TODO[R&D]: Implement XGBoost training pipeline",
            "n_samples": len(data),
        }
    
    @property
    def name(self) -> str:
        return "ml_xgboost"


class HybridPredictor(BaseSetupPredictor):
    """
    Hybrid predictor: Rule-based + ML correction.
    
    TODO[R&D]: Test hypothesis that ML correction improves on rules.
    
    Approach:
    1. Get rule-based prediction
    2. Use ML to predict residual (actual - rule_based)
    3. Final = rule_based + predicted_residual
    """
    
    def __init__(
        self,
        rule_predictor: Optional[RuleBasedPredictor] = None,
        ml_predictor: Optional[MLXGBoostPredictor] = None,
    ):
        self.rule_predictor = rule_predictor or RuleBasedPredictor()
        self.ml_predictor = ml_predictor or MLXGBoostPredictor()
        self.correction_model = None
    
    def predict(
        self,
        machine_id: str,
        from_family: str,
        to_family: str,
        features: Optional[Dict[str, Any]] = None,
    ) -> SetupPrediction:
        # Get rule-based prediction
        rule_pred = self.rule_predictor.predict(machine_id, from_family, to_family, features)
        
        if self.correction_model is None:
            # No correction model, return rule-based
            rule_pred.method = "hybrid_no_correction"
            return rule_pred
        
        # TODO[R&D]: Apply ML correction
        # correction = self.correction_model.predict(...)
        # final = rule_pred.predicted_min + correction
        
        return rule_pred
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the correction model."""
        # TODO[R&D]: Implement hybrid training
        return {"status": "not_implemented"}
    
    @property
    def name(self) -> str:
        return "hybrid"


class SetupEngine:
    """
    Main setup engine that predicts setup times.
    
    Supports pluggable predictors for experimentation.
    """
    
    PREDICTORS: Dict[SetupPredictor, type] = {
        SetupPredictor.RULE_BASED: RuleBasedPredictor,
        SetupPredictor.HISTORICAL_MEAN: HistoricalMeanPredictor,
        SetupPredictor.ML_XGBOOST: MLXGBoostPredictor,
        SetupPredictor.HYBRID: HybridPredictor,
    }
    
    def __init__(
        self,
        predictor: SetupPredictor = SetupPredictor.RULE_BASED,
        setup_matrix: Optional[Dict[Tuple[str, str], float]] = None,
    ):
        self.predictor_type = predictor
        predictor_class = self.PREDICTORS[predictor]
        
        if predictor == SetupPredictor.RULE_BASED:
            self.predictor = predictor_class(setup_matrix=setup_matrix)
        else:
            self.predictor = predictor_class()
        
        # Logging for experiment analysis
        self._prediction_log: List[Dict[str, Any]] = []
    
    def predict_setup(
        self,
        machine_id: str,
        from_family: str,
        to_family: str,
        features: Optional[Dict[str, Any]] = None,
        log_prediction: bool = True,
    ) -> SetupPrediction:
        """
        Predict setup time for a family transition.
        """
        prediction = self.predictor.predict(machine_id, from_family, to_family, features)
        
        if log_prediction:
            self._prediction_log.append({
                "machine_id": machine_id,
                "from_family": from_family,
                "to_family": to_family,
                "predicted_min": prediction.predicted_min,
                "confidence": prediction.confidence,
                "method": prediction.method,
            })
        
        return prediction
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the predictor on historical data."""
        return self.predictor.train(data)
    
    def get_prediction_log(self) -> List[Dict[str, Any]]:
        """Return logged predictions for analysis."""
        return self._prediction_log
    
    def evaluate(
        self,
        test_data: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Evaluate predictor on test data.
        
        TODO[R&D]: Use this for experiment E1.2.
        
        Args:
            test_data: DataFrame with from_family, to_family, machine_id, actual_setup_min
        
        Returns:
            Dict with MAE, RMSE, etc.
        """
        if test_data.empty:
            return {"mae": float("nan"), "rmse": float("nan")}
        
        predictions = []
        actuals = []
        
        for _, row in test_data.iterrows():
            pred = self.predict_setup(
                machine_id=str(row.get("machine_id", "")),
                from_family=str(row.get("from_family", "")),
                to_family=str(row.get("to_family", "")),
                log_prediction=False,
            )
            predictions.append(pred.predicted_min)
            actuals.append(float(row.get("actual_setup_min", 0)))
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "n_samples": len(test_data),
            "mean_actual": float(np.mean(actuals)),
            "mean_predicted": float(np.mean(predictions)),
        }


# ============================================================
# EXPERIMENT SUPPORT
# ============================================================

def run_setup_experiment(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    predictor: SetupPredictor,
) -> Dict[str, Any]:
    """
    Run a setup prediction experiment.
    
    TODO[R&D]: Entry point for experiment E1.2.
    """
    engine = SetupEngine(predictor=predictor)
    
    # Train
    train_result = engine.train(train_data)
    
    # Evaluate
    eval_result = engine.evaluate(test_data)
    
    return {
        "predictor": predictor.value,
        "train_result": train_result,
        "eval_result": eval_result,
    }



