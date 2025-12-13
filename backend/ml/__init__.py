"""
ProdPlan 4.0 - Machine Learning Engine

This package contains ML models for industrial prediction and optimization:
- Demand forecasting (time series)
- Setup time prediction
- Process time prediction
- Remaining Useful Life (RUL) estimation
- Anomaly detection

Architecture supports pluggable models:
- Classical ML (XGBoost, LightGBM, Random Forest)
- Deep Learning (LSTM, Transformers)
- Bayesian models (for uncertainty)

R&D / SIFIDE: This module is central to WP2 (Predictive Intelligence) research.
"""

from forecasting import (
    DemandForecaster,
    LeadTimeForecaster,
    forecast_demand,
    forecast_lead_time,
)
from setup_models import (
    SetupTimePredictor,
    ProcessTimePredictor,
    predict_setup_time,
    predict_process_time,
)
from rul_models import (
    RULEstimator,
    BayesianRULEstimator,
    estimate_rul,
)

__all__ = [
    # Forecasting
    "DemandForecaster",
    "LeadTimeForecaster",
    "forecast_demand",
    "forecast_lead_time",
    # Setup/Process
    "SetupTimePredictor",
    "ProcessTimePredictor",
    "predict_setup_time",
    "predict_process_time",
    # RUL
    "RULEstimator",
    "BayesianRULEstimator",
    "estimate_rul",
]



