"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — DELIVERY TIME ENGINE
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Automatic delivery time estimation with deterministic and ML-based approaches.

DELIVERY TIME COMPONENTS
════════════════════════

T_delivery = T_processing + T_setup + T_queue + T_buffer

where:
    T_processing = Σ operation_times
    T_setup = Σ setup_times between operations
    T_queue = waiting time in machine queues
    T_buffer = safety margin = k × σ_total

ESTIMATION METHODS
══════════════════

1. DETERMINISTIC (Baseline)
   ─────────────────────────
   
   Uses planned/standard times:
   - Processing from routing
   - Setup from setup matrix
   - Queue estimated from machine load
   
   Confidence based on data quality (SNR)

2. HISTORICAL (Statistical)
   ────────────────────────
   
   Uses past lead times for same product:
   - μ_historical = mean of past lead times
   - σ_historical = std dev
   - Prediction = μ + k × σ (for percentile)

3. ML-BASED (Future)
   ──────────────────
   
   Features:
   - Product type
   - Order quantity
   - Current machine loads
   - Day of week
   - Historical performance
   
   Models:
   - MVP: Linear Regression
   - TODO: XGBoost, LightGBM
   - TODO: DeepAR, NST for calendar-aware predictions

CONFIDENCE CALCULATION
══════════════════════

Confidence = f(data_coverage, historical_accuracy, current_load)

where:
    data_coverage = % of operations with known times
    historical_accuracy = 1 - MAPE_historical
    current_load = factor based on machine utilization

SNR INTEGRATION
═══════════════

SNR_delivery = Var(mean_by_type) / Var(residual)

High SNR: Delivery times are predictable by product type
Low SNR: High variability, need more features

R&D / SIFIDE: WP7 - Product Intelligence
────────────────────────────────────────
- Hypothesis H7.2: ML delivery estimates achieve MAPE <10%
- Experiment E7.1: Compare deterministic vs ML on 100 orders
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .product_classification import ProductFingerprint, ProductType

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

# Default buffer multipliers (k * σ)
BUFFER_MULTIPLIERS = {
    'conservative': 2.0,   # ~95% confidence
    'moderate': 1.5,       # ~87% confidence
    'aggressive': 1.0,     # ~68% confidence
}

# Queue time estimation by machine utilization
QUEUE_FACTORS = {
    (0, 0.5): 0.1,      # Low util: 10% of processing as queue
    (0.5, 0.7): 0.25,   # Medium: 25%
    (0.7, 0.85): 0.5,   # High: 50%
    (0.85, 1.0): 1.0,   # Very high: 100%
}

# Working hours per day (for calendar calculations)
WORKING_HOURS_PER_DAY = 8


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════════════════════════════

class EstimationMethod(str, Enum):
    """Delivery estimation method."""
    DETERMINISTIC = "deterministic"
    HISTORICAL = "historical"
    ML = "ml"


@dataclass
class DeliveryConfig:
    """Configuration for delivery estimation."""
    method: EstimationMethod = EstimationMethod.DETERMINISTIC
    buffer_strategy: str = "moderate"
    include_weekends: bool = False
    working_hours_per_day: float = 8.0
    confidence_target: float = 0.85


@dataclass
class DeliveryEstimate:
    """
    Delivery time estimation result.
    """
    order_id: str
    article_id: str
    
    # Estimates
    estimated_duration_hours: float = 0.0
    estimated_delivery_date: Optional[str] = None
    
    # Breakdown
    processing_time_hours: float = 0.0
    setup_time_hours: float = 0.0
    queue_time_hours: float = 0.0
    buffer_time_hours: float = 0.0
    
    # Confidence
    confidence_score: float = 0.5
    confidence_level: str = "MEDIUM"
    
    # SNR
    snr_estimate: float = 1.0
    snr_level: str = "FAIR"
    
    # Metadata
    estimation_method: str = "deterministic"
    estimation_date: str = ""
    
    # Range
    optimistic_hours: Optional[float] = None
    pessimistic_hours: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'article_id': self.article_id,
            'estimated_duration_hours': round(self.estimated_duration_hours, 2),
            'estimated_delivery_date': self.estimated_delivery_date,
            'breakdown': {
                'processing_hours': round(self.processing_time_hours, 2),
                'setup_hours': round(self.setup_time_hours, 2),
                'queue_hours': round(self.queue_time_hours, 2),
                'buffer_hours': round(self.buffer_time_hours, 2),
            },
            'confidence_score': round(self.confidence_score, 3),
            'confidence_level': self.confidence_level,
            'snr_estimate': round(self.snr_estimate, 2),
            'snr_level': self.snr_level,
            'estimation_method': self.estimation_method,
            'optimistic_hours': round(self.optimistic_hours, 2) if self.optimistic_hours else None,
            'pessimistic_hours': round(self.pessimistic_hours, 2) if self.pessimistic_hours else None,
        }


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def _get_queue_factor(utilization: float) -> float:
    """Get queue time factor based on machine utilization."""
    for (low, high), factor in QUEUE_FACTORS.items():
        if low <= utilization < high:
            return factor
    return 1.0


def _hours_to_business_date(
    start_date: datetime,
    hours: float,
    working_hours_per_day: float = 8.0,
    include_weekends: bool = False
) -> datetime:
    """
    Convert hours to business date.
    
    Accounts for working hours per day and optionally weekends.
    """
    days = hours / working_hours_per_day
    remaining_hours = hours % working_hours_per_day
    
    current_date = start_date
    days_added = 0
    
    while days_added < int(days):
        current_date += timedelta(days=1)
        # Skip weekends if not included
        if not include_weekends and current_date.weekday() >= 5:
            continue
        days_added += 1
    
    # Add remaining hours
    current_date += timedelta(hours=remaining_hours)
    
    return current_date


def _classify_confidence(score: float) -> str:
    """Classify confidence score into level."""
    if score >= 0.85:
        return "HIGH"
    elif score >= 0.6:
        return "MEDIUM"
    elif score >= 0.4:
        return "LOW"
    else:
        return "VERY_LOW"


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC ESTIMATION
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def _estimate_deterministic(
    order_id: str,
    article_id: str,
    routing_df: pd.DataFrame,
    machine_loads: Optional[Dict[str, float]] = None,
    config: DeliveryConfig = None,
    article_col: str = 'article_id',
    time_col: str = 'base_time_per_unit_min',
    machine_col: str = 'primary_machine_id',
    qty: float = 1.0,
) -> DeliveryEstimate:
    """
    Deterministic delivery estimation based on routing times.
    """
    config = config or DeliveryConfig()
    
    estimate = DeliveryEstimate(
        order_id=order_id,
        article_id=article_id,
        estimation_method="deterministic",
        estimation_date=datetime.now().isoformat(),
    )
    
    # Get routing for this article
    if article_col not in routing_df.columns:
        estimate.confidence_score = 0.1
        estimate.confidence_level = "VERY_LOW"
        return estimate
    
    article_routing = routing_df[routing_df[article_col] == article_id]
    
    if article_routing.empty:
        estimate.confidence_score = 0.1
        estimate.confidence_level = "VERY_LOW"
        return estimate
    
    # Processing time
    if time_col in article_routing.columns:
        base_time_min = article_routing[time_col].sum()
        estimate.processing_time_hours = (base_time_min * qty) / 60
    
    # Setup time (estimate as 5-15% of processing, depends on type)
    # TODO: Use actual setup matrix when available
    setup_pct = 0.1  # Default 10%
    estimate.setup_time_hours = estimate.processing_time_hours * setup_pct
    
    # Queue time based on machine loads
    if machine_loads and machine_col in article_routing.columns:
        machines = article_routing[machine_col].dropna().unique()
        avg_util = np.mean([machine_loads.get(m, 0.5) for m in machines])
        queue_factor = _get_queue_factor(avg_util)
        estimate.queue_time_hours = estimate.processing_time_hours * queue_factor
    else:
        # Default queue estimate
        estimate.queue_time_hours = estimate.processing_time_hours * 0.25
    
    # Buffer time
    buffer_k = BUFFER_MULTIPLIERS.get(config.buffer_strategy, 1.5)
    # Use coefficient of variation estimate
    cv = 0.2  # Assume 20% CV for deterministic
    sigma = (estimate.processing_time_hours + estimate.setup_time_hours) * cv
    estimate.buffer_time_hours = buffer_k * sigma
    
    # Total
    estimate.estimated_duration_hours = (
        estimate.processing_time_hours +
        estimate.setup_time_hours +
        estimate.queue_time_hours +
        estimate.buffer_time_hours
    )
    
    # Delivery date
    start_date = datetime.now()
    delivery_date = _hours_to_business_date(
        start_date,
        estimate.estimated_duration_hours,
        config.working_hours_per_day,
        config.include_weekends
    )
    estimate.estimated_delivery_date = delivery_date.strftime('%Y-%m-%d %H:%M')
    
    # Confidence (higher if we have good routing data)
    data_coverage = 1.0 if time_col in article_routing.columns else 0.5
    estimate.confidence_score = 0.6 * data_coverage + 0.2  # Base 20% + 60% if data
    estimate.confidence_level = _classify_confidence(estimate.confidence_score)
    
    # Optimistic/pessimistic
    estimate.optimistic_hours = estimate.estimated_duration_hours * 0.7
    estimate.pessimistic_hours = estimate.estimated_duration_hours * 1.5
    
    return estimate


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# HISTORICAL ESTIMATION
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def _estimate_historical(
    order_id: str,
    article_id: str,
    plan_df: pd.DataFrame,
    config: DeliveryConfig = None,
    article_col: str = 'article_id',
    order_col: str = 'order_id',
    start_col: str = 'start_time',
    end_col: str = 'end_time',
) -> DeliveryEstimate:
    """
    Historical delivery estimation based on past lead times.
    """
    config = config or DeliveryConfig()
    
    estimate = DeliveryEstimate(
        order_id=order_id,
        article_id=article_id,
        estimation_method="historical",
        estimation_date=datetime.now().isoformat(),
    )
    
    # Get historical lead times for this article
    if article_col not in plan_df.columns:
        estimate.confidence_score = 0.1
        estimate.confidence_level = "VERY_LOW"
        return estimate
    
    article_ops = plan_df[plan_df[article_col] == article_id]
    
    if article_ops.empty:
        estimate.confidence_score = 0.1
        estimate.confidence_level = "VERY_LOW"
        return estimate
    
    # Compute lead times per order
    lead_times = []
    
    if order_col in article_ops.columns and start_col in article_ops.columns and end_col in article_ops.columns:
        for oid, group in article_ops.groupby(order_col):
            starts = pd.to_datetime(group[start_col], errors='coerce').dropna()
            ends = pd.to_datetime(group[end_col], errors='coerce').dropna()
            
            if not starts.empty and not ends.empty:
                lt_hours = (ends.max() - starts.min()).total_seconds() / 3600
                if lt_hours > 0:
                    lead_times.append(lt_hours)
    
    if not lead_times:
        estimate.confidence_score = 0.2
        estimate.confidence_level = "VERY_LOW"
        return estimate
    
    # Statistical estimates
    mu = np.mean(lead_times)
    sigma = np.std(lead_times) if len(lead_times) > 1 else mu * 0.2
    
    buffer_k = BUFFER_MULTIPLIERS.get(config.buffer_strategy, 1.5)
    
    estimate.estimated_duration_hours = mu + buffer_k * sigma
    estimate.processing_time_hours = mu * 0.6  # Approximate breakdown
    estimate.queue_time_hours = mu * 0.3
    estimate.buffer_time_hours = buffer_k * sigma
    
    # Delivery date
    start_date = datetime.now()
    delivery_date = _hours_to_business_date(
        start_date,
        estimate.estimated_duration_hours,
        config.working_hours_per_day,
        config.include_weekends
    )
    estimate.estimated_delivery_date = delivery_date.strftime('%Y-%m-%d %H:%M')
    
    # Confidence based on sample size and variability
    sample_factor = min(1, len(lead_times) / 10)  # Saturates at 10 samples
    cv = sigma / mu if mu > 0 else 1
    variability_factor = max(0, 1 - cv)
    
    estimate.confidence_score = 0.5 * sample_factor + 0.4 * variability_factor + 0.1
    estimate.confidence_level = _classify_confidence(estimate.confidence_score)
    
    # SNR
    if len(lead_times) >= 5:
        from product_kpi_engine import compute_snr
        snr, level = compute_snr(np.array(lead_times))
        estimate.snr_estimate = snr
        estimate.snr_level = level
    
    # Optimistic/pessimistic
    estimate.optimistic_hours = mu - sigma if mu > sigma else mu * 0.5
    estimate.pessimistic_hours = mu + 2 * sigma
    
    return estimate


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# ML-BASED ESTIMATION (Future)
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def _estimate_ml(
    order_id: str,
    article_id: str,
    features: Dict[str, Any],
    config: DeliveryConfig = None,
) -> DeliveryEstimate:
    """
    TODO[R&D]: ML-based delivery estimation.
    
    Features to consider:
    - product_type: categorical
    - qty: numeric
    - num_operations: numeric
    - total_routing_time: numeric
    - machine_load_avg: numeric
    - day_of_week: categorical
    - month: categorical
    - historical_lead_time_mean: numeric
    - historical_lead_time_std: numeric
    
    Models:
    - MVP: Linear Regression / Ridge
    - Advanced: XGBoost, LightGBM
    - Time-aware: DeepAR, NST (Non-Stationary Transformer)
    
    Training:
    - Use historical order completions
    - Features extracted at order creation time
    - Target: actual lead time
    
    Evaluation:
    - MAPE, RMSE
    - Confidence calibration
    - SNR of predictions vs actuals
    """
    config = config or DeliveryConfig()
    
    estimate = DeliveryEstimate(
        order_id=order_id,
        article_id=article_id,
        estimation_method="ml",
        estimation_date=datetime.now().isoformat(),
    )
    
    # For now, fall back to simple regression approximation
    # TODO: Implement actual ML model
    
    base_hours = features.get('routing_time_hours', 1.0)
    qty = features.get('qty', 1)
    load_factor = features.get('machine_load_avg', 0.5)
    
    # Simple linear model approximation
    # T = β0 + β1*routing + β2*qty + β3*load
    beta = [0.5, 1.0, 0.01, 2.0]  # Placeholder coefficients
    
    estimate.estimated_duration_hours = (
        beta[0] +
        beta[1] * base_hours +
        beta[2] * qty +
        beta[3] * load_factor
    )
    
    estimate.processing_time_hours = base_hours * qty
    estimate.queue_time_hours = estimate.estimated_duration_hours * 0.2
    estimate.buffer_time_hours = estimate.estimated_duration_hours * 0.15
    
    # Delivery date
    start_date = datetime.now()
    delivery_date = _hours_to_business_date(
        start_date,
        estimate.estimated_duration_hours,
        config.working_hours_per_day,
        config.include_weekends
    )
    estimate.estimated_delivery_date = delivery_date.strftime('%Y-%m-%d %H:%M')
    
    # Lower confidence for ML placeholder
    estimate.confidence_score = 0.4
    estimate.confidence_level = "LOW"
    
    logger.warning("ML estimation not fully implemented. Using placeholder model.")
    
    return estimate


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def estimate_delivery_time(
    order_id: str,
    article_id: str,
    routing_df: pd.DataFrame,
    plan_df: Optional[pd.DataFrame] = None,
    machine_loads: Optional[Dict[str, float]] = None,
    qty: float = 1.0,
    config: Optional[DeliveryConfig] = None,
    **kwargs
) -> DeliveryEstimate:
    """
    Estimate delivery time for an order.
    
    Automatically selects best method based on available data.
    
    Args:
        order_id: Order identifier
        article_id: Article identifier
        routing_df: Routing data
        plan_df: Historical production plan (optional, for historical method)
        machine_loads: Current machine utilizations (optional)
        qty: Order quantity
        config: Estimation configuration
    
    Returns:
        DeliveryEstimate with time and confidence
    """
    config = config or DeliveryConfig()
    
    # Try historical first if data available
    if config.method == EstimationMethod.HISTORICAL and plan_df is not None:
        estimate = _estimate_historical(
            order_id=order_id,
            article_id=article_id,
            plan_df=plan_df,
            config=config,
            **kwargs
        )
        if estimate.confidence_score >= 0.3:
            return estimate
    
    # Try ML if configured
    if config.method == EstimationMethod.ML:
        features = {
            'routing_time_hours': routing_df[routing_df['article_id'] == article_id]['base_time_per_unit_min'].sum() / 60 if 'article_id' in routing_df.columns else 1.0,
            'qty': qty,
            'machine_load_avg': np.mean(list(machine_loads.values())) if machine_loads else 0.5,
        }
        return _estimate_ml(order_id, article_id, features, config)
    
    # Default: deterministic
    return _estimate_deterministic(
        order_id=order_id,
        article_id=article_id,
        routing_df=routing_df,
        machine_loads=machine_loads,
        config=config,
        qty=qty,
        **kwargs
    )


def estimate_all_deliveries(
    orders_df: pd.DataFrame,
    routing_df: pd.DataFrame,
    plan_df: Optional[pd.DataFrame] = None,
    machine_loads: Optional[Dict[str, float]] = None,
    config: Optional[DeliveryConfig] = None,
    order_col: str = 'order_id',
    article_col: str = 'article_id',
    qty_col: str = 'qty',
) -> Dict[str, DeliveryEstimate]:
    """
    Estimate delivery times for all orders.
    
    Returns:
        Dict mapping order_id -> DeliveryEstimate
    """
    estimates = {}
    
    if order_col not in orders_df.columns:
        return estimates
    
    for _, row in orders_df.iterrows():
        order_id = str(row[order_col])
        article_id = str(row.get(article_col, 'UNKNOWN'))
        qty = float(row.get(qty_col, 1))
        
        estimate = estimate_delivery_time(
            order_id=order_id,
            article_id=article_id,
            routing_df=routing_df,
            plan_df=plan_df,
            machine_loads=machine_loads,
            qty=qty,
            config=config,
        )
        
        estimates[order_id] = estimate
    
    return estimates



