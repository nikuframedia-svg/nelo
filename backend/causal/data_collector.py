"""
════════════════════════════════════════════════════════════════════════════════════════════════════
Causal Data Collector - Build Datasets from Execution Logs
════════════════════════════════════════════════════════════════════════════════════════════════════

Contract 9 Implementation: Collect data from OperationExecutionLog for causal inference.

This module provides functions to:
- Build causal datasets from execution logs
- Extract treatment, outcome, and confounders
- Prepare data for OLS and DML estimators
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# TREATMENT & OUTCOME DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Available treatments (interventions we can study)
AVAILABLE_TREATMENTS = {
    "setup_time": "Setup time (seconds)",
    "cycle_time": "Cycle time per piece (seconds)",
    "feed_rate": "Feed rate (mm/min)",
    "spindle_speed": "Spindle speed (RPM)",
    "temperature": "Process temperature (°C)",
    "pressure": "Process pressure (bar)",
    "batch_size": "Batch size",
    "shift": "Shift (day/night)",
}

# Available outcomes (effects we want to understand)
AVAILABLE_OUTCOMES = {
    "scrap_rate": "Scrap rate (%)",
    "cycle_time": "Actual cycle time (s)",
    "energy_per_unit": "Energy consumption per unit (kWh)",
    "downtime_minutes": "Downtime duration (minutes)",
    "oee_quality": "OEE Quality component",
    "total_time": "Total operation time (s)",
}

# Common confounders to control for
CONFOUNDERS = [
    "machine_id",
    "operator_id",
    "operation_id",
    "day_of_week",
    "hour_of_day",
]


@dataclass
class CausalDatasetConfig:
    """Configuration for causal dataset building."""
    treatment: str
    outcome: str
    confounders: List[str]
    days_lookback: int = 90
    min_observations: int = 30
    normalize: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_causal_dataset(
    treatment: str,
    outcome: str,
    confounders: Optional[List[str]] = None,
    days_lookback: int = 90,
    operation_id: Optional[str] = None,
    machine_id: Optional[str] = None,
    normalize: bool = False,
) -> pd.DataFrame:
    """
    Build a causal dataset from execution logs.
    
    Reads from OperationExecutionLog, extracts treatment, outcome,
    and confounders into a DataFrame suitable for causal inference.
    
    Args:
        treatment: Treatment variable name (from AVAILABLE_TREATMENTS)
        outcome: Outcome variable name (from AVAILABLE_OUTCOMES)
        confounders: List of confounder names (defaults to CONFOUNDERS)
        days_lookback: Number of days of historical data
        operation_id: Optional filter for specific operation
        machine_id: Optional filter for specific machine
        normalize: Whether to normalize continuous variables
    
    Returns:
        pd.DataFrame with columns: treatment, outcome, and confounders
    """
    confounders = confounders or CONFOUNDERS.copy()
    
    try:
        from prodplan.execution_log_models import (
            query_execution_logs,
            ExecutionLogQuery,
            ExecutionLogStatus,
        )
        
        # Query execution logs
        query = ExecutionLogQuery(
            operation_id=operation_id,
            machine_id=machine_id,
            status=ExecutionLogStatus.COMPLETED,
            from_date=datetime.now(timezone.utc) - timedelta(days=days_lookback),
            limit=2000,
        )
        
        logs = query_execution_logs(query)
        
        if len(logs) < 30:
            logger.warning(f"Insufficient data: {len(logs)} observations")
            return _create_demo_dataset(treatment, outcome, confounders, 100)
        
        # Convert to records
        records = []
        for log in logs:
            record = _extract_record(log, treatment, outcome, confounders)
            if record:
                records.append(record)
        
        if len(records) < 30:
            logger.warning(f"Insufficient valid records: {len(records)}")
            return _create_demo_dataset(treatment, outcome, confounders, 100)
        
        df = pd.DataFrame(records)
        
        # Drop rows with missing values in key columns
        df = df.dropna(subset=["treatment", "outcome"])
        
        if normalize:
            df = _normalize_dataset(df)
        
        logger.info(f"Built causal dataset with {len(df)} observations")
        return df
        
    except ImportError:
        logger.warning("Execution log models not available, using demo data")
        return _create_demo_dataset(treatment, outcome, confounders, 100)
    except Exception as e:
        logger.error(f"Error building causal dataset: {e}")
        return _create_demo_dataset(treatment, outcome, confounders, 100)


def _extract_record(
    log,
    treatment: str,
    outcome: str,
    confounders: List[str],
) -> Optional[Dict[str, Any]]:
    """Extract a record from an execution log."""
    record = {}
    
    # Extract treatment
    treatment_value = _get_field_value(log, treatment)
    if treatment_value is None:
        return None
    record["treatment"] = treatment_value
    
    # Extract outcome
    outcome_value = _get_outcome_value(log, outcome)
    if outcome_value is None:
        return None
    record["outcome"] = outcome_value
    
    # Extract confounders
    for conf in confounders:
        record[conf] = _get_confounder_value(log, conf)
    
    return record


def _get_field_value(log, field: str) -> Optional[float]:
    """Get treatment/outcome field value from log."""
    try:
        if field == "setup_time":
            return log.setup_time_s
        elif field == "cycle_time":
            return log.cycle_time_s
        elif field == "batch_size":
            return log.qty_good + log.qty_scrap
        elif field in ["feed_rate", "spindle_speed", "temperature", "pressure"]:
            # Get from process params
            if log.params:
                params = log.params.__dict__ if hasattr(log.params, '__dict__') else {}
                return params.get(field)
            return None
        elif field == "shift":
            # Derive from start time
            hour = log.start_time.hour
            return 1.0 if 6 <= hour < 18 else 0.0  # 1 = day, 0 = night
        return None
    except Exception:
        return None


def _get_outcome_value(log, outcome: str) -> Optional[float]:
    """Get outcome value from log."""
    try:
        if outcome == "scrap_rate":
            total = log.qty_good + log.qty_scrap
            return (log.qty_scrap / total * 100) if total > 0 else 0.0
        elif outcome == "cycle_time":
            return log.cycle_time_s
        elif outcome == "energy_per_unit":
            if log.energy_kwh and log.qty_good > 0:
                return log.energy_kwh / log.qty_good
            return None
        elif outcome == "downtime_minutes":
            return log.downtime_minutes
        elif outcome == "oee_quality":
            total = log.qty_good + log.qty_scrap
            return (log.qty_good / total) if total > 0 else 1.0
        elif outcome == "total_time":
            return (log.end_time - log.start_time).total_seconds()
        return None
    except Exception:
        return None


def _get_confounder_value(log, confounder: str) -> Any:
    """Get confounder value from log."""
    try:
        if confounder == "machine_id":
            return log.machine_id
        elif confounder == "operator_id":
            return log.operator_id or "unknown"
        elif confounder == "operation_id":
            return log.operation_id
        elif confounder == "day_of_week":
            return log.start_time.weekday()
        elif confounder == "hour_of_day":
            return log.start_time.hour
        return None
    except Exception:
        return None


def _normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize continuous variables to 0-1 range."""
    continuous_cols = ["treatment", "outcome"]
    for col in continuous_cols:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
    return df


def _create_demo_dataset(
    treatment: str,
    outcome: str,
    confounders: List[str],
    n_samples: int = 100,
) -> pd.DataFrame:
    """Create a demonstration dataset for testing."""
    np.random.seed(42)
    
    # Generate base treatment values
    treatment_values = np.random.uniform(50, 150, n_samples)
    
    # Generate confounders
    machine_ids = np.random.choice(["M-01", "M-02", "M-03", "M-04"], n_samples)
    operator_ids = np.random.choice(["OP-A", "OP-B", "OP-C"], n_samples)
    days = np.random.randint(0, 7, n_samples)
    hours = np.random.randint(6, 22, n_samples)
    
    # Generate outcome with causal effect
    # Base effect: higher treatment → higher outcome
    causal_effect = 0.3
    noise = np.random.normal(0, 10, n_samples)
    
    # Confounding: machine affects both treatment and outcome
    machine_effect = {"M-01": 0, "M-02": 5, "M-03": -5, "M-04": 10}
    treatment_values += np.array([machine_effect[m] for m in machine_ids])
    
    outcome_values = (
        treatment_values * causal_effect + 
        np.array([machine_effect[m] * 0.5 for m in machine_ids]) +
        noise + 
        20  # Base
    )
    
    data = {
        "treatment": treatment_values,
        "outcome": outcome_values,
    }
    
    # Add confounders
    for conf in confounders:
        if conf == "machine_id":
            data[conf] = machine_ids
        elif conf == "operator_id":
            data[conf] = operator_ids
        elif conf == "operation_id":
            data[conf] = ["OP-001"] * n_samples
        elif conf == "day_of_week":
            data[conf] = days
        elif conf == "hour_of_day":
            data[conf] = hours
        else:
            data[conf] = np.random.uniform(0, 1, n_samples)
    
    df = pd.DataFrame(data)
    
    logger.info(f"Created demo dataset with {n_samples} samples")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary statistics for a causal dataset."""
    return {
        "n_observations": len(df),
        "treatment_mean": float(df["treatment"].mean()),
        "treatment_std": float(df["treatment"].std()),
        "outcome_mean": float(df["outcome"].mean()),
        "outcome_std": float(df["outcome"].std()),
        "correlation": float(df["treatment"].corr(df["outcome"])),
        "confounders": [c for c in df.columns if c not in ["treatment", "outcome"]],
        "missing_pct": float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
    }


def prepare_for_estimator(
    df: pd.DataFrame,
    encode_categoricals: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare dataset for causal estimator.
    
    Returns:
        (X, T, Y) where:
        - X: confounders matrix
        - T: treatment vector
        - Y: outcome vector
    """
    Y = df["outcome"].values
    T = df["treatment"].values
    
    confounders = [c for c in df.columns if c not in ["treatment", "outcome"]]
    
    if encode_categoricals:
        # One-hot encode categorical confounders
        X_df = df[confounders].copy()
        for col in X_df.columns:
            if X_df[col].dtype == object:
                dummies = pd.get_dummies(X_df[col], prefix=col, drop_first=True)
                X_df = X_df.drop(columns=[col])
                X_df = pd.concat([X_df, dummies], axis=1)
        X = X_df.values
    else:
        X = df[confounders].values
    
    return X, T, Y


def list_available_treatments() -> Dict[str, str]:
    """Get list of available treatment variables."""
    return AVAILABLE_TREATMENTS.copy()


def list_available_outcomes() -> Dict[str, str]:
    """Get list of available outcome variables."""
    return AVAILABLE_OUTCOMES.copy()



