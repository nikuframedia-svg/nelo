"""
Offline ML helpers (heurísticas) para previsões de carga e lead time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from scheduler import build_plan
from data_loader import load_dataset


@dataclass
class LoadForecastModel:
    trained_at: datetime
    average_minutes: Dict[str, float]


@dataclass
class LeadTimeModel:
    trained_at: datetime
    avg_lead_time_days: float


_LOAD_MODEL: Optional[LoadForecastModel] = None
_LEAD_MODEL: Optional[LeadTimeModel] = None


def train_load_forecast_model(training_data: pd.DataFrame) -> LoadForecastModel:
    """
    Calcula médias simples de minutos por máquina (heurística rápida para MVP).
    """
    if training_data.empty:
        raise ValueError("training_data vazio para load forecast.")
    if "machine_id" not in training_data or "duration_min" not in training_data:
        raise ValueError("training_data requer colunas machine_id e duration_min.")

    grouped = (
        training_data.groupby("machine_id")["duration_min"].mean().to_dict()
    )
    model = LoadForecastModel(trained_at=datetime.utcnow(), average_minutes=grouped)
    global _LOAD_MODEL
    _LOAD_MODEL = model
    return model


def train_lead_time_model(training_data: pd.DataFrame) -> LeadTimeModel:
    """
    Calcula lead time médio (due_date - hoje) apenas como referência.
    """
    if training_data.empty:
        raise ValueError("training_data vazio para lead time.")
    if "due_date" not in training_data:
        raise ValueError("training_data requer coluna due_date.")

    due_dates = pd.to_datetime(training_data["due_date"], errors="coerce").dropna()
    if due_dates.empty:
        raise ValueError("Não foi possível calcular lead time médio (due_date inválido).")

    avg_days = (due_dates - datetime.utcnow()).dt.total_seconds().mean() / 86400
    model = LeadTimeModel(trained_at=datetime.utcnow(), avg_lead_time_days=avg_days)
    global _LEAD_MODEL
    _LEAD_MODEL = model
    return model


def predict_load(context: Dict[str, Any]) -> Dict[str, float]:
    """
    Usa o modelo treinado (ou heurística direta) para prever carga por máquina.
    """
    plan = context.get("plan")
    if plan is None or plan.empty:
        data = context.get("data") or load_dataset()
        plan = build_plan(data)

    if _LOAD_MODEL is None:
        train_load_forecast_model(plan)

    grouped = plan.groupby("machine_id")["duration_min"].sum().to_dict()
    # combinar previsão (médias) com carga atual para dar ideia de trend
    predictions = {}
    for machine, total in grouped.items():
        avg = _LOAD_MODEL.average_minutes.get(machine, total / max(len(plan), 1))
        predictions[machine] = {
            "forecast_minutes": round(avg, 2),
            "scheduled_minutes": round(total, 2),
        }
    return predictions


def predict_lead_time(context: Dict[str, Any]) -> Dict[str, float]:
    """
    Lead time heurístico por ordem (due_date - agora). Se não houver due_date,
    usa o valor médio do modelo.
    """
    orders = context.get("orders")
    if orders is None:
        data = context.get("data") or load_dataset()
        orders = data.orders

    if _LEAD_MODEL is None:
        train_lead_time_model(orders)

    predictions: Dict[str, float] = {}
    now = datetime.utcnow()
    due_dates = pd.to_datetime(orders["due_date"], errors="coerce")

    for order_id, due in zip(orders["order_id"], due_dates):
        if pd.isna(due):
            predictions[str(order_id)] = round(_LEAD_MODEL.avg_lead_time_days, 2)
        else:
            predictions[str(order_id)] = round((due - now).total_seconds() / 86400, 2)
    return predictions


# TODO[ML_ENGINE]: adicionar deteção de anomalias e previsões de throughput (séries temporais).

