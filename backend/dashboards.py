"""
Geradores de dados para dashboards (Gantt comparativo, heatmap, projeções).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from data_loader import as_records


def build_gantt_comparison(
    baseline_plan: pd.DataFrame,
    scenario_plan: pd.DataFrame,
    *,
    max_operations: int = 200,
) -> Dict[str, Any]:
    """
    Produz dados para um Gantt comparando baseline vs cenário.
    """
    return {
        "baseline": {
            "total_operations": int(len(baseline_plan)),
            "operations": as_records(baseline_plan.head(max_operations)),
        },
        "scenario": {
            "total_operations": int(len(scenario_plan)),
            "operations": as_records(scenario_plan.head(max_operations)),
        },
    }


def build_heatmap_machine_load(plan: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Calcula carga total por máquina para alimentar heatmaps.
    """
    if plan.empty:
        return []
    load = (
        plan.groupby("machine_id")["duration_min"].sum().reset_index().rename(columns={"duration_min": "total_minutes"})
    )
    total = load["total_minutes"].sum() or 1.0
    load["load_share"] = load["total_minutes"] / total
    return load.to_dict(orient="records")


def build_annual_projection(plan: pd.DataFrame, ml_predictions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combina dados do plano com previsões (ML) para KPIs anuais simplificados.
    """
    total_minutes = float(plan["duration_min"].sum()) if not plan.empty else 0.0
    total_orders = int(plan["order_id"].nunique()) if not plan.empty else 0
    predicted_load = ml_predictions.get("load") if isinstance(ml_predictions, dict) else {}

    return {
        "scheduled_hours": round(total_minutes / 60.0, 2),
        "orders_planned": total_orders,
        "predicted_load": predicted_load,
    }


# TODO[DASHBOARDS]: adicionar drill-down (operadores, cadeias, mapas de impacto).

