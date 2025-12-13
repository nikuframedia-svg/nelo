"""
Conectores ERP/MES simplificados (placeholders) para o MVP.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ..data_loader import load_dataset, as_records
from ..scheduler import build_plan

EXPORT_DIR = Path(__file__).resolve().parents[2] / "data" / "erp_exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_orders_from_erp(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Emulação de fetch ERP: filtra ordens do Excel conforme critérios (datas, prioridade, artigo).
    """
    data = load_dataset()
    orders = data.orders.copy()

    sku = filters.get("article_id")
    if sku:
        orders = orders[orders["article_id"] == sku]

    min_due = filters.get("due_date_from")
    max_due = filters.get("due_date_to")
    if min_due:
        orders = orders[pd.to_datetime(orders["due_date"]) >= pd.to_datetime(min_due)]
    if max_due:
        orders = orders[pd.to_datetime(orders["due_date"]) <= pd.to_datetime(max_due)]

    return as_records(orders)


def push_plan_to_erp(plan_payload: Dict[str, Any]) -> Path:
    """
    Guarda o plano aprovado num JSON pronto a ser importado pelo ERP (simulação).
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    target = EXPORT_DIR / f"plan_export_{timestamp}.json"
    target.write_text(json.dumps(plan_payload, indent=2, ensure_ascii=False))
    return target


def fetch_machine_status_from_mes() -> List[Dict[str, Any]]:
    """
    Emula leitura MES: agrega carga por máquina e devolve estado (livre/ocupado).
    """
    data = load_dataset()
    plan = build_plan(data)
    if plan.empty:
        return []

    now = datetime.utcnow()
    statuses = []
    for machine_id, df_machine in plan.groupby("machine_id"):
        last_end = pd.to_datetime(df_machine["end_time"]).max()
        status = "ocupada" if last_end and last_end > now else "livre"
        statuses.append(
            {
                "machine_id": machine_id,
                "status": status,
                "scheduled_minutes": float(df_machine["duration_min"].sum()),
            }
        )
    return statuses


# TODO[ERP_MES_CONNECTOR]: ligar estes métodos a conectores reais (SQL Server / REST / SOAP).

