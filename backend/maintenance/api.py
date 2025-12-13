"""
════════════════════════════════════════════════════════════════════════════════════════════════════
MAINTENANCE API - Endpoints para Gestão de Ordens de Manutenção
════════════════════════════════════════════════════════════════════════════════════════════════════

API REST para:
- CRUD de ordens de manutenção
- Avaliação automática e criação de WOs via PredictiveCare
- Sugestão de janelas de manutenção
- KPIs e estatísticas

R&D / SIFIDE: WP1 - Digital Twin para manutenção preditiva
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
import json

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field

from maintenance.models import (
    MaintenancePriority,
    MaintenanceType,
    MaintenanceStatus,
    WorkOrderSource,
    WorkOrderCreate,
    WorkOrderUpdate,
    WorkOrderResponse,
    MaintenanceWindowSuggestion,
)
from maintenance.predictivecare_bridge import (
    PredictiveCareBridge,
    get_predictivecare_bridge,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/maintenance", tags=["Maintenance"])


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class WorkOrderListResponse(BaseModel):
    """Response for work order list."""
    total: int
    items: List[Dict[str, Any]]
    filters: Dict[str, Any]


class EvaluationResponse(BaseModel):
    """Response for PredictiveCare evaluation."""
    evaluated_machines: int
    work_orders_created: int
    work_orders: List[Dict[str, Any]]


class MaintenanceKPIs(BaseModel):
    """Maintenance KPIs."""
    total_work_orders: int
    open_work_orders: int
    overdue_work_orders: int
    completed_last_30d: int
    mttr_hours: Optional[float]  # Mean Time To Repair
    failures_prevented: int
    predictive_maintenance_rate: float  # % of WOs that are predictive


class CompleteWorkOrderInput(BaseModel):
    """Input for completing a work order."""
    resolution_notes: str = Field(..., description="Description of work performed")
    failure_prevented: bool = Field(True, description="Whether this prevented a failure")
    parts_replaced: List[str] = Field(default_factory=list, description="Parts replaced")
    actual_duration_hours: Optional[float] = Field(None, ge=0)


# ═══════════════════════════════════════════════════════════════════════════════
# WORK ORDER CRUD
# ═══════════════════════════════════════════════════════════════════════════════

# In-memory store for demo (replace with database in production)
_work_orders_store: Dict[int, Dict[str, Any]] = {}
_next_id = 1


@router.get("/workorders", summary="List work orders")
async def list_workorders(
    status: Optional[str] = Query(None, description="Filter by status"),
    machine_id: Optional[str] = Query(None, description="Filter by machine"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    maintenance_type: Optional[str] = Query(None, description="Filter by type"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> WorkOrderListResponse:
    """
    List work orders with optional filters.
    """
    items = list(_work_orders_store.values())
    
    # Apply filters
    if status:
        items = [w for w in items if w.get("status") == status]
    if machine_id:
        items = [w for w in items if w.get("machine_id") == machine_id]
    if priority:
        items = [w for w in items if w.get("priority") == priority]
    if maintenance_type:
        items = [w for w in items if w.get("maintenance_type") == maintenance_type]
    
    # Sort by priority and created_at
    priority_order = {"EMERGENCY": 0, "CRITICAL": 1, "HIGH": 2, "MEDIUM": 3, "LOW": 4}
    items.sort(key=lambda x: (
        priority_order.get(x.get("priority", "MEDIUM"), 3),
        x.get("created_at", ""),
    ))
    
    total = len(items)
    items = items[offset:offset + limit]
    
    return WorkOrderListResponse(
        total=total,
        items=items,
        filters={"status": status, "machine_id": machine_id, "priority": priority},
    )


@router.post("/workorders", summary="Create work order")
async def create_workorder(body: WorkOrderCreate) -> Dict[str, Any]:
    """
    Create a new work order manually.
    """
    global _next_id
    
    now = datetime.now(timezone.utc)
    wo_number = f"WO-{now.strftime('%Y%m%d')}-{_next_id:04d}"
    
    wo = {
        "id": _next_id,
        "work_order_number": wo_number,
        "machine_id": body.machine_id,
        "title": body.title,
        "description": body.description,
        "priority": body.priority.value,
        "maintenance_type": body.maintenance_type.value,
        "status": MaintenanceStatus.OPEN.value,
        "source": WorkOrderSource.MANUAL.value,
        "scheduled_start": body.scheduled_start.isoformat() if body.scheduled_start else None,
        "scheduled_end": body.scheduled_end.isoformat() if body.scheduled_end else None,
        "estimated_duration_hours": body.estimated_duration_hours,
        "assigned_technician_id": body.assigned_technician_id,
        "spare_parts": body.spare_parts,
        "spare_parts_ready": False,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    
    _work_orders_store[_next_id] = wo
    _next_id += 1
    
    logger.info(f"Created work order {wo_number}")
    return wo


@router.get("/workorders/{work_order_id}", summary="Get work order details")
async def get_workorder(work_order_id: int) -> Dict[str, Any]:
    """
    Get details of a specific work order.
    """
    if work_order_id not in _work_orders_store:
        raise HTTPException(status_code=404, detail="Work order not found")
    
    return _work_orders_store[work_order_id]


@router.patch("/workorders/{work_order_id}", summary="Update work order")
async def update_workorder(
    work_order_id: int,
    body: WorkOrderUpdate,
) -> Dict[str, Any]:
    """
    Update a work order.
    """
    if work_order_id not in _work_orders_store:
        raise HTTPException(status_code=404, detail="Work order not found")
    
    wo = _work_orders_store[work_order_id]
    
    update_data = body.dict(exclude_unset=True)
    for key, value in update_data.items():
        if value is not None:
            if isinstance(value, datetime):
                wo[key] = value.isoformat()
            elif hasattr(value, 'value'):  # Enum
                wo[key] = value.value
            else:
                wo[key] = value
    
    wo["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    logger.info(f"Updated work order {wo['work_order_number']}")
    return wo


@router.delete("/workorders/{work_order_id}", summary="Delete work order")
async def delete_workorder(work_order_id: int) -> Dict[str, str]:
    """
    Delete a work order (soft delete - marks as cancelled).
    """
    if work_order_id not in _work_orders_store:
        raise HTTPException(status_code=404, detail="Work order not found")
    
    wo = _work_orders_store[work_order_id]
    wo["status"] = MaintenanceStatus.CANCELLED.value
    wo["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    return {"status": "cancelled", "work_order_number": wo["work_order_number"]}


@router.post("/workorders/{work_order_id}/complete", summary="Complete work order")
async def complete_workorder(
    work_order_id: int,
    body: CompleteWorkOrderInput,
) -> Dict[str, Any]:
    """
    Mark a work order as completed.
    """
    if work_order_id not in _work_orders_store:
        raise HTTPException(status_code=404, detail="Work order not found")
    
    wo = _work_orders_store[work_order_id]
    now = datetime.now(timezone.utc)
    
    wo["status"] = MaintenanceStatus.COMPLETED.value
    wo["actual_end"] = now.isoformat()
    wo["resolution_notes"] = body.resolution_notes
    wo["failure_prevented"] = body.failure_prevented
    wo["parts_replaced"] = body.parts_replaced
    
    if body.actual_duration_hours:
        wo["actual_duration_hours"] = body.actual_duration_hours
    
    wo["updated_at"] = now.isoformat()
    
    logger.info(f"Completed work order {wo['work_order_number']}")
    return wo


@router.post("/workorders/{work_order_id}/start", summary="Start work order")
async def start_workorder(work_order_id: int) -> Dict[str, Any]:
    """
    Mark a work order as in progress.
    """
    if work_order_id not in _work_orders_store:
        raise HTTPException(status_code=404, detail="Work order not found")
    
    wo = _work_orders_store[work_order_id]
    now = datetime.now(timezone.utc)
    
    wo["status"] = MaintenanceStatus.IN_PROGRESS.value
    wo["actual_start"] = now.isoformat()
    wo["updated_at"] = now.isoformat()
    
    return wo


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTIVECARE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/predictivecare/evaluate", summary="Evaluate machines and create WOs")
async def evaluate_and_create(
    bridge: PredictiveCareBridge = Depends(get_predictivecare_bridge),
) -> EvaluationResponse:
    """
    Evaluate all machines via PredictiveCare and automatically
    create work orders for machines requiring attention.
    """
    global _next_id
    
    created_orders = bridge.evaluate_and_create_workorders()
    
    # Store in our in-memory store
    for wo_data in created_orders:
        if "id" not in wo_data:
            wo_data["id"] = _next_id
            _next_id += 1
        _work_orders_store[wo_data["id"]] = wo_data
    
    return EvaluationResponse(
        evaluated_machines=10,  # TODO: get actual count
        work_orders_created=len(created_orders),
        work_orders=created_orders,
    )


@router.get("/predictivecare/suggest-window/{machine_id}", summary="Suggest maintenance window")
async def suggest_window(
    machine_id: str,
    horizon_days: int = Query(7, ge=1, le=30),
    bridge: PredictiveCareBridge = Depends(get_predictivecare_bridge),
) -> Dict[str, Any]:
    """
    Get suggested maintenance window for a machine.
    
    Considers production plan and RUL to find optimal timing.
    """
    suggestion = bridge.suggest_maintenance_window(machine_id, horizon_days)
    
    if suggestion is None:
        raise HTTPException(status_code=404, detail="Could not suggest window")
    
    # Convert datetime objects to strings
    if "window_start" in suggestion and suggestion["window_start"]:
        suggestion["window_start"] = suggestion["window_start"].isoformat()
    if "window_end" in suggestion and suggestion["window_end"]:
        suggestion["window_end"] = suggestion["window_end"].isoformat()
    if "alternative_windows" in suggestion:
        for alt in suggestion["alternative_windows"]:
            if "start" in alt and alt["start"]:
                alt["start"] = alt["start"].isoformat()
            if "end" in alt and alt["end"]:
                alt["end"] = alt["end"].isoformat()
    
    return suggestion


# ═══════════════════════════════════════════════════════════════════════════════
# KPIs AND STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/kpis", summary="Get maintenance KPIs")
async def get_kpis() -> MaintenanceKPIs:
    """
    Get maintenance KPIs and statistics.
    """
    items = list(_work_orders_store.values())
    now = datetime.now(timezone.utc)
    thirty_days_ago = now - timedelta(days=30)
    
    open_statuses = [
        MaintenanceStatus.OPEN.value,
        MaintenanceStatus.PLANNED.value,
        MaintenanceStatus.IN_PROGRESS.value,
    ]
    
    total = len(items)
    open_count = len([w for w in items if w.get("status") in open_statuses])
    overdue = len([
        w for w in items
        if w.get("status") in open_statuses
        and w.get("scheduled_end")
        and datetime.fromisoformat(w["scheduled_end"]) < now
    ])
    
    completed_recent = [
        w for w in items
        if w.get("status") == MaintenanceStatus.COMPLETED.value
        and w.get("actual_end")
        and datetime.fromisoformat(w["actual_end"]) > thirty_days_ago
    ]
    
    # Calculate MTTR
    durations = [w.get("actual_duration_hours", 0) for w in completed_recent if w.get("actual_duration_hours")]
    mttr = sum(durations) / len(durations) if durations else None
    
    # Failures prevented
    failures_prevented = len([w for w in completed_recent if w.get("failure_prevented", False)])
    
    # Predictive rate
    predictive_count = len([w for w in items if w.get("maintenance_type") == MaintenanceType.PREDICTIVE.value])
    predictive_rate = predictive_count / total if total > 0 else 0
    
    return MaintenanceKPIs(
        total_work_orders=total,
        open_work_orders=open_count,
        overdue_work_orders=overdue,
        completed_last_30d=len(completed_recent),
        mttr_hours=round(mttr, 1) if mttr else None,
        failures_prevented=failures_prevented,
        predictive_maintenance_rate=round(predictive_rate, 2),
    )


@router.get("/schedule", summary="Get maintenance schedule")
async def get_schedule(
    days: int = Query(7, ge=1, le=90),
) -> Dict[str, Any]:
    """
    Get maintenance schedule for the next N days.
    """
    now = datetime.now(timezone.utc)
    end_date = now + timedelta(days=days)
    
    items = list(_work_orders_store.values())
    
    scheduled = [
        w for w in items
        if w.get("status") in [MaintenanceStatus.PLANNED.value, MaintenanceStatus.OPEN.value]
        and (
            (w.get("scheduled_start") and datetime.fromisoformat(w["scheduled_start"]) <= end_date) or
            (w.get("suggested_start") and datetime.fromisoformat(w["suggested_start"]) <= end_date)
        )
    ]
    
    # Group by date
    by_date: Dict[str, List[Dict]] = {}
    for w in scheduled:
        date_str = None
        if w.get("scheduled_start"):
            date_str = datetime.fromisoformat(w["scheduled_start"]).date().isoformat()
        elif w.get("suggested_start"):
            date_str = datetime.fromisoformat(w["suggested_start"]).date().isoformat()
        
        if date_str:
            if date_str not in by_date:
                by_date[date_str] = []
            by_date[date_str].append({
                "id": w["id"],
                "work_order_number": w["work_order_number"],
                "machine_id": w["machine_id"],
                "title": w["title"],
                "priority": w["priority"],
                "status": w["status"],
            })
    
    return {
        "start_date": now.date().isoformat(),
        "end_date": end_date.date().isoformat(),
        "total_scheduled": len(scheduled),
        "by_date": by_date,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND REFERENCE DATA
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/reference/priorities", summary="List priorities")
async def list_priorities() -> List[str]:
    """List available priority levels."""
    return [p.value for p in MaintenancePriority]


@router.get("/reference/types", summary="List maintenance types")
async def list_types() -> List[str]:
    """List available maintenance types."""
    return [t.value for t in MaintenanceType]


@router.get("/reference/statuses", summary="List statuses")
async def list_statuses() -> List[str]:
    """List available work order statuses."""
    return [s.value for s in MaintenanceStatus]


