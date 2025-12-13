"""
════════════════════════════════════════════════════════════════════════════════════════════════════
PREVENTION GUARD API - REST Endpoints for Error Prevention
════════════════════════════════════════════════════════════════════════════════════════════════════

Endpoints for prevention guard operations:
- POST /guard/validate/product-release - Validate product for release
- POST /guard/validate/order-start - Validate before starting order
- POST /guard/predict-risk - Predict defect risk
- POST /guard/exceptions - Request/manage exceptions
- GET /guard/rules - List validation rules
- GET /guard/events - Guard event log

R&D / SIFIDE: WP4 - Zero Defect Manufacturing
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

from .prevention_guard import (
    PreventionGuardService,
    ValidationRule,
    ValidationResult,
    RiskPrediction,
    ExceptionRequest,
    GuardEvent,
    ValidationCategory,
    ValidationSeverity,
    ValidationAction,
    RiskLevel,
    ExceptionStatus,
    get_prevention_guard_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/guard", tags=["Prevention Guard"])


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class BomComponentInput(BaseModel):
    """BOM component input."""
    component_id: str
    qty_per_unit: float = Field(1, gt=0)
    status: str = "active"


class RoutingOperationInput(BaseModel):
    """Routing operation input."""
    operation_id: str
    setup_time: float = Field(0, ge=0)
    cycle_time: float = Field(0, ge=0)
    work_center_id: Optional[str] = None
    machine_id: Optional[str] = None


class AttachmentInput(BaseModel):
    """Attachment input."""
    attachment_id: str
    type: str  # drawing, cad, work_instruction, quality_plan


class ProductReleaseValidationInput(BaseModel):
    """Input for product release validation."""
    item_id: str
    revision: str = "A"
    product_type: str = "standard"
    bom_components: List[BomComponentInput] = Field(default_factory=list)
    routing_operations: List[RoutingOperationInput] = Field(default_factory=list)
    attachments: List[AttachmentInput] = Field(default_factory=list)


class MaterialInput(BaseModel):
    """Material input."""
    sku: str
    revision: Optional[str] = None
    batch_id: Optional[str] = None
    expiry_date: Optional[str] = None


class MachineInput(BaseModel):
    """Machine input."""
    machine_id: str
    health_index: float = Field(0.8, ge=0, le=1)


class ToolInput(BaseModel):
    """Tool input."""
    tool_id: str
    calibrated: bool = True


class ContextInput(BaseModel):
    """Context input for shopfloor validation."""
    machine_id: str
    operator_id: str
    shift: int = Field(1, ge=1, le=3)
    operator_experience: float = Field(0.5, ge=0, le=1)
    machine_health: float = Field(0.8, ge=0, le=1)
    material_batch: str = ""
    temperature: float = 20
    humidity: float = 50
    tools: List[ToolInput] = Field(default_factory=list)


class OrderStartValidationInput(BaseModel):
    """Input for order start validation."""
    order_id: str
    product_id: str
    quantity: float = Field(1, gt=0)
    operation_id: Optional[str] = None
    scanned_materials: List[MaterialInput] = Field(default_factory=list)
    required_materials: List[MaterialInput] = Field(default_factory=list)
    machine: MachineInput
    context: ContextInput


class RiskPredictionInput(BaseModel):
    """Input for risk prediction."""
    order_id: str
    product_id: str
    quantity: float = 1
    context: ContextInput


class ExceptionRequestInput(BaseModel):
    """Input for exception request."""
    issue_id: str
    order_id: str
    operation_id: str = ""
    requested_by: str
    reason: str


class ExceptionResolutionInput(BaseModel):
    """Input for exception resolution."""
    resolved_by: str
    note: str = ""


class ValidationRuleInput(BaseModel):
    """Input for custom validation rule."""
    rule_id: str
    name: str
    description: str = ""
    category: str
    severity: str = "warning"
    action: str = "warn"
    condition: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class HistoricalDataInput(BaseModel):
    """Input for adding historical data."""
    order_id: str
    product_id: str
    machine_id: str
    operator_id: str
    shift: int = 1
    had_defect: bool
    defect_type: Optional[str] = None
    defect_cause: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/status")
async def get_status():
    """Get prevention guard status."""
    service = get_prevention_guard_service()
    stats = service.get_statistics()
    
    return {
        "service": "Prevention Guard",
        "version": "1.0.0",
        "status": "operational",
        "statistics": stats,
        "categories": [c.value for c in ValidationCategory],
        "severities": [s.value for s in ValidationSeverity],
        "actions": [a.value for a in ValidationAction],
        "risk_levels": [r.value for r in RiskLevel],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/validate/product-release")
async def validate_product_release(data: ProductReleaseValidationInput):
    """
    Validate a product for release.
    
    Checks BOM, routing, and documentation.
    """
    service = get_prevention_guard_service()
    
    item_data = {
        "item_id": data.item_id,
        "revision": data.revision,
        "product_type": data.product_type,
    }
    
    bom_components = [
        {
            "component_id": c.component_id,
            "qty_per_unit": c.qty_per_unit,
            "status": c.status,
        }
        for c in data.bom_components
    ]
    
    routing_operations = [
        {
            "operation_id": op.operation_id,
            "setup_time": op.setup_time,
            "cycle_time": op.cycle_time,
            "work_center_id": op.work_center_id,
            "machine_id": op.machine_id,
        }
        for op in data.routing_operations
    ]
    
    attachments = [
        {"attachment_id": a.attachment_id, "type": a.type}
        for a in data.attachments
    ]
    
    result = service.validate_product_release(
        item_data, bom_components, routing_operations, attachments
    )
    
    return result.to_dict()


@router.post("/validate/order-start")
async def validate_order_start(data: OrderStartValidationInput):
    """
    Validate before starting a production order.
    
    Checks materials, equipment, and predicts risk.
    """
    service = get_prevention_guard_service()
    
    order_data = {
        "order_id": data.order_id,
        "product_id": data.product_id,
        "quantity": data.quantity,
        "operation_id": data.operation_id,
    }
    
    scanned_materials = [
        {"sku": m.sku, "revision": m.revision, "expiry_date": m.expiry_date}
        for m in data.scanned_materials
    ]
    
    required_materials = [
        {"sku": m.sku, "revision": m.revision}
        for m in data.required_materials
    ]
    
    machine_data = {
        "machine_id": data.machine.machine_id,
        "health_index": data.machine.health_index,
    }
    
    context = {
        "machine_id": data.context.machine_id,
        "operator_id": data.context.operator_id,
        "shift": data.context.shift,
        "operator_experience": data.context.operator_experience,
        "machine_health": data.context.machine_health,
        "material_batch": data.context.material_batch,
        "temperature": data.context.temperature,
        "humidity": data.context.humidity,
        "tools": [{"tool_id": t.tool_id, "calibrated": t.calibrated} for t in data.context.tools],
    }
    
    validation_result, risk_prediction = service.validate_order_start(
        order_data, scanned_materials, required_materials, machine_data, context
    )
    
    return {
        "validation": validation_result.to_dict(),
        "risk_prediction": risk_prediction.to_dict(),
    }


@router.post("/predict-risk")
async def predict_risk(data: RiskPredictionInput):
    """Predict defect risk for an order."""
    service = get_prevention_guard_service()
    
    order_data = {
        "order_id": data.order_id,
        "product_id": data.product_id,
        "quantity": data.quantity,
    }
    
    context = {
        "machine_id": data.context.machine_id,
        "operator_id": data.context.operator_id,
        "shift": data.context.shift,
        "operator_experience": data.context.operator_experience,
        "machine_health": data.context.machine_health,
        "material_batch": data.context.material_batch,
        "temperature": data.context.temperature,
        "humidity": data.context.humidity,
    }
    
    risk = service.predictive_guard.predict_risk(order_data, context)
    
    return risk.to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/exceptions")
async def request_exception(data: ExceptionRequestInput):
    """Request an exception for a blocked validation."""
    service = get_prevention_guard_service()
    
    exception = service.request_exception(
        issue_id=data.issue_id,
        order_id=data.order_id,
        operation_id=data.operation_id,
        requested_by=data.requested_by,
        reason=data.reason,
    )
    
    return exception.to_dict()


@router.get("/exceptions")
async def list_exceptions(
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
):
    """List exception requests."""
    service = get_prevention_guard_service()
    
    exceptions = list(service.exception_manager.exceptions.values())
    
    if status:
        exceptions = [e for e in exceptions if e.status.value == status]
    
    exceptions.sort(key=lambda e: e.requested_at, reverse=True)
    
    return {
        "total": len(exceptions),
        "exceptions": [e.to_dict() for e in exceptions[:limit]],
    }


@router.get("/exceptions/pending")
async def get_pending_exceptions():
    """Get pending exceptions awaiting approval."""
    service = get_prevention_guard_service()
    
    pending = service.exception_manager.get_pending_exceptions()
    
    return {
        "total": len(pending),
        "exceptions": [e.to_dict() for e in pending],
    }


@router.post("/exceptions/{exception_id}/approve")
async def approve_exception(exception_id: str, data: ExceptionResolutionInput):
    """Approve an exception request."""
    service = get_prevention_guard_service()
    
    success, message = service.approve_exception(
        exception_id, data.resolved_by, data.note
    )
    
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    exception = service.exception_manager.exceptions.get(exception_id)
    
    return {
        "success": True,
        "message": message,
        "exception": exception.to_dict() if exception else None,
    }


@router.post("/exceptions/{exception_id}/reject")
async def reject_exception(exception_id: str, data: ExceptionResolutionInput):
    """Reject an exception request."""
    service = get_prevention_guard_service()
    
    success, message = service.exception_manager.reject_exception(
        exception_id, data.resolved_by, data.note
    )
    
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    exception = service.exception_manager.exceptions.get(exception_id)
    
    return {
        "success": True,
        "message": message,
        "exception": exception.to_dict() if exception else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RULES ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/rules")
async def list_rules(category: Optional[str] = None):
    """List all validation rules."""
    service = get_prevention_guard_service()
    
    # Combine rules from both engines
    all_rules = {}
    all_rules.update(service.pdm_guard.rules)
    all_rules.update(service.shopfloor_guard.rules)
    
    rules = list(all_rules.values())
    
    if category:
        rules = [r for r in rules if r.category.value == category]
    
    return {
        "total": len(rules),
        "rules": [r.to_dict() for r in rules],
    }


@router.post("/rules")
async def add_custom_rule(data: ValidationRuleInput):
    """Add a custom validation rule."""
    service = get_prevention_guard_service()
    
    try:
        category = ValidationCategory(data.category)
        severity = ValidationSeverity(data.severity)
        action = ValidationAction(data.action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    rule = ValidationRule(
        rule_id=data.rule_id,
        name=data.name,
        description=data.description,
        category=category,
        severity=severity,
        action=action,
        condition=data.condition,
        parameters=data.parameters,
        enabled=data.enabled,
    )
    
    # Add to appropriate engine
    if category in [ValidationCategory.BOM, ValidationCategory.ROUTING, ValidationCategory.DOCUMENTATION]:
        service.pdm_guard.add_rule(rule)
    else:
        service.shopfloor_guard.rules[rule.rule_id] = rule
    
    return rule.to_dict()


@router.patch("/rules/{rule_id}/toggle")
async def toggle_rule(rule_id: str, enabled: bool = True):
    """Enable or disable a rule."""
    service = get_prevention_guard_service()
    
    # Find rule
    if rule_id in service.pdm_guard.rules:
        service.pdm_guard.rules[rule_id].enabled = enabled
        return {"success": True, "rule_id": rule_id, "enabled": enabled}
    elif rule_id in service.shopfloor_guard.rules:
        service.shopfloor_guard.rules[rule_id].enabled = enabled
        return {"success": True, "rule_id": rule_id, "enabled": enabled}
    else:
        raise HTTPException(status_code=404, detail="Rule not found")


# ═══════════════════════════════════════════════════════════════════════════════
# EVENTS & STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/events")
async def list_events(
    event_type: Optional[str] = None,
    entity_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
):
    """List guard events."""
    service = get_prevention_guard_service()
    
    events = list(service.events)
    
    if event_type:
        events = [e for e in events if e.event_type.value == event_type]
    
    if entity_type:
        events = [e for e in events if e.entity_type == entity_type]
    
    events.sort(key=lambda e: e.timestamp, reverse=True)
    
    return {
        "total": len(events),
        "events": [e.to_dict() for e in events[:limit]],
    }


@router.get("/statistics")
async def get_statistics():
    """Get guard statistics."""
    service = get_prevention_guard_service()
    
    return service.get_statistics()


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING & DEMO
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/training/add-data")
async def add_training_data(data: HistoricalDataInput):
    """Add historical data for predictive model training."""
    service = get_prevention_guard_service()
    
    order_data = {
        "order_id": data.order_id,
        "product_id": data.product_id,
    }
    
    context = {
        "machine_id": data.machine_id,
        "operator_id": data.operator_id,
        "shift": data.shift,
    }
    
    defect_details = None
    if data.had_defect:
        defect_details = {
            "type": data.defect_type,
            "cause": data.defect_cause,
        }
    
    service.predictive_guard.add_historical_data(
        order_data, context, data.had_defect, defect_details
    )
    
    return {
        "success": True,
        "training_samples": len(service.predictive_guard.training_data),
    }


@router.post("/training/train")
async def train_model():
    """Train the predictive model."""
    service = get_prevention_guard_service()
    
    result = service.predictive_guard.train()
    
    return result


@router.post("/demo")
async def run_demo():
    """Run demo validation scenarios."""
    service = get_prevention_guard_service()
    
    results = {}
    
    # Demo 1: Product release with issues
    product_result = service.validate_product_release(
        item_data={"item_id": "PROD-001", "revision": "A"},
        bom_components=[
            {"component_id": "COMP-001", "qty_per_unit": 2, "status": "active"},
            {"component_id": "COMP-002", "qty_per_unit": 0, "status": "active"},  # Zero qty - error
            {"component_id": "COMP-003", "qty_per_unit": 1, "status": "obsolete"},  # Obsolete - warning
        ],
        routing_operations=[
            {"operation_id": "OP-10", "setup_time": 15, "cycle_time": 30, "work_center_id": "WC-01"},
            {"operation_id": "OP-20", "setup_time": 0, "cycle_time": 0},  # No times - error
        ],
        attachments=[
            {"attachment_id": "DWG-001", "type": "drawing"},
        ],
    )
    results["product_release"] = product_result.to_dict()
    
    # Demo 2: Order start with material mismatch
    order_result, risk_result = service.validate_order_start(
        order_data={"order_id": "OP-2024-001", "product_id": "PROD-001", "quantity": 50},
        scanned_materials=[
            {"sku": "MAT-002", "revision": "B"},  # Wrong material
        ],
        required_materials=[
            {"sku": "MAT-001", "revision": "A"},
        ],
        machine_data={"machine_id": "MC-01", "health_index": 0.55},  # Low health
        context={
            "machine_id": "MC-01",
            "operator_id": "OP-001",
            "shift": 3,  # Night shift
            "operator_experience": 0.3,  # Inexperienced
            "machine_health": 0.55,
            "material_batch": "MB-001",
            "temperature": 25,
            "humidity": 60,
            "tools": [
                {"tool_id": "TOOL-001", "calibrated": True},
                {"tool_id": "TOOL-002", "calibrated": False},  # Not calibrated
            ],
        },
    )
    results["order_start"] = {
        "validation": order_result.to_dict(),
        "risk": risk_result.to_dict(),
    }
    
    return {
        "demo": True,
        "results": results,
        "summary": {
            "product_release_passed": product_result.passed,
            "order_start_passed": order_result.passed,
            "risk_level": risk_result.risk_level.value,
            "total_issues": len(product_result.issues) + len(order_result.issues),
        },
    }



