"""
════════════════════════════════════════════════════════════════════════════════════════════════════
MRP Complete API - Material Requirements Planning REST Endpoints
════════════════════════════════════════════════════════════════════════════════════════════════════

Endpoints for MRP operations:
- POST /mrp/run - Execute MRP run
- GET /mrp/runs - List MRP runs
- GET /mrp/runs/{id} - Get run details
- POST /mrp/demands - Add demand entries
- GET /mrp/item-plans/{sku} - Get item plan
- GET /mrp/planned-orders - List planned orders
- POST /mrp/demo - Run demo MRP

R&D / SIFIDE: WP3 - Inventory & Capacity Optimization
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

from .mrp_complete import (
    MRPConfig,
    MRPCompleteEngine,
    MRPService,
    MRPRunResult,
    ItemMRPParameters,
    DemandEntry,
    InventoryPosition,
    BomComponent,
    PlannedOrder,
    ItemSource,
    OrderSource,
    PlannedOrderType,
    PlannedOrderStatus,
    get_mrp_service,
    reset_mrp_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mrp", tags=["MRP"])


@router.get("/status")
async def get_mrp_status():
    """Get MRP module status."""
    service = get_mrp_service()
    runs_count = len(service.runs) if hasattr(service, 'runs') else 0
    return {
        "service": "MRP - Material Requirements Planning",
        "version": "2.0.0",
        "status": "operational",
        "runs_count": runs_count,
        "features": ["multi_level_bom", "lot_sizing", "capacity_check", "forecast_integration"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# IN-MEMORY STORAGE
# ═══════════════════════════════════════════════════════════════════════════════

_mrp_runs: Dict[str, MRPRunResult] = {}


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ItemParametersInput(BaseModel):
    """Item MRP parameters input."""
    item_id: int
    sku: str
    name: str = ""
    source: str = Field("manufactured", description="manufactured, purchased, mixed")
    safety_stock: float = Field(0, ge=0)
    min_stock: float = Field(0, ge=0)
    max_stock: float = Field(float('inf'), ge=0)
    moq: float = Field(1, ge=0)
    multiple: float = Field(1, ge=1)
    scrap_rate: float = Field(0, ge=0, le=1)
    lead_time_days: float = Field(7, ge=0)
    unit: str = "pcs"


class DemandInput(BaseModel):
    """Demand entry input."""
    demand_id: Optional[str] = None
    item_id: int
    sku: str
    quantity: float = Field(..., gt=0)
    due_date: str  # ISO format
    source: str = Field("sales_order", description="sales_order, forecast, manual")
    priority: int = Field(1, ge=1, le=10)
    reference_id: Optional[str] = None


class InventoryInput(BaseModel):
    """Inventory position input."""
    item_id: int
    sku: str
    on_hand: float = Field(0, ge=0)
    allocated: float = Field(0, ge=0)
    on_order: float = Field(0, ge=0)


class BomInput(BaseModel):
    """BOM structure input."""
    parent_item_id: int
    components: List[Dict[str, Any]]


class MRPRunRequest(BaseModel):
    """MRP run request."""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    horizon_days: int = Field(90, ge=7, le=365)
    period_days: int = Field(7, ge=1, le=30)
    
    # Optional: Include data inline
    item_parameters: Optional[List[ItemParametersInput]] = None
    demands: Optional[List[DemandInput]] = None
    inventory: Optional[List[InventoryInput]] = None
    bom_structure: Optional[List[BomInput]] = None
    
    load_from_pdm: bool = Field(True, description="Load BOM from PDM module")


class MRPDemoRequest(BaseModel):
    """Demo MRP request."""
    num_products: int = Field(3, ge=1, le=10)
    num_orders: int = Field(5, ge=1, le=20)
    horizon_days: int = Field(90, ge=7, le=180)


class PlannedOrderResponse(BaseModel):
    """Planned order response."""
    order_id: str
    item_id: int
    sku: str
    order_type: str
    status: str
    quantity: float
    start_date: str
    due_date: str
    lead_time_days: float


class MRPRunSummary(BaseModel):
    """MRP run summary."""
    run_id: str
    run_timestamp: str
    items_processed: int
    demands_processed: int
    purchase_orders_count: int
    manufacture_orders_count: int
    shortage_alerts_count: int
    capacity_alerts_count: int


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO date string."""
    if not date_str:
        return None
    return datetime.fromisoformat(date_str.replace("Z", "+00:00"))


def _source_str_to_enum(source: str) -> ItemSource:
    """Convert source string to enum."""
    mapping = {
        "manufactured": ItemSource.MANUFACTURED,
        "purchased": ItemSource.PURCHASED,
        "mixed": ItemSource.MIXED,
    }
    return mapping.get(source.lower(), ItemSource.MANUFACTURED)


def _order_source_to_enum(source: str) -> OrderSource:
    """Convert order source string to enum."""
    mapping = {
        "sales_order": OrderSource.SALES_ORDER,
        "forecast": OrderSource.FORECAST,
        "manual": OrderSource.MANUAL,
        "dependent_demand": OrderSource.DEPENDENT_DEMAND,
        "safety_stock": OrderSource.SAFETY_STOCK,
    }
    return mapping.get(source.lower(), OrderSource.SALES_ORDER)


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/status")
async def get_mrp_status():
    """Get MRP module status."""
    service = get_mrp_service()
    
    return {
        "service": "MRP - Material Requirements Planning",
        "version": "2.0.0",
        "status": "operational",
        "config": {
            "horizon_days": service.config.horizon_days,
            "period_days": service.config.period_days,
            "enable_forecast": service.config.enable_forecast,
            "enable_capacity_check": service.config.enable_capacity_check,
        },
        "runs_stored": len(_mrp_runs),
        "item_sources": [s.value for s in ItemSource],
        "order_sources": [s.value for s in OrderSource],
    }


@router.post("/run")
async def run_mrp(request: MRPRunRequest = Body(...)):
    """
    Execute MRP run.
    
    Can load data from:
    1. PDM module (if load_from_pdm=True)
    2. Inline data in request body
    3. Previously loaded data in service
    """
    try:
        # Create new engine with config
        config = MRPConfig(
            horizon_days=request.horizon_days,
            period_days=request.period_days,
        )
        engine = MRPCompleteEngine(config)
        
        # Load from PDM if requested
        if request.load_from_pdm:
            try:
                from duplios.models import SessionLocal
                db = SessionLocal()
                engine.load_bom_from_pdm(db)
                db.close()
            except Exception as e:
                logger.warning(f"Could not load from PDM: {e}")
        
        # Load inline data
        if request.item_parameters:
            for p in request.item_parameters:
                engine.set_item_parameter(ItemMRPParameters(
                    item_id=p.item_id,
                    sku=p.sku,
                    name=p.name,
                    source=_source_str_to_enum(p.source),
                    safety_stock=p.safety_stock,
                    min_stock=p.min_stock,
                    max_stock=p.max_stock,
                    moq=p.moq,
                    multiple=p.multiple,
                    scrap_rate=p.scrap_rate,
                    lead_time_days=p.lead_time_days,
                    unit=p.unit,
                ))
        
        if request.demands:
            for d in request.demands:
                engine.add_demand(DemandEntry(
                    demand_id=d.demand_id or f"D-{d.item_id}-{len(engine.demands)}",
                    item_id=d.item_id,
                    sku=d.sku,
                    quantity=d.quantity,
                    due_date=_parse_date(d.due_date) or datetime.now() + timedelta(days=7),
                    source=_order_source_to_enum(d.source),
                    priority=d.priority,
                    reference_id=d.reference_id,
                ))
        
        if request.inventory:
            for inv in request.inventory:
                engine.set_inventory_position(InventoryPosition(
                    item_id=inv.item_id,
                    sku=inv.sku,
                    on_hand=inv.on_hand,
                    allocated=inv.allocated,
                    on_order=inv.on_order,
                ))
        
        if request.bom_structure:
            for bom in request.bom_structure:
                components = []
                for c in bom.components:
                    components.append(BomComponent(
                        component_item_id=c.get("component_item_id", c.get("item_id")),
                        component_sku=c.get("component_sku", c.get("sku", "")),
                        component_name=c.get("component_name", c.get("name", "")),
                        qty_per_unit=c.get("qty_per_unit", c.get("qty", 1)),
                        scrap_rate=c.get("scrap_rate", 0),
                    ))
                engine.set_bom_structure(bom.parent_item_id, components)
        
        # Parse dates
        start_date = _parse_date(request.start_date)
        end_date = _parse_date(request.end_date)
        
        # Run MRP
        result = engine.run_mrp(start_date, end_date)
        
        # Store result
        _mrp_runs[result.run_id] = result
        
        logger.info(f"MRP run {result.run_id} completed")
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"MRP run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs")
async def list_mrp_runs(
    limit: int = Query(20, ge=1, le=100),
):
    """List MRP runs."""
    runs = []
    for run_id, result in sorted(
        _mrp_runs.items(),
        key=lambda x: x[1].run_timestamp,
        reverse=True
    )[:limit]:
        runs.append(MRPRunSummary(
            run_id=run_id,
            run_timestamp=result.run_timestamp.isoformat(),
            items_processed=result.items_processed,
            demands_processed=result.demands_processed,
            purchase_orders_count=len(result.purchase_orders),
            manufacture_orders_count=len(result.manufacture_orders),
            shortage_alerts_count=len(result.shortage_alerts),
            capacity_alerts_count=len(result.capacity_alerts),
        ))
    
    return {
        "total": len(_mrp_runs),
        "returned": len(runs),
        "runs": [r.model_dump() for r in runs],
    }


@router.get("/runs/{run_id}")
async def get_mrp_run(run_id: str):
    """Get MRP run details."""
    if run_id not in _mrp_runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    return _mrp_runs[run_id].to_dict()


@router.get("/runs/{run_id}/purchase-orders")
async def get_purchase_orders(run_id: str):
    """Get purchase orders from an MRP run."""
    if run_id not in _mrp_runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    result = _mrp_runs[run_id]
    
    return {
        "run_id": run_id,
        "total": len(result.purchase_orders),
        "orders": [o.to_dict() for o in result.purchase_orders],
    }


@router.get("/runs/{run_id}/manufacture-orders")
async def get_manufacture_orders(run_id: str):
    """Get manufacture orders from an MRP run."""
    if run_id not in _mrp_runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    result = _mrp_runs[run_id]
    
    return {
        "run_id": run_id,
        "total": len(result.manufacture_orders),
        "orders": [o.to_dict() for o in result.manufacture_orders],
    }


@router.get("/runs/{run_id}/alerts")
async def get_mrp_alerts(run_id: str):
    """Get alerts from an MRP run."""
    if run_id not in _mrp_runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    result = _mrp_runs[run_id]
    
    return {
        "run_id": run_id,
        "shortage_alerts": result.shortage_alerts,
        "capacity_alerts": [a.to_dict() for a in result.capacity_alerts],
        "warnings": result.warnings,
    }


@router.get("/runs/{run_id}/item-plan/{sku}")
async def get_item_plan(run_id: str, sku: str):
    """Get item plan from an MRP run."""
    if run_id not in _mrp_runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    result = _mrp_runs[run_id]
    
    if sku not in result.item_plans:
        raise HTTPException(status_code=404, detail=f"Item {sku} not found in run")
    
    return result.item_plans[sku].to_dict()


@router.post("/demo")
async def run_demo_mrp(request: MRPDemoRequest = Body(...)):
    """
    Run demo MRP with generated data.
    """
    try:
        config = MRPConfig(
            horizon_days=request.horizon_days,
            period_days=7,
        )
        engine = MRPCompleteEngine(config)
        
        # Generate demo products
        products = []
        components = []
        
        for i in range(request.num_products):
            prod_id = 100 + i
            prod_sku = f"PROD-{i+1:03d}"
            
            products.append(ItemMRPParameters(
                item_id=prod_id,
                sku=prod_sku,
                name=f"Product {i+1}",
                source=ItemSource.MANUFACTURED,
                safety_stock=10,
                moq=5,
                multiple=5,
                lead_time_days=5,
            ))
            
            # Add 2-3 components per product
            num_comps = 2 + (i % 2)
            prod_components = []
            
            for j in range(num_comps):
                comp_id = 200 + i * 10 + j
                comp_sku = f"COMP-{i+1:03d}-{j+1:02d}"
                
                components.append(ItemMRPParameters(
                    item_id=comp_id,
                    sku=comp_sku,
                    name=f"Component {j+1} for Product {i+1}",
                    source=ItemSource.PURCHASED,
                    safety_stock=20,
                    moq=50,
                    multiple=10,
                    scrap_rate=0.02,
                    lead_time_days=14,
                ))
                
                prod_components.append(BomComponent(
                    component_item_id=comp_id,
                    component_sku=comp_sku,
                    component_name=f"Component {j+1}",
                    qty_per_unit=1 + j * 0.5,
                    scrap_rate=0.02,
                ))
            
            engine.set_bom_structure(prod_id, prod_components)
        
        # Load parameters
        for p in products + components:
            engine.set_item_parameter(p)
            
            # Set some inventory
            engine.set_inventory_position(InventoryPosition(
                item_id=p.item_id,
                sku=p.sku,
                on_hand=float(50 if "PROD" in p.sku else 100),
                allocated=0,
                on_order=0,
            ))
        
        # Generate demo orders
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        for i in range(request.num_orders):
            prod_idx = i % request.num_products
            prod = products[prod_idx]
            
            due_date = start_date + timedelta(days=7 + i * 7)
            
            engine.add_demand(DemandEntry(
                demand_id=f"SO-{i+1:04d}",
                item_id=prod.item_id,
                sku=prod.sku,
                quantity=10 + (i * 5),
                due_date=due_date,
                source=OrderSource.SALES_ORDER,
                priority=1,
            ))
        
        # Run MRP
        result = engine.run_mrp(start_date)
        
        # Store result
        _mrp_runs[result.run_id] = result
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Demo MRP failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forecast")
async def load_forecast(
    forecast_data: List[Dict[str, Any]] = Body(...),
):
    """
    Load forecast data into MRP service.
    
    Expected format:
    [{"item_id": 1, "sku": "X", "date": "2025-01-15", "quantity": 100}, ...]
    """
    service = get_mrp_service()
    count = service.load_forecast(forecast_data)
    
    return {
        "success": True,
        "forecast_entries_loaded": count,
    }


@router.delete("/reset")
async def reset_mrp():
    """Reset MRP service and clear runs."""
    global _mrp_runs
    _mrp_runs = {}
    reset_mrp_service()
    
    return {
        "success": True,
        "message": "MRP service reset",
    }


