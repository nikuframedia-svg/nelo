"""
════════════════════════════════════════════════════════════════════════════════════════════════════
PDM API - Product Data Management REST Endpoints
════════════════════════════════════════════════════════════════════════════════════════════════════

Endpoints for PDM operations:
- Items: CRUD operations for master items
- Revisions: Create, release, obsolete revisions
- BOM: Manage bill of materials
- Routing: Manage manufacturing routing
- ECR/ECO: Engineering change management
- Validation: Pre-release validation
- Comparison: Revision diff

R&D / SIFIDE: WP1 - PLM/PDM Core
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Body, Depends
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from .models import SessionLocal
from .pdm_models import (
    Item, ItemType,
    ItemRevision, RevisionStatus,
    BomLine, RoutingOperation,
    ECR, ECO, ECRStatus,
    WorkInstruction,
)
from .pdm_core import (
    PDMService, PDMConfig,
    ValidationResult, ValidationIssue, ValidationSeverity,
    BomExplosion, ImpactAnalysis, RevisionDiff,
    get_pdm_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pdm", tags=["PDM"])


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE DEPENDENCY
# ═══════════════════════════════════════════════════════════════════════════════

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ItemCreate(BaseModel):
    """Create item request."""
    sku: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=255)
    type: str = Field("FINISHED", description="FINISHED, SEMI_FINISHED, RAW_MATERIAL, TOOLING, PACKAGING")
    unit: str = Field("pcs", max_length=20)
    family: Optional[str] = None
    weight_kg: Optional[float] = None
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        valid_types = ["FINISHED", "SEMI_FINISHED", "RAW_MATERIAL", "TOOLING", "PACKAGING"]
        if v not in valid_types:
            raise ValueError(f"type must be one of: {valid_types}")
        return v


class ItemResponse(BaseModel):
    """Item response."""
    id: int
    sku: str
    name: str
    type: str
    unit: str
    family: Optional[str]
    weight_kg: Optional[float]
    current_revision: Optional[str] = None


class RevisionCreate(BaseModel):
    """Create revision request."""
    code: Optional[str] = Field(None, max_length=10)
    copy_from_revision_id: Optional[int] = None
    notes: Optional[str] = None


class RevisionResponse(BaseModel):
    """Revision response."""
    id: int
    item_id: int
    code: str
    status: str
    effective_from: Optional[str]
    effective_to: Optional[str]
    notes: Optional[str]
    created_at: str


class BomLineCreate(BaseModel):
    """Create BOM line request."""
    component_revision_id: int
    qty_per_unit: float = Field(1.0, gt=0)
    scrap_rate: float = Field(0.0, ge=0, le=1)
    position: Optional[str] = None
    notes: Optional[str] = None


class BomLineResponse(BaseModel):
    """BOM line response."""
    id: int
    parent_revision_id: int
    component_revision_id: int
    component_sku: str
    component_name: str
    component_revision_code: str
    qty_per_unit: float
    scrap_rate: float
    position: Optional[str]
    unit: str


class RoutingOperationCreate(BaseModel):
    """Create routing operation request."""
    op_code: str = Field(..., min_length=1, max_length=50)
    sequence: int = Field(10, ge=1)
    machine_group: Optional[str] = None
    nominal_setup_time: float = Field(0.0, ge=0)
    nominal_cycle_time: float = Field(0.0, ge=0)
    tool_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    is_critical: bool = False
    requires_inspection: bool = False


class RoutingOperationResponse(BaseModel):
    """Routing operation response."""
    id: int
    revision_id: int
    sequence: int
    op_code: str
    name: Optional[str]
    machine_group: Optional[str]
    setup_time: float
    cycle_time: float
    tool_id: Optional[str]
    is_critical: bool
    requires_inspection: bool


class ECRCreate(BaseModel):
    """Create ECR request."""
    item_id: int
    title: str = Field(..., min_length=1, max_length=255)
    description: str
    reason: Optional[str] = None
    priority: str = Field("MEDIUM", pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    requested_by: Optional[str] = None


class ECRResponse(BaseModel):
    """ECR response."""
    id: int
    item_id: int
    title: str
    description: str
    reason: Optional[str]
    priority: str
    status: str
    requested_by: Optional[str]
    requested_at: str


class ValidationResultResponse(BaseModel):
    """Validation result response."""
    valid: bool
    errors_count: int
    warnings_count: int
    issues: List[Dict[str, Any]]


class ReleaseRequest(BaseModel):
    """Release revision request."""
    released_by: str = "system"
    force: bool = Field(False, description="Force release even with warnings")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _item_to_response(item: Item, current_rev: Optional[ItemRevision] = None) -> ItemResponse:
    return ItemResponse(
        id=item.id,
        sku=item.sku,
        name=item.name,
        type=item.type.value,
        unit=item.unit,
        family=item.family,
        weight_kg=item.weight_kg,
        current_revision=current_rev.code if current_rev else None,
    )


def _revision_to_response(rev: ItemRevision) -> RevisionResponse:
    return RevisionResponse(
        id=rev.id,
        item_id=rev.item_id,
        code=rev.code,
        status=rev.status.value,
        effective_from=rev.effective_from.isoformat() if rev.effective_from else None,
        effective_to=rev.effective_to.isoformat() if rev.effective_to else None,
        notes=rev.notes,
        created_at=rev.created_at.isoformat() if rev.created_at else "",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ITEM ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/items", response_model=List[ItemResponse])
async def list_items(
    type: Optional[str] = Query(None, description="Filter by type"),
    family: Optional[str] = Query(None, description="Filter by family"),
    search: Optional[str] = Query(None, description="Search by SKU or name"),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """List all items with optional filters."""
    query = db.query(Item)
    
    if type:
        try:
            item_type = ItemType(type)
            query = query.filter(Item.type == item_type)
        except ValueError:
            pass
    
    if family:
        query = query.filter(Item.family == family)
    
    if search:
        query = query.filter(
            (Item.sku.ilike(f"%{search}%")) | (Item.name.ilike(f"%{search}%"))
        )
    
    items = query.limit(limit).all()
    
    result = []
    for item in items:
        current_rev = db.query(ItemRevision).filter(
            ItemRevision.item_id == item.id,
            ItemRevision.status == RevisionStatus.RELEASED
        ).first()
        result.append(_item_to_response(item, current_rev))
    
    return result


@router.post("/items", response_model=ItemResponse)
async def create_item(
    data: ItemCreate = Body(...),
    db: Session = Depends(get_db),
):
    """Create a new item."""
    # Check if SKU exists
    existing = db.query(Item).filter(Item.sku == data.sku).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Item with SKU {data.sku} already exists")
    
    item = Item(
        sku=data.sku,
        name=data.name,
        type=ItemType(data.type),
        unit=data.unit,
        family=data.family,
        weight_kg=data.weight_kg,
    )
    
    db.add(item)
    db.commit()
    db.refresh(item)
    
    logger.info(f"Created item {item.id} ({item.sku})")
    
    return _item_to_response(item)


@router.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(
    item_id: int,
    db: Session = Depends(get_db),
):
    """Get item by ID."""
    item = db.query(Item).filter(Item.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    current_rev = db.query(ItemRevision).filter(
        ItemRevision.item_id == item.id,
        ItemRevision.status == RevisionStatus.RELEASED
    ).first()
    
    return _item_to_response(item, current_rev)


@router.get("/items/by-sku/{sku}", response_model=ItemResponse)
async def get_item_by_sku(
    sku: str,
    db: Session = Depends(get_db),
):
    """Get item by SKU."""
    item = db.query(Item).filter(Item.sku == sku).first()
    if not item:
        raise HTTPException(status_code=404, detail=f"Item with SKU {sku} not found")
    
    current_rev = db.query(ItemRevision).filter(
        ItemRevision.item_id == item.id,
        ItemRevision.status == RevisionStatus.RELEASED
    ).first()
    
    return _item_to_response(item, current_rev)


# ═══════════════════════════════════════════════════════════════════════════════
# REVISION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/items/{item_id}/revisions", response_model=List[RevisionResponse])
async def list_revisions(
    item_id: int,
    status: Optional[str] = Query(None, description="Filter by status"),
    db: Session = Depends(get_db),
):
    """List revisions for an item."""
    item = db.query(Item).filter(Item.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    query = db.query(ItemRevision).filter(ItemRevision.item_id == item_id)
    
    if status:
        try:
            rev_status = RevisionStatus(status)
            query = query.filter(ItemRevision.status == rev_status)
        except ValueError:
            pass
    
    revisions = query.order_by(ItemRevision.code).all()
    
    return [_revision_to_response(rev) for rev in revisions]


@router.post("/items/{item_id}/revisions", response_model=RevisionResponse)
async def create_revision(
    item_id: int,
    data: RevisionCreate = Body(...),
    db: Session = Depends(get_db),
):
    """Create a new draft revision."""
    service = PDMService(db=db)
    
    rev = service.create_new_revision(
        item_id=item_id,
        code=data.code,
        copy_from=data.copy_from_revision_id,
    )
    
    if not rev:
        raise HTTPException(status_code=400, detail="Failed to create revision")
    
    if data.notes:
        rev.notes = data.notes
        db.commit()
        db.refresh(rev)
    
    return _revision_to_response(rev)


@router.get("/revisions/{revision_id}", response_model=RevisionResponse)
async def get_revision(
    revision_id: int,
    db: Session = Depends(get_db),
):
    """Get revision by ID."""
    rev = db.query(ItemRevision).filter(ItemRevision.id == revision_id).first()
    if not rev:
        raise HTTPException(status_code=404, detail="Revision not found")
    
    return _revision_to_response(rev)


@router.post("/revisions/{revision_id}/release")
async def release_revision(
    revision_id: int,
    data: ReleaseRequest = Body(...),
    db: Session = Depends(get_db),
):
    """Release a revision."""
    service = PDMService(db=db)
    
    success, rev, validation = service.release_revision(
        revision_id=revision_id,
        released_by=data.released_by,
        force=data.force,
    )
    
    return {
        "success": success,
        "revision": _revision_to_response(rev).model_dump() if rev else None,
        "validation": validation.to_dict(),
    }


@router.post("/revisions/{revision_id}/obsolete")
async def obsolete_revision(
    revision_id: int,
    reason: Optional[str] = Body(None, embed=True),
    db: Session = Depends(get_db),
):
    """Mark a revision as obsolete."""
    from .pdm_core import RevisionWorkflowEngine, PDMConfig
    
    engine = RevisionWorkflowEngine(PDMConfig(), db)
    success, rev = engine.obsolete_revision(revision_id, reason=reason)
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to obsolete revision")
    
    return {
        "success": True,
        "revision": _revision_to_response(rev).model_dump() if rev else None,
    }


@router.get("/revisions/{revision_id}/validate")
async def validate_revision(
    revision_id: int,
    db: Session = Depends(get_db),
):
    """Validate a revision for release."""
    service = PDMService(db=db)
    validation = service.validate_for_release(revision_id)
    
    return validation.to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# BOM ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/revisions/{revision_id}/bom")
async def get_bom(
    revision_id: int,
    db: Session = Depends(get_db),
):
    """Get BOM for a revision."""
    rev = db.query(ItemRevision).filter(ItemRevision.id == revision_id).first()
    if not rev:
        raise HTTPException(status_code=404, detail="Revision not found")
    
    bom_lines = db.query(BomLine).filter(
        BomLine.parent_revision_id == revision_id
    ).all()
    
    result = []
    for line in bom_lines:
        comp_rev = db.query(ItemRevision).filter(
            ItemRevision.id == line.component_revision_id
        ).first()
        comp_item = db.query(Item).filter(
            Item.id == comp_rev.item_id
        ).first() if comp_rev else None
        
        result.append({
            "id": line.id,
            "parent_revision_id": line.parent_revision_id,
            "component_revision_id": line.component_revision_id,
            "component_sku": comp_item.sku if comp_item else "",
            "component_name": comp_item.name if comp_item else "",
            "component_revision_code": comp_rev.code if comp_rev else "",
            "qty_per_unit": line.qty_per_unit,
            "scrap_rate": line.scrap_rate,
            "position": line.position,
            "unit": comp_item.unit if comp_item else "pcs",
        })
    
    return result


@router.post("/revisions/{revision_id}/bom")
async def add_bom_line(
    revision_id: int,
    data: BomLineCreate = Body(...),
    db: Session = Depends(get_db),
):
    """Add a BOM line to a revision."""
    rev = db.query(ItemRevision).filter(ItemRevision.id == revision_id).first()
    if not rev:
        raise HTTPException(status_code=404, detail="Revision not found")
    
    if rev.status != RevisionStatus.DRAFT:
        raise HTTPException(status_code=400, detail="Can only modify DRAFT revisions")
    
    # Validate component exists
    comp_rev = db.query(ItemRevision).filter(
        ItemRevision.id == data.component_revision_id
    ).first()
    if not comp_rev:
        raise HTTPException(status_code=400, detail="Component revision not found")
    
    # Check for self-reference
    if data.component_revision_id == revision_id:
        raise HTTPException(status_code=400, detail="Cannot add item to its own BOM")
    
    line = BomLine(
        parent_revision_id=revision_id,
        component_revision_id=data.component_revision_id,
        qty_per_unit=data.qty_per_unit,
        scrap_rate=data.scrap_rate,
        position=data.position,
        notes=data.notes,
    )
    
    db.add(line)
    db.commit()
    db.refresh(line)
    
    comp_item = db.query(Item).filter(Item.id == comp_rev.item_id).first()
    
    return {
        "id": line.id,
        "parent_revision_id": line.parent_revision_id,
        "component_revision_id": line.component_revision_id,
        "component_sku": comp_item.sku if comp_item else "",
        "component_name": comp_item.name if comp_item else "",
        "qty_per_unit": line.qty_per_unit,
    }


@router.delete("/bom/{bom_line_id}")
async def delete_bom_line(
    bom_line_id: int,
    db: Session = Depends(get_db),
):
    """Delete a BOM line."""
    line = db.query(BomLine).filter(BomLine.id == bom_line_id).first()
    if not line:
        raise HTTPException(status_code=404, detail="BOM line not found")
    
    rev = db.query(ItemRevision).filter(ItemRevision.id == line.parent_revision_id).first()
    if rev and rev.status != RevisionStatus.DRAFT:
        raise HTTPException(status_code=400, detail="Can only modify DRAFT revisions")
    
    db.delete(line)
    db.commit()
    
    return {"success": True}


@router.get("/revisions/{revision_id}/bom/explode")
async def explode_bom(
    revision_id: int,
    qty: float = Query(1.0, ge=0.001),
    db: Session = Depends(get_db),
):
    """Explode BOM recursively."""
    service = PDMService(db=db)
    explosion = service.explode_bom(revision_id, qty)
    
    return {
        "total_components": explosion.total_components,
        "max_depth": explosion.max_depth,
        "has_cycle": explosion.has_cycle,
        "flat_list": explosion.flat_list,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTING ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/revisions/{revision_id}/routing")
async def get_routing(
    revision_id: int,
    db: Session = Depends(get_db),
):
    """Get routing for a revision."""
    rev = db.query(ItemRevision).filter(ItemRevision.id == revision_id).first()
    if not rev:
        raise HTTPException(status_code=404, detail="Revision not found")
    
    operations = db.query(RoutingOperation).filter(
        RoutingOperation.revision_id == revision_id
    ).order_by(RoutingOperation.sequence).all()
    
    return [{
        "id": op.id,
        "revision_id": op.revision_id,
        "sequence": op.sequence,
        "op_code": op.op_code,
        "name": op.name,
        "machine_group": op.machine_group,
        "setup_time": op.nominal_setup_time,
        "cycle_time": op.nominal_cycle_time,
        "tool_id": op.tool_id,
        "is_critical": op.is_critical,
        "requires_inspection": op.requires_inspection,
    } for op in operations]


@router.post("/revisions/{revision_id}/routing")
async def add_routing_operation(
    revision_id: int,
    data: RoutingOperationCreate = Body(...),
    db: Session = Depends(get_db),
):
    """Add a routing operation."""
    rev = db.query(ItemRevision).filter(ItemRevision.id == revision_id).first()
    if not rev:
        raise HTTPException(status_code=404, detail="Revision not found")
    
    if rev.status != RevisionStatus.DRAFT:
        raise HTTPException(status_code=400, detail="Can only modify DRAFT revisions")
    
    # Check for duplicate sequence
    existing = db.query(RoutingOperation).filter(
        RoutingOperation.revision_id == revision_id,
        RoutingOperation.sequence == data.sequence
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Operation with sequence {data.sequence} already exists")
    
    op = RoutingOperation(
        revision_id=revision_id,
        op_code=data.op_code,
        sequence=data.sequence,
        machine_group=data.machine_group,
        nominal_setup_time=data.nominal_setup_time,
        nominal_cycle_time=data.nominal_cycle_time,
        tool_id=data.tool_id,
        name=data.name,
        description=data.description,
        is_critical=data.is_critical,
        requires_inspection=data.requires_inspection,
    )
    
    db.add(op)
    db.commit()
    db.refresh(op)
    
    return {
        "id": op.id,
        "revision_id": op.revision_id,
        "sequence": op.sequence,
        "op_code": op.op_code,
    }


@router.delete("/routing/{operation_id}")
async def delete_routing_operation(
    operation_id: int,
    db: Session = Depends(get_db),
):
    """Delete a routing operation."""
    op = db.query(RoutingOperation).filter(RoutingOperation.id == operation_id).first()
    if not op:
        raise HTTPException(status_code=404, detail="Operation not found")
    
    rev = db.query(ItemRevision).filter(ItemRevision.id == op.revision_id).first()
    if rev and rev.status != RevisionStatus.DRAFT:
        raise HTTPException(status_code=400, detail="Can only modify DRAFT revisions")
    
    db.delete(op)
    db.commit()
    
    return {"success": True}


# ═══════════════════════════════════════════════════════════════════════════════
# ECR/ECO ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/ecr")
async def list_ecrs(
    item_id: Optional[int] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """List ECRs."""
    query = db.query(ECR)
    
    if item_id:
        query = query.filter(ECR.item_id == item_id)
    
    if status:
        try:
            ecr_status = ECRStatus(status)
            query = query.filter(ECR.status == ecr_status)
        except ValueError:
            pass
    
    ecrs = query.order_by(ECR.created_at.desc()).limit(limit).all()
    
    return [{
        "id": ecr.id,
        "item_id": ecr.item_id,
        "title": ecr.title,
        "description": ecr.description,
        "reason": ecr.reason,
        "priority": ecr.priority,
        "status": ecr.status.value,
        "requested_by": ecr.requested_by,
        "requested_at": ecr.requested_at.isoformat() if ecr.requested_at else None,
    } for ecr in ecrs]


@router.post("/ecr")
async def create_ecr(
    data: ECRCreate = Body(...),
    db: Session = Depends(get_db),
):
    """Create an ECR."""
    service = PDMService(db=db)
    
    ecr = service.create_ecr(
        item_id=data.item_id,
        title=data.title,
        description=data.description,
        reason=data.reason,
        priority=data.priority,
        requested_by=data.requested_by or "system",
    )
    
    if not ecr:
        raise HTTPException(status_code=400, detail="Failed to create ECR")
    
    return {
        "id": ecr.id,
        "item_id": ecr.item_id,
        "title": ecr.title,
        "status": ecr.status.value,
    }


@router.get("/ecr/{ecr_id}/impact")
async def analyze_ecr_impact(
    ecr_id: int,
    db: Session = Depends(get_db),
):
    """Analyze impact of an ECR."""
    ecr = db.query(ECR).filter(ECR.id == ecr_id).first()
    if not ecr:
        raise HTTPException(status_code=404, detail="ECR not found")
    
    service = PDMService(db=db)
    impact = service.analyze_impact(ecr.item_id)
    
    return impact.to_dict()


@router.get("/items/{item_id}/impact")
async def analyze_item_impact(
    item_id: int,
    revision_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """Analyze impact of changing an item/revision."""
    service = PDMService(db=db)
    impact = service.analyze_impact(item_id, revision_id)
    
    return impact.to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/revisions/compare")
async def compare_revisions(
    from_revision_id: int = Query(...),
    to_revision_id: int = Query(...),
    db: Session = Depends(get_db),
):
    """Compare two revisions."""
    service = PDMService(db=db)
    diff = service.compare_revisions(from_revision_id, to_revision_id)
    
    return diff.to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# STATUS ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/status")
async def get_pdm_status(db: Session = Depends(get_db)):
    """Get PDM module status."""
    total_items = db.query(Item).count()
    total_revisions = db.query(ItemRevision).count()
    released_revisions = db.query(ItemRevision).filter(
        ItemRevision.status == RevisionStatus.RELEASED
    ).count()
    draft_revisions = db.query(ItemRevision).filter(
        ItemRevision.status == RevisionStatus.DRAFT
    ).count()
    open_ecrs = db.query(ECR).filter(ECR.status == ECRStatus.OPEN).count()
    
    return {
        "service": "PDM - Product Data Management",
        "version": "1.0.0",
        "status": "operational",
        "statistics": {
            "total_items": total_items,
            "total_revisions": total_revisions,
            "released_revisions": released_revisions,
            "draft_revisions": draft_revisions,
            "open_ecrs": open_ecrs,
        },
        "item_types": [t.value for t in ItemType],
        "revision_statuses": [s.value for s in RevisionStatus],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO DATA ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/demo/seed")
async def seed_demo_data(db: Session = Depends(get_db)):
    """Seed demo PDM data."""
    try:
        # Create items
        items_data = [
            {"sku": "PROD-001", "name": "Finished Product A", "type": ItemType.FINISHED, "unit": "pcs", "weight_kg": 2.5},
            {"sku": "SEMI-001", "name": "Sub-Assembly A", "type": ItemType.SEMI_FINISHED, "unit": "pcs", "weight_kg": 1.2},
            {"sku": "RAW-001", "name": "Steel Bar", "type": ItemType.RAW_MATERIAL, "unit": "kg"},
            {"sku": "RAW-002", "name": "Aluminum Sheet", "type": ItemType.RAW_MATERIAL, "unit": "m2"},
            {"sku": "PACK-001", "name": "Cardboard Box", "type": ItemType.PACKAGING, "unit": "pcs"},
        ]
        
        created_items = []
        for data in items_data:
            existing = db.query(Item).filter(Item.sku == data["sku"]).first()
            if not existing:
                item = Item(**data)
                db.add(item)
                db.flush()
                created_items.append(item)
            else:
                created_items.append(existing)
        
        # Create revisions for each item
        for item in created_items:
            existing_rev = db.query(ItemRevision).filter(
                ItemRevision.item_id == item.id
            ).first()
            
            if not existing_rev:
                rev = ItemRevision(
                    item_id=item.id,
                    code="A",
                    status=RevisionStatus.RELEASED,
                    effective_from=datetime.utcnow(),
                )
                db.add(rev)
        
        db.commit()
        
        # Get final state
        items = db.query(Item).all()
        revisions = db.query(ItemRevision).all()
        
        return {
            "success": True,
            "items_created": len([i for i in created_items if i.id]),
            "total_items": len(items),
            "total_revisions": len(revisions),
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error seeding demo data: {e}")
        raise HTTPException(status_code=500, detail=str(e))



