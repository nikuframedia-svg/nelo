"""
════════════════════════════════════════════════════════════════════════════════════════════════════
PDM SERVICE - Product Data Management Service Layer
════════════════════════════════════════════════════════════════════════════════════════════════════

Contract 5 Implementation: PDM Lite Service Functions

Services for:
- Item management (create, read, update)
- Revision management (create, release, obsolete)
- BOM management (get, copy)
- Routing management (get, copy)
- ECR/ECO workflow
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session

from duplios.pdm_models import (
    Item, ItemType, ItemRevision, RevisionStatus,
    BomLine, RoutingOperation, ECR, ECRStatus, ECO,
    WorkInstruction
)
from duplios.models import SessionLocal

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ITEM MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def create_item_with_initial_revision(
    sku: str,
    name: str,
    item_type: str = "FINISHED",
    unit: str = "pcs",
    family: Optional[str] = None,
    weight_kg: Optional[float] = None,
    db: Optional[Session] = None,
) -> Dict[str, Any]:
    """
    Create a new Item with an initial DRAFT revision.
    
    Args:
        sku: Unique SKU
        name: Item name
        item_type: FINISHED, SEMI_FINISHED, RAW_MATERIAL, TOOLING, PACKAGING
        unit: Unit of measure
        family: Product family (optional)
        weight_kg: Unit weight (optional)
    
    Returns:
        Dict with item and revision data
    """
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    
    try:
        # Check if SKU exists
        existing = db.query(Item).filter(Item.sku == sku).first()
        if existing:
            raise ValueError(f"Item with SKU '{sku}' already exists")
        
        # Parse item type
        try:
            item_type_enum = ItemType(item_type.upper())
        except ValueError:
            item_type_enum = ItemType.FINISHED
        
        # Create item
        item = Item(
            sku=sku,
            name=name,
            type=item_type_enum,
            unit=unit,
            family=family,
            weight_kg=weight_kg,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(item)
        db.flush()  # Get the ID
        
        # Create initial revision
        revision = ItemRevision(
            item_id=item.id,
            code="A",  # First revision
            status=RevisionStatus.DRAFT,
            notes="Initial revision",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(revision)
        
        db.commit()
        db.refresh(item)
        db.refresh(revision)
        
        logger.info(f"Created item {sku} with initial revision {revision.code}")
        
        return {
            "item": _item_to_dict(item),
            "revision": _revision_to_dict(revision),
        }
        
    finally:
        if close_session:
            db.close()


def get_item(item_id: int, db: Optional[Session] = None) -> Optional[Dict[str, Any]]:
    """Get item by ID with active revision."""
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    
    try:
        item = db.query(Item).filter(Item.id == item_id).first()
        if not item:
            return None
        
        active_rev = get_active_revision(item_id, db)
        
        result = _item_to_dict(item)
        result["active_revision"] = _revision_to_dict(active_rev) if active_rev else None
        
        return result
        
    finally:
        if close_session:
            db.close()


def get_item_by_sku(sku: str, db: Optional[Session] = None) -> Optional[Dict[str, Any]]:
    """Get item by SKU."""
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    
    try:
        item = db.query(Item).filter(Item.sku == sku).first()
        if not item:
            return None
        
        return get_item(item.id, db)
        
    finally:
        if close_session:
            db.close()


def list_items(
    item_type: Optional[str] = None,
    family: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Optional[Session] = None,
) -> List[Dict[str, Any]]:
    """List items with optional filters."""
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    
    try:
        query = db.query(Item)
        
        if item_type:
            try:
                type_enum = ItemType(item_type.upper())
                query = query.filter(Item.type == type_enum)
            except ValueError:
                pass
        
        if family:
            query = query.filter(Item.family.ilike(f"%{family}%"))
        
        items = query.order_by(Item.name).offset(offset).limit(limit).all()
        
        result = []
        for item in items:
            active_rev = get_active_revision(item.id, db)
            item_dict = _item_to_dict(item)
            item_dict["active_revision"] = _revision_to_dict(active_rev) if active_rev else None
            result.append(item_dict)
        
        return result
        
    finally:
        if close_session:
            db.close()


# ═══════════════════════════════════════════════════════════════════════════════
# REVISION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def get_active_revision(item_id: int, db: Optional[Session] = None) -> Optional[ItemRevision]:
    """
    Get the active (RELEASED) revision for an item.
    
    If no RELEASED revision, returns the latest DRAFT.
    """
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    
    try:
        # First try to get RELEASED revision
        revision = db.query(ItemRevision).filter(
            ItemRevision.item_id == item_id,
            ItemRevision.status == RevisionStatus.RELEASED
        ).order_by(ItemRevision.effective_from.desc().nullslast()).first()
        
        if revision:
            return revision
        
        # Fall back to latest DRAFT
        return db.query(ItemRevision).filter(
            ItemRevision.item_id == item_id,
            ItemRevision.status == RevisionStatus.DRAFT
        ).order_by(ItemRevision.created_at.desc()).first()
        
    finally:
        if close_session:
            db.close()


def get_revisions_for_item(item_id: int, db: Optional[Session] = None) -> List[Dict[str, Any]]:
    """Get all revisions for an item."""
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    
    try:
        revisions = db.query(ItemRevision).filter(
            ItemRevision.item_id == item_id
        ).order_by(ItemRevision.code.desc()).all()
        
        return [_revision_to_dict(r) for r in revisions]
        
    finally:
        if close_session:
            db.close()


def create_new_revision_from_previous(
    item_id: int,
    base_revision_id: Optional[int] = None,
    changes: Optional[Dict[str, Any]] = None,
    db: Optional[Session] = None,
) -> Dict[str, Any]:
    """
    Create a new revision based on an existing one.
    
    Duplicates BOM and Routing from base revision.
    
    Args:
        item_id: Item to create revision for
        base_revision_id: Revision to copy from (uses active if None)
        changes: Dict with changes to apply
    
    Returns:
        Dict with new revision data
    """
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    
    try:
        # Get base revision
        if base_revision_id:
            base_rev = db.query(ItemRevision).filter(
                ItemRevision.id == base_revision_id
            ).first()
        else:
            base_rev = get_active_revision(item_id, db)
        
        if not base_rev:
            raise ValueError("No base revision found")
        
        # Generate next revision code
        next_code = _get_next_revision_code(item_id, db)
        
        # Create new revision
        new_rev = ItemRevision(
            item_id=item_id,
            code=next_code,
            status=RevisionStatus.DRAFT,
            notes=changes.get("notes") if changes else f"Based on revision {base_rev.code}",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(new_rev)
        db.flush()
        
        # Copy BOM lines
        bom_lines = db.query(BomLine).filter(
            BomLine.parent_revision_id == base_rev.id
        ).all()
        
        for bom_line in bom_lines:
            new_bom = BomLine(
                parent_revision_id=new_rev.id,
                component_revision_id=bom_line.component_revision_id,
                qty_per_unit=bom_line.qty_per_unit,
                scrap_rate=bom_line.scrap_rate,
                position=bom_line.position,
                notes=bom_line.notes,
            )
            db.add(new_bom)
        
        # Copy routing operations
        routing_ops = db.query(RoutingOperation).filter(
            RoutingOperation.revision_id == base_rev.id
        ).all()
        
        for op in routing_ops:
            new_op = RoutingOperation(
                revision_id=new_rev.id,
                op_code=op.op_code,
                sequence=op.sequence,
                machine_group=op.machine_group,
                nominal_setup_time=op.nominal_setup_time,
                nominal_cycle_time=op.nominal_cycle_time,
                tool_id=op.tool_id,
                description=op.description,
            )
            db.add(new_op)
        
        db.commit()
        db.refresh(new_rev)
        
        logger.info(f"Created revision {next_code} for item {item_id} based on {base_rev.code}")
        
        return {
            "revision": _revision_to_dict(new_rev),
            "base_revision": base_rev.code,
            "bom_lines_copied": len(bom_lines),
            "routing_ops_copied": len(routing_ops),
        }
        
    finally:
        if close_session:
            db.close()


class PdmReleaseError(Exception):
    """Error raised when Poka-Yoke validation fails on revision release."""
    def __init__(self, message: str, validation_errors: List[str]):
        super().__init__(message)
        self.validation_errors = validation_errors


def release_revision(
    revision_id: int, 
    db: Optional[Session] = None,
    strict_mode: bool = True,
) -> Dict[str, Any]:
    """
    Release a revision (DRAFT -> RELEASED).
    
    Validates that revision has BOM and Routing (Poka-Yoke Digital).
    Obsoletes previous RELEASED revisions of the same item.
    
    Args:
        revision_id: ID of the revision to release
        db: Optional database session
        strict_mode: If True (default), prevents release if validation fails.
                     If False, releases with warnings.
    
    Raises:
        PdmReleaseError: If strict_mode and validation fails
    """
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    
    try:
        revision = db.query(ItemRevision).filter(
            ItemRevision.id == revision_id
        ).first()
        
        if not revision:
            raise ValueError(f"Revision {revision_id} not found")
        
        if revision.status != RevisionStatus.DRAFT:
            raise ValueError(f"Only DRAFT revisions can be released")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # POKA-YOKE DIGITAL - CONTRACT 9
        # Validation before release to prevent incomplete revisions from production
        # ═══════════════════════════════════════════════════════════════════════════
        
        validation_errors = []
        warnings = []
        
        # 1. Validate BOM exists
        bom_count = db.query(BomLine).filter(
            BomLine.parent_revision_id == revision_id
        ).count()
        
        if bom_count == 0:
            validation_errors.append("Poka-Yoke: BOM vazio - adicione pelo menos 1 componente")
        
        # 2. Validate Routing exists
        routing_ops = db.query(RoutingOperation).filter(
            RoutingOperation.revision_id == revision_id
        ).all()
        
        if len(routing_ops) == 0:
            validation_errors.append("Poka-Yoke: Routing vazio - defina pelo menos 1 operação")
        
        # 3. Optional: Check WorkInstructions for critical operations
        try:
            from prodplan.work_instructions import WorkInstructionService
            wi_service = WorkInstructionService()
            
            critical_ops_without_wi = []
            for op in routing_ops:
                is_critical = getattr(op, 'is_critical', False)
                if is_critical:  # Check if operation is marked as critical
                    wi = wi_service.get_work_instructions(revision_id, op.id)
                    if not wi:
                        op_name = getattr(op, 'name', None) or op.op_code or f"OP-{op.id}"
                        critical_ops_without_wi.append(op_name)
        except ImportError:
            # WorkInstructionService not available, skip this check
            critical_ops_without_wi = []
        
        if critical_ops_without_wi:
            warnings.append(f"Operações críticas sem instruções: {', '.join(critical_ops_without_wi)}")
        
        # Apply strict mode validation
        if strict_mode and validation_errors:
            raise PdmReleaseError(
                f"Revisão não pode ser liberada: {len(validation_errors)} erro(s) de validação",
                validation_errors
            )
        
        # If not strict, convert errors to warnings
        if not strict_mode:
            warnings.extend([e.replace("Poka-Yoke: ", "AVISO: ") for e in validation_errors])
            validation_errors = []
        
        now = datetime.utcnow()
        
        # Obsolete other RELEASED revisions
        other_released = db.query(ItemRevision).filter(
            ItemRevision.item_id == revision.item_id,
            ItemRevision.status == RevisionStatus.RELEASED,
            ItemRevision.id != revision_id
        ).all()
        
        obsoleted_count = 0
        for rev in other_released:
            rev.status = RevisionStatus.OBSOLETE
            rev.effective_to = now
            rev.updated_at = now
            obsoleted_count += 1
        
        # Release this revision
        revision.status = RevisionStatus.RELEASED
        revision.effective_from = now
        revision.updated_at = now
        
        db.commit()
        db.refresh(revision)
        
        logger.info(f"Released revision {revision.code} for item {revision.item_id}")
        
        return {
            "revision": _revision_to_dict(revision),
            "obsoleted_revisions": obsoleted_count,
            "warnings": warnings,
        }
        
    finally:
        if close_session:
            db.close()


# ═══════════════════════════════════════════════════════════════════════════════
# BOM MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def get_bom(revision_id: int, db: Optional[Session] = None) -> List[Dict[str, Any]]:
    """Get BOM for a revision."""
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    
    try:
        bom_lines = db.query(BomLine).filter(
            BomLine.parent_revision_id == revision_id
        ).all()
        
        result = []
        for line in bom_lines:
            # Get component info
            comp_rev = db.query(ItemRevision).filter(
                ItemRevision.id == line.component_revision_id
            ).first()
            
            comp_item = None
            if comp_rev:
                comp_item = db.query(Item).filter(
                    Item.id == comp_rev.item_id
                ).first()
            
            result.append({
                "id": line.id,
                "component_revision_id": line.component_revision_id,
                "component_sku": comp_item.sku if comp_item else None,
                "component_name": comp_item.name if comp_item else None,
                "component_revision_code": comp_rev.code if comp_rev else None,
                "qty_per_unit": line.qty_per_unit,
                "scrap_rate": line.scrap_rate,
                "position": line.position,
                "notes": line.notes,
            })
        
        return result
        
    finally:
        if close_session:
            db.close()


def add_bom_line(
    parent_revision_id: int,
    component_revision_id: int,
    qty_per_unit: float,
    scrap_rate: float = 0.0,
    position: Optional[str] = None,
    notes: Optional[str] = None,
    db: Optional[Session] = None,
) -> Dict[str, Any]:
    """Add a BOM line."""
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    
    try:
        bom_line = BomLine(
            parent_revision_id=parent_revision_id,
            component_revision_id=component_revision_id,
            qty_per_unit=qty_per_unit,
            scrap_rate=scrap_rate,
            position=position,
            notes=notes,
        )
        db.add(bom_line)
        db.commit()
        db.refresh(bom_line)
        
        return {
            "id": bom_line.id,
            "parent_revision_id": parent_revision_id,
            "component_revision_id": component_revision_id,
            "qty_per_unit": qty_per_unit,
        }
        
    finally:
        if close_session:
            db.close()


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTING MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def get_routing(revision_id: int, db: Optional[Session] = None) -> List[Dict[str, Any]]:
    """Get routing for a revision."""
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    
    try:
        operations = db.query(RoutingOperation).filter(
            RoutingOperation.revision_id == revision_id
        ).order_by(RoutingOperation.sequence).all()
        
        return [_routing_op_to_dict(op) for op in operations]
        
    finally:
        if close_session:
            db.close()


def add_routing_operation(
    revision_id: int,
    op_code: str,
    sequence: int,
    machine_group: Optional[str] = None,
    nominal_setup_time: float = 0.0,
    nominal_cycle_time: float = 0.0,
    tool_id: Optional[str] = None,
    description: Optional[str] = None,
    db: Optional[Session] = None,
) -> Dict[str, Any]:
    """Add a routing operation."""
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    
    try:
        operation = RoutingOperation(
            revision_id=revision_id,
            op_code=op_code,
            sequence=sequence,
            machine_group=machine_group,
            nominal_setup_time=nominal_setup_time,
            nominal_cycle_time=nominal_cycle_time,
            tool_id=tool_id,
            description=description,
        )
        db.add(operation)
        db.commit()
        db.refresh(operation)
        
        return _routing_op_to_dict(operation)
        
    finally:
        if close_session:
            db.close()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_next_revision_code(item_id: int, db: Session) -> str:
    """Generate next revision code (A, B, C, ...)."""
    revisions = db.query(ItemRevision).filter(
        ItemRevision.item_id == item_id
    ).all()
    
    if not revisions:
        return "A"
    
    # Get highest code
    codes = [r.code for r in revisions]
    highest = max(codes)
    
    # Increment
    if len(highest) == 1 and highest.isalpha():
        if highest.upper() == "Z":
            return "AA"
        return chr(ord(highest.upper()) + 1)
    else:
        # Handle multi-char codes
        return f"{highest}+"


def _item_to_dict(item: Item) -> Dict[str, Any]:
    """Convert Item to dictionary."""
    return {
        "id": item.id,
        "sku": item.sku,
        "name": item.name,
        "type": item.type.value,
        "unit": item.unit,
        "family": item.family,
        "weight_kg": item.weight_kg,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
    }


def _revision_to_dict(revision: ItemRevision) -> Dict[str, Any]:
    """Convert ItemRevision to dictionary."""
    if not revision:
        return None
    return {
        "id": revision.id,
        "item_id": revision.item_id,
        "code": revision.code,
        "status": revision.status.value,
        "effective_from": revision.effective_from.isoformat() if revision.effective_from else None,
        "effective_to": revision.effective_to.isoformat() if revision.effective_to else None,
        "notes": revision.notes,
        "created_at": revision.created_at.isoformat() if revision.created_at else None,
        "updated_at": revision.updated_at.isoformat() if revision.updated_at else None,
    }


def _routing_op_to_dict(op: RoutingOperation) -> Dict[str, Any]:
    """Convert RoutingOperation to dictionary."""
    return {
        "id": op.id,
        "revision_id": op.revision_id,
        "op_code": op.op_code,
        "sequence": op.sequence,
        "machine_group": op.machine_group,
        "nominal_setup_time": op.nominal_setup_time,
        "nominal_cycle_time": op.nominal_cycle_time,
        "tool_id": op.tool_id,
        "description": op.description,
    }

