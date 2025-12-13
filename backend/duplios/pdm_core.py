"""
════════════════════════════════════════════════════════════════════════════════════════════════════
PDM CORE - Product Data Management Engine
════════════════════════════════════════════════════════════════════════════════════════════════════

Core PDM engine with:
- BOM cycle detection (DAG validation)
- Revision workflow (Draft → Released → Obsolete)
- Release validation engine
- ECR/ECO impact analysis
- Consistency checks

Features:
- Graph algorithms for BOM validation
- Workflow state machine for revisions
- Pre-release validation rules
- Impact analysis for engineering changes
- Integration points for ProdPlan, SmartInventory, Duplios

R&D / SIFIDE: WP1 - PLM/PDM Core
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from collections import defaultdict

from sqlalchemy.orm import Session

from .models import SessionLocal
from .pdm_models import (
    Item, ItemType,
    ItemRevision, RevisionStatus,
    BomLine, RoutingOperation,
    ECR, ECO, ECRStatus,
    WorkInstruction,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PDMConfig:
    """Configuration for PDM validation and workflow."""
    
    # BOM Validation
    max_bom_depth: int = 20  # Maximum BOM explosion depth
    allow_draft_components: bool = False  # Allow Draft components in Released BOM
    
    # Routing Validation
    require_routing_for_manufactured: bool = True
    require_work_instructions_for_critical: bool = True
    
    # Attachments
    required_attachments_finished: List[str] = field(default_factory=lambda: ["CAD"])
    required_attachments_semi: List[str] = field(default_factory=list)
    
    # Release Validation
    require_bom_for_release: bool = True
    require_routing_for_release: bool = True
    require_quality_plan: bool = False
    
    # ECR/ECO
    auto_obsolete_on_release: bool = True
    require_ecr_for_changes: bool = False


class ValidationSeverity(str, Enum):
    """Severity of validation issues."""
    ERROR = "error"  # Blocks release
    WARNING = "warning"  # Allows release with acknowledgment
    INFO = "info"  # Information only


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationIssue:
    """A validation issue found during checks."""
    code: str
    message: str
    severity: ValidationSeverity
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "field": self.field,
            "details": self.details,
        }


@dataclass
class ValidationResult:
    """Result of validation checks."""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings_count: int = 0
    errors_count: int = 0
    
    def add_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.errors_count += 1
            self.valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warnings_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors_count": self.errors_count,
            "warnings_count": self.warnings_count,
            "issues": [i.to_dict() for i in self.issues],
        }


@dataclass
class BomNode:
    """Node in BOM graph for cycle detection."""
    revision_id: int
    item_id: int
    sku: str
    qty: float = 1.0
    level: int = 0
    children: List['BomNode'] = field(default_factory=list)


@dataclass
class BomExplosion:
    """Result of BOM explosion."""
    root: BomNode
    flat_list: List[Dict[str, Any]] = field(default_factory=list)
    total_components: int = 0
    max_depth: int = 0
    has_cycle: bool = False
    cycle_path: Optional[List[str]] = None


@dataclass
class ImpactAnalysis:
    """Impact analysis result for ECR/ECO."""
    affected_items: List[Dict[str, Any]]
    open_production_orders: List[Dict[str, Any]]
    wip_inventory: List[Dict[str, Any]]
    finished_inventory: List[Dict[str, Any]]
    affected_dpps: List[Dict[str, Any]]
    total_affected_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "affected_items": self.affected_items,
            "open_production_orders": self.open_production_orders,
            "wip_inventory": self.wip_inventory,
            "finished_inventory": self.finished_inventory,
            "affected_dpps": self.affected_dpps,
            "total_affected_count": self.total_affected_count,
        }


@dataclass
class RevisionDiff:
    """Difference between two revisions."""
    from_revision: str
    to_revision: str
    bom_added: List[Dict[str, Any]] = field(default_factory=list)
    bom_removed: List[Dict[str, Any]] = field(default_factory=list)
    bom_changed: List[Dict[str, Any]] = field(default_factory=list)
    routing_added: List[Dict[str, Any]] = field(default_factory=list)
    routing_removed: List[Dict[str, Any]] = field(default_factory=list)
    routing_changed: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# BOM VALIDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class BomValidationEngine:
    """
    Engine for BOM structure validation.
    
    Features:
    - Cycle detection using DFS
    - Structure consistency checks
    - Component status validation
    """
    
    def __init__(self, config: PDMConfig, db: Session):
        self.config = config
        self.db = db
    
    def detect_cycle(self, revision_id: int) -> Tuple[bool, Optional[List[str]]]:
        """
        Detect cycles in BOM using DFS.
        
        A BOM must be a DAG (Directed Acyclic Graph).
        
        Returns:
            (has_cycle, cycle_path) where cycle_path shows the cycle if found
        """
        visited: Set[int] = set()
        rec_stack: Set[int] = set()
        path: List[int] = []
        
        def dfs(rev_id: int) -> bool:
            visited.add(rev_id)
            rec_stack.add(rev_id)
            path.append(rev_id)
            
            # Get BOM lines for this revision
            bom_lines = self.db.query(BomLine).filter(
                BomLine.parent_revision_id == rev_id
            ).all()
            
            for line in bom_lines:
                child_id = line.component_revision_id
                
                if child_id not in visited:
                    if dfs(child_id):
                        return True
                elif child_id in rec_stack:
                    # Cycle detected!
                    path.append(child_id)
                    return True
            
            path.pop()
            rec_stack.remove(rev_id)
            return False
        
        has_cycle = dfs(revision_id)
        
        if has_cycle:
            # Build readable cycle path
            cycle_path = []
            for rev_id in path:
                rev = self.db.query(ItemRevision).filter(
                    ItemRevision.id == rev_id
                ).first()
                if rev:
                    item = self.db.query(Item).filter(Item.id == rev.item_id).first()
                    cycle_path.append(f"{item.sku}:{rev.code}" if item else f"Rev:{rev_id}")
            return True, cycle_path
        
        return False, None
    
    def validate_bom_structure(self, revision_id: int) -> ValidationResult:
        """
        Validate BOM structure and consistency.
        """
        result = ValidationResult(valid=True)
        
        # Check for cycles
        has_cycle, cycle_path = self.detect_cycle(revision_id)
        if has_cycle:
            result.add_issue(ValidationIssue(
                code="BOM_CYCLE",
                message=f"BOM contains cycle: {' → '.join(cycle_path or [])}",
                severity=ValidationSeverity.ERROR,
                details={"cycle_path": cycle_path},
            ))
            return result  # No point checking further
        
        # Get revision and item
        revision = self.db.query(ItemRevision).filter(
            ItemRevision.id == revision_id
        ).first()
        
        if not revision:
            result.add_issue(ValidationIssue(
                code="REVISION_NOT_FOUND",
                message=f"Revision {revision_id} not found",
                severity=ValidationSeverity.ERROR,
            ))
            return result
        
        item = self.db.query(Item).filter(Item.id == revision.item_id).first()
        
        # Get BOM lines
        bom_lines = self.db.query(BomLine).filter(
            BomLine.parent_revision_id == revision_id
        ).all()
        
        # Check if BOM is required
        if not bom_lines:
            if item.type in [ItemType.FINISHED, ItemType.SEMI_FINISHED]:
                if self.config.require_bom_for_release:
                    result.add_issue(ValidationIssue(
                        code="BOM_EMPTY",
                        message="BOM is empty for manufactured item",
                        severity=ValidationSeverity.ERROR,
                        field="bom",
                    ))
                else:
                    result.add_issue(ValidationIssue(
                        code="BOM_EMPTY_WARNING",
                        message="BOM is empty - verify this is intentional",
                        severity=ValidationSeverity.WARNING,
                        field="bom",
                    ))
        
        # Validate each BOM line
        for line in bom_lines:
            # Check component revision exists
            comp_rev = self.db.query(ItemRevision).filter(
                ItemRevision.id == line.component_revision_id
            ).first()
            
            if not comp_rev:
                result.add_issue(ValidationIssue(
                    code="BOM_INVALID_COMPONENT",
                    message=f"Component revision {line.component_revision_id} not found",
                    severity=ValidationSeverity.ERROR,
                    field="bom",
                    details={"component_revision_id": line.component_revision_id},
                ))
                continue
            
            # Check component status
            if comp_rev.status == RevisionStatus.DRAFT:
                if not self.config.allow_draft_components:
                    result.add_issue(ValidationIssue(
                        code="BOM_DRAFT_COMPONENT",
                        message=f"Component {comp_rev.id} is still in DRAFT status",
                        severity=ValidationSeverity.ERROR,
                        field="bom",
                        details={"component_revision_id": comp_rev.id, "status": comp_rev.status.value},
                    ))
                else:
                    result.add_issue(ValidationIssue(
                        code="BOM_DRAFT_COMPONENT_WARNING",
                        message=f"Component {comp_rev.id} is in DRAFT status",
                        severity=ValidationSeverity.WARNING,
                        field="bom",
                    ))
            
            # Check quantity
            if line.qty_per_unit <= 0:
                result.add_issue(ValidationIssue(
                    code="BOM_INVALID_QTY",
                    message=f"Invalid quantity {line.qty_per_unit} for component",
                    severity=ValidationSeverity.ERROR,
                    field="bom",
                    details={"component_revision_id": line.component_revision_id, "qty": line.qty_per_unit},
                ))
            
            # Check scrap rate
            if line.scrap_rate < 0 or line.scrap_rate > 1:
                result.add_issue(ValidationIssue(
                    code="BOM_INVALID_SCRAP",
                    message=f"Scrap rate {line.scrap_rate} must be between 0 and 1",
                    severity=ValidationSeverity.WARNING,
                    field="bom",
                ))
        
        return result
    
    def explode_bom(
        self,
        revision_id: int,
        qty: float = 1.0,
        max_depth: Optional[int] = None,
    ) -> BomExplosion:
        """
        Explode BOM recursively.
        
        Returns hierarchical structure and flat list of all components.
        """
        max_depth = max_depth or self.config.max_bom_depth
        explosion = BomExplosion(root=None, flat_list=[], has_cycle=False)
        visited: Set[int] = set()
        
        def build_tree(rev_id: int, qty_multiplier: float, level: int) -> Optional[BomNode]:
            if level > max_depth:
                return None
            
            if rev_id in visited:
                explosion.has_cycle = True
                return None
            
            visited.add(rev_id)
            
            revision = self.db.query(ItemRevision).filter(
                ItemRevision.id == rev_id
            ).first()
            
            if not revision:
                return None
            
            item = self.db.query(Item).filter(Item.id == revision.item_id).first()
            
            node = BomNode(
                revision_id=rev_id,
                item_id=revision.item_id,
                sku=item.sku if item else "",
                qty=qty_multiplier,
                level=level,
            )
            
            # Add to flat list
            explosion.flat_list.append({
                "level": level,
                "revision_id": rev_id,
                "item_id": revision.item_id,
                "sku": item.sku if item else "",
                "name": item.name if item else "",
                "qty": qty_multiplier,
                "unit": item.unit if item else "pcs",
            })
            
            # Get children
            bom_lines = self.db.query(BomLine).filter(
                BomLine.parent_revision_id == rev_id
            ).all()
            
            for line in bom_lines:
                child_qty = qty_multiplier * line.qty_per_unit * (1 + line.scrap_rate)
                child_node = build_tree(line.component_revision_id, child_qty, level + 1)
                if child_node:
                    node.children.append(child_node)
            
            visited.remove(rev_id)
            explosion.max_depth = max(explosion.max_depth, level)
            return node
        
        explosion.root = build_tree(revision_id, qty, 0)
        explosion.total_components = len(explosion.flat_list)
        
        return explosion


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTING VALIDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class RoutingValidationEngine:
    """
    Engine for Routing validation.
    
    Checks:
    - Sequence integrity
    - Required fields
    - Work instructions for critical operations
    """
    
    def __init__(self, config: PDMConfig, db: Session):
        self.config = config
        self.db = db
    
    def validate_routing(self, revision_id: int) -> ValidationResult:
        """Validate routing structure and consistency."""
        result = ValidationResult(valid=True)
        
        # Get revision and item
        revision = self.db.query(ItemRevision).filter(
            ItemRevision.id == revision_id
        ).first()
        
        if not revision:
            result.add_issue(ValidationIssue(
                code="REVISION_NOT_FOUND",
                message=f"Revision {revision_id} not found",
                severity=ValidationSeverity.ERROR,
            ))
            return result
        
        item = self.db.query(Item).filter(Item.id == revision.item_id).first()
        
        # Get routing operations
        operations = self.db.query(RoutingOperation).filter(
            RoutingOperation.revision_id == revision_id
        ).order_by(RoutingOperation.sequence).all()
        
        # Check if routing is required
        is_manufactured = item.type in [ItemType.FINISHED, ItemType.SEMI_FINISHED]
        
        if not operations:
            if is_manufactured and self.config.require_routing_for_manufactured:
                result.add_issue(ValidationIssue(
                    code="ROUTING_EMPTY",
                    message="Routing is empty for manufactured item",
                    severity=ValidationSeverity.ERROR,
                    field="routing",
                ))
            return result
        
        # Check for duplicate sequences
        sequences = [op.sequence for op in operations]
        if len(sequences) != len(set(sequences)):
            result.add_issue(ValidationIssue(
                code="ROUTING_DUPLICATE_SEQ",
                message="Duplicate sequence numbers in routing",
                severity=ValidationSeverity.ERROR,
                field="routing",
            ))
        
        # Validate each operation
        for op in operations:
            # Check required fields
            if not op.op_code:
                result.add_issue(ValidationIssue(
                    code="ROUTING_MISSING_OP_CODE",
                    message=f"Operation {op.sequence} missing op_code",
                    severity=ValidationSeverity.ERROR,
                    field="routing",
                    details={"operation_id": op.id, "sequence": op.sequence},
                ))
            
            # Check times
            if op.nominal_cycle_time < 0:
                result.add_issue(ValidationIssue(
                    code="ROUTING_INVALID_TIME",
                    message=f"Operation {op.sequence} has invalid cycle time",
                    severity=ValidationSeverity.ERROR,
                    field="routing",
                ))
            
            if op.nominal_cycle_time == 0 and op.nominal_setup_time == 0:
                result.add_issue(ValidationIssue(
                    code="ROUTING_ZERO_TIME",
                    message=f"Operation {op.sequence} has zero cycle and setup time",
                    severity=ValidationSeverity.WARNING,
                    field="routing",
                ))
            
            # Check work instructions for critical operations
            if op.is_critical and self.config.require_work_instructions_for_critical:
                wi = self.db.query(WorkInstruction).filter(
                    WorkInstruction.revision_id == revision_id,
                    WorkInstruction.operation_id == op.id
                ).first()
                
                if not wi:
                    result.add_issue(ValidationIssue(
                        code="ROUTING_MISSING_WI",
                        message=f"Critical operation {op.sequence} ({op.op_code}) missing work instructions",
                        severity=ValidationSeverity.ERROR,
                        field="routing",
                        details={"operation_id": op.id, "op_code": op.op_code},
                    ))
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# RELEASE VALIDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ReleaseValidationEngine:
    """
    Engine for pre-release validation.
    
    Performs all checks before allowing a revision to be released.
    """
    
    def __init__(self, config: PDMConfig, db: Session):
        self.config = config
        self.db = db
        self.bom_engine = BomValidationEngine(config, db)
        self.routing_engine = RoutingValidationEngine(config, db)
    
    def validate_for_release(self, revision_id: int) -> ValidationResult:
        """
        Perform all pre-release validations.
        
        Checks:
        - BOM structure and consistency
        - Routing structure and consistency
        - Required attachments
        - Work instructions for critical operations
        """
        result = ValidationResult(valid=True)
        
        # Get revision
        revision = self.db.query(ItemRevision).filter(
            ItemRevision.id == revision_id
        ).first()
        
        if not revision:
            result.add_issue(ValidationIssue(
                code="REVISION_NOT_FOUND",
                message=f"Revision {revision_id} not found",
                severity=ValidationSeverity.ERROR,
            ))
            return result
        
        # Check current status
        if revision.status == RevisionStatus.RELEASED:
            result.add_issue(ValidationIssue(
                code="ALREADY_RELEASED",
                message="Revision is already released",
                severity=ValidationSeverity.ERROR,
            ))
            return result
        
        if revision.status == RevisionStatus.OBSOLETE:
            result.add_issue(ValidationIssue(
                code="OBSOLETE_REVISION",
                message="Cannot release an obsolete revision",
                severity=ValidationSeverity.ERROR,
            ))
            return result
        
        # BOM validation
        bom_result = self.bom_engine.validate_bom_structure(revision_id)
        for issue in bom_result.issues:
            result.add_issue(issue)
        
        # Routing validation
        routing_result = self.routing_engine.validate_routing(revision_id)
        for issue in routing_result.issues:
            result.add_issue(issue)
        
        # Attachment validation
        from .pdm_models import Attachment
        attachments = self.db.query(Attachment).filter(
            Attachment.revision_id == revision_id
        ).all()
        
        attachment_types = {att.file_type for att in attachments}
        
        # Check required attachments based on item type
        if item.type == ItemType.FINISHED:
            required = self.config.required_attachments_finished
            for req_type in required:
                if req_type not in attachment_types:
                    result.add_issue(ValidationIssue(
                        code="MISSING_ATTACHMENT",
                        message=f"Required attachment type '{req_type}' not found for finished product",
                        severity=ValidationSeverity.ERROR if self.config.require_quality_plan else ValidationSeverity.WARNING,
                        field="attachments",
                        details={"required_type": req_type, "item_type": item.type.value},
                    ))
        elif item.type == ItemType.SEMI_FINISHED:
            required = self.config.required_attachments_semi
            for req_type in required:
                if req_type not in attachment_types:
                    result.add_issue(ValidationIssue(
                        code="MISSING_ATTACHMENT",
                        message=f"Required attachment type '{req_type}' not found for semi-finished product",
                        severity=ValidationSeverity.WARNING,
                        field="attachments",
                        details={"required_type": req_type, "item_type": item.type.value},
                    ))
        
        # Check for existing Released revision
        item = self.db.query(Item).filter(Item.id == revision.item_id).first()
        existing_released = self.db.query(ItemRevision).filter(
            ItemRevision.item_id == revision.item_id,
            ItemRevision.status == RevisionStatus.RELEASED,
            ItemRevision.id != revision_id
        ).first()
        
        if existing_released and not self.config.auto_obsolete_on_release:
            result.add_issue(ValidationIssue(
                code="EXISTING_RELEASED",
                message=f"Item already has a released revision ({existing_released.code})",
                severity=ValidationSeverity.WARNING,
                details={"existing_revision_id": existing_released.id, "existing_code": existing_released.code},
            ))
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# REVISION WORKFLOW ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class RevisionWorkflowEngine:
    """
    Engine for revision workflow management.
    
    Handles state transitions:
    - Draft → Released
    - Released → Obsolete
    """
    
    def __init__(self, config: PDMConfig, db: Session):
        self.config = config
        self.db = db
        self.release_validator = ReleaseValidationEngine(config, db)
    
    def release_revision(
        self,
        revision_id: int,
        released_by: str = "system",
        force: bool = False,
    ) -> Tuple[bool, Optional[ItemRevision], ValidationResult]:
        """
        Release a revision.
        
        Args:
            revision_id: ID of revision to release
            released_by: User performing the release
            force: If True, release even with warnings
        
        Returns:
            (success, revision, validation_result)
        """
        # Validate first
        validation = self.release_validator.validate_for_release(revision_id)
        
        if not validation.valid:
            return False, None, validation
        
        if validation.warnings_count > 0 and not force:
            return False, None, validation
        
        # Perform release
        try:
            revision = self.db.query(ItemRevision).filter(
                ItemRevision.id == revision_id
            ).first()
            
            if not revision:
                validation.add_issue(ValidationIssue(
                    code="REVISION_NOT_FOUND",
                    message=f"Revision {revision_id} not found",
                    severity=ValidationSeverity.ERROR,
                ))
                return False, None, validation
            
            now = datetime.utcnow()
            prev_released = []
            
            # Obsolete previous released revisions
            if self.config.auto_obsolete_on_release:
                prev_released = self.db.query(ItemRevision).filter(
                    ItemRevision.item_id == revision.item_id,
                    ItemRevision.status == RevisionStatus.RELEASED,
                    ItemRevision.id != revision_id
                ).all()
                
                for prev in prev_released:
                    prev.status = RevisionStatus.OBSOLETE
                    prev.effective_to = now
                    prev.updated_at = now
                    logger.info(f"Obsoleted revision {prev.id} ({prev.code})")
            
            # Release current revision
            revision.status = RevisionStatus.RELEASED
            revision.effective_from = now
            revision.updated_at = now
            
            self.db.commit()
            self.db.refresh(revision)
            
            logger.info(f"Released revision {revision.id} ({revision.code}) by {released_by}")
            
            # Signal affected orders and notify stakeholders
            self._notify_revision_release(revision, prev_released)
            
            return True, revision, validation
        
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to release revision {revision_id}: {e}")
            validation.add_issue(ValidationIssue(
                code="RELEASE_ERROR",
                message=f"Release failed: {str(e)}",
                severity=ValidationSeverity.ERROR,
            ))
            return False, None, validation
    
    def _notify_revision_release(
        self,
        released_revision: ItemRevision,
        obsoleted_revisions: List[ItemRevision],
    ) -> None:
        """
        Notify stakeholders and signal affected orders when a revision is released.
        
        As specified: "o sistema sinalize ordens abertas ou stock em curso afetado 
        por uma alteração de revisão (evitando uso inadvertido de versões antigas)"
        """
        item = self.db.query(Item).filter(Item.id == released_revision.item_id).first()
        if not item:
            return
        
        # Build notification message
        notification = {
            "type": "revision_released",
            "item_sku": item.sku,
            "item_name": item.name,
            "released_revision": {
                "id": released_revision.id,
                "code": released_revision.code,
            },
            "obsoleted_revisions": [
                {"id": rev.id, "code": rev.code} for rev in obsoleted_revisions
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Signal affected production orders (integration point)
        # TODO: Integrate with ProdPlan module to query open orders
        # affected_orders = query_production_orders(item_id=item.id, revision_id__in=[rev.id for rev in obsoleted_revisions])
        # for order in affected_orders:
        #     signal_order_revision_change(order.id, new_revision_id=released_revision.id)
        
        # Signal affected inventory (integration point)
        # TODO: Integrate with SmartInventory module to flag stock with old revision
        # affected_stock = query_inventory(item_id=item.id, revision_id__in=[rev.id for rev in obsoleted_revisions])
        # for stock in affected_stock:
        #     flag_stock_revision_obsolete(stock.id, obsolete_revision_id=stock.revision_id)
        
        logger.info(
            f"Revision release notification: {item.sku}:{released_revision.code} released, "
            f"{len(obsoleted_revisions)} revisions obsoleted"
        )
        
        # Store notification for API access
        # In a real system, this would be sent to a message queue or notification service
        # For now, we log it and it can be queried via API
    
    def obsolete_revision(
        self,
        revision_id: int,
        obsoleted_by: str = "system",
        reason: Optional[str] = None,
    ) -> Tuple[bool, Optional[ItemRevision]]:
        """
        Mark a revision as obsolete.
        """
        try:
            revision = self.db.query(ItemRevision).filter(
                ItemRevision.id == revision_id
            ).first()
            
            if not revision:
                return False, None
            
            if revision.status == RevisionStatus.OBSOLETE:
                return True, revision  # Already obsolete
            
            now = datetime.utcnow()
            revision.status = RevisionStatus.OBSOLETE
            revision.effective_to = now
            revision.updated_at = now
            if reason:
                revision.notes = f"{revision.notes or ''}\nObsolete reason: {reason}"
            
            self.db.commit()
            self.db.refresh(revision)
            
            logger.info(f"Obsoleted revision {revision.id} ({revision.code}) by {obsoleted_by}")
            
            return True, revision
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error obsoleting revision {revision_id}: {e}")
            return False, None
    
    def create_new_revision(
        self,
        item_id: int,
        code: Optional[str] = None,
        copy_from_revision_id: Optional[int] = None,
        created_by: str = "system",
    ) -> Optional[ItemRevision]:
        """
        Create a new Draft revision for an item.
        
        Optionally copies BOM and routing from another revision.
        """
        try:
            item = self.db.query(Item).filter(Item.id == item_id).first()
            if not item:
                logger.error(f"Item {item_id} not found")
                return None
            
            # Generate revision code if not provided
            if not code:
                existing_revs = self.db.query(ItemRevision).filter(
                    ItemRevision.item_id == item_id
                ).count()
                code = chr(ord('A') + existing_revs) if existing_revs < 26 else f"{existing_revs + 1:02d}"
            
            # Create new revision
            new_rev = ItemRevision(
                item_id=item_id,
                code=code,
                status=RevisionStatus.DRAFT,
                notes=f"Created by {created_by}",
            )
            
            self.db.add(new_rev)
            self.db.flush()  # Get ID
            
            # Copy from existing revision if requested
            if copy_from_revision_id:
                self._copy_revision_content(copy_from_revision_id, new_rev.id)
            
            self.db.commit()
            self.db.refresh(new_rev)
            
            logger.info(f"Created new revision {new_rev.id} ({new_rev.code}) for item {item_id}")
            
            return new_rev
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating new revision for item {item_id}: {e}")
            return None
    
    def _copy_revision_content(self, from_revision_id: int, to_revision_id: int) -> None:
        """Copy BOM and routing from one revision to another."""
        # Copy BOM
        bom_lines = self.db.query(BomLine).filter(
            BomLine.parent_revision_id == from_revision_id
        ).all()
        
        for line in bom_lines:
            new_line = BomLine(
                parent_revision_id=to_revision_id,
                component_revision_id=line.component_revision_id,
                qty_per_unit=line.qty_per_unit,
                scrap_rate=line.scrap_rate,
                position=line.position,
                notes=line.notes,
            )
            self.db.add(new_line)
        
        # Copy routing
        operations = self.db.query(RoutingOperation).filter(
            RoutingOperation.revision_id == from_revision_id
        ).all()
        
        for op in operations:
            new_op = RoutingOperation(
                revision_id=to_revision_id,
                op_code=op.op_code,
                sequence=op.sequence,
                machine_group=op.machine_group,
                nominal_setup_time=op.nominal_setup_time,
                nominal_cycle_time=op.nominal_cycle_time,
                tool_id=op.tool_id,
                description=op.description,
                name=op.name,
                is_critical=op.is_critical,
                requires_inspection=op.requires_inspection,
            )
            self.db.add(new_op)


# ═══════════════════════════════════════════════════════════════════════════════
# ECR/ECO ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ECREngine:
    """
    Engine for ECR/ECO management and impact analysis.
    """
    
    def __init__(self, config: PDMConfig, db: Session):
        self.config = config
        self.db = db
        self.bom_engine = BomValidationEngine(config, db)
    
    def create_ecr(
        self,
        item_id: int,
        title: str,
        description: str,
        reason: Optional[str] = None,
        priority: str = "MEDIUM",
        requested_by: str = "system",
    ) -> Optional[ECR]:
        """Create a new ECR."""
        try:
            ecr = ECR(
                item_id=item_id,
                title=title,
                description=description,
                reason=reason,
                priority=priority,
                status=ECRStatus.OPEN,
                requested_by=requested_by,
            )
            
            self.db.add(ecr)
            self.db.commit()
            self.db.refresh(ecr)
            
            logger.info(f"Created ECR {ecr.id} for item {item_id}")
            
            return ecr
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating ECR: {e}")
            return None
    
    def analyze_impact(self, item_id: int, revision_id: Optional[int] = None) -> ImpactAnalysis:
        """
        Analyze impact of changing an item/revision.
        
        Finds:
        - Other items that use this component in their BOM
        - Open production orders
        - WIP inventory
        - Finished goods inventory
        - DPP records
        """
        impact = ImpactAnalysis(
            affected_items=[],
            open_production_orders=[],
            wip_inventory=[],
            finished_inventory=[],
            affected_dpps=[],
        )
        
        # Find items that use this as a component
        if revision_id:
            usages = self.db.query(BomLine).filter(
                BomLine.component_revision_id == revision_id
            ).all()
        else:
            # Find all revisions of item
            revisions = self.db.query(ItemRevision).filter(
                ItemRevision.item_id == item_id
            ).all()
            rev_ids = [r.id for r in revisions]
            
            usages = self.db.query(BomLine).filter(
                BomLine.component_revision_id.in_(rev_ids)
            ).all() if rev_ids else []
        
        # Build affected items list
        seen_items = set()
        for usage in usages:
            parent_rev = self.db.query(ItemRevision).filter(
                ItemRevision.id == usage.parent_revision_id
            ).first()
            
            if parent_rev and parent_rev.item_id not in seen_items:
                seen_items.add(parent_rev.item_id)
                parent_item = self.db.query(Item).filter(
                    Item.id == parent_rev.item_id
                ).first()
                
                if parent_item:
                    impact.affected_items.append({
                        "item_id": parent_item.id,
                        "sku": parent_item.sku,
                        "name": parent_item.name,
                        "revision_id": parent_rev.id,
                        "revision_code": parent_rev.code,
                        "revision_status": parent_rev.status.value,
                    })
        
        # TODO: Query production orders, inventory from respective modules
        # These would require integration with ProdPlan and SmartInventory
        
        # Find affected DPPs
        try:
            from .dpp_models import DppRecord
            
            if revision_id:
                dpps = self.db.query(DppRecord).filter(
                    DppRecord.revision_id == revision_id
                ).all()
            else:
                revisions = self.db.query(ItemRevision).filter(
                    ItemRevision.item_id == item_id
                ).all()
                rev_ids = [r.id for r in revisions]
                dpps = self.db.query(DppRecord).filter(
                    DppRecord.revision_id.in_(rev_ids)
                ).all() if rev_ids else []
            
            for dpp in dpps:
                impact.affected_dpps.append({
                    "dpp_id": dpp.id,
                    "gtin": dpp.gtin,
                    "status": dpp.status,
                })
        except Exception:
            pass  # DppRecord might not exist
        
        impact.total_affected_count = (
            len(impact.affected_items) +
            len(impact.open_production_orders) +
            len(impact.affected_dpps)
        )
        
        return impact
    
    def create_eco(
        self,
        ecr_id: int,
        from_revision_id: Optional[int],
        to_revision_id: int,
        approved_by: str = "system",
    ) -> Optional[ECO]:
        """Create an ECO from an ECR."""
        try:
            eco = ECO(
                ecr_id=ecr_id,
                from_revision_id=from_revision_id,
                to_revision_id=to_revision_id,
                approved_by=approved_by,
                approved_at=datetime.utcnow(),
            )
            
            self.db.add(eco)
            self.db.commit()
            self.db.refresh(eco)
            
            logger.info(f"Created ECO {eco.id} for ECR {ecr_id}")
            
            return eco
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating ECO: {e}")
            return None
    
    def implement_eco(
        self,
        eco_id: int,
        implementation_notes: Optional[str] = None,
    ) -> Tuple[bool, Optional[ECO]]:
        """Mark an ECO as implemented."""
        try:
            eco = self.db.query(ECO).filter(ECO.id == eco_id).first()
            if not eco:
                return False, None
            
            eco.implemented_at = datetime.utcnow()
            eco.implementation_notes = implementation_notes
            
            # Close the ECR
            ecr = self.db.query(ECR).filter(ECR.id == eco.ecr_id).first()
            if ecr:
                ecr.status = ECRStatus.CLOSED
                ecr.closed_at = datetime.utcnow()
            
            self.db.commit()
            self.db.refresh(eco)
            
            logger.info(f"Implemented ECO {eco_id}")
            
            return True, eco
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error implementing ECO: {e}")
            return False, None


# ═══════════════════════════════════════════════════════════════════════════════
# REVISION COMPARISON ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class RevisionComparisonEngine:
    """
    Engine for comparing revisions.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def compare_revisions(self, from_revision_id: int, to_revision_id: int) -> RevisionDiff:
        """
        Compare two revisions and return differences.
        """
        from_rev = self.db.query(ItemRevision).filter(
            ItemRevision.id == from_revision_id
        ).first()
        to_rev = self.db.query(ItemRevision).filter(
            ItemRevision.id == to_revision_id
        ).first()
        
        diff = RevisionDiff(
            from_revision=from_rev.code if from_rev else "",
            to_revision=to_rev.code if to_rev else "",
        )
        
        # Compare BOM
        from_bom = {
            line.component_revision_id: line
            for line in self.db.query(BomLine).filter(
                BomLine.parent_revision_id == from_revision_id
            ).all()
        } if from_rev else {}
        
        to_bom = {
            line.component_revision_id: line
            for line in self.db.query(BomLine).filter(
                BomLine.parent_revision_id == to_revision_id
            ).all()
        } if to_rev else {}
        
        # Find added, removed, changed
        for comp_id, line in to_bom.items():
            comp_rev = self.db.query(ItemRevision).filter(
                ItemRevision.id == comp_id
            ).first()
            comp_item = self.db.query(Item).filter(
                Item.id == comp_rev.item_id
            ).first() if comp_rev else None
            
            line_info = {
                "component_revision_id": comp_id,
                "sku": comp_item.sku if comp_item else "",
                "qty": line.qty_per_unit,
            }
            
            if comp_id not in from_bom:
                diff.bom_added.append(line_info)
            elif from_bom[comp_id].qty_per_unit != line.qty_per_unit:
                line_info["old_qty"] = from_bom[comp_id].qty_per_unit
                diff.bom_changed.append(line_info)
        
        for comp_id, line in from_bom.items():
            if comp_id not in to_bom:
                comp_rev = self.db.query(ItemRevision).filter(
                    ItemRevision.id == comp_id
                ).first()
                comp_item = self.db.query(Item).filter(
                    Item.id == comp_rev.item_id
                ).first() if comp_rev else None
                
                diff.bom_removed.append({
                    "component_revision_id": comp_id,
                    "sku": comp_item.sku if comp_item else "",
                    "qty": line.qty_per_unit,
                })
        
        # Compare Routing
        from_routing = {
            op.sequence: op
            for op in self.db.query(RoutingOperation).filter(
                RoutingOperation.revision_id == from_revision_id
            ).all()
        } if from_rev else {}
        
        to_routing = {
            op.sequence: op
            for op in self.db.query(RoutingOperation).filter(
                RoutingOperation.revision_id == to_revision_id
            ).all()
        } if to_rev else {}
        
        for seq, op in to_routing.items():
            op_info = {
                "sequence": seq,
                "op_code": op.op_code,
                "cycle_time": op.nominal_cycle_time,
                "setup_time": op.nominal_setup_time,
            }
            
            if seq not in from_routing:
                diff.routing_added.append(op_info)
            else:
                old_op = from_routing[seq]
                if (old_op.op_code != op.op_code or
                    old_op.nominal_cycle_time != op.nominal_cycle_time or
                    old_op.nominal_setup_time != op.nominal_setup_time):
                    op_info["old_op_code"] = old_op.op_code
                    op_info["old_cycle_time"] = old_op.nominal_cycle_time
                    op_info["old_setup_time"] = old_op.nominal_setup_time
                    diff.routing_changed.append(op_info)
        
        for seq, op in from_routing.items():
            if seq not in to_routing:
                diff.routing_removed.append({
                    "sequence": seq,
                    "op_code": op.op_code,
                    "cycle_time": op.nominal_cycle_time,
                    "setup_time": op.nominal_setup_time,
                })
        
        return diff


# ═══════════════════════════════════════════════════════════════════════════════
# PDM SERVICE (FACADE)
# ═══════════════════════════════════════════════════════════════════════════════

class PDMService:
    """
    Main PDM service facade.
    
    Provides high-level operations for PDM management.
    
    Usage:
        service = PDMService()
        
        # Get current revision
        rev = service.get_current_revision(item_id)
        
        # Get BOM
        bom = service.get_bom(item_id, rev.code)
        
        # Release revision
        success, rev, validation = service.release_revision(revision_id)
    """
    
    def __init__(self, config: Optional[PDMConfig] = None, db: Optional[Session] = None):
        self.config = config or PDMConfig()
        self._db = db
    
    @property
    def db(self) -> Session:
        if self._db is None:
            self._db = SessionLocal()
        return self._db
    
    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None
    
    # Item operations
    def get_item(self, item_id: Optional[int] = None, sku: Optional[str] = None) -> Optional[Item]:
        """Get item by ID or SKU."""
        if item_id:
            return self.db.query(Item).filter(Item.id == item_id).first()
        if sku:
            return self.db.query(Item).filter(Item.sku == sku).first()
        return None
    
    def get_current_revision(self, item_id: int) -> Optional[ItemRevision]:
        """Get the currently released revision for an item."""
        return self.db.query(ItemRevision).filter(
            ItemRevision.item_id == item_id,
            ItemRevision.status == RevisionStatus.RELEASED
        ).first()
    
    # BOM operations
    def get_bom(self, item_id: int, revision_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get BOM for an item/revision."""
        if revision_code:
            revision = self.db.query(ItemRevision).filter(
                ItemRevision.item_id == item_id,
                ItemRevision.code == revision_code
            ).first()
        else:
            revision = self.get_current_revision(item_id)
        
        if not revision:
            return []
        
        bom_lines = self.db.query(BomLine).filter(
            BomLine.parent_revision_id == revision.id
        ).all()
        
        result = []
        for line in bom_lines:
            comp_rev = self.db.query(ItemRevision).filter(
                ItemRevision.id == line.component_revision_id
            ).first()
            comp_item = self.db.query(Item).filter(
                Item.id == comp_rev.item_id
            ).first() if comp_rev else None
            
            result.append({
                "bom_line_id": line.id,
                "component_revision_id": line.component_revision_id,
                "component_item_id": comp_item.id if comp_item else None,
                "component_sku": comp_item.sku if comp_item else "",
                "component_name": comp_item.name if comp_item else "",
                "revision_code": comp_rev.code if comp_rev else "",
                "qty_per_unit": line.qty_per_unit,
                "scrap_rate": line.scrap_rate,
                "unit": comp_item.unit if comp_item else "pcs",
            })
        
        return result
    
    # Routing operations
    def get_routing(self, item_id: int, revision_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get routing for an item/revision."""
        if revision_code:
            revision = self.db.query(ItemRevision).filter(
                ItemRevision.item_id == item_id,
                ItemRevision.code == revision_code
            ).first()
        else:
            revision = self.get_current_revision(item_id)
        
        if not revision:
            return []
        
        operations = self.db.query(RoutingOperation).filter(
            RoutingOperation.revision_id == revision.id
        ).order_by(RoutingOperation.sequence).all()
        
        return [{
            "operation_id": op.id,
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
    
    # Workflow operations
    def release_revision(
        self,
        revision_id: int,
        released_by: str = "system",
        force: bool = False,
    ) -> Tuple[bool, Optional[ItemRevision], ValidationResult]:
        """Release a revision."""
        engine = RevisionWorkflowEngine(self.config, self.db)
        return engine.release_revision(revision_id, released_by, force)
    
    def create_new_revision(
        self,
        item_id: int,
        code: Optional[str] = None,
        copy_from: Optional[int] = None,
        created_by: str = "system",
    ) -> Optional[ItemRevision]:
        """Create a new draft revision."""
        engine = RevisionWorkflowEngine(self.config, self.db)
        return engine.create_new_revision(item_id, code, copy_from, created_by)
    
    # Validation operations
    def validate_for_release(self, revision_id: int) -> ValidationResult:
        """Validate a revision for release."""
        engine = ReleaseValidationEngine(self.config, self.db)
        return engine.validate_for_release(revision_id)
    
    # BOM operations
    def explode_bom(self, revision_id: int, qty: float = 1.0) -> BomExplosion:
        """Explode BOM recursively."""
        engine = BomValidationEngine(self.config, self.db)
        return engine.explode_bom(revision_id, qty)
    
    # ECR/ECO operations
    def create_ecr(
        self,
        item_id: int,
        title: str,
        description: str,
        **kwargs
    ) -> Optional[ECR]:
        """Create an ECR."""
        engine = ECREngine(self.config, self.db)
        return engine.create_ecr(item_id, title, description, **kwargs)
    
    def analyze_impact(self, item_id: int, revision_id: Optional[int] = None) -> ImpactAnalysis:
        """Analyze change impact."""
        engine = ECREngine(self.config, self.db)
        return engine.analyze_impact(item_id, revision_id)
    
    # Comparison operations
    def compare_revisions(self, from_revision_id: int, to_revision_id: int) -> RevisionDiff:
        """Compare two revisions."""
        engine = RevisionComparisonEngine(self.db)
        return engine.compare_revisions(from_revision_id, to_revision_id)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_pdm_service() -> PDMService:
    """Get a PDM service instance."""
    return PDMService()


def get_current_revision(item_id: int) -> Optional[ItemRevision]:
    """Convenience function to get current revision."""
    service = get_pdm_service()
    try:
        return service.get_current_revision(item_id)
    finally:
        service.close()


def get_bom(item_id: int, revision_code: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convenience function to get BOM."""
    service = get_pdm_service()
    try:
        return service.get_bom(item_id, revision_code)
    finally:
        service.close()


def get_routing(item_id: int, revision_code: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convenience function to get routing."""
    service = get_pdm_service()
    try:
        return service.get_routing(item_id, revision_code)
    finally:
        service.close()


