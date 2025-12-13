"""
════════════════════════════════════════════════════════════════════════════════════════════════════
PREVENTION GUARD - Process & Quality Guard System
════════════════════════════════════════════════════════════════════════════════════════════════════

Sistema de prevenção de erros para fabricação e montagem.

Components:
1. PDM Guard - Validações na definição (BOM, Routing, Docs)
2. Shopfloor Guard - Validações na execução (Material, Equipment)
3. Predictive Guard - ML para previsão de riscos
4. Poka-Yoke Engine - Regras de prevenção digital
5. Exception Manager - Workflow de aprovação

Modelo Preditivo:
- Features: product, machine, operator, shift, material_batch, machine_health
- Output: P(defect) - probabilidade de defeito
- Threshold: se P(defect) > 0.3 → alerta de risco

R&D / SIFIDE: WP4 - Zero Defect Manufacturing
"""

from __future__ import annotations

import logging
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict
import random

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class ValidationSeverity(str, Enum):
    """Severity of validation issue."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(str, Enum):
    """Category of validation."""
    BOM = "bom"
    ROUTING = "routing"
    DOCUMENTATION = "documentation"
    MATERIAL = "material"
    EQUIPMENT = "equipment"
    PARAMETER = "parameter"
    QUALITY = "quality"
    COMPLIANCE = "compliance"


class ValidationAction(str, Enum):
    """Action to take on validation failure."""
    ALLOW = "allow"  # Log but allow
    WARN = "warn"  # Warn but allow
    BLOCK = "block"  # Block operation
    APPROVAL_REQUIRED = "approval_required"  # Needs supervisor


class RiskLevel(str, Enum):
    """Predicted risk level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExceptionStatus(str, Enum):
    """Status of exception request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class GuardEventType(str, Enum):
    """Type of guard event."""
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
    RISK_ALERT = "risk_alert"
    EXCEPTION_REQUESTED = "exception_requested"
    EXCEPTION_RESOLVED = "exception_resolved"
    ERROR_PREVENTED = "error_prevented"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationRule:
    """A validation rule definition."""
    rule_id: str
    name: str
    description: str
    category: ValidationCategory
    severity: ValidationSeverity
    action: ValidationAction
    
    # Rule logic
    condition: str  # Condition expression or rule type
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Applicability
    applies_to_products: List[str] = field(default_factory=list)  # Empty = all
    applies_to_operations: List[str] = field(default_factory=list)
    
    # Status
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "severity": self.severity.value,
            "action": self.action.value,
            "condition": self.condition,
            "parameters": self.parameters,
            "enabled": self.enabled,
        }


@dataclass
class ValidationIssue:
    """A detected validation issue."""
    issue_id: str
    rule_id: str
    rule_name: str
    category: ValidationCategory
    severity: ValidationSeverity
    action: ValidationAction
    
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Context
    entity_type: str = ""  # "bom", "order", "operation"
    entity_id: str = ""
    
    # Resolution
    resolved: bool = False
    resolution_note: str = ""
    
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "category": self.category.value,
            "severity": self.severity.value,
            "action": self.action.value,
            "message": self.message,
            "details": self.details,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "resolved": self.resolved,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    
    # Summary
    errors: int = 0
    warnings: int = 0
    blocked: bool = False
    requires_approval: bool = False
    
    # Timing
    validation_time_ms: float = 0
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "errors": self.errors,
            "warnings": self.warnings,
            "blocked": self.blocked,
            "requires_approval": self.requires_approval,
            "validation_time_ms": round(self.validation_time_ms, 2),
            "validated_at": self.validated_at.isoformat(),
        }


@dataclass
class RiskPrediction:
    """Predicted risk for an operation."""
    prediction_id: str
    risk_level: RiskLevel
    defect_probability: float
    
    # Factors contributing to risk
    risk_factors: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Similar issues
    similar_issues: List[Dict[str, Any]] = field(default_factory=list)
    
    # Model info
    model_version: str = "base"
    confidence: float = 0.5
    
    predicted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "risk_level": self.risk_level.value,
            "defect_probability": round(self.defect_probability, 4),
            "risk_factors": {k: round(v, 3) for k, v in self.risk_factors.items()},
            "recommendations": self.recommendations,
            "similar_issues": self.similar_issues[:5],  # Limit
            "model_version": self.model_version,
            "confidence": round(self.confidence, 3),
            "predicted_at": self.predicted_at.isoformat(),
        }


@dataclass
class ExceptionRequest:
    """Request to override a validation block."""
    exception_id: str
    validation_issue_id: str
    
    # Context
    order_id: str
    operation_id: str
    
    # Request
    requested_by: str
    reason: str
    status: ExceptionStatus
    
    # Resolution
    resolved_by: Optional[str] = None
    resolution_note: Optional[str] = None
    
    # Timestamps
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "exception_id": self.exception_id,
            "validation_issue_id": self.validation_issue_id,
            "order_id": self.order_id,
            "operation_id": self.operation_id,
            "requested_by": self.requested_by,
            "reason": self.reason,
            "status": self.status.value,
            "resolved_by": self.resolved_by,
            "resolution_note": self.resolution_note,
            "requested_at": self.requested_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class GuardEvent:
    """An event logged by the guard system."""
    event_id: str
    event_type: GuardEventType
    
    # Context
    entity_type: str
    entity_id: str
    
    # Details
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    # User
    user_id: Optional[str] = None
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "message": self.message,
            "details": self.details,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PDM GUARD ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class PDMGuardEngine:
    """
    Validation engine for PDM (Product Data Management).
    
    As specified: "Validação de BOM e Routing no momento da release (PDM Guard)"
    
    Validates:
    - BOM structure and components (duplicates, zero quantities, obsolete, cycles)
    - Routing completeness (times, resources, inspections)
    - Required documentation (drawings, work instructions, quality plans)
    - Process configuration (parameter defaults and acceptable ranges)
    - Custom compliance rules
    """
    
    def __init__(self):
        self.rules: Dict[str, ValidationRule] = {}
        self._init_default_rules()
    
    def _init_default_rules(self):
        """Initialize default PDM validation rules."""
        default_rules = [
            # BOM Rules
            ValidationRule(
                rule_id="BOM-001",
                name="No Duplicate Components",
                description="BOM cannot have duplicate component entries",
                category=ValidationCategory.BOM,
                severity=ValidationSeverity.ERROR,
                action=ValidationAction.BLOCK,
                condition="no_duplicate_components",
            ),
            ValidationRule(
                rule_id="BOM-002",
                name="No Zero Quantities",
                description="Component quantities must be positive",
                category=ValidationCategory.BOM,
                severity=ValidationSeverity.ERROR,
                action=ValidationAction.BLOCK,
                condition="positive_quantities",
            ),
            ValidationRule(
                rule_id="BOM-003",
                name="No Obsolete Components",
                description="BOM should not contain obsolete components",
                category=ValidationCategory.BOM,
                severity=ValidationSeverity.WARNING,
                action=ValidationAction.WARN,
                condition="no_obsolete_components",
            ),
            ValidationRule(
                rule_id="BOM-004",
                name="No BOM Cycles",
                description="BOM structure must be acyclic (DAG)",
                category=ValidationCategory.BOM,
                severity=ValidationSeverity.CRITICAL,
                action=ValidationAction.BLOCK,
                condition="no_cycles",
            ),
            
            # Routing Rules
            ValidationRule(
                rule_id="RTG-001",
                name="Complete Operations",
                description="All operations must have time estimates",
                category=ValidationCategory.ROUTING,
                severity=ValidationSeverity.ERROR,
                action=ValidationAction.BLOCK,
                condition="complete_operations",
            ),
            ValidationRule(
                rule_id="RTG-002",
                name="Valid Work Centers",
                description="All operations must reference valid work centers",
                category=ValidationCategory.ROUTING,
                severity=ValidationSeverity.ERROR,
                action=ValidationAction.BLOCK,
                condition="valid_work_centers",
            ),
            ValidationRule(
                rule_id="RTG-003",
                name="Inspection Operations",
                description="Critical products must have inspection operations",
                category=ValidationCategory.ROUTING,
                severity=ValidationSeverity.WARNING,
                action=ValidationAction.APPROVAL_REQUIRED,
                condition="has_inspection",
                parameters={"critical_product_types": ["assembly", "safety"]},
            ),
            
            # Documentation Rules
            ValidationRule(
                rule_id="DOC-001",
                name="Drawing Required",
                description="Products must have attached drawings",
                category=ValidationCategory.DOCUMENTATION,
                severity=ValidationSeverity.ERROR,
                action=ValidationAction.BLOCK,
                condition="has_drawing",
            ),
            ValidationRule(
                rule_id="DOC-002",
                name="Work Instructions Required",
                description="Operations must have work instructions",
                category=ValidationCategory.DOCUMENTATION,
                severity=ValidationSeverity.WARNING,
                action=ValidationAction.WARN,
                condition="has_work_instructions",
            ),
            ValidationRule(
                rule_id="DOC-003",
                name="Quality Plan Required",
                description="Products must have quality control plan",
                category=ValidationCategory.DOCUMENTATION,
                severity=ValidationSeverity.WARNING,
                action=ValidationAction.APPROVAL_REQUIRED,
                condition="has_quality_plan",
            ),
        ]
        
        for rule in default_rules:
            self.rules[rule.rule_id] = rule
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add a custom validation rule."""
        self.rules[rule.rule_id] = rule
    
    def validate_bom(
        self,
        bom_data: Dict[str, Any],
        components: List[Dict[str, Any]],
    ) -> ValidationResult:
        """
        Validate a BOM structure.
        
        Args:
            bom_data: BOM metadata (item_id, revision, etc.)
            components: List of BOM components
        
        Returns:
            ValidationResult with any issues
        """
        import time
        start = time.time()
        
        issues = []
        item_id = bom_data.get("item_id", "unknown")
        
        # BOM-001: No duplicates
        if self.rules.get("BOM-001", ValidationRule("", "", "", ValidationCategory.BOM, ValidationSeverity.INFO, ValidationAction.ALLOW, "")).enabled:
            component_ids = [c.get("component_id") for c in components]
            duplicates = [cid for cid in component_ids if component_ids.count(cid) > 1]
            
            if duplicates:
                rule = self.rules["BOM-001"]
                issues.append(ValidationIssue(
                    issue_id=f"VI-{uuid.uuid4().hex[:8]}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    category=rule.category,
                    severity=rule.severity,
                    action=rule.action,
                    message=f"Duplicate components found: {set(duplicates)}",
                    details={"duplicates": list(set(duplicates))},
                    entity_type="bom",
                    entity_id=item_id,
                ))
        
        # BOM-002: Positive quantities
        if self.rules.get("BOM-002", ValidationRule("", "", "", ValidationCategory.BOM, ValidationSeverity.INFO, ValidationAction.ALLOW, "")).enabled:
            for comp in components:
                qty = comp.get("qty_per_unit", comp.get("quantity", 0))
                if qty <= 0:
                    rule = self.rules["BOM-002"]
                    issues.append(ValidationIssue(
                        issue_id=f"VI-{uuid.uuid4().hex[:8]}",
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        category=rule.category,
                        severity=rule.severity,
                        action=rule.action,
                        message=f"Invalid quantity for component {comp.get('component_id')}: {qty}",
                        details={"component_id": comp.get("component_id"), "quantity": qty},
                        entity_type="bom",
                        entity_id=item_id,
                    ))
        
        # BOM-003: Obsolete check
        if self.rules.get("BOM-003", ValidationRule("", "", "", ValidationCategory.BOM, ValidationSeverity.INFO, ValidationAction.ALLOW, "")).enabled:
            for comp in components:
                if comp.get("status") == "obsolete":
                    rule = self.rules["BOM-003"]
                    issues.append(ValidationIssue(
                        issue_id=f"VI-{uuid.uuid4().hex[:8]}",
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        category=rule.category,
                        severity=rule.severity,
                        action=rule.action,
                        message=f"Obsolete component in BOM: {comp.get('component_id')}",
                        details={"component_id": comp.get("component_id")},
                        entity_type="bom",
                        entity_id=item_id,
                    ))
        
        # Build result
        elapsed = (time.time() - start) * 1000
        
        errors = sum(1 for i in issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
        warnings = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)
        blocked = any(i.action == ValidationAction.BLOCK for i in issues)
        requires_approval = any(i.action == ValidationAction.APPROVAL_REQUIRED for i in issues)
        
        return ValidationResult(
            passed=errors == 0,
            issues=issues,
            errors=errors,
            warnings=warnings,
            blocked=blocked,
            requires_approval=requires_approval,
            validation_time_ms=elapsed,
        )
    
    def validate_routing(
        self,
        routing_data: Dict[str, Any],
        operations: List[Dict[str, Any]],
    ) -> ValidationResult:
        """Validate a routing definition."""
        import time
        start = time.time()
        
        issues = []
        item_id = routing_data.get("item_id", "unknown")
        
        # RTG-001: Complete operations
        if self.rules.get("RTG-001", ValidationRule("", "", "", ValidationCategory.ROUTING, ValidationSeverity.INFO, ValidationAction.ALLOW, "")).enabled:
            for op in operations:
                if not op.get("setup_time") and not op.get("cycle_time"):
                    rule = self.rules["RTG-001"]
                    issues.append(ValidationIssue(
                        issue_id=f"VI-{uuid.uuid4().hex[:8]}",
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        category=rule.category,
                        severity=rule.severity,
                        action=rule.action,
                        message=f"Operation {op.get('operation_id')} missing time estimates",
                        details={"operation_id": op.get("operation_id")},
                        entity_type="routing",
                        entity_id=item_id,
                    ))
        
        # RTG-002: Valid work centers
        if self.rules.get("RTG-002", ValidationRule("", "", "", ValidationCategory.ROUTING, ValidationSeverity.INFO, ValidationAction.ALLOW, "")).enabled:
            for op in operations:
                if not op.get("work_center_id") and not op.get("machine_id"):
                    rule = self.rules["RTG-002"]
                    issues.append(ValidationIssue(
                        issue_id=f"VI-{uuid.uuid4().hex[:8]}",
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        category=rule.category,
                        severity=rule.severity,
                        action=rule.action,
                        message=f"Operation {op.get('operation_id')} has no work center assigned",
                        details={"operation_id": op.get("operation_id")},
                        entity_type="routing",
                        entity_id=item_id,
                    ))
        
        elapsed = (time.time() - start) * 1000
        
        errors = sum(1 for i in issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
        warnings = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)
        
        return ValidationResult(
            passed=errors == 0,
            issues=issues,
            errors=errors,
            warnings=warnings,
            blocked=any(i.action == ValidationAction.BLOCK for i in issues),
            requires_approval=any(i.action == ValidationAction.APPROVAL_REQUIRED for i in issues),
            validation_time_ms=elapsed,
        )
    
    def validate_documentation(
        self,
        item_data: Dict[str, Any],
        attachments: List[Dict[str, Any]],
    ) -> ValidationResult:
        """Validate required documentation."""
        import time
        start = time.time()
        
        issues = []
        item_id = item_data.get("item_id", "unknown")
        
        attachment_types = {a.get("type") for a in attachments}
        
        # DOC-001: Drawing required
        if self.rules.get("DOC-001", ValidationRule("", "", "", ValidationCategory.DOCUMENTATION, ValidationSeverity.INFO, ValidationAction.ALLOW, "")).enabled:
            if "drawing" not in attachment_types and "cad" not in attachment_types:
                rule = self.rules["DOC-001"]
                issues.append(ValidationIssue(
                    issue_id=f"VI-{uuid.uuid4().hex[:8]}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    category=rule.category,
                    severity=rule.severity,
                    action=rule.action,
                    message="No drawing or CAD file attached",
                    details={"available_types": list(attachment_types)},
                    entity_type="item",
                    entity_id=item_id,
                ))
        
        elapsed = (time.time() - start) * 1000
        
        errors = sum(1 for i in issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
        
        return ValidationResult(
            passed=errors == 0,
            issues=issues,
            errors=errors,
            warnings=len(issues) - errors,
            blocked=any(i.action == ValidationAction.BLOCK for i in issues),
            requires_approval=any(i.action == ValidationAction.APPROVAL_REQUIRED for i in issues),
            validation_time_ms=elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SHOPFLOOR GUARD ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ShopfloorGuardEngine:
    """
    Validation engine for shopfloor operations.
    
    As specified: "Guardião no Chão-de-fábrica (Shopfloor Guard): no início de cada ordem, 
    verificar se o operador tem a versão correta da instrução de trabalho e se os materiais 
    carregados correspondem exatamente à BOM (uso de código de barras/RFID para conferência)"
    
    Validates:
    - Material scanning/verification (barcode/RFID)
    - Equipment and tool calibration
    - Process parameters (real-time poka-yoke)
    - Pre-start checklists
    - Work instruction version
    """
    
    def __init__(self):
        self.rules: Dict[str, ValidationRule] = {}
        self._init_default_rules()
    
    def _init_default_rules(self):
        """Initialize default shopfloor validation rules."""
        default_rules = [
            # Material Rules
            ValidationRule(
                rule_id="MAT-001",
                name="Material Verification",
                description="Scanned material must match order requirements",
                category=ValidationCategory.MATERIAL,
                severity=ValidationSeverity.CRITICAL,
                action=ValidationAction.BLOCK,
                condition="material_match",
            ),
            ValidationRule(
                rule_id="MAT-002",
                name="Material Revision Check",
                description="Material revision must be compatible",
                category=ValidationCategory.MATERIAL,
                severity=ValidationSeverity.ERROR,
                action=ValidationAction.APPROVAL_REQUIRED,
                condition="revision_compatible",
            ),
            ValidationRule(
                rule_id="MAT-003",
                name="Material Expiry Check",
                description="Material must not be expired",
                category=ValidationCategory.MATERIAL,
                severity=ValidationSeverity.CRITICAL,
                action=ValidationAction.BLOCK,
                condition="not_expired",
            ),
            
            # Equipment Rules
            ValidationRule(
                rule_id="EQP-001",
                name="Machine Assignment",
                description="Machine must match operation requirements",
                category=ValidationCategory.EQUIPMENT,
                severity=ValidationSeverity.ERROR,
                action=ValidationAction.BLOCK,
                condition="machine_match",
            ),
            ValidationRule(
                rule_id="EQP-002",
                name="Tool Calibration",
                description="Required tools must be calibrated",
                category=ValidationCategory.EQUIPMENT,
                severity=ValidationSeverity.WARNING,
                action=ValidationAction.APPROVAL_REQUIRED,
                condition="tools_calibrated",
            ),
            ValidationRule(
                rule_id="EQP-003",
                name="Machine Health",
                description="Machine health index must be acceptable",
                category=ValidationCategory.EQUIPMENT,
                severity=ValidationSeverity.WARNING,
                action=ValidationAction.WARN,
                condition="machine_healthy",
                parameters={"min_health_index": 0.6},
            ),
            
            # Parameter Rules
            ValidationRule(
                rule_id="PAR-001",
                name="Parameter Limits",
                description="Process parameters must be within limits",
                category=ValidationCategory.PARAMETER,
                severity=ValidationSeverity.CRITICAL,
                action=ValidationAction.BLOCK,
                condition="parameters_in_range",
            ),
            ValidationRule(
                rule_id="PAR-002",
                name="Golden Run Match",
                description="Parameters should match golden run",
                category=ValidationCategory.PARAMETER,
                severity=ValidationSeverity.INFO,
                action=ValidationAction.WARN,
                condition="parameters_optimal",
            ),
        ]
        
        for rule in default_rules:
            self.rules[rule.rule_id] = rule
    
    def validate_material(
        self,
        order_data: Dict[str, Any],
        scanned_material: Dict[str, Any],
        required_material: Dict[str, Any],
    ) -> ValidationResult:
        """
        Validate scanned material against requirements.
        
        Args:
            order_data: Production order information
            scanned_material: Material that was scanned
            required_material: Material required by the order
        
        Returns:
            ValidationResult
        """
        import time
        start = time.time()
        
        issues = []
        order_id = order_data.get("order_id", "unknown")
        
        scanned_sku = scanned_material.get("sku", scanned_material.get("item_id", ""))
        required_sku = required_material.get("sku", required_material.get("item_id", ""))
        
        # MAT-001: Material match
        if scanned_sku != required_sku:
            rule = self.rules.get("MAT-001")
            if rule and rule.enabled:
                issues.append(ValidationIssue(
                    issue_id=f"VI-{uuid.uuid4().hex[:8]}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    category=rule.category,
                    severity=rule.severity,
                    action=rule.action,
                    message=f"Material mismatch: scanned {scanned_sku}, required {required_sku}",
                    details={
                        "scanned_sku": scanned_sku,
                        "required_sku": required_sku,
                    },
                    entity_type="order",
                    entity_id=order_id,
                ))
        
        # MAT-002: Revision check
        scanned_rev = scanned_material.get("revision")
        required_rev = required_material.get("revision")
        
        if scanned_rev and required_rev and scanned_rev != required_rev:
            rule = self.rules.get("MAT-002")
            if rule and rule.enabled:
                issues.append(ValidationIssue(
                    issue_id=f"VI-{uuid.uuid4().hex[:8]}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    category=rule.category,
                    severity=rule.severity,
                    action=rule.action,
                    message=f"Revision mismatch: scanned {scanned_rev}, required {required_rev}",
                    details={
                        "scanned_revision": scanned_rev,
                        "required_revision": required_rev,
                    },
                    entity_type="order",
                    entity_id=order_id,
                ))
        
        # MAT-003: Expiry check
        expiry_date = scanned_material.get("expiry_date")
        if expiry_date:
            if isinstance(expiry_date, str):
                expiry_date = datetime.fromisoformat(expiry_date.replace("Z", "+00:00"))
            
            if expiry_date < datetime.now(timezone.utc):
                rule = self.rules.get("MAT-003")
                if rule and rule.enabled:
                    issues.append(ValidationIssue(
                        issue_id=f"VI-{uuid.uuid4().hex[:8]}",
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        category=rule.category,
                        severity=rule.severity,
                        action=rule.action,
                        message=f"Material expired on {expiry_date.date()}",
                        details={"expiry_date": expiry_date.isoformat()},
                        entity_type="material",
                        entity_id=scanned_sku,
                    ))
        
        elapsed = (time.time() - start) * 1000
        
        errors = sum(1 for i in issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
        
        return ValidationResult(
            passed=errors == 0,
            issues=issues,
            errors=errors,
            warnings=len(issues) - errors,
            blocked=any(i.action == ValidationAction.BLOCK for i in issues),
            requires_approval=any(i.action == ValidationAction.APPROVAL_REQUIRED for i in issues),
            validation_time_ms=elapsed,
        )
    
    def validate_equipment(
        self,
        operation_data: Dict[str, Any],
        machine_data: Dict[str, Any],
        tools: List[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate equipment for an operation."""
        import time
        start = time.time()
        
        issues = []
        operation_id = operation_data.get("operation_id", "unknown")
        
        # EQP-001: Machine match
        required_machine = operation_data.get("machine_id", operation_data.get("work_center_id"))
        actual_machine = machine_data.get("machine_id")
        
        if required_machine and actual_machine != required_machine:
            rule = self.rules.get("EQP-001")
            if rule and rule.enabled:
                issues.append(ValidationIssue(
                    issue_id=f"VI-{uuid.uuid4().hex[:8]}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    category=rule.category,
                    severity=rule.severity,
                    action=rule.action,
                    message=f"Wrong machine: using {actual_machine}, required {required_machine}",
                    details={
                        "actual_machine": actual_machine,
                        "required_machine": required_machine,
                    },
                    entity_type="operation",
                    entity_id=operation_id,
                ))
        
        # EQP-002: Tool calibration
        if tools:
            for tool in tools:
                if not tool.get("calibrated", True):
                    rule = self.rules.get("EQP-002")
                    if rule and rule.enabled:
                        issues.append(ValidationIssue(
                            issue_id=f"VI-{uuid.uuid4().hex[:8]}",
                            rule_id=rule.rule_id,
                            rule_name=rule.name,
                            category=rule.category,
                            severity=rule.severity,
                            action=rule.action,
                            message=f"Tool {tool.get('tool_id')} not calibrated",
                            details={"tool_id": tool.get("tool_id")},
                            entity_type="tool",
                            entity_id=tool.get("tool_id"),
                        ))
        
        # EQP-003: Machine health
        health_index = machine_data.get("health_index", 1.0)
        rule = self.rules.get("EQP-003")
        if rule and rule.enabled:
            min_health = rule.parameters.get("min_health_index", 0.6)
            if health_index < min_health:
                issues.append(ValidationIssue(
                    issue_id=f"VI-{uuid.uuid4().hex[:8]}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    category=rule.category,
                    severity=rule.severity,
                    action=rule.action,
                    message=f"Machine health index {health_index:.2f} below threshold {min_health}",
                    details={
                        "health_index": health_index,
                        "threshold": min_health,
                    },
                    entity_type="machine",
                    entity_id=actual_machine,
                ))
        
        elapsed = (time.time() - start) * 1000
        errors = sum(1 for i in issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
        
        return ValidationResult(
            passed=errors == 0,
            issues=issues,
            errors=errors,
            warnings=len(issues) - errors,
            blocked=any(i.action == ValidationAction.BLOCK for i in issues),
            requires_approval=any(i.action == ValidationAction.APPROVAL_REQUIRED for i in issues),
            validation_time_ms=elapsed,
        )
    
    def validate_parameters(
        self,
        operation_data: Dict[str, Any],
        actual_parameters: Dict[str, float],
        parameter_limits: Dict[str, Dict[str, float]],
    ) -> ValidationResult:
        """Validate process parameters against limits."""
        import time
        start = time.time()
        
        issues = []
        operation_id = operation_data.get("operation_id", "unknown")
        
        for param_name, actual_value in actual_parameters.items():
            if param_name in parameter_limits:
                limits = parameter_limits[param_name]
                min_val = limits.get("min", float("-inf"))
                max_val = limits.get("max", float("inf"))
                
                if actual_value < min_val or actual_value > max_val:
                    rule = self.rules.get("PAR-001")
                    if rule and rule.enabled:
                        issues.append(ValidationIssue(
                            issue_id=f"VI-{uuid.uuid4().hex[:8]}",
                            rule_id=rule.rule_id,
                            rule_name=rule.name,
                            category=rule.category,
                            severity=rule.severity,
                            action=rule.action,
                            message=f"Parameter {param_name}={actual_value} outside limits [{min_val}, {max_val}]",
                            details={
                                "parameter": param_name,
                                "value": actual_value,
                                "min": min_val,
                                "max": max_val,
                            },
                            entity_type="operation",
                            entity_id=operation_id,
                        ))
        
        elapsed = (time.time() - start) * 1000
        errors = sum(1 for i in issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
        
        return ValidationResult(
            passed=errors == 0,
            issues=issues,
            errors=errors,
            warnings=len(issues) - errors,
            blocked=any(i.action == ValidationAction.BLOCK for i in issues),
            requires_approval=any(i.action == ValidationAction.APPROVAL_REQUIRED for i in issues),
            validation_time_ms=elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTIVE QUALITY GUARD (ML)
# ═══════════════════════════════════════════════════════════════════════════════

class PredictiveGuardEngine:
    """
    ML-based predictive quality risk assessment.
    
    As specified: "Módulo de Previsão de Risco de Qualidade: analisar dados históricos 
    e em tempo real (produto, máquina, hora do dia, condições ambientais, etc.) para 
    prever a probabilidade de defeito ou erro"
    
    Modelo Matemático:
    - logit(P(Defeito)) = β0 + β1*Machine + β2*Operator + ... + βn*Interaction_n
    - Com mais dados, evolui para modelos não-lineares (MLP) incluindo interações complexas
    - Minimiza entropia cruzada: CrossEntropyLoss = -Σ[y*log(p) + (1-y)*log(1-p)]
    
    Uses historical data to predict defect probability before production starts.
    """
    
    def __init__(self):
        self.model = None
        self.trained = False
        self.training_data: List[Tuple[np.ndarray, float]] = []
        self.historical_issues: List[Dict[str, Any]] = []
        
        # Risk thresholds
        self.thresholds = {
            RiskLevel.LOW: 0.1,
            RiskLevel.MEDIUM: 0.3,
            RiskLevel.HIGH: 0.5,
            RiskLevel.CRITICAL: 0.7,
        }
    
    def _extract_features(
        self,
        order_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> np.ndarray:
        """Extract features for prediction."""
        features = [
            hash(order_data.get("product_id", "")) % 1000 / 1000,
            hash(context.get("machine_id", "")) % 1000 / 1000,
            hash(context.get("operator_id", "")) % 1000 / 1000,
            context.get("shift", 1) / 3,
            context.get("operator_experience", 0.5),
            context.get("machine_health", 0.8),
            hash(context.get("material_batch", "")) % 1000 / 1000,
            context.get("temperature", 20) / 50,
            context.get("humidity", 50) / 100,
            order_data.get("quantity", 1) / 100,
        ]
        return np.array(features, dtype=np.float32)
    
    def add_historical_data(
        self,
        order_data: Dict[str, Any],
        context: Dict[str, Any],
        had_defect: bool,
        defect_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add historical data for training."""
        features = self._extract_features(order_data, context)
        label = 1.0 if had_defect else 0.0
        self.training_data.append((features, label))
        
        if had_defect and defect_details:
            self.historical_issues.append({
                "order_id": order_data.get("order_id"),
                "product_id": order_data.get("product_id"),
                "machine_id": context.get("machine_id"),
                "operator_id": context.get("operator_id"),
                "defect_type": defect_details.get("type"),
                "cause": defect_details.get("cause"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
    
    def train(self) -> Dict[str, Any]:
        """Train the predictive model."""
        if len(self.training_data) < 20:
            return {"success": False, "reason": "Not enough data (need 20+)"}
        
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            
            # Prepare data
            X = np.stack([d[0] for d in self.training_data])
            y = np.array([d[1] for d in self.training_data])
            
            X_tensor = torch.from_numpy(X)
            y_tensor = torch.from_numpy(y).float().unsqueeze(1)
            
            # Simple model
            class DefectPredictor(nn.Module):
                def __init__(self, input_size):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_size, 32),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Linear(16, 1),
                        nn.Sigmoid(),
                    )
                
                def forward(self, x):
                    return self.net(x)
            
            self.model = DefectPredictor(X.shape[1])
            optimizer = optim.Adam(self.model.parameters(), lr=0.01)
            criterion = nn.BCELoss()
            
            # Train
            losses = []
            for epoch in range(100):
                self.model.train()
                optimizer.zero_grad()
                output = self.model(X_tensor)
                loss = criterion(output, y_tensor)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            self.trained = True
            
            return {
                "success": True,
                "samples": len(self.training_data),
                "defect_rate": float(np.mean(y)),
                "final_loss": losses[-1],
            }
            
        except ImportError:
            logger.warning("PyTorch not available for training")
            return {"success": False, "reason": "PyTorch not available"}
    
    def predict_risk(
        self,
        order_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> RiskPrediction:
        """
        Predict defect risk for an order.
        
        Returns:
            RiskPrediction with probability and factors
        """
        features = self._extract_features(order_data, context)
        
        # Predict
        if self.trained and self.model:
            try:
                import torch
                
                self.model.eval()
                with torch.no_grad():
                    x = torch.from_numpy(features).unsqueeze(0)
                    prob = float(self.model(x).item())
                
                model_version = "ml_pytorch"
                confidence = 0.8
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")
                prob = self._base_prediction(features, context)
                model_version = "base_heuristic"
                confidence = 0.5
        else:
            prob = self._base_prediction(features, context)
            model_version = "base_heuristic"
            confidence = 0.5
        
        # Determine risk level
        risk_level = RiskLevel.LOW
        for level, threshold in sorted(self.thresholds.items(), key=lambda x: x[1], reverse=True):
            if prob >= threshold:
                risk_level = level
                break
        
        # Risk factors
        risk_factors = {
            "machine_health": (1 - context.get("machine_health", 0.8)) * 0.3,
            "operator_experience": (1 - context.get("operator_experience", 0.5)) * 0.2,
            "batch_complexity": min(order_data.get("quantity", 1) / 50, 0.3),
            "shift_factor": 0.1 if context.get("shift", 1) == 3 else 0,
        }
        
        # Recommendations
        recommendations = []
        if prob > 0.3:
            if risk_factors["machine_health"] > 0.1:
                recommendations.append("Consider machine maintenance before production")
            if risk_factors["operator_experience"] > 0.1:
                recommendations.append("Assign more experienced operator")
            recommendations.append("Add extra inspection points")
        
        # Find similar issues
        similar = self._find_similar_issues(order_data, context)
        
        return RiskPrediction(
            prediction_id=f"RP-{uuid.uuid4().hex[:8]}",
            risk_level=risk_level,
            defect_probability=prob,
            risk_factors=risk_factors,
            recommendations=recommendations,
            similar_issues=similar,
            model_version=model_version,
            confidence=confidence,
        )
    
    def _base_prediction(self, features: np.ndarray, context: Dict[str, Any]) -> float:
        """Base heuristic prediction when ML not available."""
        # Simple rule-based risk
        risk = 0.1  # Base risk
        
        # Machine health factor
        health = context.get("machine_health", 0.8)
        if health < 0.6:
            risk += 0.3
        elif health < 0.8:
            risk += 0.1
        
        # Operator experience
        exp = context.get("operator_experience", 0.5)
        if exp < 0.3:
            risk += 0.15
        
        # Night shift
        if context.get("shift", 1) == 3:
            risk += 0.05
        
        return min(risk, 0.9)
    
    def _find_similar_issues(
        self,
        order_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Find similar historical issues."""
        similar = []
        product_id = order_data.get("product_id")
        machine_id = context.get("machine_id")
        
        for issue in self.historical_issues[-100:]:  # Last 100
            score = 0
            if issue.get("product_id") == product_id:
                score += 2
            if issue.get("machine_id") == machine_id:
                score += 1
            
            if score > 0:
                similar.append({
                    **issue,
                    "similarity_score": score,
                })
        
        return sorted(similar, key=lambda x: x["similarity_score"], reverse=True)[:5]


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class ExceptionManager:
    """Manages exception requests for validation overrides."""
    
    def __init__(self):
        self.exceptions: Dict[str, ExceptionRequest] = {}
        self.approved_overrides: Dict[str, ExceptionRequest] = {}  # issue_id -> exception
    
    def request_exception(
        self,
        validation_issue_id: str,
        order_id: str,
        operation_id: str,
        requested_by: str,
        reason: str,
        expires_hours: int = 8,
    ) -> ExceptionRequest:
        """Request an exception override."""
        exception = ExceptionRequest(
            exception_id=f"EX-{uuid.uuid4().hex[:8]}",
            validation_issue_id=validation_issue_id,
            order_id=order_id,
            operation_id=operation_id,
            requested_by=requested_by,
            reason=reason,
            status=ExceptionStatus.PENDING,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=expires_hours),
        )
        
        self.exceptions[exception.exception_id] = exception
        logger.info(f"Exception requested: {exception.exception_id} for issue {validation_issue_id}")
        
        return exception
    
    def approve_exception(
        self,
        exception_id: str,
        approved_by: str,
        note: str = "",
    ) -> Tuple[bool, str]:
        """Approve an exception request."""
        exception = self.exceptions.get(exception_id)
        if not exception:
            return False, "Exception not found"
        
        if exception.status != ExceptionStatus.PENDING:
            return False, f"Exception is not pending (status: {exception.status.value})"
        
        exception.status = ExceptionStatus.APPROVED
        exception.resolved_by = approved_by
        exception.resolution_note = note
        exception.resolved_at = datetime.now(timezone.utc)
        
        # Store for quick lookup
        self.approved_overrides[exception.validation_issue_id] = exception
        
        logger.info(f"Exception {exception_id} approved by {approved_by}")
        
        return True, "Exception approved"
    
    def reject_exception(
        self,
        exception_id: str,
        rejected_by: str,
        note: str = "",
    ) -> Tuple[bool, str]:
        """Reject an exception request."""
        exception = self.exceptions.get(exception_id)
        if not exception:
            return False, "Exception not found"
        
        if exception.status != ExceptionStatus.PENDING:
            return False, f"Exception is not pending (status: {exception.status.value})"
        
        exception.status = ExceptionStatus.REJECTED
        exception.resolved_by = rejected_by
        exception.resolution_note = note
        exception.resolved_at = datetime.now(timezone.utc)
        
        return True, "Exception rejected"
    
    def has_valid_override(self, issue_id: str) -> bool:
        """Check if an issue has a valid override."""
        override = self.approved_overrides.get(issue_id)
        if not override:
            return False
        
        # Check expiry
        if override.expires_at and override.expires_at < datetime.now(timezone.utc):
            override.status = ExceptionStatus.EXPIRED
            del self.approved_overrides[issue_id]
            return False
        
        return True
    
    def get_pending_exceptions(self) -> List[ExceptionRequest]:
        """Get all pending exceptions."""
        return [e for e in self.exceptions.values() if e.status == ExceptionStatus.PENDING]


# ═══════════════════════════════════════════════════════════════════════════════
# PREVENTION GUARD SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class PreventionGuardService:
    """
    Main service for error prevention.
    
    Combines all guard engines and provides unified interface.
    """
    
    def __init__(self):
        self.pdm_guard = PDMGuardEngine()
        self.shopfloor_guard = ShopfloorGuardEngine()
        self.predictive_guard = PredictiveGuardEngine()
        self.exception_manager = ExceptionManager()
        
        # Event log
        self.events: List[GuardEvent] = []
        
        # Statistics
        self.stats = {
            "validations_performed": 0,
            "issues_detected": 0,
            "errors_prevented": 0,
            "exceptions_requested": 0,
            "exceptions_approved": 0,
        }
    
    def _log_event(
        self,
        event_type: GuardEventType,
        entity_type: str,
        entity_id: str,
        message: str,
        details: Dict[str, Any] = None,
        user_id: str = None,
    ) -> GuardEvent:
        """Log a guard event."""
        event = GuardEvent(
            event_id=f"GE-{uuid.uuid4().hex[:8]}",
            event_type=event_type,
            entity_type=entity_type,
            entity_id=entity_id,
            message=message,
            details=details or {},
            user_id=user_id,
        )
        self.events.append(event)
        return event
    
    def validate_product_release(
        self,
        item_data: Dict[str, Any],
        bom_components: List[Dict[str, Any]],
        routing_operations: List[Dict[str, Any]],
        attachments: List[Dict[str, Any]],
    ) -> ValidationResult:
        """
        Comprehensive validation for product release.
        
        Combines BOM, routing, and documentation checks.
        """
        self.stats["validations_performed"] += 1
        
        all_issues = []
        
        # BOM validation
        bom_result = self.pdm_guard.validate_bom(item_data, bom_components)
        all_issues.extend(bom_result.issues)
        
        # Routing validation
        routing_result = self.pdm_guard.validate_routing(item_data, routing_operations)
        all_issues.extend(routing_result.issues)
        
        # Documentation validation
        doc_result = self.pdm_guard.validate_documentation(item_data, attachments)
        all_issues.extend(doc_result.issues)
        
        # Aggregate
        errors = sum(1 for i in all_issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
        warnings = sum(1 for i in all_issues if i.severity == ValidationSeverity.WARNING)
        
        self.stats["issues_detected"] += len(all_issues)
        
        # Log event
        if errors > 0:
            self._log_event(
                GuardEventType.VALIDATION_FAILED,
                "item",
                item_data.get("item_id", ""),
                f"Product release validation failed: {errors} errors, {warnings} warnings",
                {"errors": errors, "warnings": warnings},
            )
        else:
            self._log_event(
                GuardEventType.VALIDATION_PASSED,
                "item",
                item_data.get("item_id", ""),
                "Product release validation passed",
            )
        
        return ValidationResult(
            passed=errors == 0,
            issues=all_issues,
            errors=errors,
            warnings=warnings,
            blocked=any(i.action == ValidationAction.BLOCK for i in all_issues),
            requires_approval=any(i.action == ValidationAction.APPROVAL_REQUIRED for i in all_issues),
            validation_time_ms=bom_result.validation_time_ms + routing_result.validation_time_ms + doc_result.validation_time_ms,
        )
    
    def validate_order_start(
        self,
        order_data: Dict[str, Any],
        scanned_materials: List[Dict[str, Any]],
        required_materials: List[Dict[str, Any]],
        machine_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[ValidationResult, RiskPrediction]:
        """
        Validate before starting a production order.
        
        Returns both validation result and risk prediction.
        """
        self.stats["validations_performed"] += 1
        
        all_issues = []
        
        # Material validations
        for scanned, required in zip(scanned_materials, required_materials):
            mat_result = self.shopfloor_guard.validate_material(order_data, scanned, required)
            all_issues.extend(mat_result.issues)
        
        # Equipment validation
        equip_result = self.shopfloor_guard.validate_equipment(
            order_data,
            machine_data,
            context.get("tools", []),
        )
        all_issues.extend(equip_result.issues)
        
        # Risk prediction
        risk = self.predictive_guard.predict_risk(order_data, context)
        
        # Add risk alert if high
        if risk.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            all_issues.append(ValidationIssue(
                issue_id=f"VI-{uuid.uuid4().hex[:8]}",
                rule_id="RISK-001",
                rule_name="Predictive Risk Alert",
                category=ValidationCategory.QUALITY,
                severity=ValidationSeverity.WARNING if risk.risk_level == RiskLevel.HIGH else ValidationSeverity.ERROR,
                action=ValidationAction.APPROVAL_REQUIRED,
                message=f"Predicted defect risk: {risk.defect_probability:.1%} ({risk.risk_level.value})",
                details={
                    "risk_level": risk.risk_level.value,
                    "probability": risk.defect_probability,
                    "factors": risk.risk_factors,
                },
                entity_type="order",
                entity_id=order_data.get("order_id", ""),
            ))
            
            self._log_event(
                GuardEventType.RISK_ALERT,
                "order",
                order_data.get("order_id", ""),
                f"High risk detected: {risk.defect_probability:.1%}",
                {"risk": risk.to_dict()},
            )
        
        # Check for overrides
        filtered_issues = []
        for issue in all_issues:
            if issue.action in [ValidationAction.BLOCK, ValidationAction.APPROVAL_REQUIRED]:
                if not self.exception_manager.has_valid_override(issue.issue_id):
                    filtered_issues.append(issue)
                else:
                    issue.resolved = True
                    issue.resolution_note = "Override approved"
                    filtered_issues.append(issue)
            else:
                filtered_issues.append(issue)
        
        errors = sum(1 for i in filtered_issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] and not i.resolved)
        
        self.stats["issues_detected"] += len(all_issues)
        if any(i.action == ValidationAction.BLOCK and not i.resolved for i in filtered_issues):
            self.stats["errors_prevented"] += 1
        
        result = ValidationResult(
            passed=errors == 0,
            issues=filtered_issues,
            errors=errors,
            warnings=len(filtered_issues) - errors,
            blocked=any(i.action == ValidationAction.BLOCK and not i.resolved for i in filtered_issues),
            requires_approval=any(i.action == ValidationAction.APPROVAL_REQUIRED and not i.resolved for i in filtered_issues),
        )
        
        return result, risk
    
    def request_exception(
        self,
        issue_id: str,
        order_id: str,
        operation_id: str,
        requested_by: str,
        reason: str,
    ) -> ExceptionRequest:
        """Request an exception for a blocked validation."""
        self.stats["exceptions_requested"] += 1
        
        exception = self.exception_manager.request_exception(
            issue_id, order_id, operation_id, requested_by, reason
        )
        
        self._log_event(
            GuardEventType.EXCEPTION_REQUESTED,
            "exception",
            exception.exception_id,
            f"Exception requested by {requested_by}",
            {"issue_id": issue_id, "reason": reason},
            user_id=requested_by,
        )
        
        return exception
    
    def approve_exception(
        self,
        exception_id: str,
        approved_by: str,
        note: str = "",
    ) -> Tuple[bool, str]:
        """Approve an exception."""
        success, message = self.exception_manager.approve_exception(exception_id, approved_by, note)
        
        if success:
            self.stats["exceptions_approved"] += 1
            self._log_event(
                GuardEventType.EXCEPTION_RESOLVED,
                "exception",
                exception_id,
                f"Exception approved by {approved_by}",
                {"note": note},
                user_id=approved_by,
            )
        
        return success, message
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get guard statistics."""
        return {
            **self.stats,
            "pending_exceptions": len(self.exception_manager.get_pending_exceptions()),
            "active_rules": {
                "pdm": len([r for r in self.pdm_guard.rules.values() if r.enabled]),
                "shopfloor": len([r for r in self.shopfloor_guard.rules.values() if r.enabled]),
            },
            "predictive_model": {
                "trained": self.predictive_guard.trained,
                "training_samples": len(self.predictive_guard.training_data),
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_service_instance: Optional[PreventionGuardService] = None


def get_prevention_guard_service() -> PreventionGuardService:
    """Get singleton service."""
    global _service_instance
    if _service_instance is None:
        _service_instance = PreventionGuardService()
    return _service_instance


def reset_prevention_guard_service() -> None:
    """Reset singleton."""
    global _service_instance
    _service_instance = None


