"""
════════════════════════════════════════════════════════════════════════════════════════════════════
Work Instructions Service
════════════════════════════════════════════════════════════════════════════════════════════════════

Contract 8 Implementation: Work Instructions for Shopfloor App

Features:
- Work instructions linked to ItemRevision + RoutingOperation
- Step-by-step instructions with text, images
- Quality checkpoints (checkbox, numeric, text)
- Order execution reporting (start, pause, terminate)
- Downtime tracking

Models stored in separate SQLite for shopfloor data.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Path to shopfloor database
SHOPFLOOR_DB_PATH = Path(__file__).parent.parent.parent / "data" / "shopfloor.db"


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class CheckpointType(str, Enum):
    """Type of quality checkpoint."""
    CHECKBOX = "checkbox"
    NUMERIC = "numeric"
    TEXT = "text"


class DowntimeReason(str, Enum):
    """Predefined downtime reasons."""
    MACHINE_FAILURE = "machine_failure"
    MATERIAL_SHORTAGE = "material_shortage"
    TOOL_CHANGE = "tool_change"
    QUALITY_ISSUE = "quality_issue"
    OPERATOR_BREAK = "operator_break"
    MAINTENANCE = "maintenance"
    SETUP = "setup"
    OTHER = "other"


class ExecutionStatus(str, Enum):
    """Order execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class QualityCheckpoint(BaseModel):
    """A quality checkpoint in work instructions."""
    id: str
    label: str
    type: CheckpointType
    required: bool = True
    unit: Optional[str] = None  # For numeric
    min_value: Optional[float] = None  # For numeric
    max_value: Optional[float] = None  # For numeric
    default_value: Optional[str] = None


class WorkInstructionStep(BaseModel):
    """A single step in work instructions."""
    step_number: int
    title: str
    description: str
    image_url: Optional[str] = None
    duration_minutes: Optional[int] = None
    checkpoints: List[QualityCheckpoint] = []
    safety_warning: Optional[str] = None


class WorkInstructionData(BaseModel):
    """Complete work instructions for an operation."""
    id: int = 0
    revision_id: int
    operation_id: int
    operation_code: str
    title: str
    version: str = "1.0"
    steps: List[WorkInstructionStep] = []
    total_estimated_time: int = 0  # minutes
    tools_required: List[str] = []
    materials_required: List[str] = []
    safety_equipment: List[str] = []
    created_at: str = ""
    updated_at: str = ""


class QualityCheckResult(BaseModel):
    """Result of a quality checkpoint."""
    checkpoint_id: str
    value: str
    passed: bool
    timestamp: str


class OrderExecutionReport(BaseModel):
    """Report for order execution on shopfloor."""
    order_id: str
    operation_id: int
    machine_id: str
    operator_id: Optional[str] = None
    
    # Quantities
    good_qty: int = 0
    scrap_qty: int = 0
    rework_qty: int = 0
    
    # Quality checkpoints
    quality_results: List[QualityCheckResult] = []
    
    # Timing
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    pause_time_minutes: int = 0
    
    # Downtime
    downtime_reason: Optional[DowntimeReason] = None
    downtime_minutes: int = 0
    downtime_notes: Optional[str] = None
    
    # Status
    status: ExecutionStatus = ExecutionStatus.PENDING
    notes: Optional[str] = None


class ShopfloorOrder(BaseModel):
    """Order as seen on shopfloor."""
    id: str
    article_id: str
    article_name: str
    operation_code: str
    machine_id: str
    planned_qty: int
    good_qty: int = 0
    scrap_qty: int = 0
    status: ExecutionStatus
    planned_start: str
    planned_end: str
    actual_start: Optional[str] = None
    actual_end: Optional[str] = None
    priority: int = 0
    has_work_instructions: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_db_exists() -> None:
    """Create shopfloor database and tables."""
    SHOPFLOOR_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(SHOPFLOOR_DB_PATH)
    cursor = conn.cursor()
    
    # Work Instructions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS work_instructions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            revision_id INTEGER NOT NULL,
            operation_id INTEGER NOT NULL,
            operation_code TEXT,
            title TEXT NOT NULL,
            version TEXT DEFAULT '1.0',
            steps TEXT DEFAULT '[]',
            total_estimated_time INTEGER DEFAULT 0,
            tools_required TEXT DEFAULT '[]',
            materials_required TEXT DEFAULT '[]',
            safety_equipment TEXT DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT,
            UNIQUE(revision_id, operation_id)
        )
    """)
    
    # Order Execution Reports table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS order_execution_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT NOT NULL,
            operation_id INTEGER,
            machine_id TEXT NOT NULL,
            operator_id TEXT,
            good_qty INTEGER DEFAULT 0,
            scrap_qty INTEGER DEFAULT 0,
            rework_qty INTEGER DEFAULT 0,
            quality_results TEXT DEFAULT '[]',
            start_time TEXT,
            end_time TEXT,
            pause_time_minutes INTEGER DEFAULT 0,
            downtime_reason TEXT,
            downtime_minutes INTEGER DEFAULT 0,
            downtime_notes TEXT,
            status TEXT DEFAULT 'pending',
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT
        )
    """)
    
    conn.commit()
    conn.close()


# Ensure DB exists on module load
_ensure_db_exists()


# ═══════════════════════════════════════════════════════════════════════════════
# WORK INSTRUCTIONS SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class WorkInstructionService:
    """Service for managing work instructions."""
    
    @staticmethod
    def get(revision_id: int, operation_id: int) -> Optional[WorkInstructionData]:
        """Get work instructions for a revision/operation."""
        conn = sqlite3.connect(SHOPFLOOR_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM work_instructions
            WHERE revision_id = ? AND operation_id = ?
        """, (revision_id, operation_id))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return WorkInstructionData(
            id=row["id"],
            revision_id=row["revision_id"],
            operation_id=row["operation_id"],
            operation_code=row["operation_code"] or "",
            title=row["title"],
            version=row["version"] or "1.0",
            steps=[WorkInstructionStep(**s) for s in json.loads(row["steps"] or "[]")],
            total_estimated_time=row["total_estimated_time"] or 0,
            tools_required=json.loads(row["tools_required"] or "[]"),
            materials_required=json.loads(row["materials_required"] or "[]"),
            safety_equipment=json.loads(row["safety_equipment"] or "[]"),
            created_at=row["created_at"],
            updated_at=row["updated_at"] or "",
        )
    
    @staticmethod
    def create(data: WorkInstructionData) -> WorkInstructionData:
        """Create work instructions."""
        conn = sqlite3.connect(SHOPFLOOR_DB_PATH)
        cursor = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        
        cursor.execute("""
            INSERT INTO work_instructions (
                revision_id, operation_id, operation_code, title, version,
                steps, total_estimated_time, tools_required, materials_required,
                safety_equipment, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.revision_id,
            data.operation_id,
            data.operation_code,
            data.title,
            data.version,
            json.dumps([s.dict() for s in data.steps]),
            data.total_estimated_time,
            json.dumps(data.tools_required),
            json.dumps(data.materials_required),
            json.dumps(data.safety_equipment),
            now,
            now,
        ))
        
        data.id = cursor.lastrowid
        data.created_at = now
        data.updated_at = now
        
        conn.commit()
        conn.close()
        
        return data
    
    @staticmethod
    def update(data: WorkInstructionData) -> WorkInstructionData:
        """Update work instructions."""
        conn = sqlite3.connect(SHOPFLOOR_DB_PATH)
        cursor = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        
        cursor.execute("""
            UPDATE work_instructions SET
                operation_code = ?,
                title = ?,
                version = ?,
                steps = ?,
                total_estimated_time = ?,
                tools_required = ?,
                materials_required = ?,
                safety_equipment = ?,
                updated_at = ?
            WHERE id = ?
        """, (
            data.operation_code,
            data.title,
            data.version,
            json.dumps([s.dict() for s in data.steps]),
            data.total_estimated_time,
            json.dumps(data.tools_required),
            json.dumps(data.materials_required),
            json.dumps(data.safety_equipment),
            now,
            data.id,
        ))
        
        data.updated_at = now
        
        conn.commit()
        conn.close()
        
        return data
    
    @staticmethod
    def generate_default(
        revision_id: int,
        operation_id: int,
        operation_code: str,
        article_name: str = "Product",
    ) -> WorkInstructionData:
        """Generate default work instructions for an operation."""
        steps = [
            WorkInstructionStep(
                step_number=1,
                title="Preparação",
                description=f"Verificar material e ferramentas para {operation_code}. Confirmar que a máquina está pronta.",
                duration_minutes=5,
                checkpoints=[
                    QualityCheckpoint(
                        id="prep_material",
                        label="Material verificado",
                        type=CheckpointType.CHECKBOX,
                        required=True,
                    ),
                    QualityCheckpoint(
                        id="prep_tools",
                        label="Ferramentas OK",
                        type=CheckpointType.CHECKBOX,
                        required=True,
                    ),
                ],
            ),
            WorkInstructionStep(
                step_number=2,
                title="Execução",
                description=f"Executar operação {operation_code} conforme especificação.",
                duration_minutes=15,
                safety_warning="Usar equipamento de proteção individual (EPI).",
            ),
            WorkInstructionStep(
                step_number=3,
                title="Verificação",
                description="Verificar qualidade da peça produzida.",
                duration_minutes=5,
                checkpoints=[
                    QualityCheckpoint(
                        id="dim_check",
                        label="Dimensão principal (mm)",
                        type=CheckpointType.NUMERIC,
                        required=True,
                        unit="mm",
                        min_value=0.0,
                        max_value=1000.0,
                    ),
                    QualityCheckpoint(
                        id="visual_check",
                        label="Inspeção visual OK",
                        type=CheckpointType.CHECKBOX,
                        required=True,
                    ),
                    QualityCheckpoint(
                        id="notes",
                        label="Observações",
                        type=CheckpointType.TEXT,
                        required=False,
                    ),
                ],
            ),
        ]
        
        return WorkInstructionData(
            revision_id=revision_id,
            operation_id=operation_id,
            operation_code=operation_code,
            title=f"Instrução de Trabalho - {operation_code}",
            version="1.0",
            steps=steps,
            total_estimated_time=25,
            tools_required=["Ferramenta standard", "Calibrador"],
            materials_required=[article_name],
            safety_equipment=["Luvas", "Óculos de proteção"],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ORDER EXECUTION SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

def report_order_execution(report: OrderExecutionReport) -> int:
    """Save order execution report."""
    conn = sqlite3.connect(SHOPFLOOR_DB_PATH)
    cursor = conn.cursor()
    
    now = datetime.utcnow().isoformat()
    
    cursor.execute("""
        INSERT INTO order_execution_reports (
            order_id, operation_id, machine_id, operator_id,
            good_qty, scrap_qty, rework_qty, quality_results,
            start_time, end_time, pause_time_minutes,
            downtime_reason, downtime_minutes, downtime_notes,
            status, notes, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report.order_id,
        report.operation_id,
        report.machine_id,
        report.operator_id,
        report.good_qty,
        report.scrap_qty,
        report.rework_qty,
        json.dumps([r.dict() for r in report.quality_results]),
        report.start_time,
        report.end_time,
        report.pause_time_minutes,
        report.downtime_reason.value if report.downtime_reason else None,
        report.downtime_minutes,
        report.downtime_notes,
        report.status.value,
        report.notes,
        now,
        now,
    ))
    
    report_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    
    return report_id


def get_order_reports(order_id: str) -> List[Dict[str, Any]]:
    """Get all execution reports for an order."""
    conn = sqlite3.connect(SHOPFLOOR_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM order_execution_reports
        WHERE order_id = ?
        ORDER BY created_at DESC
    """, (order_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_work_instructions(
    revision_id: int,
    operation_id: int,
    auto_generate: bool = True,
    operation_code: str = "OP",
    article_name: str = "Product",
) -> Optional[WorkInstructionData]:
    """
    Get work instructions for a revision/operation.
    If not found and auto_generate=True, creates default instructions.
    """
    wi = WorkInstructionService.get(revision_id, operation_id)
    
    if wi:
        return wi
    
    if auto_generate:
        default_wi = WorkInstructionService.generate_default(
            revision_id, operation_id, operation_code, article_name
        )
        return WorkInstructionService.create(default_wi)
    
    return None


def get_shopfloor_orders(
    machine_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
) -> List[ShopfloorOrder]:
    """
    Get orders for shopfloor display.
    
    This is a demo implementation - in production would query the main scheduler.
    """
    # Demo orders
    demo_orders = [
        ShopfloorOrder(
            id="ORD-001",
            article_id="ART-100",
            article_name="Peça A",
            operation_code="OP10",
            machine_id="CNC-01",
            planned_qty=100,
            good_qty=45,
            scrap_qty=2,
            status=ExecutionStatus.IN_PROGRESS,
            planned_start="2025-01-09T08:00:00",
            planned_end="2025-01-09T12:00:00",
            actual_start="2025-01-09T08:15:00",
            priority=1,
            has_work_instructions=True,
        ),
        ShopfloorOrder(
            id="ORD-002",
            article_id="ART-200",
            article_name="Peça B",
            operation_code="OP20",
            machine_id="CNC-01",
            planned_qty=50,
            status=ExecutionStatus.PENDING,
            planned_start="2025-01-09T12:00:00",
            planned_end="2025-01-09T14:00:00",
            priority=2,
            has_work_instructions=True,
        ),
        ShopfloorOrder(
            id="ORD-003",
            article_id="ART-100",
            article_name="Peça A",
            operation_code="OP30",
            machine_id="MILL-01",
            planned_qty=80,
            status=ExecutionStatus.PENDING,
            planned_start="2025-01-09T08:00:00",
            planned_end="2025-01-09T11:00:00",
            priority=1,
            has_work_instructions=False,
        ),
    ]
    
    result = demo_orders
    
    if machine_id:
        result = [o for o in result if o.machine_id == machine_id]
    
    if status:
        result = [o for o in result if o.status.value == status]
    
    return result[:limit]


# ═══════════════════════════════════════════════════════════════════════════════
# POKA-YOKE DIGITAL - SHOPFLOOR VALIDATION (CONTRACT 9)
# ═══════════════════════════════════════════════════════════════════════════════

class ShopfloorValidationError(Exception):
    """Error raised when Poka-Yoke validation fails on shopfloor start."""
    def __init__(self, message: str, validation_errors: List[str], warnings: List[str] = None):
        super().__init__(message)
        self.validation_errors = validation_errors
        self.warnings = warnings or []


@dataclass
class ProcessParamSpec:
    """Specification for a process parameter."""
    name: str
    spec_min: Optional[float] = None
    spec_max: Optional[float] = None
    nominal: Optional[float] = None
    unit: str = ""


class ShopfloorValidator:
    """
    Poka-Yoke Digital validator for Shopfloor operations.
    
    Validates before operation START:
    1. Correct Revision (RELEASED status)
    2. WorkInstruction availability
    3. Correct Material (if barcode/material_id provided)
    4. Process Parameters in range (if specs defined)
    """
    
    def __init__(self):
        self.wi_service = WorkInstructionService()
    
    def validate_start(
        self,
        order_id: str,
        machine_id: str,
        revision_id: Optional[int] = None,
        operation_id: Optional[int] = None,
        material_barcode: Optional[str] = None,
        process_params: Optional[Dict[str, float]] = None,
        strict_mode: bool = True,
    ) -> Dict[str, Any]:
        """
        Validate an operation before starting.
        
        Args:
            order_id: Order ID
            machine_id: Machine ID
            revision_id: Optional revision ID to validate
            operation_id: Optional operation ID
            material_barcode: Optional material barcode to verify
            process_params: Optional process parameters to validate
            strict_mode: If True, raises error on validation failure
        
        Returns:
            Dict with validation result and any warnings
            
        Raises:
            ShopfloorValidationError: If strict_mode and validation fails
        """
        errors = []
        warnings = []
        
        # 1. Validate Revision Status
        if revision_id:
            rev_validation = self._validate_revision(revision_id)
            errors.extend(rev_validation.get("errors", []))
            warnings.extend(rev_validation.get("warnings", []))
        
        # 2. Validate WorkInstruction availability
        if revision_id and operation_id:
            wi_validation = self._validate_work_instructions(revision_id, operation_id)
            if wi_validation.get("error"):
                errors.append(wi_validation["error"])
            if wi_validation.get("warning"):
                warnings.append(wi_validation["warning"])
        
        # 3. Validate Material (if barcode provided)
        if material_barcode and revision_id:
            mat_validation = self._validate_material(revision_id, material_barcode)
            if mat_validation.get("error"):
                errors.append(mat_validation["error"])
        
        # 4. Validate Process Parameters (if specs available)
        if process_params and revision_id and operation_id:
            param_validation = self._validate_process_params(
                revision_id, operation_id, process_params
            )
            errors.extend(param_validation.get("errors", []))
            warnings.extend(param_validation.get("warnings", []))
        
        # Apply strict mode
        if strict_mode and errors:
            raise ShopfloorValidationError(
                f"Operação não pode iniciar: {len(errors)} erro(s) de validação",
                errors,
                warnings
            )
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "can_proceed": len(errors) == 0 or not strict_mode,
        }
    
    def _validate_revision(self, revision_id: int) -> Dict[str, Any]:
        """Validate that revision is RELEASED."""
        errors = []
        warnings = []
        
        try:
            # Import both pdm_models and dpp_models to resolve relationships
            from duplios import pdm_models, dpp_models
            from duplios.pdm_models import ItemRevision, RevisionStatus
            from duplios.models import SessionLocal
            
            db = SessionLocal()
            revision = db.query(ItemRevision).filter(
                ItemRevision.id == revision_id
            ).first()
            db.close()
            
            if not revision:
                errors.append(f"Poka-Yoke: Revisão {revision_id} não encontrada")
            elif revision.status != RevisionStatus.RELEASED:
                errors.append(
                    f"Poka-Yoke: Revisão {revision.code} não está RELEASED "
                    f"(status atual: {revision.status.value})"
                )
        except ImportError:
            # PDM models not available, skip validation
            pass
        except Exception as e:
            warnings.append(f"Não foi possível validar revisão: {e}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_work_instructions(
        self, revision_id: int, operation_id: int
    ) -> Dict[str, Any]:
        """Check if work instructions are available."""
        try:
            wi = self.wi_service.get_work_instructions(revision_id, operation_id)
            if not wi:
                return {
                    "warning": f"Poka-Yoke: Sem instruções de trabalho para operação {operation_id}"
                }
        except Exception as e:
            return {"warning": f"Não foi possível verificar instruções: {e}"}
        
        return {}
    
    def _validate_material(
        self, revision_id: int, barcode: str
    ) -> Dict[str, Any]:
        """Validate material against BOM."""
        try:
            # Import both pdm_models and dpp_models to resolve relationships
            from duplios import pdm_models, dpp_models
            from duplios.pdm_models import BomLine, Item
            from duplios.models import SessionLocal
            
            db = SessionLocal()
            
            # Get BOM components
            bom_lines = db.query(BomLine).filter(
                BomLine.parent_revision_id == revision_id
            ).all()
            
            # Get component SKUs
            component_ids = [bl.component_item_id for bl in bom_lines]
            valid_skus = set()
            
            if component_ids:
                components = db.query(Item).filter(
                    Item.id.in_(component_ids)
                ).all()
                valid_skus = {c.sku for c in components}
            
            db.close()
            
            # Check if barcode matches any valid component
            if barcode not in valid_skus:
                return {
                    "error": f"Poka-Yoke: Material {barcode} não consta na BOM da revisão"
                }
        except ImportError:
            pass  # PDM not available
        except Exception as e:
            return {"warning": f"Não foi possível validar material: {e}"}
        
        return {}
    
    def _validate_process_params(
        self,
        revision_id: int,
        operation_id: int,
        params: Dict[str, float],
    ) -> Dict[str, Any]:
        """Validate process parameters against specs."""
        errors = []
        warnings = []
        
        try:
            # Get parameter specs (could be from DB or routing)
            specs = self._get_param_specs(revision_id, operation_id)
            
            for param_name, value in params.items():
                spec = specs.get(param_name)
                if not spec:
                    continue  # No spec defined, skip
                
                if spec.spec_min is not None and value < spec.spec_min:
                    errors.append(
                        f"Poka-Yoke: {param_name}={value}{spec.unit} abaixo do mínimo "
                        f"({spec.spec_min}{spec.unit})"
                    )
                elif spec.spec_max is not None and value > spec.spec_max:
                    errors.append(
                        f"Poka-Yoke: {param_name}={value}{spec.unit} acima do máximo "
                        f"({spec.spec_max}{spec.unit})"
                    )
        except Exception as e:
            warnings.append(f"Não foi possível validar parâmetros: {e}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _get_param_specs(
        self, revision_id: int, operation_id: int
    ) -> Dict[str, ProcessParamSpec]:
        """Get process parameter specifications for operation."""
        # Default specs (could be loaded from DB)
        return {
            "temperatura": ProcessParamSpec(
                name="temperatura",
                spec_min=150.0,
                spec_max=250.0,
                nominal=200.0,
                unit="°C"
            ),
            "pressao": ProcessParamSpec(
                name="pressao",
                spec_min=5.0,
                spec_max=15.0,
                nominal=10.0,
                unit="bar"
            ),
            "feed_rate": ProcessParamSpec(
                name="feed_rate",
                spec_min=50.0,
                spec_max=200.0,
                nominal=100.0,
                unit="mm/min"
            ),
            "spindle_speed": ProcessParamSpec(
                name="spindle_speed",
                spec_min=500.0,
                spec_max=5000.0,
                nominal=2000.0,
                unit="rpm"
            ),
        }


# Singleton instance
_shopfloor_validator: Optional[ShopfloorValidator] = None


def get_shopfloor_validator() -> ShopfloorValidator:
    """Get singleton ShopfloorValidator instance."""
    global _shopfloor_validator
    if _shopfloor_validator is None:
        _shopfloor_validator = ShopfloorValidator()
    return _shopfloor_validator


def validate_operation_start(
    order_id: str,
    machine_id: str,
    revision_id: Optional[int] = None,
    operation_id: Optional[int] = None,
    material_barcode: Optional[str] = None,
    process_params: Optional[Dict[str, float]] = None,
    strict_mode: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to validate operation start.
    
    Wraps ShopfloorValidator.validate_start().
    """
    validator = get_shopfloor_validator()
    return validator.validate_start(
        order_id=order_id,
        machine_id=machine_id,
        revision_id=revision_id,
        operation_id=operation_id,
        material_barcode=material_barcode,
        process_params=process_params,
        strict_mode=strict_mode,
    )

