"""
════════════════════════════════════════════════════════════════════════════════════════════════════
WORK INSTRUCTIONS - Digital Work Instructions & Checklists
════════════════════════════════════════════════════════════════════════════════════════════════════

Sistema de instruções de trabalho digitais para operadores no chão de fábrica.

Features:
- Instruções passo-a-passo com texto, imagens, 3D
- Checklists de qualidade interativos
- Poka-yoke de sequência (não permite saltar passos)
- Captura de evidências (valores, fotos, confirmações)
- Rastreabilidade completa (as-built record)
- Suporte multilíngua
- Integração com PDM e ProdPlan

Integração:
- PDM: Vincula instruções a revisões de produto/operações
- ProdPlan: Carrega instruções automaticamente ao iniciar ordem
- ZDM: Reporta NOKs ao sistema de qualidade

R&D / SIFIDE: WP1 - Digital Twin & Shopfloor
"""

from __future__ import annotations

import logging
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Text, DateTime,
    ForeignKey, Enum as SQLEnum, JSON, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship, Session
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class StepType(str, Enum):
    """Type of instruction step."""
    INSTRUCTION = "instruction"  # Simple instruction
    MEASUREMENT = "measurement"  # Requires numeric value input
    CHECKLIST = "checklist"  # Yes/No or OK/NOK check
    PHOTO = "photo"  # Requires photo evidence
    CONFIRMATION = "confirmation"  # Simple confirmation
    CONDITIONAL = "conditional"  # Conditional branching


class InputType(str, Enum):
    """Type of input expected from operator."""
    NONE = "none"  # Just read and confirm
    NUMERIC = "numeric"  # Enter a number
    TEXT = "text"  # Enter text
    SELECT = "select"  # Choose from options
    BOOLEAN = "boolean"  # Yes/No
    PHOTO = "photo"  # Take/upload photo


class CheckResult(str, Enum):
    """Result of a checklist item."""
    OK = "ok"
    NOK = "nok"
    NA = "na"  # Not applicable


class ExecutionStatus(str, Enum):
    """Status of instruction execution."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PAUSED = "paused"
    ABORTED = "aborted"


class StepStatus(str, Enum):
    """Status of a single step."""
    PENDING = "pending"
    CURRENT = "current"
    COMPLETED = "completed"
    SKIPPED = "skipped"  # Only for conditional steps that don't apply


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VisualReference:
    """
    Visual reference for a step (image or 3D model).
    
    As specified: "permitir anotações no modelo (ex.: setas, destaques) por passo"
    """
    type: str  # "image", "3d_model", "video"
    url: str
    caption: Optional[str] = None
    highlight_region: Optional[Dict[str, Any]] = None  # For 3D: {"x": 0, "y": 0, "z": 0, "radius": 1}
    annotations: Optional[List[Dict[str, Any]]] = None  # Annotations: [{"type": "arrow", "from": {...}, "to": {...}}, ...]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        return result


@dataclass
class Tolerance:
    """Tolerance specification for measurements."""
    nominal: float
    min_value: float
    max_value: float
    unit: str = ""
    
    def is_within_tolerance(self, value: float) -> bool:
        return self.min_value <= value <= self.max_value
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InstructionStep:
    """A single step in a work instruction."""
    step_id: str
    sequence: int
    title: str
    description: str
    step_type: StepType
    input_type: InputType
    
    # Optional fields
    visual_references: List[VisualReference] = field(default_factory=list)
    tolerance: Optional[Tolerance] = None
    options: List[str] = field(default_factory=list)  # For SELECT type
    is_critical: bool = False
    is_quality_check: bool = False
    required: bool = True
    
    # Conditional logic
    condition: Optional[Dict[str, Any]] = None  # {"step_id": "X", "value": "Y", "operator": "=="}
    
    # Translations
    translations: Dict[str, Dict[str, str]] = field(default_factory=dict)  # {lang: {title, description}}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "sequence": self.sequence,
            "title": self.title,
            "description": self.description,
            "step_type": self.step_type.value,
            "input_type": self.input_type.value,
            "visual_references": [v.to_dict() for v in self.visual_references],
            "tolerance": self.tolerance.to_dict() if self.tolerance else None,
            "options": self.options,
            "is_critical": self.is_critical,
            "is_quality_check": self.is_quality_check,
            "required": self.required,
            "condition": self.condition,
            "translations": self.translations,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'InstructionStep':
        return InstructionStep(
            step_id=data.get("step_id", str(uuid.uuid4())[:8]),
            sequence=data.get("sequence", 0),
            title=data.get("title", ""),
            description=data.get("description", ""),
            step_type=StepType(data.get("step_type", "instruction")),
            input_type=InputType(data.get("input_type", "none")),
            visual_references=[
                VisualReference(**v) for v in data.get("visual_references", [])
            ],
            tolerance=Tolerance(**data["tolerance"]) if data.get("tolerance") else None,
            options=data.get("options", []),
            is_critical=data.get("is_critical", False),
            is_quality_check=data.get("is_quality_check", False),
            required=data.get("required", True),
            condition=data.get("condition"),
            translations=data.get("translations", {}),
        )


@dataclass
class QualityCheckItem:
    """A quality checklist item."""
    check_id: str
    sequence: int
    question: str
    check_type: str  # "ok_nok", "numeric", "visual", "dimensional"
    
    # For numeric checks
    tolerance: Optional[Tolerance] = None
    
    # Metadata
    is_critical: bool = False
    fail_action: str = "pause"  # "pause", "alert", "block"
    
    # Translations
    translations: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "sequence": self.sequence,
            "question": self.question,
            "check_type": self.check_type,
            "tolerance": self.tolerance.to_dict() if self.tolerance else None,
            "is_critical": self.is_critical,
            "fail_action": self.fail_action,
            "translations": self.translations,
        }


@dataclass
class WorkInstructionDefinition:
    """Complete work instruction definition."""
    instruction_id: str
    revision_id: int  # PDM revision
    operation_id: int  # PDM routing operation
    
    title: str
    description: str
    version: int
    
    steps: List[InstructionStep]
    quality_checks: List[QualityCheckItem]
    
    # Metadata
    author: str = ""
    language: str = "pt"  # Default language
    supported_languages: List[str] = field(default_factory=lambda: ["pt"])  # Multilingual support
    estimated_time_minutes: float = 0
    
    # 3D model reference
    model_3d_url: Optional[str] = None
    model_3d_type: str = "glb"  # glb, gltf, obj, step
    model_3d_annotations: Optional[List[Dict[str, Any]]] = None  # Global 3D annotations
    
    # Status
    status: str = "draft"  # draft, released, obsolete
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction_id": self.instruction_id,
            "revision_id": self.revision_id,
            "operation_id": self.operation_id,
            "title": self.title,
            "description": self.description,
            "version": self.version,
            "steps": [s.to_dict() for s in self.steps],
            "quality_checks": [q.to_dict() for q in self.quality_checks],
            "author": self.author,
            "language": self.language,
            "supported_languages": self.supported_languages,
            "estimated_time_minutes": self.estimated_time_minutes,
            "model_3d_url": self.model_3d_url,
            "model_3d_type": self.model_3d_type,
            "model_3d_annotations": self.model_3d_annotations,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class StepExecution:
    """Execution record for a single step."""
    step_id: str
    status: StepStatus
    
    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Input values
    input_value: Optional[Any] = None
    photo_url: Optional[str] = None
    
    # Quality check result
    check_result: Optional[CheckResult] = None
    
    # Validation
    within_tolerance: Optional[bool] = None
    
    # Operator
    completed_by: Optional[str] = None
    
    # Notes
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "input_value": self.input_value,
            "photo_url": self.photo_url,
            "check_result": self.check_result.value if self.check_result else None,
            "within_tolerance": self.within_tolerance,
            "completed_by": self.completed_by,
            "notes": self.notes,
        }


@dataclass
class QualityCheckExecution:
    """Execution record for a quality check."""
    check_id: str
    result: CheckResult
    
    # For numeric checks
    measured_value: Optional[float] = None
    within_tolerance: Optional[bool] = None
    
    # Timestamp
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checked_by: Optional[str] = None
    
    # If NOK
    defect_description: Optional[str] = None
    defect_photo_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "result": self.result.value,
            "measured_value": self.measured_value,
            "within_tolerance": self.within_tolerance,
            "checked_at": self.checked_at.isoformat(),
            "checked_by": self.checked_by,
            "defect_description": self.defect_description,
            "defect_photo_url": self.defect_photo_url,
        }


@dataclass
class InstructionExecution:
    """Complete execution record for a work instruction."""
    execution_id: str
    instruction_id: str
    order_id: str  # Production order
    
    # Status
    status: ExecutionStatus
    current_step_index: int = 0
    
    # Operator
    operator_id: str = ""
    operator_name: str = ""
    
    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    
    # Execution records
    step_executions: List[StepExecution] = field(default_factory=list)
    quality_check_executions: List[QualityCheckExecution] = field(default_factory=list)
    
    # Summary
    total_steps: int = 0
    completed_steps: int = 0
    nok_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "instruction_id": self.instruction_id,
            "order_id": self.order_id,
            "status": self.status.value,
            "current_step_index": self.current_step_index,
            "operator_id": self.operator_id,
            "operator_name": self.operator_name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "paused_at": self.paused_at.isoformat() if self.paused_at else None,
            "step_executions": [s.to_dict() for s in self.step_executions],
            "quality_check_executions": [q.to_dict() for q in self.quality_check_executions],
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "nok_count": self.nok_count,
            "progress_percent": round(self.completed_steps / self.total_steps * 100 if self.total_steps > 0 else 0, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# POKA-YOKE EXECUTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class PokaYokeValidationError(Exception):
    """Raised when poka-yoke validation fails."""
    pass


class WorkInstructionExecutionEngine:
    """
    Execution engine with Poka-Yoke validation.
    
    Features:
    - Enforces step sequence (no skipping)
    - Validates required inputs
    - Checks tolerance limits
    - Handles conditional steps
    - Manages quality check failures
    
    Usage:
        engine = WorkInstructionExecutionEngine()
        execution = engine.start_execution(instruction, order_id, operator)
        
        # Complete each step
        engine.complete_step(execution, step_id, input_value="50.2")
        
        # Handle quality checks
        engine.record_quality_check(execution, check_id, CheckResult.OK)
    """
    
    def __init__(self):
        self.executions: Dict[str, InstructionExecution] = {}
        self.instructions: Dict[str, WorkInstructionDefinition] = {}
    
    def register_instruction(self, instruction: WorkInstructionDefinition) -> None:
        """Register an instruction definition."""
        self.instructions[instruction.instruction_id] = instruction
    
    def start_execution(
        self,
        instruction_id: str,
        order_id: str,
        operator_id: str,
        operator_name: str = "",
    ) -> InstructionExecution:
        """
        Start a new execution of a work instruction.
        
        Initializes all step executions as PENDING.
        """
        instruction = self.instructions.get(instruction_id)
        if not instruction:
            raise ValueError(f"Instruction {instruction_id} not found")
        
        execution_id = f"EX-{uuid.uuid4().hex[:8]}"
        
        # Initialize step executions
        step_executions = []
        for step in instruction.steps:
            step_executions.append(StepExecution(
                step_id=step.step_id,
                status=StepStatus.PENDING,
            ))
        
        # Mark first applicable step as current
        first_step_idx = self._find_first_applicable_step(instruction, {})
        if first_step_idx >= 0:
            step_executions[first_step_idx].status = StepStatus.CURRENT
        
        execution = InstructionExecution(
            execution_id=execution_id,
            instruction_id=instruction_id,
            order_id=order_id,
            status=ExecutionStatus.IN_PROGRESS,
            current_step_index=first_step_idx,
            operator_id=operator_id,
            operator_name=operator_name,
            started_at=datetime.now(timezone.utc),
            step_executions=step_executions,
            total_steps=len(instruction.steps),
            completed_steps=0,
        )
        
        self.executions[execution_id] = execution
        logger.info(f"Started execution {execution_id} for instruction {instruction_id}")
        
        return execution
    
    def complete_step(
        self,
        execution_id: str,
        step_id: str,
        input_value: Optional[Any] = None,
        photo_url: Optional[str] = None,
        notes: Optional[str] = None,
        operator_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Complete a step in the execution.
        
        Performs poka-yoke validation:
        - Verifies step is the current step
        - Validates required input
        - Checks tolerance limits
        
        Returns:
            (success, message)
        """
        execution = self.executions.get(execution_id)
        if not execution:
            return False, f"Execution {execution_id} not found"
        
        instruction = self.instructions.get(execution.instruction_id)
        if not instruction:
            return False, "Instruction not found"
        
        # Find step
        step_exec = next((s for s in execution.step_executions if s.step_id == step_id), None)
        if not step_exec:
            return False, f"Step {step_id} not found"
        
        step_def = next((s for s in instruction.steps if s.step_id == step_id), None)
        if not step_def:
            return False, f"Step definition {step_id} not found"
        
        # POKA-YOKE: Check if this is the current step
        if step_exec.status != StepStatus.CURRENT:
            return False, f"Cannot complete step {step_id}: not the current step (poka-yoke)"
        
        # POKA-YOKE: Validate required input
        if step_def.input_type != InputType.NONE and step_def.required:
            if step_def.input_type == InputType.PHOTO and not photo_url:
                return False, "Photo evidence required for this step"
            elif step_def.input_type == InputType.NUMERIC and input_value is None:
                return False, "Numeric value required for this step"
            elif step_def.input_type == InputType.TEXT and not input_value:
                return False, "Text input required for this step"
        
        # POKA-YOKE: Check tolerance if applicable
        within_tolerance = None
        if step_def.tolerance and input_value is not None:
            try:
                numeric_value = float(input_value)
                within_tolerance = step_def.tolerance.is_within_tolerance(numeric_value)
                
                if not within_tolerance and step_def.is_critical:
                    return False, f"Value {numeric_value} outside tolerance [{step_def.tolerance.min_value}, {step_def.tolerance.max_value}]"
            except (ValueError, TypeError):
                return False, "Invalid numeric value"
        
        # Update step execution
        now = datetime.now(timezone.utc)
        step_exec.status = StepStatus.COMPLETED
        step_exec.completed_at = now
        step_exec.input_value = input_value
        step_exec.photo_url = photo_url
        step_exec.within_tolerance = within_tolerance
        step_exec.completed_by = operator_id or execution.operator_id
        step_exec.notes = notes
        
        if not step_exec.started_at:
            step_exec.started_at = now
        
        execution.completed_steps += 1
        
        # Advance to next step
        next_idx = self._find_next_applicable_step(instruction, execution)
        
        if next_idx >= 0:
            execution.current_step_index = next_idx
            execution.step_executions[next_idx].status = StepStatus.CURRENT
            execution.step_executions[next_idx].started_at = now
        else:
            # All steps completed
            execution.status = ExecutionStatus.COMPLETED
            execution.completed_at = now
            logger.info(f"Execution {execution_id} completed")
        
        return True, "Step completed successfully"
    
    def record_quality_check(
        self,
        execution_id: str,
        check_id: str,
        result: CheckResult,
        measured_value: Optional[float] = None,
        defect_description: Optional[str] = None,
        defect_photo_url: Optional[str] = None,
        operator_id: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Record a quality check result.
        
        Returns:
            (success, message, action_required) - action_required is set if NOK
        """
        execution = self.executions.get(execution_id)
        if not execution:
            return False, f"Execution {execution_id} not found", None
        
        instruction = self.instructions.get(execution.instruction_id)
        if not instruction:
            return False, "Instruction not found", None
        
        check_def = next((q for q in instruction.quality_checks if q.check_id == check_id), None)
        if not check_def:
            return False, f"Quality check {check_id} not found", None
        
        # Validate tolerance if numeric
        within_tolerance = None
        if check_def.tolerance and measured_value is not None:
            within_tolerance = check_def.tolerance.is_within_tolerance(measured_value)
        
        # Create execution record
        check_exec = QualityCheckExecution(
            check_id=check_id,
            result=result,
            measured_value=measured_value,
            within_tolerance=within_tolerance,
            checked_by=operator_id or execution.operator_id,
            defect_description=defect_description,
            defect_photo_url=defect_photo_url,
        )
        
        execution.quality_check_executions.append(check_exec)
        
        # Handle NOK
        action_required = None
        if result == CheckResult.NOK:
            execution.nok_count += 1
            
            if check_def.is_critical:
                if check_def.fail_action == "block":
                    execution.status = ExecutionStatus.PAUSED
                    execution.paused_at = datetime.now(timezone.utc)
                    action_required = "block"
                    logger.warning(f"Execution {execution_id} blocked due to critical NOK on {check_id}")
                elif check_def.fail_action == "pause":
                    execution.status = ExecutionStatus.PAUSED
                    execution.paused_at = datetime.now(timezone.utc)
                    action_required = "pause"
                else:
                    action_required = "alert"
        
        return True, "Quality check recorded", action_required
    
    def resume_execution(self, execution_id: str) -> Tuple[bool, str]:
        """Resume a paused execution."""
        execution = self.executions.get(execution_id)
        if not execution:
            return False, f"Execution {execution_id} not found"
        
        if execution.status != ExecutionStatus.PAUSED:
            return False, "Execution is not paused"
        
        execution.status = ExecutionStatus.IN_PROGRESS
        execution.paused_at = None
        
        return True, "Execution resumed"
    
    def abort_execution(self, execution_id: str, reason: str = "") -> Tuple[bool, str]:
        """Abort an execution."""
        execution = self.executions.get(execution_id)
        if not execution:
            return False, f"Execution {execution_id} not found"
        
        execution.status = ExecutionStatus.ABORTED
        execution.completed_at = datetime.now(timezone.utc)
        
        logger.info(f"Execution {execution_id} aborted: {reason}")
        
        return True, "Execution aborted"
    
    def get_current_step(self, execution_id: str) -> Optional[Tuple[InstructionStep, StepExecution]]:
        """Get current step definition and execution."""
        execution = self.executions.get(execution_id)
        if not execution:
            return None
        
        instruction = self.instructions.get(execution.instruction_id)
        if not instruction:
            return None
        
        if execution.current_step_index < 0 or execution.current_step_index >= len(instruction.steps):
            return None
        
        step_def = instruction.steps[execution.current_step_index]
        step_exec = execution.step_executions[execution.current_step_index]
        
        return step_def, step_exec
    
    def _find_first_applicable_step(
        self,
        instruction: WorkInstructionDefinition,
        step_values: Dict[str, Any],
    ) -> int:
        """Find the first applicable step (considering conditions)."""
        for i, step in enumerate(instruction.steps):
            if self._is_step_applicable(step, step_values):
                return i
        return -1
    
    def _find_next_applicable_step(
        self,
        instruction: WorkInstructionDefinition,
        execution: InstructionExecution,
    ) -> int:
        """Find the next applicable step."""
        # Build step values from execution
        step_values = {
            se.step_id: se.input_value
            for se in execution.step_executions
            if se.status == StepStatus.COMPLETED
        }
        
        current_idx = execution.current_step_index
        
        for i in range(current_idx + 1, len(instruction.steps)):
            step = instruction.steps[i]
            if self._is_step_applicable(step, step_values):
                return i
            else:
                # Mark as skipped
                execution.step_executions[i].status = StepStatus.SKIPPED
        
        return -1  # No more steps
    
    def _is_step_applicable(
        self,
        step: InstructionStep,
        step_values: Dict[str, Any],
    ) -> bool:
        """Check if a step is applicable based on its condition."""
        if not step.condition:
            return True
        
        cond_step_id = step.condition.get("step_id")
        cond_value = step.condition.get("value")
        cond_operator = step.condition.get("operator", "==")
        
        if cond_step_id not in step_values:
            return False  # Condition step not completed yet
        
        actual_value = step_values[cond_step_id]
        
        if cond_operator == "==":
            return str(actual_value) == str(cond_value)
        elif cond_operator == "!=":
            return str(actual_value) != str(cond_value)
        elif cond_operator == ">":
            return float(actual_value) > float(cond_value)
        elif cond_operator == "<":
            return float(actual_value) < float(cond_value)
        
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# WORK INSTRUCTION SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class WorkInstructionService:
    """
    Service for managing work instructions.
    
    Provides:
    - CRUD operations for instructions
    - Execution management
    - Integration with PDM
    """
    
    def __init__(self):
        self.engine = WorkInstructionExecutionEngine()
        self.instructions: Dict[str, WorkInstructionDefinition] = {}
    
    def create_instruction(
        self,
        revision_id: int,
        operation_id: int,
        title: str,
        description: str = "",
        steps: Optional[List[Dict[str, Any]]] = None,
        quality_checks: Optional[List[Dict[str, Any]]] = None,
        author: str = "",
        language: str = "pt",
        model_3d_url: Optional[str] = None,
    ) -> WorkInstructionDefinition:
        """Create a new work instruction."""
        instruction_id = f"WI-{uuid.uuid4().hex[:8]}"
        
        # Parse steps
        parsed_steps = []
        if steps:
            for i, s in enumerate(steps):
                s["sequence"] = i + 1
                if "step_id" not in s:
                    s["step_id"] = f"S{i+1:02d}"
                parsed_steps.append(InstructionStep.from_dict(s))
        
        # Parse quality checks
        parsed_checks = []
        if quality_checks:
            for i, q in enumerate(quality_checks):
                check = QualityCheckItem(
                    check_id=q.get("check_id", f"QC{i+1:02d}"),
                    sequence=i + 1,
                    question=q.get("question", ""),
                    check_type=q.get("check_type", "ok_nok"),
                    tolerance=Tolerance(**q["tolerance"]) if q.get("tolerance") else None,
                    is_critical=q.get("is_critical", False),
                    fail_action=q.get("fail_action", "pause"),
                )
                parsed_checks.append(check)
        
        instruction = WorkInstructionDefinition(
            instruction_id=instruction_id,
            revision_id=revision_id,
            operation_id=operation_id,
            title=title,
            description=description,
            version=1,
            steps=parsed_steps,
            quality_checks=parsed_checks,
            author=author,
            language=language,
            model_3d_url=model_3d_url,
            status="draft",
        )
        
        self.instructions[instruction_id] = instruction
        self.engine.register_instruction(instruction)
        
        logger.info(f"Created instruction {instruction_id}")
        
        return instruction
    
    def get_instruction(self, instruction_id: str) -> Optional[WorkInstructionDefinition]:
        """Get instruction by ID."""
        return self.instructions.get(instruction_id)
    
    def get_instruction_for_operation(
        self,
        revision_id: int,
        operation_id: int,
    ) -> Optional[WorkInstructionDefinition]:
        """
        Get instruction for a specific operation.
        
        As specified: "ao iniciar uma ordem, apresentar automaticamente as instruções 
        correspondentes àquele produto e versão, evitando uso de instruções desatualizadas"
        """
        for instr in self.instructions.values():
            if instr.revision_id == revision_id and instr.operation_id == operation_id:
                if instr.status == "released":
                    return instr
        return None
    
    def get_instruction_for_order(
        self,
        order_id: str,
        db_session=None,
    ) -> Optional[WorkInstructionDefinition]:
        """
        Get instruction automatically for a production order.
        
        As specified: "ao iniciar uma ordem, apresentar automaticamente as instruções 
        correspondentes àquele produto e versão"
        
        This method queries the production order to get the item revision and operation,
        then loads the corresponding work instruction.
        """
        try:
            # Try to get order info from ProdPlan/PDM
            if db_session:
                from duplios.pdm_models import ItemRevision, RoutingOperation
                # TODO: Query production order to get revision_id and operation_id
                # For now, return None if order info not available
                pass
            
            # Fallback: try to find instruction by order_id pattern
            # This assumes order_id contains item/revision info
            # In real implementation, query production order table
            
            return None
        except Exception as e:
            logger.warning(f"Failed to get instruction for order {order_id}: {e}")
            return None
    
    def release_instruction(self, instruction_id: str) -> Tuple[bool, str]:
        """Release an instruction for use."""
        instruction = self.instructions.get(instruction_id)
        if not instruction:
            return False, "Instruction not found"
        
        if not instruction.steps:
            return False, "Instruction has no steps"
        
        instruction.status = "released"
        instruction.updated_at = datetime.now(timezone.utc)
        
        return True, "Instruction released"
    
    def start_execution(
        self,
        instruction_id: str,
        order_id: str,
        operator_id: str,
        operator_name: str = "",
    ) -> InstructionExecution:
        """Start executing an instruction."""
        return self.engine.start_execution(
            instruction_id=instruction_id,
            order_id=order_id,
            operator_id=operator_id,
            operator_name=operator_name,
        )
    
    def get_execution(self, execution_id: str) -> Optional[InstructionExecution]:
        """Get execution by ID."""
        return self.engine.executions.get(execution_id)
    
    def complete_step(
        self,
        execution_id: str,
        step_id: str,
        **kwargs,
    ) -> Tuple[bool, str]:
        """Complete a step."""
        return self.engine.complete_step(execution_id, step_id, **kwargs)
    
    def record_quality_check(
        self,
        execution_id: str,
        check_id: str,
        result: CheckResult,
        **kwargs,
    ) -> Tuple[bool, str, Optional[str]]:
        """Record a quality check."""
        return self.engine.record_quality_check(execution_id, check_id, result, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_service_instance: Optional[WorkInstructionService] = None


def get_work_instruction_service() -> WorkInstructionService:
    """Get singleton service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = WorkInstructionService()
    return _service_instance


def reset_work_instruction_service() -> None:
    """Reset singleton (for testing)."""
    global _service_instance
    _service_instance = None


