"""
════════════════════════════════════════════════════════════════════════════════════════════════════
WORK INSTRUCTIONS API - REST Endpoints for Digital Work Instructions
════════════════════════════════════════════════════════════════════════════════════════════════════

Endpoints for work instructions management and execution:
- POST /work-instructions - Create new instruction
- GET /work-instructions/{id} - Get instruction
- POST /work-instructions/{id}/execute - Start execution
- POST /executions/{id}/steps/{step_id}/complete - Complete step
- POST /executions/{id}/quality-checks - Record quality check

R&D / SIFIDE: WP1 - Digital Twin & Shopfloor
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

from .work_instructions import (
    WorkInstructionService,
    WorkInstructionDefinition,
    InstructionExecution,
    InstructionStep,
    QualityCheckItem,
    StepType,
    InputType,
    CheckResult,
    ExecutionStatus,
    get_work_instruction_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/work-instructions", tags=["Work Instructions"])


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class VisualReferenceInput(BaseModel):
    type: str = Field("image", description="image, 3d_model, video")
    url: str
    caption: Optional[str] = None
    highlight_region: Optional[Dict[str, Any]] = None


class ToleranceInput(BaseModel):
    nominal: float
    min_value: float
    max_value: float
    unit: str = ""


class StepInput(BaseModel):
    step_id: Optional[str] = None
    title: str
    description: str
    step_type: str = Field("instruction", description="instruction, measurement, checklist, photo, confirmation")
    input_type: str = Field("none", description="none, numeric, text, select, boolean, photo")
    visual_references: List[VisualReferenceInput] = Field(default_factory=list)
    tolerance: Optional[ToleranceInput] = None
    options: List[str] = Field(default_factory=list)
    is_critical: bool = False
    is_quality_check: bool = False
    required: bool = True
    condition: Optional[Dict[str, Any]] = None


class QualityCheckInput(BaseModel):
    check_id: Optional[str] = None
    question: str
    check_type: str = Field("ok_nok", description="ok_nok, numeric, visual, dimensional")
    tolerance: Optional[ToleranceInput] = None
    is_critical: bool = False
    fail_action: str = Field("pause", description="pause, alert, block")


class CreateInstructionRequest(BaseModel):
    revision_id: int
    operation_id: int
    title: str
    description: str = ""
    steps: List[StepInput] = Field(default_factory=list)
    quality_checks: List[QualityCheckInput] = Field(default_factory=list)
    author: str = ""
    language: str = "pt"
    model_3d_url: Optional[str] = None
    estimated_time_minutes: float = 0


class StartExecutionRequest(BaseModel):
    order_id: str
    operator_id: str
    operator_name: str = ""


class CompleteStepRequest(BaseModel):
    input_value: Optional[Any] = None
    photo_url: Optional[str] = None
    notes: Optional[str] = None
    operator_id: Optional[str] = None


class RecordQualityCheckRequest(BaseModel):
    check_id: str
    result: str = Field(..., description="ok, nok, na")
    measured_value: Optional[float] = None
    defect_description: Optional[str] = None
    defect_photo_url: Optional[str] = None
    operator_id: Optional[str] = None


class DemoInstructionRequest(BaseModel):
    product_name: str = "Motor Assembly"
    num_steps: int = Field(5, ge=3, le=15)


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/status")
async def get_status():
    """Get work instructions module status."""
    service = get_work_instruction_service()
    
    return {
        "service": "Work Instructions & Checklists",
        "version": "1.0.0",
        "status": "operational",
        "instructions_count": len(service.instructions),
        "active_executions": len([
            e for e in service.engine.executions.values()
            if e.status == ExecutionStatus.IN_PROGRESS
        ]),
        "step_types": [t.value for t in StepType],
        "input_types": [t.value for t in InputType],
        "check_results": [r.value for r in CheckResult],
    }


@router.post("")
async def create_instruction(request: CreateInstructionRequest):
    """Create a new work instruction."""
    service = get_work_instruction_service()
    
    # Convert inputs
    steps_data = []
    for s in request.steps:
        step_dict = {
            "step_id": s.step_id,
            "title": s.title,
            "description": s.description,
            "step_type": s.step_type,
            "input_type": s.input_type,
            "visual_references": [v.model_dump() for v in s.visual_references],
            "tolerance": s.tolerance.model_dump() if s.tolerance else None,
            "options": s.options,
            "is_critical": s.is_critical,
            "is_quality_check": s.is_quality_check,
            "required": s.required,
            "condition": s.condition,
        }
        steps_data.append(step_dict)
    
    checks_data = []
    for q in request.quality_checks:
        check_dict = {
            "check_id": q.check_id,
            "question": q.question,
            "check_type": q.check_type,
            "tolerance": q.tolerance.model_dump() if q.tolerance else None,
            "is_critical": q.is_critical,
            "fail_action": q.fail_action,
        }
        checks_data.append(check_dict)
    
    instruction = service.create_instruction(
        revision_id=request.revision_id,
        operation_id=request.operation_id,
        title=request.title,
        description=request.description,
        steps=steps_data,
        quality_checks=checks_data,
        author=request.author,
        language=request.language,
        model_3d_url=request.model_3d_url,
    )
    
    return instruction.to_dict()


@router.get("/{instruction_id}")
async def get_instruction(instruction_id: str):
    """Get instruction by ID."""
    service = get_work_instruction_service()
    
    instruction = service.get_instruction(instruction_id)
    if not instruction:
        raise HTTPException(status_code=404, detail="Instruction not found")
    
    return instruction.to_dict()


@router.get("")
async def list_instructions(
    revision_id: Optional[int] = None,
    operation_id: Optional[int] = None,
    status: Optional[str] = None,
):
    """List all instructions."""
    service = get_work_instruction_service()
    
    instructions = list(service.instructions.values())
    
    if revision_id is not None:
        instructions = [i for i in instructions if i.revision_id == revision_id]
    
    if operation_id is not None:
        instructions = [i for i in instructions if i.operation_id == operation_id]
    
    if status:
        instructions = [i for i in instructions if i.status == status]
    
    return {
        "total": len(instructions),
        "instructions": [i.to_dict() for i in instructions],
    }


@router.post("/{instruction_id}/release")
async def release_instruction(instruction_id: str):
    """Release an instruction for use."""
    service = get_work_instruction_service()
    
    success, message = service.release_instruction(instruction_id)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    return {"success": True, "message": message}


@router.post("/{instruction_id}/execute")
async def start_execution(instruction_id: str, request: StartExecutionRequest):
    """Start executing an instruction."""
    service = get_work_instruction_service()
    
    instruction = service.get_instruction(instruction_id)
    if not instruction:
        raise HTTPException(status_code=404, detail="Instruction not found")
    
    try:
        execution = service.start_execution(
            instruction_id=instruction_id,
            order_id=request.order_id,
            operator_id=request.operator_id,
            operator_name=request.operator_name,
        )
        
        # Include instruction and current step in response
        current_step = service.engine.get_current_step(execution.execution_id)
        
        response = execution.to_dict()
        response["instruction"] = instruction.to_dict()
        if current_step:
            step_def, step_exec = current_step
            response["current_step"] = step_def.to_dict()
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/executions/{execution_id}")
async def get_execution(execution_id: str):
    """Get execution status."""
    service = get_work_instruction_service()
    
    execution = service.get_execution(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    instruction = service.get_instruction(execution.instruction_id)
    
    response = execution.to_dict()
    response["instruction"] = instruction.to_dict() if instruction else None
    
    current_step = service.engine.get_current_step(execution_id)
    if current_step:
        step_def, step_exec = current_step
        response["current_step"] = step_def.to_dict()
    
    return response


@router.post("/executions/{execution_id}/steps/{step_id}/complete")
async def complete_step(execution_id: str, step_id: str, request: CompleteStepRequest):
    """Complete a step in the execution."""
    service = get_work_instruction_service()
    
    success, message = service.complete_step(
        execution_id=execution_id,
        step_id=step_id,
        input_value=request.input_value,
        photo_url=request.photo_url,
        notes=request.notes,
        operator_id=request.operator_id,
    )
    
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    # Get updated execution
    execution = service.get_execution(execution_id)
    response = execution.to_dict()
    
    # Include next step if available
    current_step = service.engine.get_current_step(execution_id)
    if current_step:
        step_def, step_exec = current_step
        response["current_step"] = step_def.to_dict()
    else:
        response["current_step"] = None
    
    return response


@router.post("/executions/{execution_id}/quality-checks")
async def record_quality_check(execution_id: str, request: RecordQualityCheckRequest):
    """Record a quality check result."""
    service = get_work_instruction_service()
    
    try:
        result = CheckResult(request.result)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid result: {request.result}")
    
    success, message, action = service.record_quality_check(
        execution_id=execution_id,
        check_id=request.check_id,
        result=result,
        measured_value=request.measured_value,
        defect_description=request.defect_description,
        defect_photo_url=request.defect_photo_url,
        operator_id=request.operator_id,
    )
    
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    execution = service.get_execution(execution_id)
    
    return {
        "success": True,
        "message": message,
        "action_required": action,
        "execution": execution.to_dict(),
    }


@router.post("/executions/{execution_id}/resume")
async def resume_execution(execution_id: str):
    """Resume a paused execution."""
    service = get_work_instruction_service()
    
    success, message = service.engine.resume_execution(execution_id)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    execution = service.get_execution(execution_id)
    return execution.to_dict()


@router.post("/executions/{execution_id}/abort")
async def abort_execution(execution_id: str, reason: str = ""):
    """Abort an execution."""
    service = get_work_instruction_service()
    
    success, message = service.engine.abort_execution(execution_id, reason)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    execution = service.get_execution(execution_id)
    return execution.to_dict()


@router.get("/executions")
async def list_executions(
    status: Optional[str] = None,
    order_id: Optional[str] = None,
    operator_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
):
    """List executions."""
    service = get_work_instruction_service()
    
    executions = list(service.engine.executions.values())
    
    if status:
        executions = [e for e in executions if e.status.value == status]
    
    if order_id:
        executions = [e for e in executions if e.order_id == order_id]
    
    if operator_id:
        executions = [e for e in executions if e.operator_id == operator_id]
    
    # Sort by started_at descending
    executions.sort(key=lambda x: x.started_at or datetime.min, reverse=True)
    
    return {
        "total": len(executions),
        "executions": [e.to_dict() for e in executions[:limit]],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/demo")
async def create_demo_instruction(request: DemoInstructionRequest = Body(...)):
    """Create a demo work instruction with sample steps."""
    service = get_work_instruction_service()
    
    # Generate demo steps
    demo_steps = [
        {
            "title": "Preparação da Estação",
            "description": "Verifique se todas as ferramentas estão disponíveis e a estação de trabalho está limpa.",
            "step_type": "confirmation",
            "input_type": "boolean",
            "visual_references": [
                {"type": "image", "url": "/images/demo/workstation.png", "caption": "Estação de trabalho preparada"}
            ],
        },
        {
            "title": f"Identificação do {request.product_name}",
            "description": "Escaneie o código de barras ou insira o número de série do produto.",
            "step_type": "instruction",
            "input_type": "text",
            "required": True,
        },
        {
            "title": "Inspeção Visual Inicial",
            "description": "Verifique se o produto não apresenta danos visíveis, riscos ou deformações.",
            "step_type": "checklist",
            "input_type": "boolean",
            "is_quality_check": True,
        },
        {
            "title": "Medição de Dimensão Crítica",
            "description": "Meça o diâmetro externo utilizando o paquímetro calibrado.",
            "step_type": "measurement",
            "input_type": "numeric",
            "tolerance": {"nominal": 50.0, "min_value": 49.5, "max_value": 50.5, "unit": "mm"},
            "is_critical": True,
            "visual_references": [
                {"type": "image", "url": "/images/demo/measurement.png", "caption": "Posição correta do paquímetro"}
            ],
        },
        {
            "title": "Aperto de Parafusos",
            "description": "Aperte os 4 parafusos M8 ao torque especificado de 25 Nm ±2 Nm.",
            "step_type": "measurement",
            "input_type": "numeric",
            "tolerance": {"nominal": 25.0, "min_value": 23.0, "max_value": 27.0, "unit": "Nm"},
            "is_critical": True,
        },
        {
            "title": "Teste Funcional",
            "description": "Execute o teste funcional e confirme que o motor arranca corretamente.",
            "step_type": "checklist",
            "input_type": "boolean",
            "is_quality_check": True,
            "is_critical": True,
        },
        {
            "title": "Captura de Evidência",
            "description": "Tire uma foto do produto montado mostrando a etiqueta de identificação.",
            "step_type": "photo",
            "input_type": "photo",
            "required": True,
        },
        {
            "title": "Verificação Final",
            "description": "Confirme que todas as etapas foram concluídas e o produto está pronto para embalagem.",
            "step_type": "confirmation",
            "input_type": "boolean",
        },
    ]
    
    # Use only requested number of steps
    selected_steps = demo_steps[:request.num_steps]
    
    # Demo quality checks
    quality_checks = [
        {
            "question": "Superfície sem riscos ou marcas?",
            "check_type": "ok_nok",
            "is_critical": False,
            "fail_action": "alert",
        },
        {
            "question": "Dimensões dentro da tolerância?",
            "check_type": "ok_nok",
            "is_critical": True,
            "fail_action": "pause",
        },
        {
            "question": "Funcionamento correto?",
            "check_type": "ok_nok",
            "is_critical": True,
            "fail_action": "block",
        },
    ]
    
    instruction = service.create_instruction(
        revision_id=1,
        operation_id=1,
        title=f"Montagem de {request.product_name}",
        description=f"Instruções completas para montagem e teste de {request.product_name}",
        steps=selected_steps,
        quality_checks=quality_checks,
        author="Sistema Demo",
        language="pt",
        model_3d_url="/models/demo/motor_assembly.glb",
    )
    
    # Auto-release
    service.release_instruction(instruction.instruction_id)
    
    return instruction.to_dict()


@router.post("/demo/execute")
async def run_demo_execution():
    """Create demo instruction and start execution."""
    service = get_work_instruction_service()
    
    # Create instruction if none exists
    if not service.instructions:
        demo_request = DemoInstructionRequest()
        instruction_data = await create_demo_instruction(demo_request)
        instruction_id = instruction_data["instruction_id"]
    else:
        # Use first released instruction
        instruction = next(
            (i for i in service.instructions.values() if i.status == "released"),
            None
        )
        if not instruction:
            demo_request = DemoInstructionRequest()
            instruction_data = await create_demo_instruction(demo_request)
            instruction_id = instruction_data["instruction_id"]
        else:
            instruction_id = instruction.instruction_id
    
    # Start execution
    request = StartExecutionRequest(
        order_id="OP-2024-0001",
        operator_id="OP001",
        operator_name="João Silva",
    )
    
    return await start_execution(instruction_id, request)



