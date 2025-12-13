"""
Shopfloor Module - Digital Shopfloor Management
===============================================

Components:
- Work Instructions: Digital work instructions & checklists
- Poka-Yoke: Error prevention mechanisms
- Execution Engine: Step-by-step execution with validation
"""

from .work_instructions import (
    WorkInstructionService,
    WorkInstructionExecutionEngine,
    WorkInstructionDefinition,
    InstructionStep,
    InstructionExecution,
    QualityCheckItem,
    StepType,
    InputType,
    CheckResult,
    ExecutionStatus,
    StepStatus,
    get_work_instruction_service,
)

from .api_work_instructions import router as work_instructions_router

__all__ = [
    "WorkInstructionService",
    "WorkInstructionExecutionEngine",
    "WorkInstructionDefinition",
    "InstructionStep",
    "InstructionExecution",
    "QualityCheckItem",
    "StepType",
    "InputType",
    "CheckResult",
    "ExecutionStatus",
    "StepStatus",
    "get_work_instruction_service",
    "work_instructions_router",
]



