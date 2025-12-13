"""
════════════════════════════════════════════════════════════════════════════════════════════════════
PRODPLAN MODULE - Production Planning & Work Instructions
════════════════════════════════════════════════════════════════════════════════════════════════════

Contract 8 Implementation: Work Instructions for Shopfloor

This module contains:
- Work Instructions models and service
- Shopfloor order execution API
- Quality reporting integration
"""

from prodplan.work_instructions import (
    WorkInstructionStep,
    WorkInstructionData,
    QualityCheckpoint,
    WorkInstructionService,
    get_work_instructions,
    report_order_execution,
    OrderExecutionReport,
    DowntimeReason,
)

__all__ = [
    "WorkInstructionStep",
    "WorkInstructionData",
    "QualityCheckpoint",
    "WorkInstructionService",
    "get_work_instructions",
    "report_order_execution",
    "OrderExecutionReport",
    "DowntimeReason",
]



