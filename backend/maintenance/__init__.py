"""
Maintenance Module - Work Orders and CMMS Integration

Este módulo gere ordens de trabalho de manutenção, integrando-se com:
- PredictiveCare (Digital Twin) para criação automática
- ProdPlan para agendamento
- CMMS externos (Odoo, SAP PM, etc.)

R&D / SIFIDE: WP1 - Digital Twin para manutenção preditiva
"""

from .models import (
    MaintenanceWorkOrder,
    MaintenancePriority,
    MaintenanceType,
    MaintenanceStatus,
    WorkOrderSource,
)

from .predictivecare_bridge import PredictiveCareBridge

__all__ = [
    "MaintenanceWorkOrder",
    "MaintenancePriority",
    "MaintenanceType",
    "MaintenanceStatus",
    "WorkOrderSource",
    "PredictiveCareBridge",
]


