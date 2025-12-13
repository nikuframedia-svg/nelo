"""
Quality Module - Prevention Guard & Zero Defect Manufacturing
=============================================================

Components:
- PDM Guard: Validation for product definition
- Shopfloor Guard: Validation for production execution
- Predictive Guard: ML-based risk prediction
- Exception Manager: Override approval workflow

R&D / SIFIDE: WP4 - Zero Defect Manufacturing
"""

from .prevention_guard import (
    PreventionGuardService,
    PDMGuardEngine,
    ShopfloorGuardEngine,
    PredictiveGuardEngine,
    ExceptionManager,
    ValidationRule,
    ValidationResult,
    ValidationIssue,
    RiskPrediction,
    ExceptionRequest,
    GuardEvent,
    ValidationCategory,
    ValidationSeverity,
    ValidationAction,
    RiskLevel,
    ExceptionStatus,
    GuardEventType,
    get_prevention_guard_service,
)

from .api_prevention_guard import router as prevention_guard_router

__all__ = [
    "PreventionGuardService",
    "PDMGuardEngine",
    "ShopfloorGuardEngine",
    "PredictiveGuardEngine",
    "ExceptionManager",
    "ValidationRule",
    "ValidationResult",
    "ValidationIssue",
    "RiskPrediction",
    "ExceptionRequest",
    "GuardEvent",
    "ValidationCategory",
    "ValidationSeverity",
    "ValidationAction",
    "RiskLevel",
    "ExceptionStatus",
    "GuardEventType",
    "get_prevention_guard_service",
    "prevention_guard_router",
]


