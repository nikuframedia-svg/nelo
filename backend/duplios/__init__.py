"""
════════════════════════════════════════════════════════════════════════════════════════════════════
DUPLIOS Package - Digital Product Passport & PDM
════════════════════════════════════════════════════════════════════════════════════════════════════

This package provides:
- DPP (Digital Product Passport) management
- PDM (Product Data Management) core
- LCA (Life Cycle Assessment) calculations

Modules:
- models: SQLAlchemy models for Duplios
- pdm_models: PDM-specific models (Item, Revision, BOM, Routing, ECR/ECO)
- pdm_core: PDM business logic and validation engines
- dpp_models: DPP-specific models
- lca_models: LCA calculation models
- api_duplios: REST API for Duplios
- api_pdm: REST API for PDM
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Models
try:
    from .models import SessionLocal, Base, engine
except ImportError as e:
    logger.warning(f"Duplios models not available: {e}")

# PDM Models
try:
    from .pdm_models import (
        Item, ItemType,
        ItemRevision, RevisionStatus,
        BomLine, RoutingOperation,
        ECR, ECO, ECRStatus,
        WorkInstruction,
    )
except ImportError as e:
    logger.warning(f"PDM models not available: {e}")

# PDM Core
try:
    from .pdm_core import (
        PDMService, PDMConfig,
        BomValidationEngine, RoutingValidationEngine,
        ReleaseValidationEngine, RevisionWorkflowEngine,
        ECREngine, RevisionComparisonEngine,
        ValidationResult, ValidationIssue, ValidationSeverity,
        BomExplosion, ImpactAnalysis, RevisionDiff,
        get_pdm_service, get_current_revision, get_bom, get_routing,
    )
except ImportError as e:
    logger.warning(f"PDM core not available: {e}")

# DPP Models
try:
    from .dpp_models import (
        DppRecord, DigitalIdentity,
        DppStatus, VerificationStatus,
        ProductConformanceSnapshot, ConformityStatus,
        GoldenRun,
    )
except ImportError as e:
    logger.warning(f"DPP models not available: {e}")

# APIs
try:
    from .api_pdm import router as pdm_router
except ImportError as e:
    logger.warning(f"PDM API not available: {e}")
    pdm_router = None

try:
    from .api_duplios import router as duplios_router
except ImportError as e:
    logger.warning(f"Duplios API not available: {e}")
    duplios_router = None

__all__ = [
    # Models base
    "SessionLocal", "Base", "engine",
    # PDM Models
    "Item", "ItemType",
    "ItemRevision", "RevisionStatus",
    "BomLine", "RoutingOperation",
    "ECR", "ECO", "ECRStatus",
    "WorkInstruction",
    # PDM Core
    "PDMService", "PDMConfig",
    "BomValidationEngine", "RoutingValidationEngine",
    "ReleaseValidationEngine", "RevisionWorkflowEngine",
    "ECREngine", "RevisionComparisonEngine",
    "ValidationResult", "ValidationIssue", "ValidationSeverity",
    "BomExplosion", "ImpactAnalysis", "RevisionDiff",
    "get_pdm_service", "get_current_revision", "get_bom", "get_routing",
    # DPP Models
    "DppRecord", "DigitalIdentity",
    "DppStatus", "VerificationStatus",
    "ProductConformanceSnapshot", "ConformityStatus",
    "GoldenRun",
    # APIs
    "pdm_router", "duplios_router",
]


