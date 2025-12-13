"""
═══════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 - MATHEMATICAL CORE ENGINE
═══════════════════════════════════════════════════════════════════════════════

A rigorous, modular, and explainable optimization & ML backend for
Advanced Planning & Scheduling (APS) in manufacturing.

Architecture Overview
=====================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         APPLICATION LAYER                                │
    │    (FastAPI endpoints, CLI tools, dashboards)                           │
    └────────────────────────────────┬────────────────────────────────────────┘
                                     │
    ┌────────────────────────────────▼────────────────────────────────────────┐
    │                           CORE ENGINE                                    │
    │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐│
    │  │ optimization │ │     ml       │ │  evaluation  │ │  explainability  ││
    │  │              │ │              │ │              │ │                  ││
    │  │ • MILP       │ │ • Forecasting│ │ • KPIs       │ │ • Schedule XAI   ││
    │  │ • CP-SAT     │ │ • Setup pred │ │ • SNR        │ │ • Forecast XAI   ││
    │  │ • Heuristics │ │ • RUL        │ │ • Data qual  │ │ • Confidence     ││
    │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────────┘│
    └────────────────────────────────┬────────────────────────────────────────┘
                                     │
    ┌────────────────────────────────▼────────────────────────────────────────┐
    │                           DATA LAYER                                     │
    │    (DataBundle, Excel loader, database connectors)                       │
    └─────────────────────────────────────────────────────────────────────────┘

Core Concept: Signal-to-Noise Ratio (SNR)
=========================================

SNR is the fundamental metric for data quality, model robustness, and
prediction confidence throughout the system.

Definition:
    SNR = σ²_signal / σ²_noise = Var(predictable) / Var(residual)

Interpretation:
    SNR > 10   : EXCELLENT - High predictability, reliable decisions
    3 < SNR ≤ 10 : GOOD - Moderate predictability, useful with monitoring
    1 < SNR ≤ 3  : FAIR - Limited predictability, use with caution
    SNR ≤ 1    : POOR - Noise-dominated, predictions unreliable

Relationship to R²:
    R² = SNR / (1 + SNR)  ⟺  SNR = R² / (1 - R²)

R&D / SIFIDE Alignment
======================
Work Package 1: Intelligent APS Core (Optimization)
Work Package 2: Predictive Intelligence (ML)
Work Package 3: SmartInventory Integration
Work Package 4: Evaluation & Explainability

References
==========
[1] Pinedo, M. (2016). Scheduling: Theory, Algorithms, and Systems. Springer.
[2] Hyndman & Athanasopoulos (2021). Forecasting: Principles and Practice. OTexts.
[3] Box, Hunter & Hunter (2005). Statistics for Experimenters. Wiley.
[4] Fisher, R.A. (1925). Statistical Methods for Research Workers.
"""

__version__ = "4.0.0"
__author__ = "ProdPlan R&D Team"

from typing import Final

# SNR interpretation thresholds
SNR_EXCELLENT: Final[float] = 10.0
SNR_GOOD: Final[float] = 3.0
SNR_FAIR: Final[float] = 1.0
