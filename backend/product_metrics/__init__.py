"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — PRODUCT METRICS MODULE
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Industrial product metrics, classification, and automatic delivery estimation.

CAPABILITIES
════════════

1. PRODUCT CLASSIFICATION
   - Automatic classification by type (vidro_duplo, vidro_triplo, vidro_laminado)
   - Fingerprinting based on operations, times, machines, routes
   - Family grouping for statistical analysis

2. PRODUCT KPIs
   - Processing time (mean, std, sigma levels)
   - Setup time analysis
   - Waste/scrap estimation
   - Lead time distribution
   - Signal-to-Noise Ratio (SNR) for process stability

3. DELIVERY TIME ESTIMATION
   - Deterministic: Processing + Setup + Queue
   - Probabilistic: ML-based with confidence intervals
   - Calendar-aware (future: consider holidays, shifts)

PRODUCT TYPES (Glass Industry)
══════════════════════════════

VIDRO DUPLO (Double Glazing):
- 2 glass panes with air/gas gap
- Typical ops: CUT → EDGE → WASH → ASSEMBLE → SEAL
- Processing: ~30-45 min

VIDRO TRIPLO (Triple Glazing):
- 3 glass panes with 2 gaps
- Additional: lamination, coating
- Processing: ~45-60 min

VIDRO LAMINADO (Laminated Glass):
- Multiple layers with PVB/EVA interlayer
- Requires autoclave
- Processing: ~60-90 min

MATHEMATICAL FRAMEWORK
══════════════════════

Process Capability (Cp):
    Cp = (USL - LSL) / (6σ)

Process SNR:
    SNR = Var(μ_group) / Var(residual)

Delivery Time Estimation:
    T_delivery = T_processing + T_setup + T_queue + T_buffer
    
    where T_buffer = k × σ_total (safety margin)

R&D / SIFIDE ALIGNMENT
──────────────────────
Work Package 7: Product Intelligence
- Hypothesis H7.1: Fingerprinting improves classification accuracy >95%
- Hypothesis H7.2: ML delivery estimates have MAPE <10%
- Experiment E7.1: Compare deterministic vs ML delivery estimation
"""

from .product_classification import (
    ProductType,
    ProductFingerprint,
    classify_product,
    classify_all_products,
    build_product_fingerprint,
    get_product_family,
)
from .product_kpi_engine import (
    ProductKPIs,
    ProductTypeKPIs,
    GlobalProductKPIs,
    compute_product_kpis,
    compute_product_type_kpis,
    compute_all_product_kpis,
)
from .delivery_time_engine import (
    DeliveryEstimate,
    estimate_delivery_time,
    estimate_all_deliveries,
    DeliveryConfig,
    EstimationMethod,
)

__all__ = [
    # Classification
    "ProductType",
    "ProductFingerprint",
    "classify_product",
    "classify_all_products",
    "build_product_fingerprint",
    "get_product_family",
    # KPIs
    "ProductKPIs",
    "ProductTypeKPIs",
    "GlobalProductKPIs",
    "compute_product_kpis",
    "compute_product_type_kpis",
    "compute_all_product_kpis",
    # Delivery
    "DeliveryEstimate",
    "estimate_delivery_time",
    "estimate_all_deliveries",
    "DeliveryConfig",
    "EstimationMethod",
]

