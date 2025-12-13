"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — SMART INVENTORY MODULE
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Arquitetura Principal: SmartInventory On-Prem

Este módulo implementa um motor de inventário ultra-avançado, on-prem, que:

1. **Ingestão IoT/RFID/Visão**: Stock em tempo real via múltiplas fontes
2. **Digital Twin**: Estado de inventário multi-armazém sincronizado
3. **Forecasting Avançado**: N-BEATS, NST, D-Linear (com fallback ARIMA/Prophet)
4. **ROP Dinâmico**: Ponto de encomenda adaptativo com risco 30 dias
5. **Sinais Externos**: Integração com APIs de preços, notícias, indicadores macro
6. **Otimização Multi-Armazém**: MILP para redistribuição e compras
7. **Sugestões Inteligentes**: Recomendações automáticas (comprar, transferir, reduzir)

Arquitetura:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    SmartInventory Core                           │
    ├─────────────────────────────────────────────────────────────────┤
    │  IoT Ingestion                                                  │
    │    ├─ RFID Events                                               │
    │    ├─ Vision Events                                             │
    │    └─ Manual Events                                             │
    ├─────────────────────────────────────────────────────────────────┤
    │  Stock State (Digital Twin)                                     │
    │    ├─ Multi-Warehouse State                                     │
    │    ├─ Committed Stock                                           │
    │    └─ In-Transit Tracking                                       │
    ├─────────────────────────────────────────────────────────────────┤
    │  Demand Forecasting                                             │
    │    ├─ ARIMA/Prophet (MVP)                                       │
    │    ├─ N-BEATS (TODO[R&D])                                       │
    │    ├─ Non-Stationary Transformer (TODO[R&D])                    │
    │    └─ SNR-based Confidence                                      │
    ├─────────────────────────────────────────────────────────────────┤
    │  ROP Engine                                                     │
    │    ├─ Dynamic ROP Calculation                                  │
    │    ├─ Safety Stock                                              │
    │    └─ 30-Day Risk Simulation                                    │
    ├─────────────────────────────────────────────────────────────────┤
    │  External Signals                                               │
    │    ├─ Commodity Prices                                          │
    │    ├─ News Sentiment                                            │
    │    └─ Macro Indicators                                         │
    ├─────────────────────────────────────────────────────────────────┤
    │  Multi-Warehouse Optimizer                                      │
    │    ├─ Transfer Optimization                                    │
    │    ├─ Purchase Planning                                         │
    │    └─ MILP Formulation                                         │
    └─────────────────────────────────────────────────────────────────┘

SIFIDE R&D Classification:
- Technical Uncertainty: Can advanced ML forecasting improve inventory optimization?
- Scientific Novelty: Integration of external signals (news, prices) into inventory decisions
- Experimental Nature: Comparative evaluation of forecasting models (ARIMA vs N-BEATS vs NST)

Mathematical Foundations:
───────────────────────
    ROP = μ_d * L + z * σ_d * sqrt(L)
    where:
        μ_d = mean daily demand (from forecast)
        σ_d = std dev of daily demand
        L = lead time (days)
        z = service level quantile (e.g., 1.96 for 95%)

    SNR = Var(signal) / Var(noise)
    where:
        signal = forecasted trend
        noise = residuals

TODO[R&D]: Future enhancements:
    - Real-time computer vision integration for stock counting
    - Reinforcement learning for dynamic ROP adjustment
    - Graph neural networks for multi-warehouse optimization
    - Integration with APS for joint production-inventory planning

Dependencies:
    - pandas: Data manipulation
    - numpy: Numerical operations
    - statsmodels: ARIMA forecasting
    - prophet: Prophet forecasting (optional)
    - scipy: Statistical functions
    - ortools: MILP optimization

Author: ProdPlan R&D Team
Version: 0.1.0 (MVP)
"""

# Conditional imports
try:
    from .iot_ingestion import (
        StockEvent,
        StockEventSource,
        ingest_rfid_event,
        ingest_vision_event,
        ingest_manual_event,
        apply_stock_event,
    )
    _HAS_IOT = True
except ImportError:
    _HAS_IOT = False
    StockEvent = None
    StockEventSource = None

try:
    from .stock_state import (
        StockState,
        WarehouseStock,
        get_realtime_stock,
        get_global_stock,
        create_stock_state,
        snapshot_stock_state,
    )
    _HAS_STOCK = True
except ImportError:
    _HAS_STOCK = False

try:
    from .demand_forecasting import (
        ForecastResult,
        ForecastModel,
        forecast_demand,
        compute_snr_forecast,
        ForecastConfig,
    )
    _HAS_FORECAST = True
except ImportError:
    _HAS_FORECAST = False

try:
    from .rop_engine import (
        ROPResult,
        compute_dynamic_rop,
        compute_risk_30d,
        ROPConfig,
    )
    _HAS_ROP = True
except ImportError:
    _HAS_ROP = False

try:
    from .external_signals import (
        ExternalSignal,
        get_commodity_price,
        fetch_news_headlines,
        get_macro_indicator,
        ExternalSignalConfig,
    )
    _HAS_SIGNALS = True
except ImportError:
    _HAS_SIGNALS = False

try:
    from .multi_warehouse_optimizer import (
        Warehouse,
        MultiWarehousePlan,
        optimize_multi_warehouse,
        OptimizationConfig,
    )
    _HAS_MULTI_WH = True
except ImportError:
    _HAS_MULTI_WH = False

try:
    from .suggestion_engine import (
        InventorySuggestion,
        SuggestionType,
        SuggestionPriority,
        generate_inventory_suggestions,
    )
    _HAS_SUGGESTIONS = True
except ImportError:
    _HAS_SUGGESTIONS = False

# BOM Engine
try:
    from .bom_engine import (
        BOMEngine,
        BOMItem,
        BOMComponent,
        BOMItemType,
        ExplodedRequirement,
        create_sample_bom,
    )
    _HAS_BOM = True
except ImportError:
    _HAS_BOM = False

# MRP Engine
try:
    from .mrp_engine import (
        MRPEngine,
        MRPResult,
        PlannedOrder,
        GrossRequirement,
        InventoryPosition,
        OrderType,
        RequirementSource,
        run_simple_mrp,
        # New high-level API
        Order,
        MRPFromOrdersEngine,
        MRPFromOrdersResult,
        PurchaseSuggestion,
        InternalOrderSuggestion,
        MRPParameters,
        run_mrp_from_orders,
    )
    _HAS_MRP = True
except ImportError:
    _HAS_MRP = False

# Forecasting Engine (Unified)
try:
    from .forecasting_engine import (
        ForecastEngineBase,
        ClassicalForecastEngine,
        AdvancedForecastEngine,
        ForecastModelType,
        ForecastConfig as UnifiedForecastConfig,
        ForecastResult as UnifiedForecastResult,
        SNRClass,
        get_forecast_engine,
        forecast_demand as unified_forecast_demand,
    )
    _HAS_UNIFIED_FORECAST = True
except ImportError:
    _HAS_UNIFIED_FORECAST = False

# BOM high-level API
try:
    from .bom_engine import (
        explode_bom,
        ComponentRequirement,
        aggregate_requirements,
    )
    _HAS_BOM_API = True
except ImportError:
    _HAS_BOM_API = False

__all__ = [
    # IoT Ingestion
    "StockEvent",
    "StockEventSource",
    "ingest_rfid_event",
    "ingest_vision_event",
    "ingest_manual_event",
    "apply_stock_event",
    # Stock State
    "StockState",
    "WarehouseStock",
    "get_realtime_stock",
    "get_global_stock",
    "create_stock_state",
    "snapshot_stock_state",
    # Forecasting
    "ForecastResult",
    "ForecastModel",
    "forecast_demand",
    "compute_snr_forecast",
    "ForecastConfig",
    # ROP
    "ROPResult",
    "compute_dynamic_rop",
    "compute_risk_30d",
    "ROPConfig",
    # External Signals
    "ExternalSignal",
    "get_commodity_price",
    "fetch_news_headlines",
    "get_macro_indicator",
    "ExternalSignalConfig",
    # Multi-Warehouse
    "Warehouse",
    "MultiWarehousePlan",
    "optimize_multi_warehouse",
    "OptimizationConfig",
    # Suggestions
    "InventorySuggestion",
    "SuggestionType",
    "SuggestionPriority",
    "generate_inventory_suggestions",
    # BOM Engine
    "BOMEngine",
    "BOMItem",
    "BOMComponent",
    "BOMItemType",
    "ExplodedRequirement",
    "create_sample_bom",
    # MRP Engine
    "MRPEngine",
    "MRPResult",
    "PlannedOrder",
    "GrossRequirement",
    "InventoryPosition",
    "OrderType",
    "RequirementSource",
    "run_simple_mrp",
    # MRP High-Level API
    "Order",
    "MRPFromOrdersEngine",
    "MRPFromOrdersResult",
    "PurchaseSuggestion",
    "InternalOrderSuggestion",
    "MRPParameters",
    "run_mrp_from_orders",
    # Unified Forecasting Engine
    "ForecastEngineBase",
    "ClassicalForecastEngine",
    "AdvancedForecastEngine",
    "ForecastModelType",
    "UnifiedForecastConfig",
    "UnifiedForecastResult",
    "SNRClass",
    "get_forecast_engine",
    "unified_forecast_demand",
    # BOM High-Level API
    "explode_bom",
    "ComponentRequirement",
    "aggregate_requirements",
]

__version__ = "0.1.0"
__author__ = "ProdPlan R&D Team"

