"""
════════════════════════════════════════════════════════════════════════════════════════════════════
DIGITAL TWIN MODULE - PdM-Integrated Production Scheduling (PdM-IPS)
════════════════════════════════════════════════════════════════════════════════════════════════════

Este módulo implementa um Digital Twin para Prognóstico de Vida Útil Remanescente (RUL) por 
máquina/operação, usando modelos generativos (CVAE) e integração com o APS.

Componentes:
- health_indicator_cvae.py: CVAE para extrair Health Indicators (HI) específicos por operação
- rul_estimator.py: Estimação de RUL com incerteza (Gaussian Process / Bayesian)
- rul_integration_scheduler.py: Integração RUL-APS para scheduling adaptativo

Inovação:
- O scheduling passa a ser **PdM-integrated**: o plano "sabe" que máquina está perto de falhar
  e adapta-se preventivamente.
- Usa CVAE para aprender Health Indicators específicos por operação (HI), com embedding do
  contexto da operação.

TODO[R&D]:
- Deep Bayesian (MC Dropout, HMC/VI) para melhor estimativa de incerteza
- Integração com sensores IoT reais
- Transfer learning para novos tipos de máquinas
- Explicabilidade das decisões de manutenção

Referências:
- "Remaining Useful Life Estimation Using CVAE" (Li et al., 2020)
- "Integrated Production and Maintenance Scheduling" (Xiao et al., 2019)
- "Health Indicator Construction for Rotating Machinery" (Wang et al., 2021)
"""

from digital_twin.health_indicator_cvae import (
    CVAE,
    CVAEConfig,
    SensorSnapshot,
    OperationContext,
    HealthIndicatorResult,
    train_cvae,
    infer_hi,
    create_demo_dataset,
)

from digital_twin.rul_estimator import (
    RULEstimate,
    RULEstimatorConfig,
    RULEstimator,
    estimate_rul,
    get_machine_health_status,
    HealthStatus,
    create_demo_hi_history,
)

from digital_twin.rul_integration_scheduler import (
    RULAdjustmentConfig,
    MachineRULInfo,
    PlanAdjustmentResult,
    adjust_plan_with_rul,
    get_rul_penalties,
    should_avoid_machine,
)

# XAI Digital Twin Geometry (Contract 4 & 6)
try:
    from .xai_dt_geometry import (
        # Data structures
        DeviationField,
        PODConfig,
        PODResult,
        XAIConfig,
        XaiDtExplanation,
        ProvableCause,
        AnomalyScore,
        SensitivityResult,
        # Engines
        DeviationEngineBase,
        SimpleDeviationEngine,
        PodDeviationEngine,
        DeviationSurrogateModel,
        # Factory & convenience
        get_deviation_engine,
        explain_deviation,
        create_test_deviation_field,
        analyze_machine_data,
        compute_pca_pod,
        build_surrogate_model,
        explain_rul_factors,
        # Legacy alias
        PODEngine,
    )
    _HAS_XAI_GEOMETRY = True
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"XAI Geometry not available: {e}")
    _HAS_XAI_GEOMETRY = False

# Process Optimization (Contract 6)
try:
    from .process_optimization import (
        compute_golden_runs,
        suggest_process_params,
        get_golden_runs,
        analyze_parameter_impact,
        predict_quality,
    )
    _HAS_PROCESS_OPTIMIZATION = True
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Process Optimization not available: {e}")
    _HAS_PROCESS_OPTIMIZATION = False

# API Router (Contract 6)
try:
    from .api_xai_dt import router as xai_dt_router
    _HAS_XAI_API = True
except ImportError:
    xai_dt_router = None
    _HAS_XAI_API = False

# SHI-DT (Smart Health Index Digital Twin)
try:
    from digital_twin.shi_dt import (
        SHIDT,
        SHIDTConfig,
        get_shidt,
        reset_shidt,
        HealthIndexReading,
        RULEstimate as SHIDTRULEstimate,
        SensorContribution,
        HealthAlert,
        MachineHealthStatus,
        OperationalProfile,
        AlertSeverity,
        ProfileDetector,
        RULCalculator,
        ExplainabilityEngine,
        AlertGenerator,
    )
    from digital_twin.api_shi_dt import router as shi_dt_router
    _HAS_SHIDT = True
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"SHI-DT not available: {e}")
    shi_dt_router = None
    _HAS_SHIDT = False

# XAI-DT Product (Explainable Digital Twin do Produto)
try:
    from digital_twin.xai_dt_product import (
        XAIDTConfig,
        XAIDTProductAnalyzer,
        get_xai_dt_analyzer,
        reset_xai_dt_analyzer,
        PointCloud,
        DeviationField3D,
        PCAResult,
        RegionalAnalysis,
        IdentifiedPattern,
        RootCause,
        CorrectiveAction,
        XAIDTAnalysisResult,
        ICPAligner,
        DeviationComputer,
        PatternAnalyzer,
        RootCauseAnalyzer,
        DeviationPattern,
        RootCauseCategory,
        create_demo_cad_scan,
    )
    from digital_twin.api_xai_dt_product import router as xai_dt_product_router
    _HAS_XAI_DT_PRODUCT = True
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"XAI-DT Product not available: {e}")
    xai_dt_product_router = None
    _HAS_XAI_DT_PRODUCT = False

__all__ = [
    # CVAE Health Indicator
    "CVAE",
    "CVAEConfig",
    "SensorSnapshot",
    "OperationContext",
    "HealthIndicatorResult",
    "train_cvae",
    "infer_hi",
    "create_demo_dataset",
    # RUL Estimator
    "RULEstimate",
    "RULEstimatorConfig",
    "RULEstimator",
    "estimate_rul",
    "get_machine_health_status",
    "HealthStatus",
    "create_demo_hi_history",
    # Scheduler Integration
    "RULAdjustmentConfig",
    "MachineRULInfo",
    "PlanAdjustmentResult",
    "adjust_plan_with_rul",
    "get_rul_penalties",
    "should_avoid_machine",
    # XAI Geometry (Contract 4 & 6)
    "DeviationField",
    "PODConfig",
    "PODResult",
    "XAIConfig",
    "XaiDtExplanation",
    "ProvableCause",
    "AnomalyScore",
    "SensitivityResult",
    "DeviationEngineBase",
    "SimpleDeviationEngine",
    "PodDeviationEngine",
    "DeviationSurrogateModel",
    "get_deviation_engine",
    "explain_deviation",
    "create_test_deviation_field",
    "analyze_machine_data",
    "compute_pca_pod",
    "build_surrogate_model",
    "explain_rul_factors",
    "PODEngine",  # Legacy alias
    # Process Optimization (Contract 6)
    "compute_golden_runs",
    "suggest_process_params",
    "get_golden_runs",
    "analyze_parameter_impact",
    "predict_quality",
    # API Router
    "xai_dt_router",
    # SHI-DT (Smart Health Index Digital Twin)
    "SHIDT",
    "SHIDTConfig",
    "get_shidt",
    "reset_shidt",
    "HealthIndexReading",
    "SHIDTRULEstimate",
    "SensorContribution",
    "HealthAlert",
    "MachineHealthStatus",
    "OperationalProfile",
    "AlertSeverity",
    "ProfileDetector",
    "RULCalculator",
    "ExplainabilityEngine",
    "AlertGenerator",
    "shi_dt_router",
    # XAI-DT Product
    "XAIDTConfig",
    "XAIDTProductAnalyzer",
    "get_xai_dt_analyzer",
    "reset_xai_dt_analyzer",
    "PointCloud",
    "DeviationField3D",
    "PCAResult",
    "RegionalAnalysis",
    "IdentifiedPattern",
    "RootCause",
    "CorrectiveAction",
    "XAIDTAnalysisResult",
    "ICPAligner",
    "DeviationComputer",
    "PatternAnalyzer",
    "RootCauseAnalyzer",
    "DeviationPattern",
    "RootCauseCategory",
    "create_demo_cad_scan",
    "xai_dt_product_router",
]

