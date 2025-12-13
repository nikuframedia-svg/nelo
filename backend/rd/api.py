"""
ProdPlan 4.0 - R&D API Endpoints
================================

API REST para módulo de Research & Development.

Endpoints:
- GET  /rd/status               - Status geral do módulo R&D
- GET  /rd/experiments          - Lista experiências
- GET  /rd/experiments/{id}     - Detalhes de experiência
- POST /rd/wp1/run              - Executar experiência WP1 (Routing)
- POST /rd/wp2/evaluate         - Avaliar sugestão WP2
- POST /rd/wp2/evaluate-batch   - Avaliação em batch WP2
- POST /rd/wp3/run-scenario     - Executar cenário WP3 (Inventory)
- POST /rd/wp3/compare          - Comparar políticas WP3
- POST /rd/wp4/run-episode      - Executar episódio WP4 (Learning)
- GET  /rd/report/summary       - Resumo R&D para período
- GET  /rd/report/export        - Exportar relatório SIFIDE
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .experiments_core import (
    WorkPackage,
    ExperimentStatus,
    RDExperiment,
    list_experiments,
    get_experiment,
    get_experiments_summary,
)
from .wp1_routing_experiments import (
    WP1RoutingRequest,
    WP1RoutingExperiment,
    WP1PolicyResult,
    run_routing_comparison,
    get_routing_strategies,
)
from .wp2_suggestions_eval import (
    WP2EvaluationRequest,
    WP2EvaluationResult,
    WP2BatchEvaluationRequest,
    WP2BatchEvaluationResult,
    SuggestionRecord,
    SuggestionType,
    SuggestionLabel,
    evaluate_suggestion,
    evaluate_suggestions_batch,
    get_all_suggestions,
    log_suggestion,
)
from .wp3_inventory_capacity import (
    WP3ScenarioRequest,
    WP3ScenarioResult,
    WP3Experiment,
    WP3ComparisonRequest,
    InventoryPolicy,
    DateRange,
    run_inventory_capacity_scenario,
    compare_inventory_policies,
    get_default_policies,
)
from .wp4_learning_scheduler import (
    WP4RunRequest,
    WP4ExperimentResult,
    PolicyStats,
    EpisodeResult,
    run_learning_experiment,
    BanditScheduler,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rd", tags=["R&D"])


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class RDStatusResponse(BaseModel):
    """Status do módulo R&D."""
    available: bool = True
    version: str = "2.0.0"
    work_packages: List[str] = ["WP1_ROUTING", "WP2_SUGGESTIONS", "WP3_INVENTORY_CAPACITY", "WP4_LEARNING_SCHEDULER"]
    experiments_summary: dict
    feature_flags_active: bool = True


class ExperimentCreatedResponse(BaseModel):
    """Resposta de criação de experiência."""
    experiment_id: int
    status: str
    message: str


# ═══════════════════════════════════════════════════════════════════════════════
# GENERAL ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/status", response_model=RDStatusResponse)
async def get_rd_status():
    """
    Obtém status geral do módulo R&D.
    """
    try:
        summary = get_experiments_summary()
        return RDStatusResponse(
            available=True,
            experiments_summary=summary,
        )
    except Exception as e:
        logger.error(f"Error getting R&D status: {e}")
        return RDStatusResponse(
            available=False,
            experiments_summary={"error": str(e)},
        )


@router.get("/experiments", response_model=List[RDExperiment])
async def list_rd_experiments(
    wp: Optional[str] = Query(None, description="Filtrar por Work Package"),
    status: Optional[str] = Query(None, description="Filtrar por status"),
    limit: int = Query(50, ge=1, le=500, description="Limite de resultados"),
):
    """
    Lista experiências R&D com filtros opcionais.
    """
    try:
        wp_filter = WorkPackage(wp) if wp else None
        status_filter = ExperimentStatus(status) if status else None
        
        experiments = list_experiments(
            wp=wp_filter,
            status=status_filter,
            limit=limit,
        )
        return experiments
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}", response_model=RDExperiment)
async def get_rd_experiment(experiment_id: int):
    """
    Obtém detalhes de uma experiência específica.
    """
    experiment = get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    return experiment


# ═══════════════════════════════════════════════════════════════════════════════
# WP1 - ROUTING ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/wp1/strategies")
async def get_wp1_strategies():
    """
    Obtém estratégias de routing disponíveis.
    """
    return get_routing_strategies()


@router.post("/wp1/run", response_model=WP1RoutingExperiment)
async def run_wp1_experiment(request: WP1RoutingRequest):
    """
    Executa experiência de comparação de routing (WP1).
    
    Compara múltiplas políticas de scheduling e calcula KPIs:
    - Makespan
    - Tardiness
    - OTD Rate
    - Setup Time
    
    Políticas disponíveis: FIFO, SPT, EDD, CR, WSPT, MILP, CPSAT
    """
    try:
        result = run_routing_comparison(request)
        return result
    except Exception as e:
        logger.error(f"WP1 experiment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# WP2 - SUGGESTIONS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/wp2/suggestion-types")
async def get_wp2_suggestion_types():
    """
    Obtém tipos de sugestões disponíveis.
    """
    return [
        {"id": t.value, "name": t.value.replace("_", " ").title()}
        for t in SuggestionType
    ]


@router.get("/wp2/suggestions", response_model=List[SuggestionRecord])
async def list_wp2_suggestions(
    time_window_days: int = Query(30, description="Dias para filtrar"),
    suggestion_type: Optional[str] = Query(None, description="Filtrar por tipo"),
    label: Optional[str] = Query(None, description="Filtrar por label"),
):
    """
    Lista sugestões registadas para avaliação.
    """
    try:
        suggestions = get_all_suggestions(
            time_window_days=time_window_days,
            suggestion_type=suggestion_type,
            label=label,
        )
        return suggestions
    except Exception as e:
        logger.error(f"Error listing suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class LogSuggestionRequest(BaseModel):
    """Request para registar sugestão."""
    suggestion_type: str
    origin: str
    title: str
    description: str
    pre_kpis: Dict[str, Any]


@router.post("/wp2/log-suggestion")
async def log_wp2_suggestion(request: LogSuggestionRequest):
    """
    Regista uma sugestão para avaliação futura.
    """
    try:
        suggestion_id = log_suggestion(
            suggestion_type=request.suggestion_type,
            origin=request.origin,
            title=request.title,
            description=request.description,
            pre_plan_kpis=request.pre_kpis,
        )
        return {"suggestion_id": suggestion_id, "status": "logged"}
    except Exception as e:
        logger.error(f"Error logging suggestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/wp2/evaluate-suggestion", response_model=WP2EvaluationResult)
async def run_wp2_evaluate_suggestion(request: WP2EvaluationRequest):
    """
    Avalia impacto de uma sugestão em modo shadow (WP2).
    
    Processo:
    1. Reconstroi plano baseline
    2. Aplica sugestão (shadow)
    3. Compara KPIs antes/depois
    4. Classifica como BENEFICIAL / NEUTRAL / HARMFUL
    """
    try:
        result = evaluate_suggestion(request.suggestion_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"WP2 evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/wp2/evaluate-batch", response_model=WP2BatchEvaluationResult)
async def run_wp2_batch_evaluation(request: WP2BatchEvaluationRequest):
    """
    Avalia múltiplas sugestões em batch (WP2).
    
    Calcula métricas agregadas:
    - Precision: % de sugestões benéficas
    - Recall: % de cobertura
    - F1 Score
    """
    try:
        result = evaluate_suggestions_batch(request)
        return result
    except Exception as e:
        logger.error(f"WP2 batch evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# WP3 - INVENTORY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/wp3/policies", response_model=List[InventoryPolicy])
async def get_wp3_policies():
    """
    Obtém políticas de inventário padrão para comparação.
    """
    return get_default_policies()


@router.post("/wp3/run-scenario")
async def run_wp3_scenario(request: WP3ScenarioRequest):
    """
    Executa cenário de inventário + capacidade (WP3).
    
    Simula uma política de inventário e calcula KPIs:
    - Service Level
    - Stock médio
    - Stockouts
    - Custo total
    - Impacto em OTD
    """
    try:
        experiment = run_inventory_capacity_scenario(
            policy=request.policy,
            horizon=request.horizon,
            baseline_policy=request.baseline_policy,
            experiment_name=request.name,
        )
        return {"experiment_id": experiment.id, "status": experiment.status.value}
    except Exception as e:
        logger.error(f"WP3 scenario failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/wp3/compare", response_model=WP3Experiment)
async def run_wp3_comparison(request: WP3ComparisonRequest):
    """
    Compara múltiplas políticas de inventário (WP3).
    
    Executa cada política no mesmo horizonte e identifica:
    - Melhor trade-off custo/serviço
    - Recomendação de política
    """
    try:
        result = compare_inventory_policies(request)
        return result
    except Exception as e:
        logger.error(f"WP3 comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# WP4 - LEARNING SCHEDULER ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/wp4/bandit-types")
async def get_wp4_bandit_types():
    """
    Obtém tipos de algoritmos bandit disponíveis.
    """
    return [
        {"id": "epsilon_greedy", "name": "Epsilon-Greedy", "description": "Exploração com probabilidade epsilon"},
        {"id": "ucb1", "name": "UCB1", "description": "Upper Confidence Bound"},
        {"id": "thompson_sampling", "name": "Thompson Sampling", "description": "Amostragem Bayesiana"},
    ]


@router.get("/wp4/reward-types")
async def get_wp4_reward_types():
    """
    Obtém tipos de reward disponíveis.
    """
    return [
        {"id": "makespan", "name": "Makespan", "description": "Minimizar makespan"},
        {"id": "otd", "name": "OTD Rate", "description": "Maximizar taxa de entrega a tempo"},
        {"id": "combined", "name": "Combinado", "description": "Ponderação de tardiness + makespan + setups"},
    ]


@router.post("/wp4/run-episode", response_model=WP4ExperimentResult)
async def run_wp4_episode(request: WP4RunRequest):
    """
    Executa experiência de learning scheduler (WP4).
    
    Usa multi-armed bandit para aprender a melhor política:
    - Epsilon-greedy ou UCB1 ou Thompson Sampling
    - Corre N episódios
    - Calcula regret vs baseline
    - Identifica melhor política
    """
    try:
        result = run_learning_experiment(request)
        return result
    except Exception as e:
        logger.error(f"WP4 experiment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE FLAGS ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/feature-flags")
async def get_feature_flags():
    """
    Obtém feature flags atuais.
    """
    try:
        from feature_flags import FeatureFlags
        return FeatureFlags.to_dict()
    except ImportError:
        return {"error": "Feature flags module not available"}
    except Exception as e:
        return {"error": str(e)}


@router.post("/feature-flags/{engine_type}/{value}")
async def set_feature_flag(engine_type: str, value: str):
    """
    Define feature flag em runtime (apenas para R&D/testing).
    """
    try:
        from feature_flags import FeatureFlags
        success = FeatureFlags.set_engine(engine_type, value)
        if success:
            return {"status": "updated", "engine": engine_type, "value": value}
        else:
            raise HTTPException(status_code=400, detail=f"Invalid engine type or value")
    except ImportError:
        raise HTTPException(status_code=500, detail="Feature flags module not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING ENDPOINTS (CONTRACT 11 - SIFIDE)
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/report/summary")
async def get_rd_report_summary(
    start: str = Query(..., description="Data início (YYYY-MM-DD)"),
    end: str = Query(..., description="Data fim (YYYY-MM-DD)"),
):
    """
    Obtém resumo consolidado de R&D para um período.
    
    Útil para relatórios SIFIDE e dossiês de I&D.
    
    Inclui:
    - Total de experiências por WP
    - Hipóteses testadas e resultados
    - Métricas agregadas
    """
    try:
        from .reporting import build_rd_summary, generate_demo_rd_data
        from datetime import datetime
        
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date()
        
        summary = build_rd_summary(start_date, end_date)
        
        # If no real experiments, use demo data
        if summary.get("total_experiments", 0) == 0:
            year = start_date.year
            demo = generate_demo_rd_data(year)
            demo["is_demo"] = True
            return demo
        
        return summary
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logger.error(f"Error building R&D summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report/export")
async def export_rd_report(
    start: str = Query(..., description="Data início (YYYY-MM-DD)"),
    end: str = Query(..., description="Data fim (YYYY-MM-DD)"),
    format: str = Query("json", description="Formato: json ou pdf"),
):
    """
    Exporta relatório R&D para download.
    
    Formatos disponíveis:
    - json: Estrutura completa em JSON
    - pdf: Relatório formatado (requer reportlab)
    """
    try:
        from .reporting import export_rd_report as do_export, generate_demo_rd_data
        from datetime import datetime
        import json
        
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date()
        
        content, filename, content_type = do_export(start_date, end_date, format)
        
        # If no content (no experiments), generate demo
        if len(content) < 100:
            demo = generate_demo_rd_data(start_date.year)
            if format.lower() == "json":
                content = json.dumps(demo, indent=2, ensure_ascii=False).encode("utf-8")
        
        return Response(
            content=content,
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format or format type: {e}")
    except Exception as e:
        logger.error(f"Error exporting R&D report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report/years")
async def get_available_years():
    """
    Obtém anos disponíveis para relatórios.
    """
    try:
        from .experiments_core import get_db_session
        from sqlalchemy import text
        
        with get_db_session() as db:
            query = text("""
                SELECT DISTINCT strftime('%Y', created_at) as year
                FROM rd_experiments
                ORDER BY year DESC
            """)
            rows = db.execute(query).fetchall()
            years = [int(row[0]) for row in rows if row[0]]
        
        # Always include current year
        from datetime import datetime
        current_year = datetime.now().year
        if current_year not in years:
            years.insert(0, current_year)
        
        return {"years": years}
        
    except Exception as e:
        logger.warning(f"Error getting available years: {e}")
        # Fallback to recent years
        from datetime import datetime
        current_year = datetime.now().year
        return {"years": [current_year, current_year - 1]}


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/summary/wp1")
async def get_wp1_summary():
    """
    Obtém resumo das experiências WP1.
    """
    experiments = list_experiments(wp=WorkPackage.WP1_ROUTING, limit=100)
    
    finished = [e for e in experiments if e.status == ExperimentStatus.FINISHED]
    
    return {
        "total_experiments": len(experiments),
        "finished": len(finished),
        "avg_improvement_pct": sum(e.summary.get("improvement_pct", 0) for e in finished) / len(finished) if finished else 0,
        "best_policies": _get_best_policies_from_experiments(finished),
    }


@router.get("/summary/wp2")
async def get_wp2_summary():
    """
    Obtém resumo das avaliações WP2.
    """
    experiments = list_experiments(wp=WorkPackage.WP2_SUGGESTIONS, limit=100)
    suggestions = get_all_suggestions(time_window_days=30)
    
    return {
        "total_experiments": len(experiments),
        "total_suggestions": len(suggestions),
        "by_label": {
            "BENEFICIAL": len([s for s in suggestions if s.label == SuggestionLabel.BENEFICIAL.value]),
            "NEUTRAL": len([s for s in suggestions if s.label == SuggestionLabel.NEUTRAL.value]),
            "HARMFUL": len([s for s in suggestions if s.label == SuggestionLabel.HARMFUL.value]),
            "UNKNOWN": len([s for s in suggestions if s.label == SuggestionLabel.UNKNOWN.value]),
        },
    }


@router.get("/summary/wp3")
async def get_wp3_summary():
    """
    Obtém resumo das experiências WP3.
    """
    experiments = list_experiments(wp=WorkPackage.WP3_INVENTORY, limit=100)
    
    finished = [e for e in experiments if e.status == ExperimentStatus.FINISHED]
    
    return {
        "total_experiments": len(experiments),
        "finished": len(finished),
        "avg_service_level": sum(e.kpis.get("service_level_pct", 0) for e in finished) / len(finished) if finished else 0,
        "recommended_policies": _get_recommended_policies(finished),
    }


@router.get("/summary/wp4")
async def get_wp4_summary():
    """
    Obtém resumo das experiências WP4.
    """
    experiments = list_experiments(wp=WorkPackage.WP4_LEARNING, limit=100)
    
    finished = [e for e in experiments if e.status == ExperimentStatus.FINISHED]
    
    return {
        "total_experiments": len(experiments),
        "finished": len(finished),
        "total_episodes": sum(e.parameters.get("num_episodes", 0) for e in finished),
        "avg_cumulative_regret": sum(e.kpis.get("cumulative_regret", 0) for e in finished) / len(finished) if finished else 0,
        "best_policies_learned": _get_best_learned_policies(finished),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_best_policies_from_experiments(experiments: List[RDExperiment]) -> Dict[str, int]:
    """Conta políticas mais frequentemente selecionadas como melhores."""
    counts = {}
    for exp in experiments:
        best = exp.summary.get("best_policy")
        if best:
            counts[best] = counts.get(best, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5])


def _get_recommended_policies(experiments: List[RDExperiment]) -> List[str]:
    """Obtém políticas recomendadas das experiências."""
    policies = set()
    for exp in experiments:
        policy = exp.summary.get("best_policy") or exp.summary.get("policy_name")
        if policy:
            policies.add(policy)
    return list(policies)[:5]


def _get_best_learned_policies(experiments: List[RDExperiment]) -> Dict[str, int]:
    """Conta políticas aprendidas como melhores."""
    counts = {}
    for exp in experiments:
        best = exp.summary.get("best_policy")
        if best:
            counts[best] = counts.get(best, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5])
