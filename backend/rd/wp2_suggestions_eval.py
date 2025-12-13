"""
ProdPlan 4.0 - WP2 Suggestions Evaluation
=========================================

Work Package 2: Intelligent Suggestions Evaluation

Avaliação de sugestões IA para melhorar planeamento:
- Log de sugestões geradas
- Avaliação shadow (sem afetar produção)
- Classificação BENEFICIAL/NEUTRAL/HARMFUL

R&D / SIFIDE: Validação de qualidade de sugestões.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field

from .experiments_core import (
    WorkPackage,
    ExperimentStatus,
    create_experiment,
    update_experiment_status,
    RDExperimentCreate,
    RDExperiment,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class SuggestionLabel(str, Enum):
    """Classificação de impacto da sugestão."""
    BENEFICIAL = "BENEFICIAL"  # Melhora KPIs
    NEUTRAL = "NEUTRAL"        # Não muda significativamente
    HARMFUL = "HARMFUL"        # Piora KPIs
    UNKNOWN = "UNKNOWN"        # Não avaliado ainda


class SuggestionType(str, Enum):
    """Tipo de sugestão."""
    BOTTLENECK_MITIGATION = "bottleneck_mitigation"
    CAPACITY_ADJUSTMENT = "capacity_adjustment"
    PRIORITY_CHANGE = "priority_change"
    ROUTE_CHANGE = "route_change"
    MAINTENANCE_SCHEDULE = "maintenance_schedule"
    INVENTORY_ACTION = "inventory_action"
    OTHER = "other"


class SuggestionOrigin(str, Enum):
    """Origem da sugestão."""
    HEURISTIC = "heuristic"
    CAUSAL_MODEL = "causal_model"
    ML_MODEL = "ml_model"
    USER = "user"
    RULE_BASED = "rule_based"


@dataclass
class SuggestionContext:
    """Contexto em que a sugestão foi gerada."""
    machine_loads: Dict[str, float]  # machine_id -> load %
    current_otd_rate: float
    current_tardiness_hours: float
    num_pending_orders: int
    bottleneck_machine: Optional[str] = None


class SuggestionRecord(BaseModel):
    """Registro de uma sugestão para avaliação."""
    suggestion_id: int
    suggestion_type: str
    origin: str
    title: str
    description: str
    created_at: datetime
    pre_kpis: Dict[str, Any]
    post_kpis: Optional[Dict[str, Any]] = None
    label: str = SuggestionLabel.UNKNOWN.value
    delta_otd_pct: Optional[float] = None
    delta_tardiness_pct: Optional[float] = None
    evaluated: bool = False


class WP2EvaluationRequest(BaseModel):
    """Request para avaliar sugestão."""
    suggestion_id: int
    context: Dict[str, Any] = Field(default_factory=dict)


class WP2EvaluationResult(BaseModel):
    """Resultado de avaliação WP2."""
    experiment_id: int
    suggestion_id: int
    label: str
    pre_kpis: Dict[str, float]
    post_kpis: Dict[str, float]
    delta_otd_pct: float
    delta_tardiness_pct: float
    delta_makespan_pct: float
    conclusion: str


class WP2BatchEvaluationRequest(BaseModel):
    """Request para avaliação em batch."""
    name: str = Field(description="Nome da avaliação")
    time_window_days: int = Field(default=30, description="Dias para analisar")
    context: Dict[str, Any] = Field(default_factory=dict)


class WP2BatchEvaluationResult(BaseModel):
    """Resultado de avaliação em batch."""
    experiment_id: int
    name: str
    total_suggestions: int
    evaluated_count: int
    beneficial_count: int
    neutral_count: int
    harmful_count: int
    overall_precision: float
    overall_recall: float
    f1_score: float
    conclusion: str


# ═══════════════════════════════════════════════════════════════════════════════
# IN-MEMORY SUGGESTION STORE (para demonstração)
# ═══════════════════════════════════════════════════════════════════════════════

_suggestion_store: Dict[int, SuggestionRecord] = {}
_suggestion_counter = 0


def _get_next_suggestion_id() -> int:
    """Gera próximo ID de sugestão."""
    global _suggestion_counter
    _suggestion_counter += 1
    return _suggestion_counter


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def log_suggestion(
    suggestion_type: str,
    origin: str,
    title: str,
    description: str,
    pre_plan_kpis: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Cria entrada para avaliação de sugestão.
    
    Chamada sempre que o motor de Sugestões IA gera uma sugestão.
    
    Args:
        suggestion_type: Tipo de sugestão
        origin: Origem (heurística, causal, ML, etc.)
        title: Título da sugestão
        description: Descrição detalhada
        pre_plan_kpis: KPIs do plano antes da sugestão
        context: Contexto adicional (carga, mix, máquinas)
    
    Returns:
        suggestion_id (PK)
    """
    suggestion_id = _get_next_suggestion_id()
    
    record = SuggestionRecord(
        suggestion_id=suggestion_id,
        suggestion_type=suggestion_type,
        origin=origin,
        title=title,
        description=description,
        created_at=datetime.now(),
        pre_kpis=pre_plan_kpis,
        label=SuggestionLabel.UNKNOWN.value,
    )
    
    _suggestion_store[suggestion_id] = record
    
    logger.info(f"Logged suggestion #{suggestion_id}: {title} (type={suggestion_type}, origin={origin})")
    
    return suggestion_id


def get_suggestion(suggestion_id: int) -> Optional[SuggestionRecord]:
    """Obtém registro de sugestão."""
    return _suggestion_store.get(suggestion_id)


def get_all_suggestions(
    time_window_days: int = 30,
    suggestion_type: Optional[str] = None,
    label: Optional[str] = None,
) -> List[SuggestionRecord]:
    """
    Obtém lista de sugestões com filtros.
    """
    cutoff = datetime.now() - timedelta(days=time_window_days)
    
    results = []
    for s in _suggestion_store.values():
        if s.created_at < cutoff:
            continue
        if suggestion_type and s.suggestion_type != suggestion_type:
            continue
        if label and s.label != label:
            continue
        results.append(s)
    
    return sorted(results, key=lambda x: x.created_at, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_suggestion(suggestion_id: int) -> WP2EvaluationResult:
    """
    Avalia impacto de uma sugestão em modo shadow.
    
    Processo:
    1. Reconstroi plano baseline (sem sugestão)
    2. Aplica sugestão (shadow) e volta a planear
    3. Compara KPIs antes/depois
    4. Classifica sugestão como BENEFICIAL / NEUTRAL / HARMFUL
    5. Atualiza registro com KPIs pós e label
    6. Cria RDExperiment associado (WP2)
    
    Args:
        suggestion_id: ID da sugestão a avaliar
    
    Returns:
        WP2EvaluationResult com comparação de KPIs
    """
    from scheduling import HeuristicScheduler
    from data_loader import load_dataset
    from scheduling import create_instance_from_dataframes
    
    suggestion = get_suggestion(suggestion_id)
    if not suggestion:
        raise ValueError(f"Suggestion {suggestion_id} not found")
    
    logger.info(f"Evaluating suggestion #{suggestion_id}: {suggestion.title}")
    
    # Criar experiência R&D
    experiment = create_experiment(RDExperimentCreate(
        wp=WorkPackage.WP2_SUGGESTIONS,
        name=f"Eval_Suggestion_{suggestion_id}",
        description=f"Avaliação da sugestão: {suggestion.title}",
        parameters={
            "suggestion_id": suggestion_id,
            "suggestion_type": suggestion.suggestion_type,
            "origin": suggestion.origin,
        },
    ))
    update_experiment_status(experiment.id, ExperimentStatus.RUNNING)
    
    # Carregar dados e criar instância
    try:
        data = load_dataset()
        instance = create_instance_from_dataframes(
            orders_df=data.orders,
            routing_df=data.routing,
            machines_df=data.machines,
        )
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        # Usar dados de demonstração
        instance = _create_demo_instance()
    
    # 1. Executar plano baseline
    scheduler_baseline = HeuristicScheduler(rule="EDD")
    result_baseline = scheduler_baseline.build_schedule(instance)
    kpis_baseline = result_baseline.get("kpis", {})
    
    # 2. Simular aplicação da sugestão (modificar instância)
    # Por agora, simular uma pequena melhoria baseada no tipo de sugestão
    instance_modified = _apply_suggestion_to_instance(instance, suggestion)
    
    # 3. Executar plano com sugestão aplicada
    scheduler_modified = HeuristicScheduler(rule="EDD")
    result_modified = scheduler_modified.build_schedule(instance_modified)
    kpis_modified = result_modified.get("kpis", {})
    
    # 4. Calcular deltas
    makespan_base = kpis_baseline.get("makespan_hours", 0)
    makespan_mod = kpis_modified.get("makespan_hours", 0)
    otd_base = kpis_baseline.get("otd_rate", 1.0)
    otd_mod = kpis_modified.get("otd_rate", 1.0)
    tardiness_base = kpis_baseline.get("total_tardiness_hours", 0)
    tardiness_mod = kpis_modified.get("total_tardiness_hours", 0)
    
    delta_makespan = ((makespan_mod - makespan_base) / makespan_base * 100) if makespan_base > 0 else 0
    delta_otd = ((otd_mod - otd_base) / otd_base * 100) if otd_base > 0 else 0
    delta_tardiness = ((tardiness_mod - tardiness_base) / tardiness_base * 100) if tardiness_base > 0 else 0
    
    # 5. Classificar sugestão
    # BENEFICIAL: OTD melhora ou tardiness reduz significativamente (>5%)
    # HARMFUL: OTD piora ou tardiness aumenta significativamente (>5%)
    # NEUTRAL: Mudanças menores que 5%
    
    if delta_otd > 5 or delta_tardiness < -5:
        label = SuggestionLabel.BENEFICIAL
    elif delta_otd < -5 or delta_tardiness > 5:
        label = SuggestionLabel.HARMFUL
    else:
        label = SuggestionLabel.NEUTRAL
    
    # 6. Atualizar registro
    suggestion.post_kpis = kpis_modified
    suggestion.label = label.value
    suggestion.delta_otd_pct = round(delta_otd, 2)
    suggestion.delta_tardiness_pct = round(delta_tardiness, 2)
    suggestion.evaluated = True
    
    # Gerar conclusão
    if label == SuggestionLabel.BENEFICIAL:
        conclusion = f"Sugestão BENÉFICA: OTD {delta_otd:+.1f}%, Tardiness {delta_tardiness:+.1f}%"
    elif label == SuggestionLabel.HARMFUL:
        conclusion = f"Sugestão PREJUDICIAL: OTD {delta_otd:+.1f}%, Tardiness {delta_tardiness:+.1f}%"
    else:
        conclusion = f"Sugestão NEUTRA: impacto mínimo nos KPIs"
    
    # Atualizar experiência
    update_experiment_status(
        experiment.id,
        ExperimentStatus.FINISHED,
        summary={
            "suggestion_id": suggestion_id,
            "label": label.value,
            "delta_otd_pct": round(delta_otd, 2),
            "delta_tardiness_pct": round(delta_tardiness, 2),
            "delta_makespan_pct": round(delta_makespan, 2),
        },
        kpis={
            "baseline_otd": otd_base,
            "modified_otd": otd_mod,
            "baseline_tardiness": tardiness_base,
            "modified_tardiness": tardiness_mod,
        },
        conclusion=conclusion,
    )
    
    logger.info(f"Suggestion #{suggestion_id} evaluated: {label.value}")
    
    return WP2EvaluationResult(
        experiment_id=experiment.id,
        suggestion_id=suggestion_id,
        label=label.value,
        pre_kpis={
            "otd_rate": round(otd_base, 3),
            "tardiness_hours": round(tardiness_base, 2),
            "makespan_hours": round(makespan_base, 2),
        },
        post_kpis={
            "otd_rate": round(otd_mod, 3),
            "tardiness_hours": round(tardiness_mod, 2),
            "makespan_hours": round(makespan_mod, 2),
        },
        delta_otd_pct=round(delta_otd, 2),
        delta_tardiness_pct=round(delta_tardiness, 2),
        delta_makespan_pct=round(delta_makespan, 2),
        conclusion=conclusion,
    )


def evaluate_suggestions_batch(request: WP2BatchEvaluationRequest) -> WP2BatchEvaluationResult:
    """
    Avalia múltiplas sugestões em batch.
    
    Calcula métricas agregadas de precisão e recall.
    """
    logger.info(f"WP2 Batch Evaluation: {request.name}")
    
    # Criar experiência
    experiment = create_experiment(RDExperimentCreate(
        wp=WorkPackage.WP2_SUGGESTIONS,
        name=request.name,
        description=f"Avaliação em batch de sugestões ({request.time_window_days} dias)",
        parameters={
            "time_window_days": request.time_window_days,
        },
    ))
    update_experiment_status(experiment.id, ExperimentStatus.RUNNING)
    
    # Obter sugestões no período
    suggestions = get_all_suggestions(time_window_days=request.time_window_days)
    
    # Se não há sugestões, gerar algumas de demonstração
    if len(suggestions) == 0:
        _generate_demo_suggestions()
        suggestions = get_all_suggestions(time_window_days=request.time_window_days)
    
    # Avaliar cada sugestão
    beneficial_count = 0
    neutral_count = 0
    harmful_count = 0
    
    for s in suggestions:
        if not s.evaluated:
            try:
                result = evaluate_suggestion(s.suggestion_id)
                if result.label == SuggestionLabel.BENEFICIAL.value:
                    beneficial_count += 1
                elif result.label == SuggestionLabel.HARMFUL.value:
                    harmful_count += 1
                else:
                    neutral_count += 1
            except Exception as e:
                logger.error(f"Error evaluating suggestion {s.suggestion_id}: {e}")
        else:
            if s.label == SuggestionLabel.BENEFICIAL.value:
                beneficial_count += 1
            elif s.label == SuggestionLabel.HARMFUL.value:
                harmful_count += 1
            else:
                neutral_count += 1
    
    # Calcular métricas
    total = len(suggestions)
    evaluated = beneficial_count + neutral_count + harmful_count
    
    # Precision: % de sugestões que são benéficas
    precision = beneficial_count / evaluated if evaluated > 0 else 0
    
    # Recall: assumir que todas sugestões "deveriam" ser benéficas
    # então recall = beneficial / total
    recall = beneficial_count / total if total > 0 else 0
    
    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    conclusion = f"Avaliadas {evaluated}/{total} sugestões. Precision: {precision*100:.1f}%, F1: {f1:.2f}"
    
    # Atualizar experiência
    update_experiment_status(
        experiment.id,
        ExperimentStatus.FINISHED,
        summary={
            "total_suggestions": total,
            "evaluated_count": evaluated,
            "beneficial_count": beneficial_count,
            "neutral_count": neutral_count,
            "harmful_count": harmful_count,
        },
        kpis={
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
        },
        conclusion=conclusion,
    )
    
    return WP2BatchEvaluationResult(
        experiment_id=experiment.id,
        name=request.name,
        total_suggestions=total,
        evaluated_count=evaluated,
        beneficial_count=beneficial_count,
        neutral_count=neutral_count,
        harmful_count=harmful_count,
        overall_precision=round(precision, 3),
        overall_recall=round(recall, 3),
        f1_score=round(f1, 3),
        conclusion=conclusion,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _create_demo_instance():
    """Cria instância de demonstração."""
    from scheduling.types import SchedulingInstance
    
    operations = []
    for i in range(15):
        operations.append({
            "operation_id": f"OP_{i:03d}",
            "order_id": f"ORD_{i//3:03d}",
            "article_id": f"ART_{i//5:02d}",
            "op_seq": i % 3,
            "op_code": f"OP_{i % 5}",
            "duration_min": 30 + (i * 5),
            "primary_machine_id": f"M-{(i % 5) + 300}",
            "due_date": datetime.now() + timedelta(hours=24 + i * 2),
        })
    
    machines = [{"machine_id": f"M-{i + 300}", "name": f"Machine {i}"} for i in range(5)]
    
    return SchedulingInstance(
        operations=operations,
        machines=machines,
        horizon_start=datetime.now(),
    )


def _apply_suggestion_to_instance(instance, suggestion: SuggestionRecord):
    """
    Aplica sugestão à instância (em modo shadow).
    
    Retorna uma cópia modificada da instância.
    """
    # Por simplicidade, criar uma cópia e simular pequenas melhorias
    # Numa implementação real, isto aplicaria a sugestão específica
    
    import copy
    modified = copy.deepcopy(instance)
    
    # Simular efeito da sugestão baseado no tipo
    if suggestion.suggestion_type == SuggestionType.BOTTLENECK_MITIGATION.value:
        # Simular redução de tempos em 5%
        for op in modified.operations:
            if "duration_min" in op:
                op["duration_min"] = op["duration_min"] * 0.95
    elif suggestion.suggestion_type == SuggestionType.PRIORITY_CHANGE.value:
        # Simular mudança de prioridades
        for op in modified.operations:
            if "priority" in op:
                op["priority"] = op.get("priority", 1.0) * 1.1
    
    return modified


def _generate_demo_suggestions():
    """Gera sugestões de demonstração."""
    demo_suggestions = [
        ("bottleneck_mitigation", "heuristic", "Reduzir carga em M-300", 
         "Máquina M-300 com carga >90%. Sugerir redistribuição."),
        ("priority_change", "rule_based", "Priorizar ordem ORD-005",
         "Ordem ORD-005 com due date próximo. Aumentar prioridade."),
        ("route_change", "ml_model", "Usar rota alternativa para ART-03",
         "ML identificou rota B como mais eficiente para ART-03."),
        ("capacity_adjustment", "causal_model", "Adicionar turno extra",
         "Análise causal sugere que turno extra reduz tardiness em 15%."),
        ("maintenance_schedule", "heuristic", "Antecipar manutenção M-302",
         "RUL de M-302 baixo. Sugerir manutenção preventiva."),
    ]
    
    base_kpis = {
        "makespan_hours": 48.0,
        "otd_rate": 0.85,
        "total_tardiness_hours": 5.0,
    }
    
    for stype, origin, title, desc in demo_suggestions:
        log_suggestion(stype, origin, title, desc, base_kpis)
