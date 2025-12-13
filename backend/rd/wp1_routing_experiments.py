"""
ProdPlan 4.0 - WP1 Routing Experiments
======================================

Work Package 1: Routing Intelligence

Experiências para comparar e otimizar estratégias de routing:
- Comparação de heurísticas (FIFO, SPT, EDD, CR, WSPT)
- MILP vs heurísticas
- Routing dinâmico vs estático

R&D / SIFIDE: Documentação de experiências de routing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

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

class WP1RoutingRequest(BaseModel):
    """Request para experiência WP1."""
    name: str = Field(description="Nome da experiência")
    policies: List[str] = Field(
        default=["FIFO", "SPT", "EDD"],
        description="Políticas a comparar"
    )
    baseline_policy: str = Field(
        default="FIFO",
        description="Política baseline para comparação"
    )
    date_start: Optional[datetime] = Field(
        default=None,
        description="Data início para carregar dados"
    )
    date_end: Optional[datetime] = Field(
        default=None,
        description="Data fim para carregar dados"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Contexto adicional"
    )


class WP1PolicyResult(BaseModel):
    """Resultado de uma política."""
    policy: str
    makespan_hours: float
    tardiness_hours: float
    setup_hours: float
    otd_rate: float
    num_late_orders: int
    total_operations: int
    solve_time_sec: float
    vs_baseline_makespan_pct: Optional[float] = None
    vs_baseline_tardiness_pct: Optional[float] = None


class WP1RoutingExperiment(BaseModel):
    """Resultado completo de experiência WP1."""
    experiment_id: int
    name: str
    status: str
    policies_tested: List[str]
    baseline_policy: str
    results: List[WP1PolicyResult]
    best_policy: Optional[str] = None
    improvement_vs_baseline_pct: Optional[float] = None
    conclusion: Optional[str] = None
    total_time_sec: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_routing_experiment(
    instance,  # SchedulingInstance
    policies: List[str],
    baseline_policy: str,
    experiment_name: str = None,
) -> RDExperiment:
    """
    Executa experiência de comparação de routing.
    
    Para o mesmo instance (ordens+máquinas):
    - Corre cada política pedida (FIFO, SPT, EDD, SQ, SETUP_AWARE, MILP/CP-SAT)
    - Calcula KPIs (SchedulingKPIs) com base em SchedulingResult
    - Compara com a baseline
    - Grava em rd_experiments
    - Devolve RDExperiment com summary: {policy -> KPIs, best_policy, regrets}
    
    Args:
        instance: SchedulingInstance com operações e máquinas
        policies: Lista de políticas a testar
        baseline_policy: Política para comparação
        experiment_name: Nome da experiência
    
    Returns:
        RDExperiment com resultados
    """
    from scheduling import HeuristicScheduler, solve_milp, solve_cpsat
    
    start_time = time.time()
    experiment_name = experiment_name or f"WP1_Routing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Starting WP1 Routing Experiment: {experiment_name}")
    logger.info(f"Policies: {policies}, Baseline: {baseline_policy}")
    
    # Criar experiência
    experiment = create_experiment(RDExperimentCreate(
        wp=WorkPackage.WP1_ROUTING,
        name=experiment_name,
        description=f"Comparação de {len(policies)} políticas de routing",
        parameters={
            "policies": policies,
            "baseline_policy": baseline_policy,
            "num_operations": len(instance.operations) if hasattr(instance, 'operations') else 0,
        },
    ))
    
    update_experiment_status(experiment.id, ExperimentStatus.RUNNING)
    
    # Executar cada política
    policy_results: Dict[str, Dict] = {}
    
    for policy in policies:
        logger.info(f"Running policy: {policy}")
        
        try:
            if policy.upper() == "MILP":
                result = solve_milp(instance, time_limit_sec=60.0)
            elif policy.upper() == "CPSAT":
                result = solve_cpsat(instance, time_limit_sec=60.0)
            else:
                # Heurística
                scheduler = HeuristicScheduler(rule=policy)
                result = scheduler.build_schedule(instance)
            
            kpis = result.get("kpis", {})
            policy_results[policy] = {
                "makespan_hours": kpis.get("makespan_hours", 0),
                "tardiness_hours": kpis.get("total_tardiness_hours", 0),
                "setup_hours": kpis.get("total_setup_time_hours", 0),
                "otd_rate": kpis.get("otd_rate", 1.0),
                "num_late_orders": kpis.get("num_late_orders", 0),
                "total_operations": kpis.get("total_operations", 0),
                "solve_time_sec": result.get("solve_time_sec", 0),
                "status": result.get("status", "unknown"),
            }
            
            logger.info(f"Policy {policy}: makespan={kpis.get('makespan_hours', 0):.2f}h, "
                       f"OTD={kpis.get('otd_rate', 1.0)*100:.1f}%")
            
        except Exception as e:
            logger.error(f"Policy {policy} failed: {e}")
            policy_results[policy] = {
                "makespan_hours": float('inf'),
                "tardiness_hours": float('inf'),
                "setup_hours": 0,
                "otd_rate": 0,
                "num_late_orders": 0,
                "total_operations": 0,
                "solve_time_sec": 0,
                "status": "error",
                "error": str(e),
            }
    
    # Calcular comparações com baseline
    baseline_result = policy_results.get(baseline_policy, {})
    baseline_makespan = baseline_result.get("makespan_hours", 0)
    baseline_tardiness = baseline_result.get("tardiness_hours", 0)
    
    results_with_comparison: List[WP1PolicyResult] = []
    best_policy = baseline_policy
    best_makespan = baseline_makespan
    
    for policy, result in policy_results.items():
        makespan = result.get("makespan_hours", float('inf'))
        tardiness = result.get("tardiness_hours", float('inf'))
        
        # Calcular % vs baseline
        vs_makespan = None
        vs_tardiness = None
        if baseline_makespan > 0:
            vs_makespan = round((makespan - baseline_makespan) / baseline_makespan * 100, 2)
        if baseline_tardiness > 0:
            vs_tardiness = round((tardiness - baseline_tardiness) / baseline_tardiness * 100, 2)
        
        results_with_comparison.append(WP1PolicyResult(
            policy=policy,
            makespan_hours=round(makespan, 2),
            tardiness_hours=round(tardiness, 2),
            setup_hours=round(result.get("setup_hours", 0), 2),
            otd_rate=round(result.get("otd_rate", 0), 3),
            num_late_orders=result.get("num_late_orders", 0),
            total_operations=result.get("total_operations", 0),
            solve_time_sec=round(result.get("solve_time_sec", 0), 3),
            vs_baseline_makespan_pct=vs_makespan,
            vs_baseline_tardiness_pct=vs_tardiness,
        ))
        
        # Track best
        if makespan < best_makespan and makespan != float('inf'):
            best_makespan = makespan
            best_policy = policy
    
    # Calcular improvement
    improvement_pct = 0.0
    if baseline_makespan > 0 and best_makespan != float('inf'):
        improvement_pct = round((baseline_makespan - best_makespan) / baseline_makespan * 100, 2)
    
    total_time = time.time() - start_time
    
    # Gerar conclusão
    if best_policy == baseline_policy:
        conclusion = f"Baseline {baseline_policy} é a melhor política testada."
    else:
        conclusion = f"Política {best_policy} reduz makespan em {improvement_pct:.1f}% vs {baseline_policy}."
    
    # Atualizar experiência com resultados
    summary = {
        "policies_tested": policies,
        "baseline_policy": baseline_policy,
        "best_policy": best_policy,
        "improvement_vs_baseline_pct": improvement_pct,
        "results_by_policy": {r.policy: r.dict() for r in results_with_comparison},
    }
    
    kpis = {
        "best_makespan_hours": best_makespan,
        "baseline_makespan_hours": baseline_makespan,
        "best_otd_rate": policy_results.get(best_policy, {}).get("otd_rate", 0),
    }
    
    update_experiment_status(
        experiment.id,
        ExperimentStatus.FINISHED,
        summary=summary,
        kpis=kpis,
        conclusion=conclusion,
    )
    
    logger.info(f"WP1 Experiment finished: {conclusion}")
    
    return experiment


def run_routing_comparison(request: WP1RoutingRequest) -> WP1RoutingExperiment:
    """
    Executa experiência de comparação de routing usando dados do sistema.
    
    Esta é a função de alto nível que:
    1. Carrega dados de produção (ou usa dados demo)
    2. Cria SchedulingInstance
    3. Chama run_routing_experiment()
    4. Formata resultado para API
    """
    from data_loader import load_dataset
    from scheduling import create_instance_from_dataframes
    
    start_time = time.time()
    
    logger.info(f"WP1 Routing Comparison: {request.name}")
    logger.info(f"Policies: {request.policies}")
    
    # Carregar dados
    try:
        data = load_dataset()
        instance = create_instance_from_dataframes(
            orders_df=data.orders,
            routing_df=data.routing,
            machines_df=data.machines,
            horizon_start=request.date_start or datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        # Criar instância de demonstração
        instance = _create_demo_instance()
    
    # Executar experiência
    experiment = run_routing_experiment(
        instance=instance,
        policies=request.policies,
        baseline_policy=request.baseline_policy,
        experiment_name=request.name,
    )
    
    # Reconstituir resultados para resposta
    # Precisamos de re-executar para obter os resultados formatados
    from scheduling import HeuristicScheduler, solve_milp, solve_cpsat
    
    results: List[WP1PolicyResult] = []
    best_makespan = float('inf')
    best_policy = request.baseline_policy
    baseline_makespan = 0.0
    
    for policy in request.policies:
        try:
            if policy.upper() == "MILP":
                result = solve_milp(instance, time_limit_sec=30.0)
            elif policy.upper() == "CPSAT":
                result = solve_cpsat(instance, time_limit_sec=30.0)
            else:
                scheduler = HeuristicScheduler(rule=policy)
                result = scheduler.build_schedule(instance)
            
            kpis = result.get("kpis", {})
            makespan = kpis.get("makespan_hours", 0)
            
            if policy == request.baseline_policy:
                baseline_makespan = makespan
            
            if makespan < best_makespan:
                best_makespan = makespan
                best_policy = policy
            
            results.append(WP1PolicyResult(
                policy=policy,
                makespan_hours=round(makespan, 2),
                tardiness_hours=round(kpis.get("total_tardiness_hours", 0), 2),
                setup_hours=round(kpis.get("total_setup_time_hours", 0), 2),
                otd_rate=round(kpis.get("otd_rate", 1.0), 3),
                num_late_orders=kpis.get("num_late_orders", 0),
                total_operations=kpis.get("total_operations", 0),
                solve_time_sec=round(result.get("solve_time_sec", 0), 3),
            ))
        except Exception as e:
            logger.error(f"Policy {policy} failed in comparison: {e}")
    
    # Calcular comparações vs baseline
    for r in results:
        if baseline_makespan > 0:
            r.vs_baseline_makespan_pct = round((r.makespan_hours - baseline_makespan) / baseline_makespan * 100, 2)
    
    improvement_pct = 0.0
    if baseline_makespan > 0:
        improvement_pct = round((baseline_makespan - best_makespan) / baseline_makespan * 100, 2)
    
    if best_policy == request.baseline_policy:
        conclusion = f"Baseline {request.baseline_policy} é a melhor política testada."
    else:
        conclusion = f"Política {best_policy} reduz makespan em {improvement_pct:.1f}% vs {request.baseline_policy}."
    
    total_time = time.time() - start_time
    
    return WP1RoutingExperiment(
        experiment_id=experiment.id,
        name=request.name,
        status="finished",
        policies_tested=request.policies,
        baseline_policy=request.baseline_policy,
        results=results,
        best_policy=best_policy,
        improvement_vs_baseline_pct=improvement_pct,
        conclusion=conclusion,
        total_time_sec=round(total_time, 2),
    )


def _create_demo_instance():
    """Cria instância de demonstração para testes."""
    from scheduling.types import SchedulingInstance
    
    # Criar operações de demonstração
    operations = []
    for i in range(20):
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


def get_routing_strategies() -> List[Dict[str, str]]:
    """Retorna lista de estratégias de routing disponíveis."""
    return [
        {"id": "FIFO", "name": "First In, First Out", "description": "Processa ordens pela ordem de chegada"},
        {"id": "SPT", "name": "Shortest Processing Time", "description": "Prioriza operações mais curtas"},
        {"id": "EDD", "name": "Earliest Due Date", "description": "Prioriza ordens com due date mais cedo"},
        {"id": "CR", "name": "Critical Ratio", "description": "Prioriza por rácio crítico"},
        {"id": "WSPT", "name": "Weighted SPT", "description": "SPT ponderado por prioridade"},
        {"id": "MILP", "name": "Otimização MILP", "description": "Otimização matemática (mais lento)"},
        {"id": "CPSAT", "name": "Constraint Programming", "description": "CP-SAT para instâncias maiores"},
    ]
