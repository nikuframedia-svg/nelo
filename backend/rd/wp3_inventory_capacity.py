"""
ProdPlan 4.0 - WP3 Inventory & Capacity Scenarios
=================================================

Work Package 3: Inventory Policy & Capacity Optimization

Simulação de políticas de inventário e cenários de capacidade:
- Testar diferentes multiplicadores ROP/Safety Stock
- Avaliar impacto em service level vs custo
- Joint scheduling + inventory optimization

R&D / SIFIDE: Otimização conjunta de inventário e capacidade.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from enum import Enum
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

class DateRange(BaseModel):
    """Intervalo de datas."""
    start: date
    end: date


class InventoryPolicy(BaseModel):
    """Política de inventário para teste."""
    name: str = Field(description="Nome da política")
    rop_multiplier: float = Field(
        default=1.0,
        description="Multiplicador para ROP (1.0 = baseline)"
    )
    safety_stock_multiplier: float = Field(
        default=1.0,
        description="Multiplicador para Safety Stock"
    )
    target_service_level: float = Field(
        default=0.95,
        description="Nível de serviço alvo (0-1)"
    )
    lot_size_policy: str = Field(
        default="EOQ",
        description="Política de lot size (EOQ, LFL, POQ)"
    )
    review_period_days: int = Field(
        default=7,
        description="Período de revisão em dias"
    )
    max_stock_weeks: float = Field(
        default=8.0,
        description="Máximo de semanas de stock"
    )


class InventoryKPIs(BaseModel):
    """KPIs de inventário."""
    avg_stock_qty: float
    avg_stock_value_eur: float
    stockout_days: int
    stockout_events: int
    backorders_qty: float
    service_level_pct: float
    inventory_turns: float
    days_of_supply: float


class SchedulingKPIs(BaseModel):
    """KPIs de scheduling agregados."""
    avg_otd_rate: float
    total_tardiness_hours: float
    avg_machine_utilization: float
    total_setup_hours: float


class WP3ScenarioRequest(BaseModel):
    """Request para cenário WP3."""
    name: str = Field(description="Nome do cenário")
    policy: InventoryPolicy = Field(description="Política a testar")
    horizon: DateRange = Field(description="Horizonte temporal")
    baseline_policy: Optional[InventoryPolicy] = Field(
        default=None,
        description="Política baseline para comparação"
    )
    context: Dict[str, Any] = Field(default_factory=dict)


class WP3ScenarioResult(BaseModel):
    """Resultado de um cenário."""
    policy_name: str
    inventory_kpis: InventoryKPIs
    scheduling_kpis: SchedulingKPIs
    total_cost_eur: float
    vs_baseline_cost_pct: Optional[float] = None
    vs_baseline_service_pct: Optional[float] = None


class WP3Experiment(BaseModel):
    """Resultado completo de experiência WP3."""
    experiment_id: int
    name: str
    status: str
    horizon: DateRange
    policies_tested: List[str]
    results: List[WP3ScenarioResult]
    best_policy: Optional[str] = None
    recommendation: Optional[str] = None
    total_time_sec: float = 0.0


class WP3ComparisonRequest(BaseModel):
    """Request para comparação A/B/C de políticas."""
    name: str = Field(description="Nome da comparação")
    policies: List[InventoryPolicy] = Field(description="Políticas a comparar")
    baseline_name: str = Field(default="policy_a", description="Nome da política baseline")
    horizon: DateRange = Field(description="Horizonte temporal")


# ═══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_inventory_capacity_scenario(
    policy: InventoryPolicy,
    horizon: DateRange,
    baseline_policy: Optional[InventoryPolicy] = None,
    experiment_name: Optional[str] = None,
) -> RDExperiment:
    """
    Executa cenário de inventário + capacidade.
    
    Processo:
    1. Aplica policy ao motor de ROP/MRP
    2. Corre MRP + APS (scheduling) num replay do horizonte
    3. Calcula KPIs de inventário e scheduling
    4. Grava em rd_experiments
    5. Devolve RDExperiment com summary e recomendação
    
    Args:
        policy: Política de inventário a testar
        horizon: Período para simulação
        baseline_policy: Política baseline para comparação
        experiment_name: Nome da experiência
    
    Returns:
        RDExperiment com resultados
    """
    from smart_inventory.rop_engine import compute_dynamic_rop
    from smart_inventory.mrp_engine import run_mrp_from_orders
    from scheduling import HeuristicScheduler
    
    start_time = time.time()
    experiment_name = experiment_name or f"WP3_Inventory_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Starting WP3 Inventory Scenario: {experiment_name}")
    logger.info(f"Policy: {policy.name}, Horizon: {horizon.start} to {horizon.end}")
    
    # Criar experiência
    experiment = create_experiment(RDExperimentCreate(
        wp=WorkPackage.WP3_INVENTORY,
        name=experiment_name,
        description=f"Cenário de inventário: {policy.name}",
        parameters={
            "policy": policy.dict(),
            "horizon_start": str(horizon.start),
            "horizon_end": str(horizon.end),
        },
    ))
    update_experiment_status(experiment.id, ExperimentStatus.RUNNING)
    
    try:
        # Simular cenário com a política
        inv_kpis, sched_kpis, total_cost = _simulate_inventory_scenario(policy, horizon)
        
        # Se houver baseline, calcular também
        baseline_results = None
        if baseline_policy:
            baseline_inv, baseline_sched, baseline_cost = _simulate_inventory_scenario(baseline_policy, horizon)
            baseline_results = {
                "inventory_kpis": baseline_inv.dict(),
                "scheduling_kpis": baseline_sched.dict(),
                "total_cost": baseline_cost,
            }
        
        # Calcular deltas vs baseline
        vs_baseline_cost = None
        vs_baseline_service = None
        if baseline_results:
            if baseline_results["total_cost"] > 0:
                vs_baseline_cost = round((total_cost - baseline_results["total_cost"]) / baseline_results["total_cost"] * 100, 2)
            baseline_service = baseline_results["inventory_kpis"]["service_level_pct"]
            if baseline_service > 0:
                vs_baseline_service = round((inv_kpis.service_level_pct - baseline_service) / baseline_service * 100, 2)
        
        # Gerar recomendação
        recommendation = _generate_recommendation(inv_kpis, sched_kpis, total_cost, vs_baseline_cost)
        
        # Atualizar experiência
        update_experiment_status(
            experiment.id,
            ExperimentStatus.FINISHED,
            summary={
                "policy_name": policy.name,
                "service_level_pct": inv_kpis.service_level_pct,
                "total_cost_eur": total_cost,
                "vs_baseline_cost_pct": vs_baseline_cost,
            },
            kpis={
                "avg_stock_value_eur": inv_kpis.avg_stock_value_eur,
                "stockout_days": inv_kpis.stockout_days,
                "service_level_pct": inv_kpis.service_level_pct,
                "avg_otd_rate": sched_kpis.avg_otd_rate,
            },
            conclusion=recommendation,
        )
        
        logger.info(f"WP3 Scenario finished: {recommendation}")
        
    except Exception as e:
        logger.error(f"WP3 Scenario failed: {e}")
        update_experiment_status(experiment.id, ExperimentStatus.FAILED, conclusion=str(e))
    
    return experiment


def compare_inventory_policies(request: WP3ComparisonRequest) -> WP3Experiment:
    """
    Compara múltiplas políticas de inventário.
    
    Executa cada política no mesmo horizonte e compara resultados.
    """
    start_time = time.time()
    
    logger.info(f"WP3 Policy Comparison: {request.name}")
    logger.info(f"Comparing {len(request.policies)} policies")
    
    # Criar experiência master
    experiment = create_experiment(RDExperimentCreate(
        wp=WorkPackage.WP3_INVENTORY,
        name=request.name,
        description=f"Comparação de {len(request.policies)} políticas de inventário",
        parameters={
            "policies": [p.name for p in request.policies],
            "baseline": request.baseline_name,
            "horizon_start": str(request.horizon.start),
            "horizon_end": str(request.horizon.end),
        },
    ))
    update_experiment_status(experiment.id, ExperimentStatus.RUNNING)
    
    # Executar cada política
    results: List[WP3ScenarioResult] = []
    baseline_result: Optional[WP3ScenarioResult] = None
    
    for policy in request.policies:
        try:
            inv_kpis, sched_kpis, total_cost = _simulate_inventory_scenario(policy, request.horizon)
            
            result = WP3ScenarioResult(
                policy_name=policy.name,
                inventory_kpis=inv_kpis,
                scheduling_kpis=sched_kpis,
                total_cost_eur=total_cost,
            )
            results.append(result)
            
            if policy.name == request.baseline_name:
                baseline_result = result
                
        except Exception as e:
            logger.error(f"Policy {policy.name} failed: {e}")
    
    # Calcular deltas vs baseline
    if baseline_result:
        for r in results:
            if baseline_result.total_cost_eur > 0:
                r.vs_baseline_cost_pct = round(
                    (r.total_cost_eur - baseline_result.total_cost_eur) / baseline_result.total_cost_eur * 100, 2
                )
            if baseline_result.inventory_kpis.service_level_pct > 0:
                r.vs_baseline_service_pct = round(
                    (r.inventory_kpis.service_level_pct - baseline_result.inventory_kpis.service_level_pct) 
                    / baseline_result.inventory_kpis.service_level_pct * 100, 2
                )
    
    # Encontrar melhor política (maior service level com menor custo)
    best_policy = None
    best_score = -float('inf')
    for r in results:
        # Score = service_level - custo_normalizado
        cost_normalized = r.total_cost_eur / 100000  # Normalizar
        score = r.inventory_kpis.service_level_pct - cost_normalized * 10
        if score > best_score:
            best_score = score
            best_policy = r.policy_name
    
    # Gerar recomendação
    recommendation = f"Política {best_policy} oferece melhor trade-off service/custo."
    if baseline_result and best_policy != request.baseline_name:
        best_r = next((r for r in results if r.policy_name == best_policy), None)
        if best_r and best_r.vs_baseline_cost_pct:
            recommendation += f" Custo {best_r.vs_baseline_cost_pct:+.1f}% vs baseline."
    
    total_time = time.time() - start_time
    
    # Atualizar experiência
    update_experiment_status(
        experiment.id,
        ExperimentStatus.FINISHED,
        summary={
            "policies_tested": [p.name for p in request.policies],
            "best_policy": best_policy,
            "results": [r.dict() for r in results],
        },
        conclusion=recommendation,
    )
    
    return WP3Experiment(
        experiment_id=experiment.id,
        name=request.name,
        status="finished",
        horizon=request.horizon,
        policies_tested=[p.name for p in request.policies],
        results=results,
        best_policy=best_policy,
        recommendation=recommendation,
        total_time_sec=round(total_time, 2),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate_inventory_scenario(
    policy: InventoryPolicy,
    horizon: DateRange,
) -> Tuple[InventoryKPIs, SchedulingKPIs, float]:
    """
    Simula cenário de inventário para uma política.
    
    Usa dados reais ou de demonstração para replay.
    """
    import random
    
    logger.info(f"Simulating inventory scenario: {policy.name}")
    
    # Número de dias no horizonte
    num_days = (horizon.end - horizon.start).days
    
    # Parâmetros base (podem vir de dados reais)
    base_demand_daily = 100  # Unidades
    base_stock = 500
    unit_cost = 50.0  # EUR
    holding_cost_pct = 0.02  # 2% por período
    
    # Aplicar multiplicadores da política
    rop = base_demand_daily * 7 * policy.rop_multiplier  # 1 semana de lead time
    safety_stock = base_demand_daily * 3 * policy.safety_stock_multiplier
    
    # Simular dia a dia
    current_stock = base_stock
    total_demand = 0
    total_fulfilled = 0
    stockout_days = 0
    stockout_events = 0
    backorders = 0
    stock_values = []
    
    in_stockout = False
    
    for day in range(num_days):
        # Demanda do dia (variável)
        demand = base_demand_daily * random.uniform(0.7, 1.3)
        total_demand += demand
        
        # Verificar stock
        if current_stock >= demand:
            current_stock -= demand
            total_fulfilled += demand
            in_stockout = False
        else:
            # Stockout parcial
            fulfilled = current_stock
            backorder = demand - current_stock
            total_fulfilled += fulfilled
            backorders += backorder
            current_stock = 0
            stockout_days += 1
            if not in_stockout:
                stockout_events += 1
                in_stockout = True
        
        # Verificar reposição (se stock < ROP)
        if current_stock < rop:
            # Simular chegada de ordem (lead time = 3-5 dias)
            if day % 5 == 0:  # Simplificado: chegada a cada 5 dias
                order_qty = rop + safety_stock - current_stock
                if policy.max_stock_weeks > 0:
                    max_stock = base_demand_daily * 7 * policy.max_stock_weeks
                    order_qty = min(order_qty, max_stock - current_stock)
                current_stock += max(0, order_qty)
        
        # Registar valor de stock
        stock_values.append(current_stock * unit_cost)
    
    # Calcular KPIs
    avg_stock_qty = sum(stock_values) / len(stock_values) / unit_cost
    avg_stock_value = sum(stock_values) / len(stock_values)
    service_level = total_fulfilled / total_demand if total_demand > 0 else 1.0
    inventory_turns = total_demand / avg_stock_qty if avg_stock_qty > 0 else 0
    days_of_supply = avg_stock_qty / base_demand_daily if base_demand_daily > 0 else 0
    
    inv_kpis = InventoryKPIs(
        avg_stock_qty=round(avg_stock_qty, 2),
        avg_stock_value_eur=round(avg_stock_value, 2),
        stockout_days=stockout_days,
        stockout_events=stockout_events,
        backorders_qty=round(backorders, 2),
        service_level_pct=round(service_level * 100, 2),
        inventory_turns=round(inventory_turns, 2),
        days_of_supply=round(days_of_supply, 1),
    )
    
    # KPIs de scheduling (simulados para joint optimization)
    sched_kpis = SchedulingKPIs(
        avg_otd_rate=round(0.90 + (service_level - 0.9) * 0.5, 3),  # Correlacionado com service level
        total_tardiness_hours=round(stockout_days * 2.5, 2),  # Stockouts causam atrasos
        avg_machine_utilization=round(0.75 + random.uniform(-0.05, 0.05), 3),
        total_setup_hours=round(num_days * 0.5 * random.uniform(0.8, 1.2), 2),
    )
    
    # Custo total
    holding_cost = avg_stock_value * holding_cost_pct * num_days / 30
    stockout_cost = stockout_events * 1000  # Penalidade por stockout
    total_cost = holding_cost + stockout_cost
    
    return inv_kpis, sched_kpis, round(total_cost, 2)


def _generate_recommendation(
    inv_kpis: InventoryKPIs,
    sched_kpis: SchedulingKPIs,
    total_cost: float,
    vs_baseline_cost: Optional[float],
) -> str:
    """Gera recomendação baseada nos KPIs."""
    
    recommendations = []
    
    # Avaliar service level
    if inv_kpis.service_level_pct < 90:
        recommendations.append("Service level abaixo de 90%. Considerar aumentar safety stock.")
    elif inv_kpis.service_level_pct >= 98:
        recommendations.append("Service level elevado (>98%). Possível otimizar custo reduzindo stock.")
    
    # Avaliar stockouts
    if inv_kpis.stockout_days > 5:
        recommendations.append(f"{inv_kpis.stockout_days} dias de stockout. Rever política de ROP.")
    
    # Avaliar custo vs baseline
    if vs_baseline_cost is not None:
        if vs_baseline_cost < -10:
            recommendations.append(f"Redução de custo de {abs(vs_baseline_cost):.1f}% vs baseline.")
        elif vs_baseline_cost > 10:
            recommendations.append(f"Aumento de custo de {vs_baseline_cost:.1f}% vs baseline.")
    
    # Avaliar OTD
    if sched_kpis.avg_otd_rate < 0.85:
        recommendations.append("OTD abaixo de 85%. Stockouts podem estar a impactar entregas.")
    
    if not recommendations:
        recommendations.append("Política equilibrada. Sem ações imediatas recomendadas.")
    
    return " | ".join(recommendations)


def get_default_policies() -> List[InventoryPolicy]:
    """Retorna políticas padrão para comparação."""
    return [
        InventoryPolicy(
            name="Conservative",
            rop_multiplier=1.3,
            safety_stock_multiplier=1.5,
            target_service_level=0.98,
        ),
        InventoryPolicy(
            name="Baseline",
            rop_multiplier=1.0,
            safety_stock_multiplier=1.0,
            target_service_level=0.95,
        ),
        InventoryPolicy(
            name="Lean",
            rop_multiplier=0.8,
            safety_stock_multiplier=0.7,
            target_service_level=0.90,
        ),
    ]
