"""
════════════════════════════════════════════════════════════════════════════════════════════════════
RECOVERY STRATEGY ENGINE - Motor de Estratégias de Recuperação ZDM
════════════════════════════════════════════════════════════════════════════════════════════════════

Define e avalia políticas de recuperação para diferentes tipos de falha.

Estratégias Disponíveis:
- LOCAL_REPLAN: Replaneamento local (reordenar operações)
- VIP_PRIORITY: Priorizar encomendas VIP
- ADD_SHIFT: Adicionar turno extra
- CUT_LOWPRIORITY: Cortar produtos não prioritários
- REROUTE: Reencaminhar para máquinas alternativas
- PARTIAL_BATCH: Dividir lote em sub-lotes
- OUTSOURCE: Subcontratar externamente
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np

from .failure_scenario_generator import FailureScenario, FailureType


class RecoveryStrategy(Enum):
    """Estratégias de recuperação disponíveis."""
    LOCAL_REPLAN = "local_replan"         # Replaneamento local
    VIP_PRIORITY = "vip_priority"         # Priorizar VIP
    ADD_SHIFT = "add_shift"               # Adicionar turno
    CUT_LOWPRIORITY = "cut_lowpriority"   # Cortar baixa prioridade
    REROUTE = "reroute"                   # Reencaminhar
    PARTIAL_BATCH = "partial_batch"       # Dividir lote
    OUTSOURCE = "outsource"               # Subcontratar
    MAINTENANCE_URGENT = "maintenance_urgent"  # Manutenção urgente
    DO_NOTHING = "do_nothing"             # Não fazer nada (absorver)


@dataclass
class RecoveryAction:
    """
    Uma ação de recuperação específica.
    """
    strategy: RecoveryStrategy
    target_machine: Optional[str] = None
    target_orders: List[str] = field(default_factory=list)
    
    # Parâmetros específicos
    shift_hours: float = 0.0              # Para ADD_SHIFT
    alternative_machine: Optional[str] = None  # Para REROUTE
    outsource_vendor: Optional[str] = None     # Para OUTSOURCE
    
    # Impacto estimado
    recovery_time_hours: float = 0.0
    cost_eur: float = 0.0
    effectiveness_pct: float = 0.0        # 0-100
    
    # Descrição
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Converte para dicionário."""
        return {
            "strategy": self.strategy.value,
            "target_machine": self.target_machine,
            "target_orders": self.target_orders[:5],  # Limitar
            "shift_hours": round(self.shift_hours, 1),
            "alternative_machine": self.alternative_machine,
            "outsource_vendor": self.outsource_vendor,
            "recovery_time_hours": round(self.recovery_time_hours, 1),
            "cost_eur": round(self.cost_eur, 0),
            "effectiveness_pct": round(self.effectiveness_pct, 1),
            "description": self.description,
        }


@dataclass
class RecoveryPlan:
    """
    Plano de recuperação completo para um cenário de falha.
    """
    scenario_id: str
    primary_action: RecoveryAction
    secondary_actions: List[RecoveryAction] = field(default_factory=list)
    
    # Métricas agregadas
    total_cost_eur: float = 0.0
    total_recovery_time_hours: float = 0.0
    expected_effectiveness_pct: float = 0.0
    
    # Status
    feasible: bool = True
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH
    confidence: float = 0.0
    
    # Explicação
    rationale: str = ""
    warnings: List[str] = field(default_factory=list)
    
    def compute_totals(self):
        """Calcula totais do plano."""
        all_actions = [self.primary_action] + self.secondary_actions
        
        self.total_cost_eur = sum(a.cost_eur for a in all_actions)
        self.total_recovery_time_hours = max(a.recovery_time_hours for a in all_actions)
        
        # Effectiveness é combinação (não simplesmente soma)
        base_eff = self.primary_action.effectiveness_pct / 100
        for action in self.secondary_actions:
            # Cada ação adicional melhora o que resta
            remaining = 1 - base_eff
            base_eff += remaining * (action.effectiveness_pct / 100) * 0.5
        
        self.expected_effectiveness_pct = min(100, base_eff * 100)
        self.confidence = min(0.95, 0.5 + len(all_actions) * 0.1)
    
    def to_dict(self) -> Dict:
        """Converte para dicionário."""
        return {
            "scenario_id": self.scenario_id,
            "primary_action": self.primary_action.to_dict(),
            "secondary_actions": [a.to_dict() for a in self.secondary_actions],
            "total_cost_eur": round(self.total_cost_eur, 0),
            "total_recovery_time_hours": round(self.total_recovery_time_hours, 1),
            "expected_effectiveness_pct": round(self.expected_effectiveness_pct, 1),
            "feasible": self.feasible,
            "risk_level": self.risk_level,
            "confidence": round(self.confidence, 2),
            "rationale": self.rationale,
            "warnings": self.warnings,
        }


@dataclass
class RecoveryConfig:
    """
    Configuração do motor de recuperação.
    """
    # Custos por estratégia (€/hora ou €/ação)
    cost_shift_per_hour: float = 50.0
    cost_outsource_per_unit: float = 100.0
    cost_maintenance_urgent: float = 500.0
    cost_reroute_setup: float = 100.0
    
    # Efetividade base por estratégia
    effectiveness_local_replan: float = 60.0
    effectiveness_vip_priority: float = 40.0
    effectiveness_add_shift: float = 70.0
    effectiveness_cut_lowpriority: float = 50.0
    effectiveness_reroute: float = 80.0
    effectiveness_partial_batch: float = 55.0
    effectiveness_outsource: float = 75.0
    
    # Limites
    max_shift_hours: float = 8.0
    max_outsource_pct: float = 30.0
    
    # Prioridades VIP (orders com priority >= threshold são VIP)
    vip_priority_threshold: float = 8.0
    
    # Máquinas alternativas conhecidas
    alternative_machines: Dict[str, List[str]] = field(default_factory=dict)


def evaluate_strategy_fit(
    strategy: RecoveryStrategy,
    scenario: FailureScenario,
    plan_df: pd.DataFrame,
    config: RecoveryConfig,
) -> Tuple[float, str]:
    """
    Avalia quão adequada é uma estratégia para um cenário.
    
    Returns:
        (score 0-100, razão)
    """
    score = 50.0  # Base
    reason = ""
    
    # Avaliar por tipo de falha
    if scenario.failure_type == FailureType.SUDDEN:
        if strategy == RecoveryStrategy.REROUTE:
            score = 90.0
            reason = "Reencaminhamento ideal para falha súbita"
        elif strategy == RecoveryStrategy.ADD_SHIFT:
            score = 75.0
            reason = "Turno extra pode compensar paragem"
        elif strategy == RecoveryStrategy.MAINTENANCE_URGENT:
            score = 70.0
            reason = "Manutenção urgente reduz downtime"
        elif strategy == RecoveryStrategy.DO_NOTHING:
            score = 20.0
            reason = "Absorver falha súbita tem alto impacto"
    
    elif scenario.failure_type == FailureType.GRADUAL:
        if strategy == RecoveryStrategy.LOCAL_REPLAN:
            score = 85.0
            reason = "Replaneamento ajusta para degradação"
        elif strategy == RecoveryStrategy.MAINTENANCE_URGENT:
            score = 80.0
            reason = "Manutenção previne piora"
        elif strategy == RecoveryStrategy.ADD_SHIFT:
            score = 70.0
            reason = "Horas extra compensam ciclos mais longos"
    
    elif scenario.failure_type == FailureType.QUALITY:
        if strategy == RecoveryStrategy.PARTIAL_BATCH:
            score = 80.0
            reason = "Lotes menores isolam defeitos"
        elif strategy == RecoveryStrategy.CUT_LOWPRIORITY:
            score = 65.0
            reason = "Cortar baixa prioridade libera capacidade para retrabalho"
        elif strategy == RecoveryStrategy.MAINTENANCE_URGENT:
            score = 75.0
            reason = "Corrigir causa raiz do defeito"
    
    elif scenario.failure_type == FailureType.MATERIAL:
        if strategy == RecoveryStrategy.OUTSOURCE:
            score = 85.0
            reason = "Subcontratação contorna falta de material"
        elif strategy == RecoveryStrategy.VIP_PRIORITY:
            score = 70.0
            reason = "Priorizar VIP com material disponível"
        elif strategy == RecoveryStrategy.LOCAL_REPLAN:
            score = 75.0
            reason = "Reordenar para usar materiais disponíveis"
    
    elif scenario.failure_type == FailureType.OPERATOR:
        if strategy == RecoveryStrategy.ADD_SHIFT:
            score = 80.0
            reason = "Turno extra traz operadores alternativos"
        elif strategy == RecoveryStrategy.REROUTE:
            score = 70.0
            reason = "Redirecionar para células com operadores"
    
    # Ajustar por severidade
    if scenario.severity > 0.8:
        if strategy in [RecoveryStrategy.REROUTE, RecoveryStrategy.OUTSOURCE]:
            score *= 1.1  # Boost para estratégias mais robustas
        elif strategy == RecoveryStrategy.DO_NOTHING:
            score *= 0.5  # Penalizar inação em cenários graves
    
    # Ajustar por duração
    if scenario.duration_hours > 24:
        if strategy == RecoveryStrategy.ADD_SHIFT:
            score *= 0.8  # Turnos extra não são sustentáveis a longo prazo
        if strategy == RecoveryStrategy.OUTSOURCE:
            score *= 1.2  # Outsource mais viável para períodos longos
    
    return min(100, max(0, score)), reason


def create_recovery_action(
    strategy: RecoveryStrategy,
    scenario: FailureScenario,
    plan_df: pd.DataFrame,
    config: RecoveryConfig,
) -> RecoveryAction:
    """
    Cria uma ação de recuperação concreta.
    """
    action = RecoveryAction(
        strategy=strategy,
        target_machine=scenario.machine_id,
    )
    
    # Configurar parâmetros por estratégia
    if strategy == RecoveryStrategy.LOCAL_REPLAN:
        action.recovery_time_hours = scenario.duration_hours * 0.3
        action.cost_eur = 50.0  # Custo de replaneamento
        action.effectiveness_pct = config.effectiveness_local_replan
        action.description = f"Reordenar operações após {scenario.machine_id} para minimizar atrasos"
    
    elif strategy == RecoveryStrategy.VIP_PRIORITY:
        action.recovery_time_hours = 1.0
        action.cost_eur = 0.0
        action.effectiveness_pct = config.effectiveness_vip_priority
        action.description = "Priorizar encomendas VIP, atrasar outras"
    
    elif strategy == RecoveryStrategy.ADD_SHIFT:
        shift_hours = min(config.max_shift_hours, scenario.duration_hours * 0.5)
        action.shift_hours = shift_hours
        action.recovery_time_hours = scenario.duration_hours * 0.5
        action.cost_eur = shift_hours * config.cost_shift_per_hour
        action.effectiveness_pct = config.effectiveness_add_shift
        action.description = f"Adicionar {shift_hours:.1f}h extra para compensar"
    
    elif strategy == RecoveryStrategy.CUT_LOWPRIORITY:
        action.recovery_time_hours = 0.5
        action.cost_eur = 0.0  # Custo de oportunidade não contabilizado aqui
        action.effectiveness_pct = config.effectiveness_cut_lowpriority
        action.description = "Cortar ou adiar ordens de baixa prioridade"
    
    elif strategy == RecoveryStrategy.REROUTE:
        alternatives = config.alternative_machines.get(scenario.machine_id, [])
        if alternatives:
            action.alternative_machine = alternatives[0]
            action.description = f"Reencaminhar operações para {alternatives[0]}"
        else:
            # Inferir alternativa
            prefix = scenario.machine_id.split('-')[0] if '-' in scenario.machine_id else scenario.machine_id[0]
            action.alternative_machine = f"{prefix}-ALT"
            action.description = f"Reencaminhar para máquina alternativa do grupo {prefix}"
        
        action.recovery_time_hours = 2.0  # Setup + transferência
        action.cost_eur = config.cost_reroute_setup
        action.effectiveness_pct = config.effectiveness_reroute
    
    elif strategy == RecoveryStrategy.PARTIAL_BATCH:
        action.recovery_time_hours = scenario.duration_hours * 0.4
        action.cost_eur = 50.0
        action.effectiveness_pct = config.effectiveness_partial_batch
        action.description = "Dividir lote para processar parcialmente"
    
    elif strategy == RecoveryStrategy.OUTSOURCE:
        action.outsource_vendor = "Fornecedor Externo"
        action.recovery_time_hours = scenario.duration_hours * 0.3
        action.cost_eur = config.cost_outsource_per_unit * 10  # Estimativa
        action.effectiveness_pct = config.effectiveness_outsource
        action.description = "Subcontratar produção externamente"
    
    elif strategy == RecoveryStrategy.MAINTENANCE_URGENT:
        action.recovery_time_hours = scenario.duration_hours * 0.6
        action.cost_eur = config.cost_maintenance_urgent
        action.effectiveness_pct = 70.0
        action.description = f"Manutenção urgente em {scenario.machine_id}"
    
    else:  # DO_NOTHING
        action.recovery_time_hours = scenario.duration_hours
        action.cost_eur = 0.0
        action.effectiveness_pct = 10.0
        action.description = "Absorver impacto sem ação corretiva"
    
    return action


def suggest_best_recovery(
    plan_df: pd.DataFrame,
    scenario: FailureScenario,
    config: Optional[RecoveryConfig] = None,
) -> RecoveryPlan:
    """
    Sugere o melhor plano de recuperação para um cenário de falha.
    
    Args:
        plan_df: DataFrame do plano de produção
        scenario: Cenário de falha
        config: Configuração (usa default se None)
    
    Returns:
        RecoveryPlan com ações recomendadas
    """
    if config is None:
        config = RecoveryConfig()
    
    # Avaliar todas as estratégias
    strategy_scores = []
    for strategy in RecoveryStrategy:
        score, reason = evaluate_strategy_fit(strategy, scenario, plan_df, config)
        strategy_scores.append((strategy, score, reason))
    
    # Ordenar por score
    strategy_scores.sort(key=lambda x: -x[1])
    
    # Selecionar melhor estratégia como primária
    best_strategy, best_score, best_reason = strategy_scores[0]
    primary_action = create_recovery_action(best_strategy, scenario, plan_df, config)
    
    # Adicionar estratégias secundárias complementares
    secondary_actions = []
    used_strategies = {best_strategy}
    
    for strategy, score, reason in strategy_scores[1:4]:  # Top 3 secundárias
        if score > 40 and strategy not in used_strategies:
            # Verificar se é complementar (não conflitante)
            if not _strategies_conflict(best_strategy, strategy):
                action = create_recovery_action(strategy, scenario, plan_df, config)
                secondary_actions.append(action)
                used_strategies.add(strategy)
    
    # Criar plano
    plan = RecoveryPlan(
        scenario_id=scenario.scenario_id,
        primary_action=primary_action,
        secondary_actions=secondary_actions,
    )
    
    # Calcular totais
    plan.compute_totals()
    
    # Determinar risco
    if plan.expected_effectiveness_pct >= 70:
        plan.risk_level = "LOW"
    elif plan.expected_effectiveness_pct >= 50:
        plan.risk_level = "MEDIUM"
    else:
        plan.risk_level = "HIGH"
    
    # Gerar rationale
    plan.rationale = (
        f"Para a falha tipo '{scenario.failure_type.value}' em {scenario.machine_id}, "
        f"recomendamos '{best_strategy.value}' ({best_reason}). "
        f"Efetividade esperada: {plan.expected_effectiveness_pct:.0f}%."
    )
    
    # Warnings
    if scenario.severity > 0.8:
        plan.warnings.append("Cenário de alta severidade - monitorizar de perto")
    if plan.total_cost_eur > 1000:
        plan.warnings.append(f"Custo de recuperação elevado: €{plan.total_cost_eur:.0f}")
    if scenario.triggered_by_rul:
        plan.warnings.append("Falha associada a RUL baixo - considerar substituição preventiva")
    
    return plan


def _strategies_conflict(s1: RecoveryStrategy, s2: RecoveryStrategy) -> bool:
    """Verifica se duas estratégias são conflitantes."""
    conflicts = {
        (RecoveryStrategy.DO_NOTHING, RecoveryStrategy.ADD_SHIFT),
        (RecoveryStrategy.DO_NOTHING, RecoveryStrategy.REROUTE),
        (RecoveryStrategy.DO_NOTHING, RecoveryStrategy.OUTSOURCE),
        (RecoveryStrategy.CUT_LOWPRIORITY, RecoveryStrategy.VIP_PRIORITY),  # Redundante
    }
    
    return (s1, s2) in conflicts or (s2, s1) in conflicts


def apply_recovery_strategy(
    plan_df: pd.DataFrame,
    recovery_plan: RecoveryPlan,
    scenario: FailureScenario,
) -> pd.DataFrame:
    """
    Aplica um plano de recuperação ao DataFrame do plano.
    
    TODO[R&D]: Implementar ajustes reais no plano:
    - Recalcular datas de operações
    - Atualizar máquinas atribuídas
    - Ajustar prioridades
    
    Args:
        plan_df: DataFrame do plano original
        recovery_plan: Plano de recuperação a aplicar
        scenario: Cenário de falha
    
    Returns:
        DataFrame ajustado
    """
    adjusted = plan_df.copy()
    
    # Por agora, apenas marca as operações afetadas
    if 'machine_id' in adjusted.columns:
        affected_mask = adjusted['machine_id'] == scenario.machine_id
        
        if recovery_plan.primary_action.strategy == RecoveryStrategy.REROUTE:
            alt_machine = recovery_plan.primary_action.alternative_machine
            if alt_machine:
                # Simular reencaminhamento
                adjusted.loc[affected_mask, 'machine_id'] = alt_machine
                adjusted.loc[affected_mask, 'recovery_applied'] = True
    
    return adjusted


def get_recovery_recommendations(
    plan_df: pd.DataFrame,
    scenarios: List[FailureScenario],
    config: Optional[RecoveryConfig] = None,
) -> List[RecoveryPlan]:
    """
    Gera recomendações de recuperação para múltiplos cenários.
    
    Args:
        plan_df: DataFrame do plano
        scenarios: Lista de cenários de falha
        config: Configuração
    
    Returns:
        Lista de RecoveryPlan ordenada por prioridade
    """
    if config is None:
        config = RecoveryConfig()
    
    plans = []
    for scenario in scenarios:
        plan = suggest_best_recovery(plan_df, scenario, config)
        plans.append(plan)
    
    # Ordenar por: risco alto primeiro, depois por effectiveness
    risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    plans.sort(key=lambda p: (risk_order.get(p.risk_level, 3), -p.expected_effectiveness_pct))
    
    return plans


