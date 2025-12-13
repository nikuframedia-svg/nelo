"""
════════════════════════════════════════════════════════════════════════════════════════════════════
RUL INTEGRATION SCHEDULER - PdM-Integrated Production Scheduling (PdM-IPS)
════════════════════════════════════════════════════════════════════════════════════════════════════

Integração do RUL com o APS (Advanced Planning and Scheduling).

O scheduling passa a ser **PdM-integrated**: o plano "sabe" que máquina está perto de falhar
e adapta-se preventivamente.

Funcionalidades:
1. Penalizar uso de máquinas com baixo HI para operações críticas
2. Antecipar ordens de manutenção
3. Evitar colocar operações longas perto do fim de vida
4. Redistribuir carga para máquinas saudáveis

Integração:
- Adicionar penalização no custo para usar máquina com baixa HI
- Restrições de indisponibilidade parcial
- Decisões documentadas (não agendar OP crítica em máquina com RUL baixo)

TODO[R&D]:
- Otimização conjunta produção + manutenção (MILP integrado)
- Stochastic scheduling com incerteza de RUL
- Reinforcement Learning para política adaptativa
- Multi-objective: makespan vs reliability vs maintenance cost
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd

from digital_twin.rul_estimator import RULEstimate, HealthStatus, RULEstimatorConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RULAdjustmentConfig:
    """Configuração para ajuste do plano baseado em RUL."""
    
    # Thresholds de RUL (em horas)
    rul_threshold_critical: float = 24.0  # < 24h = evitar completamente
    rul_threshold_warning: float = 72.0  # < 72h = evitar operações longas
    rul_threshold_caution: float = 168.0  # < 1 semana = monitorar
    
    # Penalizações (multiplicadores de custo)
    penalty_critical: float = 100.0  # Máquina crítica - penalização muito alta
    penalty_warning: float = 5.0  # Máquina em warning
    penalty_caution: float = 1.5  # Máquina em caution
    penalty_healthy: float = 1.0  # Máquina saudável (sem penalização)
    
    # Operações críticas
    critical_op_duration_threshold_min: float = 60.0  # Ops > 1h são críticas
    critical_op_penalty_multiplier: float = 2.0  # Penalização extra para ops críticas
    
    # Redistribuição
    enable_load_redistribution: bool = True
    max_load_increase_percent: float = 20.0  # Máximo aumento de carga em máquinas saudáveis
    
    # Manutenção preventiva
    schedule_maintenance_for_critical: bool = True
    maintenance_window_hours: float = 4.0  # Duração da manutenção preventiva
    
    # Logging
    log_decisions: bool = True


@dataclass
class MachineRULInfo:
    """Informação de RUL para uma máquina."""
    machine_id: str
    rul_estimate: Optional[RULEstimate]
    
    # Derived
    @property
    def rul_hours(self) -> float:
        """RUL em horas (ou infinito se não disponível)."""
        if self.rul_estimate is None:
            return float('inf')
        return self.rul_estimate.rul_mean_hours
    
    @property
    def health_status(self) -> HealthStatus:
        """Estado de saúde."""
        if self.rul_estimate is None:
            return HealthStatus.HEALTHY
        return self.rul_estimate.health_status
    
    @property
    def current_hi(self) -> float:
        """HI atual."""
        if self.rul_estimate is None:
            return 1.0
        return self.rul_estimate.current_hi
    
    @property
    def is_safe_for_production(self) -> bool:
        """Verifica se a máquina é segura para produção."""
        return self.health_status not in (HealthStatus.CRITICAL,)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            "machine_id": self.machine_id,
            "rul_hours": round(self.rul_hours, 1) if self.rul_hours != float('inf') else None,
            "health_status": self.health_status.value,
            "current_hi": round(self.current_hi, 4),
            "is_safe_for_production": self.is_safe_for_production,
        }


@dataclass
class PlanAdjustmentDecision:
    """Decisão de ajuste do plano."""
    decision_type: str  # "AVOID", "REDISTRIBUTE", "MAINTENANCE", "PENALTY"
    machine_id: str
    operation_id: Optional[str]
    reason: str
    original_value: Optional[Any]
    adjusted_value: Optional[Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_type": self.decision_type,
            "machine_id": self.machine_id,
            "operation_id": self.operation_id,
            "reason": self.reason,
            "original_value": str(self.original_value),
            "adjusted_value": str(self.adjusted_value),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PlanAdjustmentResult:
    """Resultado do ajuste do plano."""
    original_plan_df: pd.DataFrame
    adjusted_plan_df: pd.DataFrame
    decisions: List[PlanAdjustmentDecision]
    machine_rul_info: Dict[str, MachineRULInfo]
    
    # Métricas
    operations_redistributed: int = 0
    operations_avoided: int = 0
    maintenance_scheduled: int = 0
    total_penalty_applied: float = 0.0
    
    def summary(self) -> Dict[str, Any]:
        """Resumo do ajuste."""
        return {
            "operations_redistributed": self.operations_redistributed,
            "operations_avoided": self.operations_avoided,
            "maintenance_scheduled": self.maintenance_scheduled,
            "total_penalty_applied": round(self.total_penalty_applied, 2),
            "num_decisions": len(self.decisions),
            "machines_at_risk": sum(
                1 for m in self.machine_rul_info.values()
                if m.health_status in (HealthStatus.CRITICAL, HealthStatus.WARNING)
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ADJUSTMENT FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def adjust_plan_with_rul(
    plan_df: pd.DataFrame,
    rul_info: Dict[str, RULEstimate],
    config: Optional[RULAdjustmentConfig] = None,
) -> PlanAdjustmentResult:
    """
    Ajustar plano de produção baseado em informação de RUL.
    
    Args:
        plan_df: DataFrame com o plano de produção
                 Colunas esperadas: order_id, machine_id, start_time, end_time, duration_min
        rul_info: Dicionário {machine_id: RULEstimate}
        config: Configuração do ajuste
    
    Returns:
        PlanAdjustmentResult com plano ajustado e decisões
    
    Pipeline:
    1. Identificar máquinas em risco
    2. Para cada operação em máquina de risco:
       a. Se crítica: tentar redistribuir para máquina saudável
       b. Se warning: penalizar e marcar para monitoramento
       c. Se caution: apenas penalizar ligeiramente
    3. Opcionalmente agendar manutenção preventiva
    4. Documentar todas as decisões
    """
    config = config or RULAdjustmentConfig()
    decisions: List[PlanAdjustmentDecision] = []
    
    # Criar cópia do plano
    adjusted_df = plan_df.copy()
    
    # Converter RUL info para MachineRULInfo
    machine_rul: Dict[str, MachineRULInfo] = {}
    for machine_id in plan_df['machine_id'].unique():
        estimate = rul_info.get(machine_id)
        machine_rul[machine_id] = MachineRULInfo(machine_id=machine_id, rul_estimate=estimate)
    
    # Adicionar máquinas do rul_info que podem não estar no plano
    for machine_id, estimate in rul_info.items():
        if machine_id not in machine_rul:
            machine_rul[machine_id] = MachineRULInfo(machine_id=machine_id, rul_estimate=estimate)
    
    # Métricas
    operations_redistributed = 0
    operations_avoided = 0
    maintenance_scheduled = 0
    total_penalty = 0.0
    
    # 1. Identificar máquinas críticas
    critical_machines: Set[str] = set()
    warning_machines: Set[str] = set()
    caution_machines: Set[str] = set()
    healthy_machines: Set[str] = set()
    
    for machine_id, info in machine_rul.items():
        if info.rul_hours < config.rul_threshold_critical:
            critical_machines.add(machine_id)
        elif info.rul_hours < config.rul_threshold_warning:
            warning_machines.add(machine_id)
        elif info.rul_hours < config.rul_threshold_caution:
            caution_machines.add(machine_id)
        else:
            healthy_machines.add(machine_id)
    
    if config.log_decisions:
        logger.info(f"Máquinas críticas: {critical_machines}")
        logger.info(f"Máquinas em warning: {warning_machines}")
        logger.info(f"Máquinas em caution: {caution_machines}")
        logger.info(f"Máquinas saudáveis: {healthy_machines}")
    
    # 2. Processar operações
    if 'rul_penalty' not in adjusted_df.columns:
        adjusted_df['rul_penalty'] = 1.0
    if 'rul_decision' not in adjusted_df.columns:
        adjusted_df['rul_decision'] = ''
    
    for idx, row in adjusted_df.iterrows():
        machine_id = row['machine_id']
        order_id = row.get('order_id', f'OP-{idx}')
        duration_min = row.get('duration_min', 0)
        
        # Verificar se é operação crítica (longa)
        is_critical_op = duration_min > config.critical_op_duration_threshold_min
        
        # Determinar penalização
        penalty = config.penalty_healthy
        decision_type = None
        reason = ""
        
        if machine_id in critical_machines:
            # Máquina crítica - evitar se possível
            penalty = config.penalty_critical
            
            if config.enable_load_redistribution and healthy_machines:
                # Tentar redistribuir
                alternative = _find_alternative_machine(
                    row, healthy_machines, adjusted_df, config
                )
                if alternative:
                    # Redistribuir
                    decisions.append(PlanAdjustmentDecision(
                        decision_type="REDISTRIBUTE",
                        machine_id=machine_id,
                        operation_id=order_id,
                        reason=f"Máquina {machine_id} crítica (RUL={machine_rul[machine_id].rul_hours:.1f}h). "
                               f"Redistribuída para {alternative}.",
                        original_value=machine_id,
                        adjusted_value=alternative,
                    ))
                    adjusted_df.at[idx, 'machine_id'] = alternative
                    adjusted_df.at[idx, 'rul_decision'] = f"REDISTRIBUTE:{alternative}"
                    operations_redistributed += 1
                    penalty = config.penalty_healthy
                else:
                    # Não foi possível redistribuir - marcar como evitar
                    decisions.append(PlanAdjustmentDecision(
                        decision_type="AVOID_RECOMMENDED",
                        machine_id=machine_id,
                        operation_id=order_id,
                        reason=f"Máquina {machine_id} crítica (RUL={machine_rul[machine_id].rul_hours:.1f}h). "
                               f"Sem alternativa disponível. RISCO ELEVADO.",
                        original_value=None,
                        adjusted_value=None,
                    ))
                    adjusted_df.at[idx, 'rul_decision'] = "CRITICAL_NO_ALTERNATIVE"
                    operations_avoided += 1
            else:
                decisions.append(PlanAdjustmentDecision(
                    decision_type="PENALTY",
                    machine_id=machine_id,
                    operation_id=order_id,
                    reason=f"Máquina {machine_id} crítica. Penalização alta aplicada.",
                    original_value=1.0,
                    adjusted_value=penalty,
                ))
                adjusted_df.at[idx, 'rul_decision'] = "CRITICAL_PENALIZED"
        
        elif machine_id in warning_machines:
            penalty = config.penalty_warning
            
            # Evitar operações longas em máquinas warning
            if is_critical_op and config.enable_load_redistribution and healthy_machines:
                alternative = _find_alternative_machine(
                    row, healthy_machines, adjusted_df, config
                )
                if alternative:
                    decisions.append(PlanAdjustmentDecision(
                        decision_type="REDISTRIBUTE",
                        machine_id=machine_id,
                        operation_id=order_id,
                        reason=f"Operação longa ({duration_min:.0f}min) em máquina warning. "
                               f"Redistribuída para {alternative}.",
                        original_value=machine_id,
                        adjusted_value=alternative,
                    ))
                    adjusted_df.at[idx, 'machine_id'] = alternative
                    adjusted_df.at[idx, 'rul_decision'] = f"REDISTRIBUTE:{alternative}"
                    operations_redistributed += 1
                    penalty = config.penalty_healthy
                else:
                    adjusted_df.at[idx, 'rul_decision'] = "WARNING_PENALIZED"
            else:
                adjusted_df.at[idx, 'rul_decision'] = "WARNING_MONITORED"
        
        elif machine_id in caution_machines:
            penalty = config.penalty_caution
            adjusted_df.at[idx, 'rul_decision'] = "CAUTION_MONITORED"
        
        else:
            adjusted_df.at[idx, 'rul_decision'] = "HEALTHY"
        
        # Aplicar penalização extra para operações críticas
        if is_critical_op and machine_id in (critical_machines | warning_machines):
            penalty *= config.critical_op_penalty_multiplier
        
        adjusted_df.at[idx, 'rul_penalty'] = penalty
        total_penalty += penalty - 1.0  # Penalização acumulada (excesso sobre 1.0)
    
    # 3. Agendar manutenção preventiva
    if config.schedule_maintenance_for_critical:
        for machine_id in critical_machines:
            decisions.append(PlanAdjustmentDecision(
                decision_type="MAINTENANCE",
                machine_id=machine_id,
                operation_id=None,
                reason=f"Manutenção preventiva recomendada para {machine_id}. "
                       f"RUL={machine_rul[machine_id].rul_hours:.1f}h, "
                       f"HI={machine_rul[machine_id].current_hi:.2f}",
                original_value=None,
                adjusted_value=f"{config.maintenance_window_hours}h",
            ))
            maintenance_scheduled += 1
    
    return PlanAdjustmentResult(
        original_plan_df=plan_df,
        adjusted_plan_df=adjusted_df,
        decisions=decisions,
        machine_rul_info=machine_rul,
        operations_redistributed=operations_redistributed,
        operations_avoided=operations_avoided,
        maintenance_scheduled=maintenance_scheduled,
        total_penalty_applied=total_penalty,
    )


def _find_alternative_machine(
    operation_row: pd.Series,
    candidate_machines: Set[str],
    plan_df: pd.DataFrame,
    config: RULAdjustmentConfig,
) -> Optional[str]:
    """
    Encontrar máquina alternativa para uma operação.
    
    TODO[R&D]: Implementar lógica mais sofisticada:
    - Verificar compatibilidade da máquina com a operação
    - Considerar tempo de setup
    - Verificar capacidade disponível
    - Otimizar para minimizar impacto no makespan
    """
    if not candidate_machines:
        return None
    
    # Por agora, retornar a primeira máquina candidata com menor carga
    # Em produção, verificar routing para compatibilidade
    machine_loads = {}
    for machine in candidate_machines:
        load = plan_df[plan_df['machine_id'] == machine]['duration_min'].sum()
        machine_loads[machine] = load
    
    if machine_loads:
        # Escolher máquina com menor carga
        return min(machine_loads, key=machine_loads.get)
    
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_rul_penalties(
    machines: List[str],
    rul_info: Dict[str, RULEstimate],
    config: Optional[RULAdjustmentConfig] = None,
) -> Dict[str, float]:
    """
    Obter penalizações de RUL para um conjunto de máquinas.
    
    Útil para integrar com heurísticos/MILP que usam custos.
    
    Returns:
        Dict[machine_id, penalty_multiplier]
    """
    config = config or RULAdjustmentConfig()
    penalties = {}
    
    for machine_id in machines:
        estimate = rul_info.get(machine_id)
        
        if estimate is None:
            penalties[machine_id] = config.penalty_healthy
            continue
        
        rul_hours = estimate.rul_mean_hours
        
        if rul_hours < config.rul_threshold_critical:
            penalties[machine_id] = config.penalty_critical
        elif rul_hours < config.rul_threshold_warning:
            penalties[machine_id] = config.penalty_warning
        elif rul_hours < config.rul_threshold_caution:
            penalties[machine_id] = config.penalty_caution
        else:
            penalties[machine_id] = config.penalty_healthy
    
    return penalties


def should_avoid_machine(
    machine_id: str,
    rul_estimate: Optional[RULEstimate],
    operation_duration_min: float,
    config: Optional[RULAdjustmentConfig] = None,
) -> Tuple[bool, str]:
    """
    Verificar se uma máquina deve ser evitada para uma operação.
    
    Returns:
        (should_avoid, reason)
    """
    config = config or RULAdjustmentConfig()
    
    if rul_estimate is None:
        return False, "Sem dados de RUL"
    
    rul_hours = rul_estimate.rul_mean_hours
    
    # Crítica - sempre evitar se possível
    if rul_hours < config.rul_threshold_critical:
        return True, f"Máquina crítica (RUL={rul_hours:.1f}h < {config.rul_threshold_critical}h)"
    
    # Warning - evitar operações longas
    if rul_hours < config.rul_threshold_warning:
        if operation_duration_min > config.critical_op_duration_threshold_min:
            return True, f"Operação longa em máquina warning (RUL={rul_hours:.1f}h)"
        return False, f"Máquina warning mas operação curta"
    
    return False, "Máquina saudável"


def compute_plan_reliability(
    plan_df: pd.DataFrame,
    rul_info: Dict[str, RULEstimate],
) -> float:
    """
    Calcular "reliability score" do plano (0-1).
    
    Score alto = plano usa maioritariamente máquinas saudáveis.
    Score baixo = plano depende de máquinas em risco.
    
    TODO[R&D]: Integrar com probabilidade de falha durante o plano.
    """
    if plan_df.empty:
        return 1.0
    
    total_weight = 0.0
    reliability_sum = 0.0
    
    for _, row in plan_df.iterrows():
        machine_id = row['machine_id']
        duration = row.get('duration_min', 1.0)
        
        estimate = rul_info.get(machine_id)
        if estimate is None:
            machine_reliability = 1.0
        else:
            # Reliability baseado em HI
            machine_reliability = estimate.current_hi
        
        reliability_sum += machine_reliability * duration
        total_weight += duration
    
    if total_weight > 0:
        return reliability_sum / total_weight
    return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION HOOK FOR SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════

def create_rul_aware_machine_costs(
    machines: List[str],
    base_costs: Optional[Dict[str, float]] = None,
    rul_info: Optional[Dict[str, RULEstimate]] = None,
    config: Optional[RULAdjustmentConfig] = None,
) -> Dict[str, float]:
    """
    Criar custos de máquina ajustados por RUL.
    
    Para usar com MILP/CP-SAT: minimizar sum(cost[m] * usage[m])
    
    Args:
        machines: Lista de IDs de máquinas
        base_costs: Custos base (default=1.0 para todos)
        rul_info: Informação de RUL
        config: Configuração
    
    Returns:
        Dict[machine_id, adjusted_cost]
    """
    config = config or RULAdjustmentConfig()
    base_costs = base_costs or {m: 1.0 for m in machines}
    rul_info = rul_info or {}
    
    penalties = get_rul_penalties(machines, rul_info, config)
    
    adjusted_costs = {}
    for machine_id in machines:
        base = base_costs.get(machine_id, 1.0)
        penalty = penalties.get(machine_id, 1.0)
        adjusted_costs[machine_id] = base * penalty
    
    return adjusted_costs



