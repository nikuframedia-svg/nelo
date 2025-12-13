"""
════════════════════════════════════════════════════════════════════════════════════════════════════
ZDM SIMULATOR - Simulador de Resiliência Zero Disruption Manufacturing
════════════════════════════════════════════════════════════════════════════════════════════════════

Simula a execução do plano de produção em cenários de falha e calcula métricas de resiliência.

Features:
- Simulação de perturbações no cronograma
- Estratégias de auto-recuperação
- Cálculo de métricas de impacto
- Resilience Score agregado
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np

from .failure_scenario_generator import FailureScenario, FailureType


class RecoveryStatus(Enum):
    """Status de recuperação após uma falha."""
    SUCCESS = "success"           # Recuperação total
    PARTIAL = "partial"           # Recuperação parcial
    FAILED = "failed"             # Não conseguiu recuperar
    NOT_ATTEMPTED = "not_attempted"


@dataclass
class ImpactMetrics:
    """
    Métricas de impacto de um cenário de falha.
    """
    # Tempo
    downtime_hours: float = 0.0
    recovery_time_hours: float = 0.0
    total_delay_hours: float = 0.0
    
    # Throughput
    orders_impacted: int = 0
    operations_delayed: int = 0
    throughput_loss_pct: float = 0.0
    
    # OTD (On-Time Delivery)
    otd_deviation_hours: float = 0.0
    orders_at_risk: int = 0
    otd_impact_pct: float = 0.0
    
    # Qualidade
    defective_units: int = 0
    rework_hours: float = 0.0
    
    # Custo estimado
    estimated_cost_eur: float = 0.0
    
    def severity_score(self) -> float:
        """Calcula score de severidade (0-100, maior = pior)."""
        score = 0.0
        
        # Tempo (peso 30%)
        time_score = min(100, (self.downtime_hours / 24) * 50 + (self.recovery_time_hours / 12) * 50)
        score += time_score * 0.3
        
        # Throughput (peso 30%)
        throughput_score = min(100, self.throughput_loss_pct * 2 + self.operations_delayed * 5)
        score += throughput_score * 0.3
        
        # OTD (peso 40%)
        otd_score = min(100, self.otd_impact_pct * 2 + self.orders_at_risk * 10)
        score += otd_score * 0.4
        
        return round(min(100, score), 1)
    
    def to_dict(self) -> Dict:
        """Converte para dicionário."""
        return {
            "downtime_hours": round(self.downtime_hours, 1),
            "recovery_time_hours": round(self.recovery_time_hours, 1),
            "total_delay_hours": round(self.total_delay_hours, 1),
            "orders_impacted": self.orders_impacted,
            "operations_delayed": self.operations_delayed,
            "throughput_loss_pct": round(self.throughput_loss_pct, 1),
            "otd_deviation_hours": round(self.otd_deviation_hours, 1),
            "orders_at_risk": self.orders_at_risk,
            "otd_impact_pct": round(self.otd_impact_pct, 1),
            "defective_units": self.defective_units,
            "rework_hours": round(self.rework_hours, 1),
            "estimated_cost_eur": round(self.estimated_cost_eur, 0),
            "severity_score": self.severity_score(),
        }


@dataclass
class SimulationResult:
    """
    Resultado da simulação de um cenário de falha.
    """
    scenario: FailureScenario
    impact: ImpactMetrics
    recovery_status: RecoveryStatus
    recovery_actions_taken: List[str]
    adjusted_plan: Optional[pd.DataFrame] = None
    
    def to_dict(self) -> Dict:
        """Converte para dicionário."""
        return {
            "scenario": self.scenario.to_dict(),
            "impact": self.impact.to_dict(),
            "recovery_status": self.recovery_status.value,
            "recovery_actions_taken": self.recovery_actions_taken,
            "resilience_score": 100 - self.impact.severity_score(),
        }


@dataclass
class ResilienceReport:
    """
    Relatório de resiliência agregado de múltiplas simulações.
    """
    simulation_results: List[SimulationResult]
    
    # Scores agregados
    overall_resilience_score: float = 0.0
    avg_recovery_time_hours: float = 0.0
    avg_throughput_loss_pct: float = 0.0
    avg_otd_impact_pct: float = 0.0
    
    # Estatísticas
    scenarios_simulated: int = 0
    full_recovery_count: int = 0
    partial_recovery_count: int = 0
    failed_recovery_count: int = 0
    
    # Máquinas mais críticas
    critical_machines: List[str] = field(default_factory=list)
    
    # Recomendações
    recommendations: List[str] = field(default_factory=list)
    
    def compute_aggregates(self):
        """Calcula métricas agregadas."""
        if not self.simulation_results:
            return
        
        self.scenarios_simulated = len(self.simulation_results)
        
        # Contagens de recovery status
        self.full_recovery_count = sum(
            1 for r in self.simulation_results if r.recovery_status == RecoveryStatus.SUCCESS
        )
        self.partial_recovery_count = sum(
            1 for r in self.simulation_results if r.recovery_status == RecoveryStatus.PARTIAL
        )
        self.failed_recovery_count = sum(
            1 for r in self.simulation_results if r.recovery_status == RecoveryStatus.FAILED
        )
        
        # Médias
        self.avg_recovery_time_hours = np.mean([
            r.impact.recovery_time_hours for r in self.simulation_results
        ])
        self.avg_throughput_loss_pct = np.mean([
            r.impact.throughput_loss_pct for r in self.simulation_results
        ])
        self.avg_otd_impact_pct = np.mean([
            r.impact.otd_impact_pct for r in self.simulation_results
        ])
        
        # Resilience Score (inverso da severidade média)
        avg_severity = np.mean([
            r.impact.severity_score() for r in self.simulation_results
        ])
        self.overall_resilience_score = max(0, 100 - avg_severity)
        
        # Máquinas mais críticas
        machine_impact = {}
        for result in self.simulation_results:
            machine = result.scenario.machine_id
            if machine not in machine_impact:
                machine_impact[machine] = []
            machine_impact[machine].append(result.impact.severity_score())
        
        # Top 5 máquinas por severidade média
        machine_avg = {m: np.mean(scores) for m, scores in machine_impact.items()}
        self.critical_machines = sorted(machine_avg.keys(), key=lambda m: -machine_avg[m])[:5]
        
        # Gerar recomendações
        self._generate_recommendations()
    
    def _generate_recommendations(self):
        """Gera recomendações baseadas nos resultados."""
        self.recommendations = []
        
        # Resilience Score baixo
        if self.overall_resilience_score < 50:
            self.recommendations.append(
                "⚠️ Resilience Score crítico (<50). Considerar redundância de máquinas críticas."
            )
        elif self.overall_resilience_score < 70:
            self.recommendations.append(
                "⚡ Resilience Score moderado. Melhorar planos de contingência."
            )
        
        # Taxa de recuperação
        recovery_rate = self.full_recovery_count / max(1, self.scenarios_simulated) * 100
        if recovery_rate < 50:
            self.recommendations.append(
                f"📉 Taxa de recuperação total baixa ({recovery_rate:.0f}%). Investir em flexibilidade produtiva."
            )
        
        # Tempo de recuperação
        if self.avg_recovery_time_hours > 8:
            self.recommendations.append(
                f"⏱️ Tempo médio de recuperação elevado ({self.avg_recovery_time_hours:.1f}h). Otimizar processos de manutenção."
            )
        
        # OTD
        if self.avg_otd_impact_pct > 10:
            self.recommendations.append(
                f"📦 Impacto OTD significativo ({self.avg_otd_impact_pct:.1f}%). Criar buffers de segurança."
            )
        
        # Máquinas críticas
        if self.critical_machines:
            self.recommendations.append(
                f"🔧 Máquinas críticas: {', '.join(self.critical_machines[:3])}. Priorizar manutenção preventiva."
            )
        
        if not self.recommendations:
            self.recommendations.append(
                "✅ Plano apresenta boa resiliência. Manter programa de manutenção atual."
            )
    
    def to_dict(self) -> Dict:
        """Converte para dicionário."""
        return {
            "overall_resilience_score": round(self.overall_resilience_score, 1),
            "scenarios_simulated": self.scenarios_simulated,
            "recovery_stats": {
                "full_recovery": self.full_recovery_count,
                "partial_recovery": self.partial_recovery_count,
                "failed_recovery": self.failed_recovery_count,
                "success_rate_pct": round(
                    self.full_recovery_count / max(1, self.scenarios_simulated) * 100, 1
                ),
            },
            "averages": {
                "recovery_time_hours": round(self.avg_recovery_time_hours, 1),
                "throughput_loss_pct": round(self.avg_throughput_loss_pct, 1),
                "otd_impact_pct": round(self.avg_otd_impact_pct, 1),
            },
            "critical_machines": self.critical_machines,
            "recommendations": self.recommendations,
            "scenario_details": [r.to_dict() for r in self.simulation_results[:10]],  # Top 10
        }


@dataclass
class SimulationConfig:
    """
    Configuração do simulador ZDM.
    """
    # Estratégias de recuperação a tentar
    enable_rerouting: bool = True          # Reencaminhar para máquinas alternativas
    enable_overtime: bool = True           # Adicionar horas extra
    enable_priority_shuffle: bool = True   # Reordenar prioridades
    enable_partial_batch: bool = False     # Dividir lotes
    
    # Parâmetros
    max_overtime_hours: float = 4.0        # Máximo de horas extra por dia
    rerouting_efficiency: float = 0.8      # Eficiência após reencaminhamento
    cost_per_downtime_hour: float = 500.0  # € por hora de paragem
    cost_per_delayed_order: float = 200.0  # € por encomenda atrasada
    
    # Simulação
    use_monte_carlo: bool = False          # Usar Monte Carlo para variabilidade
    monte_carlo_iterations: int = 100


class ZDMSimulator:
    """
    Simulador de Zero Disruption Manufacturing.
    
    Simula o impacto de falhas no plano de produção e tenta
    estratégias de recuperação automática.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Inicializa o simulador.
        
        Args:
            config: Configuração do simulador
        """
        self.config = config or SimulationConfig()
        self._alternative_machines: Dict[str, List[str]] = {}
    
    def set_alternative_machines(self, alternatives: Dict[str, List[str]]):
        """
        Define máquinas alternativas para reencaminhamento.
        
        Args:
            alternatives: Dict {machine_id: [alternative_machine_ids]}
        """
        self._alternative_machines = alternatives
    
    def simulate_scenario(
        self,
        plan_df: pd.DataFrame,
        scenario: FailureScenario,
    ) -> SimulationResult:
        """
        Simula um único cenário de falha.
        
        Args:
            plan_df: DataFrame do plano de produção
            scenario: Cenário de falha a simular
        
        Returns:
            SimulationResult com impacto e recuperação
        """
        # 1. Calcular impacto inicial
        impact = self._calculate_initial_impact(plan_df, scenario)
        
        # 2. Tentar estratégias de recuperação
        recovery_status, actions, adjusted_plan = self._attempt_recovery(
            plan_df, scenario, impact
        )
        
        # 3. Recalcular impacto após recuperação
        if recovery_status != RecoveryStatus.NOT_ATTEMPTED:
            impact = self._calculate_post_recovery_impact(
                plan_df, adjusted_plan, scenario, recovery_status
            )
        
        return SimulationResult(
            scenario=scenario,
            impact=impact,
            recovery_status=recovery_status,
            recovery_actions_taken=actions,
            adjusted_plan=adjusted_plan,
        )
    
    def simulate_all(
        self,
        plan_df: pd.DataFrame,
        scenarios: List[FailureScenario],
    ) -> ResilienceReport:
        """
        Simula múltiplos cenários e gera relatório de resiliência.
        
        Args:
            plan_df: DataFrame do plano de produção
            scenarios: Lista de cenários de falha
        
        Returns:
            ResilienceReport agregado
        """
        results = []
        for scenario in scenarios:
            result = self.simulate_scenario(plan_df, scenario)
            results.append(result)
        
        report = ResilienceReport(simulation_results=results)
        report.compute_aggregates()
        
        return report
    
    def _calculate_initial_impact(
        self,
        plan_df: pd.DataFrame,
        scenario: FailureScenario,
    ) -> ImpactMetrics:
        """Calcula impacto inicial da falha (sem recuperação)."""
        impact = ImpactMetrics()
        
        if plan_df.empty:
            return impact
        
        # Identificar operações afetadas
        machine_ops = plan_df[plan_df['machine_id'] == scenario.machine_id].copy()
        
        if machine_ops.empty:
            return impact
        
        # Converter tempos
        try:
            machine_ops['start_time'] = pd.to_datetime(machine_ops['start_time'])
            machine_ops['end_time'] = pd.to_datetime(machine_ops['end_time'])
        except Exception:
            pass
        
        # Calcular overlap com período de falha
        failure_start = scenario.start_time
        failure_end = scenario.start_time + timedelta(hours=scenario.duration_hours)
        
        affected_ops = machine_ops[
            (machine_ops['start_time'] < failure_end) &
            (machine_ops['end_time'] > failure_start)
        ]
        
        impact.operations_delayed = len(affected_ops)
        impact.downtime_hours = scenario.duration_hours
        
        # Estimar ordens impactadas
        if 'order_id' in affected_ops.columns:
            impact.orders_impacted = affected_ops['order_id'].nunique()
        else:
            impact.orders_impacted = max(1, len(affected_ops) // 3)
        
        # Throughput loss
        total_ops = len(plan_df)
        if total_ops > 0:
            impact.throughput_loss_pct = (impact.operations_delayed / total_ops) * 100 * scenario.severity
        
        # OTD impact
        impact.orders_at_risk = int(impact.orders_impacted * scenario.severity)
        impact.otd_deviation_hours = scenario.duration_hours * scenario.severity
        impact.otd_impact_pct = (impact.orders_at_risk / max(1, impact.orders_impacted)) * 100
        
        # Qualidade (para falhas de qualidade)
        if scenario.failure_type == FailureType.QUALITY:
            if 'qty' in affected_ops.columns:
                total_qty = affected_ops['qty'].sum()
                impact.defective_units = int(total_qty * scenario.quality_reject_rate)
            else:
                impact.defective_units = int(10 * scenario.quality_reject_rate * 100)
            impact.rework_hours = impact.defective_units * 0.5  # 30 min por unidade
        
        # Custo estimado
        impact.estimated_cost_eur = (
            impact.downtime_hours * self.config.cost_per_downtime_hour +
            impact.orders_at_risk * self.config.cost_per_delayed_order
        )
        
        return impact
    
    def _attempt_recovery(
        self,
        plan_df: pd.DataFrame,
        scenario: FailureScenario,
        impact: ImpactMetrics,
    ) -> Tuple[RecoveryStatus, List[str], pd.DataFrame]:
        """
        Tenta estratégias de recuperação.
        
        Returns:
            (status, actions_taken, adjusted_plan)
        """
        actions = []
        adjusted_plan = plan_df.copy()
        
        if impact.operations_delayed == 0:
            return RecoveryStatus.SUCCESS, ["Sem operações afetadas"], adjusted_plan
        
        recovery_success = 0.0  # 0 a 1
        
        # Estratégia 1: Reencaminhamento para máquinas alternativas
        if self.config.enable_rerouting:
            alternatives = self._alternative_machines.get(scenario.machine_id, [])
            if not alternatives:
                # Gerar alternativas automáticas (máquinas do mesmo tipo)
                prefix = scenario.machine_id.split('-')[0] if '-' in scenario.machine_id else scenario.machine_id[0]
                alternatives = [
                    m for m in plan_df['machine_id'].unique()
                    if m != scenario.machine_id and m.startswith(prefix)
                ][:2]
            
            if alternatives:
                actions.append(f"Reencaminhamento para {alternatives[0]}")
                recovery_success += 0.4 * self.config.rerouting_efficiency
        
        # Estratégia 2: Horas extra
        if self.config.enable_overtime:
            overtime_potential = self.config.max_overtime_hours / max(1, scenario.duration_hours)
            if overtime_potential > 0.2:
                actions.append(f"Adição de {min(self.config.max_overtime_hours, scenario.duration_hours):.1f}h extra")
                recovery_success += 0.3 * min(1.0, overtime_potential)
        
        # Estratégia 3: Repriorização
        if self.config.enable_priority_shuffle:
            actions.append("Repriorização de ordens VIP")
            recovery_success += 0.2
        
        # Determinar status de recuperação
        if recovery_success >= 0.8:
            status = RecoveryStatus.SUCCESS
        elif recovery_success >= 0.4:
            status = RecoveryStatus.PARTIAL
        elif recovery_success > 0:
            status = RecoveryStatus.FAILED
        else:
            status = RecoveryStatus.NOT_ATTEMPTED
            actions = ["Nenhuma estratégia aplicável"]
        
        return status, actions, adjusted_plan
    
    def _calculate_post_recovery_impact(
        self,
        original_plan: pd.DataFrame,
        adjusted_plan: pd.DataFrame,
        scenario: FailureScenario,
        recovery_status: RecoveryStatus,
    ) -> ImpactMetrics:
        """Recalcula impacto após tentativa de recuperação."""
        # Começar com impacto inicial
        impact = self._calculate_initial_impact(original_plan, scenario)
        
        # Reduzir baseado no status de recuperação
        reduction_factor = {
            RecoveryStatus.SUCCESS: 0.2,      # 80% de redução
            RecoveryStatus.PARTIAL: 0.5,      # 50% de redução
            RecoveryStatus.FAILED: 0.9,       # 10% de redução
            RecoveryStatus.NOT_ATTEMPTED: 1.0,
        }[recovery_status]
        
        # Aplicar redução
        impact.throughput_loss_pct *= reduction_factor
        impact.otd_impact_pct *= reduction_factor
        impact.orders_at_risk = int(impact.orders_at_risk * reduction_factor)
        impact.otd_deviation_hours *= reduction_factor
        impact.estimated_cost_eur *= reduction_factor
        
        # Tempo de recuperação
        impact.recovery_time_hours = scenario.duration_hours * (1 - reduction_factor) + scenario.duration_hours * 0.3
        impact.total_delay_hours = impact.downtime_hours + impact.recovery_time_hours
        
        return impact


def quick_resilience_check(
    plan_df: pd.DataFrame,
    n_scenarios: int = 5,
) -> Dict:
    """
    Verificação rápida de resiliência do plano.
    
    Args:
        plan_df: DataFrame do plano
        n_scenarios: Número de cenários a simular
    
    Returns:
        Dict com score e resumo
    """
    from .failure_scenario_generator import generate_failure_scenarios
    
    scenarios = generate_failure_scenarios(plan_df, n_scenarios=n_scenarios)
    simulator = ZDMSimulator()
    report = simulator.simulate_all(plan_df, scenarios)
    
    return {
        "resilience_score": round(report.overall_resilience_score, 1),
        "scenarios_tested": report.scenarios_simulated,
        "recovery_success_rate": round(
            report.full_recovery_count / max(1, report.scenarios_simulated) * 100, 1
        ),
        "critical_machines": report.critical_machines[:3],
        "top_recommendation": report.recommendations[0] if report.recommendations else None,
    }


