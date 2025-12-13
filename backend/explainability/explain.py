"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — EXPLAINABILITY ENGINE (XAI)
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Human-readable explanations for all APS and ML decisions, with mathematical rigor and SNR integration.

EXPLAINABLE AI PRINCIPLES
═════════════════════════

1. TRANSPARENCY: Why was this decision made?
2. CAUSALITY: What factors influenced the outcome?
3. COUNTERFACTUAL: What would change the decision?
4. CONFIDENCE: How reliable is this decision? (via SNR)
5. AUDITABILITY: Full trace for SIFIDE compliance

EXPLANATION STRUCTURE
─────────────────────

Every explanation includes:

    ┌─────────────────────────────────────────────────────────────────┐
    │ DECISÃO                                                         │
    │ O que foi decidido                                              │
    ├─────────────────────────────────────────────────────────────────┤
    │ JUSTIFICAÇÃO                                                    │
    │ • Fator 1: Descrição + Fórmula + Valor                         │
    │ • Fator 2: Descrição + Fórmula + Valor                         │
    ├─────────────────────────────────────────────────────────────────┤
    │ CONFIANÇA                                                       │
    │ SNR = X.X → Nível: HIGH/MEDIUM/LOW                             │
    │ [████████░░] 80%                                                │
    ├─────────────────────────────────────────────────────────────────┤
    │ ALTERNATIVAS                                                    │
    │ • Alternativa 1: Razão da rejeição                             │
    │ • Alternativa 2: Razão da rejeição                             │
    ├─────────────────────────────────────────────────────────────────┤
    │ CONTRAFACTUAL                                                   │
    │ "Se X fosse diferente, então Y seria selecionado"              │
    └─────────────────────────────────────────────────────────────────┘

MATHEMATICAL NOTATION IN EXPLANATIONS
─────────────────────────────────────

Explanations include mathematical formulas for transparency:

    "O tempo de setup foi calculado como:
     S(FAM-A → FAM-B) = s_{AB} = 15.0 min
     
     onde s_{AB} é o tempo da matriz de setup."

R&D / SIFIDE: WP4 - Explainability
─────────────────────────────────
- Hypothesis H4.4: XAI improves operator trust calibration
- Experiment E4.4: Compare acceptance rates with/without explanations

REFERENCES
──────────
[1] Miller, T. (2019). Explanation in AI. Artificial Intelligence.
[2] Arrieta et al. (2020). Explainable AI (XAI): Concepts and challenges.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# SNR HELPERS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def format_snr_bar(snr: float, width: int = 10) -> str:
    """Create visual SNR bar."""
    confidence = snr / (1 + snr) if snr > 0 else 0
    filled = int(confidence * width)
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}] {confidence * 100:.0f}%"


def snr_level_pt(snr: float) -> str:
    """Get SNR level in Portuguese."""
    if snr >= 10:
        return "EXCELENTE"
    elif snr >= 5:
        return "ALTO"
    elif snr >= 2:
        return "MODERADO"
    elif snr >= 1:
        return "BAIXO"
    else:
        return "MUITO BAIXO"


def snr_description_pt(snr: float) -> str:
    """Get SNR description in Portuguese."""
    if snr >= 10:
        return "Alta previsibilidade. Decisão muito fiável."
    elif snr >= 5:
        return "Boa previsibilidade. Decisão fiável."
    elif snr >= 2:
        return "Previsibilidade moderada. Monitorizar resultado."
    elif snr >= 1:
        return "Previsibilidade limitada. Usar com cautela."
    else:
        return "Previsibilidade muito baixa. Considerar alternativas."


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# EXPLANATION DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class Factor:
    """A factor that influenced a decision."""
    name: str
    description: str
    value: Any
    formula: Optional[str] = None
    weight: str = "medium"  # "high", "medium", "low"
    
    def to_text(self) -> str:
        text = f"• {self.name}: {self.description}"
        if self.formula:
            text += f"\n  Fórmula: {self.formula}"
        text += f"\n  Valor: {self.value}"
        return text


@dataclass
class Alternative:
    """An alternative that was considered but not chosen."""
    option: str
    reason_rejected: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_text(self) -> str:
        return f"• {self.option}: {self.reason_rejected}"


@dataclass
class Explanation:
    """Complete explanation for a decision."""
    decision_type: str
    decision_summary: str
    
    # Factors that influenced the decision
    factors: List[Factor] = field(default_factory=list)
    
    # Alternatives considered
    alternatives: List[Alternative] = field(default_factory=list)
    
    # Counterfactual
    counterfactual: Optional[str] = None
    
    # SNR-based confidence
    snr: float = 1.0
    snr_level: str = "BAIXO"
    snr_description: str = ""
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'decision_type': self.decision_type,
            'decision_summary': self.decision_summary,
            'factors': [{'name': f.name, 'description': f.description, 'value': f.value, 'formula': f.formula} for f in self.factors],
            'alternatives': [{'option': a.option, 'reason': a.reason_rejected} for a in self.alternatives],
            'counterfactual': self.counterfactual,
            'snr': round(self.snr, 2),
            'snr_level': self.snr_level,
            'confidence_pct': round(self.snr / (1 + self.snr) * 100, 1),
            'timestamp': self.timestamp,
        }
    
    def to_text(self) -> str:
        """Generate human-readable explanation in Portuguese."""
        lines = [
            "╔═══════════════════════════════════════════════════════════════════╗",
            f"║ {self.decision_type.upper().center(65)} ║",
            "╠═══════════════════════════════════════════════════════════════════╣",
            "",
            "📋 DECISÃO",
            f"   {self.decision_summary}",
            "",
        ]
        
        # Factors
        if self.factors:
            lines.append("📊 JUSTIFICAÇÃO (fatores considerados)")
            for factor in self.factors:
                lines.append(factor.to_text())
            lines.append("")
        
        # Confidence
        lines.extend([
            "🎯 CONFIANÇA",
            f"   SNR = {self.snr:.1f} → Nível: {self.snr_level}",
            f"   {format_snr_bar(self.snr)}",
            f"   {self.snr_description}",
            "",
        ])
        
        # Alternatives
        if self.alternatives:
            lines.append("🔄 ALTERNATIVAS CONSIDERADAS")
            for alt in self.alternatives:
                lines.append(alt.to_text())
            lines.append("")
        
        # Counterfactual
        if self.counterfactual:
            lines.extend([
                "❓ CONTRAFACTUAL",
                f"   {self.counterfactual}",
                "",
            ])
        
        lines.append("╚═══════════════════════════════════════════════════════════════════╝")
        
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# EXPLANATION FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def explain_routing_decision(
    operation_id: str,
    chosen_machine: str,
    eligible_machines: List[str],
    processing_times: Dict[str, float],
    machine_loads: Optional[Dict[str, float]] = None,
    setup_times: Optional[Dict[str, float]] = None,
    snr: float = 1.0,
    context: Optional[Dict[str, Any]] = None
) -> Explanation:
    """
    Explain why an operation was routed to a specific machine.
    
    Args:
        operation_id: ID of the operation
        chosen_machine: Machine that was selected
        eligible_machines: List of machines that could process the operation
        processing_times: Dict machine_id -> processing time (minutes)
        machine_loads: Dict machine_id -> current load (0-1)
        setup_times: Dict machine_id -> setup time from previous operation
        snr: Signal-to-Noise Ratio for this decision
        context: Additional context
    
    Returns:
        Explanation object
    """
    factors = []
    alternatives = []
    
    # Processing time factor
    chosen_time = processing_times.get(chosen_machine, 0)
    min_time = min(processing_times.values()) if processing_times else 0
    
    if processing_times:
        if chosen_time == min_time:
            factors.append(Factor(
                name="Tempo de processamento",
                description="Máquina com menor tempo de processamento",
                value=f"{chosen_time:.1f} min",
                formula=f"p(o, {chosen_machine}) = {chosen_time:.1f} min = min{{p(o, m) : m ∈ M}}",
                weight="high"
            ))
        else:
            factors.append(Factor(
                name="Tempo de processamento",
                description=f"Tempo de processamento na máquina selecionada",
                value=f"{chosen_time:.1f} min (mínimo disponível: {min_time:.1f} min)",
                formula=f"p(o, {chosen_machine}) = {chosen_time:.1f} min",
                weight="medium"
            ))
    
    # Machine load factor
    if machine_loads:
        chosen_load = machine_loads.get(chosen_machine, 0)
        min_load = min(machine_loads.values())
        
        if chosen_load == min_load:
            factors.append(Factor(
                name="Carga da máquina",
                description="Máquina com menor carga atual",
                value=f"{chosen_load * 100:.1f}%",
                formula=f"load({chosen_machine}) = {chosen_load * 100:.1f}% = min{{load(m) : m ∈ M}}",
                weight="high"
            ))
        else:
            factors.append(Factor(
                name="Carga da máquina",
                description="Carga atual da máquina selecionada",
                value=f"{chosen_load * 100:.1f}%",
                weight="low"
            ))
    
    # Setup time factor
    if setup_times:
        chosen_setup = setup_times.get(chosen_machine, 0)
        min_setup = min(setup_times.values())
        
        if chosen_setup == min_setup and chosen_setup > 0:
            factors.append(Factor(
                name="Tempo de setup",
                description="Menor tempo de mudança de família de setup",
                value=f"{chosen_setup:.1f} min",
                formula=f"setup(prev → {chosen_machine}) = s_{{f_{{prev}}, f_{{cur}}}} = {chosen_setup:.1f} min",
                weight="high"
            ))
    
    # Alternatives
    for m in eligible_machines:
        if m != chosen_machine:
            reasons = []
            if processing_times and processing_times.get(m, 0) > chosen_time:
                diff = processing_times[m] - chosen_time
                reasons.append(f"+{diff:.1f} min de processamento")
            if machine_loads and machine_loads.get(m, 0) > machine_loads.get(chosen_machine, 0):
                reasons.append("maior carga")
            if setup_times and setup_times.get(m, 0) > setup_times.get(chosen_machine, 0):
                reasons.append("maior setup")
            
            alternatives.append(Alternative(
                option=m,
                reason_rejected=", ".join(reasons) if reasons else "não otimal"
            ))
    
    # Counterfactual
    counterfactual = None
    if machine_loads and len(eligible_machines) > 1:
        # Find machine with lowest load that isn't chosen
        other_machines = [m for m in eligible_machines if m != chosen_machine]
        if other_machines:
            best_alt = min(other_machines, key=lambda m: machine_loads.get(m, 1))
            alt_load = machine_loads.get(best_alt, 0)
            chosen_load_val = machine_loads.get(chosen_machine, 0)
            if alt_load < chosen_load_val:
                counterfactual = (
                    f"Se {best_alt} tivesse o mesmo tempo de processamento "
                    f"que {chosen_machine}, seria preferida devido à menor carga "
                    f"({alt_load * 100:.0f}% vs {chosen_load_val * 100:.0f}%)."
                )
    
    return Explanation(
        decision_type="Decisão de Rota",
        decision_summary=f"Operação {operation_id} atribuída à máquina {chosen_machine}",
        factors=factors,
        alternatives=alternatives[:3],  # Top 3 alternatives
        counterfactual=counterfactual,
        snr=snr,
        snr_level=snr_level_pt(snr),
        snr_description=snr_description_pt(snr),
        context=context or {},
    )


def explain_setup_cost(
    from_family: str,
    to_family: str,
    setup_time: float,
    snr: float = 1.0,
    historical_avg: Optional[float] = None,
    historical_std: Optional[float] = None
) -> Explanation:
    """
    Explain the setup cost calculation.
    
    Args:
        from_family: Previous setup family
        to_family: Next setup family
        setup_time: Calculated setup time (minutes)
        snr: SNR of the setup time estimate
        historical_avg: Historical average (if available)
        historical_std: Historical standard deviation
    
    Returns:
        Explanation object
    """
    factors = []
    
    # Matrix lookup factor
    factors.append(Factor(
        name="Matriz de Setup",
        description=f"Tempo de transição da família {from_family} para {to_family}",
        value=f"{setup_time:.1f} min",
        formula=f"S({from_family} → {to_family}) = s_{{{from_family},{to_family}}} = {setup_time:.1f} min",
        weight="high"
    ))
    
    # Historical comparison
    if historical_avg is not None:
        factors.append(Factor(
            name="Comparação Histórica",
            description="Média histórica para esta transição",
            value=f"{historical_avg:.1f} ± {historical_std:.1f} min" if historical_std else f"{historical_avg:.1f} min",
            formula=f"μ_{{hist}} = {historical_avg:.1f}, σ_{{hist}} = {historical_std:.1f}" if historical_std else None,
            weight="medium"
        ))
    
    counterfactual = None
    if from_family != to_family:
        counterfactual = (
            f"Se a operação anterior fosse da família {to_family}, "
            f"o setup seria mínimo (~5 min para mesma família)."
        )
    
    return Explanation(
        decision_type="Cálculo de Setup",
        decision_summary=f"Setup de {from_family} para {to_family}: {setup_time:.1f} min",
        factors=factors,
        counterfactual=counterfactual,
        snr=snr,
        snr_level=snr_level_pt(snr),
        snr_description=snr_description_pt(snr),
    )


def explain_forecast(
    article_id: str,
    model_name: str,
    predicted_value: float,
    unit: str,
    snr: float,
    features_used: List[str],
    historical_mean: Optional[float] = None,
    prediction_interval: Optional[Tuple[float, float]] = None,
    mape: Optional[float] = None
) -> Explanation:
    """
    Explain a forecast/prediction.
    
    Args:
        article_id: Article being forecast
        model_name: Name of the forecasting model
        predicted_value: Predicted value
        unit: Unit of measurement
        snr: SNR of the forecast
        features_used: Features used in the model
        historical_mean: Historical mean (if available)
        prediction_interval: (lower, upper) confidence bounds
        mape: Mean Absolute Percentage Error
    
    Returns:
        Explanation object
    """
    factors = []
    
    # Model factor
    factors.append(Factor(
        name="Modelo",
        description=f"Previsão usando {model_name}",
        value=f"{predicted_value:.2f} {unit}",
        formula=f"ŷ = f({', '.join(features_used[:3])}{'...' if len(features_used) > 3 else ''})",
        weight="high"
    ))
    
    # Features
    if features_used:
        factors.append(Factor(
            name="Variáveis",
            description="Features usadas no modelo",
            value=", ".join(features_used[:5]) + ("..." if len(features_used) > 5 else ""),
            weight="medium"
        ))
    
    # Historical comparison
    if historical_mean is not None:
        deviation = ((predicted_value - historical_mean) / historical_mean * 100) if historical_mean != 0 else 0
        factors.append(Factor(
            name="Contexto Histórico",
            description=f"Desvio de {'+'  if deviation > 0 else ''}{deviation:.1f}% da média histórica",
            value=f"Média histórica: {historical_mean:.2f} {unit}",
            weight="low"
        ))
    
    # Prediction interval
    if prediction_interval:
        factors.append(Factor(
            name="Intervalo de Confiança (95%)",
            description="Limites inferior e superior da previsão",
            value=f"[{prediction_interval[0]:.2f}, {prediction_interval[1]:.2f}] {unit}",
            formula=f"IC_{{95%}} = ŷ ± 1.96 × σ_{{pred}}",
            weight="medium"
        ))
    
    # MAPE
    if mape is not None:
        factors.append(Factor(
            name="Erro Médio (MAPE)",
            description="Erro médio absoluto percentual do modelo",
            value=f"{mape:.1f}%",
            formula="MAPE = (1/n) × Σ|yᵢ - ŷᵢ|/|yᵢ| × 100%",
            weight="medium"
        ))
    
    return Explanation(
        decision_type="Previsão",
        decision_summary=f"Previsão para {article_id}: {predicted_value:.2f} {unit}",
        factors=factors,
        snr=snr,
        snr_level=snr_level_pt(snr),
        snr_description=snr_description_pt(snr),
    )


def explain_rl_policy_decision(
    state: Dict[str, Any],
    chosen_action: str,
    available_actions: List[str],
    action_values: Dict[str, float],
    reward: float,
    cumulative_regret: float,
    policy_name: str,
    snr: float = 1.0,
    exploration: bool = False
) -> Explanation:
    """
    Explain a Reinforcement Learning / Bandit policy decision.
    
    Args:
        state: Current state
        chosen_action: Action that was selected
        available_actions: All available actions
        action_values: Estimated value for each action
        reward: Observed reward
        cumulative_regret: Total regret so far
        policy_name: Name of the policy (UCB, Thompson, etc.)
        snr: SNR of the context
        exploration: Whether this was an exploration step
    
    Returns:
        Explanation object
    """
    factors = []
    
    # Policy factor
    factors.append(Factor(
        name="Política",
        description=f"Decisão usando {policy_name}",
        value=f"{'Exploração' if exploration else 'Exploitação'}",
        weight="high"
    ))
    
    # Action values
    if action_values:
        chosen_value = action_values.get(chosen_action, 0)
        max_value = max(action_values.values())
        
        if policy_name.upper() == "UCB":
            factors.append(Factor(
                name="Valor UCB",
                description=f"Upper Confidence Bound para {chosen_action}",
                value=f"{chosen_value:.3f}",
                formula=f"UCB(a) = Q̂(a) + c × √(ln(t) / n(a))",
                weight="high"
            ))
        elif policy_name.upper() == "THOMPSON":
            factors.append(Factor(
                name="Amostra Thompson",
                description=f"Amostra da distribuição posterior para {chosen_action}",
                value=f"{chosen_value:.3f}",
                formula="θ ~ Beta(α, β) ou N(μ, σ²)",
                weight="high"
            ))
        else:
            factors.append(Factor(
                name="Valor Estimado",
                description=f"Q-value estimado para {chosen_action}",
                value=f"{chosen_value:.3f}",
                formula="Q̂(a) = (1/n) × Σ rᵢ",
                weight="high"
            ))
    
    # Reward
    factors.append(Factor(
        name="Recompensa Observada",
        description="Recompensa obtida neste passo",
        value=f"{reward:.4f}",
        weight="medium"
    ))
    
    # Regret
    factors.append(Factor(
        name="Regret Acumulado",
        description="Arrependimento total até ao momento",
        value=f"{cumulative_regret:.4f}",
        formula="Regret(T) = T × μ* - Σₜ rₜ",
        weight="low"
    ))
    
    # Alternatives
    alternatives = []
    for action in available_actions:
        if action != chosen_action and action in action_values:
            val = action_values[action]
            alternatives.append(Alternative(
                option=action,
                reason_rejected=f"Valor estimado: {val:.3f}" + 
                               (f" (< {action_values.get(chosen_action, 0):.3f})" if val < action_values.get(chosen_action, 0) else "")
            ))
    
    # Counterfactual
    counterfactual = None
    if exploration:
        counterfactual = (
            f"Esta foi uma decisão de exploração. "
            f"Com 100% exploitação, {max(action_values, key=action_values.get)} seria escolhida."
        )
    
    return Explanation(
        decision_type="Decisão de Política RL",
        decision_summary=f"Ação {chosen_action} selecionada por {policy_name}",
        factors=factors,
        alternatives=alternatives[:3],
        counterfactual=counterfactual,
        snr=snr,
        snr_level=snr_level_pt(snr),
        snr_description=snr_description_pt(snr),
        context={'state': state, 'policy': policy_name, 'exploration': exploration},
    )


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# BATCH EXPLANATION
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def explain_schedule(
    schedule_df,
    routing_df=None,
    setup_matrix=None,
    snr_by_operation: Optional[Dict[str, float]] = None
) -> List[Explanation]:
    """
    Generate explanations for all operations in a schedule.
    
    Args:
        schedule_df: DataFrame with scheduled operations
        routing_df: Optional routing information
        setup_matrix: Optional setup matrix
        snr_by_operation: Optional SNR per operation
    
    Returns:
        List of Explanation objects
    """
    explanations = []
    
    # Group by machine to analyze sequencing
    for machine_id in schedule_df['machine_id'].unique():
        machine_ops = schedule_df[schedule_df['machine_id'] == machine_id].sort_values('start_time')
        
        for i, (_, row) in enumerate(machine_ops.iterrows()):
            op_id = row.get('operation_id', row.get('id', f'op_{i}'))
            snr = snr_by_operation.get(op_id, 1.0) if snr_by_operation else 1.0
            
            # Create simplified routing explanation
            explanation = Explanation(
                decision_type="Agendamento",
                decision_summary=f"Operação {op_id} agendada em {machine_id} às {row.get('start_time', 'N/A')}",
                factors=[
                    Factor(
                        name="Máquina",
                        description="Máquina atribuída",
                        value=machine_id,
                        weight="high"
                    ),
                    Factor(
                        name="Duração",
                        description="Tempo de processamento",
                        value=f"{row.get('duration_min', 0):.1f} min",
                        weight="medium"
                    ),
                ],
                snr=snr,
                snr_level=snr_level_pt(snr),
                snr_description=snr_description_pt(snr),
            )
            
            explanations.append(explanation)
    
    return explanations



