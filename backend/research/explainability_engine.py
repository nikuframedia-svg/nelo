"""
Explainability Engine — Decision Justification & Explanations

R&D Module for WP2: What-If + Explainable AI

Research Questions:
    Q2: Can we automatically generate human-readable suggestions that
        production directors consider credible and useful?
    Q4: Can we create an explainable "AI Co-pilot for factories" that
        justifies APS choices without becoming a black box?

Hypotheses:
    H2.1: System-generated suggestions rated "useful" by directors ≥70%
    H4.1: Explanations increase user trust by ≥1 point (5-point scale)
    H4.2: Decision justifications match actual scheduling rationale ≥90%

Technical Uncertainty:
    - What level of detail is appropriate for different users?
    - How to explain trade-offs without overwhelming?
    - Can we detect and avoid "plausible but wrong" explanations?
    - How to measure explanation fidelity?

Usage:
    from backend.research.explainability_engine import ExplainabilityEngine
    
    engine = ExplainabilityEngine()
    explanation = engine.explain_routing_decision(
        decision=routing_decision,
        context=scheduling_context,
        target_audience="plant_manager"
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
import json


class AudienceLevel(Enum):
    """Target audience for explanations."""
    OPERATOR = "operator"           # Shop floor, needs simple actionable info
    SUPERVISOR = "supervisor"       # Shift manager, needs operational details
    PLANT_MANAGER = "plant_manager" # Needs strategic overview and KPIs
    ENGINEER = "engineer"           # Technical deep-dive, full data access


class DecisionType(Enum):
    """Types of decisions that can be explained."""
    ROUTING = "routing"             # Why this route was selected
    SCHEDULING = "scheduling"       # Why this order comes before another
    MACHINE_ASSIGNMENT = "machine"  # Why this machine was chosen
    SUGGESTION = "suggestion"       # Why this suggestion was generated
    BOTTLENECK = "bottleneck"       # Why this is identified as bottleneck
    RISK = "risk"                   # Why this is flagged as risk


@dataclass
class ExplanationElement:
    """A single element of an explanation."""
    type: str  # "fact", "reasoning", "evidence", "trade_off", "alternative"
    content: str
    data: Optional[Dict[str, Any]] = None
    importance: float = 1.0  # 0-1, for filtering by audience


@dataclass
class Explanation:
    """Complete explanation for a decision."""
    decision_type: DecisionType
    summary: str  # One-line summary for quick understanding
    elements: List[ExplanationElement]
    audience_level: AudienceLevel
    confidence: float  # 0-1, how confident is this explanation
    data_sources: List[str]  # What data was used
    
    def to_text(self, max_elements: int = 5) -> str:
        """Convert to human-readable text."""
        lines = [self.summary, ""]
        for elem in self.elements[:max_elements]:
            prefix = {
                "fact": "📊",
                "reasoning": "💡",
                "evidence": "📈",
                "trade_off": "⚖️",
                "alternative": "🔄",
            }.get(elem.type, "•")
            lines.append(f"{prefix} {elem.content}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "decision_type": self.decision_type.value,
            "summary": self.summary,
            "elements": [
                {"type": e.type, "content": e.content, "data": e.data}
                for e in self.elements
            ],
            "audience_level": self.audience_level.value,
            "confidence": self.confidence,
            "data_sources": self.data_sources,
        }


class ExplainabilityEngine:
    """
    Main engine for generating explanations.
    
    Transforms raw APS decisions into natural language explanations
    referencing data and KPIs.
    """
    
    def __init__(self, language: str = "pt-PT"):
        self.language = language
        self._explanation_log: List[Dict[str, Any]] = []
    
    def explain_routing_decision(
        self,
        selected_route: str,
        selected_machine: str,
        alternatives: List[Dict[str, Any]],
        scoring_method: str,
        scores: Dict[str, float],
        context: Dict[str, Any],
        audience: AudienceLevel = AudienceLevel.SUPERVISOR,
    ) -> Explanation:
        """
        Explain why a specific route was selected.
        
        TODO[R&D]: Test explanation fidelity (H4.2).
        """
        elements = []
        
        # Fact: What was decided
        elements.append(ExplanationElement(
            type="fact",
            content=f"Selecionada rota {selected_route} na máquina {selected_machine}.",
            data={"route": selected_route, "machine": selected_machine},
        ))
        
        # Reasoning: Why this route
        best_score = scores.get(selected_route, 0)
        elements.append(ExplanationElement(
            type="reasoning",
            content=f"Método de seleção: {scoring_method}. Score: {best_score:.2f} (menor = melhor).",
            data={"method": scoring_method, "score": best_score},
        ))
        
        # Evidence: Comparison with alternatives
        if alternatives:
            alt_text = ", ".join([
                f"{alt['route']}: {scores.get(alt['route'], '?'):.2f}"
                for alt in alternatives[:3]
            ])
            elements.append(ExplanationElement(
                type="evidence",
                content=f"Alternativas consideradas: {alt_text}.",
                data={"alternatives": alternatives},
            ))
        
        # Trade-off (if applicable)
        if scoring_method == "multi_objective":
            elements.append(ExplanationElement(
                type="trade_off",
                content="Ponderados: tempo de produção, setup, carga da máquina e urgência.",
                importance=0.8,
            ))
        
        # Build summary
        summary = f"Rota {selected_route} ({selected_machine}) selecionada por {scoring_method}."
        
        return Explanation(
            decision_type=DecisionType.ROUTING,
            summary=summary,
            elements=self._filter_by_audience(elements, audience),
            audience_level=audience,
            confidence=0.9,
            data_sources=["routing_df", "machine_loads"],
        )
    
    def explain_bottleneck(
        self,
        machine_id: str,
        load_minutes: float,
        total_ops: int,
        utilization_pct: float,
        comparison_machines: List[Dict[str, Any]],
        audience: AudienceLevel = AudienceLevel.PLANT_MANAGER,
    ) -> Explanation:
        """
        Explain why a machine is identified as bottleneck.
        """
        elements = []
        
        # Fact
        elements.append(ExplanationElement(
            type="fact",
            content=f"{machine_id} identificada como gargalo com {load_minutes/60:.1f}h de carga ({total_ops} operações).",
            data={"machine_id": machine_id, "load_h": load_minutes/60, "ops": total_ops},
        ))
        
        # Evidence
        elements.append(ExplanationElement(
            type="evidence",
            content=f"Utilização: {utilization_pct:.0f}% (>80% = sobrecarga).",
            data={"utilization_pct": utilization_pct},
        ))
        
        # Comparison
        if comparison_machines:
            avg_load = sum(m.get("load_min", 0) for m in comparison_machines) / len(comparison_machines)
            elements.append(ExplanationElement(
                type="evidence",
                content=f"Carga média das outras máquinas: {avg_load/60:.1f}h ({(load_minutes/avg_load - 1)*100:.0f}% acima).",
                data={"avg_load_h": avg_load/60},
            ))
        
        # Reasoning
        elements.append(ExplanationElement(
            type="reasoning",
            content="Máquina com maior tempo total de processamento no plano atual.",
        ))
        
        summary = f"Gargalo: {machine_id} com {utilization_pct:.0f}% de utilização."
        
        return Explanation(
            decision_type=DecisionType.BOTTLENECK,
            summary=summary,
            elements=self._filter_by_audience(elements, audience),
            audience_level=audience,
            confidence=0.95,
            data_sources=["production_plan", "machine_loads"],
        )
    
    def explain_suggestion(
        self,
        suggestion_type: str,
        action: str,
        expected_impact: Dict[str, Any],
        supporting_data: Dict[str, Any],
        audience: AudienceLevel = AudienceLevel.SUPERVISOR,
    ) -> Explanation:
        """
        Explain why a suggestion was generated.
        
        TODO[R&D]: Use for user study (H2.1).
        """
        elements = []
        
        # Fact: What is suggested
        elements.append(ExplanationElement(
            type="fact",
            content=action,
            data={"action": action},
        ))
        
        # Evidence: Why this was identified
        if suggestion_type == "overload_reduction":
            elements.append(ExplanationElement(
                type="evidence",
                content=f"Máquina com {supporting_data.get('utilization_pct', 0):.0f}% de utilização.",
                data=supporting_data,
            ))
        elif suggestion_type == "idle_gap":
            elements.append(ExplanationElement(
                type="evidence",
                content=f"Gap de {supporting_data.get('gap_min', 0)/60:.1f}h identificado.",
                data=supporting_data,
            ))
        elif suggestion_type == "product_risk":
            elements.append(ExplanationElement(
                type="evidence",
                content=f"Espera de {supporting_data.get('wait_min', 0)/60:.1f}h entre operações.",
                data=supporting_data,
            ))
        
        # Impact
        if expected_impact:
            impact_text = ", ".join([
                f"{k}: {v:+.1f}%" if isinstance(v, (int, float)) else f"{k}: {v}"
                for k, v in expected_impact.items()
            ])
            elements.append(ExplanationElement(
                type="reasoning",
                content=f"Impacto esperado: {impact_text}.",
                data=expected_impact,
            ))
        
        # Trade-off
        elements.append(ExplanationElement(
            type="trade_off",
            content="Implementar esta sugestão pode requerer reorganização do plano.",
            importance=0.6,
        ))
        
        summary = f"Sugestão ({suggestion_type}): {action[:50]}..."
        
        return Explanation(
            decision_type=DecisionType.SUGGESTION,
            summary=summary,
            elements=self._filter_by_audience(elements, audience),
            audience_level=audience,
            confidence=0.75,  # Suggestions have inherent uncertainty
            data_sources=["production_plan", "suggestions_engine"],
        )
    
    def explain_schedule_order(
        self,
        earlier_op: Dict[str, Any],
        later_op: Dict[str, Any],
        reasons: List[str],
        audience: AudienceLevel = AudienceLevel.SUPERVISOR,
    ) -> Explanation:
        """
        Explain why one operation is scheduled before another.
        """
        elements = []
        
        # Fact
        elements.append(ExplanationElement(
            type="fact",
            content=f"{earlier_op.get('op_code')} ({earlier_op.get('order_id')}) agendada antes de {later_op.get('op_code')} ({later_op.get('order_id')}).",
        ))
        
        # Reasoning
        for reason in reasons:
            elements.append(ExplanationElement(
                type="reasoning",
                content=reason,
            ))
        
        summary = f"{earlier_op.get('op_code')} antes de {later_op.get('op_code')} por: {reasons[0] if reasons else 'sequência padrão'}."
        
        return Explanation(
            decision_type=DecisionType.SCHEDULING,
            summary=summary,
            elements=self._filter_by_audience(elements, audience),
            audience_level=audience,
            confidence=0.85,
            data_sources=["scheduler", "priority_rules"],
        )
    
    def _filter_by_audience(
        self,
        elements: List[ExplanationElement],
        audience: AudienceLevel,
    ) -> List[ExplanationElement]:
        """Filter explanation elements by audience level."""
        thresholds = {
            AudienceLevel.OPERATOR: 0.9,
            AudienceLevel.SUPERVISOR: 0.6,
            AudienceLevel.PLANT_MANAGER: 0.4,
            AudienceLevel.ENGINEER: 0.0,
        }
        threshold = thresholds.get(audience, 0.5)
        return [e for e in elements if e.importance >= threshold]
    
    def log_explanation(self, explanation: Explanation) -> None:
        """Log explanation for analysis."""
        self._explanation_log.append(explanation.to_dict())
    
    def get_explanation_log(self) -> List[Dict[str, Any]]:
        """Return logged explanations."""
        return self._explanation_log


# ============================================================
# LLM INTEGRATION (TODO[R&D])
# ============================================================

class LLMExplainer:
    """
    Use LLM to enhance explanations with natural language.
    
    TODO[R&D]: Implement LLM-enhanced explanations.
    TODO[R&D]: Measure hallucination rate.
    TODO[R&D]: Compare LLM vs template explanations in user study.
    
    Important: LLM should ONLY explain, never decide.
    The deterministic engine makes decisions; LLM makes them understandable.
    """
    
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
    
    def enhance_explanation(
        self,
        base_explanation: Explanation,
        context: Dict[str, Any],
    ) -> Explanation:
        """
        Use LLM to make explanation more natural and contextual.
        
        TODO[R&D]: Implement with hallucination guardrails.
        """
        # For now, return base explanation unchanged
        return base_explanation
    
    def generate_natural_summary(
        self,
        elements: List[ExplanationElement],
        audience: AudienceLevel,
    ) -> str:
        """
        Generate a natural language summary from elements.
        
        TODO[R&D]: Implement with LLM.
        """
        # Template-based fallback
        return " ".join(e.content for e in elements[:3])


# ============================================================
# EXPERIMENT SUPPORT
# ============================================================

def run_explanation_fidelity_test(
    explanations: List[Explanation],
    ground_truth: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Test whether explanations match actual decision reasons.
    
    TODO[R&D]: Define fidelity metrics and implement.
    
    Experiment E4.2: Explanation fidelity test.
    """
    # Placeholder
    return {
        "fidelity_score": 0.0,
        "n_tested": len(explanations),
        "status": "not_implemented",
    }



