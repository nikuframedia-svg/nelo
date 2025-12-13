"""
════════════════════════════════════════════════════════════════════════════════════════════════════
COMPLEXITY DASHBOARD ENGINE - Motor de Análise de Complexidade Causal
════════════════════════════════════════════════════════════════════════════════════════════════════

Analisa a complexidade do sistema causal e gera insights sobre trade-offs.

Features:
- Métricas de complexidade do grafo causal
- Identificação de interações não-lineares
- Geração de insights em linguagem natural
- Trade-off analysis entre objetivos
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd

from .causal_graph_builder import CausalGraph, VariableType, learn_causal_graph
from .causal_effect_estimator import (
    CausalEffect,
    CausalEffectEstimator,
    estimate_effect,
    get_all_effects_for_outcome,
    get_all_effects_from_treatment,
)


class InsightType(Enum):
    """Tipos de insight causal."""
    TRADEOFF = "tradeoff"               # Trade-off entre objetivos
    LEVERAGE = "leverage"               # Ponto de alavancagem (alto impacto)
    RISK = "risk"                       # Risco identificado
    OPPORTUNITY = "opportunity"         # Oportunidade de melhoria
    INTERACTION = "interaction"         # Interação não-óbvia
    BOTTLENECK = "bottleneck"           # Bottleneck causal
    RESILIENCE = "resilience"           # Factor de resiliência


@dataclass
class CausalInsight:
    """
    Um insight derivado da análise causal.
    """
    insight_type: InsightType
    title: str
    description: str
    priority: str  # "high", "medium", "low"
    
    # Variáveis envolvidas
    treatments: List[str]
    outcomes: List[str]
    
    # Evidência
    confidence: float  # 0-1
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    # Ações sugeridas
    suggested_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "type": self.insight_type.value,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "treatments": self.treatments,
            "outcomes": self.outcomes,
            "confidence": round(self.confidence, 2),
            "evidence": self.evidence,
            "suggested_actions": self.suggested_actions,
        }


@dataclass
class ComplexityMetrics:
    """
    Métricas de complexidade do sistema causal.
    """
    # Estrutura do grafo
    n_variables: int = 0
    n_relations: int = 0
    n_treatments: int = 0
    n_outcomes: int = 0
    n_confounders: int = 0
    
    # Métricas de rede
    density: float = 0.0                    # n_edges / n_possible_edges
    avg_in_degree: float = 0.0              # Média de pais por variável
    avg_out_degree: float = 0.0             # Média de filhos por variável
    max_path_length: int = 0                # Caminho mais longo
    
    # Métricas causais
    n_significant_relations: int = 0        # Relações com p < 0.05
    avg_effect_strength: float = 0.0        # Força média dos efeitos
    effect_variance: float = 0.0            # Variância dos efeitos
    
    # Complexidade
    nonlinearity_score: float = 0.0         # Score de não-linearidade
    interaction_score: float = 0.0          # Score de interações
    overall_complexity: float = 0.0         # Score global de complexidade
    
    def to_dict(self) -> Dict:
        return {
            "structure": {
                "n_variables": self.n_variables,
                "n_relations": self.n_relations,
                "n_treatments": self.n_treatments,
                "n_outcomes": self.n_outcomes,
                "n_confounders": self.n_confounders,
            },
            "network": {
                "density": round(self.density, 3),
                "avg_in_degree": round(self.avg_in_degree, 2),
                "avg_out_degree": round(self.avg_out_degree, 2),
                "max_path_length": self.max_path_length,
            },
            "causal": {
                "n_significant_relations": self.n_significant_relations,
                "avg_effect_strength": round(self.avg_effect_strength, 3),
                "effect_variance": round(self.effect_variance, 3),
            },
            "complexity": {
                "nonlinearity_score": round(self.nonlinearity_score, 2),
                "interaction_score": round(self.interaction_score, 2),
                "overall_complexity": round(self.overall_complexity, 2),
            }
        }


class ComplexityDashboard:
    """
    Dashboard de análise de complexidade causal.
    """
    
    def __init__(self, 
                 causal_graph: Optional[CausalGraph] = None,
                 data: Optional[pd.DataFrame] = None):
        """
        Inicializa o dashboard.
        
        Args:
            causal_graph: Grafo causal
            data: Dados observacionais
        """
        self.graph = causal_graph or learn_causal_graph(data)
        self.data = data
        self._metrics: Optional[ComplexityMetrics] = None
        self._insights: List[CausalInsight] = []
    
    def compute_metrics(self) -> ComplexityMetrics:
        """Calcula métricas de complexidade."""
        metrics = ComplexityMetrics()
        
        # Estrutura do grafo
        metrics.n_variables = len(self.graph.variables)
        metrics.n_relations = len(self.graph.relations)
        metrics.n_treatments = len(self.graph.get_treatments())
        metrics.n_outcomes = len(self.graph.get_outcomes())
        metrics.n_confounders = len(self.graph.get_confounders())
        
        # Métricas de rede
        if metrics.n_variables > 1:
            max_edges = metrics.n_variables * (metrics.n_variables - 1)
            metrics.density = metrics.n_relations / max_edges if max_edges > 0 else 0
        
        # In-degree e out-degree
        in_degrees = {}
        out_degrees = {}
        for var in self.graph.variables:
            in_degrees[var] = len(self.graph.get_parents(var))
            out_degrees[var] = len(self.graph.get_children(var))
        
        if in_degrees:
            metrics.avg_in_degree = np.mean(list(in_degrees.values()))
            metrics.avg_out_degree = np.mean(list(out_degrees.values()))
        
        # Força média dos efeitos
        strengths = [abs(r.strength) for r in self.graph.relations]
        if strengths:
            metrics.avg_effect_strength = np.mean(strengths)
            metrics.effect_variance = np.var(strengths)
        
        # Relações significativas (strength > 0.2)
        metrics.n_significant_relations = sum(1 for r in self.graph.relations if abs(r.strength) > 0.2)
        
        # Score de não-linearidade (estimativa baseada em variância)
        metrics.nonlinearity_score = min(100, metrics.effect_variance * 500)
        
        # Score de interações (baseado em densidade e graus)
        metrics.interaction_score = min(100, metrics.density * 100 + metrics.avg_out_degree * 10)
        
        # Complexidade global
        metrics.overall_complexity = (
            0.3 * min(100, metrics.n_relations * 3) +
            0.2 * metrics.nonlinearity_score +
            0.2 * metrics.interaction_score +
            0.3 * min(100, metrics.n_significant_relations * 5)
        )
        
        self._metrics = metrics
        return metrics
    
    def generate_insights(self) -> List[CausalInsight]:
        """Gera insights a partir da análise causal."""
        insights = []
        
        # Assegurar que temos métricas
        if self._metrics is None:
            self.compute_metrics()
        
        # 1. Identificar trade-offs
        insights.extend(self._find_tradeoffs())
        
        # 2. Identificar pontos de alavancagem
        insights.extend(self._find_leverage_points())
        
        # 3. Identificar riscos
        insights.extend(self._find_risks())
        
        # 4. Identificar oportunidades
        insights.extend(self._find_opportunities())
        
        # 5. Identificar interações
        insights.extend(self._find_interactions())
        
        # Ordenar por prioridade
        priority_order = {"high": 0, "medium": 1, "low": 2}
        insights.sort(key=lambda i: (priority_order.get(i.priority, 3), -i.confidence))
        
        self._insights = insights
        return insights
    
    def _find_tradeoffs(self) -> List[CausalInsight]:
        """Identifica trade-offs entre outcomes."""
        insights = []
        
        treatments = self.graph.get_treatments()
        outcomes = self.graph.get_outcomes()
        
        for treatment in treatments:
            effects = get_all_effects_from_treatment(treatment, self.graph, self.data)
            
            # Procurar efeitos opostos em diferentes outcomes
            positive_outcomes = [e for e in effects if e.estimate > 0.1 and e.significance != "not_significant"]
            negative_outcomes = [e for e in effects if e.estimate < -0.1 and e.significance != "not_significant"]
            
            if positive_outcomes and negative_outcomes:
                treatment_desc = self.graph.variables.get(treatment, treatment)
                if hasattr(treatment_desc, 'description'):
                    treatment_desc = treatment_desc.description
                
                pos_names = [self._get_outcome_name(e.outcome) for e in positive_outcomes[:2]]
                neg_names = [self._get_outcome_name(e.outcome) for e in negative_outcomes[:2]]
                
                # Calcular confiança baseada em p_values
                p_values = [e.p_value for e in effects if hasattr(e, 'p_value')]
                avg_confidence = 1 - np.mean(p_values) if p_values else 0.5
                
                insight = CausalInsight(
                    insight_type=InsightType.TRADEOFF,
                    title=f"Trade-off identificado: {treatment_desc}",
                    description=(
                        f"Aumentar '{treatment_desc}' melhora {', '.join(pos_names)} "
                        f"mas piora {', '.join(neg_names)}. "
                        f"É necessário balancear estes objetivos."
                    ),
                    priority="high" if len(positive_outcomes) > 1 and len(negative_outcomes) > 1 else "medium",
                    treatments=[treatment],
                    outcomes=[e.outcome for e in positive_outcomes + negative_outcomes],
                    confidence=avg_confidence,
                    evidence={
                        "positive_effects": [e.to_dict() for e in positive_outcomes[:2]],
                        "negative_effects": [e.to_dict() for e in negative_outcomes[:2]],
                    },
                    suggested_actions=[
                        f"Definir prioridades claras entre os objetivos em conflito",
                        f"Considerar abordagem multi-objetivo para optimização",
                    ]
                )
                insights.append(insight)
        
        return insights
    
    def _find_leverage_points(self) -> List[CausalInsight]:
        """Identifica pontos de alavancagem de alto impacto."""
        insights = []
        
        treatments = self.graph.get_treatments()
        
        for treatment in treatments:
            effects = get_all_effects_from_treatment(treatment, self.graph, self.data)
            
            # Contar outcomes significativamente afetados
            significant_effects = [e for e in effects if abs(e.estimate) > 0.2]
            
            if len(significant_effects) >= 3:
                treatment_desc = self._get_treatment_name(treatment)
                
                # Calcular confiança baseada em p_values
                p_values = [e.p_value for e in significant_effects if hasattr(e, 'p_value')]
                avg_confidence = 1 - np.mean(p_values) if p_values else 0.5
                
                insight = CausalInsight(
                    insight_type=InsightType.LEVERAGE,
                    title=f"Ponto de alavancagem: {treatment_desc}",
                    description=(
                        f"'{treatment_desc}' tem impacto significativo em {len(significant_effects)} "
                        f"outcomes diferentes. Pequenas mudanças nesta variável podem ter efeitos amplificados."
                    ),
                    priority="high",
                    treatments=[treatment],
                    outcomes=[e.outcome for e in significant_effects],
                    confidence=avg_confidence,
                    evidence={
                        "n_affected_outcomes": len(significant_effects),
                        "avg_effect_strength": np.mean([abs(e.estimate) for e in significant_effects]),
                    },
                    suggested_actions=[
                        f"Monitorizar de perto alterações em '{treatment_desc}'",
                        f"Avaliar cuidadosamente antes de fazer mudanças",
                    ]
                )
                insights.append(insight)
        
        return insights
    
    def _find_risks(self) -> List[CausalInsight]:
        """Identifica riscos baseados em relações causais."""
        insights = []
        
        # Riscos relacionados com falhas
        for rel in self.graph.relations:
            if rel.effect == "failure_prob" and rel.strength > 0.4:
                cause_desc = self._get_treatment_name(rel.cause)
                
                insight = CausalInsight(
                    insight_type=InsightType.RISK,
                    title=f"Risco de falha: {cause_desc}",
                    description=(
                        f"'{cause_desc}' tem um impacto forte (+{rel.strength*100:.0f}%) "
                        f"na probabilidade de falhas de máquinas. Controlar esta variável é crítico."
                    ),
                    priority="high",
                    treatments=[rel.cause],
                    outcomes=["failure_prob"],
                    confidence=rel.confidence,
                    evidence={"effect_strength": rel.strength},
                    suggested_actions=[
                        f"Implementar monitorização contínua de '{cause_desc}'",
                        f"Definir limites de alerta e acção",
                    ]
                )
                insights.append(insight)
        
        # Riscos relacionados com stress dos operadores
        for rel in self.graph.relations:
            if rel.effect == "operator_stress" and rel.strength > 0.3:
                cause_desc = self._get_treatment_name(rel.cause)
                
                insight = CausalInsight(
                    insight_type=InsightType.RISK,
                    title=f"Risco de stress: {cause_desc}",
                    description=(
                        f"'{cause_desc}' contribui significativamente (+{rel.strength*100:.0f}%) "
                        f"para o stress dos operadores. Considerar o factor humano."
                    ),
                    priority="medium",
                    treatments=[rel.cause],
                    outcomes=["operator_stress"],
                    confidence=rel.confidence,
                    evidence={"effect_strength": rel.strength},
                    suggested_actions=[
                        f"Balancear produtividade com bem-estar dos colaboradores",
                        f"Consultar equipa antes de alterações",
                    ]
                )
                insights.append(insight)
        
        return insights
    
    def _find_opportunities(self) -> List[CausalInsight]:
        """Identifica oportunidades de melhoria."""
        insights = []
        
        # Oportunidades de redução de custo
        for rel in self.graph.relations:
            if rel.effect == "energy_cost" and rel.strength > 0.3:
                cause_desc = self._get_treatment_name(rel.cause)
                
                insight = CausalInsight(
                    insight_type=InsightType.OPPORTUNITY,
                    title=f"Oportunidade de poupança energética",
                    description=(
                        f"Reduzir '{cause_desc}' pode diminuir significativamente (-{rel.strength*100:.0f}%) "
                        f"os custos energéticos."
                    ),
                    priority="medium",
                    treatments=[rel.cause],
                    outcomes=["energy_cost"],
                    confidence=rel.confidence,
                    evidence={"potential_reduction_pct": rel.strength * 100},
                    suggested_actions=[
                        f"Avaliar possibilidade de optimizar '{cause_desc}'",
                        f"Calcular ROI de intervenções",
                    ]
                )
                insights.append(insight)
        
        # Oportunidades de melhoria de OTD
        for rel in self.graph.relations:
            if rel.effect == "otd_rate" and rel.strength > 0.2:
                cause_desc = self._get_treatment_name(rel.cause)
                
                insight = CausalInsight(
                    insight_type=InsightType.OPPORTUNITY,
                    title=f"Oportunidade de melhoria OTD",
                    description=(
                        f"Optimizar '{cause_desc}' pode melhorar a taxa de entregas a tempo "
                        f"em até {rel.strength*100:.0f}%."
                    ),
                    priority="medium",
                    treatments=[rel.cause],
                    outcomes=["otd_rate"],
                    confidence=rel.confidence,
                    evidence={"potential_improvement_pct": rel.strength * 100},
                    suggested_actions=[
                        f"Analisar impacto detalhado de alterações em '{cause_desc}'",
                    ]
                )
                insights.append(insight)
        
        return insights
    
    def _find_interactions(self) -> List[CausalInsight]:
        """Identifica interações não-óbvias entre variáveis."""
        insights = []
        
        # Procurar caminhos indirectos
        treatments = self.graph.get_treatments()
        outcomes = self.graph.get_outcomes()
        
        for treatment in treatments[:3]:  # Limitar para não gerar demasiados
            for outcome in outcomes[:3]:
                # Verificar se existe caminho indirecto
                direct_effect = None
                for rel in self.graph.relations:
                    if rel.cause == treatment and rel.effect == outcome:
                        direct_effect = rel.strength
                        break
                
                if direct_effect is None or abs(direct_effect) < 0.1:
                    # Procurar caminho via mediador
                    for mediator_rel in self.graph.relations:
                        if mediator_rel.cause == treatment:
                            mediator = mediator_rel.effect
                            for final_rel in self.graph.relations:
                                if final_rel.cause == mediator and final_rel.effect == outcome:
                                    indirect_effect = mediator_rel.strength * final_rel.strength
                                    if abs(indirect_effect) > 0.1:
                                        treatment_desc = self._get_treatment_name(treatment)
                                        mediator_desc = self._get_outcome_name(mediator)
                                        outcome_desc = self._get_outcome_name(outcome)
                                        
                                        insight = CausalInsight(
                                            insight_type=InsightType.INTERACTION,
                                            title=f"Efeito indirecto via {mediator_desc}",
                                            description=(
                                                f"'{treatment_desc}' afecta '{outcome_desc}' indirectamente "
                                                f"através de '{mediator_desc}'. Este caminho causal pode não ser óbvio."
                                            ),
                                            priority="low",
                                            treatments=[treatment],
                                            outcomes=[mediator, outcome],
                                            confidence=min(mediator_rel.confidence, final_rel.confidence) * 0.8,
                                            evidence={
                                                "path": [treatment, mediator, outcome],
                                                "indirect_effect": indirect_effect,
                                            },
                                            suggested_actions=[
                                                f"Considerar efeitos em '{mediator_desc}' ao decidir sobre '{treatment_desc}'",
                                            ]
                                        )
                                        insights.append(insight)
        
        return insights[:5]  # Limitar número de interações
    
    def _get_treatment_name(self, var_name: str) -> str:
        """Obtém nome legível de um tratamento."""
        if var_name in self.graph.variables:
            return self.graph.variables[var_name].description
        return var_name.replace("_", " ")
    
    def _get_outcome_name(self, var_name: str) -> str:
        """Obtém nome legível de um outcome."""
        if var_name in self.graph.variables:
            return self.graph.variables[var_name].description
        return var_name.replace("_", " ")
    
    def to_dict(self) -> Dict:
        """Exporta dashboard completo para dicionário."""
        if self._metrics is None:
            self.compute_metrics()
        if not self._insights:
            self.generate_insights()
        
        return {
            "metrics": self._metrics.to_dict(),
            "insights": [i.to_dict() for i in self._insights],
            "graph_summary": self.graph.to_dict(),
        }


def compute_complexity_metrics(
    causal_graph: Optional[CausalGraph] = None,
    data: Optional[pd.DataFrame] = None,
) -> ComplexityMetrics:
    """
    Função principal para calcular métricas de complexidade.
    """
    dashboard = ComplexityDashboard(causal_graph, data)
    return dashboard.compute_metrics()


def generate_causal_insights(
    causal_graph: Optional[CausalGraph] = None,
    data: Optional[pd.DataFrame] = None,
) -> List[CausalInsight]:
    """
    Função principal para gerar insights causais.
    """
    dashboard = ComplexityDashboard(causal_graph, data)
    return dashboard.generate_insights()


def generate_tradeoff_analysis(
    treatment: str,
    causal_graph: Optional[CausalGraph] = None,
    data: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Análise detalhada de trade-offs para um tratamento específico.
    """
    if causal_graph is None:
        causal_graph = learn_causal_graph(data)
    
    effects = get_all_effects_from_treatment(treatment, causal_graph, data)
    
    positive = [e for e in effects if e.estimate > 0]
    negative = [e for e in effects if e.estimate < 0]
    
    treatment_desc = causal_graph.variables[treatment].description if treatment in causal_graph.variables else treatment
    
    return {
        "treatment": treatment,
        "treatment_description": treatment_desc,
        "positive_effects": [e.to_dict() for e in sorted(positive, key=lambda x: -x.estimate)],
        "negative_effects": [e.to_dict() for e in sorted(negative, key=lambda x: x.estimate)],
        "net_benefit_score": sum(e.estimate for e in positive) + sum(e.estimate for e in negative),
        "recommendation": (
            "Benefício líquido positivo - considerar aumentar" 
            if sum(e.estimate for e in positive) > abs(sum(e.estimate for e in negative))
            else "Trade-offs significativos - avaliar cuidadosamente"
        ),
    }

