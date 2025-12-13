"""
════════════════════════════════════════════════════════════════════════════════════════════════════
CAUSAL GRAPH BUILDER - Construção de Grafos Causais
════════════════════════════════════════════════════════════════════════════════════════════════════

Usa inferência causal para construir e aprender estruturas de grafos dirigidos acíclicos (DAG)
que representam relações causais entre variáveis de decisão e outcomes.

Bibliotecas suportadas:
- pgmpy (estrutura de DAG e inferência)
- DoWhy (identificação e estimação causal)
- EconML (Machine Learning causal - CATE)

Variáveis:
- X (Treatment): decisões de scheduling (policy, sequência, carga, setups)
- Y (Outcome): outputs (makespan, tardiness, energia, desgaste, acidentes, stress)
- Z (Confounder/Context): sazonalidade, procura, mix de produto
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any, Set
import json
import numpy as np
import pandas as pd


class VariableType(Enum):
    """Tipos de variáveis no grafo causal."""
    TREATMENT = "treatment"         # Decisões (X) - scheduling choices
    OUTCOME = "outcome"             # Resultados (Y) - KPIs
    CONFOUNDER = "confounder"       # Confounders (Z) - contexto
    MEDIATOR = "mediator"           # Mediadores - mecanismos intermediários
    INSTRUMENT = "instrument"       # Variáveis instrumentais


@dataclass
class CausalVariable:
    """
    Representa uma variável no grafo causal.
    """
    name: str
    var_type: VariableType
    description: str
    unit: str = ""
    domain: str = ""  # "continuous", "binary", "categorical", "count"
    
    # Metadados
    source: str = ""  # De onde vem o dado
    importance: float = 1.0  # Peso relativo
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.var_type.value,
            "description": self.description,
            "unit": self.unit,
            "domain": self.domain,
            "importance": self.importance,
        }


@dataclass
class CausalRelation:
    """
    Representa uma relação causal entre duas variáveis.
    """
    cause: str          # Nome da variável causa
    effect: str         # Nome da variável efeito
    strength: float     # Força estimada da relação (-1 a 1)
    confidence: float   # Confiança na relação (0-1)
    
    # Tipo de relação
    is_direct: bool = True
    mechanism: str = ""  # Descrição do mecanismo causal
    
    # Evidência
    evidence_score: float = 0.0
    data_points: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "strength": round(self.strength, 3),
            "confidence": round(self.confidence, 2),
            "is_direct": self.is_direct,
            "mechanism": self.mechanism,
            "evidence_score": round(self.evidence_score, 2),
        }


@dataclass
class CausalGraph:
    """
    Grafo causal completo (DAG).
    """
    variables: Dict[str, CausalVariable]
    relations: List[CausalRelation]
    
    # Metadados
    created_at: datetime = field(default_factory=datetime.now)
    data_source: str = ""
    n_observations: int = 0
    
    def get_parents(self, var_name: str) -> List[str]:
        """Retorna variáveis que causam diretamente var_name."""
        return [r.cause for r in self.relations if r.effect == var_name and r.is_direct]
    
    def get_children(self, var_name: str) -> List[str]:
        """Retorna variáveis diretamente causadas por var_name."""
        return [r.effect for r in self.relations if r.cause == var_name and r.is_direct]
    
    def get_ancestors(self, var_name: str, visited: Optional[Set] = None) -> Set[str]:
        """Retorna todos os ancestrais (causas diretas e indiretas)."""
        if visited is None:
            visited = set()
        parents = self.get_parents(var_name)
        for parent in parents:
            if parent not in visited:
                visited.add(parent)
                self.get_ancestors(parent, visited)
        return visited
    
    def get_treatments(self) -> List[str]:
        """Retorna variáveis de tratamento."""
        return [v.name for v in self.variables.values() if v.var_type == VariableType.TREATMENT]
    
    def get_outcomes(self) -> List[str]:
        """Retorna variáveis de outcome."""
        return [v.name for v in self.variables.values() if v.var_type == VariableType.OUTCOME]
    
    def get_confounders(self) -> List[str]:
        """Retorna confounders."""
        return [v.name for v in self.variables.values() if v.var_type == VariableType.CONFOUNDER]
    
    def to_dict(self) -> Dict:
        return {
            "variables": {k: v.to_dict() for k, v in self.variables.items()},
            "relations": [r.to_dict() for r in self.relations],
            "n_variables": len(self.variables),
            "n_relations": len(self.relations),
            "treatments": self.get_treatments(),
            "outcomes": self.get_outcomes(),
            "confounders": self.get_confounders(),
            "created_at": self.created_at.isoformat(),
            "n_observations": self.n_observations,
        }
    
    def to_adjacency_matrix(self) -> Tuple[pd.DataFrame, List[str]]:
        """Converte para matriz de adjacência."""
        var_names = list(self.variables.keys())
        n = len(var_names)
        adj = np.zeros((n, n))
        
        for rel in self.relations:
            if rel.cause in var_names and rel.effect in var_names:
                i = var_names.index(rel.cause)
                j = var_names.index(rel.effect)
                adj[i, j] = rel.strength
        
        return pd.DataFrame(adj, index=var_names, columns=var_names), var_names


class CausalGraphBuilder:
    """
    Construtor de grafos causais.
    
    Pode aprender estrutura a partir de dados ou usar conhecimento de domínio.
    """
    
    def __init__(self):
        self.variables: Dict[str, CausalVariable] = {}
        self.relations: List[CausalRelation] = []
        self._domain_knowledge_applied = False
    
    def add_variable(self, var: CausalVariable) -> 'CausalGraphBuilder':
        """Adiciona uma variável."""
        self.variables[var.name] = var
        return self
    
    def add_relation(self, cause: str, effect: str, 
                     strength: float = 0.0, confidence: float = 0.5,
                     mechanism: str = "") -> 'CausalGraphBuilder':
        """Adiciona uma relação causal."""
        rel = CausalRelation(
            cause=cause,
            effect=effect,
            strength=strength,
            confidence=confidence,
            mechanism=mechanism,
        )
        self.relations.append(rel)
        return self
    
    def apply_domain_knowledge(self) -> 'CausalGraphBuilder':
        """
        Aplica conhecimento de domínio sobre scheduling industrial.
        """
        if self._domain_knowledge_applied:
            return self
        
        # ═══════════════════════════════════════════════════════════════════
        # VARIÁVEIS DE DECISÃO (TREATMENT)
        # ═══════════════════════════════════════════════════════════════════
        
        self.add_variable(CausalVariable(
            name="setup_frequency",
            var_type=VariableType.TREATMENT,
            description="Frequência de setups/changeovers por turno",
            unit="setups/turno",
            domain="count",
        ))
        
        self.add_variable(CausalVariable(
            name="batch_size",
            var_type=VariableType.TREATMENT,
            description="Tamanho médio dos lotes de produção",
            unit="unidades",
            domain="continuous",
        ))
        
        self.add_variable(CausalVariable(
            name="machine_load",
            var_type=VariableType.TREATMENT,
            description="Carga média das máquinas",
            unit="%",
            domain="continuous",
        ))
        
        self.add_variable(CausalVariable(
            name="night_shifts",
            var_type=VariableType.TREATMENT,
            description="Número de turnos noturnos por semana",
            unit="turnos",
            domain="count",
        ))
        
        self.add_variable(CausalVariable(
            name="overtime_hours",
            var_type=VariableType.TREATMENT,
            description="Horas extra por semana",
            unit="horas",
            domain="continuous",
        ))
        
        self.add_variable(CausalVariable(
            name="maintenance_delay",
            var_type=VariableType.TREATMENT,
            description="Dias de adiamento de manutenção preventiva",
            unit="dias",
            domain="count",
        ))
        
        self.add_variable(CausalVariable(
            name="priority_changes",
            var_type=VariableType.TREATMENT,
            description="Alterações de prioridade de ordens por dia",
            unit="alterações/dia",
            domain="count",
        ))
        
        # ═══════════════════════════════════════════════════════════════════
        # VARIÁVEIS DE OUTCOME
        # ═══════════════════════════════════════════════════════════════════
        
        self.add_variable(CausalVariable(
            name="energy_cost",
            var_type=VariableType.OUTCOME,
            description="Custo energético total",
            unit="€/semana",
            domain="continuous",
        ))
        
        self.add_variable(CausalVariable(
            name="makespan",
            var_type=VariableType.OUTCOME,
            description="Tempo total de produção",
            unit="horas",
            domain="continuous",
        ))
        
        self.add_variable(CausalVariable(
            name="tardiness",
            var_type=VariableType.OUTCOME,
            description="Atraso total em entregas",
            unit="horas",
            domain="continuous",
        ))
        
        self.add_variable(CausalVariable(
            name="otd_rate",
            var_type=VariableType.OUTCOME,
            description="Taxa de On-Time Delivery",
            unit="%",
            domain="continuous",
        ))
        
        self.add_variable(CausalVariable(
            name="machine_wear",
            var_type=VariableType.OUTCOME,
            description="Desgaste acumulado das máquinas",
            unit="score 0-100",
            domain="continuous",
        ))
        
        self.add_variable(CausalVariable(
            name="failure_prob",
            var_type=VariableType.OUTCOME,
            description="Probabilidade de falha nas próximas 48h",
            unit="%",
            domain="continuous",
        ))
        
        self.add_variable(CausalVariable(
            name="operator_stress",
            var_type=VariableType.OUTCOME,
            description="Índice de stress dos operadores",
            unit="score 0-100",
            domain="continuous",
        ))
        
        self.add_variable(CausalVariable(
            name="quality_defects",
            var_type=VariableType.OUTCOME,
            description="Taxa de defeitos de qualidade",
            unit="%",
            domain="continuous",
        ))
        
        self.add_variable(CausalVariable(
            name="production_stability",
            var_type=VariableType.OUTCOME,
            description="Estabilidade do plano de produção",
            unit="score 0-100",
            domain="continuous",
        ))
        
        # ═══════════════════════════════════════════════════════════════════
        # VARIÁVEIS DE CONTEXTO (CONFOUNDER)
        # ═══════════════════════════════════════════════════════════════════
        
        self.add_variable(CausalVariable(
            name="demand_volume",
            var_type=VariableType.CONFOUNDER,
            description="Volume de procura semanal",
            unit="unidades",
            domain="continuous",
        ))
        
        self.add_variable(CausalVariable(
            name="product_mix",
            var_type=VariableType.CONFOUNDER,
            description="Diversidade de produtos (número de SKUs activos)",
            unit="SKUs",
            domain="count",
        ))
        
        self.add_variable(CausalVariable(
            name="seasonality",
            var_type=VariableType.CONFOUNDER,
            description="Índice de sazonalidade",
            unit="score 0-1",
            domain="continuous",
        ))
        
        self.add_variable(CausalVariable(
            name="machine_age",
            var_type=VariableType.CONFOUNDER,
            description="Idade média do equipamento",
            unit="anos",
            domain="continuous",
        ))
        
        self.add_variable(CausalVariable(
            name="workforce_experience",
            var_type=VariableType.CONFOUNDER,
            description="Experiência média dos operadores",
            unit="anos",
            domain="continuous",
        ))
        
        # ═══════════════════════════════════════════════════════════════════
        # RELAÇÕES CAUSAIS (baseadas em conhecimento de domínio)
        # ═══════════════════════════════════════════════════════════════════
        
        # Setup frequency effects
        self.add_relation("setup_frequency", "energy_cost", 
                         strength=0.35, confidence=0.85,
                         mechanism="Cada setup consome energia adicional para aquecimento e estabilização")
        
        self.add_relation("setup_frequency", "makespan",
                         strength=0.25, confidence=0.80,
                         mechanism="Setups adicionam tempo não-produtivo")
        
        self.add_relation("setup_frequency", "machine_wear",
                         strength=0.20, confidence=0.70,
                         mechanism="Ciclos térmicos e mecânicos aceleram desgaste")
        
        self.add_relation("setup_frequency", "quality_defects",
                         strength=-0.15, confidence=0.65,
                         mechanism="Lotes menores permitem detectar defeitos mais cedo")
        
        # Batch size effects
        self.add_relation("batch_size", "setup_frequency",
                         strength=-0.60, confidence=0.90,
                         mechanism="Lotes maiores reduzem necessidade de changeovers")
        
        self.add_relation("batch_size", "production_stability",
                         strength=0.30, confidence=0.75,
                         mechanism="Lotes maiores reduzem variabilidade no plano")
        
        self.add_relation("batch_size", "tardiness",
                         strength=0.20, confidence=0.70,
                         mechanism="Lotes grandes podem atrasar outras encomendas")
        
        # Machine load effects
        self.add_relation("machine_load", "energy_cost",
                         strength=0.50, confidence=0.90,
                         mechanism="Maior utilização = maior consumo energético")
        
        self.add_relation("machine_load", "machine_wear",
                         strength=0.45, confidence=0.85,
                         mechanism="Uso intensivo acelera degradação")
        
        self.add_relation("machine_load", "failure_prob",
                         strength=0.40, confidence=0.80,
                         mechanism="Sobrecarga aumenta probabilidade de falha")
        
        self.add_relation("machine_load", "makespan",
                         strength=-0.35, confidence=0.85,
                         mechanism="Maior utilização reduz tempo total")
        
        # Night shifts effects
        self.add_relation("night_shifts", "operator_stress",
                         strength=0.55, confidence=0.90,
                         mechanism="Turnos noturnos perturbam ritmo circadiano")
        
        self.add_relation("night_shifts", "quality_defects",
                         strength=0.25, confidence=0.75,
                         mechanism="Fadiga aumenta erros humanos")
        
        self.add_relation("night_shifts", "energy_cost",
                         strength=0.15, confidence=0.70,
                         mechanism="Tarifas energéticas podem variar")
        
        self.add_relation("night_shifts", "makespan",
                         strength=-0.40, confidence=0.85,
                         mechanism="Mais horas produtivas reduzem tempo total")
        
        # Overtime effects
        self.add_relation("overtime_hours", "operator_stress",
                         strength=0.45, confidence=0.88,
                         mechanism="Horas extra aumentam fadiga e burnout")
        
        self.add_relation("overtime_hours", "energy_cost",
                         strength=0.20, confidence=0.80,
                         mechanism="Energia adicional consumida")
        
        self.add_relation("overtime_hours", "otd_rate",
                         strength=0.35, confidence=0.85,
                         mechanism="Capacidade extra permite cumprir prazos")
        
        # Maintenance delay effects
        self.add_relation("maintenance_delay", "failure_prob",
                         strength=0.65, confidence=0.92,
                         mechanism="Manutenção adiada aumenta risco de avaria")
        
        self.add_relation("maintenance_delay", "machine_wear",
                         strength=0.50, confidence=0.88,
                         mechanism="Sem manutenção, desgaste acumula mais rápido")
        
        self.add_relation("maintenance_delay", "quality_defects",
                         strength=0.30, confidence=0.75,
                         mechanism="Equipamento degradado produz mais defeitos")
        
        self.add_relation("maintenance_delay", "makespan",
                         strength=-0.10, confidence=0.70,
                         mechanism="Evitar paragens de manutenção a curto prazo")
        
        # Priority changes effects
        self.add_relation("priority_changes", "production_stability",
                         strength=-0.55, confidence=0.85,
                         mechanism="Mudanças frequentes desestabilizam o plano")
        
        self.add_relation("priority_changes", "operator_stress",
                         strength=0.30, confidence=0.78,
                         mechanism="Incerteza e mudanças aumentam stress")
        
        self.add_relation("priority_changes", "setup_frequency",
                         strength=0.25, confidence=0.72,
                         mechanism="Reordenações podem exigir setups adicionais")
        
        # Confounder relationships
        self.add_relation("demand_volume", "machine_load",
                         strength=0.60, confidence=0.90,
                         mechanism="Mais procura = maior utilização")
        
        self.add_relation("demand_volume", "overtime_hours",
                         strength=0.45, confidence=0.85,
                         mechanism="Picos de procura exigem horas extra")
        
        self.add_relation("product_mix", "setup_frequency",
                         strength=0.50, confidence=0.88,
                         mechanism="Mais SKUs = mais changeovers")
        
        self.add_relation("machine_age", "failure_prob",
                         strength=0.55, confidence=0.90,
                         mechanism="Equipamento mais antigo falha mais")
        
        self.add_relation("machine_age", "energy_cost",
                         strength=0.25, confidence=0.75,
                         mechanism="Equipamento antigo é menos eficiente")
        
        self.add_relation("workforce_experience", "quality_defects",
                         strength=-0.40, confidence=0.80,
                         mechanism="Operadores experientes cometem menos erros")
        
        self.add_relation("workforce_experience", "operator_stress",
                         strength=-0.30, confidence=0.75,
                         mechanism="Experiência ajuda a lidar com pressão")
        
        # Outcome to outcome (mediators)
        self.add_relation("machine_wear", "failure_prob",
                         strength=0.70, confidence=0.92,
                         mechanism="Desgaste aumenta probabilidade de falha")
        
        self.add_relation("operator_stress", "quality_defects",
                         strength=0.35, confidence=0.80,
                         mechanism="Stress leva a mais erros")
        
        self.add_relation("quality_defects", "tardiness",
                         strength=0.30, confidence=0.75,
                         mechanism="Defeitos requerem retrabalho, causando atrasos")
        
        self.add_relation("failure_prob", "tardiness",
                         strength=0.45, confidence=0.85,
                         mechanism="Falhas causam paragens e atrasos")
        
        self.add_relation("tardiness", "otd_rate",
                         strength=-0.80, confidence=0.95,
                         mechanism="Atrasos reduzem entregas a tempo")
        
        self._domain_knowledge_applied = True
        return self
    
    def learn_from_data(self, data: pd.DataFrame, 
                        method: str = "constraint") -> 'CausalGraphBuilder':
        """
        Aprende estrutura causal a partir de dados.
        
        TODO[R&D]: Implementar com:
        - pgmpy PC algorithm (constraint-based)
        - NOTEARS (score-based, continuous optimization)
        - DoWhy causal discovery
        
        Args:
            data: DataFrame com observações
            method: "constraint", "score", "hybrid"
        """
        # Por agora, usar conhecimento de domínio e refinar com correlações
        if not self._domain_knowledge_applied:
            self.apply_domain_knowledge()
        
        # Calcular correlações para refinar strengths
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for rel in self.relations:
            if rel.cause in numeric_cols and rel.effect in numeric_cols:
                try:
                    corr = data[rel.cause].corr(data[rel.effect])
                    if not np.isnan(corr):
                        # Ajustar strength baseado em evidência
                        rel.strength = (rel.strength + corr) / 2
                        rel.data_points = len(data)
                        rel.evidence_score = abs(corr) * 0.5 + rel.confidence * 0.5
                except Exception:
                    pass
        
        return self
    
    def build(self, n_observations: int = 0) -> CausalGraph:
        """Constrói o grafo causal final."""
        return CausalGraph(
            variables=self.variables.copy(),
            relations=self.relations.copy(),
            n_observations=n_observations,
        )


def learn_causal_graph(historical_data: Optional[pd.DataFrame] = None,
                       use_domain_knowledge: bool = True) -> CausalGraph:
    """
    Função principal para aprender um grafo causal.
    
    Args:
        historical_data: DataFrame com dados históricos (opcional)
        use_domain_knowledge: Se deve aplicar conhecimento de domínio
    
    Returns:
        CausalGraph aprendido
    """
    builder = CausalGraphBuilder()
    
    if use_domain_knowledge:
        builder.apply_domain_knowledge()
    
    if historical_data is not None and not historical_data.empty:
        builder.learn_from_data(historical_data)
        n_obs = len(historical_data)
    else:
        n_obs = 0
    
    return builder.build(n_observations=n_obs)


def generate_synthetic_data(n_samples: int = 1000, 
                           noise_level: float = 0.1) -> pd.DataFrame:
    """
    Gera dados sintéticos para testar o módulo causal.
    
    Simula relações causais conhecidas com ruído.
    """
    np.random.seed(42)
    
    # Confounders
    demand_volume = np.random.normal(1000, 200, n_samples)
    product_mix = np.random.poisson(15, n_samples)
    seasonality = 0.5 + 0.5 * np.sin(np.linspace(0, 4*np.pi, n_samples))
    machine_age = np.random.uniform(1, 15, n_samples)
    workforce_experience = np.random.uniform(1, 20, n_samples)
    
    # Treatments (influenciados por confounders)
    setup_frequency = 5 + 0.3 * product_mix + np.random.normal(0, 2, n_samples)
    batch_size = 100 - 2 * setup_frequency + np.random.normal(0, 10, n_samples)
    machine_load = 50 + 0.03 * demand_volume + np.random.normal(0, 10, n_samples)
    night_shifts = np.clip(np.floor(demand_volume / 500), 0, 5)
    overtime_hours = np.clip(demand_volume / 100 - 8 + np.random.normal(0, 2, n_samples), 0, 20)
    maintenance_delay = np.random.poisson(3, n_samples)
    priority_changes = np.random.poisson(5, n_samples)
    
    # Outcomes (determinados por treatments e confounders)
    energy_cost = (
        500 + 
        35 * setup_frequency + 
        0.5 * machine_load * 10 + 
        15 * night_shifts +
        25 * machine_age +
        np.random.normal(0, 100, n_samples)
    )
    
    machine_wear = (
        20 + 
        2 * setup_frequency + 
        0.4 * machine_load + 
        5 * maintenance_delay +
        2 * machine_age +
        np.random.normal(0, 10, n_samples)
    )
    
    failure_prob = np.clip(
        5 + 
        0.7 * machine_wear +
        6 * maintenance_delay +
        0.5 * machine_load / 10 +
        np.random.normal(0, 5, n_samples),
        0, 100
    )
    
    operator_stress = np.clip(
        20 +
        5 * night_shifts +
        2 * overtime_hours +
        3 * priority_changes -
        workforce_experience +
        np.random.normal(0, 10, n_samples),
        0, 100
    )
    
    quality_defects = np.clip(
        2 +
        0.03 * operator_stress +
        0.02 * machine_wear +
        0.2 * night_shifts -
        0.1 * workforce_experience +
        np.random.normal(0, 1, n_samples),
        0, 20
    )
    
    makespan = (
        100 +
        2.5 * setup_frequency -
        0.35 * machine_load -
        4 * night_shifts +
        0.3 * quality_defects * 10 +  # Retrabalho
        np.random.normal(0, 10, n_samples)
    )
    
    tardiness = np.clip(
        -10 +
        0.1 * makespan +
        3 * priority_changes +
        0.5 * failure_prob +
        2 * quality_defects +
        np.random.normal(0, 5, n_samples),
        0, 100
    )
    
    otd_rate = np.clip(
        100 - 0.8 * tardiness + np.random.normal(0, 3, n_samples),
        0, 100
    )
    
    production_stability = np.clip(
        80 -
        5 * priority_changes +
        0.3 * batch_size / 10 -
        0.2 * failure_prob +
        np.random.normal(0, 8, n_samples),
        0, 100
    )
    
    return pd.DataFrame({
        # Confounders
        "demand_volume": demand_volume,
        "product_mix": product_mix,
        "seasonality": seasonality,
        "machine_age": machine_age,
        "workforce_experience": workforce_experience,
        # Treatments
        "setup_frequency": setup_frequency,
        "batch_size": batch_size,
        "machine_load": machine_load,
        "night_shifts": night_shifts,
        "overtime_hours": overtime_hours,
        "maintenance_delay": maintenance_delay,
        "priority_changes": priority_changes,
        # Outcomes
        "energy_cost": energy_cost,
        "makespan": makespan,
        "tardiness": tardiness,
        "otd_rate": otd_rate,
        "machine_wear": machine_wear,
        "failure_prob": failure_prob,
        "operator_stress": operator_stress,
        "quality_defects": quality_defects,
        "production_stability": production_stability,
    })



