"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    REPORT GENERATOR — LLM-Powered Report Generation
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Generates natural language reports using LLM for:
- Executive summaries
- Technical explanations
- Scenario comparisons
- Algorithm descriptions
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import logging

from .comparison_engine import ScenarioComparison, ComparisonMetrics

logger = logging.getLogger(__name__)


@dataclass
class ExecutiveReport:
    """Executive summary report."""
    title: str
    summary: str
    key_findings: List[str]
    comparison_table: Dict[str, Any]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    conclusion: str
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "key_findings": self.key_findings,
            "comparison_table": self.comparison_table,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendations": self.recommendations,
            "conclusion": self.conclusion,
            "generated_at": self.generated_at.isoformat(),
        }
    
    def to_markdown(self) -> str:
        """Convert report to markdown format."""
        md = f"# {self.title}\n\n"
        md += f"*Gerado em: {self.generated_at.strftime('%d/%m/%Y %H:%M')}*\n\n"
        
        md += "## Resumo Executivo\n\n"
        md += f"{self.summary}\n\n"
        
        md += "## Principais Conclusões\n\n"
        for i, finding in enumerate(self.key_findings, 1):
            md += f"{i}. {finding}\n"
        md += "\n"
        
        md += "## Comparação de Métricas\n\n"
        md += "| Métrica | Plano Base | Novo Cenário | Diferença |\n"
        md += "|---------|------------|--------------|----------|\n"
        for metric, values in self.comparison_table.items():
            base = values.get("baseline", "-")
            scen = values.get("scenario", "-")
            delta = values.get("delta", "-")
            md += f"| {metric} | {base} | {scen} | {delta} |\n"
        md += "\n"
        
        if self.strengths:
            md += "## Pontos Fortes do Novo Cenário\n\n"
            for s in self.strengths:
                md += f"✅ {s}\n"
            md += "\n"
        
        if self.weaknesses:
            md += "## Pontos de Atenção\n\n"
            for w in self.weaknesses:
                md += f"⚠️ {w}\n"
            md += "\n"
        
        if self.recommendations:
            md += "## Recomendações\n\n"
            for r in self.recommendations:
                md += f"💡 {r}\n"
            md += "\n"
        
        md += "## Conclusão\n\n"
        md += f"{self.conclusion}\n"
        
        return md


@dataclass
class TechnicalReport:
    """Technical explanation report."""
    title: str
    algorithm_name: str
    algorithm_description: str
    objective_function: str
    constraints: List[str]
    parameters: Dict[str, str]
    examples: List[str]
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "algorithm_name": self.algorithm_name,
            "algorithm_description": self.algorithm_description,
            "objective_function": self.objective_function,
            "constraints": self.constraints,
            "parameters": self.parameters,
            "examples": self.examples,
            "generated_at": self.generated_at.isoformat(),
        }
    
    def to_markdown(self) -> str:
        md = f"# {self.title}\n\n"
        md += f"*Documentação técnica gerada em: {self.generated_at.strftime('%d/%m/%Y %H:%M')}*\n\n"
        
        md += f"## Algoritmo: {self.algorithm_name}\n\n"
        md += f"{self.algorithm_description}\n\n"
        
        md += "## Função Objetivo\n\n"
        md += f"```\n{self.objective_function}\n```\n\n"
        
        md += "## Restrições\n\n"
        for c in self.constraints:
            md += f"- {c}\n"
        md += "\n"
        
        md += "## Parâmetros\n\n"
        for param, desc in self.parameters.items():
            md += f"- **{param}**: {desc}\n"
        md += "\n"
        
        if self.examples:
            md += "## Exemplos Práticos\n\n"
            for i, ex in enumerate(self.examples, 1):
                md += f"### Exemplo {i}\n{ex}\n\n"
        
        return md


class ReportGenerator:
    """
    Generates reports using LLM.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize report generator.
        
        Args:
            llm_client: Optional LLM client (uses OpenAI if not provided)
        """
        self.llm_client = llm_client
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM for text generation."""
        try:
            from ..openai_client import ask_openai
            return ask_openai(system_prompt, user_prompt)
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return ""
    
    def generate_executive_report(
        self,
        comparison: ScenarioComparison,
        context: Optional[str] = None,
    ) -> ExecutiveReport:
        """
        Generate executive summary report from scenario comparison.
        
        Args:
            comparison: ScenarioComparison object
            context: Optional additional context about the scenarios
            
        Returns:
            ExecutiveReport
        """
        # Build comparison table
        comparison_table = {}
        
        metric_labels = {
            "makespan_hours": "Makespan (horas)",
            "lead_time_avg_days": "Lead Time Médio (dias)",
            "throughput_units_per_week": "Throughput Semanal (unid.)",
            "orders_late": "Ordens Atrasadas",
            "otd_pct": "OTD (%)",
            "total_tardiness_hours": "Atrasos Totais (horas)",
            "total_setup_hours": "Setup Total (horas)",
            "avg_utilization_pct": "Utilização Média (%)",
            "bottleneck_utilization": "Utilização Gargalo (%)",
        }
        
        for name, delta in comparison.deltas.items():
            label = metric_labels.get(name, name)
            comparison_table[label] = {
                "baseline": f"{delta.baseline_value:.1f}",
                "scenario": f"{delta.scenario_value:.1f}",
                "delta": f"{'+' if delta.absolute_delta > 0 else ''}{delta.percent_delta:.1f}%",
            }
        
        # Generate summary with LLM
        summary = self._generate_summary_llm(comparison, context)
        conclusion = self._generate_conclusion_llm(comparison)
        
        # Key findings
        key_findings = []
        
        # Add most significant changes
        significant_deltas = sorted(
            comparison.deltas.items(),
            key=lambda x: abs(x[1].percent_delta),
            reverse=True
        )[:5]
        
        for name, delta in significant_deltas:
            label = metric_labels.get(name, name)
            direction = "melhorou" if delta.is_improvement else "piorou"
            key_findings.append(
                f"{label} {direction} {abs(delta.percent_delta):.1f}% "
                f"({delta.baseline_value:.1f} → {delta.scenario_value:.1f})"
            )
        
        return ExecutiveReport(
            title=f"Relatório Comparativo: {comparison.baseline.scenario_name} vs {comparison.scenario.scenario_name}",
            summary=summary,
            key_findings=key_findings,
            comparison_table=comparison_table,
            strengths=comparison.strengths,
            weaknesses=comparison.weaknesses,
            recommendations=comparison.recommendations,
            conclusion=conclusion,
        )
    
    def _generate_summary_llm(
        self,
        comparison: ScenarioComparison,
        context: Optional[str],
    ) -> str:
        """Generate executive summary using LLM."""
        system_prompt = """És um especialista em planeamento de produção industrial em Portugal.
Gera resumos executivos claros e concisos em Português de Portugal.
Usa linguagem profissional mas acessível. Foca nos impactos de negócio."""
        
        data = {
            "baseline": comparison.baseline.scenario_name,
            "scenario": comparison.scenario.scenario_name,
            "description": comparison.scenario.scenario_description,
            "overall_improvement": comparison.overall_improvement,
            "improvement_score": comparison.improvement_score,
            "key_changes": [
                f"{name}: {d.baseline_value:.1f} → {d.scenario_value:.1f} ({'+' if d.absolute_delta > 0 else ''}{d.percent_delta:.1f}%)"
                for name, d in list(comparison.deltas.items())[:5]
            ],
            "strengths_count": len(comparison.strengths),
            "weaknesses_count": len(comparison.weaknesses),
        }
        
        user_prompt = f"""Gera um resumo executivo (3-4 frases) comparando dois cenários de planeamento:

Dados:
{json.dumps(data, indent=2, ensure_ascii=False)}

{f'Contexto adicional: {context}' if context else ''}

O resumo deve:
1. Identificar a mudança principal entre cenários
2. Quantificar o impacto mais significativo
3. Dar uma recomendação clara (adotar ou não o novo cenário)"""
        
        summary = self._call_llm(system_prompt, user_prompt)
        
        if not summary:
            # Fallback summary
            if comparison.overall_improvement:
                summary = (
                    f"O {comparison.scenario.scenario_name} apresenta melhorias significativas em relação ao "
                    f"{comparison.baseline.scenario_name}. O score de melhoria global é de "
                    f"{comparison.improvement_score:.1f} pontos, com destaque para "
                    f"{len(comparison.strengths)} pontos fortes identificados. "
                    f"Recomenda-se a adoção do novo cenário."
                )
            else:
                summary = (
                    f"A análise comparativa entre {comparison.baseline.scenario_name} e "
                    f"{comparison.scenario.scenario_name} revela que o cenário atual é preferível. "
                    f"Foram identificados {len(comparison.weaknesses)} pontos de atenção no novo cenário. "
                    f"Recomenda-se manter o plano base ou rever as alterações propostas."
                )
        
        return summary
    
    def _generate_conclusion_llm(self, comparison: ScenarioComparison) -> str:
        """Generate conclusion using LLM."""
        system_prompt = """És um especialista em planeamento de produção industrial.
Gera conclusões objetivas e acionáveis em Português de Portugal."""
        
        user_prompt = f"""Com base nesta comparação de cenários, gera uma conclusão final (2-3 frases):

Melhoria global: {'Sim' if comparison.overall_improvement else 'Não'}
Score: {comparison.improvement_score:.1f}
Pontos fortes: {len(comparison.strengths)}
Pontos fracos: {len(comparison.weaknesses)}
Principais melhorias: {', '.join(comparison.strengths[:2]) if comparison.strengths else 'Nenhuma significativa'}
Principais problemas: {', '.join(comparison.weaknesses[:2]) if comparison.weaknesses else 'Nenhum significativo'}

A conclusão deve dar uma recomendação clara e justificada."""
        
        conclusion = self._call_llm(system_prompt, user_prompt)
        
        if not conclusion:
            if comparison.overall_improvement:
                conclusion = (
                    f"Em conclusão, o novo cenário representa uma melhoria substancial no planeamento. "
                    f"Recomenda-se a sua implementação, monitorizando os pontos de atenção identificados."
                )
            else:
                conclusion = (
                    f"Em conclusão, o cenário proposto não apresenta melhorias suficientes para justificar "
                    f"a sua adoção. Recomenda-se rever os parâmetros ou manter o plano atual."
                )
        
        return conclusion
    
    def generate_technical_explanation(
        self,
        algorithm: str = "dispatching",
        machine_id: Optional[str] = None,
        include_examples: bool = True,
    ) -> TechnicalReport:
        """
        Generate technical explanation of scheduling algorithms.
        
        Args:
            algorithm: Algorithm type (dispatching, flow_shop, setup_optimization)
            machine_id: Specific machine to explain (optional)
            include_examples: Include practical examples
            
        Returns:
            TechnicalReport
        """
        if algorithm == "dispatching":
            return self._explain_dispatching(machine_id, include_examples)
        elif algorithm == "flow_shop":
            return self._explain_flow_shop(include_examples)
        elif algorithm == "setup_optimization":
            return self._explain_setup_optimization(include_examples)
        else:
            return self._explain_dispatching(machine_id, include_examples)
    
    def _explain_dispatching(
        self,
        machine_id: Optional[str],
        include_examples: bool,
    ) -> TechnicalReport:
        """Explain dispatching rules."""
        description = """O motor de planeamento utiliza regras de dispatching para sequenciar operações 
em cada máquina. Estas regras determinam a ordem em que as operações são processadas quando 
múltiplas operações estão disponíveis para execução.

As regras disponíveis são:

**EDD (Earliest Due Date)** - Prioriza operações cujas encomendas têm a data de entrega mais próxima.
Esta regra minimiza o atraso máximo e é ideal quando cumprir prazos é crítico.

**SPT (Shortest Processing Time)** - Prioriza operações com menor tempo de processamento.
Minimiza o tempo médio de espera na fila e maximiza o throughput a curto prazo.

**FIFO (First In First Out)** - Processa operações na ordem de chegada.
Garante justiça temporal mas não otimiza nenhuma métrica específica.

**CR (Critical Ratio)** - Calcula a razão entre tempo restante até a data de entrega e tempo de 
processamento restante. Valores menores indicam maior urgência.

**WSPT (Weighted SPT)** - SPT ponderado pela prioridade da encomenda. 
Equilibra urgência com importância estratégica das encomendas."""
        
        objective = """Minimizar F(π) = α·Σ T_j + β·Σ C_j + γ·Σ setup(j-1, j)

Onde:
- T_j = max(0, C_j - d_j) é o atraso da ordem j
- C_j é o tempo de conclusão da ordem j
- d_j é a data de entrega da ordem j
- setup(i,j) é o tempo de setup entre ordens consecutivas
- α, β, γ são pesos configuráveis"""
        
        constraints = [
            "Cada operação só pode ser processada após a operação precedente estar completa",
            "Cada máquina só pode processar uma operação de cada vez",
            "Tempos de setup dependem da sequência (família de produto anterior)",
            "Datas de disponibilidade de material são respeitadas",
            "Capacidade de máquina é finita (turnos e manutenções)",
        ]
        
        parameters = {
            "Regra de Dispatching": "EDD, SPT, FIFO, CR ou WSPT",
            "Horizonte de Planeamento": "Período temporal considerado (ex: 2 semanas)",
            "Agrupamento por Setup": "Se ativo, agrupa operações da mesma família",
            "Pesos de Prioridade": "Multiplicadores por encomenda/cliente",
        }
        
        examples = []
        if include_examples:
            examples = [
                """**Situação**: 3 operações aguardam na máquina M-101:
- Op1: Duração 30min, Due date em 2h
- Op2: Duração 60min, Due date em 3h
- Op3: Duração 15min, Due date em 4h

**Com EDD**: Ordem = Op1, Op2, Op3 (por proximidade da due date)
**Com SPT**: Ordem = Op3, Op1, Op2 (por duração)

Resultado: EDD minimiza atrasos; SPT maximiza throughput imediato.""",
                
                """**Situação**: Encomendas com prioridades diferentes:
- Enc1: Prioridade Alta (peso=2), due date longe
- Enc2: Prioridade Normal (peso=1), due date próxima

**Com EDD**: Enc2 primeiro (due date)
**Com WSPT**: Enc1 pode subir na fila devido ao peso

Resultado: WSPT equilibra urgência comercial com datas.""",
            ]
        
        return TechnicalReport(
            title=f"Critérios de Sequenciamento{f' - Máquina {machine_id}' if machine_id else ''}",
            algorithm_name="Regras de Dispatching",
            algorithm_description=description,
            objective_function=objective,
            constraints=constraints,
            parameters=parameters,
            examples=examples,
        )
    
    def _explain_flow_shop(self, include_examples: bool) -> TechnicalReport:
        """Explain flow shop scheduling."""
        description = """O planeamento encadeado (flow shop) sincroniza a sequência de operações 
através de múltiplas máquinas em cadeia. Ao contrário do planeamento independente, onde cada 
máquina otimiza localmente, o flow shop considera o fluxo global para minimizar o tempo total 
(makespan) e garantir fluxo contínuo.

O algoritmo utiliza:

1. **Heurística NEH** (Nawaz-Enscore-Ham) para construção inicial:
   - Ordena jobs por tempo total de processamento (decrescente)
   - Insere cada job na posição que minimiza makespan parcial

2. **Melhoria Local 2-opt**:
   - Troca pares de posições e avalia impacto
   - Repete até não haver melhorias

3. **CP-SAT** (opcional) para solução ótima:
   - Modelo de programação por restrições
   - Garante solução ótima para instâncias pequenas/médias"""
        
        objective = """Minimizar C_max = max_j {C_{j,m}}

Onde:
- C_max é o makespan (tempo de conclusão do último job na última máquina)
- C_{j,m} é o tempo de conclusão do job j na máquina m (última da cadeia)
- j ∈ J é o conjunto de jobs
- m = último estágio da cadeia"""
        
        constraints = [
            "Precedência intra-job: S_{j,k+1} ≥ C_{j,k} + buffer_{k,k+1}",
            "Não-sobreposição: Cada máquina processa um job de cada vez",
            "Buffer entre estágios: Tempo mínimo de transferência/arrefecimento",
            "Sequência permutação: Mesma ordem de jobs em todas as máquinas (opcional)",
        ]
        
        parameters = {
            "Cadeia de Máquinas": "Sequência ordenada de máquinas (ex: M-101 → M-102 → M-103)",
            "Buffer Default": "Tempo de transferência entre máquinas (default: 30 min)",
            "Buffers Específicos": "Buffers customizados por par de máquinas",
            "Solver": "HEURISTIC (rápido) ou CPSAT (ótimo)",
        }
        
        examples = []
        if include_examples:
            examples = [
                """**Situação**: Cadeia M-101 → M-102 → M-103 com 3 jobs
Buffer entre máquinas: 30 minutos

Job1: [20, 30, 25] min por máquina
Job2: [35, 20, 30] min
Job3: [25, 25, 20] min

**Resultado NEH**: Sequência Job2-Job1-Job3
Makespan: 195 min (vs 210 min com FIFO)

Melhoria: 7% de redução no makespan.""",
            ]
        
        return TechnicalReport(
            title="Planeamento Encadeado (Flow Shop)",
            algorithm_name="Flow Shop Scheduling com NEH + Local Search",
            algorithm_description=description,
            objective_function=objective,
            constraints=constraints,
            parameters=parameters,
            examples=examples,
        )
    
    def _explain_setup_optimization(self, include_examples: bool) -> TechnicalReport:
        """Explain setup time optimization."""
        description = """A otimização de tempos de setup minimiza o tempo gasto em trocas/preparações 
entre operações consecutivas. Quando produtos de famílias diferentes são processados em sequência, 
é necessário tempo de setup (limpeza, ajustes, troca de ferramentas).

O algoritmo funciona como um TSP (Traveling Salesman Problem):
- Cada "cidade" é uma família de produto
- A "distância" é o tempo de setup entre famílias
- Objetivo: encontrar a sequência que minimiza a soma dos setups

Estratégia de otimização:
1. **Greedy Nearest Neighbor**: Começa pela operação mais urgente, depois escolhe sempre 
   a próxima com menor setup
2. **2-opt**: Melhora local trocando pares de posições
3. **Algoritmo Genético**: Para instâncias grandes (>20 operações)"""
        
        objective = """Minimizar Σ setup(f(π_i), f(π_{i+1}))

Sujeito a:
- T_j ≤ T_max para todas as ordens (limite de atraso)

Onde:
- π é a sequência de operações
- f(op) é a família de setup da operação
- setup(A, B) é o tempo de setup da família A para B
- T_j é o atraso da ordem j"""
        
        constraints = [
            "Respeito às datas de entrega (soft constraint com penalização)",
            "Limite máximo de atraso por ordem",
            "Manutenção da precedência operacional",
        ]
        
        parameters = {
            "Matriz de Setup": "Tempos de setup entre todas as famílias de produto",
            "Max Tardiness": "Atraso máximo permitido por ordem (minutos)",
            "Algoritmo": "greedy, greedy_2opt ou genetic",
            "Respeitar Due Dates": "Se True, penaliza sequências que causam atrasos",
        }
        
        examples = []
        if include_examples:
            examples = [
                """**Situação**: 5 operações de 3 famílias (A, B, C)
Matriz de Setup:
       A    B    C
  A    0   15   30
  B   20    0   10
  C   25   15    0

Sequência FIFO: A-B-A-C-B → Setup total: 15+20+30+15 = 80 min
Sequência Otimizada: A-A-B-C-B → Setup total: 0+20+10+15 = 45 min

Poupança: 35 minutos (44% redução)"""
            ]
        
        return TechnicalReport(
            title="Otimização de Tempos de Setup",
            algorithm_name="Setup Minimization (TSP-like)",
            algorithm_description=description,
            objective_function=objective,
            constraints=constraints,
            parameters=parameters,
            examples=examples,
        )
    
    def generate_scenario_summary(
        self,
        scenario: ComparisonMetrics,
        changes_description: str,
        previous_scenario: Optional[ComparisonMetrics] = None,
    ) -> str:
        """
        Generate a narrative summary of a scenario.
        
        Args:
            scenario: Current scenario metrics
            changes_description: Description of what changed
            previous_scenario: Previous scenario for context (optional)
            
        Returns:
            Narrative summary text
        """
        system_prompt = """És um especialista em planeamento de produção industrial em Portugal.
Gera resumos narrativos claros em Português de Portugal, explicando cenários de 
planeamento de forma compreensível para gestores de produção."""
        
        data = {
            "nome": scenario.scenario_name,
            "descricao_mudancas": changes_description,
            "makespan_h": scenario.makespan_hours,
            "throughput_semanal": scenario.throughput_units_per_week,
            "lead_time_medio_dias": scenario.lead_time_avg_days,
            "ordens_atrasadas": scenario.orders_late,
            "otd_pct": scenario.otd_pct,
            "utilizacao_media": scenario.avg_utilization_pct,
            "gargalo": scenario.bottleneck_machine,
            "utilizacao_gargalo": scenario.bottleneck_utilization,
            "num_maquinas": scenario.num_machines,
            "num_ordens": scenario.num_orders,
        }
        
        if previous_scenario:
            data["anterior"] = {
                "makespan_h": previous_scenario.makespan_hours,
                "ordens_atrasadas": previous_scenario.orders_late,
                "utilizacao_media": previous_scenario.avg_utilization_pct,
            }
        
        user_prompt = f"""Resume o seguinte cenário de planeamento em linguagem clara (4-5 frases):

Dados do Cenário:
{json.dumps(data, indent=2, ensure_ascii=False)}

O resumo deve:
1. Descrever as mudanças feitas (máquinas, turnos, etc.)
2. Explicar o objetivo dessas mudanças
3. Apresentar os principais resultados (throughput, prazos, utilização)
4. Identificar áreas críticas ou de sucesso"""
        
        summary = self._call_llm(system_prompt, user_prompt)
        
        if not summary:
            # Fallback
            summary = f"""O cenário "{scenario.scenario_name}" {changes_description}. 

Os principais resultados são:
- Makespan de {scenario.makespan_hours:.0f} horas para completar todas as ordens
- Throughput semanal de {scenario.throughput_units_per_week:.0f} unidades
- Lead time médio de {scenario.lead_time_avg_days:.1f} dias
- {scenario.orders_late} ordens com atraso (OTD: {scenario.otd_pct:.0f}%)

A máquina {scenario.bottleneck_machine} é o gargalo atual com {scenario.bottleneck_utilization:.0f}% de utilização."""
        
        return summary


def generate_executive_report(
    comparison: ScenarioComparison,
    context: Optional[str] = None,
) -> ExecutiveReport:
    """
    Convenience function to generate executive report.
    """
    generator = ReportGenerator()
    return generator.generate_executive_report(comparison, context)


def generate_technical_explanation(
    algorithm: str = "dispatching",
    machine_id: Optional[str] = None,
) -> TechnicalReport:
    """
    Convenience function to generate technical explanation.
    """
    generator = ReportGenerator()
    return generator.generate_technical_explanation(algorithm, machine_id)


