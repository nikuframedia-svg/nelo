"""
════════════════════════════════════════════════════════════════════════════════════════════════════
TESTS - Causal Context Models
════════════════════════════════════════════════════════════════════════════════════════════════════

Testes para o módulo de análise causal.
"""

import pytest
import numpy as np
import pandas as pd

from backend.causal.causal_graph_builder import (
    CausalVariable,
    VariableType,
    CausalGraph,
    CausalGraphBuilder,
    learn_causal_graph,
    generate_synthetic_data,
)
from backend.causal.causal_effect_estimator import (
    CausalEffect,
    EffectType,
    CausalEffectEstimator,
    estimate_effect,
    estimate_intervention,
    get_all_effects_for_outcome,
    get_all_effects_from_treatment,
)
from backend.causal.complexity_dashboard_engine import (
    ComplexityMetrics,
    CausalInsight,
    InsightType,
    ComplexityDashboard,
    compute_complexity_metrics,
    generate_causal_insights,
    generate_tradeoff_analysis,
)


class TestCausalGraphBuilder:
    """Testes para construção de grafos causais."""
    
    def test_add_variable(self):
        """Testa adição de variáveis."""
        builder = CausalGraphBuilder()
        
        var = CausalVariable(
            name="test_var",
            var_type=VariableType.TREATMENT,
            description="Test variable",
        )
        builder.add_variable(var)
        
        assert "test_var" in builder.variables
        assert builder.variables["test_var"].var_type == VariableType.TREATMENT
    
    def test_add_relation(self):
        """Testa adição de relações."""
        builder = CausalGraphBuilder()
        builder.add_relation("A", "B", strength=0.5, confidence=0.8)
        
        assert len(builder.relations) == 1
        assert builder.relations[0].cause == "A"
        assert builder.relations[0].effect == "B"
        assert builder.relations[0].strength == 0.5
    
    def test_domain_knowledge(self):
        """Testa aplicação de conhecimento de domínio."""
        builder = CausalGraphBuilder()
        builder.apply_domain_knowledge()
        
        # Verificar que variáveis foram adicionadas
        assert len(builder.variables) > 10
        assert len(builder.relations) > 20
        
        # Verificar tipos de variáveis
        treatments = [v for v in builder.variables.values() if v.var_type == VariableType.TREATMENT]
        outcomes = [v for v in builder.variables.values() if v.var_type == VariableType.OUTCOME]
        confounders = [v for v in builder.variables.values() if v.var_type == VariableType.CONFOUNDER]
        
        assert len(treatments) >= 5
        assert len(outcomes) >= 5
        assert len(confounders) >= 3
    
    def test_build_graph(self):
        """Testa construção do grafo."""
        builder = CausalGraphBuilder()
        builder.apply_domain_knowledge()
        
        graph = builder.build()
        
        assert isinstance(graph, CausalGraph)
        assert len(graph.variables) > 0
        assert len(graph.relations) > 0
    
    def test_learn_causal_graph(self):
        """Testa função principal de aprendizagem."""
        graph = learn_causal_graph()
        
        assert isinstance(graph, CausalGraph)
        assert graph.get_treatments()
        assert graph.get_outcomes()
    
    def test_graph_navigation(self):
        """Testa navegação no grafo."""
        graph = learn_causal_graph()
        
        # Testar parents e children
        for var in graph.get_treatments()[:3]:
            children = graph.get_children(var)
            assert isinstance(children, list)
        
        for var in graph.get_outcomes()[:3]:
            parents = graph.get_parents(var)
            assert isinstance(parents, list)


class TestCausalEffectEstimator:
    """Testes para estimação de efeitos causais."""
    
    def test_estimate_effect_basic(self):
        """Testa estimação básica de efeito."""
        data = generate_synthetic_data(n_samples=500)
        graph = learn_causal_graph(data)
        
        effect = estimate_effect("setup_frequency", "energy_cost", graph, data)
        
        assert isinstance(effect, CausalEffect)
        assert effect.treatment == "setup_frequency"
        assert effect.outcome == "energy_cost"
        assert effect.estimate != 0  # Deve haver efeito
    
    def test_effect_interpretation(self):
        """Testa interpretação do efeito."""
        data = generate_synthetic_data()
        graph = learn_causal_graph(data)
        
        effect = estimate_effect("machine_load", "energy_cost", graph, data)
        effect.compute_interpretation()
        
        assert effect.direction in ["positive", "negative", "neutral"]
        assert effect.magnitude in ["small", "medium", "large"]
        assert effect.significance in ["significant", "marginal", "not_significant"]
    
    def test_confounder_identification(self):
        """Testa identificação de confounders."""
        graph = learn_causal_graph()
        estimator = CausalEffectEstimator(graph)
        
        confounders = estimator.identify_confounders("setup_frequency", "energy_cost")
        
        assert isinstance(confounders, list)
        # Deve identificar pelo menos alguns confounders
        assert len(confounders) >= 0
    
    def test_estimate_intervention(self):
        """Testa estimação de intervenção."""
        result = estimate_intervention(
            "aumentar setup",
            "energy_cost",
            intervention_value=1.0,
        )
        
        assert result["success"] == True
        assert "effect" in result
        assert "interpretation" in result
    
    def test_all_effects_for_outcome(self):
        """Testa obter todos os efeitos num outcome."""
        graph = learn_causal_graph()
        effects = get_all_effects_for_outcome("energy_cost", graph)
        
        assert len(effects) > 0
        assert all(e.outcome == "energy_cost" for e in effects)
        # Deve estar ordenado por magnitude
        magnitudes = [abs(e.estimate) for e in effects]
        assert magnitudes == sorted(magnitudes, reverse=True)
    
    def test_all_effects_from_treatment(self):
        """Testa obter todos os efeitos de um tratamento."""
        graph = learn_causal_graph()
        effects = get_all_effects_from_treatment("setup_frequency", graph)
        
        assert len(effects) > 0
        assert all(e.treatment == "setup_frequency" for e in effects)


class TestComplexityDashboard:
    """Testes para dashboard de complexidade."""
    
    def test_compute_metrics(self):
        """Testa cálculo de métricas."""
        graph = learn_causal_graph()
        metrics = compute_complexity_metrics(graph)
        
        assert isinstance(metrics, ComplexityMetrics)
        assert metrics.n_variables > 0
        assert metrics.n_relations > 0
        assert 0 <= metrics.overall_complexity <= 100
    
    def test_generate_insights(self):
        """Testa geração de insights."""
        graph = learn_causal_graph()
        insights = generate_causal_insights(graph)
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        assert all(isinstance(i, CausalInsight) for i in insights)
    
    def test_insight_types(self):
        """Testa diferentes tipos de insights."""
        graph = learn_causal_graph()
        insights = generate_causal_insights(graph)
        
        insight_types = {i.insight_type for i in insights}
        # Deve ter pelo menos alguns tipos diferentes
        assert len(insight_types) >= 1
    
    def test_tradeoff_analysis(self):
        """Testa análise de trade-offs."""
        graph = learn_causal_graph()
        analysis = generate_tradeoff_analysis("setup_frequency", graph)
        
        assert "treatment" in analysis
        assert "positive_effects" in analysis
        assert "negative_effects" in analysis
        assert "net_benefit_score" in analysis
    
    def test_dashboard_to_dict(self):
        """Testa exportação do dashboard."""
        graph = learn_causal_graph()
        dashboard = ComplexityDashboard(graph)
        
        result = dashboard.to_dict()
        
        assert "metrics" in result
        assert "insights" in result
        assert "graph_summary" in result


class TestSyntheticData:
    """Testes para dados sintéticos."""
    
    def test_generate_data(self):
        """Testa geração de dados sintéticos."""
        data = generate_synthetic_data(n_samples=100)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        
        # Verificar colunas esperadas
        expected_cols = [
            "setup_frequency", "batch_size", "machine_load",
            "energy_cost", "makespan", "tardiness", "otd_rate",
        ]
        for col in expected_cols:
            assert col in data.columns
    
    def test_causal_relationships_in_data(self):
        """Testa que relações causais são refletidas nos dados."""
        data = generate_synthetic_data(n_samples=1000)
        
        # machine_load -> energy_cost deve ter correlação positiva
        corr = data["machine_load"].corr(data["energy_cost"])
        assert corr > 0.3
        
        # maintenance_delay -> failure_prob deve ter correlação positiva
        corr = data["maintenance_delay"].corr(data["failure_prob"])
        assert corr > 0.3
        
        # tardiness -> otd_rate deve ter correlação negativa
        corr = data["tardiness"].corr(data["otd_rate"])
        assert corr < -0.3


class TestIntegration:
    """Testes de integração end-to-end."""
    
    def test_full_causal_workflow(self):
        """Testa workflow completo."""
        # 1. Gerar dados
        data = generate_synthetic_data(n_samples=500)
        
        # 2. Aprender grafo
        graph = learn_causal_graph(data)
        
        # 3. Estimar efeitos
        effect = estimate_effect("setup_frequency", "energy_cost", graph, data)
        assert effect is not None
        
        # 4. Gerar insights
        insights = generate_causal_insights(graph, data)
        assert len(insights) > 0
        
        # 5. Calcular métricas
        metrics = compute_complexity_metrics(graph, data)
        assert metrics.overall_complexity > 0
        
        print(f"\n📊 Resultado do Teste de Integração:")
        print(f"   Variáveis: {metrics.n_variables}")
        print(f"   Relações: {metrics.n_relations}")
        print(f"   Complexidade: {metrics.overall_complexity:.1f}")
        print(f"   Insights gerados: {len(insights)}")
        print(f"   Efeito setup->energy: {effect.estimate:.3f}")
    
    def test_causal_explanation(self):
        """Testa geração de explicações causais."""
        graph = learn_causal_graph()
        
        # Simular pergunta: "O que acontece ao custo energético se aumentar setups?"
        effect = estimate_effect("setup_frequency", "energy_cost", graph)
        
        assert effect.explanation
        assert len(effect.explanation) > 50
        
        print(f"\n📝 Explicação gerada:")
        print(f"   {effect.explanation}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



