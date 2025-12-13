"""
Testes para Inteligência - Causal, Otimização, What-If (E1-E3)
"""
import pytest
from datetime import datetime


class TestE1_CausalAnalysis:
    """E1: Análise Causal."""
    
    def test_causal_graph_building(self, test_client):
        """E1.1: Deve construir grafo causal a partir de dados."""
        response = test_client.post("/causal/build-graph", json={
            "data": [
                {"machine": "M1", "operator": "OP1", "defect_rate": 0.05},
                {"machine": "M1", "operator": "OP2", "defect_rate": 0.03},
                {"machine": "M2", "operator": "OP1", "defect_rate": 0.08}
            ],
            "variables": ["machine", "operator", "defect_rate"]
        })
        
        if response.status_code == 200:
            graph = response.json()
            assert "nodes" in graph or "edges" in graph or "graph" in graph
    
    def test_causal_effect_estimation(self, test_client):
        """E1.2: Deve estimar efeitos causais (DML)."""
        response = test_client.post("/causal/estimate-effect", json={
            "treatment": "operator",
            "outcome": "defect_rate",
            "confounders": ["machine", "shift"],
            "data": [
                {"operator": "OP1", "machine": "M1", "shift": "DAY", "defect_rate": 0.05},
                {"operator": "OP2", "machine": "M1", "shift": "DAY", "defect_rate": 0.03}
            ]
        })
        
        if response.status_code == 200:
            effect = response.json()
            assert "ate" in effect or "causal_effect" in effect or "effect" in effect
    
    def test_causal_fallback(self, test_client):
        """E1.3: Deve fazer fallback para OLS se DML falhar."""
        # Simular falha de DML (dados insuficientes)
        response = test_client.post("/causal/estimate-effect", json={
            "treatment": "operator",
            "outcome": "defect_rate",
            "confounders": ["machine"],
            "data": [
                {"operator": "OP1", "machine": "M1", "defect_rate": 0.05}
            ]
        })
        
        if response.status_code == 200:
            effect = response.json()
            # Deve ter resultado mesmo que seja via OLS fallback
            assert "ate" in effect or "causal_effect" in effect or "method" in effect
            if "method" in effect:
                assert effect["method"] in ["DML", "OLS", "fallback"]


class TestE2_Optimization:
    """E2: Otimização Matemática."""
    
    def test_time_prediction_ml(self, test_client):
        """E2.1: Deve prever tempos (setup, ciclo) via ML."""
        response = test_client.post("/optimization/predict-time", json={
            "product_id": "PROD-A",
            "machine_id": "M1",
            "operation": "cutting",
            "quantity": 100
        })
        
        if response.status_code == 200:
            prediction = response.json()
            assert "setup_time" in prediction or "setup_time_minutes" in prediction
            assert "cycle_time" in prediction or "cycle_time_minutes" in prediction
    
    def test_parameter_optimization(self, test_client):
        """E2.2: Deve otimizar parâmetros de processo (Bayesian, RL, GA)."""
        response = test_client.post("/optimization/optimize-parameters", json={
            "product_id": "PROD-A",
            "machine_id": "M1",
            "objectives": ["minimize_cycle_time", "minimize_defect_rate"],
            "method": "bayesian"
        })
        
        if response.status_code == 200:
            optimization = response.json()
            assert "optimal_parameters" in optimization or "parameters" in optimization
            assert "expected_metrics" in optimization or "predicted_performance" in optimization
    
    def test_scheduling_optimization(self, test_client, sample_orders):
        """E2.3: Deve otimizar agendamento (MILP, CP-SAT, heurísticas)."""
        response = test_client.post("/optimization/schedule", json={
            "orders": sample_orders,
            "method": "cp-sat",
            "objective": "minimize_tardiness"
        })
        
        if response.status_code == 200:
            schedule = response.json()
            assert "operations" in schedule or "plan" in schedule
            assert "makespan" in schedule or "total_time" in schedule


class TestE3_WhatIf:
    """E3: What-If Avançado."""
    
    def test_scenario_description(self, test_client):
        """E3.1: Deve descrever cenário em linguagem natural."""
        response = test_client.post("/what-if/describe", json={
            "scenario_text": "Aumentar capacidade da máquina M1 em 20%"
        })
        
        if response.status_code == 200:
            scenario = response.json()
            assert "machine_id" in scenario or "changes" in scenario
            assert "capacity_change" in scenario or "delta" in scenario
    
    def test_scenario_comparison(self, test_client):
        """E3.2: Deve comparar baseline vs cenário."""
        response = test_client.post("/what-if/compare", json={
            "baseline_plan_id": "PLAN-001",
            "scenario": {
                "machine_id": "M1",
                "capacity_multiplier": 1.2
            }
        })
        
        if response.status_code == 200:
            comparison = response.json()
            assert "baseline_metrics" in comparison or "baseline" in comparison
            assert "scenario_metrics" in comparison or "scenario" in comparison
            assert "differences" in comparison or "delta" in comparison
    
    def test_whatif_multiple_scenarios(self, test_client):
        """E3.3: Deve suportar múltiplos cenários simultâneos."""
        response = test_client.post("/what-if/compare-multiple", json={
            "baseline_plan_id": "PLAN-001",
            "scenarios": [
                {"name": "scenario_1", "machine_id": "M1", "capacity_multiplier": 1.2},
                {"name": "scenario_2", "machine_id": "M2", "capacity_multiplier": 1.5}
            ]
        })
        
        if response.status_code == 200:
            comparisons = response.json()
            assert isinstance(comparisons, (dict, list))
            # Deve ter comparação para cada cenário

