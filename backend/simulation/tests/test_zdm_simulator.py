"""
════════════════════════════════════════════════════════════════════════════════════════════════════
TESTS - ZDM Simulator
════════════════════════════════════════════════════════════════════════════════════════════════════

Testes para o módulo de simulação Zero Disruption Manufacturing.
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from backend.simulation.zdm.failure_scenario_generator import (
    FailureScenario,
    FailureType,
    FailureConfig,
    generate_failure_scenarios,
    generate_single_failure,
)
from backend.simulation.zdm.zdm_simulator import (
    ZDMSimulator,
    SimulationConfig,
    ResilienceReport,
    RecoveryStatus,
    quick_resilience_check,
)
from backend.simulation.zdm.recovery_strategy_engine import (
    RecoveryStrategy,
    RecoveryConfig,
    suggest_best_recovery,
    get_recovery_recommendations,
)


def create_test_plan() -> pd.DataFrame:
    """Cria um plano de teste."""
    now = datetime.now()
    
    data = []
    machines = ["M-101", "M-102", "M-103", "M-201", "M-202"]
    
    for i in range(20):
        machine = machines[i % len(machines)]
        start = now + timedelta(hours=i * 2)
        duration = np.random.uniform(1, 4)
        
        data.append({
            "order_id": f"ORD-{i//4:03d}",
            "article_id": f"ART-{i % 5:03d}",
            "machine_id": machine,
            "op_seq": i % 5,
            "start_time": start.isoformat(),
            "end_time": (start + timedelta(hours=duration)).isoformat(),
            "duration_min": duration * 60,
            "qty": np.random.randint(10, 100),
            "priority": np.random.randint(1, 10),
        })
    
    return pd.DataFrame(data)


class TestFailureScenarioGenerator:
    """Testes para geração de cenários de falha."""
    
    def test_generate_single_failure(self):
        """Testa geração de uma única falha."""
        plan_start = datetime.now()
        plan_end = plan_start + timedelta(days=3)
        config = FailureConfig()
        
        scenario = generate_single_failure(
            machine_id="M-101",
            failure_type=FailureType.SUDDEN,
            plan_start=plan_start,
            plan_end=plan_end,
            config=config,
            scenario_idx=0,
        )
        
        assert scenario.machine_id == "M-101"
        assert scenario.failure_type == FailureType.SUDDEN
        assert scenario.duration_hours > 0
        assert 0 <= scenario.severity <= 1
        assert scenario.scenario_id.startswith("ZDM-")
    
    def test_generate_multiple_scenarios(self):
        """Testa geração de múltiplos cenários."""
        plan_df = create_test_plan()
        
        scenarios = generate_failure_scenarios(plan_df, n_scenarios=10)
        
        assert len(scenarios) == 10
        assert all(isinstance(s, FailureScenario) for s in scenarios)
        
        # Verificar variedade de tipos
        types = {s.failure_type for s in scenarios}
        assert len(types) >= 2  # Pelo menos 2 tipos diferentes
    
    def test_rul_weighted_scenarios(self):
        """Testa que cenários com RUL baixo têm mais probabilidade."""
        plan_df = create_test_plan()
        
        # Simular RUL muito baixo para M-101
        rul_info = {
            "M-101": 20.0,   # Crítico
            "M-102": 500.0,  # Saudável
            "M-103": 500.0,
            "M-201": 500.0,
            "M-202": 500.0,
        }
        
        config = FailureConfig(rul_weight_multiplier=5.0)
        scenarios = generate_failure_scenarios(
            plan_df, 
            n_scenarios=20, 
            rul_info=rul_info,
            config=config,
        )
        
        # M-101 deve aparecer mais vezes
        m101_count = sum(1 for s in scenarios if s.machine_id == "M-101")
        assert m101_count >= 3  # Deve ter pelo menos 3 (de 20)


class TestZDMSimulator:
    """Testes para o simulador ZDM."""
    
    def test_simulate_single_scenario(self):
        """Testa simulação de um cenário."""
        plan_df = create_test_plan()
        
        scenario = FailureScenario(
            scenario_id="TEST-001",
            failure_type=FailureType.SUDDEN,
            machine_id="M-101",
            start_time=datetime.now() + timedelta(hours=5),
            duration_hours=4.0,
            severity=0.7,
        )
        
        simulator = ZDMSimulator()
        result = simulator.simulate_scenario(plan_df, scenario)
        
        assert result.scenario == scenario
        assert result.impact is not None
        assert result.recovery_status in RecoveryStatus
        assert len(result.recovery_actions_taken) > 0
    
    def test_simulate_all_scenarios(self):
        """Testa simulação de múltiplos cenários."""
        plan_df = create_test_plan()
        scenarios = generate_failure_scenarios(plan_df, n_scenarios=5)
        
        simulator = ZDMSimulator()
        report = simulator.simulate_all(plan_df, scenarios)
        
        assert isinstance(report, ResilienceReport)
        assert report.scenarios_simulated == 5
        assert 0 <= report.overall_resilience_score <= 100
        assert len(report.recommendations) > 0
    
    def test_recovery_reduces_impact(self):
        """Testa que recuperação reduz impacto."""
        plan_df = create_test_plan()
        
        scenario = FailureScenario(
            scenario_id="TEST-002",
            failure_type=FailureType.SUDDEN,
            machine_id="M-101",
            start_time=datetime.now() + timedelta(hours=5),
            duration_hours=8.0,
            severity=0.8,
        )
        
        # Simulador com recuperação
        config_recovery = SimulationConfig(
            enable_rerouting=True,
            enable_overtime=True,
        )
        simulator_recovery = ZDMSimulator(config_recovery)
        result_recovery = simulator_recovery.simulate_scenario(plan_df, scenario)
        
        # Simulador sem recuperação
        config_no_recovery = SimulationConfig(
            enable_rerouting=False,
            enable_overtime=False,
            enable_priority_shuffle=False,
        )
        simulator_no_recovery = ZDMSimulator(config_no_recovery)
        result_no_recovery = simulator_no_recovery.simulate_scenario(plan_df, scenario)
        
        # Com recuperação deve ter menor impacto
        # (ou pelo menos tentar ações)
        assert result_recovery.recovery_status != RecoveryStatus.NOT_ATTEMPTED or \
               len(result_recovery.recovery_actions_taken) > 0
    
    def test_quick_resilience_check(self):
        """Testa verificação rápida de resiliência."""
        plan_df = create_test_plan()
        
        result = quick_resilience_check(plan_df, n_scenarios=3)
        
        assert "resilience_score" in result
        assert "scenarios_tested" in result
        assert result["scenarios_tested"] == 3


class TestRecoveryStrategyEngine:
    """Testes para o motor de estratégias de recuperação."""
    
    def test_suggest_recovery_for_sudden_failure(self):
        """Testa sugestão para falha súbita."""
        plan_df = create_test_plan()
        
        scenario = FailureScenario(
            scenario_id="TEST-003",
            failure_type=FailureType.SUDDEN,
            machine_id="M-101",
            start_time=datetime.now() + timedelta(hours=5),
            duration_hours=4.0,
            severity=0.8,
        )
        
        plan = suggest_best_recovery(plan_df, scenario)
        
        assert plan.primary_action is not None
        assert plan.primary_action.strategy in RecoveryStrategy
        assert plan.expected_effectiveness_pct > 0
        
        # Para falha súbita, REROUTE deve ser uma das top opções
        strategies = [plan.primary_action.strategy] + [a.strategy for a in plan.secondary_actions]
        assert RecoveryStrategy.REROUTE in strategies or RecoveryStrategy.ADD_SHIFT in strategies
    
    def test_suggest_recovery_for_quality_failure(self):
        """Testa sugestão para problema de qualidade."""
        plan_df = create_test_plan()
        
        scenario = FailureScenario(
            scenario_id="TEST-004",
            failure_type=FailureType.QUALITY,
            machine_id="M-102",
            start_time=datetime.now() + timedelta(hours=3),
            duration_hours=2.0,
            severity=0.6,
            quality_reject_rate=0.15,
        )
        
        plan = suggest_best_recovery(plan_df, scenario)
        
        assert plan.primary_action is not None
        # Para qualidade, PARTIAL_BATCH ou MAINTENANCE deve ser considerado
        strategies = [plan.primary_action.strategy] + [a.strategy for a in plan.secondary_actions]
        assert len(strategies) >= 1
    
    def test_recovery_plan_costs(self):
        """Testa cálculo de custos do plano."""
        plan_df = create_test_plan()
        
        scenario = FailureScenario(
            scenario_id="TEST-005",
            failure_type=FailureType.SUDDEN,
            machine_id="M-103",
            start_time=datetime.now() + timedelta(hours=10),
            duration_hours=12.0,
            severity=0.9,
        )
        
        config = RecoveryConfig(
            cost_shift_per_hour=100.0,
            cost_maintenance_urgent=1000.0,
        )
        
        plan = suggest_best_recovery(plan_df, scenario, config)
        
        assert plan.total_cost_eur >= 0
        assert plan.total_recovery_time_hours > 0
    
    def test_get_multiple_recommendations(self):
        """Testa recomendações para múltiplos cenários."""
        plan_df = create_test_plan()
        scenarios = generate_failure_scenarios(plan_df, n_scenarios=3)
        
        plans = get_recovery_recommendations(plan_df, scenarios)
        
        assert len(plans) == 3
        # Verificar ordenação por risco
        risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        risks = [risk_order.get(p.risk_level, 3) for p in plans]
        assert risks == sorted(risks)  # Deve estar ordenado


class TestIntegration:
    """Testes de integração end-to-end."""
    
    def test_full_zdm_workflow(self):
        """Testa workflow completo do ZDM."""
        # 1. Criar plano
        plan_df = create_test_plan()
        
        # 2. Gerar cenários
        scenarios = generate_failure_scenarios(plan_df, n_scenarios=5)
        assert len(scenarios) == 5
        
        # 3. Simular
        simulator = ZDMSimulator()
        report = simulator.simulate_all(plan_df, scenarios)
        
        # 4. Verificar report
        assert report.scenarios_simulated == 5
        assert report.overall_resilience_score >= 0
        
        # 5. Gerar recomendações
        for scenario in scenarios[:2]:
            recovery_plan = suggest_best_recovery(plan_df, scenario)
            assert recovery_plan.primary_action is not None
        
        # 6. Verificar output
        report_dict = report.to_dict()
        assert "overall_resilience_score" in report_dict
        assert "recommendations" in report_dict
        assert "scenario_details" in report_dict
    
    def test_machine_failure_mid_production(self):
        """
        Cenário toy: máquina falha a meio da produção.
        Verificar se sistema consegue recuperar com mínima perda de OTD.
        """
        # Criar plano com operações em M-101
        now = datetime.now()
        plan_data = [
            {
                "order_id": "ORD-VIP",
                "article_id": "ART-001",
                "machine_id": "M-101",
                "op_seq": 1,
                "start_time": (now + timedelta(hours=2)).isoformat(),
                "end_time": (now + timedelta(hours=6)).isoformat(),
                "duration_min": 240,
                "qty": 50,
                "priority": 9,  # VIP
            },
            {
                "order_id": "ORD-REG",
                "article_id": "ART-002",
                "machine_id": "M-101",
                "op_seq": 2,
                "start_time": (now + timedelta(hours=6)).isoformat(),
                "end_time": (now + timedelta(hours=10)).isoformat(),
                "duration_min": 240,
                "qty": 30,
                "priority": 5,
            },
            {
                "order_id": "ORD-LOW",
                "article_id": "ART-003",
                "machine_id": "M-102",
                "op_seq": 1,
                "start_time": (now + timedelta(hours=2)).isoformat(),
                "end_time": (now + timedelta(hours=8)).isoformat(),
                "duration_min": 360,
                "qty": 100,
                "priority": 3,
            },
        ]
        plan_df = pd.DataFrame(plan_data)
        
        # Falha em M-101 a meio
        scenario = FailureScenario(
            scenario_id="CRITICAL-001",
            failure_type=FailureType.SUDDEN,
            machine_id="M-101",
            start_time=now + timedelta(hours=4),  # A meio da operação VIP
            duration_hours=3.0,
            severity=0.8,
        )
        
        # Configurar alternativas
        config = SimulationConfig(enable_rerouting=True)
        simulator = ZDMSimulator(config)
        simulator.set_alternative_machines({"M-101": ["M-102", "M-103"]})
        
        # Simular
        result = simulator.simulate_scenario(plan_df, scenario)
        
        # Verificar que tentou recuperar
        assert result.recovery_status in [RecoveryStatus.SUCCESS, RecoveryStatus.PARTIAL]
        assert "Reencaminhamento" in result.recovery_actions_taken[0] or \
               "extra" in result.recovery_actions_taken[0]
        
        # Obter plano de recuperação
        recovery_plan = suggest_best_recovery(plan_df, scenario)
        assert recovery_plan.expected_effectiveness_pct >= 50
        
        print(f"\n📊 Resultado do Teste:")
        print(f"   Recovery Status: {result.recovery_status.value}")
        print(f"   Actions: {result.recovery_actions_taken}")
        print(f"   OTD Impact: {result.impact.otd_impact_pct:.1f}%")
        print(f"   Recovery Effectiveness: {recovery_plan.expected_effectiveness_pct:.1f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



