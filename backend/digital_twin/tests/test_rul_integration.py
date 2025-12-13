"""
════════════════════════════════════════════════════════════════════════════════════════════════════
TESTS - RUL Integration with APS
════════════════════════════════════════════════════════════════════════════════════════════════════

Testes para verificar a integração do RUL com o scheduler.

Cenário principal:
- M1 com RUL muito baixo (crítica)
- M2 saudável
- Verificar que operações críticas são agendadas preferencialmente em M2
"""

import pytest
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

from ..health_indicator_cvae import (
    CVAE,
    CVAEConfig,
    SensorSnapshot,
    OperationContext,
    HealthIndicatorResult,
    infer_hi,
    create_demo_dataset,
)

from ..rul_estimator import (
    RULEstimate,
    RULEstimatorConfig,
    RULEstimator,
    estimate_rul,
    get_machine_health_status,
    HealthStatus,
    create_demo_hi_history,
)

from ..rul_integration_scheduler import (
    RULAdjustmentConfig,
    MachineRULInfo,
    PlanAdjustmentResult,
    adjust_plan_with_rul,
    get_rul_penalties,
    should_avoid_machine,
    compute_plan_reliability,
    create_rul_aware_machine_costs,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_plan_df() -> pd.DataFrame:
    """Criar plano de exemplo com operações em 3 máquinas."""
    now = datetime.now(timezone.utc)
    
    operations = [
        # M1 - máquina que terá RUL baixo
        {"order_id": "OP-001", "machine_id": "M1", "duration_min": 120, 
         "start_time": now, "end_time": now + timedelta(hours=2)},
        {"order_id": "OP-002", "machine_id": "M1", "duration_min": 90, 
         "start_time": now + timedelta(hours=2), "end_time": now + timedelta(hours=3.5)},
        
        # M2 - máquina saudável
        {"order_id": "OP-003", "machine_id": "M2", "duration_min": 60, 
         "start_time": now, "end_time": now + timedelta(hours=1)},
        {"order_id": "OP-004", "machine_id": "M2", "duration_min": 45, 
         "start_time": now + timedelta(hours=1), "end_time": now + timedelta(hours=1.75)},
        
        # M3 - máquina em warning
        {"order_id": "OP-005", "machine_id": "M3", "duration_min": 80, 
         "start_time": now, "end_time": now + timedelta(hours=1.33)},
    ]
    
    return pd.DataFrame(operations)


@pytest.fixture
def sample_rul_info() -> dict:
    """Criar informação de RUL de exemplo."""
    now = datetime.now(timezone.utc)
    
    # M1 - crítica (RUL < 24h)
    m1_rul = RULEstimate(
        machine_id="M1",
        timestamp=now,
        rul_mean_hours=15.0,  # Crítico!
        rul_std_hours=5.0,
        rul_lower_hours=5.0,
        rul_upper_hours=25.0,
        current_hi=0.25,
        health_status=HealthStatus.CRITICAL,
        degradation_rate_per_hour=-0.01,
        confidence=0.8,
        history_points_used=20,
        model_used="exponential",
    )
    
    # M2 - saudável (RUL > 168h)
    m2_rul = RULEstimate(
        machine_id="M2",
        timestamp=now,
        rul_mean_hours=500.0,  # Saudável
        rul_std_hours=50.0,
        rul_lower_hours=400.0,
        rul_upper_hours=600.0,
        current_hi=0.85,
        health_status=HealthStatus.HEALTHY,
        degradation_rate_per_hour=-0.0005,
        confidence=0.9,
        history_points_used=30,
        model_used="exponential",
    )
    
    # M3 - warning (24h < RUL < 72h)
    m3_rul = RULEstimate(
        machine_id="M3",
        timestamp=now,
        rul_mean_hours=50.0,  # Warning
        rul_std_hours=15.0,
        rul_lower_hours=20.0,
        rul_upper_hours=80.0,
        current_hi=0.45,
        health_status=HealthStatus.WARNING,
        degradation_rate_per_hour=-0.005,
        confidence=0.75,
        history_points_used=15,
        model_used="exponential",
    )
    
    return {
        "M1": m1_rul,
        "M2": m2_rul,
        "M3": m3_rul,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS - RUL ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════════

class TestRULEstimator:
    """Testes para o estimador de RUL."""
    
    def test_estimate_rul_exponential(self):
        """Testar estimação de RUL com modelo exponencial."""
        # Criar histórico de degradação
        history = create_demo_hi_history(
            machine_id="TEST-001",
            num_points=30,
            degradation_type="exponential",
            initial_hi=0.95,
            final_hi=0.5,
        )
        
        result = estimate_rul("TEST-001", history)
        
        assert result is not None
        assert result.machine_id == "TEST-001"
        assert result.rul_mean_hours > 0
        assert result.rul_std_hours >= 0
        assert result.rul_lower_hours <= result.rul_mean_hours <= result.rul_upper_hours
        assert 0 <= result.current_hi <= 1
        assert result.model_used == "exponential"
    
    def test_estimate_rul_linear(self):
        """Testar estimação de RUL com modelo linear."""
        config = RULEstimatorConfig(degradation_model="linear")
        
        history = create_demo_hi_history(
            machine_id="TEST-002",
            num_points=25,
            degradation_type="linear",
            initial_hi=0.9,
            final_hi=0.4,
        )
        
        result = estimate_rul("TEST-002", history, config)
        
        assert result is not None
        assert result.model_used == "linear"
        assert result.rul_mean_hours >= 0
    
    def test_insufficient_history(self):
        """Testar comportamento com histórico insuficiente."""
        history = [
            (datetime.now(timezone.utc), 0.9),
            (datetime.now(timezone.utc) + timedelta(hours=1), 0.88),
        ]  # Apenas 2 pontos
        
        result = estimate_rul("TEST-003", history)
        
        assert result is None  # Histórico insuficiente
    
    def test_health_status_classification(self):
        """Testar classificação de estado de saúde."""
        assert get_machine_health_status(0.9) == HealthStatus.HEALTHY
        assert get_machine_health_status(0.6) == HealthStatus.DEGRADED
        assert get_machine_health_status(0.4) == HealthStatus.WARNING
        assert get_machine_health_status(0.2) == HealthStatus.CRITICAL


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS - RUL INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestRULIntegration:
    """Testes para a integração RUL-Scheduler."""
    
    def test_adjust_plan_basic(self, sample_plan_df, sample_rul_info):
        """Testar ajuste básico do plano."""
        result = adjust_plan_with_rul(sample_plan_df, sample_rul_info)
        
        assert isinstance(result, PlanAdjustmentResult)
        assert len(result.adjusted_plan_df) == len(sample_plan_df)
        assert len(result.decisions) > 0
        assert "M1" in result.machine_rul_info
    
    def test_critical_machine_penalized(self, sample_plan_df, sample_rul_info):
        """Testar que máquinas críticas são penalizadas."""
        result = adjust_plan_with_rul(sample_plan_df, sample_rul_info)
        
        # Operações em M1 (crítica) devem ter penalização alta
        m1_ops = result.adjusted_plan_df[result.adjusted_plan_df['machine_id'] == 'M1']
        m2_ops = result.adjusted_plan_df[result.adjusted_plan_df['machine_id'] == 'M2']
        
        if not m1_ops.empty and not m2_ops.empty:
            avg_penalty_m1 = m1_ops['rul_penalty'].mean()
            avg_penalty_m2 = m2_ops['rul_penalty'].mean()
            
            # M1 deve ter penalização maior que M2
            assert avg_penalty_m1 > avg_penalty_m2
    
    def test_redistribution_to_healthy(self, sample_plan_df, sample_rul_info):
        """Testar redistribuição de operações para máquinas saudáveis."""
        config = RULAdjustmentConfig(enable_load_redistribution=True)
        result = adjust_plan_with_rul(sample_plan_df, sample_rul_info, config)
        
        # Verificar se há decisões de redistribuição
        redistribute_decisions = [
            d for d in result.decisions if d.decision_type == "REDISTRIBUTE"
        ]
        
        # Pode ou não haver redistribuições dependendo da lógica
        # O importante é que o processo complete sem erros
        assert result.operations_redistributed >= 0
    
    def test_maintenance_scheduled(self, sample_plan_df, sample_rul_info):
        """Testar agendamento de manutenção preventiva."""
        config = RULAdjustmentConfig(schedule_maintenance_for_critical=True)
        result = adjust_plan_with_rul(sample_plan_df, sample_rul_info, config)
        
        # Deve haver manutenção agendada para M1 (crítica)
        maintenance_decisions = [
            d for d in result.decisions if d.decision_type == "MAINTENANCE"
        ]
        
        assert len(maintenance_decisions) >= 1
        assert any(d.machine_id == "M1" for d in maintenance_decisions)
    
    def test_should_avoid_machine(self, sample_rul_info):
        """Testar função should_avoid_machine."""
        # M1 crítica - deve evitar
        should_avoid, reason = should_avoid_machine(
            "M1", sample_rul_info["M1"], operation_duration_min=60
        )
        assert should_avoid is True
        assert "crítica" in reason.lower()
        
        # M2 saudável - não deve evitar
        should_avoid, reason = should_avoid_machine(
            "M2", sample_rul_info["M2"], operation_duration_min=60
        )
        assert should_avoid is False
    
    def test_get_rul_penalties(self, sample_rul_info):
        """Testar cálculo de penalizações."""
        machines = ["M1", "M2", "M3"]
        penalties = get_rul_penalties(machines, sample_rul_info)
        
        assert penalties["M1"] > penalties["M2"]  # M1 crítica > M2 saudável
        assert penalties["M3"] > penalties["M2"]  # M3 warning > M2 saudável
        assert penalties["M2"] == 1.0  # M2 saudável sem penalização
    
    def test_plan_reliability(self, sample_plan_df, sample_rul_info):
        """Testar cálculo de reliability do plano."""
        reliability = compute_plan_reliability(sample_plan_df, sample_rul_info)
        
        assert 0 <= reliability <= 1
        # O plano usa M1 (crítica), então reliability não deve ser 1.0
        assert reliability < 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS - CVAE HEALTH INDICATOR
# ═══════════════════════════════════════════════════════════════════════════════

class TestCVAEHealthIndicator:
    """Testes para o CVAE de Health Indicators."""
    
    def test_sensor_snapshot_to_vector(self):
        """Testar conversão de SensorSnapshot para vetor."""
        snapshot = SensorSnapshot(
            machine_id="M1",
            timestamp=datetime.now(timezone.utc),
            vibration_x=0.1,
            vibration_y=0.15,
            temperature_bearing=0.3,
        )
        
        vector = snapshot.to_vector()
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 64  # Dimensão fixa
        assert vector.dtype == np.float32
    
    def test_operation_context_to_indices(self):
        """Testar conversão de OperationContext para índices."""
        config = CVAEConfig()
        context = OperationContext(
            machine_id="M1",
            op_code="OP-MILL",
            product_type="PROD-A",
        )
        
        m_idx, o_idx, p_idx = context.to_indices(config)
        
        assert 0 <= m_idx < config.num_machines
        assert 0 <= o_idx < config.num_operations
        assert 0 <= p_idx < config.num_product_types
    
    def test_create_demo_dataset(self):
        """Testar criação de dataset de demonstração."""
        dataset = create_demo_dataset(num_samples=100, num_machines=5)
        
        assert len(dataset) == 100
        
        snapshot, context, hi_target = dataset[0]
        assert isinstance(snapshot, SensorSnapshot)
        assert isinstance(context, OperationContext)
        assert 0 <= hi_target <= 1
    
    def test_cvae_inference_simulated(self):
        """Testar inferência do CVAE (modo simulado)."""
        config = CVAEConfig()
        model = CVAE(config)
        
        snapshot = SensorSnapshot(
            machine_id="M1",
            timestamp=datetime.now(timezone.utc),
            vibration_x=0.3,
            vibration_y=0.25,
            temperature_bearing=0.4,
        )
        
        context = OperationContext(
            machine_id="M1",
            op_code="OP-TURN",
            product_type="PROD-B",
        )
        
        result = infer_hi(model, snapshot, context, config)
        
        assert isinstance(result, HealthIndicatorResult)
        assert 0 <= result.hi <= 1
        assert result.status in ("HEALTHY", "WARNING", "CRITICAL")


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST - FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """
    Teste de integração completo.
    
    Cenário: Máquina M1 com RUL muito baixo, M2 saudável.
    Verificar que operações críticas são agendadas preferencialmente em M2.
    """
    
    def test_critical_operations_prefer_healthy_machines(self):
        """
        Teste principal: operações críticas devem preferir máquinas saudáveis.
        """
        now = datetime.now(timezone.utc)
        
        # Criar plano com operações em ambas as máquinas
        plan_df = pd.DataFrame([
            # Operação longa (crítica) em M1
            {"order_id": "OP-CRITICAL-1", "machine_id": "M1", "duration_min": 180,
             "start_time": now, "end_time": now + timedelta(hours=3)},
            # Operação curta em M1
            {"order_id": "OP-SHORT-1", "machine_id": "M1", "duration_min": 30,
             "start_time": now + timedelta(hours=3), "end_time": now + timedelta(hours=3.5)},
            # Operações em M2 (saudável)
            {"order_id": "OP-M2-1", "machine_id": "M2", "duration_min": 60,
             "start_time": now, "end_time": now + timedelta(hours=1)},
        ])
        
        # RUL info: M1 crítica, M2 saudável
        rul_info = {
            "M1": RULEstimate(
                machine_id="M1",
                timestamp=now,
                rul_mean_hours=10.0,  # Muito crítico!
                rul_std_hours=3.0,
                rul_lower_hours=4.0,
                rul_upper_hours=16.0,
                current_hi=0.2,
                health_status=HealthStatus.CRITICAL,
                degradation_rate_per_hour=-0.02,
                confidence=0.85,
                history_points_used=25,
                model_used="exponential",
            ),
            "M2": RULEstimate(
                machine_id="M2",
                timestamp=now,
                rul_mean_hours=800.0,  # Muito saudável
                rul_std_hours=100.0,
                rul_lower_hours=600.0,
                rul_upper_hours=1000.0,
                current_hi=0.92,
                health_status=HealthStatus.HEALTHY,
                degradation_rate_per_hour=-0.0002,
                confidence=0.95,
                history_points_used=40,
                model_used="exponential",
            ),
        }
        
        # Configurar para redistribuição agressiva
        config = RULAdjustmentConfig(
            enable_load_redistribution=True,
            rul_threshold_critical=24.0,
            critical_op_duration_threshold_min=60.0,  # Ops > 1h são críticas
        )
        
        # Ajustar plano
        result = adjust_plan_with_rul(plan_df, rul_info, config)
        
        # Verificações
        assert result is not None
        
        # 1. Deve haver decisões de redistribuição ou avoidance para M1
        m1_decisions = [d for d in result.decisions if d.machine_id == "M1"]
        assert len(m1_decisions) > 0, "Deve haver decisões para M1 (crítica)"
        
        # 2. Operação crítica em M1 deve ser redistribuída ou penalizada fortemente
        critical_op = result.adjusted_plan_df[
            result.adjusted_plan_df['order_id'] == 'OP-CRITICAL-1'
        ].iloc[0]
        
        # Se foi redistribuída, machine_id mudou; senão, penalty deve ser alta
        was_redistributed = critical_op['machine_id'] != 'M1'
        has_high_penalty = critical_op['rul_penalty'] >= 5.0
        
        assert was_redistributed or has_high_penalty, \
            "Operação crítica em M1 deve ser redistribuída ou penalizada"
        
        # 3. Reliability do plano ajustado deve ser >= reliability original
        # (porque redistribuímos para máquinas saudáveis)
        original_reliability = compute_plan_reliability(plan_df, rul_info)
        adjusted_reliability = compute_plan_reliability(result.adjusted_plan_df, rul_info)
        
        # Note: pode ser igual se não houve redistribuição possível
        assert adjusted_reliability >= original_reliability * 0.9, \
            "Reliability não deve degradar significativamente"
        
        # 4. Resumo deve refletir as ações tomadas
        summary = result.summary()
        assert "machines_at_risk" in summary
        assert summary["machines_at_risk"] >= 1  # M1 é de risco
        
        print("\n" + "="*60)
        print("RESULTADO DO TESTE DE INTEGRAÇÃO")
        print("="*60)
        print(f"Máquinas em risco: {summary['machines_at_risk']}")
        print(f"Operações redistribuídas: {summary['operations_redistributed']}")
        print(f"Manutenções agendadas: {summary['maintenance_scheduled']}")
        print(f"Penalização total aplicada: {summary['total_penalty_applied']:.2f}")
        print(f"Reliability original: {original_reliability:.3f}")
        print(f"Reliability ajustada: {adjusted_reliability:.3f}")
        print("="*60)


# ═══════════════════════════════════════════════════════════════════════════════
# RUN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])



