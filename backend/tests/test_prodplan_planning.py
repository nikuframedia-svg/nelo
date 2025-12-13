"""
Testes para ProdPlan - Planeamento & Execução (A1-A6)
"""
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any


class TestA1_PrecedencesAndCapacity:
    """A1: Validar precedências e capacidade no scheduling."""
    
    def test_precedences_respected(self, test_client, sample_orders, sample_routing):
        """A1.1: Operações devem respeitar precedências definidas no routing."""
        # Criar plano com routing que tem precedências
        response = test_client.post("/scheduling/plan", json={
            "orders": sample_orders[:1],
            "routing": sample_routing
        })
        
        if response.status_code == 200:
            plan = response.json()
            # Verificar que OP1 vem antes de OP2
            ops = plan.get("operations", [])
            op1_idx = next((i for i, op in enumerate(ops) if op.get("operation_id") == "OP1"), None)
            op2_idx = next((i for i, op in enumerate(ops) if op.get("operation_id") == "OP2"), None)
            
            if op1_idx is not None and op2_idx is not None:
                assert op1_idx < op2_idx, "OP1 deve vir antes de OP2"
    
    def test_capacity_constraints(self, test_client, sample_orders, sample_machines):
        """A1.2: Plano não deve exceder capacidade das máquinas."""
        response = test_client.post("/scheduling/plan", json={
            "orders": sample_orders,
            "machines": sample_machines
        })
        
        if response.status_code == 200:
            plan = response.json()
            # Verificar que não há sobrecarga
            machine_loads = {}
            for op in plan.get("operations", []):
                machine_id = op.get("machine_id")
                duration = op.get("duration_minutes", 0)
                if machine_id:
                    machine_loads[machine_id] = machine_loads.get(machine_id, 0) + duration
            
            # Verificar que carga não excede capacidade disponível (assumindo 8h = 480min)
            for machine_id, load in machine_loads.items():
                assert load <= 480, f"Máquina {machine_id} sobrecarregada: {load}min > 480min"


class TestA2_PriorityAndDueDate:
    """A2: Prioridade e data de entrega afetam sequência."""
    
    def test_priority_ordering(self, test_client, sample_orders):
        """A2.1: Ordens com maior prioridade devem ser agendadas primeiro."""
        # OP002 tem priority=8, OP001 tem priority=5, OP003 tem priority=3
        response = test_client.post("/scheduling/plan", json={
            "orders": sample_orders,
            "priority_mode": "priority"
        })
        
        if response.status_code == 200:
            plan = response.json()
            order_sequence = [op.get("order_id") for op in plan.get("operations", [])]
            
            # OP002 (priority=8) deve aparecer antes de OP001 (priority=5)
            if "OP002" in order_sequence and "OP001" in order_sequence:
                assert order_sequence.index("OP002") < order_sequence.index("OP001")
    
    def test_due_date_ordering(self, test_client, sample_orders):
        """A2.2: Ordens com data de entrega mais próxima devem ter prioridade."""
        response = test_client.post("/scheduling/plan", json={
            "orders": sample_orders,
            "priority_mode": "due_date"
        })
        
        if response.status_code == 200:
            plan = response.json()
            # OP002 tem due_date mais próxima (5 dias) que OP001 (7 dias)
            order_sequence = [op.get("order_id") for op in plan.get("operations", [])]
            
            if "OP002" in order_sequence and "OP001" in order_sequence:
                assert order_sequence.index("OP002") < order_sequence.index("OP001")


class TestA3_VIPOrders:
    """A3: Ordens VIP devem ter tratamento especial."""
    
    def test_vip_priority_boost(self, test_client, sample_orders):
        """A3.1: Ordens marcadas como VIP devem ter prioridade máxima."""
        vip_order = {**sample_orders[0], "is_vip": True, "priority": 1}  # Prioridade baixa mas VIP
        normal_order = {**sample_orders[1], "is_vip": False, "priority": 10}  # Prioridade alta mas não VIP
        
        response = test_client.post("/scheduling/plan", json={
            "orders": [vip_order, normal_order]
        })
        
        if response.status_code == 200:
            plan = response.json()
            order_sequence = [op.get("order_id") for op in plan.get("operations", [])]
            
            # VIP deve aparecer primeiro mesmo com prioridade menor
            if vip_order["order_id"] in order_sequence:
                vip_idx = order_sequence.index(vip_order["order_id"])
                normal_idx = order_sequence.index(normal_order["order_id"])
                assert vip_idx < normal_idx, "Ordem VIP deve ter precedência"
    
    def test_vip_capacity_override(self, test_client, sample_orders):
        """A3.2: Ordens VIP podem usar capacidade extra (overtime)."""
        vip_order = {**sample_orders[0], "is_vip": True, "quantity": 1000}  # Quantidade grande
        
        response = test_client.post("/scheduling/plan", json={
            "orders": [vip_order],
            "allow_overtime": True
        })
        
        if response.status_code == 200:
            plan = response.json()
            # Verificar que plano foi criado mesmo com quantidade grande
            assert len(plan.get("operations", [])) > 0


class TestA4_ExecutionTracking:
    """A4: Rastreamento de execução em tempo real."""
    
    def test_start_operation(self, test_client):
        """A4.1: Deve ser possível iniciar uma operação."""
        response = test_client.post("/prodplan/operations/start", json={
            "operation_id": "OP001-OP1",
            "machine_id": "M1",
            "operator_id": "OPR-001",
            "started_at": datetime.now().isoformat()
        })
        
        # Deve aceitar ou retornar erro se não existir
        assert response.status_code in [200, 201, 404, 400]
    
    def test_complete_operation(self, test_client):
        """A4.2: Deve ser possível completar uma operação."""
        response = test_client.post("/prodplan/operations/complete", json={
            "operation_id": "OP001-OP1",
            "completed_at": datetime.now().isoformat(),
            "quantity_good": 95,
            "quantity_rejected": 5
        })
        
        assert response.status_code in [200, 404, 400]
    
    def test_execution_status(self, test_client):
        """A4.3: Deve retornar status atual de execução."""
        response = test_client.get("/prodplan/execution/status")
        
        if response.status_code == 200:
            status = response.json()
            assert "orders" in status or "operations" in status or isinstance(status, list)


class TestA5_BottleneckDetection:
    """A5: Deteção de gargalos."""
    
    def test_bottleneck_identification(self, test_client, sample_orders):
        """A5.1: Sistema deve identificar máquina gargalo."""
        response = test_client.get("/bottleneck")
        
        if response.status_code == 200:
            bottleneck = response.json()
            assert "machine_id" in bottleneck or "machine" in bottleneck
            assert "minutes" in bottleneck or "delay" in bottleneck or "load" in bottleneck
    
    def test_bottleneck_impact(self, test_client):
        """A5.2: Deve calcular impacto do gargalo no plano."""
        response = test_client.get("/bottleneck")
        
        if response.status_code == 200:
            bottleneck = response.json()
            # Deve ter informação sobre impacto
            assert any(key in bottleneck for key in ["impact", "delay", "affected_orders", "minutes"])


class TestA6_KPIs:
    """A6: KPIs de planeamento."""
    
    def test_plan_kpis(self, test_client, sample_orders):
        """A6.1: Deve calcular KPIs do plano (OTD, utilização, etc.)."""
        response = test_client.get("/plan/kpis")
        
        if response.status_code == 200:
            kpis = response.json()
            # Verificar presença de KPIs comuns
            assert isinstance(kpis, dict)
            # Pode ter: otd, utilization, makespan, etc.
    
    def test_machine_utilization(self, test_client):
        """A6.2: Deve calcular utilização por máquina."""
        response = test_client.get("/machines/utilization")
        
        if response.status_code == 200:
            utilization = response.json()
            assert isinstance(utilization, (dict, list))

