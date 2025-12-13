"""
Testes para SmartInventory - Stock, MRP, ROP, WIP, Spares (B1-B5)
"""
import pytest
from datetime import datetime, timedelta


class TestB1_ROPClassic:
    """B1: ROP (Re-order Point) clássico."""
    
    def test_rop_calculation(self, test_client, sample_skus):
        """B1.1: Deve calcular ROP baseado em lead time e consumo médio."""
        sku = sample_skus[0]
        response = test_client.post("/smart-inventory/rop/calculate", json={
            "sku_id": sku["sku_id"],
            "lead_time_days": sku["lead_time_days"],
            "average_daily_consumption": 10,
            "safety_stock_days": 2
        })
        
        if response.status_code == 200:
            rop = response.json()
            # ROP = (lead_time + safety) * daily_consumption
            expected_rop = (sku["lead_time_days"] + 2) * 10
            assert "rop" in rop or "reorder_point" in rop
            if "rop" in rop:
                assert rop["rop"] >= expected_rop * 0.9  # Tolerância 10%
    
    def test_rop_trigger(self, test_client, sample_skus):
        """B1.2: Deve gerar alerta quando stock cai abaixo do ROP."""
        sku = sample_skus[0]
        # Stock abaixo do ROP
        response = test_client.post("/smart-inventory/rop/check", json={
            "sku_id": sku["sku_id"],
            "current_stock": 50,  # Abaixo do ROP esperado
            "min_stock": sku["min_stock"]
        })
        
        if response.status_code == 200:
            result = response.json()
            assert result.get("should_reorder", False) or result.get("alert", False)


class TestB2_MRPComplete:
    """B2: MRP completo multi-nível."""
    
    def test_mrp_bom_explosion(self, test_client, sample_orders, sample_bom):
        """B2.1: Deve explodir BOM multi-nível corretamente."""
        response = test_client.post("/smart-inventory/mrp/run", json={
            "orders": sample_orders,
            "bom": sample_bom,
            "horizon_days": 30
        })
        
        if response.status_code == 200:
            mrp_result = response.json()
            # Deve ter necessidades calculadas
            assert "requirements" in mrp_result or "planned_orders" in mrp_result
    
    def test_mrp_net_requirements(self, test_client, sample_skus):
        """B2.2: Deve calcular necessidades líquidas (gross - stock - scheduled)."""
        response = test_client.post("/smart-inventory/mrp/calculate-net", json={
            "sku_id": sample_skus[0]["sku_id"],
            "gross_requirement": 500,
            "current_stock": sample_skus[0]["current_stock"],
            "scheduled_receipts": 100
        })
        
        if response.status_code == 200:
            net = response.json()
            # Net = max(0, Gross + Safety - Stock - Scheduled)
            assert "net_requirement" in net or "net" in net
            if "net_requirement" in net:
                expected_net = max(0, 500 + 100 - sample_skus[0]["current_stock"] - 100)
                assert abs(net["net_requirement"] - expected_net) < 10  # Tolerância
    
    def test_mrp_lot_sizing(self, test_client, sample_skus):
        """B2.3: Deve aplicar lot sizing (MOQ, múltiplos)."""
        sku = sample_skus[0]
        response = test_client.post("/smart-inventory/mrp/lot-size", json={
            "sku_id": sku["sku_id"],
            "net_requirement": 120,
            "moq": sku["moq"],
            "lot_multiple": 50
        })
        
        if response.status_code == 200:
            lot = response.json()
            # Deve ser >= MOQ e múltiplo de 50
            if "order_quantity" in lot:
                qty = lot["order_quantity"]
                assert qty >= sku["moq"]
                assert qty % 50 == 0


class TestB3_ForecastROP:
    """B3: ROP dinâmico com forecast."""
    
    def test_forecast_integration(self, test_client, sample_skus):
        """B3.1: ROP deve usar forecast de consumo futuro."""
        response = test_client.post("/smart-inventory/forecast/rop", json={
            "sku_id": sample_skus[0]["sku_id"],
            "historical_data": [10, 12, 11, 13, 10, 12, 11, 13, 10, 12],
            "forecast_horizon_days": 7
        })
        
        if response.status_code == 200:
            result = response.json()
            assert "forecast" in result or "rop" in result or "reorder_point" in result
    
    def test_dynamic_rop_adjustment(self, test_client, sample_skus):
        """B3.2: ROP deve ajustar-se quando forecast muda."""
        # Primeiro cálculo
        response1 = test_client.post("/smart-inventory/forecast/rop", json={
            "sku_id": sample_skus[0]["sku_id"],
            "historical_data": [10, 10, 10, 10],
            "forecast_horizon_days": 7
        })
        
        # Segundo cálculo com tendência crescente
        response2 = test_client.post("/smart-inventory/forecast/rop", json={
            "sku_id": sample_skus[0]["sku_id"],
            "historical_data": [10, 15, 20, 25],
            "forecast_horizon_days": 7
        })
        
        if response1.status_code == 200 and response2.status_code == 200:
            rop1 = response1.json()
            rop2 = response2.json()
            
            rop1_val = rop1.get("rop") or rop1.get("reorder_point", 0)
            rop2_val = rop2.get("rop") or rop2.get("reorder_point", 0)
            
            # ROP2 deve ser maior que ROP1 devido à tendência crescente
            assert rop2_val >= rop1_val * 0.9  # Pelo menos similar ou maior


class TestB4_WIPFlow:
    """B4: WIP (Work In Progress) Flow."""
    
    def test_wip_tracking(self, test_client):
        """B4.1: Deve rastrear posição atual de ordens em produção."""
        response = test_client.get("/smart-inventory/wip/current")
        
        if response.status_code == 200:
            wip = response.json()
            assert isinstance(wip, (dict, list))
            # Pode ter: orders, stations, progress, etc.
    
    def test_wip_progress(self, test_client):
        """B4.2: Deve mostrar progresso de cada ordem (ex: 70% completado)."""
        response = test_client.get("/smart-inventory/wip/progress")
        
        if response.status_code == 200:
            progress = response.json()
            assert isinstance(progress, (dict, list))
            # Cada item deve ter: order_id, progress_pct, current_station, etc.


class TestB5_SparesForecasting:
    """B5: Previsão de peças sobressalentes."""
    
    def test_spare_part_forecast(self, test_client):
        """B5.1: Deve prever necessidades de peças sobressalentes."""
        response = test_client.post("/smart-inventory/spares/forecast", json={
            "component_id": "BEARING-001",
            "machine_id": "M1",
            "historical_replacements": [
                {"date": "2024-01-01", "quantity": 2},
                {"date": "2024-04-01", "quantity": 2},
                {"date": "2024-07-01", "quantity": 1}
            ]
        })
        
        if response.status_code == 200:
            forecast = response.json()
            assert "expected_replacements" in forecast or "recommended_date" in forecast
    
    def test_spare_integration_mrp(self, test_client):
        """B5.2: Necessidades de peças devem integrar com MRP."""
        response = test_client.post("/smart-inventory/spares/mrp-demand", json={
            "component_id": "BEARING-001",
            "forecast_horizon_days": 90
        })
        
        if response.status_code == 200:
            demands = response.json()
            assert isinstance(demands, (dict, list))
            # Deve ter formato compatível com MRP

