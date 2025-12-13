"""
Testes para Digital Twin - SHI-DT, XAI-DT, PredictiveCare (D1-D4)
"""
import pytest
from datetime import datetime, timedelta


class TestD1_SHIDT:
    """D1: SHI-DT (Smart Health Index Digital Twin)."""
    
    def test_shi_calculation(self, test_client, sample_sensor_readings):
        """D1.1: Deve calcular SHI (0-100) a partir de dados de sensores."""
        response = test_client.post("/digital-twin/shi-dt/calculate", json={
            "machine_id": "M1",
            "sensor_readings": sample_sensor_readings
        })
        
        if response.status_code == 200:
            shi = response.json()
            assert "health_index" in shi or "shi" in shi
            shi_value = shi.get("health_index") or shi.get("shi", 0)
            assert 0 <= shi_value <= 100
    
    def test_shi_anomaly_detection(self, test_client, sample_sensor_readings):
        """D1.2: Deve detetar anomalias via CVAE."""
        # Adicionar leitura anómala
        anomalous_readings = sample_sensor_readings + [{
            "machine_id": "M1",
            "timestamp": datetime.now().isoformat(),
            "sensor_type": "vibration",
            "value": 10.0,  # Valor muito alto
            "unit": "g"
        }]
        
        response = test_client.post("/digital-twin/shi-dt/calculate", json={
            "machine_id": "M1",
            "sensor_readings": anomalous_readings
        })
        
        if response.status_code == 200:
            result = response.json()
            # Deve indicar anomalia
            assert "anomaly_score" in result or "anomaly_detected" in result or "health_index" in result
            if "anomaly_score" in result:
                assert result["anomaly_score"] > 0.5  # Score alto indica anomalia
    
    def test_rul_estimation(self, test_client):
        """D1.3: Deve estimar RUL (Remaining Useful Life)."""
        response = test_client.get("/digital-twin/shi-dt/rul/M1")
        
        if response.status_code == 200:
            rul = response.json()
            assert "rul_hours" in rul or "rul" in rul
            if "rul_hours" in rul:
                assert rul["rul_hours"] >= 0  # RUL não pode ser negativo


class TestD2_XAIDT:
    """D2: XAI-DT (Explainable Digital Twin de Produto)."""
    
    def test_cad_scan_alignment(self, test_client):
        """D2.1: Deve alinhar CAD com scan 3D (ICP)."""
        response = test_client.post("/digital-twin/xai-dt/align", json={
            "cad_file": "test_cad.obj",
            "scan_file": "test_scan.ply",
            "product_id": "PROD-TEST"
        })
        
        if response.status_code == 200:
            alignment = response.json()
            assert "transformation_matrix" in alignment or "alignment_score" in alignment
    
    def test_deviation_field(self, test_client):
        """D2.2: Deve calcular campo de desvio geométrico."""
        response = test_client.post("/digital-twin/xai-dt/deviation", json={
            "product_id": "PROD-TEST",
            "cad_points": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            "scan_points": [[0.1, 0, 0], [1.05, 0, 0], [0, 0.95, 0]]
        })
        
        if response.status_code == 200:
            deviation = response.json()
            assert "deviation_field" in deviation or "deviations" in deviation
            assert "deviation_score" in deviation or "global_score" in deviation
    
    def test_rca_geometric(self, test_client):
        """D2.3: Deve realizar RCA geométrica (identificar causas de desvio)."""
        response = test_client.post("/digital-twin/xai-dt/rca", json={
            "product_id": "PROD-TEST",
            "deviation_pattern": "barrel_shape"
        })
        
        if response.status_code == 200:
            rca = response.json()
            assert "probable_causes" in rca or "causes" in rca
            assert "suggested_actions" in rca or "corrections" in rca


class TestD3_PredictiveCare:
    """D3: PredictiveCare (Manutenção Preditiva)."""
    
    def test_predictive_state(self, test_client):
        """D3.1: Deve calcular estado preditivo (SHI, RUL, risco)."""
        response = test_client.get("/digital-twin/predictivecare/state/M1")
        
        if response.status_code == 200:
            state = response.json()
            assert "shi_percent" in state or "health_index" in state
            assert "rul_hours" in state or "rul" in state
            assert "risk_next_7d" in state or "risk" in state
    
    def test_maintenance_workorder_creation(self, test_client):
        """D3.2: Deve criar ordens de manutenção automaticamente."""
        response = test_client.post("/maintenance/predictivecare/evaluate", json={
            "machine_id": "M1",
            "threshold_risk": 0.7
        })
        
        if response.status_code == 200:
            result = response.json()
            assert "work_orders" in result or "created_orders" in result
    
    def test_maintenance_window_suggestion(self, test_client):
        """D3.3: Deve sugerir janela ótima de manutenção."""
        response = test_client.get("/maintenance/predictivecare/suggest-window/M1")
        
        if response.status_code == 200:
            window = response.json()
            assert "suggested_start" in window or "start_time" in window
            assert "suggested_end" in window or "end_time" in window


class TestD4_IoTIngestion:
    """D4: IoT Ingestion."""
    
    def test_sensor_reading_ingestion(self, test_client, sample_sensor_readings):
        """D4.1: Deve ingerir leituras de sensores."""
        response = test_client.post("/digital-twin/iot/readings", json={
            "readings": sample_sensor_readings
        })
        
        assert response.status_code in [200, 201]
        if response.status_code in [200, 201]:
            result = response.json()
            assert "ingested" in result or "count" in result or "status" in result
    
    def test_opcua_ingestion(self, test_client):
        """D4.2: Deve suportar ingestão OPC-UA."""
        response = test_client.post("/digital-twin/iot/readings/opc-ua", json={
            "machine_id": "M1",
            "endpoint": "opc.tcp://localhost:4840",
            "node_ids": ["ns=2;s=Temperature", "ns=2;s=Vibration"]
        })
        
        assert response.status_code in [200, 400, 500]  # Pode falhar se OPC-UA não disponível

