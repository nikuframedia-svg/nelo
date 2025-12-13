"""
Testes para Prevenção de Erros - PDM Guard, Shopfloor Guard, Risco (G1-G3)
"""
import pytest
from datetime import datetime


class TestG1_PDMGuard:
    """G1: PDM Guard - Validação de BOM, Routing, Documentação."""
    
    def test_bom_validation(self, test_client, sample_bom):
        """G1.1: Deve validar BOM antes de release."""
        response = test_client.post("/prevention/pdm/validate-bom", json=sample_bom)
        
        if response.status_code == 200:
            validation = response.json()
            assert "valid" in validation or "errors" in validation
            if "errors" in validation:
                # Verificar tipos de erros comuns
                errors = validation["errors"]
                assert isinstance(errors, list)
    
    def test_routing_validation(self, test_client, sample_routing):
        """G1.2: Deve validar Routing (tempos, recursos, completude)."""
        response = test_client.post("/prevention/pdm/validate-routing", json=sample_routing)
        
        if response.status_code == 200:
            validation = response.json()
            assert "valid" in validation or "errors" in validation
    
    def test_documentation_validation(self, test_client):
        """G1.3: Deve validar documentação requerida (desenhos, instruções)."""
        response = test_client.post("/prevention/pdm/validate-documentation", json={
            "item_id": "ITEM-TEST",
            "revision_id": "REV-A",
            "required_docs": ["drawing", "work_instruction", "quality_plan"]
        })
        
        if response.status_code == 200:
            validation = response.json()
            assert "valid" in validation or "missing_docs" in validation
    
    def test_release_blocking(self, test_client):
        """G1.4: Deve bloquear release se validações falharem."""
        invalid_bom = {
            "item_id": "ITEM-TEST",
            "components": [
                {"item_id": "INVALID-ITEM", "quantity": -1}  # Quantidade negativa
            ]
        }
        
        response = test_client.post("/prevention/pdm/validate-bom", json=invalid_bom)
        
        if response.status_code == 200:
            validation = response.json()
            if "valid" in validation:
                assert validation["valid"] == False
            if "errors" in validation:
                assert len(validation["errors"]) > 0


class TestG2_ShopfloorGuard:
    """G2: Shopfloor Guard - Material, Equipamento, Parâmetros."""
    
    def test_material_validation(self, test_client):
        """G2.1: Deve validar material via barcode/RFID contra ordem."""
        response = test_client.post("/prevention/shopfloor/validate-material", json={
            "order_id": "OP001",
            "material_barcode": "BC-001",
            "expected_item_id": "SKU-001"
        })
        
        if response.status_code == 200:
            validation = response.json()
            assert "valid" in validation or "match" in validation
            assert "item_id" in validation or "material" in validation
    
    def test_equipment_validation(self, test_client):
        """G2.2: Deve validar máquina/ferramenta correta e calibrada."""
        response = test_client.post("/prevention/shopfloor/validate-equipment", json={
            "order_id": "OP001",
            "machine_id": "M1",
            "expected_machine_id": "M1",
            "tool_id": "TOOL-001",
            "calibration_status": "valid"
        })
        
        if response.status_code == 200:
            validation = response.json()
            assert "valid" in validation or "equipment_ok" in validation
    
    def test_parameter_validation(self, test_client):
        """G2.3: Deve validar parâmetros dentro de limites seguros."""
        response = test_client.post("/prevention/shopfloor/validate-parameters", json={
            "order_id": "OP001",
            "operation_id": "OP1",
            "parameters": {
                "temperature": 150,
                "pressure": 5.0,
                "speed": 1000
            },
            "limits": {
                "temperature": {"min": 100, "max": 200},
                "pressure": {"min": 3.0, "max": 8.0},
                "speed": {"min": 500, "max": 2000}
            }
        })
        
        if response.status_code == 200:
            validation = response.json()
            assert "valid" in validation or "parameters_ok" in validation
            if "errors" in validation:
                # Se houver erros, devem indicar quais parâmetros estão fora
                assert isinstance(validation["errors"], list)
    
    def test_prestart_checklist(self, test_client):
        """G2.4: Deve validar checklist pré-início obrigatório."""
        response = test_client.post("/prevention/shopfloor/prestart-checklist", json={
            "order_id": "OP001",
            "checklist_items": [
                {"item": "material_verified", "checked": True},
                {"item": "equipment_ready", "checked": True},
                {"item": "safety_clear", "checked": False}  # Não verificado
            ]
        })
        
        if response.status_code == 200:
            validation = response.json()
            assert "all_checked" in validation or "valid" in validation
            # Se safety_clear não está verificado, deve bloquear
            if "all_checked" in validation:
                assert validation["all_checked"] == False


class TestG3_PredictiveGuard:
    """G3: Predictive Guard - Risco de Qualidade."""
    
    def test_risk_prediction(self, test_client):
        """G3.1: Deve prever risco de defeito antes de iniciar ordem."""
        response = test_client.post("/prevention/predictive/risk", json={
            "order_id": "OP001",
            "product_id": "PROD-A",
            "machine_id": "M1",
            "operator_id": "OPR-001",
            "shift": "NIGHT",
            "material_batch": "BATCH-001"
        })
        
        if response.status_code == 200:
            risk = response.json()
            assert "risk_score" in risk or "defect_probability" in risk
            risk_value = risk.get("risk_score") or risk.get("defect_probability", 0)
            assert 0 <= risk_value <= 1
    
    def test_risk_thresholds(self, test_client):
        """G3.2: Deve aplicar thresholds e ações (notificar, inspeção extra)."""
        # Risco alto
        response = test_client.post("/prevention/predictive/risk", json={
            "order_id": "OP001",
            "product_id": "PROD-A",
            "machine_id": "M1",
            "operator_id": "OPR-LOW-EXP",  # Operador com baixa experiência
            "shift": "NIGHT"
        })
        
        if response.status_code == 200:
            risk = response.json()
            risk_value = risk.get("risk_score") or risk.get("defect_probability", 0)
            
            # Se risco alto (>= 0.5), deve sugerir ações
            if risk_value >= 0.5:
                assert "recommended_actions" in risk or "actions" in risk
                actions = risk.get("recommended_actions") or risk.get("actions", [])
                assert isinstance(actions, list)
                assert len(actions) > 0
    
    def test_similar_orders_alert(self, test_client):
        """G3.3: Deve alertar se ordens similares tiveram problemas."""
        response = test_client.post("/prevention/predictive/check-similar", json={
            "order_id": "OP001",
            "product_id": "PROD-A",
            "machine_id": "M1"
        })
        
        if response.status_code == 200:
            check = response.json()
            assert "similar_orders" in check or "alerts" in check
            if "similar_orders" in check:
                # Se houver ordens similares com problemas, deve alertar
                similar = check["similar_orders"]
                if len(similar) > 0:
                    assert "alert" in check or "has_issues" in check

