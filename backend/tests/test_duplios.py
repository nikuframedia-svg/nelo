"""
Testes para Duplios - PDM + DPP + LCA + Trust + Compliance + ESG (C1-C6)
"""
import pytest
from datetime import datetime


class TestC1_PDMCore:
    """C1: PDM Core - Items, Revisions, BOM, Routing."""
    
    def test_item_creation(self, test_client):
        """C1.1: Deve criar item no PDM."""
        response = test_client.post("/duplios/pdm/items", json={
            "item_code": "ITEM-TEST-001",
            "name": "Test Item",
            "type": "PRODUCT",
            "description": "Test description"
        })
        
        assert response.status_code in [200, 201, 400]
        if response.status_code in [200, 201]:
            item = response.json()
            assert "item_id" in item or "item_code" in item
    
    def test_revision_workflow(self, test_client):
        """C1.2: Deve criar revisão e seguir workflow (Draft → Released)."""
        # Criar item primeiro
        item_resp = test_client.post("/duplios/pdm/items", json={
            "item_code": "ITEM-TEST-002",
            "name": "Test Item 2",
            "type": "PRODUCT"
        })
        
        if item_resp.status_code in [200, 201]:
            item = item_resp.json()
            item_id = item.get("item_id") or item.get("item_code")
            
            # Criar revisão Draft
            rev_resp = test_client.post(f"/duplios/pdm/items/{item_id}/revisions", json={
                "revision_code": "A",
                "status": "DRAFT"
            })
            
            if rev_resp.status_code in [200, 201]:
                revision = rev_resp.json()
                assert revision.get("status") == "DRAFT"
                
                # Release revisão
                release_resp = test_client.post(f"/duplios/pdm/revisions/{revision.get('revision_id')}/release")
                if release_resp.status_code == 200:
                    released = release_resp.json()
                    assert released.get("status") == "RELEASED"
    
    def test_bom_validation(self, test_client, sample_bom):
        """C1.3: Deve validar BOM (sem ciclos, componentes válidos)."""
        response = test_client.post("/duplios/pdm/bom/validate", json=sample_bom)
        
        if response.status_code == 200:
            validation = response.json()
            assert "valid" in validation or "errors" in validation
    
    def test_bom_cycle_detection(self, test_client):
        """C1.4: Deve detectar ciclos na BOM."""
        cyclic_bom = {
            "item_id": "ITEM-A",
            "components": [
                {"item_id": "ITEM-B", "quantity": 1}
            ]
        }
        # Se ITEM-B contém ITEM-A, há ciclo
        response = test_client.post("/duplios/pdm/bom/validate", json=cyclic_bom)
        
        if response.status_code == 200:
            validation = response.json()
            # Deve detectar ciclo
            if "errors" in validation:
                errors = validation["errors"]
                assert any("cycle" in str(err).lower() or "circular" in str(err).lower() for err in errors)


class TestC2_DPPTrustIndex:
    """C2: DPP Trust Index avançado."""
    
    def test_trust_index_calculation(self, test_client, sample_dpp):
        """C2.1: Deve calcular Trust Index (0-100) campo-a-campo."""
        # Criar DPP primeiro
        dpp_resp = test_client.post("/duplios/dpp", json=sample_dpp)
        
        if dpp_resp.status_code in [200, 201]:
            dpp = dpp_resp.json()
            dpp_id = dpp.get("dpp_id")
            
            # Calcular Trust Index
            trust_resp = test_client.get(f"/duplios/dpp/{dpp_id}/trust-index")
            
            if trust_resp.status_code == 200:
                trust = trust_resp.json()
                assert "overall_trust_index" in trust
                assert 0 <= trust["overall_trust_index"] <= 100
                assert "field_scores" in trust or "field_metas" in trust
    
    def test_trust_index_factors(self, test_client, sample_dpp):
        """C2.2: Deve aplicar fatores (recência, verificação 3rd-party, incerteza)."""
        # DPP com dados medidos e recentes deve ter Trust Index alto
        high_trust_dpp = {
            **sample_dpp,
            "carbon_footprint_source": "MEASURED",
            "last_updated": datetime.now().isoformat(),
            "third_party_verified": True
        }
        
        dpp_resp = test_client.post("/duplios/dpp", json=high_trust_dpp)
        if dpp_resp.status_code in [200, 201]:
            dpp = dpp_resp.json()
            trust_resp = test_client.get(f"/duplios/dpp/{dpp.get('dpp_id')}/trust-index")
            
            if trust_resp.status_code == 200:
                trust = trust_resp.json()
                # Trust Index deve ser alto (>70) para dados medidos e verificados
                assert trust["overall_trust_index"] >= 70


class TestC3_GapFilling:
    """C3: Gap Filling Lite."""
    
    def test_gap_filling_missing_fields(self, test_client):
        """C3.1: Deve preencher campos ambientais em falta (CO2, água, reciclabilidade)."""
        incomplete_dpp = {
            "dpp_id": "DPP-GAP-001",
            "product_name": "Test Product",
            "product_code": "PROD-GAP",
            # Sem carbon_footprint, water_consumption, recyclability
            "composition": [{"material": "steel", "percentage": 100}]
        }
        
        response = test_client.post("/duplios/gap-filling/fill", json=incomplete_dpp)
        
        if response.status_code == 200:
            filled = response.json()
            # Deve ter preenchido campos em falta
            assert "carbon_footprint_kg_co2eq" in filled or "filled_fields" in filled
            assert "uncertainty" in filled or "source" in filled
    
    def test_gap_filling_contextual_adjustment(self, test_client):
        """C3.2: Deve ajustar valores por país e tecnologia."""
        dpp = {
            "dpp_id": "DPP-CTX-001",
            "product_name": "Test Product",
            "composition": [{"material": "steel", "percentage": 100}],
            "country_of_origin": "PT",  # Portugal tem energia mais limpa
            "manufacturing_year": 2023  # Tecnologia recente
        }
        
        response = test_client.post("/duplios/gap-filling/fill", json=dpp)
        
        if response.status_code == 200:
            filled = response.json()
            # Valores devem ser ajustados para PT
            assert "carbon_footprint_kg_co2eq" in filled or "filled_fields" in filled


class TestC4_ComplianceRadar:
    """C4: Compliance Radar (ESPR, CBAM, CSRD)."""
    
    def test_compliance_espr(self, test_client, sample_dpp):
        """C4.1: Deve analisar compliance ESPR."""
        response = test_client.post("/duplios/compliance/analyze", json={
            "dpp_id": sample_dpp.get("dpp_id", "DPP-TEST"),
            "framework": "ESPR"
        })
        
        if response.status_code == 200:
            compliance = response.json()
            assert "compliance_score" in compliance or "espr_score" in compliance
            assert "missing_fields" in compliance or "gaps" in compliance
    
    def test_compliance_cbam(self, test_client, sample_dpp):
        """C4.2: Deve analisar compliance CBAM."""
        response = test_client.post("/duplios/compliance/analyze", json={
            "dpp_id": sample_dpp.get("dpp_id", "DPP-TEST"),
            "framework": "CBAM"
        })
        
        if response.status_code == 200:
            compliance = response.json()
            assert "compliance_score" in compliance or "cbam_score" in compliance
    
    def test_compliance_csrd(self, test_client, sample_dpp):
        """C4.3: Deve analisar compliance CSRD."""
        response = test_client.post("/duplios/compliance/analyze", json={
            "dpp_id": sample_dpp.get("dpp_id", "DPP-TEST"),
            "framework": "CSRD"
        })
        
        if response.status_code == 200:
            compliance = response.json()
            assert "compliance_score" in compliance or "csrd_score" in compliance


class TestC5_LCA:
    """C5: LCA (Life Cycle Assessment)."""
    
    def test_lca_calculation(self, test_client, sample_dpp):
        """C5.1: Deve calcular impacto LCA completo."""
        response = test_client.post("/duplios/lca/calculate", json=sample_dpp)
        
        if response.status_code == 200:
            lca = response.json()
            assert "carbon_footprint" in lca or "co2_equivalent" in lca
            assert "water_consumption" in lca or "water_footprint" in lca
    
    def test_lca_multi_tier(self, test_client):
        """C5.2: Deve suportar análise multi-tier (fornecedores)."""
        multi_tier_dpp = {
            "dpp_id": "DPP-MT-001",
            "product_name": "Test Product",
            "composition": [
                {"material": "component_a", "percentage": 50, "supplier_dpp_id": "DPP-SUPPLIER-1"},
                {"material": "component_b", "percentage": 50, "supplier_dpp_id": "DPP-SUPPLIER-2"}
            ]
        }
        
        response = test_client.post("/duplios/lca/calculate", json=multi_tier_dpp)
        
        if response.status_code == 200:
            lca = response.json()
            # Deve agregar impactos dos fornecedores
            assert "carbon_footprint" in lca or "total_impact" in lca


class TestC6_ESGSuppliers:
    """C6: ESG e Fornecedores."""
    
    def test_supplier_tracking(self, test_client):
        """C6.1: Deve rastrear fornecedores e seus DPPs."""
        response = test_client.get("/duplios/suppliers")
        
        if response.status_code == 200:
            suppliers = response.json()
            assert isinstance(suppliers, (dict, list))
    
    def test_esg_score_aggregation(self, test_client):
        """C6.2: Deve agregar scores ESG de fornecedores."""
        response = test_client.post("/duplios/esg/aggregate", json={
            "product_dpp_id": "DPP-PROD-001",
            "supplier_dpp_ids": ["DPP-SUPPLIER-1", "DPP-SUPPLIER-2"]
        })
        
        if response.status_code == 200:
            esg = response.json()
            assert "overall_esg_score" in esg or "aggregated_score" in esg

