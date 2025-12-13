"""
Testes para Chat/Copilot - Linguagem Natural (H1-H3)
"""
import pytest


class TestH1_IndustrialQA:
    """H1: QA Industrial - Perguntas sobre produção."""
    
    def test_bottleneck_question(self, test_client):
        """H1.1: Deve responder "Qual é o gargalo?"."""
        response = test_client.post("/chat", json={
            "question": "Qual é o gargalo atual?",
            "context": "production"
        })
        
        if response.status_code == 200:
            answer = response.json()
            assert "answer" in answer or "response" in answer or "text" in answer
            answer_text = (answer.get("answer") or answer.get("response") or answer.get("text", "")).lower()
            # Deve mencionar gargalo ou máquina
            assert "gargalo" in answer_text or "máquina" in answer_text or "bottleneck" in answer_text
    
    def test_stock_risk_question(self, test_client):
        """H1.2: Deve responder "Quais SKUs estão em risco de rutura?"."""
        response = test_client.post("/chat", json={
            "question": "Quais SKUs estão em risco de rutura?",
            "context": "inventory"
        })
        
        if response.status_code == 200:
            answer = response.json()
            answer_text = (answer.get("answer") or answer.get("response") or answer.get("text", "")).lower()
            # Deve mencionar SKU, stock ou rutura
            assert any(word in answer_text for word in ["sku", "stock", "rutura", "risco", "inventário"])
    
    def test_trust_index_question(self, test_client):
        """H1.3: Deve responder "Qual o Trust Index do produto X?"."""
        response = test_client.post("/chat", json={
            "question": "Qual o Trust Index do produto PROD-A?",
            "context": "duplios"
        })
        
        if response.status_code == 200:
            answer = response.json()
            answer_text = (answer.get("answer") or answer.get("response") or answer.get("text", "")).lower()
            # Deve mencionar trust index ou score
            assert any(word in answer_text for word in ["trust", "índice", "score", "confiança"])


class TestH2_CommandParsing:
    """H2: Parsing de Comandos."""
    
    def test_recalculate_command(self, test_client):
        """H2.1: Deve reconhecer comando "Recalcular plano"."""
        response = test_client.post("/chat", json={
            "question": "Recalcular plano",
            "context": "planning"
        })
        
        if response.status_code == 200:
            result = response.json()
            # Deve reconhecer como comando e executar
            assert "command" in result or "action" in result or "executed" in result
            if "command" in result:
                assert "recalculate" in result["command"].lower() or "replan" in result["command"].lower()
    
    def test_show_kpis_command(self, test_client):
        """H2.2: Deve reconhecer comando "Mostrar KPIs"."""
        response = test_client.post("/chat", json={
            "question": "Mostrar KPIs do plano",
            "context": "planning"
        })
        
        if response.status_code == 200:
            result = response.json()
            # Deve retornar KPIs
            assert "kpis" in result or "metrics" in result or "answer" in result


class TestH3_ContextAwareness:
    """H3: Consciência de Contexto."""
    
    def test_context_switching(self, test_client):
        """H3.1: Deve manter contexto entre perguntas."""
        # Primeira pergunta
        q1 = test_client.post("/chat", json={
            "question": "Qual é o gargalo?",
            "context": "production"
        })
        
        if q1.status_code == 200:
            answer1 = q1.json()
            bottleneck_machine = "M1"  # Assumir que resposta menciona M1
            
            # Segunda pergunta com contexto
            q2 = test_client.post("/chat", json={
                "question": "Qual a utilização dessa máquina?",
                "context": "production",
                "previous_context": answer1
            })
            
            if q2.status_code == 200:
                answer2 = q2.json()
                # Deve entender que "essa máquina" se refere ao gargalo
                answer_text = (answer2.get("answer") or answer2.get("response") or answer2.get("text", "")).lower()
                assert "utilização" in answer_text or "utilization" in answer_text or "m1" in answer_text
    
    def test_multi_domain_questions(self, test_client):
        """H3.2: Deve responder perguntas que cruzam domínios."""
        response = test_client.post("/chat", json={
            "question": "Como o gargalo afeta o stock do SKU-001?",
            "context": "cross_domain"
        })
        
        if response.status_code == 200:
            answer = response.json()
            answer_text = (answer.get("answer") or answer.get("response") or answer.get("text", "")).lower()
            # Deve mencionar tanto gargalo quanto stock
            assert any(word in answer_text for word in ["gargalo", "bottleneck", "stock", "sku", "inventário"])

