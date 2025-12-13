"""
════════════════════════════════════════════════════════════════════════════════════════════════════
Chat Engine for Industrial Copilot
════════════════════════════════════════════════════════════════════════════════════════════════════

Main orchestrator for the Industrial Copilot chat system.

Features:
- Intent routing
- Skill-based response generation
- KPI payload for rich UI responses
- Context management
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from chat.router import Intent, route_intent, IntentMatch

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class KpiPayload(BaseModel):
    """KPI data to display in chat UI."""
    label: str
    value: str
    unit: Optional[str] = None
    trend: Optional[str] = None  # up, down, neutral
    color: Optional[str] = None  # green, yellow, red, blue


class ChatRequest(BaseModel):
    """Incoming chat request."""
    message: str
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response from chat engine."""
    message: str
    intent: str
    confidence: float
    kpis: List[KpiPayload] = []
    suggestions: List[str] = []
    actions: List[Dict[str, str]] = []  # {label, action_type, action_data}
    timestamp: str


# ═══════════════════════════════════════════════════════════════════════════════
# SKILLS
# ═══════════════════════════════════════════════════════════════════════════════

def scheduler_skill(message: str, context: Dict[str, Any]) -> ChatResponse:
    """Handle scheduling-related queries."""
    # TODO: Connect to actual scheduler service
    # For now, provide demo response
    
    kpis = [
        KpiPayload(label="Makespan", value="48.5", unit="h", trend="down", color="green"),
        KpiPayload(label="OTD", value="94.2", unit="%", trend="up", color="green"),
        KpiPayload(label="Tardiness", value="3.2", unit="h", trend="down", color="green"),
        KpiPayload(label="Utilização", value="87", unit="%", color="blue"),
    ]
    
    suggestions = [
        "Ver plano atual no Gantt",
        "Executar reotimização com MILP",
        "Verificar ordens atrasadas",
    ]
    
    actions = [
        {"label": "Ver Gantt", "action_type": "navigate", "action_data": "/prodplan"},
        {"label": "Otimizar", "action_type": "api_call", "action_data": "/api/optimize"},
    ]
    
    return ChatResponse(
        message=(
            "📊 **Estado do Planeamento**\n\n"
            "O plano atual está a decorrer normalmente. "
            "O makespan estimado é de 48.5 horas e o OTD está em 94.2%.\n\n"
            "Existem 3 ordens com potencial atraso que requerem atenção. "
            "A utilização média das máquinas é de 87%.\n\n"
            "💡 *Sugestão:* Considere reotimizar com MILP para reduzir o makespan."
        ),
        intent=Intent.SCHEDULER.value,
        confidence=0.85,
        kpis=kpis,
        suggestions=suggestions,
        actions=actions,
        timestamp=datetime.now().isoformat(),
    )


def inventory_skill(message: str, context: Dict[str, Any]) -> ChatResponse:
    """Handle inventory-related queries."""
    # TODO: Connect to actual inventory service
    
    kpis = [
        KpiPayload(label="SKUs em Risco", value="12", color="red"),
        KpiPayload(label="Cobertura Média", value="15.3", unit="dias", color="yellow"),
        KpiPayload(label="Valor Stock", value="€145K", color="blue"),
        KpiPayload(label="Taxa Serviço", value="96.8", unit="%", color="green"),
    ]
    
    suggestions = [
        "Ver SKUs críticos",
        "Executar previsão de procura",
        "Recalcular ROPs",
    ]
    
    return ChatResponse(
        message=(
            "📦 **Estado do Inventário**\n\n"
            "Atualmente existem **12 SKUs em risco** de rutura. "
            "A cobertura média é de 15.3 dias.\n\n"
            "O valor total em stock é de €145K com uma taxa de serviço de 96.8%.\n\n"
            "⚠️ *Alerta:* 3 materiais críticos (Classe A) têm cobertura < 7 dias."
        ),
        intent=Intent.INVENTORY.value,
        confidence=0.85,
        kpis=kpis,
        suggestions=suggestions,
        actions=[
            {"label": "SmartInventory", "action_type": "navigate", "action_data": "/inventory"},
        ],
        timestamp=datetime.now().isoformat(),
    )


def duplios_skill(message: str, context: Dict[str, Any]) -> ChatResponse:
    """Handle DPP/compliance-related queries."""
    
    kpis = [
        KpiPayload(label="Total DPPs", value="156", color="blue"),
        KpiPayload(label="ESPR", value="89", unit="%", color="green"),
        KpiPayload(label="CBAM", value="78", unit="%", color="yellow"),
        KpiPayload(label="Carbono Total", value="2.4K", unit="kg CO₂e", color="green"),
    ]
    
    suggestions = [
        "Ver DPPs não conformes",
        "Recalcular LCA",
        "Exportar relatório compliance",
    ]
    
    return ChatResponse(
        message=(
            "📋 **Estado dos Passaportes Digitais**\n\n"
            "Existem **156 DPPs** no sistema. "
            "Conformidade ESPR: 89%, CBAM: 78%, CSRD: 72%.\n\n"
            "A pegada carbónica total é de 2.4K kg CO₂e.\n\n"
            "✅ 14 novos DPPs criados este mês com compliance >90%."
        ),
        intent=Intent.DUPLIOS.value,
        confidence=0.85,
        kpis=kpis,
        suggestions=suggestions,
        actions=[
            {"label": "Duplios", "action_type": "navigate", "action_data": "/duplios"},
        ],
        timestamp=datetime.now().isoformat(),
    )


def digital_twin_skill(message: str, context: Dict[str, Any]) -> ChatResponse:
    """Handle digital twin/maintenance-related queries."""
    
    kpis = [
        KpiPayload(label="Saúde Global", value="82", unit="%", color="green"),
        KpiPayload(label="Críticas", value="2", color="red"),
        KpiPayload(label="RUL Médio", value="420", unit="h", color="blue"),
        KpiPayload(label="Manutenções", value="3", color="yellow"),
    ]
    
    suggestions = [
        "Ver máquinas críticas",
        "Ajustar plano com RUL",
        "Agendar manutenção preventiva",
    ]
    
    return ChatResponse(
        message=(
            "🔧 **Estado das Máquinas (Digital Twin)**\n\n"
            "A saúde global da fábrica é de **82%**. "
            "Existem **2 máquinas em estado crítico** que requerem atenção urgente:\n\n"
            "- CNC-03: RUL estimado 48h (Health Index: 35%)\n"
            "- MILL-02: RUL estimado 72h (Health Index: 42%)\n\n"
            "⚠️ *Recomendação:* Agendar manutenção preventiva para CNC-03 nas próximas 24h."
        ),
        intent=Intent.DIGITAL_TWIN.value,
        confidence=0.85,
        kpis=kpis,
        suggestions=suggestions,
        actions=[
            {"label": "Digital Twin", "action_type": "navigate", "action_data": "/prodplan"},
        ],
        timestamp=datetime.now().isoformat(),
    )


def rd_skill(message: str, context: Dict[str, Any]) -> ChatResponse:
    """Handle R&D-related queries."""
    
    kpis = [
        KpiPayload(label="Experiências", value="47", color="blue"),
        KpiPayload(label="WP1 Routing", value="+12%", color="green"),
        KpiPayload(label="WP4 Bandit", value="85", unit="ep", color="blue"),
        KpiPayload(label="Sugestões OK", value="78", unit="%", color="green"),
    ]
    
    suggestions = [
        "Ver resultados WP1",
        "Executar experiência WP4",
        "Exportar relatório R&D",
    ]
    
    return ChatResponse(
        message=(
            "🔬 **Estado de R&D**\n\n"
            "O módulo de R&D tem **47 experiências** registadas.\n\n"
            "**Resultados principais:**\n"
            "- WP1 Routing: Melhoria de +12% vs baseline FIFO\n"
            "- WP2 Sugestões: 78% das sugestões avaliadas como benéficas\n"
            "- WP4 Learning: 85 episódios executados, reward estabilizando\n\n"
            "💡 A estratégia CR (Critical Ratio) mostrou os melhores resultados."
        ),
        intent=Intent.RD.value,
        confidence=0.85,
        kpis=kpis,
        suggestions=suggestions,
        actions=[
            {"label": "R&D Lab", "action_type": "navigate", "action_data": "/research"},
        ],
        timestamp=datetime.now().isoformat(),
    )


def causal_skill(message: str, context: Dict[str, Any]) -> ChatResponse:
    """Handle causal analysis queries."""
    
    kpis = [
        KpiPayload(label="Variáveis", value="24", color="blue"),
        KpiPayload(label="Efeitos", value="18", color="green"),
        KpiPayload(label="ATE Setup→Energy", value="+15", unit="%", color="yellow"),
    ]
    
    suggestions = [
        "Ver grafo causal",
        "Analisar trade-offs",
        "Estimar efeito de intervenção",
    ]
    
    return ChatResponse(
        message=(
            "🔗 **Análise Causal**\n\n"
            "O módulo causal analisou **24 variáveis** e identificou **18 relações causais**.\n\n"
            "**Trade-offs identificados:**\n"
            "- Aumentar setup em 10% → +15% energia, -8% tardiness\n"
            "- Reduzir batch size → -5% stock, +3% setup\n\n"
            "💡 Use análise causal para entender impactos antes de decisões."
        ),
        intent=Intent.CAUSAL.value,
        confidence=0.85,
        kpis=kpis,
        suggestions=suggestions,
        actions=[
            {"label": "Causal", "action_type": "navigate", "action_data": "/causal"},
        ],
        timestamp=datetime.now().isoformat(),
    )


def greeting_skill(message: str, context: Dict[str, Any]) -> ChatResponse:
    """Handle greetings."""
    
    return ChatResponse(
        message=(
            "👋 Olá! Sou o **Copilot Industrial** do ProdPlan 4.0.\n\n"
            "Posso ajudar-te com:\n"
            "- 📊 **Planeamento** - ordens, Gantt, otimização\n"
            "- 📦 **Inventário** - stock, previsões, ROPs\n"
            "- 📋 **Duplios** - DPPs, compliance, LCA\n"
            "- 🔧 **Digital Twin** - saúde máquinas, RUL, manutenção\n"
            "- 🔬 **R&D** - experiências, resultados\n"
            "- 🔗 **Causal** - trade-offs, impactos\n\n"
            "Como posso ajudar hoje?"
        ),
        intent=Intent.GREETING.value,
        confidence=0.95,
        kpis=[],
        suggestions=[
            "Qual o estado do plano?",
            "Há SKUs em risco?",
            "Máquinas em alerta?",
        ],
        actions=[],
        timestamp=datetime.now().isoformat(),
    )


def general_skill(message: str, context: Dict[str, Any]) -> ChatResponse:
    """Handle general/unknown queries."""
    
    return ChatResponse(
        message=(
            "🤔 Não tenho certeza sobre o que perguntas.\n\n"
            "Posso ajudar com:\n"
            "- Planeamento de produção\n"
            "- Gestão de inventário\n"
            "- Passaportes digitais (DPP)\n"
            "- Saúde das máquinas\n"
            "- Investigação (R&D)\n\n"
            "Podes reformular a tua questão?"
        ),
        intent=Intent.GENERAL.value,
        confidence=0.3,
        kpis=[],
        suggestions=[
            "Qual o estado do plano?",
            "Há alertas de stock?",
            "Saúde das máquinas?",
        ],
        actions=[],
        timestamp=datetime.now().isoformat(),
    )


# Skill registry
SKILLS: Dict[Intent, callable] = {
    Intent.SCHEDULER: scheduler_skill,
    Intent.INVENTORY: inventory_skill,
    Intent.DUPLIOS: duplios_skill,
    Intent.DIGITAL_TWIN: digital_twin_skill,
    Intent.RD: rd_skill,
    Intent.CAUSAL: causal_skill,
    Intent.GREETING: greeting_skill,
    Intent.GENERAL: general_skill,
}


# ═══════════════════════════════════════════════════════════════════════════════
# CHAT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ChatEngine:
    """
    Main chat engine orchestrator.
    
    Routes messages to appropriate skills and generates responses.
    """
    
    def __init__(self):
        self.skills = SKILLS
        self.session_context: Dict[str, Dict[str, Any]] = {}
        logger.info("ChatEngine initialized")
    
    def handle_message(self, request: ChatRequest) -> ChatResponse:
        """
        Process incoming chat message and generate response.
        
        Args:
            request: ChatRequest with message and optional context
        
        Returns:
            ChatResponse with message, KPIs, and suggestions
        """
        message = request.message.strip()
        context = request.context or {}
        session_id = request.session_id or "default"
        
        logger.info(f"Processing message: '{message[:50]}...' for session {session_id}")
        
        # Update session context
        if session_id not in self.session_context:
            self.session_context[session_id] = {}
        
        self.session_context[session_id].update(context)
        
        # Route intent
        intent_match = route_intent(message)
        
        logger.info(
            f"Routed to intent '{intent_match.intent.value}' "
            f"with confidence {intent_match.confidence:.2f}"
        )
        
        # Get appropriate skill
        skill = self.skills.get(intent_match.intent, general_skill)
        
        # Generate response
        try:
            response = skill(message, self.session_context[session_id])
            
            # Update confidence based on router
            response.confidence = intent_match.confidence
            
            return response
            
        except Exception as e:
            logger.error(f"Skill execution failed: {e}")
            return ChatResponse(
                message="Desculpa, ocorreu um erro ao processar a tua mensagem. Tenta novamente.",
                intent=Intent.GENERAL.value,
                confidence=0.1,
                kpis=[],
                suggestions=["Tenta outra questão"],
                actions=[],
                timestamp=datetime.now().isoformat(),
            )
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get context for a session."""
        return self.session_context.get(session_id, {})
    
    def clear_session(self, session_id: str) -> None:
        """Clear session context."""
        if session_id in self.session_context:
            del self.session_context[session_id]


# Singleton instance
_engine_instance: Optional[ChatEngine] = None


def get_chat_engine() -> ChatEngine:
    """Get singleton ChatEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ChatEngine()
    return _engine_instance



