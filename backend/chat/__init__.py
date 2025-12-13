"""
════════════════════════════════════════════════════════════════════════════════════════════════════
INDUSTRIAL COPILOT - Chat Module
════════════════════════════════════════════════════════════════════════════════════════════════════

Contract 7 Implementation: Industrial Copilot (Chat)

Features:
- Intent routing based on keywords
- Skill-based response generation
- KPI payload for rich responses
- Multi-domain coverage (scheduling, inventory, duplios, digital twin, R&D)
"""

from chat.engine import (
    ChatEngine,
    ChatRequest,
    ChatResponse,
    KpiPayload,
)

from chat.router import (
    Intent,
    route_intent,
)

__all__ = [
    "ChatEngine",
    "ChatRequest",
    "ChatResponse",
    "KpiPayload",
    "Intent",
    "route_intent",
]



