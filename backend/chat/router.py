"""
════════════════════════════════════════════════════════════════════════════════════════════════════
Intent Router for Industrial Copilot
════════════════════════════════════════════════════════════════════════════════════════════════════

Keyword-based intent routing for chat messages.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """Possible intents for chat messages."""
    SCHEDULER = "scheduler"       # Production planning, Gantt, orders
    INVENTORY = "inventory"       # Stock, ROP, forecasts, materials
    DUPLIOS = "duplios"          # DPP, LCA, compliance, traceability
    DIGITAL_TWIN = "digital_twin" # Machine health, RUL, maintenance
    RD = "rd"                    # R&D experiments, research
    CAUSAL = "causal"            # Causal analysis, trade-offs
    GENERAL = "general"          # General/unknown intent
    GREETING = "greeting"        # Greetings


# Keyword patterns for each intent
INTENT_PATTERNS: dict[Intent, List[str]] = {
    Intent.SCHEDULER: [
        r"\bplano\b", r"\bplaneamento\b", r"\bplanning\b", r"\bgantt\b",
        r"\bordem\b", r"\border\b", r"\bordens\b", r"\borders\b",
        r"\bmáquina\b", r"\bmachine\b", r"\bresource\b", r"\brecurso\b",
        r"\bscheduler\b", r"\bschedule\b", r"\bagendar\b",
        r"\bmakespan\b", r"\btardiness\b", r"\botd\b",
        r"\bheuristic\b", r"\bheurística\b", r"\bmilp\b", r"\bcpsat\b",
        r"\bprodução\b", r"\bproduction\b", r"\bproduzir\b",
        r"\batrasad\b", r"\bdelay\b", r"\bprazo\b", r"\bdue\b",
    ],
    Intent.INVENTORY: [
        r"\bstock\b", r"\bestoque\b", r"\binventário\b", r"\binventory\b",
        r"\brop\b", r"\breorder\b", r"\bsegurança\b", r"\bsafety\b",
        r"\bforecast\b", r"\bprevisão\b", r"\bdemand\b", r"\bprocura\b",
        r"\bmaterial\b", r"\bmatéria\b", r"\bmrp\b", r"\bbom\b",
        r"\bcompra\b", r"\bpurchase\b", r"\bfornecedor\b", r"\bsupplier\b",
        r"\babcxyz\b", r"\brisco\b", r"\bruptura\b", r"\bstockout\b",
        r"\bcobertura\b", r"\bcoverage\b", r"\barmazém\b", r"\bwarehouse\b",
    ],
    Intent.DUPLIOS: [
        r"\bdpp\b", r"\bpassaport\b", r"\bpassport\b", r"\bduplios\b",
        r"\blca\b", r"\bcarbono\b", r"\bcarbon\b", r"\bco2\b",
        r"\bcompliance\b", r"\bconform\b", r"\bespr\b", r"\bcbam\b", r"\bcsrd\b",
        r"\breciclabil\b", r"\brecyclabil\b", r"\brecicla\b", r"\brecycl\b",
        r"\bramabilidade\b", r"\btrust\b", r"\bconfiança\b",
        r"\bqr\b", r"\bcode\b", r"\bidentidade\b", r"\bidentity\b",
        r"\bserial\b", r"\bbatch\b", r"\brfid\b", r"\bgtin\b",
        r"\brevisão\b", r"\brevision\b", r"\bpdm\b",
    ],
    Intent.DIGITAL_TWIN: [
        r"\brul\b", r"\bremanescent\b", r"\bremaining\b", r"\buseful\b", r"\blife\b",
        r"\bhealth\b", r"\bsaúde\b", r"\bdegrada\b", r"\bdesgaste\b", r"\bwear\b",
        r"\bmanutenção\b", r"\bmaintenance\b", r"\bprevent\b", r"\bpredict\b",
        r"\bfalha\b", r"\bfailure\b", r"\bbreak\b", r"\bavaria\b",
        r"\bdigital\s*twin\b", r"\bdt\b", r"\bsensor\b", r"\bhist[óo]rico\b",
        r"\bconform\w*\s*produto\b", r"\bscan\b", r"\bcad\b", r"\bdesvio\b",
    ],
    Intent.RD: [
        r"\br&d\b", r"\brd\b", r"\bresearch\b", r"\bpesquisa\b",
        r"\bwp1\b", r"\bwp2\b", r"\bwp3\b", r"\bwp4\b",
        r"\bexperiment\b", r"\bexperiência\b", r"\bhipótese\b", r"\bhypothesis\b",
        r"\bbandit\b", r"\blearning\b", r"\baprendizagem\b",
        r"\brouting\b", r"\bsugest\b", r"\bsuggestion\b",
        r"\bsifide\b", r"\binvestig\b",
    ],
    Intent.CAUSAL: [
        r"\bcausal\b", r"\bcause\b", r"\bcausa\b", r"\befeito\b", r"\beffect\b",
        r"\btrade\s*off\b", r"\btradeoff\b", r"\bcompromisso\b",
        r"\bimpacto\b", r"\bimpact\b", r"\bconsequên\b",
        r"\bdag\b", r"\bgraph\b", r"\bgrafo\b",
        r"\bate\b", r"\bcate\b", r"\bdml\b", r"\bdowhy\b",
    ],
    Intent.GREETING: [
        r"\bol[áa]\b", r"\bhello\b", r"\bhi\b", r"\bbom\s*dia\b", r"\bboa\s*tarde\b",
        r"\bboa\s*noite\b", r"\bgood\b", r"\bmorning\b", r"\bafternoon\b",
        r"\bcomo\s*est[áa]\b", r"\bhow\s*are\b", r"\btudo\s*bem\b",
        r"\bobrigad\b", r"\bthank\b", r"\bagradec\b",
    ],
}


@dataclass
class IntentMatch:
    """Result of intent matching."""
    intent: Intent
    confidence: float  # 0-1
    matched_keywords: List[str]


def route_intent(message: str) -> IntentMatch:
    """
    Route a message to the most likely intent based on keyword matching.
    
    Args:
        message: User's message text
    
    Returns:
        IntentMatch with intent, confidence, and matched keywords
    """
    message_lower = message.lower()
    
    # Count matches for each intent
    intent_scores: dict[Intent, Tuple[int, List[str]]] = {}
    
    for intent, patterns in INTENT_PATTERNS.items():
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, message_lower, re.IGNORECASE)
            if found:
                matches.extend(found)
        
        if matches:
            intent_scores[intent] = (len(matches), matches)
    
    # If no matches, return GENERAL
    if not intent_scores:
        return IntentMatch(
            intent=Intent.GENERAL,
            confidence=0.3,
            matched_keywords=[],
        )
    
    # Find intent with most matches
    best_intent = max(intent_scores.items(), key=lambda x: x[1][0])
    intent = best_intent[0]
    score, keywords = best_intent[1]
    
    # Calculate confidence based on number of matches
    # More matches = higher confidence, capped at 0.95
    confidence = min(0.95, 0.4 + (score * 0.15))
    
    logger.info(f"Routed message to intent '{intent.value}' with confidence {confidence:.2f}")
    
    return IntentMatch(
        intent=intent,
        confidence=confidence,
        matched_keywords=list(set(keywords)),
    )


def get_intent_description(intent: Intent, lang: str = "pt") -> str:
    """Get human-readable description of an intent."""
    descriptions = {
        Intent.SCHEDULER: {
            "pt": "Planeamento de produção e gestão de ordens",
            "en": "Production planning and order management",
        },
        Intent.INVENTORY: {
            "pt": "Gestão de inventário e materiais",
            "en": "Inventory and materials management",
        },
        Intent.DUPLIOS: {
            "pt": "Passaportes digitais de produto e compliance",
            "en": "Digital product passports and compliance",
        },
        Intent.DIGITAL_TWIN: {
            "pt": "Saúde das máquinas e manutenção preditiva",
            "en": "Machine health and predictive maintenance",
        },
        Intent.RD: {
            "pt": "Investigação e desenvolvimento",
            "en": "Research and development",
        },
        Intent.CAUSAL: {
            "pt": "Análise causal e trade-offs",
            "en": "Causal analysis and trade-offs",
        },
        Intent.GENERAL: {
            "pt": "Informação geral",
            "en": "General information",
        },
        Intent.GREETING: {
            "pt": "Saudação",
            "en": "Greeting",
        },
    }
    
    return descriptions.get(intent, {}).get(lang, descriptions[intent]["en"])



