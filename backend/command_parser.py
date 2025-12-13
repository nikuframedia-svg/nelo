"""
Industrial Command Parser for Nikufra Production OS

Parses natural language commands in Portuguese and extracts structured intents.

Supports commands like:
- "Tira a M-301 das 8h às 12h amanhã"
- "Reforça o turno da tarde no corte em +2h"
- "Planeia só VIP até sexta-feira"
- "Mostra o percurso do ART-500"
- "Qual é o gargalo atual?"
- "Compara cenário com +1 máquina no corte"

R&D: WP2 (What-If + Explainable AI)
Hypothesis: H4.1 - LLM can interpret commands reliably without hallucination
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta


class CommandType(Enum):
    """Types of industrial commands."""
    # Machine operations
    MACHINE_DOWNTIME = "machine_downtime"      # Remove machine from schedule
    MACHINE_EXTEND = "machine_extend"          # Extend machine shift
    MACHINE_STATUS = "machine_status"          # Query machine status
    
    # Planning operations
    PLAN_PRIORITY = "plan_priority"            # Change order priorities
    PLAN_FILTER = "plan_filter"                # Filter plan by criteria
    PLAN_REGENERATE = "plan_regenerate"        # Regenerate plan
    
    # Query operations
    QUERY_ROUTE = "query_route"                # Query article route
    QUERY_BOTTLENECK = "query_bottleneck"      # Query bottleneck
    QUERY_KPI = "query_kpi"                    # Query KPIs
    QUERY_ORDER = "query_order"                # Query order status
    
    # What-If operations
    WHATIF_SCENARIO = "whatif_scenario"        # Run what-if scenario
    WHATIF_COMPARE = "whatif_compare"          # Compare scenarios
    
    # Explanation requests
    EXPLAIN_DECISION = "explain_decision"      # Explain a decision
    
    # Unknown
    UNKNOWN = "unknown"


@dataclass
class ParsedCommand:
    """Result of parsing an industrial command."""
    command_type: CommandType
    confidence: float  # 0-1
    entities: Dict[str, Any]
    original_text: str
    suggested_action: str
    requires_confirmation: bool = False


class CommandParser:
    """
    Rule-based command parser for industrial commands.
    
    Uses regex patterns and keyword matching for deterministic parsing.
    Falls back to LLM for complex cases.
    """
    
    # Patterns for machine operations
    MACHINE_PATTERNS = [
        # "Tira a M-301 das 8h às 12h"
        (r"tira\s+(?:a\s+)?(?P<machine>M-\d+)\s+das?\s+(?P<start>\d{1,2})h?\s+[àa]s?\s+(?P<end>\d{1,2})h?",
         CommandType.MACHINE_DOWNTIME),
        # "Remove M-301 amanhã"
        (r"remove\s+(?:a\s+)?(?P<machine>M-\d+)\s+(?P<when>amanhã|hoje|segunda|terça|quarta|quinta|sexta)",
         CommandType.MACHINE_DOWNTIME),
        # "Reforça turno da tarde no corte em +2h"
        (r"refor[çc]a\s+(?:o\s+)?turno\s+(?:da\s+)?(?P<shift>manhã|tarde|noite)\s+(?:no?\s+)?(?P<area>\w+)\s+em\s+\+?(?P<hours>\d+)h",
         CommandType.MACHINE_EXTEND),
        # "Estado da M-301"
        (r"estado\s+(?:da?\s+)?(?P<machine>M-\d+)",
         CommandType.MACHINE_STATUS),
    ]
    
    # Patterns for queries
    QUERY_PATTERNS = [
        # "Mostra percurso do ART-500"
        (r"(?:mostra|qual\s+[eé])\s+(?:o\s+)?percurso\s+(?:do?\s+)?(?P<article>ART-\d+)",
         CommandType.QUERY_ROUTE),
        # "Qual é o gargalo"
        (r"(?:qual|quem)\s+[eé]\s+(?:o\s+)?gargalo",
         CommandType.QUERY_BOTTLENECK),
        # "Mostra KPIs"
        (r"(?:mostra|quais\s+são)\s+(?:os\s+)?(?:kpi|indicadores)",
         CommandType.QUERY_KPI),
        # "Estado da ordem ORD-001"
        (r"(?:estado|onde\s+est[aá])\s+(?:a\s+)?ordem\s+(?P<order>ORD-\d+)",
         CommandType.QUERY_ORDER),
    ]
    
    # Patterns for planning
    PLAN_PATTERNS = [
        # "Planeia só VIP até sexta"
        (r"planei?a\s+s[oó]\s+(?P<priority>vip|urgente|normal)\s+at[eé]\s+(?P<until>\w+)",
         CommandType.PLAN_PRIORITY),
        # "Filtra por máquina M-301"
        (r"filtra\s+(?:por\s+)?(?P<filter_type>m[aá]quina|artigo|rota)\s+(?P<value>[\w-]+)",
         CommandType.PLAN_FILTER),
        # "Regenera o plano"
        (r"(?:regenera|recalcula|refaz)\s+(?:o\s+)?plano",
         CommandType.PLAN_REGENERATE),
    ]
    
    # Patterns for what-if
    WHATIF_PATTERNS = [
        # "E se adicionarmos uma máquina no corte?"
        (r"e\s+se\s+(?P<scenario>.+)",
         CommandType.WHATIF_SCENARIO),
        # "Compara cenário com +1 máquina"
        (r"compara\s+(?:cen[aá]rio\s+)?(?:com\s+)?(?P<scenario>.+)",
         CommandType.WHATIF_COMPARE),
    ]
    
    # Patterns for explanations
    EXPLAIN_PATTERNS = [
        # "Porque é que a M-301 é gargalo?"
        (r"(?:porque|porqu[eê]|explica)\s+(?:[eé]\s+que\s+)?(?:a\s+)?(?P<subject>M-\d+|ART-\d+|ORD-\d+)\s+(?P<aspect>.+)",
         CommandType.EXPLAIN_DECISION),
        # "Explica esta sugestão"
        (r"explica\s+(?:esta\s+)?(?P<what>sugest[aã]o|decis[aã]o|escolha)",
         CommandType.EXPLAIN_DECISION),
    ]
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns."""
        self.compiled_patterns = []
        
        all_patterns = (
            self.MACHINE_PATTERNS +
            self.QUERY_PATTERNS +
            self.PLAN_PATTERNS +
            self.WHATIF_PATTERNS +
            self.EXPLAIN_PATTERNS
        )
        
        for pattern, cmd_type in all_patterns:
            self.compiled_patterns.append((
                re.compile(pattern, re.IGNORECASE),
                cmd_type,
            ))
    
    def parse(self, text: str) -> ParsedCommand:
        """
        Parse a natural language command.
        
        Returns structured ParsedCommand with entities and suggested action.
        """
        text_normalized = text.strip().lower()
        
        # Try each pattern
        for pattern, cmd_type in self.compiled_patterns:
            match = pattern.search(text_normalized)
            if match:
                entities = match.groupdict()
                return self._build_command(cmd_type, entities, text)
        
        # Fallback: keyword-based classification
        return self._fallback_parse(text)
    
    def _build_command(
        self,
        cmd_type: CommandType,
        entities: Dict[str, Any],
        original_text: str,
    ) -> ParsedCommand:
        """Build a ParsedCommand from matched pattern."""
        
        # Generate suggested action based on command type
        action = self._generate_action(cmd_type, entities)
        
        # Determine if confirmation is needed
        requires_confirmation = cmd_type in (
            CommandType.MACHINE_DOWNTIME,
            CommandType.MACHINE_EXTEND,
            CommandType.PLAN_REGENERATE,
        )
        
        return ParsedCommand(
            command_type=cmd_type,
            confidence=0.85,
            entities=entities,
            original_text=original_text,
            suggested_action=action,
            requires_confirmation=requires_confirmation,
        )
    
    def _generate_action(
        self,
        cmd_type: CommandType,
        entities: Dict[str, Any],
    ) -> str:
        """Generate human-readable action description."""
        
        if cmd_type == CommandType.MACHINE_DOWNTIME:
            machine = entities.get("machine", "?")
            start = entities.get("start", "?")
            end = entities.get("end", "?")
            when = entities.get("when", "hoje")
            return f"Remover {machine} do plano das {start}h às {end}h ({when})"
        
        elif cmd_type == CommandType.MACHINE_EXTEND:
            shift = entities.get("shift", "?")
            area = entities.get("area", "?")
            hours = entities.get("hours", "?")
            return f"Estender turno da {shift} em {area} por +{hours}h"
        
        elif cmd_type == CommandType.MACHINE_STATUS:
            machine = entities.get("machine", "?")
            return f"Consultar estado da máquina {machine}"
        
        elif cmd_type == CommandType.QUERY_ROUTE:
            article = entities.get("article", "?")
            return f"Mostrar percurso/rota do artigo {article}"
        
        elif cmd_type == CommandType.QUERY_BOTTLENECK:
            return "Identificar e explicar o gargalo atual"
        
        elif cmd_type == CommandType.QUERY_KPI:
            return "Mostrar KPIs de produção atuais"
        
        elif cmd_type == CommandType.QUERY_ORDER:
            order = entities.get("order", "?")
            return f"Consultar estado da ordem {order}"
        
        elif cmd_type == CommandType.PLAN_PRIORITY:
            priority = entities.get("priority", "?")
            until = entities.get("until", "?")
            return f"Priorizar ordens {priority.upper()} até {until}"
        
        elif cmd_type == CommandType.PLAN_FILTER:
            filter_type = entities.get("filter_type", "?")
            value = entities.get("value", "?")
            return f"Filtrar plano por {filter_type}: {value}"
        
        elif cmd_type == CommandType.PLAN_REGENERATE:
            return "Regenerar plano de produção completo"
        
        elif cmd_type == CommandType.WHATIF_SCENARIO:
            scenario = entities.get("scenario", "?")
            return f"Simular cenário: {scenario}"
        
        elif cmd_type == CommandType.WHATIF_COMPARE:
            scenario = entities.get("scenario", "?")
            return f"Comparar cenário: {scenario}"
        
        elif cmd_type == CommandType.EXPLAIN_DECISION:
            subject = entities.get("subject", entities.get("what", "?"))
            return f"Explicar decisão sobre {subject}"
        
        return "Ação não reconhecida"
    
    def _fallback_parse(self, text: str) -> ParsedCommand:
        """Keyword-based fallback parsing."""
        text_lower = text.lower()
        
        # Check for keywords
        if any(w in text_lower for w in ["gargalo", "bottleneck"]):
            return ParsedCommand(
                command_type=CommandType.QUERY_BOTTLENECK,
                confidence=0.6,
                entities={},
                original_text=text,
                suggested_action="Identificar o gargalo atual",
            )
        
        if any(w in text_lower for w in ["kpi", "indicador", "métrica"]):
            return ParsedCommand(
                command_type=CommandType.QUERY_KPI,
                confidence=0.6,
                entities={},
                original_text=text,
                suggested_action="Mostrar KPIs de produção",
            )
        
        if any(w in text_lower for w in ["percurso", "rota", "routing"]):
            # Try to extract article
            art_match = re.search(r"(ART-\d+)", text, re.IGNORECASE)
            article = art_match.group(1) if art_match else None
            return ParsedCommand(
                command_type=CommandType.QUERY_ROUTE,
                confidence=0.5,
                entities={"article": article},
                original_text=text,
                suggested_action=f"Mostrar percurso do artigo {article or '?'}",
            )
        
        if any(w in text_lower for w in ["e se", "cenário", "simula"]):
            return ParsedCommand(
                command_type=CommandType.WHATIF_SCENARIO,
                confidence=0.5,
                entities={"scenario": text},
                original_text=text,
                suggested_action=f"Simular cenário: {text}",
            )
        
        # Unknown command
        return ParsedCommand(
            command_type=CommandType.UNKNOWN,
            confidence=0.0,
            entities={},
            original_text=text,
            suggested_action="Comando não reconhecido. Tente reformular.",
        )


def parse_command(text: str) -> Dict[str, Any]:
    """
    Parse an industrial command and return structured result.
    
    Main entry point for command parsing.
    """
    parser = CommandParser()
    result = parser.parse(text)
    
    return {
        "command_type": result.command_type.value,
        "confidence": result.confidence,
        "entities": result.entities,
        "suggested_action": result.suggested_action,
        "requires_confirmation": result.requires_confirmation,
        "original_text": result.original_text,
    }


def execute_command(parsed: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Execute a parsed command.
    
    Returns result of execution or error.
    
    TODO[R&D]: Implement actual command execution for each type.
    """
    cmd_type = parsed.get("command_type", "unknown")
    entities = parsed.get("entities", {})
    
    if cmd_type == "query_bottleneck":
        # Return bottleneck info
        from scheduler import compute_bottleneck
        from data_loader import load_dataset
        import pandas as pd
        from pathlib import Path
        
        plan_path = Path(__file__).parent.parent / "data" / "production_plan.csv"
        if plan_path.exists():
            plan_df = pd.read_csv(plan_path)
            bottleneck = compute_bottleneck(plan_df)
            return {
                "success": True,
                "result": bottleneck,
                "message": f"O gargalo atual é a máquina {bottleneck['machine_id']} com {bottleneck['total_minutes']/60:.1f}h de carga.",
            }
        return {"success": False, "message": "Plano não encontrado."}
    
    elif cmd_type == "query_kpi":
        from scheduler import compute_kpis
        from data_loader import load_dataset
        import pandas as pd
        from pathlib import Path
        
        plan_path = Path(__file__).parent.parent / "data" / "production_plan.csv"
        if plan_path.exists():
            plan_df = pd.read_csv(plan_path)
            data = load_dataset()
            kpis = compute_kpis(plan_df, data.orders)
            return {
                "success": True,
                "result": kpis,
                "message": f"KPIs: Makespan {kpis['makespan_hours']:.1f}h, OTD {kpis['otd_percent']:.1f}%, Setup {kpis['setup_hours']:.1f}h",
            }
        return {"success": False, "message": "Plano não encontrado."}
    
    elif cmd_type == "query_route":
        article = entities.get("article")
        if article:
            from data_loader import load_dataset
            data = load_dataset()
            routes = data.routing[data.routing["article_id"] == article]
            if not routes.empty:
                route_info = routes[["op_seq", "op_code", "primary_machine_id", "route_label"]].to_dict("records")
                return {
                    "success": True,
                    "result": route_info,
                    "message": f"Artigo {article} tem {len(routes)} operações.",
                }
        return {"success": False, "message": f"Artigo {article} não encontrado."}
    
    elif cmd_type in ("whatif_scenario", "whatif_compare"):
        scenario = entities.get("scenario", "")
        from what_if_engine import build_scenario_comparison
        result = build_scenario_comparison(scenario)
        return {
            "success": True,
            "result": result,
            "message": "Cenário simulado com sucesso.",
        }
    
    # Default: not implemented
    return {
        "success": False,
        "message": f"Execução de comando '{cmd_type}' ainda não implementada.",
        "parsed": parsed,
    }



