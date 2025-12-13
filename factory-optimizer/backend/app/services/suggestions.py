import json
import logging
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, Optional

from app.insights.engine import InsightEngine
from app.insights.prompts import SYSTEM_PROMPT, get_prompt_by_mode
from app.llm import LLMUnavailableError, LocalLLM
from app.llm.validator import validate_llm_output
from app.etl.loader import get_loader

logger = logging.getLogger(__name__)


@dataclass
class Suggestion:
    id: str
    icon: str
    action: str
    explanation: str
    impact: str
    impact_level: str  # baixo / medio / alto
    gain: str
    reasoning_markdown: str
    data_points: Dict


def _impact_level_from_value(value: float) -> str:
    if value >= 0.25:
        return "alto"
    if value >= 0.15:
        return "medio"
    return "baixo"


def _format_delta(value: float, suffix: str = "%") -> str:
    sign = "-" if value < 0 else "+"
    magnitude = abs(value)
    return f"{sign}{magnitude:.1f}{suffix}"


def _parse_sections(markdown: str) -> Dict[str, str]:
    sections = {
        "resumo": "",
        "causas": "",
        "impacto": "",
        "acoes": "",
        "ganho": "",
    }
    current = None
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("📊"):
            current = "resumo"
            sections[current] = line
            continue
        if line.startswith("🧠"):
            current = "causas"
            sections[current] = line
            continue
        if line.startswith("⚙️"):
            current = "impacto"
            sections[current] = line
            continue
        if line.startswith("🔧"):
            current = "acoes"
            sections[current] = line
            continue
        if line.startswith("💰"):
            current = "ganho"
            sections[current] = line
            continue
        if current:
            sections[current] = (sections[current] + "\n" + line).strip()
    return sections


def _extract_action_from_text(text: str, context_data: Dict) -> str:
    """Extrai ação concreta do texto, garantindo que é específica."""
    # Procurar por padrões de ação
    action_patterns = [
        r'Desviar\s+(\d+%?)\s+de\s+carga\s+de\s+([A-Z0-9-]+)\s+para\s+([A-Z0-9-]+)',
        r'Repor\s+imediatamente\s+SKU\s+([A-Z0-9-]+)',
        r'Aumentar\s+overlap\s+de\s+(\d+%?)\s+para\s+(\d+%?)\s+no\s+setor\s+([A-Za-z]+)',
        r'Colar\s+famílias\s+([A-Z0-9,\s]+)',
        r'Agendar\s+manutenção\s+preventiva\s+em\s+([A-Z0-9-]+)',
    ]
    
    for pattern in action_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    
    # Se não encontrar padrão, extrair primeira frase após "🔧" ou "Ação:"
    action_lines = []
    for line in text.split('\n'):
        if '🔧' in line or 'ação' in line.lower() or 'recomend' in line.lower():
            clean_line = re.sub(r'[🔧💰⚙️🧠📊]', '', line).strip()
            if clean_line and len(clean_line) > 10:
                action_lines.append(clean_line)
    
    if action_lines:
        return action_lines[0]
    
    return "Ação a definir"


def _generate_suggestion_from_context(
    llm: LocalLLM, mode: str, context_json: str, suggestion_id: str, data_points: Dict
) -> Suggestion:
    """Gera uma sugestão usando o contexto do Insight Engine."""
    prompt = get_prompt_by_mode("sugestoes", context_json)  # Sempre usar modo sugestoes
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}\n\nIMPORTANTE: A ação recomendada DEVE ser específica e concreta. Menciona recursos reais, SKUs reais, percentagens específicas. NÃO uses texto genérico."

    try:
        # Temperatura 0.25 para sugestões (mais determinístico)
        reasoning = llm.generate(prompt=full_prompt, temperature=0.25, max_tokens=600, num_ctx=4096)
    except LLMUnavailableError:
        raise

    # Validar output
    context_data = json.loads(context_json)
    loader = get_loader()
    
    valid_skus = []
    valid_resources = []
    
    # Extrair SKUs válidos
    if "skus_criticos" in context_data:
        valid_skus = [sku.get("sku", "") for sku in context_data.get("skus_criticos", [])]
    
    # Extrair recursos válidos
    if "gargalo_principal" in context_data:
        gargalo_id = context_data["gargalo_principal"].get("id", "")
        if gargalo_id != "N/A":
            valid_resources.append(gargalo_id)
    if "recursos_alternativos" in context_data:
        for r in context_data.get("recursos_alternativos", []):
            if isinstance(r, dict):
                valid_resources.append(r.get("recurso", ""))
            else:
                valid_resources.append(str(r))
    
    # Também buscar dos roteiros
    roteiros = loader.get_roteiros()
    if not roteiros.empty and "maquinas_possiveis" in roteiros.columns:
        all_resources = set()
        for machines in roteiros["maquinas_possiveis"].dropna():
            if isinstance(machines, str):
                resources = re.findall(r'M[-\s]?(\d+)', machines)
                all_resources.update([f"M-{r}" for r in resources])
        valid_resources.extend(list(all_resources))
    
    validation = validate_llm_output(reasoning, context_data, "sugestoes", valid_skus, valid_resources)
    
    if not validation["is_valid"] and len(validation["errors"]) > 0:
        # Tentar regenerar com prompt mais restritivo
        stricter_prompt = full_prompt + f"\n\nVALIDAÇÃO: Usa APENAS estes recursos: {', '.join(set(valid_resources))}. Usa APENAS estes SKUs: {', '.join(set(valid_skus))}."
        try:
            reasoning = llm.generate(prompt=stricter_prompt, temperature=0.2, max_tokens=600, num_ctx=4096)
            validation = validate_llm_output(reasoning, context_data, "sugestoes", valid_skus, valid_resources)
        except:
            pass
    
    reasoning_final = validation.get("sanitized_text", reasoning) if validation.get("sanitized_text") else reasoning

    sections = _parse_sections(reasoning_final)

    # Extrair ação de forma mais robusta
    action_raw = sections.get("acoes", "")
    if action_raw:
        # Tentar extrair ação concreta
        action = _extract_action_from_text(action_raw, context_data)
        if action == "Ação a definir":
            # Fallback: primeira linha útil
            lines = [l.strip() for l in action_raw.split('\n') if l.strip() and len(l.strip()) > 15]
            action = lines[0] if lines else "Ação a definir"
    else:
        action = "Ação a definir"
    
    explanation = sections.get("causas", "").split("\n")[0] if sections.get("causas") else "Análise em curso"
    impact = sections.get("impacto", "").split("\n")[0] if sections.get("impacto") else "Impacto a calcular"
    gain = sections.get("ganho", "") if sections.get("ganho") else "Ganho a estimar"

    # Calcular impacto baseado no contexto de sugestões (context_data já foi definido acima)
    impact_value = 0.2

    # Calcular impacto baseado no contexto de sugestões
    gargalo_principal = context_data.get("gargalo_principal", {})
    skus_criticos = context_data.get("skus_criticos", [])
    recursos_alternativos = context_data.get("recursos_alternativos", [])
    
    if gargalo_principal.get("id") != "N/A":
        utilizacao = gargalo_principal.get("utilizacao", 0.0)
        impact_value = min(utilizacao, 0.4)
    elif skus_criticos:
        impact_value = min(len(skus_criticos) / 10.0, 0.4)
    elif recursos_alternativos:
        impact_value = 0.3

    return Suggestion(
        id=suggestion_id,
        icon="⚙️" if mode == "planeamento" else "📦" if mode == "inventario" else "🔴",
        action=action.replace("🔧", "").replace("**", "").strip(),
        explanation=explanation.replace("🧠", "").replace("**", "").strip(),
        impact=impact.replace("⚙️", "").replace("**", "").strip(),
        impact_level=_impact_level_from_value(impact_value),
        gain=gain.replace("💰", "").replace("**", "").strip(),
        reasoning_markdown=reasoning,
        data_points=data_points,
    )


def generate_suggestions(mode: Literal["planeamento", "gargalos", "inventario", "resumo"] = "resumo") -> List[Dict]:
    """Gera sugestões usando ActionCandidates do Insight Engine."""
    try:
        engine = InsightEngine()
        
        # Gerar action candidates baseados nos modelos ML
        action_candidates = engine.build_action_candidates()
        
        if not action_candidates:
            return []
        
        # Filtrar candidatos por modo (se necessário)
        filtered_candidates = action_candidates
        if mode == "gargalos":
            filtered_candidates = [c for c in action_candidates if c.get("tipo") in ["desvio_carga", "preventiva", "ajuste_overlap"]]
        elif mode == "inventario":
            filtered_candidates = [c for c in action_candidates if c.get("tipo") in ["reposicao_stock", "reducao_excesso"]]
        elif mode == "planeamento":
            filtered_candidates = [c for c in action_candidates if c.get("tipo") in ["desvio_carga", "ajuste_overlap", "colar_familias"]]
        # mode == "resumo" usa todos os candidatos
        
        if not filtered_candidates:
            return []
        
        # Converter action candidates para JSON
        candidates_json = json.dumps(filtered_candidates, ensure_ascii=False, indent=2)
        
        # Gerar sugestões usando LLM
        llm = LocalLLM()
        from app.insights.prompts import build_suggestions_prompt, SYSTEM_PROMPT
        
        prompt = build_suggestions_prompt(candidates_json)
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        
        try:
            # Temperatura 0.25 para sugestões (mais determinístico)
            llm_response = llm.generate(prompt=full_prompt, temperature=0.25, max_tokens=1200, num_ctx=4096)
        except LLMUnavailableError:
            raise
        
        # Validar output
        loader = get_loader()
        
        # Extrair SKUs e recursos válidos dos candidatos
        valid_skus = []
        valid_resources = []
        
        for candidate in filtered_candidates:
            alvo = candidate.get("alvo", "")
            if candidate.get("tipo") in ["reposicao_stock", "reducao_excesso"]:
                if alvo:
                    valid_skus.append(alvo)
            elif candidate.get("tipo") in ["desvio_carga", "preventiva"]:
                if alvo:
                    valid_resources.append(alvo)
                alternativa = candidate.get("alternativa")
                if alternativa:
                    valid_resources.append(alternativa)
        
        # Também buscar dos roteiros
        roteiros = loader.get_roteiros()
        if not roteiros.empty and "maquinas_possiveis" in roteiros.columns:
            all_resources = set()
            for machines in roteiros["maquinas_possiveis"].dropna():
                if isinstance(machines, str):
                    import re
                    resources = re.findall(r'M[-\s]?(\d+)', machines)
                    all_resources.update([f"M-{r}" for r in resources])
            valid_resources.extend(list(all_resources))
        
        # Criar contexto para validação (formato esperado pelo validator)
        context_for_validation = {
            "action_candidates": filtered_candidates
        }
        
        from app.llm.validator import validate_llm_output
        validation = validate_llm_output(llm_response, context_for_validation, "sugestoes", valid_skus, valid_resources)
        
        if not validation["is_valid"] and len(validation["errors"]) > 0:
            # Tentar regenerar com prompt mais restritivo
            stricter_prompt = full_prompt + f"\n\nVALIDAÇÃO: Usa APENAS estes recursos: {', '.join(set(valid_resources))}. Usa APENAS estes SKUs: {', '.join(set(valid_skus))}."
            try:
                llm_response = llm.generate(prompt=stricter_prompt, temperature=0.2, max_tokens=1200, num_ctx=4096)
                validation = validate_llm_output(llm_response, context_for_validation, "sugestoes", valid_skus, valid_resources)
            except:
                pass
        
        reasoning_final = validation.get("sanitized_text", llm_response) if validation.get("sanitized_text") else llm_response
        
        # Parsear resposta do LLM em sugestões individuais
        suggestions: List[Suggestion] = []
        
        # Dividir por números (1), 2), etc.
        import re
        suggestion_blocks = re.split(r'\n\s*\d+\)\s*', reasoning_final)
        
        for idx, block in enumerate(suggestion_blocks[1:], 1):  # Ignorar primeiro (antes do 1))
            lines = [l.strip() for l in block.split('\n') if l.strip()]
            if not lines:
                continue
            
            # Primeira linha é o título
            title = lines[0].strip()
            
            # Procurar "Impacto:" e "Porquê:"
            impacto = ""
            porquê = ""
            
            for i, line in enumerate(lines[1:], 1):
                if line.lower().startswith("impacto:"):
                    impacto = line.replace("Impacto:", "").strip()
                elif line.lower().startswith("porquê:") or line.lower().startswith("porque:"):
                    porquê = line.replace("Porquê:", "").replace("Porque:", "").strip()
            
            # Se não encontrou, usar linhas seguintes
            if not impacto and len(lines) > 1:
                impacto = lines[1] if len(lines) > 1 else ""
            if not porquê and len(lines) > 2:
                porquê = lines[2] if len(lines) > 2 else ""
            
            # Obter dados do candidato correspondente (se houver)
            candidate_data = filtered_candidates[idx - 1] if idx - 1 < len(filtered_candidates) else {}
            
            # Calcular impacto baseado no candidato
            impacto_estimado = candidate_data.get("impacto_estimado", {})
            impact_value = abs(impacto_estimado.get("delta_otd_pp", 0.0)) / 100.0
            
            suggestions.append(
                Suggestion(
                    id=f"suggestion-{idx}",
                    icon="⚙️" if candidate_data.get("tipo") in ["desvio_carga", "ajuste_overlap"] else "📦" if candidate_data.get("tipo") in ["reposicao_stock", "reducao_excesso"] else "🔧",
                    action=title,
                    explanation=porquê or "Análise baseada em dados reais",
                    impact=impacto or "Impacto a calcular",
                    impact_level=_impact_level_from_value(impact_value),
                    gain=f"Lead time: {impacto_estimado.get('delta_lead_time_h', 0.0):.1f}h, OTD: {impacto_estimado.get('delta_otd_pp', 0.0):.1f}pp",
                    reasoning_markdown=block,
                    data_points=candidate_data.get("motivacao", {}),
                )
            )
        
        # Se não conseguiu parsear, criar uma sugestão genérica
        if not suggestions:
            # Tentar extrair pelo menos uma ação
            first_candidate = filtered_candidates[0] if filtered_candidates else {}
            suggestions.append(
                Suggestion(
                    id="suggestion-1",
                    icon="⚙️",
                    action=reasoning_final.split('\n')[0][:100] if reasoning_final else "Ação recomendada",
                    explanation="Análise baseada em modelos ML",
                    impact="Impacto estimado baseado nos modelos",
                    impact_level="medio",
                    gain="A calcular",
                    reasoning_markdown=reasoning_final,
                    data_points=first_candidate.get("motivacao", {}),
                )
            )
        
        enriched: List[Dict] = []
        for suggestion in suggestions[:10]:  # Máximo 10 sugestões
            enriched.append(asdict(suggestion))

        return enriched
    except LLMUnavailableError:
        raise
    except Exception as exc:
        import traceback
        logger.error(f"Erro ao gerar sugestões: {exc}\n{traceback.format_exc()}")
        return []
