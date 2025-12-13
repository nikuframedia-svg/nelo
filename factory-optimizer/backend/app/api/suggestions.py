import logging
from typing import Literal, Dict, Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.insights.engine import InsightEngine
from app.etl.loader import get_loader

logger = logging.getLogger(__name__)
router = APIRouter()


def _translate_key(key: str) -> str:
    """Traduz chaves técnicas para português legível."""
    translations = {
        "utilizacao": "Utilização",
        "prob_gargalo": "Probabilidade de gargalo",
        "fila_h": "Fila acumulada",
        "fila_zero": "Fila zero",
        "risk_30d": "Risco de rutura (30 dias)",
        "coverage_dias": "Cobertura em dias",
        "stock_atual": "Stock atual",
        "rop": "Ponto de encomenda (ROP)",
        "stock_abaixo_rop": "Stock abaixo do ROP",
        "setup_hours": "Horas de setup",
        "setup_hours_before": "Horas de setup (antes)",
        "familias": "Famílias",
        "overlap_atual": "Overlap atual",
        "overlap_recomendado": "Overlap recomendado",
        "lead_time_after": "Lead time após otimização",
        "excesso": "Excesso de stock",
        "excesso_dias": "Dias de excesso",
        "classe": "Classe ABC",
    }
    return translations.get(key, key.replace("_", " ").title())


def _generate_human_explanation(
    tipo: str, dados_base: Dict[str, Any], impacto: Dict[str, Any], alvo: str, candidate: Dict[str, Any]
) -> str:
    """Gera explicação em palavras humanas baseada no tipo de ação e dados."""
    
    if tipo == "desvio_carga":
        utilizacao = dados_base.get("utilizacao", 0.0)
        prob_gargalo = dados_base.get("prob_gargalo", 0.0)
        fila_h = dados_base.get("fila_h", 0.0)
        fila_zero = dados_base.get("fila_zero", False)
        alternativa = candidate.get("alternativa", "")
        pct_desvio = candidate.get("pct_desvio", 0.0)
        
        if fila_zero:
            return (
                f"O recurso {alvo} está a operar a {utilizacao*100:.0f}% de capacidade com {prob_gargalo*100:.0f}% "
                f"de probabilidade de se tornar um gargalo. Embora não tenha fila acumulada no momento, "
                f"a alta utilização e probabilidade de gargalo indicam risco iminente. "
                f"Desviar {pct_desvio:.0f}% da carga para {alternativa} previne a formação de filas e "
                f"distribui melhor a carga entre recursos, reduzindo o risco de atrasos futuros."
            )
        else:
            return (
                f"O recurso {alvo} está saturado ({utilizacao*100:.0f}% de utilização) com uma fila de {fila_h:.0f}h "
                f"e {prob_gargalo*100:.0f}% de probabilidade de gargalo. Esta situação está a causar atrasos "
                f"significativos no lead time. Desviar {pct_desvio:.0f}% da carga para {alternativa} "
                f"liberta capacidade imediata, reduz a fila e melhora o fluxo de produção."
            )
    
    elif tipo == "reposicao_stock":
        risk_30d = dados_base.get("risk_30d", 0.0)
        coverage_dias = dados_base.get("coverage_dias", 0.0)
        stock_atual = dados_base.get("stock_atual", 0.0)
        rop = dados_base.get("rop", 0.0)
        stock_abaixo_rop = dados_base.get("stock_abaixo_rop", False)
        classe = dados_base.get("classe", "C")
        qty_repor = candidate.get("qty_repor", 0)
        
        if stock_abaixo_rop:
            return (
                f"O SKU {alvo} (classe {classe}) está abaixo do ponto de encomenda (ROP). "
                f"Stock atual: {stock_atual:.0f} unidades, ROP: {rop:.0f} unidades. "
                f"Com apenas {coverage_dias:.0f} dias de cobertura e {risk_30d:.1f}% de risco de rutura, "
                f"existe uma probabilidade alta de stockout nos próximos 30 dias. "
                f"Repor {qty_repor:.0f} unidades garante continuidade de produção e evita paragens."
            )
        else:
            return (
                f"O SKU {alvo} (classe {classe}) tem apenas {coverage_dias:.0f} dias de cobertura "
                f"e {risk_30d:.1f}% de risco de rutura nos próximos 30 dias. "
                f"Embora o stock atual ({stock_atual:.0f} unidades) esteja acima do ROP ({rop:.0f} unidades), "
                f"a baixa cobertura indica necessidade de reposição preventiva para evitar interrupções."
            )
    
    elif tipo == "preventiva":
        utilizacao = dados_base.get("utilizacao", 0.0)
        prob_gargalo = dados_base.get("prob_gargalo", 0.0)
        fila_h = dados_base.get("fila_h", 0.0)
        
        return (
            f"O recurso {alvo} está a operar a {utilizacao*100:.0f}% de capacidade com {prob_gargalo*100:.0f}% "
            f"de probabilidade de gargalo. Com {fila_h:.0f}h de fila acumulada, este recurso está sob "
            f"pressão extrema. Uma avaria neste momento causaria atrasos significativos. "
            f"Agendar manutenção preventiva reduz o risco de paragens não planeadas e garante "
            f"continuidade operacional."
        )
    
    elif tipo == "colar_familias":
        setup_hours = dados_base.get("setup_hours", 0.0)
        familias = dados_base.get("familias", [])
        gargalo_afetado = candidate.get("gargalo_afetado", "N/A")
        
        familias_str = ", ".join(familias[:3]) if familias else "várias famílias"
        if len(familias) > 3:
            familias_str += f" e mais {len(familias) - 3}"
        
        return (
            f"O setor {alvo} está a gastar {setup_hours:.1f}h por semana em trocas entre famílias "
            f"({familias_str}). Estas trocas frequentes interrompem a produção e reduzem a eficiência. "
            f"Colar famílias semelhantes em sequência reduz drasticamente o tempo de setup, "
            f"liberta capacidade no recurso {gargalo_afetado} e melhora o throughput geral."
        )
    
    elif tipo == "ajuste_overlap":
        overlap_atual = dados_base.get("overlap_atual", 0.0)
        overlap_recomendado = dados_base.get("overlap_recomendado", "15-25%")
        lead_time_after = dados_base.get("lead_time_after", 0.0)
        
        return (
            f"O setor {alvo} está a usar apenas {overlap_atual*100:.0f}% de overlap entre operações, "
            f"enquanto o recomendado é {overlap_recomendado}. O overlap permite que operações consecutivas "
            f"comecem antes da anterior terminar completamente, reduzindo o lead time total. "
            f"Aumentar o overlap para o nível recomendado pode reduzir o lead time de {lead_time_after:.0f}h "
            f"significativamente, melhorando a agilidade da produção."
        )
    
    elif tipo == "reducao_excesso":
        coverage_dias = dados_base.get("coverage_dias", 0.0)
        excesso_dias = dados_base.get("excesso_dias", 0.0)
        stock_atual = dados_base.get("stock_atual", 0.0)
        
        return (
            f"O SKU {alvo} tem {coverage_dias:.0f} dias de cobertura ({excesso_dias:.0f} dias acima do ideal), "
            f"representando {stock_atual:.0f} unidades em stock. Este excesso imobiliza capital desnecessariamente "
            f"e aumenta o risco de obsolescência. Reduzir o stock para níveis mais adequados liberta capital "
            f"para investimento noutras áreas e reduz custos de armazenamento."
        )
    
    else:
        return (
            f"Esta ação foi recomendada com base na análise dos dados de produção e inventário. "
            f"Os indicadores mostram uma oportunidade de melhoria que pode ter impacto positivo nos KPIs."
        )


@router.get("/")
async def get_suggestions(
    mode: Literal["planeamento", "gargalos", "inventario", "resumo"] = Query("resumo", description="Modo de sugestões")
):
    """Retorna sugestões inteligentes baseadas no modo usando InsightEngine 2.0."""
    try:
        # Verificar se há dados carregados
        loader = get_loader()
        status = loader.get_status()
        
        if not status.get("has_data", False):
            return {
                "count": 0,
                "items": [],
                "mode": mode,
                "detail": "Carregue os ficheiros de produção e stocks para ativar o motor de recomendações."
            }
        
        # Usar o novo InsightEngine
        engine = InsightEngine()
        action_candidates = engine.build_action_candidates()
        
        if not action_candidates:
            return {
                "count": 0,
                "items": [],
                "mode": mode,
                "detail": "Ainda não existem sugestões. Carregue os ficheiros de produção e stocks para ativar o motor de recomendações."
            }
        
        # Filtrar por modo
        filtered_candidates = action_candidates
        if mode == "gargalos":
            filtered_candidates = [c for c in action_candidates if c.get("tipo") in ["desvio_carga", "preventiva", "ajuste_overlap"]]
        elif mode == "inventario":
            filtered_candidates = [c for c in action_candidates if c.get("tipo") in ["reposicao_stock", "reducao_excesso"]]
        elif mode == "planeamento":
            filtered_candidates = [c for c in action_candidates if c.get("tipo") in ["desvio_carga", "ajuste_overlap", "colar_familias"]]
        # mode == "resumo" usa todos os candidatos
        
        if not filtered_candidates:
            return {
                "count": 0,
                "items": [],
                "mode": mode,
                "detail": "Ainda não existem sugestões para este modo. Carregue os ficheiros de produção e stocks."
            }
        
        # Converter para formato esperado pelo frontend
        suggestions = []
        for idx, candidate in enumerate(filtered_candidates[:10], 1):
            tipo = candidate.get("tipo", "")
            alvo = candidate.get("alvo", "")
            dados_base = candidate.get("dados_base", {})
            impacto = candidate.get("impacto_estimado", {})
            prioridade = candidate.get("prioridade", "BAIXO")
            
            # Gerar título baseado no tipo
            if tipo == "desvio_carga":
                alternativa = candidate.get("alternativa", "")
                pct = candidate.get("pct_desvio", 0.0)
                titulo = f"Desviar {pct}% de carga de {alvo} para {alternativa}"
            elif tipo == "reposicao_stock":
                sku = candidate.get("sku", alvo)
                qty = candidate.get("qty_repor", 0)
                titulo = f"Repor {qty} unidades do SKU {sku}"
            elif tipo == "preventiva":
                titulo = f"Agendar manutenção preventiva em {alvo}"
            elif tipo == "colar_familias":
                titulo = f"Colar famílias no {alvo}"
            elif tipo == "ajuste_overlap":
                titulo = f"Aumentar overlap no setor {alvo}"
            elif tipo == "reducao_excesso":
                sku = candidate.get("sku", alvo)
                titulo = f"Reduzir stock excessivo do SKU {sku}"
            else:
                titulo = f"Ação: {tipo} em {alvo}"
            
            # Calcular impacto level baseado em delta_otd_pp
            delta_otd = abs(impacto.get("delta_otd_pp", 0.0))
            impact_level = "alto" if delta_otd >= 2.5 else "medio" if delta_otd >= 1.0 else "baixo"
            
            # Icon baseado no tipo
            if tipo in ["desvio_carga", "ajuste_overlap"]:
                icon = "⚙️"
            elif tipo in ["reposicao_stock", "reducao_excesso"]:
                icon = "📦"
            else:
                icon = "🔧"
            
            # Gerar explicação baseada em dados_base
            explicacao_parts = []
            if dados_base.get("utilizacao"):
                explicacao_parts.append(f"Utilização: {dados_base['utilizacao']*100:.1f}%")
            if dados_base.get("prob_gargalo"):
                explicacao_parts.append(f"Prob. gargalo: {dados_base['prob_gargalo']*100:.1f}%")
            if dados_base.get("risk_30d"):
                explicacao_parts.append(f"Risco 30d: {dados_base['risk_30d']:.1f}%")
            if dados_base.get("coverage_dias"):
                explicacao_parts.append(f"Cobertura: {dados_base['coverage_dias']:.1f} dias")
            if dados_base.get("setup_hours"):
                explicacao_parts.append(f"Setup: {dados_base['setup_hours']:.1f}h")
            
            explicacao = f"Prioridade: {prioridade}" + (f" | {', '.join(explicacao_parts)}" if explicacao_parts else "")
            
            # Formatar impacto
            impacto_parts = []
            if impacto.get("delta_lead_time_h"):
                impacto_parts.append(f"Lead time: {impacto['delta_lead_time_h']:.1f}h")
            if impacto.get("delta_otd_pp"):
                impacto_parts.append(f"OTD: {impacto['delta_otd_pp']:.1f}pp")
            if impacto.get("delta_fila_h"):
                impacto_parts.append(f"Fila: {impacto['delta_fila_h']:.1f}h")
            if impacto.get("delta_setup_h"):
                impacto_parts.append(f"Setup: {impacto['delta_setup_h']:.1f}h")
            if impacto.get("delta_risk_30d"):
                impacto_parts.append(f"Risco: {impacto['delta_risk_30d']:.1f}%")
            
            impacto_str = ", ".join(impacto_parts) if impacto_parts else "Impacto a calcular"
            
            # Formatar ganho (mesmo que impacto)
            ganho_str = impacto_str
            
            # Gerar explicação em palavras humanas baseada no tipo e dados_base
            explicacao_humana = _generate_human_explanation(tipo, dados_base, impacto, alvo, candidate)
            
            # Gerar reasoning markdown com explicação humana + dados técnicos
            reasoning_parts = [
                f"**Porquê esta sugestão?**",
                f"",
                f"{explicacao_humana}",
                f"",
                f"**Impacto esperado:** {impacto_str}",
                f"**Prioridade:** {prioridade}",
                f"",
                f"**Dados técnicos:**",
            ]
            for key, value in dados_base.items():
                if isinstance(value, (int, float)):
                    # Formatar chaves para português legível
                    key_pt = _translate_key(key)
                    if "utilizacao" in key:
                        reasoning_parts.append(f"- {key_pt}: {value*100:.1f}%")
                    elif "prob" in key or "risk" in key:
                        reasoning_parts.append(f"- {key_pt}: {value*100:.1f}%")
                    elif "coverage" in key or "dias" in key:
                        reasoning_parts.append(f"- {key_pt}: {value:.1f} dias")
                    elif "hours" in key or "h" in key:
                        reasoning_parts.append(f"- {key_pt}: {value:.1f}h")
                    else:
                        reasoning_parts.append(f"- {key_pt}: {value}")
                elif isinstance(value, bool):
                    key_pt = _translate_key(key)
                    reasoning_parts.append(f"- {key_pt}: {'Sim' if value else 'Não'}")
                elif isinstance(value, list):
                    key_pt = _translate_key(key)
                    reasoning_parts.append(f"- {key_pt}: {', '.join(map(str, value))}")
                else:
                    key_pt = _translate_key(key)
                    reasoning_parts.append(f"- {key_pt}: {value}")
            
            reasoning_markdown = "\n".join(reasoning_parts)
            
            # Extrair dados técnicos do candidate
            dados_tecnicos = candidate.get("dados_tecnicos", {})
            
            suggestions.append({
                "id": f"suggestion-{idx}",
                "icon": icon,
                "action": titulo,
                "explanation": explicacao,
                "impact": impacto_str,
                "impact_level": impact_level,
                "gain": ganho_str,
                "reasoning_markdown": reasoning_markdown,
                "data_points": dados_base,  # Para compatibilidade
                "dados_base": dados_base,  # Dados que justificam a ação
                "dados_tecnicos": dados_tecnicos,  # Dados técnicos completos
                "impacto_estimado": impacto,  # Impacto estruturado
                "prioridade": prioridade,
            })
        
        return {
            "count": len(suggestions),
            "items": suggestions,
            "mode": mode,
        }
    except Exception as exc:
        logger.exception(f"Erro ao gerar sugestões: {exc}")
        return {
            "count": 0,
            "items": [],
            "mode": mode,
            "detail": f"Erro ao gerar sugestões: {str(exc)}"
        }
