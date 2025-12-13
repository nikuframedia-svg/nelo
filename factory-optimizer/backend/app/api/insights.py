"""API endpoints para o Insight Engine."""

import json
import logging
import re
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.insights.engine import InsightEngine
from app.insights.cache import get_insight_cache
from app.llm import LLMUnavailableError, LocalLLM
from app.llm.validator import validate_llm_output
from app.llm.industrial_validator import IndustrialLLMValidator
from app.insights.prompts import SYSTEM_PROMPT, get_prompt_by_mode
from app.etl.loader import get_loader

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/context")
async def get_insight_context(
    mode: Literal["planeamento", "gargalos", "inventario", "resumo", "sugestoes"] = Query(
        "resumo", description="Modo de contexto"
    ),
    batch_id: Optional[str] = Query(None, description="Batch ID (opcional, usa o mais recente se não fornecido)"),
):
    """Retorna contexto estruturado por modo."""
    try:
        engine = InsightEngine()
        context = engine.build_context_by_mode(mode)
        return context
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha ao gerar contexto: {exc}") from exc


@router.get("/action-candidates")
async def get_action_candidates(
    batch_id: Optional[str] = Query(None, description="Batch ID (opcional)"),
):
    """
    Retorna ActionCandidates estruturados (sem LLM).
    Frontend pode usar diretamente para renderizar cards ou passar para LLM.
    """
    try:
        engine = InsightEngine()
        candidates = engine.build_action_candidates()
        
        # Formatar como cards estruturados
        cards = []
        for candidate in candidates:
            # Gerar título baseado no tipo
            tipo = candidate.get("tipo", "")
            alvo = candidate.get("alvo", "")
            titulo = ""
            
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
            
            cards.append({
                "acao": tipo,
                "titulo": titulo,
                "dados_base": candidate.get("dados_base", {}),
                "impacto_estimado": candidate.get("impacto_estimado", {}),
                "prioridade": candidate.get("prioridade", "BAIXO"),
                "alvo": alvo,
                "gargalo_afetado": candidate.get("gargalo_afetado"),
                "alternativa": candidate.get("alternativa"),
                "sku": candidate.get("sku"),
            })
        
        return {
            "count": len(cards),
            "cards": cards,
            "batch_id": batch_id,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha ao gerar action candidates: {exc}") from exc


@router.get("/generate")
async def generate_insight(
    mode: Literal["planeamento", "gargalos", "inventario", "resumo", "sugestoes"] = Query(
        "resumo", description="Modo de geração"
    ),
    batch_id: Optional[str] = Query(None, description="Batch ID (opcional, usa o mais recente se não fornecido)"),
    user_message: str = Query("", description="Mensagem opcional do utilizador"),
):
    """
    Gera insight LLM baseado no contexto do modo.
    Usa cache por batch_id+mode para evitar chamadas LLM desnecessárias.
    """
    try:
        # Obter batch_id atual se não fornecido
        if not batch_id:
            loader = get_loader()
            status = loader.get_status()
            # Tentar obter batch_id mais recente do status
            batch_id = status.get("latest_batch_id") or status.get("batch_id") or "default"
        
        # Verificar cache
        cache = get_insight_cache()
        cached_insight = cache.get(batch_id, mode)
        
        if cached_insight and not user_message:
            # Validar cache antes de devolver (especialmente para planeamento)
            if mode == "planeamento":
                # Verificar se o cache tem estrutura antiga ou placeholders
                has_old_structure = any(
                    pattern in cached_insight.lower()
                    for pattern in ["resumo executivo", "1️⃣", "2️⃣", "3️⃣", "💰 Δ", "📈 overlap", "a fábrica opera com poucas ordens"]
                )
                has_placeholders = any(
                    re.search(pattern, cached_insight, re.IGNORECASE)
                    for pattern in [r'\b(X|Y|Z|W|V|U)\s*(h|%|horas|unidades|pp)\b', r'\b[X-Z]h\b', r'\b[X-Z]\s*%\b']
                )
                starts_wrong = not cached_insight.strip().lower().startswith("nesta demonstração")
                # Verificar se não contém o novo texto esperado
                has_new_text = "análise industrial do motor nikufra ops" in cached_insight.lower() or "estudo do plano antes e depois" in cached_insight.lower()
                
                if has_old_structure or has_placeholders or starts_wrong or not has_new_text:
                    # Cache inválido - invalidar e regenerar
                    logger.warning(f"Cache inválido detectado para {batch_id}:{mode}. Invalidando e regenerando.")
                    cache.invalidate(batch_id, mode)
                    cached_insight = None
            
            if cached_insight:
                # Retornar do cache se válido
                logger.info(f"Cache hit para {batch_id}:{mode}")
                return {
                    "mode": mode,
                    "answer": cached_insight,
                    "insight": cached_insight,  # Campo adicional para compatibilidade com frontend
                    "model": "cached",
                    "batch_id": batch_id,
                    "cached": True,
                }
        
        # Cache miss ou user_message presente - gerar novo insight
        logger.info(f"Cache miss para {batch_id}:{mode}, gerando novo insight")
        
        try:
            engine = InsightEngine()
            context = engine.build_context_by_mode(mode)
            context_json = json.dumps(context, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.exception(f"Erro ao construir contexto para {mode}")
            raise HTTPException(status_code=500, detail=f"Erro ao construir contexto: {str(exc)}") from exc

        try:
            prompt = get_prompt_by_mode(mode, context_json, user_question=user_message if user_message else None)
            full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        except Exception as exc:
            logger.exception(f"Erro ao construir prompt para {mode}")
            raise HTTPException(status_code=500, detail=f"Erro ao construir prompt: {str(exc)}") from exc

        if user_message:
            full_prompt += f"\n\nPergunta do utilizador: {user_message}"

        # Temperaturas e max_tokens por modo (otimizados)
        temperature_map = {
            "planeamento": 0.25,
            "gargalos": 0.25,
            "inventario": 0.25,
            "resumo": 0.3,
            "sugestoes": 0.25,
        }
        max_tokens_map = {
            "planeamento": 400,
            "gargalos": 350,
            "inventario": 350,
            "resumo": 300,
            "sugestoes": 400,
        }
        
        temperature = temperature_map.get(mode, 0.3)
        max_tokens = max_tokens_map.get(mode, 350)
        num_ctx = 2048  # Reduzido de 4096 para melhor performance

        llm = LocalLLM()
        try:
            answer_raw = llm.generate(
                prompt=full_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                num_ctx=num_ctx
            )
            answer = answer_raw.strip()
        except LLMUnavailableError:
            return JSONResponse({"detail": "Modelo offline — iniciar Ollama."}, status_code=200)

        # Validar output do LLM
        try:
            loader = get_loader()
            valid_skus = []
            valid_resources = []
            
            # Extrair SKUs e recursos válidos do contexto
            if mode == "inventario":
                try:
                    inventory_data = loader.get_inventory_insights()
                    valid_skus = [sku.get("sku", "") for sku in inventory_data.get("skus", [])] if inventory_data else []
                except Exception as exc:
                    logger.warning(f"Erro ao obter inventory insights: {exc}")
                    valid_skus = []
            elif mode in ["gargalos", "planeamento"]:
                try:
                    # Para planeamento, extrair recursos do contexto de planeamento
                    if mode == "planeamento":
                        # Extrair recursos do contexto de planeamento
                        gargalo_principal = context.get("gargalo_principal", {})
                        gargalo_id = str(gargalo_principal.get("id", "")).replace("M-", "").replace("M", "").strip()
                        if gargalo_id and gargalo_id.isdigit():
                            valid_resources = [gargalo_id]
                        
                        # Também buscar recursos dos roteiros e operações
                        roteiros = loader.get_roteiros()
                        if roteiros is not None and not roteiros.empty and "maquinas_possiveis" in roteiros.columns:
                            all_resources = set()
                            for machines in roteiros["maquinas_possiveis"].dropna():
                                if isinstance(machines, str):
                                    # Aceitar tanto "M-27" quanto "27"
                                    resources = re.findall(r'M[-\s]?(\d+)', machines)
                                    all_resources.update(resources)  # Guardar sem prefixo M-
                            valid_resources.extend(list(all_resources))
                            valid_resources = list(set(valid_resources))  # Remover duplicados
                    else:
                        # Para gargalos, usar a lógica original
                        bottlenecks_insights = engine._extract_bottlenecks_insights()
                        valid_resources = [r.get("recurso", "") for r in bottlenecks_insights.get("top_resources", [])] if bottlenecks_insights else []
                        # Normalizar recursos (remover M- se existir)
                        valid_resources = [str(r).replace("M-", "").replace("M", "").strip() for r in valid_resources if r]
                        # Também buscar recursos dos roteiros
                        roteiros = loader.get_roteiros()
                        if roteiros is not None and not roteiros.empty and "maquinas_possiveis" in roteiros.columns:
                            all_resources = set()
                            for machines in roteiros["maquinas_possiveis"].dropna():
                                if isinstance(machines, str):
                                    resources = re.findall(r'M[-\s]?(\d+)', machines)
                                    all_resources.update(resources)  # Guardar sem prefixo M-
                            valid_resources.extend(list(all_resources))
                            valid_resources = list(set(valid_resources))  # Remover duplicados
                except Exception as exc:
                    logger.warning(f"Erro ao obter recursos para validação: {exc}")
                    valid_resources = []
        except Exception as exc:
            logger.warning(f"Erro ao obter dados para validação: {exc}")
            valid_skus = []
            valid_resources = []
        
        # Validação dupla: validador original + validador industrial enterprise
        try:
            validation = validate_llm_output(answer, context, mode, valid_skus, valid_resources)
        except Exception as exc:
            logger.warning(f"Erro na validação básica: {exc}")
            validation = {"is_valid": True, "errors": [], "warnings": [], "sanitized_text": answer, "should_regenerate": False}
        
        # Validação industrial rigorosa (nível enterprise)
        try:
            industrial_validator = IndustrialLLMValidator()
            action_candidates = context.get("actions", []) if mode == "sugestoes" else None
            industrial_validation = industrial_validator.validate(
                answer, context, mode, valid_skus, valid_resources, action_candidates
            )
        except Exception as exc:
            logger.warning(f"Erro na validação industrial: {exc}")
            industrial_validation = {"is_valid": True, "errors": [], "warnings": [], "sanitized_text": answer, "should_regenerate": False}
        
        # Combinar validações
        all_errors = validation.get("errors", []) + industrial_validation.get("errors", [])
        all_warnings = validation.get("warnings", []) + industrial_validation.get("warnings", [])
        is_valid = validation.get("is_valid", False) and industrial_validation.get("is_valid", False)
        should_regenerate = validation.get("should_regenerate", False) or industrial_validation.get("should_regenerate", False)
        
        # Se houver erros críticos ou estrutura antiga, tentar regenerar
        if not is_valid and (should_regenerate or len(all_errors) > 0):
            # Prompt mais rigoroso para planeamento
            if mode == "planeamento":
                stricter_prompt = full_prompt + "\n\n⚠️ REGRAS ABSOLUTAS CRÍTICAS:\n- O resumo DEVE começar EXATAMENTE com: 'Nesta demonstração, o sistema identificou um gargalo estrutural claro no recurso...'\n- NÃO uses placeholders genéricos (X, Y, Z, W, V, U).\n- NÃO uses a estrutura antiga ('Resumo Executivo', '1️⃣ Plano Antes', etc.).\n- Segue EXATAMENTE o exemplo fornecido no prompt (parágrafos corridos, sem bullets).\n- Podes mencionar recursos pelo número (ex: 27, 29, 248), desde que estejam no contexto.\n- Podes mencionar: gargalo, setup, famílias, sequência, ordens, carga, capacidade, produção, operações, turnos.\n- É PROIBIDO mencionar: stock, inventário, ABC, XYZ, ROP, risco, compras.\n- Se não tens um valor real, NÃO o inventes.\n- O resumo deve terminar com: 'Este resumo reflete exclusivamente os dados presentes na demo atual, sem extrapolações e sem misturas com inventário, stocks ou outros módulos.'"
            else:
                stricter_prompt = full_prompt + "\n\n⚠️ REGRAS ABSOLUTAS:\n- Usa APENAS SKUs e recursos que estão explicitamente no contexto JSON.\n- NÃO inventes números, recursos ou SKUs.\n- NÃO mistures módulos.\n- NÃO uses frases genéricas sem números."
            try:
                answer = llm.generate(prompt=stricter_prompt, temperature=temperature * 0.7, max_tokens=max_tokens, num_ctx=num_ctx).strip()
                # Re-validar
                validation = validate_llm_output(answer, context, mode, valid_skus, valid_resources)
                industrial_validation = industrial_validator.validate(
                    answer, context, mode, valid_skus, valid_resources, action_candidates
                )
                all_errors = validation.get("errors", []) + industrial_validation.get("errors", [])
                is_valid = validation.get("is_valid", False) and industrial_validation.get("is_valid", False)
                should_regenerate = validation.get("should_regenerate", False) or industrial_validation.get("should_regenerate", False)
            except:
                pass
        
        # Usar texto sanitizado do validador industrial (mais rigoroso)
        final_answer = industrial_validation.get("sanitized_text") or validation.get("sanitized_text") or answer
        
        # Se ainda houver problemas críticos após regeneração, invalidar cache e não guardar
        if not is_valid and (should_regenerate or len(all_errors) > 0):
            # Invalidar cache para forçar regeneração na próxima vez
            cache.invalidate(batch_id, mode)
            logger.warning(f"Resumo de {mode} não passou na validação. Cache invalidado. Erros: {all_errors}")
            # Ainda assim devolver o texto sanitizado, mas marcar como inválido
            final_answer = f"[AVISO: Resumo não passou na validação. Erros: {', '.join(all_errors[:3])}] {final_answer}"
        
        # Guardar no cache apenas se passar na validação e não houver user_message
        if not user_message and is_valid and not should_regenerate:
            cache.set(batch_id, mode, final_answer, metadata={"mode": mode, "batch_id": batch_id})

        return {
            "mode": mode,
            "answer": final_answer,
            "insight": final_answer,  # Campo adicional para compatibilidade com frontend
            "model": getattr(llm, "model_name", "llama3:8b"),
            "batch_id": batch_id,
            "cached": False,
            "context": context,
        }
    except HTTPException:
        # Re-raise HTTPExceptions (já têm status code apropriado)
        raise
    except Exception as exc:
        logger.exception(f"Falha ao gerar insight para {mode}")
        # Retornar mensagem de erro mais amigável
        error_detail = str(exc)
        if "LLMUnavailableError" in error_detail or "Ollama" in error_detail:
            return JSONResponse({"detail": "Modelo offline — iniciar Ollama."}, status_code=200)
        raise HTTPException(status_code=500, detail=f"Erro ao gerar insight: {error_detail}") from exc


@router.delete("/cache/{batch_id}")
async def invalidate_cache(batch_id: str, mode: Optional[str] = Query(None)):
    """Invalida cache de insights para um batch_id (e opcionalmente um mode específico)."""
    cache = get_insight_cache()
    cache.invalidate(batch_id, mode)
    return {"ok": True, "message": f"Cache invalidado para batch_id={batch_id}, mode={mode or 'todos'}"}

