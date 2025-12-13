"""
API para Chat de Planeamento - ProdPlan 4.0

Endpoints:
- POST /api/planning/chat/interpret: LLM interpreta linguagem natural → comando estruturado
- POST /api/planning/chat/apply: Aplica comando e recalcula plano
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.aps.planning_commands import PlanningCommand
from app.aps.planning_config import get_planning_config_store
from app.aps.planning_prompts import build_planning_chat_prompt, build_explanation_prompt
from app.aps.technical_queries import get_technical_queries
from app.etl.loader import get_loader
from app.llm.local import LocalLLM, LLMUnavailableError

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    """Request para interpretar comando de planeamento."""
    message: str
    batch_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response com comando interpretado."""
    command: dict
    requires_clarification: bool
    clarification_message: Optional[str] = None
    confidence: float


class ApplyCommandRequest(BaseModel):
    """Request para aplicar comando."""
    command: dict
    batch_id: Optional[str] = None


@router.post("/interpret")
async def interpret_planning_command(request: ChatRequest):
    """
    Interpreta frase do utilizador e gera comando estruturado.
    
    O LLM NUNCA gera código, patches ou modifica ficheiros.
    Apenas traduz intenção → PlanningCommand estruturado.
    
    Request esperado:
    {
        "message": "máquina 300 indisponível",
        "batch_id": "20251118225728" (opcional)
    }
    """
    # Log entrada
    logger.info(f"📥 /interpret recebido: message='{request.message}', batch_id={request.batch_id}")
    
    if not request.message or not request.message.strip():
        logger.warning("❌ /interpret: Mensagem vazia")
        raise HTTPException(status_code=400, detail="Mensagem vazia")
    
    # Obter batch_id
    if not request.batch_id:
        loader = get_loader()
        status = loader.get_status()
        batch_id = status.get("latest_batch_id") or status.get("batch_id") or "default"
    else:
        batch_id = request.batch_id
    
    try:
        # Construir contexto (máquinas disponíveis, artigos, etc.)
        context = _build_planning_context(batch_id)
        
        # Construir prompt
        prompt = build_planning_chat_prompt(request.message, context)
        
        # Chamar LLM
        llm = LocalLLM()
        response_text = llm.generate(
            prompt=prompt,
            temperature=0.3,  # Baixa temperatura para respostas mais determinísticas
            max_tokens=500,
            num_ctx=2048,
        ).strip()
        
        # Parse JSON da resposta
        # Remover markdown code blocks se existirem
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        try:
            command_dict = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Erro ao fazer parse do JSON do LLM: {e}\nResposta: {response_text}")
            # Tentar extrair JSON manualmente
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                command_dict = json.loads(json_match.group())
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"LLM não retornou JSON válido. Resposta: {response_text[:200]}"
                )
        
        # Log do que o LLM retornou ANTES de qualquer processamento
        logger.info(f"🔍 LLM retornou (raw): {json.dumps(command_dict, indent=2, default=str)}")
        
        # Normalizar command_type antes de validar (aceitar variações)
        if "command_type" in command_dict:
            command_type_str = str(command_dict["command_type"]).lower()
            logger.info(f"🔍 command_type do LLM (normalizado): '{command_type_str}'")
            # Mapear variações comuns
            if command_type_str in ["machine_unavailability", "machine_unavailable"]:
                command_dict["command_type"] = "machine_unavailable"
            elif command_type_str in ["manual_order", "add_manual_order"]:
                command_dict["command_type"] = "add_manual_order"
            elif command_type_str in ["priority_change", "change_priority"]:
                command_dict["command_type"] = "change_priority"
            elif command_type_str in ["horizon_change", "change_horizon"]:
                command_dict["command_type"] = "change_horizon"
            elif command_type_str in ["recalculate_plan", "recalculate", "refresh_plan", "refresh", "replan", "replanning"]:
                command_dict["command_type"] = "recalculate_plan"
        
        command_type_after_normalization = command_dict.get("command_type", "").lower()
        logger.info(f"🔍 command_type após normalização: '{command_type_after_normalization}'")
        
        # FALLBACK ROBUSTO: Se o LLM retornou "unknown" OU se o comando está incompleto, tentar inferir do texto original
        # Este fallback é CRÍTICO - nunca devemos aceitar "unknown" para frases claras
        # REGRA DE OURO: "máquina 300 indisponível" deve SEMPRE funcionar
        # Vamos também verificar se o comando está incompleto (ex: command_type válido mas falta machine_unavailable)
        should_use_fallback = False
        
        # PRIMEIRA VERIFICAÇÃO: Se o LLM retornou "unknown"
        if command_type_after_normalization == "unknown":
            should_use_fallback = True
            logger.warning(f"⚠️ LLM retornou 'unknown' para: '{request.message}'. Ativando fallback...")
        # SEGUNDA VERIFICAÇÃO: Se o comando está incompleto (falta machine_unavailable ou maquina_id)
        elif command_type_after_normalization == "machine_unavailable":
            if not command_dict.get("machine_unavailable") or not command_dict.get("machine_unavailable", {}).get("maquina_id"):
                should_use_fallback = True
                logger.warning(f"⚠️ LLM retornou 'machine_unavailable' mas comando está incompleto. Ativando fallback para completar...")
        # SEGUNDA-B: Se o comando está incompleto (falta machine_available ou maquina_id)
        elif command_type_after_normalization == "machine_available":
            if not command_dict.get("machine_available") or not command_dict.get("machine_available", {}).get("maquina_id"):
                should_use_fallback = True
                logger.warning(f"⚠️ LLM retornou 'machine_available' mas comando está incompleto. Ativando fallback para completar...")
        # SEGUNDA-C: Se o comando está incompleto (falta priority_change ou order_id/priority)
        elif command_type_after_normalization == "change_priority":
            if not command_dict.get("priority_change") or not command_dict.get("priority_change", {}).get("order_id") or not command_dict.get("priority_change", {}).get("new_priority"):
                should_use_fallback = True
                logger.warning(f"⚠️ LLM retornou 'change_priority' mas comando está incompleto. Ativando fallback para completar...")
        # TERCEIRA VERIFICAÇÃO (MAIS AGRESSIVA): Se a mensagem claramente é "máquina X indisponível", "máquina X disponível" ou "GOX VIP", 
        # SEMPRE usar fallback, mesmo que o LLM tenha retornado algo diferente
        else:
            # Verificar se a mensagem claramente indica indisponibilidade ou disponibilidade de máquina
            quick_check = _infer_command_type(request.message, context)
            if quick_check == "machine_unavailable":
                # A mensagem claramente é "máquina X indisponível"
                # Se o LLM não retornou isso ou retornou incompleto, usar fallback
                if command_type_after_normalization != "machine_unavailable":
                    should_use_fallback = True
                    logger.warning(f"⚠️ Mensagem claramente é 'máquina X indisponível' mas LLM retornou '{command_type_after_normalization}'. Forçando fallback...")
                elif not command_dict.get("machine_unavailable") or not command_dict.get("machine_unavailable", {}).get("maquina_id"):
                    should_use_fallback = True
                    logger.warning(f"⚠️ Mensagem claramente é 'máquina X indisponível' mas comando está incompleto. Forçando fallback...")
            elif quick_check == "machine_available":
                # A mensagem claramente é "máquina X disponível"
                # Se o LLM não retornou isso ou retornou incompleto, usar fallback
                if command_type_after_normalization != "machine_available":
                    should_use_fallback = True
                    logger.warning(f"⚠️ Mensagem claramente é 'máquina X disponível' mas LLM retornou '{command_type_after_normalization}'. Forçando fallback...")
                elif not command_dict.get("machine_available") or not command_dict.get("machine_available", {}).get("maquina_id"):
                    should_use_fallback = True
                    logger.warning(f"⚠️ Mensagem claramente é 'máquina X disponível' mas comando está incompleto. Forçando fallback...")
            elif quick_check == "change_priority":
                # A mensagem claramente é "GOX VIP" ou similar
                # Se o LLM não retornou isso ou retornou incompleto, usar fallback
                if command_type_after_normalization != "change_priority":
                    should_use_fallback = True
                    logger.warning(f"⚠️ Mensagem claramente é 'GOX VIP' mas LLM retornou '{command_type_after_normalization}'. Forçando fallback...")
                elif not command_dict.get("priority_change") or not command_dict.get("priority_change", {}).get("order_id") or not command_dict.get("priority_change", {}).get("new_priority"):
                    should_use_fallback = True
                    logger.warning(f"⚠️ Mensagem claramente é 'GOX VIP' mas comando está incompleto. Forçando fallback...")
            elif quick_check == "recalculate_plan":
                # A mensagem claramente é "otimiza o plano" ou similar
                # Se o LLM não retornou isso, usar fallback
                if command_type_after_normalization != "recalculate_plan":
                    should_use_fallback = True
                    logger.warning(f"⚠️ Mensagem claramente é 'recalcular plano' mas LLM retornou '{command_type_after_normalization}'. Forçando fallback...")
        
        if should_use_fallback:
            logger.info(f"🔍 Contexto disponível: máquinas={context.get('available_machines', [])[:10]}, horizon={context.get('current_horizon', 24)}h")
            logger.info(f"🔍 Mensagem original: '{request.message}'")
            
            # Se já temos command_type mas está incompleto, usar esse tipo
            # Senão, tentar inferir
            if command_type_after_normalization != "unknown" and command_type_after_normalization:
                inferred_type = command_type_after_normalization
                logger.info(f"🔍 Usando command_type do LLM (mas completando com fallback): {inferred_type}")
            else:
                inferred_type = _infer_command_type(request.message, context)
                logger.info(f"🔍 _infer_command_type retornou: {inferred_type}")
            
            if inferred_type:
                logger.info(f"✅ Fallback inferiu '{inferred_type}' da mensagem: '{request.message}'")
                command_dict["command_type"] = inferred_type
                command_dict["confidence"] = 0.9  # Alta confiança (fallback é robusto)
                command_dict["requires_clarification"] = False
                
                # Tentar extrair máquina_id se for indisponibilidade ou disponibilidade
                if inferred_type == "machine_unavailable":
                    logger.info(f"🔍 Tentando extrair máquina_id de: '{request.message}'")
                    machine_id = _extract_machine_id(request.message, context)
                    logger.info(f"🔍 _extract_machine_id retornou: {machine_id}")
                    
                    if machine_id:
                        if "machine_unavailable" not in command_dict:
                            command_dict["machine_unavailable"] = {}
                        command_dict["machine_unavailable"]["maquina_id"] = machine_id
                        # Adicionar datas padrão (sempre, mesmo que não especificadas)
                        now = datetime.utcnow()
                        horizon_hours = context.get("current_horizon", 24)
                        command_dict["machine_unavailable"]["start_time"] = now.isoformat()
                        command_dict["machine_unavailable"]["end_time"] = (now + timedelta(hours=horizon_hours)).isoformat()
                        logger.info(f"✅ Comando machine_unavailable criado via fallback: máquina {machine_id}, {now.isoformat()} até {(now + timedelta(hours=horizon_hours)).isoformat()}")
                    else:
                        # Se não conseguir extrair máquina_id, é erro real
                        logger.error(f"❌ Não foi possível extrair máquina_id de: '{request.message}'")
                        logger.error(f"❌ Máquinas disponíveis: {context.get('available_machines', [])}")
                        command_dict["command_type"] = "unknown"
                        command_dict["confidence"] = 0.3
                        command_dict["requires_clarification"] = True
                        command_dict["clarification_message"] = f"Não consegui identificar qual máquina na mensagem '{request.message}'. Tente especificar o número da máquina, por exemplo: 'máquina 300 indisponível'. Máquinas disponíveis: {', '.join(context.get('available_machines', [])[:10])}"
                
                elif inferred_type == "machine_available":
                    logger.info(f"🔍 Tentando extrair máquina_id de: '{request.message}'")
                    machine_id = _extract_machine_id(request.message, context)
                    logger.info(f"🔍 _extract_machine_id retornou: {machine_id}")
                    
                    if machine_id:
                        if "machine_available" not in command_dict:
                            command_dict["machine_available"] = {}
                        command_dict["machine_available"]["maquina_id"] = machine_id
                        logger.info(f"✅ Comando machine_available criado via fallback: máquina {machine_id} volta a estar disponível")
                    else:
                        # Se não conseguir extrair máquina_id, é erro real
                        logger.error(f"❌ Não foi possível extrair máquina_id de: '{request.message}'")
                        logger.error(f"❌ Máquinas disponíveis: {context.get('available_machines', [])}")
                        command_dict["command_type"] = "unknown"
                        command_dict["confidence"] = 0.3
                        command_dict["requires_clarification"] = True
                        command_dict["clarification_message"] = f"Não consegui identificar qual máquina na mensagem '{request.message}'. Tente especificar o número da máquina, por exemplo: 'máquina 300 disponível'. Máquinas disponíveis: {', '.join(context.get('available_machines', [])[:10])}"
                
                elif inferred_type == "change_priority":
                    logger.info(f"🔍 Tentando extrair informação de prioridade de: '{request.message}'")
                    priority_info = _extract_priority_info(request.message, context)
                    logger.info(f"🔍 _extract_priority_info retornou: {priority_info}")
                    
                    if priority_info:
                        if "priority_change" not in command_dict:
                            command_dict["priority_change"] = {}
                        # Usar artigo como order_id (o backend vai tratar isso)
                        command_dict["priority_change"]["order_id"] = priority_info["artigo"]
                        command_dict["priority_change"]["new_priority"] = priority_info["priority"]
                        logger.info(f"✅ Comando change_priority criado via fallback: {priority_info['artigo']} -> {priority_info['priority']}")
                    else:
                        # Se não conseguir extrair, é erro real
                        logger.error(f"❌ Não foi possível extrair informação de prioridade de: '{request.message}'")
                        logger.error(f"❌ Artigos disponíveis: {context.get('available_articles', [])[:10]}")
                        command_dict["command_type"] = "unknown"
                        command_dict["confidence"] = 0.3
                        command_dict["requires_clarification"] = True
                        command_dict["clarification_message"] = f"Não consegui identificar o artigo ou a prioridade na mensagem '{request.message}'. Tente especificar, por exemplo: 'GO4 VIP' ou 'GO Artigo 4 é VIP'. Artigos disponíveis: {', '.join(context.get('available_articles', [])[:10])}"
                
                elif inferred_type == "change_horizon":
                    logger.info(f"🔍 Tentando extrair horizonte de: '{request.message}'")
                    horizon_info = _extract_horizon_info(request.message)
                    logger.info(f"🔍 _extract_horizon_info retornou: {horizon_info}")
                    
                    if horizon_info:
                        if "horizon_change" not in command_dict:
                            command_dict["horizon_change"] = {}
                        command_dict["horizon_change"]["horizon_hours"] = horizon_info["horizon_hours"]
                        logger.info(f"✅ Comando change_horizon criado via fallback: {horizon_info['horizon_hours']} horas")
                    else:
                        # Se não conseguir extrair, é erro real
                        logger.error(f"❌ Não foi possível extrair horizonte de: '{request.message}'")
                        command_dict["command_type"] = "unknown"
                        command_dict["confidence"] = 0.3
                        command_dict["requires_clarification"] = True
                        command_dict["clarification_message"] = f"Não consegui identificar o número de horas na mensagem '{request.message}'. Tente especificar, por exemplo: 'planeia só 6 horas' ou 'mostra 4 horas'."
                
                elif inferred_type == "add_manual_order":
                    logger.info(f"🔍 Tentando extrair informação de ordem manual de: '{request.message}'")
                    order_info = _extract_manual_order_info(request.message, context)
                    logger.info(f"🔍 _extract_manual_order_info retornou: {order_info}")
                    
                    if order_info:
                        if "manual_order" not in command_dict:
                            command_dict["manual_order"] = {}
                        command_dict["manual_order"]["artigo"] = order_info["artigo"]
                        command_dict["manual_order"]["quantidade"] = order_info["quantidade"]
                        command_dict["manual_order"]["prioridade"] = order_info.get("prioridade", "NORMAL")
                        if order_info.get("due_date"):
                            command_dict["manual_order"]["due_date"] = order_info["due_date"]
                        logger.info(f"✅ Comando add_manual_order criado via fallback: {order_info['artigo']}, {order_info['quantidade']} unidades")
                    else:
                        # Se não conseguir extrair, é erro real
                        logger.error(f"❌ Não foi possível extrair informação de ordem manual de: '{request.message}'")
                        command_dict["command_type"] = "unknown"
                        command_dict["confidence"] = 0.3
                        command_dict["requires_clarification"] = True
                        command_dict["clarification_message"] = f"Não consegui identificar o artigo ou a quantidade na mensagem '{request.message}'. Tente especificar, por exemplo: 'adiciona GO3 com 50 unidades' ou 'nova ordem GO6 200 peças'. Artigos disponíveis: {', '.join(context.get('available_articles', [])[:10])}"
                
                elif inferred_type == "recalculate_plan":
                    # recalculate_plan não precisa de payload - apenas confirma que foi reconhecido
                    logger.info(f"✅ Comando recalculate_plan criado via fallback: '{request.message}'")
                    # Não precisa fazer nada mais - o comando já está completo
            else:
                # Não foi possível inferir - é realmente unknown
                logger.warning(f"⚠️ Não foi possível inferir comando de: '{request.message}'. É realmente unknown.")
                # Ajustar confidence e requires_clarification para valores sensatos
                command_dict["confidence"] = 0.2  # Muito baixa confiança para unknown real
                command_dict["requires_clarification"] = True
                command_dict["clarification_message"] = "Não consegui perceber a instrução. Reformula ou dá mais contexto. Exemplos: 'máquina 300 indisponível', 'planeia só 6 horas', 'GO4 VIP'."
        
        # Adicionar horizon_hours ao command_dict para usar em defaults
        command_dict["horizon_hours"] = context.get("current_horizon", 24)
        
        # Log para debug
        logger.info(f"📥 Comando após fallback: {command_dict.get('command_type')}, confidence={command_dict.get('confidence')}, requires_clarification={command_dict.get('requires_clarification')}")
        
        # CRÍTICO: Se ainda for "unknown" após todos os fallbacks, NÃO criar comando
        # unknown não é um comando válido - é sinal de falha de interpretação
        # NUNCA tentar deserializar ou aplicar comandos unknown
        if command_dict.get("command_type", "").lower() == "unknown":
            clarification_message = command_dict.get("clarification_message", "Não consegui perceber a instrução. Reformula ou dá mais contexto.")
            confidence = command_dict.get("confidence", 0.2)
            
            logger.warning(f"⚠️ Comando permanece 'unknown' após todos os fallbacks: '{request.message}'")
            logger.warning(f"⚠️ Retornando resposta de clarificação (NÃO é comando válido)")
            
            # Retornar resposta indicando que precisa de clarificação
            # unknown = não executar nada, apenas pedir clarificação
            return ChatResponse(
                command={
                    "command_type": "unknown",
                    "confidence": confidence,
                    "requires_clarification": True,  # SEMPRE true para unknown
                    "machine_unavailable": None,
                    "machine_available": None,
                    "manual_order": None,
                    "priority_change": None,
                    "horizon_change": None,
                },
                requires_clarification=True,
                clarification_message=clarification_message,
                confidence=confidence,
            )
        
        # Validar comando (só se não for unknown)
        try:
            command = PlanningCommand.from_dict(command_dict)
        except Exception as parse_error:
            logger.error(f"❌ Erro ao deserializar comando após fallback: {parse_error}")
            # Última tentativa: se detectar padrão de máquina indisponível, criar comando manualmente
            if "máquina" in request.message.lower() or "maquina" in request.message.lower():
                machine_id = _extract_machine_id(request.message, context)
                if machine_id:
                    now = datetime.utcnow()
                    horizon_hours = context.get("current_horizon", 24)
                    command_dict = {
                        "command_type": "machine_unavailable",
                        "machine_unavailable": {
                            "maquina_id": machine_id,
                            "start_time": now.isoformat(),
                            "end_time": (now + timedelta(hours=horizon_hours)).isoformat(),
                        },
                        "horizon_hours": horizon_hours,
                        "confidence": 0.8,
                        "requires_clarification": False,
                    }
                    command = PlanningCommand.from_dict(command_dict)
                    logger.info(f"✅ Criado comando de indisponibilidade manualmente para máquina {machine_id}")
                else:
                    # Não conseguiu extrair máquina - retornar unknown com clarificação
                    return ChatResponse(
                        command={
                            "command_type": "unknown",
                            "confidence": 0.3,
                            "requires_clarification": True,
                            "machine_unavailable": None,
                            "manual_order": None,
                            "priority_change": None,
                            "horizon_change": None,
                        },
                        requires_clarification=True,
                        clarification_message=f"Não consegui identificar qual máquina na mensagem: '{request.message}'. Tente: 'máquina 300 indisponível'.",
                        confidence=0.3,
                    )
            else:
                # Erro genérico de parse
                return ChatResponse(
                    command={
                        "command_type": "unknown",
                        "confidence": 0.2,
                        "requires_clarification": True,
                        "machine_unavailable": None,
                        "manual_order": None,
                        "priority_change": None,
                        "horizon_change": None,
                    },
                    requires_clarification=True,
                    clarification_message=f"Erro ao interpretar comando: {str(parse_error)}. Tente reformular a instrução.",
                    confidence=0.2,
                )
        
        # Validar que máquinas/artigos existem no contexto
        validation_errors = _validate_command(command, context)
        if validation_errors:
            command.requires_clarification = True
            command.clarification_message = "; ".join(validation_errors)
            command.confidence = 0.3
        
        return ChatResponse(
            command=command.to_dict(),
            requires_clarification=command.requires_clarification,
            clarification_message=command.clarification_message,
            confidence=command.confidence,
        )
        
    except HTTPException:
        # Re-raise HTTPExceptions (já têm status code e mensagem corretos)
        raise
    except LLMUnavailableError:
        logger.error("❌ /interpret: LLM não disponível")
        raise HTTPException(
            status_code=503,
            detail="Modelo LLM offline. Inicie Ollama (ollama serve) para usar o chat de planeamento."
        )
    except Exception as exc:
        error_type = type(exc).__name__
        error_msg = str(exc)
        logger.exception(f"❌ /interpret: Erro inesperado ao interpretar comando: {error_type}: {error_msg}")
        logger.error(f"❌ /interpret: Request recebido: message='{request.message}', batch_id={request.batch_id}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erro ao interpretar comando: {error_type}: {error_msg}. Verifique os logs do backend para mais detalhes."
        )


@router.post("/apply")
async def apply_planning_command(request: ApplyCommandRequest):
    """
    Aplica comando de planeamento e recalcula plano.
    
    Processo:
    1. Deserializa comando
    2. Atualiza PlanningConfig
    3. Invalida cache
    4. Retorna sucesso (frontend deve chamar /recalculate depois)
    
    Request esperado:
    {
        "command": {
            "command_type": "machine_unavailable",
            "machine_unavailable": {
                "maquina_id": "300",
                "start_time": "2025-11-19T10:00:00",
                "end_time": "2025-11-20T10:00:00"
            },
            "confidence": 0.8,
            "requires_clarification": false
        },
        "batch_id": "20251118225728" (opcional)
    }
    """
    # Log entrada detalhado
    logger.info(f"📥 /apply recebido: batch_id={request.batch_id}")
    logger.info(f"📥 /apply command recebido (tipo): {type(request.command)}")
    logger.info(f"📥 /apply command recebido (conteúdo): {json.dumps(request.command, indent=2, default=str)}")
    
    if not request.command:
        logger.error("❌ /apply: Comando vazio")
        raise HTTPException(
            status_code=400, 
            detail="Comando vazio. Formato esperado: { 'command': { 'command_type': '...', ... }, 'batch_id': '...' }"
        )
    
    # Obter batch_id
    if not request.batch_id:
        loader = get_loader()
        status = loader.get_status()
        batch_id = status.get("latest_batch_id") or status.get("batch_id") or "default"
    else:
        batch_id = request.batch_id
    
    try:
        # Obter horizon_hours atual para defaults
        from app.aps.cache import get_plan_cache
        cache = get_plan_cache()
        horizon_hours = 8  # Default
        for h in [24, 48, 72, 4, 8]:
            plan = cache.get(batch_id, h)
            if plan:
                horizon_hours = plan.horizon_hours
                break
        
        # Adicionar horizon_hours ao comando para usar em defaults
        command_dict = dict(request.command)
        command_dict["horizon_hours"] = horizon_hours
        
        logger.info(f"📋 /apply: Comando após adicionar horizon_hours: {json.dumps(command_dict, indent=2, default=str)}")
        
        # Deserializar comando com validação robusta
        try:
            command = PlanningCommand.from_dict(command_dict)
            logger.info(f"✅ /apply: Comando deserializado com sucesso: {command.command_type.value}")
        except Exception as parse_error:
            error_type = type(parse_error).__name__
            error_msg = str(parse_error)
            logger.error(f"❌ /apply: Erro ao deserializar comando: {error_type}: {error_msg}")
            logger.error(f"❌ /apply: Comando recebido (dict): {json.dumps(command_dict, indent=2, default=str)}")
            logger.error(f"❌ /apply: Traceback completo:", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail=f"Erro ao deserializar comando: {error_type}: {error_msg}. Comando recebido: {json.dumps(command_dict, default=str)}"
            )
        
        # Carregar configuração de planeamento
        config_store = get_planning_config_store()
        planning_config = config_store.get(batch_id)
        
        # Log para debug
        logger.info(f"📥 Aplicar comando: type={command.command_type.value}, "
                   f"has_machine_unavailable={command.machine_unavailable is not None}, "
                   f"has_manual_order={command.manual_order is not None}, "
                   f"has_priority_change={command.priority_change is not None}, "
                   f"has_horizon_change={command.horizon_change is not None}")
        
        # Aplicar comando à configuração com validação adicional
        if command.command_type.value == "machine_unavailable":
            if not command.machine_unavailable:
                raise HTTPException(
                    status_code=400,
                    detail="Comando 'machine_unavailable' está incompleto. Falta informação sobre a máquina e horários."
                )
            
            unavail = command.machine_unavailable
            
            # Validação final: garantir que as datas são válidas
            if not isinstance(unavail.start_time, datetime) or not isinstance(unavail.end_time, datetime):
                # Se ainda assim não forem datetime válidos, aplicar defaults
                now = datetime.utcnow()
                if not isinstance(unavail.start_time, datetime):
                    unavail.start_time = now
                    logger.warning(f"start_time inválido, usando default: {unavail.start_time}")
                if not isinstance(unavail.end_time, datetime):
                    unavail.end_time = now + timedelta(hours=horizon_hours)
                    logger.warning(f"end_time inválido, usando default: {unavail.end_time}")
            
            # Validar que end_time > start_time
            if unavail.end_time <= unavail.start_time:
                logger.warning(f"end_time <= start_time. Ajustando end_time para start_time + {horizon_hours}h")
                unavail.end_time = unavail.start_time + timedelta(hours=horizon_hours)
            
            planning_config.add_unavailability(unavail)
            
        elif command.command_type.value == "machine_available":
            if not command.machine_available:
                raise HTTPException(
                    status_code=400,
                    detail="Comando 'machine_available' está incompleto. Falta informação sobre a máquina."
                )
            
            planning_config.remove_unavailability(command.machine_available.maquina_id)
            
        elif command.command_type.value == "add_manual_order":
            if not command.manual_order:
                raise HTTPException(
                    status_code=400,
                    detail="Comando 'add_manual_order' está incompleto. Falta informação sobre a ordem."
                )
            planning_config.add_manual_order(command.manual_order)
            
        elif command.command_type.value == "change_priority":
            if not command.priority_change:
                raise HTTPException(
                    status_code=400,
                    detail="Comando 'change_priority' está incompleto. Falta informação sobre a ordem e prioridade."
                )
            
            # Validar que o artigo existe
            tech_queries = get_technical_queries()
            order_id = command.priority_change.order_id
            if not tech_queries.validate_article(order_id):
                # Tentar normalizar (GO4 -> GO Artigo 4)
                normalized = order_id.replace("GO", "GO Artigo ").replace("Artigo Artigo", "Artigo")
                if tech_queries.validate_article(normalized):
                    order_id = normalized
                else:
                    available_articles = tech_queries.get_all_articles()
                    raise HTTPException(
                        status_code=400,
                        detail=f"Artigo '{command.priority_change.order_id}' não existe no modelo. Artigos disponíveis: {', '.join(available_articles[:10])}"
                    )
            
            # Validar que a prioridade é válida
            valid_priorities = ["VIP", "ALTA", "NORMAL", "BAIXA"]
            if command.priority_change.new_priority not in valid_priorities:
                raise HTTPException(
                    status_code=400,
                    detail=f"Prioridade inválida: '{command.priority_change.new_priority}'. Válidas: {', '.join(valid_priorities)}"
                )
            
            # Se prioridade for NORMAL e já existe override, remover (voltar ao normal)
            if command.priority_change.new_priority == "NORMAL" and order_id in planning_config.priority_overrides:
                del planning_config.priority_overrides[order_id]
                logger.info(f"Prioridade removida para {order_id} (volta ao normal)")
            else:
                planning_config.set_priority(order_id, command.priority_change.new_priority)
            
        elif command.command_type.value == "change_horizon":
            if not command.horizon_change:
                raise HTTPException(
                    status_code=400,
                    detail="Comando 'change_horizon' está incompleto. Falta informação sobre o horizonte."
                )
            planning_config.set_horizon(command.horizon_change.horizon_hours)
            
        elif command.command_type.value == "recalculate_plan":
            # recalculate_plan: apenas recalcular com configuração atual
            # CRÍTICO: NÃO altera PlanningConfig - mantém indisponibilidades, VIPs, horizonte, etc.
            # CRÍTICO: NÃO altera APSConfig.routing_preferences - não mexe em preferências de rota
            # CRÍTICO: Mas garante que não há preferências globais "presas" no APSConfig
            logger.info(f"✅ Comando recalculate_plan recebido - vai recalcular plano com configuração atual")
            logger.info(f"✅ NÃO alterando PlanningConfig nem routing_preferences - apenas invalidando cache")
            
            # VALIDAÇÃO ADICIONAL: Verificar se há preferências globais no APSConfig que possam estar "presas"
            # (Isto não deveria acontecer, mas é uma camada extra de segurança)
            # O APSConfig será criado limpo no /recalculate, mas vamos garantir aqui também
            logger.info(f"✅ recalculate_plan: APSConfig será criado limpo no /recalculate (sem preferências de rota)")
            
            # Não precisa fazer nada na PlanningConfig - apenas invalidar cache
            # O recálculo vai usar a configuração atual (sem alterações)
            # NÃO guardar PlanningConfig (não alterou nada)
            # Passar diretamente para invalidar cache sem guardar
            
        elif command.command_type.value == "unknown":
            # CRÍTICO: unknown NÃO é um comando válido - é sinal de falha de interpretação
            # NUNCA tentar executar comandos unknown
            logger.warning(f"⚠️ /apply: Comando 'unknown' recebido - REJEITADO (não é comando válido)")
            logger.warning(f"⚠️ /apply: Comando completo: {json.dumps(command.to_dict(), indent=2, default=str)}")
            
            # Retornar mensagem clara ao frontend
            raise HTTPException(
                status_code=400,
                detail="Não consegui perceber a instrução. Por favor, reformule ou dê mais detalhes. Exemplos: 'máquina 300 indisponível', 'planeia só 6 horas', 'GO4 VIP'."
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de comando inválido: '{command.command_type.value}'. Tipos válidos: machine_unavailable, machine_available, add_manual_order, change_priority, change_horizon, recalculate_plan."
            )
        
        # Guardar configuração (apenas se não for recalculate_plan - que não altera nada)
        if command.command_type.value != "recalculate_plan":
            config_store.save(planning_config)
        else:
            logger.info(f"✅ recalculate_plan: NÃO guardando PlanningConfig (não alterou nada)")
        
        # Invalidar cache de planos (força recálculo)
        from app.aps.cache import get_plan_cache
        cache = get_plan_cache()
        cache.invalidate(batch_id)
        
        logger.info(f"Comando aplicado para batch_id={batch_id}: {command.command_type.value}")
        
        return {
            "ok": True,
            "batch_id": batch_id,
            "command_type": command.command_type.value,
            "message": "Comando aplicado com sucesso. O plano será recalculado.",
        }
        
    except HTTPException:
        # Re-raise HTTPExceptions (erros de validação)
        raise
    except Exception as exc:
        error_type = type(exc).__name__
        error_msg = str(exc)
        logger.exception(f"❌ /apply: Erro inesperado ao aplicar comando: {error_type}: {error_msg}")
        logger.error(f"❌ /apply: Request recebido: {json.dumps({'command': request.command, 'batch_id': request.batch_id}, default=str)}")
        
        # Retornar mensagem mais amigável para erros específicos
        if "fromisoformat" in error_msg.lower():
            error_msg = "Erro ao processar datas. Tente especificar horários explícitos, por exemplo: 'máquina 300 indisponível hoje das 14h às 18h'."
        
        raise HTTPException(
            status_code=500, 
            detail=f"Erro ao aplicar comando: {error_type}: {error_msg}. Verifique os logs do backend para mais detalhes."
        )


def _build_planning_context(batch_id: str) -> dict:
    """
    Constrói contexto MÍNIMO para o LLM (apenas labels, não planos completos).
    
    Contexto reduzido:
    - Apenas lista de máquinas disponíveis
    - Apenas lista de artigos disponíveis
    - Horizonte atual
    - NÃO inclui planos, operações agendadas, alternativas completas, etc.
    """
    # Usar TechnicalQueries para obter dados do modelo (Excel parseado)
    tech_queries = get_technical_queries()
    
    # Obter apenas labels (mínimo necessário)
    available_machines = tech_queries.get_all_machines()
    available_articles = tech_queries.get_all_articles()
    
    # Tentar obter horizon_hours da configuração de planeamento
    current_horizon = 24  # Default
    try:
        from app.aps.planning_config import get_planning_config_store
        config_store = get_planning_config_store()
        planning_config = config_store.get(batch_id)
        if planning_config.horizon_hours:
            current_horizon = planning_config.horizon_hours
    except Exception as exc:
        logger.debug(f"Erro ao obter horizon_hours da config: {exc}")
    
    # Tentar obter do plano se não houver na config
    if current_horizon == 24:
        try:
            from app.aps.cache import get_plan_cache
            cache = get_plan_cache()
            for horizon in [24, 48, 72, 4, 8]:
                plan = cache.get(batch_id, horizon)
                if plan:
                    current_horizon = plan.horizon_hours
                    break
        except Exception as exc:
            logger.debug(f"Erro ao obter horizon_hours do plano: {exc}")
    
    return {
        "available_machines": available_machines,  # Apenas labels
        "available_articles": available_articles,  # Apenas labels
        "current_horizon": current_horizon,
        "horizon_hours": current_horizon,  # Para usar no from_dict
    }


def _validate_command(command: PlanningCommand, context: dict) -> list:
    """
    Valida comando contra modelo real (TechnicalQueries).
    
    Validação rigorosa: rejeita máquinas/rotas/operações que não existem no Excel.
    """
    errors = []
    tech_queries = get_technical_queries()
    
    if command.machine_unavailable:
        maquina_id = command.machine_unavailable.maquina_id
        if not tech_queries.validate_machine(maquina_id):
            available_machines = tech_queries.get_all_machines()
            errors.append(
                f"Máquina '{maquina_id}' não existe no modelo. "
                f"Máquinas disponíveis: {', '.join(available_machines[:20])}"
                f"{'...' if len(available_machines) > 20 else ''}"
            )
    
    if command.manual_order:
        artigo = command.manual_order.artigo
        if not tech_queries.validate_article(artigo):
            # Tentar normalizar (GO6 -> GO Artigo 6)
            normalized = artigo.replace("GO", "GO Artigo ").replace("Artigo Artigo", "Artigo")
            if not tech_queries.validate_article(normalized):
                available_articles = tech_queries.get_all_articles()
                errors.append(
                    f"Artigo '{artigo}' não existe no modelo. "
                    f"Artigos disponíveis: {', '.join(available_articles[:20])}"
                    f"{'...' if len(available_articles) > 20 else ''}"
                )
    
    if command.priority_change:
        # Validar artigo
        order_id = command.priority_change.order_id
        if not tech_queries.validate_article(order_id):
            # Tentar normalizar (GO4 -> GO Artigo 4)
            normalized = order_id.replace("GO", "GO Artigo ").replace("Artigo Artigo", "Artigo")
            if not tech_queries.validate_article(normalized):
                available_articles = tech_queries.get_all_articles()
                errors.append(
                    f"Artigo '{order_id}' não existe no modelo. "
                    f"Artigos disponíveis: {', '.join(available_articles[:20])}"
                    f"{'...' if len(available_articles) > 20 else ''}"
                )
        
        # Validar prioridade
        valid_priorities = ["VIP", "ALTA", "NORMAL", "BAIXA"]
        if command.priority_change.new_priority not in valid_priorities:
            errors.append(
                f"Prioridade inválida: '{command.priority_change.new_priority}'. "
                f"Válidas: {', '.join(valid_priorities)}"
            )
    
    if command.horizon_change:
        if command.horizon_change.horizon_hours <= 0 or command.horizon_change.horizon_hours > 168:
            errors.append(
                f"Horizonte inválido: {command.horizon_change.horizon_hours}h. "
                f"Deve estar entre 1h e 168h (1 semana)"
            )
    
    # TODO: Validar force_route quando implementado
    # if command.force_route:
    #     artigo = command.force_route.artigo
    #     rota = command.force_route.rota
    #     if not tech_queries.validate_route(artigo, rota):
    #         available_routes = tech_queries.get_routes(artigo)
    #         errors.append(f"Rota '{rota}' não existe para artigo '{artigo}'. Rotas disponíveis: {', '.join(available_routes)}")
    
    return errors


def _infer_command_type(message: str, context: dict) -> Optional[str]:
    """
    Tenta inferir o tipo de comando a partir da mensagem quando o LLM retorna "unknown".
    
    REGRA DE OURO: Frases como "máquina 300 indisponível" são o "Hello World" do sistema.
    Se isso não funcionar, o planeador perde confiança no produto.
    
    Esta função é SIMPLES e ROBUSTA - não precisa de NLP avançado, apenas padrões básicos.
    """
    import re
    
    # Normalizar mensagem: minúsculas, remover acentos (simplificado)
    message_lower = message.lower()
    # Substituir acentos comuns (simplificado - não é perfeito mas funciona para casos básicos)
    message_normalized = message_lower.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    
    # CRÍTICO: Palavras-chave para INDISPONIBILIDADE (verificar PRIMEIRO)
    # Estas palavras indicam que a máquina deve ser marcada como indisponível
    unavailability_keywords = [
        "indisponivel", "indisponível", "avariada", "avariado", "parada", "parado", 
        "quebrada", "quebrado", "manutencao", "manutenção", "falta", "unavailable", 
        "down", "fora de servico", "fora de serviço", "fora servico", "fora serviço",
        "offline", "parou", "parou de funcionar", "nao funciona", "não funciona", 
        "nao esta", "não está", "nao esta a funcionar", "não está a funcionar",
        "em baixo", "em manutencao", "em manutenção", "parar", "parar a",
        "bloquear", "retirar do plano", "retirar plano", "remover do plano"
    ]
    
    # Palavras-chave para DISPONIBILIDADE (máquina volta a estar disponível)
    # IMPORTANTE: "disponível" sozinho NÃO é suficiente - precisa de contexto de "volta", "já está", etc.
    # Ou então deve estar explícito "remover indisponibilidade"
    availability_keywords_strong = [
        "volta a estar disponivel", "volta a estar disponível", "voltou a estar disponivel", "voltou a estar disponível",
        "volta a funcionar", "voltou a funcionar", "volta a operar", "voltou a operar",
        "ja esta disponivel", "já está disponível", "ja esta operacional", "já está operacional",
        "remover indisponibilidade", "remover indisponibilidade da", "remover indisponibilidade de",
        "repor no plano", "repor plano", "voltar a por no plano", "voltar a pôr no plano",
        "ligar", "reativar", "reparada", "reparado", "pronta", "pronto"
    ]
    
    # Palavras-chave fracas para disponibilidade (só contam se acompanhadas de contexto)
    availability_keywords_weak = [
        "disponivel", "disponível", "available", "funciona", "funcionando", "operacional",
        "ativa", "ativo", "online", "em funcionamento"
    ]
    
    # Palavras-chave para máquina (muito abrangente)
    machine_keywords = ["maquina", "máquina", "machine", "maq", "recurso", "equipamento"]
    
    # Verificar se tem número (sequência de dígitos)
    has_number = bool(re.search(r'\d+', message))
    
    # Verificar palavras-chave (tanto na versão original quanto normalizada)
    has_machine_word = any(kw in message_lower for kw in machine_keywords) or any(kw in message_normalized for kw in machine_keywords)
    has_unavailability_word = any(kw in message_lower for kw in unavailability_keywords) or any(kw in message_normalized for kw in unavailability_keywords)
    has_availability_strong = any(kw in message_lower for kw in availability_keywords_strong) or any(kw in message_normalized for kw in availability_keywords_strong)
    has_availability_weak = any(kw in message_lower for kw in availability_keywords_weak) or any(kw in message_normalized for kw in availability_keywords_weak)
    
    # REGRA CRÍTICA #1: Se contém "indisponível" ou palavras de indisponibilidade → SEMPRE machine_unavailable
    # Esta regra tem PRIORIDADE ABSOLUTA sobre disponibilidade
    if has_unavailability_word:
        if has_number:
            if has_machine_word:
                logger.info(f"✅ _infer_command_type: Reconhecido machine_unavailable (máquina + indisponível + número) - PRIORIDADE")
                return "machine_unavailable"
            else:
                logger.info(f"✅ _infer_command_type: Reconhecido machine_unavailable (número + indisponível) - PRIORIDADE")
                return "machine_unavailable"
    
    # REGRA #2: Se contém palavras FORTES de disponibilidade (volta, já está, remover indisponibilidade) → machine_available
    # Só depois de verificar indisponibilidade
    if has_availability_strong:
        if has_number:
            if has_machine_word:
                logger.info(f"✅ _infer_command_type: Reconhecido machine_available (máquina + volta/já está disponível + número)")
                return "machine_available"
            else:
                logger.info(f"✅ _infer_command_type: Reconhecido machine_available (número + volta/já está disponível)")
                return "machine_available"
    
    # REGRA #3: Se contém palavras FRACAS de disponibilidade (apenas "disponível", "funciona") → machine_available
    # Mas só se NÃO contiver palavras de indisponibilidade (já verificado acima)
    if has_availability_weak and not has_unavailability_word:
        if has_number:
            if has_machine_word:
                logger.info(f"✅ _infer_command_type: Reconhecido machine_available (máquina + disponível + número)")
                return "machine_available"
            else:
                logger.info(f"✅ _infer_command_type: Reconhecido machine_available (número + disponível)")
                return "machine_available"
    
    # Verificar padrões de PRIORIDADE (antes de outros comandos menos específicos)
    # Padrões: "GO4 VIP", "GO Artigo 4 é VIP", "prioridade VIP para GO3", etc.
    priority_keywords = ["vip", "alta", "normal", "baixa", "urgente", "prioridade", "priority"]
    has_priority_keyword = any(kw in message_lower for kw in priority_keywords)
    
    # Padrões para identificar artigos GO
    go_patterns = [
        r'\bgo\s*artigo\s*(\d+)\b',  # "GO Artigo 4"
        r'\bgo\s*(\d+)\b',  # "GO4", "GO 4"
        r'\bgo\s*artigo\s*(\d+)\b',  # "GO Artigo 4" (case insensitive já está no message_lower)
    ]
    
    has_go_article = False
    for pattern in go_patterns:
        if re.search(pattern, message_lower):
            has_go_article = True
            break
    
    # Se tem GO + prioridade → change_priority
    if has_go_article and has_priority_keyword:
        logger.info(f"✅ _infer_command_type: Reconhecido change_priority (GO + prioridade)")
        return "change_priority"
    
    # Se tem palavra "prioridade" + GO → change_priority
    if has_priority_keyword and ("prioridade" in message_lower or "priority" in message_lower):
        if has_go_article or re.search(r'\bgo\s*\d+', message_lower):
            logger.info(f"✅ _infer_command_type: Reconhecido change_priority (prioridade + GO)")
            return "change_priority"
    
    # PRIMEIRO: Verificar change_horizon (ANTES de recalculate_plan)
    # Padrões: "planeia só 6 horas", "planeia 4 horas", "planear para 8h", etc.
    # IMPORTANTE: Esta verificação deve acontecer ANTES de recalculate_plan
    # porque "planeia só 6 horas" deve ser change_horizon, não recalculate_plan
    import re
    has_number = bool(re.search(r'\d+', message_lower))
    horizon_keywords_in_message = ["horizonte", "horas", "horizon", "hours", "hora"]
    has_horizon_keyword = any(kw in message_lower for kw in horizon_keywords_in_message)
    
    # Se tem "planear" + número + palavras de horas → change_horizon (PRIORIDADE ALTA)
    if ("planear" in message_lower or "planeia" in message_lower or "planeie" in message_lower) and has_number:
        # Verificar se tem palavras de horas ou se o número está próximo de "horas"
        if has_horizon_keyword or re.search(r'\d+\s*(?:horas?|h)', message_lower) or re.search(r'(?:horas?|h).*\d+|\d+.*(?:horas?|h)', message_lower):
            logger.info(f"✅ _infer_command_type: Reconhecido change_horizon (planear + número + horas) - PRIORIDADE")
            return "change_horizon"
        # Se tem "planear" + número (mesmo sem "horas" explícita), assumir que é change_horizon
        logger.info(f"✅ _infer_command_type: Reconhecido change_horizon (planear + número, assumindo horas)")
        return "change_horizon"
    
    # Se tem palavras de horizonte + número → change_horizon
    if has_horizon_keyword and has_number:
        logger.info(f"✅ _infer_command_type: Reconhecido change_horizon (horizonte/horas + número)")
        return "change_horizon"
    
    # Verificar padrões de RECALCULAR PLANO (DEPOIS de change_horizon)
    # Padrões: "otimiza o plano", "recalcula o plano", "replanear", "atualiza o planeamento", etc.
    recalculate_keywords = [
        "otimiza", "otimizar", "recalcula", "recalcular", "replanear", "replaneia",
        "atualiza", "atualizar", "gera", "gerar", "faz", "fazer", "refaz", "refazer",
        "volta a calcular", "volta a planear", "reescreve", "reescrever",
        "novo plano", "plano novo", "replaneamento", "replaneia", "replaneie"
    ]
    plan_keywords = ["plano", "planeamento", "plan", "planning"]
    
    has_recalculate_keyword = any(kw in message_lower for kw in recalculate_keywords)
    has_plan_keyword = any(kw in message_lower for kw in plan_keywords)
    
    # Se tem palavra de recalcular + palavra de plano → recalculate_plan
    # Mas só se NÃO tiver palavras que indiquem horizonte (ex: "planeia só 6 horas" já foi capturado acima)
    if has_recalculate_keyword and has_plan_keyword and not has_horizon_keyword:
        logger.info(f"✅ _infer_command_type: Reconhecido recalculate_plan (recalcular + plano, sem horizonte)")
        return "recalculate_plan"
    
    # Caso especial: apenas "replanear" ou "otimiza" (sem palavra "plano" explícita)
    # Mas só se não tiver outras palavras que indiquem outro comando
    # E se não tiver número (que pode indicar horas)
    if has_recalculate_keyword and not has_machine_word and not has_go_article and not has_priority_keyword and not has_horizon_keyword and not has_number:
        logger.info(f"✅ _infer_command_type: Reconhecido recalculate_plan (apenas palavra de recalcular, sem contexto específico)")
        return "recalculate_plan"
    
    # Caso especial: "otimiza" ou "recalcula" sozinho (sem "plano" mas contexto claro)
    # Se a mensagem é muito curta e só tem palavra de recalcular, é recalculate_plan
    if has_recalculate_keyword and len(message_lower.split()) <= 3 and not has_machine_word and not has_go_article and not has_priority_keyword and not has_horizon_keyword:
        logger.info(f"✅ _infer_command_type: Reconhecido recalculate_plan (mensagem curta com palavra de recalcular)")
        return "recalculate_plan"
    
    order_keywords = ["ordem", "order", "adicionar", "adiciona", "nova ordem", "pedido"]
    if any(kw in message_lower for kw in order_keywords):
        return "add_manual_order"
    
    logger.warning(f"⚠️ _infer_command_type: Não conseguiu inferir tipo de comando de: '{message}'")
    return None


def _extract_priority_info(message: str, context: dict) -> Optional[Dict[str, str]]:
    """
    Extrai informação de prioridade da mensagem.
    
    Retorna: {"artigo": "GO Artigo 4", "priority": "VIP"} ou None
    """
    import re
    
    message_lower = message.lower()
    available_articles = context.get("available_articles", [])
    
    # Padrões para extrair artigo GO
    go_patterns = [
        (r'\bgo\s*artigo\s*(\d+)\b', lambda m: f"GO Artigo {m.group(1)}"),  # "GO Artigo 4"
        (r'\bgo\s*(\d+)\b', lambda m: f"GO Artigo {m.group(1)}"),  # "GO4", "GO 4"
    ]
    
    artigo = None
    for pattern, formatter in go_patterns:
        match = re.search(pattern, message_lower)
        if match:
            artigo_candidate = formatter(match)
            # Validar que o artigo existe
            if artigo_candidate in available_articles:
                artigo = artigo_candidate
                logger.info(f"✅ _extract_priority_info: Artigo encontrado: {artigo}")
                break
            # Tentar variações
            # Se "GO Artigo 4" não existe, tentar "GO4"
            if "GO Artigo" in artigo_candidate:
                go_num = artigo_candidate.replace("GO Artigo ", "")
                go_variant = f"GO{go_num}"
                if go_variant in available_articles:
                    artigo = go_variant
                    logger.info(f"✅ _extract_priority_info: Artigo encontrado (variação): {artigo}")
                    break
    
    if not artigo:
        logger.warning(f"⚠️ _extract_priority_info: Nenhum artigo GO válido encontrado em: '{message}'")
        logger.warning(f"⚠️ Artigos disponíveis: {available_articles[:10]}")
        return None
    
    # Extrair prioridade
    priority_keywords = {
        "vip": "VIP",
        "alta": "ALTA",
        "normal": "NORMAL",
        "baixa": "BAIXA",
        "urgente": "VIP",  # Urgente = VIP
    }
    
    # Palavras que indicam remoção de prioridade (voltar ao normal)
    remove_keywords = ["remove", "remover", "deixa de ser", "volta ao normal", "sem prioridade", "tira", "retira"]
    has_remove_keyword = any(kw in message_lower for kw in remove_keywords)
    
    priority = None
    
    # Se tem palavra de remoção explícita, prioridade é NORMAL
    if has_remove_keyword:
        priority = "NORMAL"
        logger.info(f"✅ _extract_priority_info: Prioridade removida (volta ao normal)")
    else:
        # Procurar por palavras-chave de prioridade
        for keyword, priority_value in priority_keywords.items():
            if keyword in message_lower:
                priority = priority_value
                logger.info(f"✅ _extract_priority_info: Prioridade encontrada: {priority}")
                break
    
    # Se ainda não encontrou, mas tem "é VIP", "é alta", etc.
    if not priority:
        priority_patterns = [
            (r'\bé\s+(vip|alta|normal|baixa)\b', lambda m: m.group(1).upper()),
            (r'\b(vip|alta|normal|baixa)\s+para\b', lambda m: m.group(1).upper()),
            (r'\bprioridade\s+(vip|alta|normal|baixa)\b', lambda m: m.group(1).upper()),
        ]
        for pattern, formatter in priority_patterns:
            match = re.search(pattern, message_lower)
            if match:
                priority_candidate = formatter(match)
                if priority_candidate in ["VIP", "ALTA", "NORMAL", "BAIXA"]:
                    priority = priority_candidate
                    logger.info(f"✅ _extract_priority_info: Prioridade encontrada via padrão: {priority}")
                    break
    
    if not priority:
        logger.warning(f"⚠️ _extract_priority_info: Nenhuma prioridade encontrada em: '{message}'")
        return None
    
    return {"artigo": artigo, "priority": priority}


def _extract_machine_id(message: str, context: dict) -> Optional[str]:
    """
    Extrai ID da máquina da mensagem.
    
    REGRA SIMPLES: Procura o primeiro número (sequência de dígitos) na mensagem.
    Valida contra a lista de máquinas disponíveis.
    
    Não precisa de regex complexa - apenas extrair números e validar.
    """
    import re
    
    available_machines = context.get("available_machines", [])
    
    if not available_machines:
        logger.warning(f"⚠️ _extract_machine_id: Lista de máquinas vazia no contexto")
        return None
    
    # Extrair TODOS os números da mensagem (sequências de 2-4 dígitos)
    all_numbers = re.findall(r'\b(\d{2,4})\b', message)
    
    if not all_numbers:
        logger.warning(f"⚠️ _extract_machine_id: Nenhum número encontrado em: '{message}'")
        return None
    
    logger.info(f"🔍 _extract_machine_id: Números encontrados: {all_numbers}, Máquinas disponíveis: {available_machines[:10]}...")
    
    # Tentar cada número encontrado (do primeiro ao último)
    for num in all_numbers:
        # Tentar exatamente como está
        if num in available_machines:
            logger.info(f"✅ _extract_machine_id: Máquina {num} encontrada e validada")
            return num
        
        # Tentar com zero à esquerda (ex: "300" -> "0300")
        if len(num) == 3:
            padded = "0" + num
            if padded in available_machines:
                logger.info(f"✅ _extract_machine_id: Máquina {padded} encontrada (com zero à esquerda)")
                return padded
        
        # Tentar sem zeros à esquerda (ex: "0300" -> "300")
        if len(num) == 4 and num.startswith("0"):
            unpadded = num.lstrip("0")
            if unpadded in available_machines:
                logger.info(f"✅ _extract_machine_id: Máquina {unpadded} encontrada (sem zeros à esquerda)")
                return unpadded
    
    # Nenhum número válido encontrado
    logger.warning(f"⚠️ _extract_machine_id: Nenhum dos números encontrados ({all_numbers}) corresponde a uma máquina válida")
    logger.warning(f"⚠️ _extract_machine_id: Máquinas disponíveis: {available_machines}")
    return None


def _extract_horizon_info(message: str) -> Optional[Dict[str, int]]:
    """
    Extrai número de horas do horizonte da mensagem.
    
    Retorna: {"horizon_hours": 6} ou None
    """
    import re
    
    message_lower = message.lower()
    
    # Padrões para extrair horas (ordem importa - mais específicos primeiro)
    # Exemplos: "6 horas", "4h", "planeia só 8 horas", "mostra 12h"
    patterns = [
        r'\bplaneia\s+(?:só|apenas|para)\s+(\d+)\s*(?:horas?|h)\b',  # "planeia só 6 horas" (mais específico primeiro)
        r'\bmostra\s+(?:só|apenas)\s+(\d+)\s*(?:horas?|h)\b',  # "mostra só 4 horas"
        r'\bplanear\s+(?:só|apenas|para)\s+(\d+)\s*(?:horas?|h)\b',  # "planear só 6 horas"
        r'\b(\d+)\s*horas?\s+de\s+horizonte\b',  # "6 horas de horizonte"
        r'\b(\d+)\s*horas?\b',  # "6 horas", "4 hora" (padrão genérico)
        r'\b(\d+)\s*h\b',  # "6h", "4h" (sem espaço)
        r'\b(\d+)\s*hora\b',  # "6 hora" (singular)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message_lower)
        if match:
            hours = int(match.group(1))
            # Validar que é um valor razoável (1-168 horas = 1 semana)
            if 1 <= hours <= 168:
                logger.info(f"✅ _extract_horizon_info: Horizonte encontrado: {hours} horas")
                return {"horizon_hours": hours}
            else:
                logger.warning(f"⚠️ _extract_horizon_info: Valor de horas fora do intervalo válido: {hours}")
    
    logger.warning(f"⚠️ _extract_horizon_info: Nenhum horizonte encontrado em: '{message}'")
    return None


def _extract_manual_order_info(message: str, context: dict) -> Optional[Dict[str, any]]:
    """
    Extrai informação de ordem manual da mensagem.
    
    Retorna: {"artigo": "GO Artigo 4", "quantidade": 200, "prioridade": "VIP", "due_date": ...} ou None
    """
    import re
    
    message_lower = message.lower()
    available_articles = context.get("available_articles", [])
    
    # Extrair artigo GO (mesma lógica de _extract_priority_info)
    go_patterns = [
        (r'\bgo\s*artigo\s*(\d+)\b', lambda m: f"GO Artigo {m.group(1)}"),  # "GO Artigo 4"
        (r'\bgo\s*(\d+)\b', lambda m: f"GO Artigo {m.group(1)}"),  # "GO4", "GO 4"
    ]
    
    artigo = None
    for pattern, formatter in go_patterns:
        match = re.search(pattern, message_lower)
        if match:
            artigo_candidate = formatter(match)
            # Validar que o artigo existe
            if artigo_candidate in available_articles:
                artigo = artigo_candidate
                break
            # Tentar variações
            if "GO Artigo" in artigo_candidate:
                go_num = artigo_candidate.replace("GO Artigo ", "")
                go_variant = f"GO{go_num}"
                if go_variant in available_articles:
                    artigo = go_variant
                    break
    
    if not artigo:
        logger.warning(f"⚠️ _extract_manual_order_info: Nenhum artigo GO válido encontrado em: '{message}'")
        return None
    
    # Extrair quantidade
    # Padrões: "50 unidades", "200 peças", "100 pcs", "50", etc.
    quantity_patterns = [
        r'\b(\d+)\s*(?:unidades?|peças?|pcs?|p)\b',  # "50 unidades", "200 peças"
        r'\bcom\s+(\d+)\s*(?:unidades?|peças?|pcs?)?\b',  # "com 50 unidades"
        r'\b(\d+)\s*(?:unidades?|peças?|pcs?)?\s+de\b',  # "50 unidades de"
        r'\bgo\s*\d+\s+(\d+)\b',  # "GO4 200" (artigo seguido de número)
    ]
    
    quantidade = None
    for pattern in quantity_patterns:
        match = re.search(pattern, message_lower)
        if match:
            quantidade = int(match.group(1))
            if quantidade > 0:
                logger.info(f"✅ _extract_manual_order_info: Quantidade encontrada: {quantidade}")
                break
    
    # Se não encontrou quantidade explícita, tentar número isolado após o artigo
    if not quantidade:
        # Procurar número após "GO Artigo X" ou "GOX"
        after_article_match = re.search(r'\bgo\s*(?:artigo\s*)?\d+\s+(\d+)\b', message_lower)
        if after_article_match:
            quantidade = int(after_article_match.group(1))
            if quantidade > 0:
                logger.info(f"✅ _extract_manual_order_info: Quantidade encontrada (após artigo): {quantidade}")
    
    if not quantidade or quantidade <= 0:
        logger.warning(f"⚠️ _extract_manual_order_info: Nenhuma quantidade válida encontrada em: '{message}'")
        return None
    
    # Extrair prioridade (opcional, default: NORMAL)
    priority_keywords = {
        "vip": "VIP",
        "alta": "ALTA",
        "normal": "NORMAL",
        "baixa": "BAIXA",
        "urgente": "VIP",
    }
    
    prioridade = "NORMAL"  # Default
    for keyword, priority_value in priority_keywords.items():
        if keyword in message_lower:
            prioridade = priority_value
            logger.info(f"✅ _extract_manual_order_info: Prioridade encontrada: {prioridade}")
            break
    
    result = {
        "artigo": artigo,
        "quantidade": quantidade,
        "prioridade": prioridade,
    }
    
    # Extrair due_date se mencionado (opcional)
    # Padrões: "para amanhã", "deadline 15/01", etc.
    # Por agora, não vamos extrair due_date automaticamente (muito complexo)
    # O utilizador pode adicionar depois via outro comando
    
    return result

