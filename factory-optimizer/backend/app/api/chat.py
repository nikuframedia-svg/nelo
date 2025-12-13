import json
import re
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.insights.engine import InsightEngine
from app.insights.prompts import SYSTEM_PROMPT, get_prompt_by_mode
from app.llm import LLMUnavailableError, LocalLLM
from app.llm.validator import validate_llm_output
from app.etl.loader import get_loader

router = APIRouter()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    mode: Optional[Literal["planeamento", "gargalos", "inventario", "resumo"]] = "resumo"
    temperature: Optional[float] = 0.5  # Aumentado para 0.5 por padrão para chat ser mais variado


class ChatResponse(BaseModel):
    answer: str
    model: str
    used_context: str
    mode: str


@router.post("/")
async def chat(request: ChatRequest):
    """Chat inteligente usando Insight Engine."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="Nenhuma mensagem enviada.")

    try:
        engine = InsightEngine()
        
        # Extrair última mensagem do utilizador (a pergunta atual)
        last_user_message = next(
            (msg.content for msg in reversed(request.messages) if msg.role == "user"), ""
        )
        
        if not last_user_message:
            raise HTTPException(status_code=400, detail="Nenhuma mensagem do utilizador encontrada.")

        # Detectar modo automaticamente baseado na pergunta (se não foi especificado ou é "resumo")
        detected_mode = request.mode
        if request.mode == "resumo" or not request.mode:
            question_lower = last_user_message.lower()
            if any(word in question_lower for word in ["gargalo", "recurso", "fila", "saturação", "utilização"]):
                detected_mode = "gargalos"
            elif any(word in question_lower for word in ["sku", "stock", "inventário", "inventario", "cobertura", "rop", "risco", "rutura"]):
                detected_mode = "inventario"
            elif any(word in question_lower for word in ["otd", "lead time", "leadtime", "plano", "planeamento", "setup"]):
                detected_mode = "planeamento"
            # Se não detectar, mantém "resumo"

        # Usar modo detectado para construir contexto
        context = engine.build_context_by_mode(detected_mode)
        context_json = json.dumps(context, ensure_ascii=False, indent=2)

        # Construir histórico da conversa (apenas últimas 3 mensagens para não sobrecarregar)
        conversation_lines = []
        recent_messages = request.messages[-6:] if len(request.messages) > 6 else request.messages
        for message in recent_messages:
            speaker = "Utilizador" if message.role == "user" else "Assistente"
            conversation_lines.append(f"{speaker}: {message.content}")

        conversation_text = "\n".join(conversation_lines) if len(conversation_lines) > 1 else ""

        # Construir prompt com a pergunta do utilizador em destaque
        prompt = get_prompt_by_mode(detected_mode, context_json, user_question=last_user_message)
        
        # Para chat, usar o mesmo SYSTEM_PROMPT rigoroso mas adaptado para perguntas diretas
        chat_system_prompt = SYSTEM_PROMPT + """

ADAPTAÇÃO PARA CHAT:
- Responde diretamente à pergunta do utilizador.
- Se a pergunta for específica (ex: "Qual é o SKU com maior risco?"), foca-te nisso.
- Se a pergunta for geral, responde diretamente sem fazer resumos do sistema.
- NÃO faças resumo executivo (isso só existe em Planeamento).
- Mantém sempre o rigor: identifica problemas reais, dados inconsistentes, e propõe ações concretas.
"""
        
        # Reorganizar para dar mais peso à pergunta do utilizador
        full_prompt = f"{chat_system_prompt}\n\n{prompt}"
        
        if conversation_text and len(recent_messages) > 1:
            full_prompt += f"\n\nHistórico recente da conversa:\n{conversation_text}"
        
        # Colocar a pergunta atual em destaque no final
        full_prompt += f"\n\n---\nPergunta atual do utilizador: {last_user_message}\n---\n\nResponde diretamente a esta pergunta específica, usando os dados do contexto acima. Se a pergunta for sobre algo que não está no contexto, diz isso claramente."

        # Temperatura por modo (chat = 0.45 para criatividade controlada)
        temperature = request.temperature if request.temperature is not None else 0.45

        llm = LocalLLM()
        try:
            answer_raw = llm.generate(prompt=full_prompt, temperature=temperature, max_tokens=800, num_ctx=4096)
            answer = answer_raw.strip()
        except LLMUnavailableError:
            return JSONResponse({"detail": "Modelo offline — iniciar Ollama."}, status_code=200)

        # Validar output do LLM
        loader = get_loader()
        valid_skus = []
        valid_resources = []
        
        if detected_mode == "inventario":
            inventory_data = loader.get_inventory_insights()
            valid_skus = [sku.get("sku", "") for sku in inventory_data.get("skus", [])]
        elif detected_mode in ["gargalos", "planeamento"]:
            bottlenecks_data = engine._get_bottlenecks_data()
            valid_resources = [r.get("recurso", "") for r in bottlenecks_data.get("top_resources", [])]
            # Também buscar recursos dos roteiros
            roteiros = loader.get_roteiros()
            if not roteiros.empty and "maquinas_possiveis" in roteiros.columns:
                all_resources = set()
                for machines in roteiros["maquinas_possiveis"].dropna():
                    if isinstance(machines, str):
                        # Extrair recursos do formato "M-16, M-133"
                        resources = re.findall(r'M[-\s]?(\d+)', machines)
                        all_resources.update([f"M-{r}" for r in resources])
                valid_resources.extend(list(all_resources))

        validation = validate_llm_output(answer, context, detected_mode, valid_skus, valid_resources)
        
        if not validation["is_valid"]:
            # Se houver erros críticos, tentar regenerar uma vez com prompt mais restritivo
            if len(validation["errors"]) > 0:
                stricter_prompt = full_prompt + "\n\nIMPORTANTE: Usa APENAS SKUs e recursos que estão explicitamente no contexto JSON acima. NÃO inventes."
                try:
                    answer = llm.generate(prompt=stricter_prompt, temperature=temperature * 0.8, max_tokens=800, num_ctx=4096).strip()
                    validation = validate_llm_output(answer, context, detected_mode, valid_skus, valid_resources)
                except:
                    pass
        
        # Usar texto sanitizado se houver erros
        final_answer = validation["sanitized_text"] if validation.get("sanitized_text") else answer

        model_name = getattr(llm, "model_name", "llama3:8b")
        return ChatResponse(
            answer=final_answer,
            model=model_name,
            used_context=context_json,
            mode=detected_mode,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha no chat: {exc}") from exc
