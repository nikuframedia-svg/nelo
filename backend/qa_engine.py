"""
QA engine for Nikufra Production OS MVP.

Responsável por:
- Construir contexto a partir dos dados (Excel + plano).
- Chamar o LLM (OpenAI) com prompts orientados ao domínio industrial.
- Responder a perguntas de:
  * Percurso / rotas de um artigo.
  * Gargalo atual da fábrica.
  * Perguntas gerais sobre APS.
- Interpretar comandos industriais em linguagem natural.

R&D: WP2 (What-If + Explainable AI)
Hypothesis: H4.1 - LLM + structured parsing enables reliable industrial co-pilot
"""

from __future__ import annotations

import pandas as pd

from data_loader import load_dataset
from scheduler import compute_bottleneck, PLAN_CSV_PATH
from openai_client import ask_openai
from command_parser import parse_command, execute_command, CommandType


def _build_route_context_for_article(article_id: str) -> str:
    """
    Constroi uma descrição textual das rotas e operações para um dado artigo.
    """
    data = load_dataset()
    routing = data.routing

    routes_art = routing[routing["article_id"] == article_id]
    if routes_art.empty:
        return f"Não existem rotas configuradas para o artigo {article_id}."

    parts = [f"Artigo {article_id}. Rotas disponíveis:"]
    # garantir ordenação
    routes_art = routes_art.sort_values(["route_id", "route_label", "op_seq"])

    current_route = None
    for _, row in routes_art.iterrows():
        rid = str(row["route_id"])
        rlabel = str(row["route_label"])
        op_seq = int(row["op_seq"])
        op_code = str(row["op_code"])
        machine_id = str(row["primary_machine_id"])
        if rid != current_route:
            parts.append(f"\nRota {rlabel} (route_id={rid}):")
            current_route = rid
        parts.append(f"- seq {op_seq}: operação {op_code} na máquina {machine_id}")

    return "\n".join(parts)


def _build_bottleneck_context() -> str:
    """
    Lê o plano de produção e devolve um resumo textual do gargalo atual.
    """
    try:
        plan = pd.read_csv(PLAN_CSV_PATH, parse_dates=["start_time", "end_time"])
    except FileNotFoundError:
        return "Ainda não existe um plano calculado. Corre primeiro o scheduler para gerar o production_plan."

    info = compute_bottleneck(plan)
    if not info:
        return "O plano atual está vazio ou não foi possível calcular o gargalo."

    machine_id = info["machine_id"]
    total_minutes = info["total_minutes"]

    return (
        f"A máquina com maior carga planeada é a {machine_id}, "
        f"com um total de {total_minutes:.1f} minutos de trabalho. "
        "Esta máquina é candidata principal a gargalo no plano atual."
    )


def answer_question_text(message: str) -> str:
    """
    Função principal de QA.

    Recebe uma pergunta em texto livre e devolve uma resposta em texto,
    já formatada e pronta para ser enviada para o frontend.
    """
    data = load_dataset()
    orders = data.orders

    msg_lower = message.lower()

    # 1) Perguntas sobre percurso / rota de um artigo
    if "percurso" in msg_lower or "rota" in msg_lower:
        article_id = None
        # tentar encontrar um artigo do tipo "ART-100" dentro da pergunta
        for art in orders["article_id"].astype(str).unique():
            if art in message:
                article_id = art
                break

        if not article_id:
            return "Não consegui perceber para que artigo devo mostrar o percurso. Indica, por exemplo, 'ART-100'."

        context = _build_route_context_for_article(article_id)
        prompt = f"""
{context}

Pergunta do utilizador:
\"{message}\"

Tarefa:
- Explica o percurso de produção do artigo {article_id}, rota a rota.
- Realça qual rota é mais directa/rápida se isso for claro a partir dos dados.
- Responde em português de Portugal, em 3 a 5 frases.
"""
        return ask_openai(prompt)

    # 2) Perguntas sobre gargalo
    if "gargalo" in msg_lower:
        context = _build_bottleneck_context()
        prompt = f"""
Contexto:
{context}

Pergunta do utilizador:
\"{message}\"

Tarefa:
- Identifica explicitamente qual é a máquina gargalo.
- Explica em 3 ou 4 frases porque é que ela é gargalo.
- Sugere 2 ou 3 acções práticas para reduzir ou eliminar o gargalo (ex.: usar rota alternativa, reforçar turno, redistribuir operações).
Responde em português de Portugal.
"""
        return ask_openai(prompt)

    if "encadeado" in msg_lower:
        prompt = f"""
Contexto:
- O modo ENCADEADO vai permitir definir cadeias de máquinas (ex.: corte → soldadura → pintura) com buffers e transportes.
- Atualmente o APS ainda corre em modo NORMAL, mas estamos a preparar hooks para ativar essas cadeias.

Pedido do utilizador:
"{message}"

Tarefa:
- Explica em português de Portugal o que é planeamento encadeado.
- Clarifica que, por agora, o sistema ainda corre em modo NORMAL mas já suporta pré-configuração de cadeias.
- Sugere como o utilizador poderá indicar cadeias ou requisitos (ex.: mencionar máquinas em sequência, buffers, operadores dedicados).
"""
        return ask_openai(prompt)

    # 3) Perguntas genéricas sobre APS / produção
    context = (
        f"Existem actualmente {len(data.orders)} ordens de produção e "
        f"{len(data.machines)} máquinas registadas no sistema."
    )
    prompt = f"""
Contexto:
{context}

Pergunta:
\"{message}\"

Tarefa:
- Responde em português de Portugal, de forma profissional e concisa.
- Se a pergunta for sobre APS, planeamento, gargalos, rotas, OEE, etc., dá uma explicação simples e prática.
"""
    return ask_openai(prompt)


def answer_with_command_parsing(message: str) -> dict:
    """
    Enhanced answer function with industrial command parsing.
    
    First attempts to parse as a structured command.
    If successful with high confidence, executes the command.
    Otherwise, falls back to LLM-based answering.
    
    Returns:
        Dict with:
        - answer: The response text
        - command_detected: Whether a command was detected
        - command_type: Type of command if detected
        - action_taken: Description of action taken
        - requires_confirmation: Whether user confirmation is needed
    """
    # Try to parse as command
    parsed = parse_command(message)
    
    # High confidence command detected
    if parsed["confidence"] >= 0.7 and parsed["command_type"] != "unknown":
        # Execute the command
        result = execute_command(parsed)
        
        if result.get("success"):
            # Build response with explanation
            response = {
                "answer": result.get("message", "Comando executado."),
                "command_detected": True,
                "command_type": parsed["command_type"],
                "action_taken": parsed["suggested_action"],
                "requires_confirmation": parsed["requires_confirmation"],
                "data": result.get("result"),
            }
            
            # If command requires confirmation, ask
            if parsed["requires_confirmation"]:
                response["answer"] = (
                    f"Interpretei o comando como: **{parsed['suggested_action']}**\n\n"
                    f"Confirmas que pretendes executar esta ação?"
                )
            
            return response
        else:
            # Command recognized but execution failed
            return {
                "answer": result.get("message", "Não foi possível executar o comando."),
                "command_detected": True,
                "command_type": parsed["command_type"],
                "action_taken": None,
                "error": True,
            }
    
    # Medium confidence - ask for clarification
    elif 0.4 <= parsed["confidence"] < 0.7 and parsed["command_type"] != "unknown":
        return {
            "answer": (
                f"Parece que estás a pedir: **{parsed['suggested_action']}**\n\n"
                f"É isso que pretendes? Se sim, reformula de forma mais clara ou confirma."
            ),
            "command_detected": True,
            "command_type": parsed["command_type"],
            "action_taken": None,
            "needs_clarification": True,
        }
    
    # No command detected - use LLM for general Q&A
    answer = answer_question_text(message)
    return {
        "answer": answer,
        "command_detected": False,
        "command_type": None,
        "action_taken": None,
    }

