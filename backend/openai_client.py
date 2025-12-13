"""
Thin wrapper around the OpenAI API for the Nikufra Production OS MVP.
"""

from __future__ import annotations
import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)

# Prompt base
DEFAULT_SYSTEM_PROMPT = (
    "És o assistente operativo do Nikufra Production OS. "
    "Responde sempre em português europeu, profissional e conciso. "
    "Se faltar contexto, informa o utilizador e sugere o próximo passo."
)


class OpenAIClient:
    """Wrapper simples para Chat Completions da OpenAI."""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY não encontrado no .env")
        self.client = OpenAI(api_key=api_key)

    def ask(self, user_prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        """Pergunta ao modelo gpt-4o-mini e devolve resposta em string."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Erro OpenAI: {e}")
            return "Ocorreu um erro ao contactar a IA."


# Instância global
_openai = OpenAIClient()


def ask_openai(user_prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Função pública usada pelos restantes módulos."""
    return _openai.ask(user_prompt, system_prompt)

