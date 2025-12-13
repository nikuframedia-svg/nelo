import json
import os
from typing import Any, Dict, Optional

import requests


class LLMUnavailableError(RuntimeError):
    """Raised when the local LLM endpoint cannot be reached or is offline."""


class LocalLLM:
    """Wrapper simples sobre um servidor LLM local (ex.: Ollama)."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3:8b")
        # ✅ Timeout reduzido para 30s (evita bloqueios longos)
        self.timeout = float(os.getenv("OLLAMA_TIMEOUT", 30))
        self.model_name = self.model

    def _build_payload(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        num_ctx: Optional[int],
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        options = extra_options.copy() if extra_options else {}
        if num_ctx:
            options.setdefault("num_ctx", num_ctx)

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if options:
            payload["options"] = options
        return payload

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 600,
        num_ctx: Optional[int] = 4096,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        endpoint = f"{self.base_url.rstrip('/')}/api/generate"
        payload = self._build_payload(prompt, temperature, max_tokens, num_ctx, extra_options)

        try:
            response = requests.post(endpoint, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise LLMUnavailableError("Modelo offline — iniciar Ollama.") from exc

        if not response.ok:
            detail = response.text
            try:
                detail_json = response.json()
                detail = detail_json.get("error") or detail_json.get("message") or json.dumps(detail_json)
            except ValueError:
                pass
            raise LLMUnavailableError(detail or "Modelo offline — iniciar Ollama.")

        data = response.json()
        generated = data.get("response") or data.get("data", "")
        if not isinstance(generated, str):
            generated = str(generated)
        return generated.strip()
