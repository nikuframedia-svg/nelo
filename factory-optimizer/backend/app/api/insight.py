from datetime import datetime
import json
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.etl.loader import get_loader
from app.llm import LLMUnavailableError, LocalLLM

router = APIRouter()


def _sanitize_dataframe(df: pd.DataFrame, limit: int = 12) -> list[Dict[str, Any]]:
    if df.empty:
        return []
    cleaned = df.head(limit).replace({np.nan: None})
    records: list[Dict[str, Any]] = []
    for row in cleaned.to_dict(orient="records"):
        normalised: Dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (datetime, pd.Timestamp)):
                normalised[key] = value.isoformat()
            elif isinstance(value, np.generic):
                normalised[key] = value.item()
            elif isinstance(value, (set, tuple)):
                normalised[key] = list(value)
            else:
                normalised[key] = value
        records.append(normalised)
    return records


def _build_context() -> str:
    loader = get_loader()
    status = loader.get_status()
    ordens = loader.get_ordens()
    roteiros = loader.get_roteiros()
    stocks = loader.get_stocks_snap()

    context = {
        "etl_status": status,
        "ordens_sample": _sanitize_dataframe(ordens),
        "roteiros_sample": _sanitize_dataframe(roteiros),
        "stocks_sample": _sanitize_dataframe(stocks),
        "generated_at": datetime.utcnow().isoformat(),
    }
    return json.dumps(context, ensure_ascii=False, indent=2)


@router.get("/summary")
async def insight_summary():
    contexto = _build_context()
    prompt = f"""
Tu és um especialista industrial e planeador de produção da Nikufra OPS,
que domina APS, OEE, inventário e gargalos de linha.
Tens acesso aos seguintes dados (JSON real):\n{contexto}\n\n
Deves responder em português europeu, com frases curtas e explicativas.
Usa o seguinte formato:

📊 **Resumo rápido (1 frase)**
🧠 **Causas principais**
⚙️ **Impacto**
🔧 **Ações recomendadas**
💰 **Ganho estimado**

Nunca dês respostas vagas — justifica sempre com os dados.

Objetivo: gera um resumo executivo diário do estado da fábrica (3 a 4 frases), destacando gargalos, efeitos no lead time, riscos de inventário e ação imediata recomendada.
Resposta:
"""

    llm = LocalLLM()
    try:
        summary = llm.generate(prompt=prompt, temperature=0.3, num_ctx=4096)
    except LLMUnavailableError:
        return JSONResponse({"detail": "Modelo offline — iniciar Ollama."}, status_code=200)

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "model": getattr(llm, "model_name", "llama3:8b"),
        "summary": summary.strip(),
        "context": contexto,
    }
