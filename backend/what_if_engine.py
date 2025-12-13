"""
What-If engine – converte linguagem natural em deltas estruturados e compara cenários.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from openai_client import ask_openai
from data_loader import DataBundle, load_dataset, as_records
from scheduler import build_plan, compute_bottleneck


SCENARIO_SYSTEM_PROMPT = (
    "És um engenheiro de planeamento industrial a trabalhar no Nikufra Production OS. "
    "O utilizador vai descrever um cenário What-If em linguagem natural "
    "(ex.: nova máquina, mudança de tempos, novos turnos, operadores, etc.). "
    "Tens acesso ao seguinte contexto:\n\n"
    "- Grupos de máquinas típicos: CUTTING, BENDING, WELDING, PAINTING, ASSEMBLY.\n"
    "- Exemplos de op_code: CUT-01, CUT-02, BND-01, BND-02, WLD-01, WLD-02, PNT-01, ASM-01.\n\n"
    "Deves devolver APENAS um JSON válido (sem texto à volta) com esta estrutura:\n\n"
    "{\n"
    '  "new_machines": [\n'
    "    {\n"
    '      "machine_id": str,\n'
    '      "description": str,\n'
    '      "group": str,              # ex.: \"CUTTING\", \"WELDING\"\n'
    '      "speed_factor_delta": float  # 1.2 = +20% velocidade, 0.8 = -20% velocidade\n'
    "    }\n"
    "  ],\n"
    '  "updated_times": [\n'
    "    {\n"
    '      "article_id": str,\n'
    '      "op_code": str,\n'
    '      "time_factor": float   # 0.8 = -20% tempo, 1.1 = +10% tempo\n'
    "    }\n"
    "  ],\n"
    '  "updated_shifts": [\n'
    "    {\n"
    '      "machine_id": str,\n'
    '      "shift_id": str,\n'
    '      "change": str  # ex.: \"add_night_shift\", \"remove_afternoon_shift\"\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Regras importantes:\n"
    "- Se alguma secção não fizer sentido para o cenário, devolve essa lista vazia.\n"
    "- NUNCA escrevas texto fora do JSON (sem comentários, sem explicações, sem markdown).\n"
    "- Tenta sempre preencher description, group e op_code com valores plausíveis quando possível.\n"
)


def describe_scenario_nl(scenario_text: str) -> Dict[str, Any]:
    """
    Recebe texto livre com um cenário What-If e devolve um dict JSON estruturado.
    O prompt inclui contexto real: artigos, op_codes e grupos de máquinas existentes.
    """
    data = load_dataset()
    machines = data.machines
    operations = data.operations
    orders = data.orders

    machine_groups = machines["work_center_group"].astype(str).unique().tolist()
    machine_ids = machines["machine_id"].astype(str).unique().tolist()
    op_codes = operations["op_code"].astype(str).unique().tolist()
    article_ids = orders["article_id"].astype(str).unique().tolist()

    context = (
        "Contexto actual da fábrica:\n"
        f"- Machine groups: {machine_groups}\n"
        f"- Machines: {machine_ids}\n"
        f"- op_code disponíveis: {op_codes}\n"
        f"- Artigos: {article_ids}\n\n"
    )

    user_prompt = (
        context
        + "Descrição do cenário:\n\"\"\"\n"
        + scenario_text
        + "\n\"\"\"\n\nGera o JSON conforme especificado."
    )

    raw_response = ask_openai(user_prompt, system_prompt=SCENARIO_SYSTEM_PROMPT)

    # Tentar fazer parse ao JSON que o LLM devolveu
    try:
        scenario = json.loads(raw_response)
    except json.JSONDecodeError:
        # Se o modelo meter texto fora do JSON, tenta limpar a primeira/última chaveta
        try:
            start = raw_response.index("{")
            end = raw_response.rindex("}") + 1
            scenario = json.loads(raw_response[start:end])
        except Exception:
            # Fallback: devolve estrutura vazia + debug
            return {
                "error": "Falha a interpretar JSON do cenário.",
                "raw_response": raw_response,
                "new_machines": [],
                "updated_times": [],
                "updated_shifts": [],
            }

    # Garantir que as chaves principais existem
    scenario.setdefault("new_machines", [])
    scenario.setdefault("updated_times", [])
    scenario.setdefault("updated_shifts", [])

    return scenario


@dataclass
class ScenarioDelta:
    """Delta estruturado para aplicar alterações ao DataBundle."""

    new_machines: List[Dict[str, Any]]
    updated_times: List[Dict[str, Any]]
    updated_shifts: List[Dict[str, Any]]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScenarioDelta":
        return cls(
            new_machines=payload.get("new_machines") or [],
            updated_times=payload.get("updated_times") or [],
            updated_shifts=payload.get("updated_shifts") or [],
        )


def apply_scenario_delta(bundle: DataBundle, delta: ScenarioDelta) -> DataBundle:
    """
    Aplica o ScenarioDelta a uma cópia do DataBundle e devolve um NOVO DataBundle.

    Regras MVP:
    - new_machines → adicionadas a data.machines (se não existirem).
    - updated_times → ajusta base_time_per_unit_min na routing para (article_id, op_code).
    - updated_shifts → placeholder (TODO para fases seguintes).
    """

    orders = bundle.orders.copy(deep=True)
    operations = bundle.operations.copy(deep=True)
    machines = bundle.machines.copy(deep=True)
    routing = bundle.routing.copy(deep=True)
    shifts = bundle.shifts.copy(deep=True)
    downtime = bundle.downtime.copy(deep=True)
    setup_matrix = bundle.setup_matrix.copy(deep=True)

    # 1) adicionar novas máquinas
    for machine in delta.new_machines:
        machine_id = str(machine.get("machine_id", "")).strip()
        if not machine_id:
            continue
        if "machine_id" in machines and (machines["machine_id"] == machine_id).any():
            continue

        speed_factor = float(machine.get("speed_factor_delta", 1.0) or 1.0)
        row = {
            "machine_id": machine_id,
            "description": machine.get("description", ""),
            "work_center_group": machine.get("group", ""),
            "speed_factor": speed_factor,
        }
        machines = pd.concat([machines, pd.DataFrame([row])], ignore_index=True)

    # 2) atualizar tempos de operações
    routing["base_time_per_unit_min"] = pd.to_numeric(
        routing["base_time_per_unit_min"], errors="coerce"
    ).fillna(0.0)
    for update in delta.updated_times:
        article_id = str(update.get("article_id", "")).strip()
        op_code = str(update.get("op_code", "")).strip()
        time_factor = float(update.get("time_factor", 1.0) or 1.0)
        if not op_code or time_factor == 1.0:
            continue

        mask = routing["op_code"] == op_code
        if article_id:
            mask = mask & (routing["article_id"] == article_id)
        if not mask.any():
            continue
        routing.loc[mask, "base_time_per_unit_min"] = (
            routing.loc[mask, "base_time_per_unit_min"] * time_factor
        )

    # 3) updated_shifts → reservado para futura fase
    # TODO: implementar lógica real para alterar shifts com base em delta.updated_shifts

    return DataBundle(
        orders=orders,
        operations=operations,
        machines=machines,
        routing=routing,
        shifts=shifts,
        downtime=downtime,
        setup_matrix=setup_matrix,
        raw_path=bundle.raw_path,
        loaded_at=bundle.loaded_at,
    )


def build_scenario_comparison(
    scenario_text: str,
    *,
    include_plan_details: bool = False,
    max_operations: int = 100,
) -> Dict[str, Any]:
    """
    Pipeline completo:
    - Descreve cenário via LLM
    - Aplica o delta ao DataBundle
    - Recalcula plano baseline vs cenário
    - Devolve métricas comparativas
    """

    base_bundle = load_dataset()
    base_plan = build_plan(base_bundle, mode="NORMAL")
    base_bottleneck = compute_bottleneck(base_plan) or {}

    raw_delta = describe_scenario_nl(scenario_text)
    delta = ScenarioDelta.from_dict(raw_delta)

    scenario_bundle = apply_scenario_delta(base_bundle, delta)
    scenario_plan = build_plan(scenario_bundle, mode="NORMAL")
    scenario_bottleneck = compute_bottleneck(scenario_plan) or {}

    def _total_duration(plan: pd.DataFrame) -> float:
        if plan.empty:
            return 0.0
        return float(plan["duration_min"].sum())

    result: Dict[str, Any] = {
        "delta": raw_delta,
        "baseline": {
            "bottleneck": base_bottleneck,
            "total_duration_min": _total_duration(base_plan),
            "num_operations": int(len(base_plan)),
        },
        "scenario": {
            "bottleneck": scenario_bottleneck,
            "total_duration_min": _total_duration(scenario_plan),
            "num_operations": int(len(scenario_plan)),
        },
    }

    if include_plan_details:
        result["plans"] = {
            "baseline": _plan_details(base_plan, max_operations),
            "scenario": _plan_details(scenario_plan, max_operations),
        }

    return result


def _plan_details(plan_df: pd.DataFrame, max_operations: int) -> Dict[str, Any]:
    if plan_df.empty:
        return {"operations": [], "total": 0}
    snippet = plan_df.head(max_operations)
    return {
        "total": int(len(plan_df)),
        "operations": as_records(snippet),
    }


