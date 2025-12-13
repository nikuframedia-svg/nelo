"""
ProdPlan 4.0 - Scheduling API
=============================

Endpoints REST para módulo de scheduling.

Endpoints:
- POST /scheduling/plan      - Gerar plano com engine selecionável
- GET  /scheduling/engines   - Lista engines disponíveis
- GET  /scheduling/rules     - Lista regras de dispatching
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scheduling", tags=["Scheduling"])


@router.get("/status")
async def get_scheduling_status():
    """Get scheduling module status."""
    return {
        "service": "Scheduling Engine",
        "version": "2.0.0",
        "status": "operational",
        "engines": ["heuristic", "milp", "cpsat"],
        "rules": ["FIFO", "SPT", "EDD", "CR", "WSPT", "ATC"],
        "features": ["data_driven_durations", "flow_shop", "multi_objective"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class PlanRequest(BaseModel):
    """Request para gerar plano de produção."""
    engine: str = Field(default="heuristic", description="Engine: heuristic, milp, cpsat")
    rule: str = Field(default="EDD", description="Regra de dispatching (para heurístico)")
    use_data_driven_durations: bool = Field(default=False, description="Usar durações históricas")
    time_limit_sec: float = Field(default=60.0, description="Limite de tempo para MILP/CP-SAT")
    gap_tolerance: float = Field(default=0.05, description="Gap de otimalidade para MILP")


class PlanResponse(BaseModel):
    """Response do plano gerado."""
    success: bool
    engine_used: str
    rule_used: Optional[str] = None
    solve_time_sec: float
    status: str
    operations: List[Dict[str, Any]]
    kpis: Dict[str, Any]
    warnings: List[str] = []
    data_driven_count: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/engines")
async def get_available_engines():
    """
    Lista engines de scheduling disponíveis.
    """
    return {
        "engines": [
            {
                "id": "heuristic",
                "name": "Heurístico",
                "description": "Regras de dispatching rápidas (FIFO, SPT, EDD, etc.)",
                "available": True,
            },
            {
                "id": "milp",
                "name": "MILP",
                "description": "Otimização matemática (Mixed-Integer Linear Programming)",
                "available": _check_milp_available(),
            },
            {
                "id": "cpsat",
                "name": "CP-SAT",
                "description": "Constraint Programming (mais eficiente para instâncias grandes)",
                "available": _check_cpsat_available(),
            },
            {
                "id": "drl",
                "name": "DRL (Experimental)",
                "description": "Deep Reinforcement Learning (em desenvolvimento)",
                "available": False,
            },
        ]
    }


@router.get("/rules")
async def get_dispatching_rules():
    """
    Lista regras de dispatching disponíveis.
    """
    return {
        "rules": [
            {"id": "FIFO", "name": "First In, First Out", "description": "Processa pela ordem de chegada"},
            {"id": "SPT", "name": "Shortest Processing Time", "description": "Prioriza operações mais curtas"},
            {"id": "EDD", "name": "Earliest Due Date", "description": "Prioriza por data de entrega"},
            {"id": "CR", "name": "Critical Ratio", "description": "Rácio crítico (tempo restante/processamento)"},
            {"id": "WSPT", "name": "Weighted SPT", "description": "SPT ponderado por prioridade"},
            {"id": "SQ", "name": "Shortest Queue", "description": "Máquina com menor fila"},
            {"id": "SETUP_AWARE", "name": "Setup Aware", "description": "Minimiza tempos de setup"},
        ]
    }


@router.post("/plan", response_model=PlanResponse)
async def generate_plan(request: PlanRequest):
    """
    Gera plano de produção usando engine selecionado.
    
    **Engines disponíveis:**
    - `heuristic`: Regras de dispatching (rápido)
    - `milp`: Otimização MILP (ótimo ou próximo)
    - `cpsat`: CP-SAT (eficiente para instâncias grandes)
    
    **Regras de dispatching (para heurístico):**
    - FIFO, SPT, EDD, CR, WSPT, SQ, SETUP_AWARE
    """
    try:
        # Carregar dados
        from ..data_loader import load_dataset
        data = load_dataset()
        
        # Criar instância de scheduling
        from types import create_instance_from_dataframes
        instance = create_instance_from_dataframes(
            orders_df=data.orders,
            routing_df=data.routing,
            machines_df=data.machines,
        )
        instance.use_data_driven_durations = request.use_data_driven_durations
        
        # Aplicar durações data-driven se solicitado
        if request.use_data_driven_durations:
            operations, data_driven_count = _apply_data_driven_durations(instance.operations)
            instance.operations = operations
        else:
            data_driven_count = 0
        
        # Executar engine selecionado
        if request.engine.lower() == "milp":
            from milp_models import solve_milp
            result = solve_milp(instance, request.time_limit_sec, request.gap_tolerance)
        elif request.engine.lower() == "cpsat":
            from cpsat_models import solve_cpsat
            result = solve_cpsat(instance, request.time_limit_sec)
        else:
            from heuristics import HeuristicScheduler
            scheduler = HeuristicScheduler(rule=request.rule)
            result = scheduler.build_schedule(instance)
        
        result["data_driven_count"] = data_driven_count
        
        return PlanResponse(
            success=result.get("success", True),
            engine_used=result.get("engine_used", request.engine),
            rule_used=result.get("rule_used"),
            solve_time_sec=result.get("solve_time_sec", 0),
            status=result.get("status", "unknown"),
            operations=result.get("scheduled_operations", []),
            kpis=result.get("kpis", {}),
            warnings=result.get("warnings", []),
            data_driven_count=data_driven_count,
        )
        
    except Exception as e:
        logger.error(f"Plan generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-driven/status")
async def get_data_driven_status():
    """
    Verifica disponibilidade de durações data-driven.
    """
    from data_driven_durations import DataDrivenDurations
    
    engine = DataDrivenDurations()
    stats = engine.get_statistics()
    
    return {
        "available": stats.get("loaded", False),
        "records": stats.get("records", 0),
        "operations_unique": stats.get("operations_unique", 0),
        "message": "Load historical data to enable data-driven durations" if not stats.get("loaded") else "Data-driven durations available",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _check_milp_available() -> bool:
    """Verifica se MILP está disponível."""
    try:
        from ortools.linear_solver import pywraplp
        return True
    except ImportError:
        return False


def _check_cpsat_available() -> bool:
    """Verifica se CP-SAT está disponível."""
    try:
        from ortools.sat.python import cp_model
        return True
    except ImportError:
        return False


def _apply_data_driven_durations(operations: List[Dict]) -> tuple:
    """
    Aplica durações data-driven às operações.
    
    Returns:
        (operations_updated, count_data_driven)
    """
    from data_driven_durations import DataDrivenDurations, create_sample_historical_data
    
    engine = DataDrivenDurations()
    
    # Tentar carregar dados históricos
    # TODO: Integrar com base de dados real
    # Por agora, usar dados de demonstração
    sample_data = create_sample_historical_data(500)
    engine.load_historical_data(sample_data)
    
    return engine.update_operations_with_estimates(operations)


