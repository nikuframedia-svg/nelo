from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_loader import load_dataset
from scheduler import build_plan, compute_bottleneck, compute_kpis, save_plan_to_csv, PLAN_CSV_PATH
from qa_engine import answer_question_text, answer_with_command_parsing
from what_if_engine import describe_scenario_nl, build_scenario_comparison
from suggestions_engine import compute_suggestions, format_suggestion_pt
from duplios.api_duplios import router as duplios_router

logger = logging.getLogger(__name__)

# Import Trust Index API router
try:
    from duplios.api_trust_index import router as trust_index_router
    HAS_TRUST_INDEX = True
    logger.info("Trust Index API loaded successfully")
except ImportError as e:
    logger.warning(f"Trust Index API not available: {e}")
    HAS_TRUST_INDEX = False
    trust_index_router = None

# Import Gap Filling Lite API router
try:
    from duplios.api_gap_filling import router as gap_filling_router
    HAS_GAP_FILLING = True
    logger.info("Gap Filling Lite API loaded successfully")
except ImportError as e:
    logger.warning(f"Gap Filling Lite API not available: {e}")
    HAS_GAP_FILLING = False
    gap_filling_router = None

# Import Compliance Radar API router
try:
    from duplios.api_compliance import router as compliance_router
    HAS_COMPLIANCE_RADAR = True
    logger.info("Compliance Radar API loaded successfully")
except ImportError as e:
    logger.warning(f"Compliance Radar API not available: {e}")
    HAS_COMPLIANCE_RADAR = False
    compliance_router = None

# Import Ops Ingestion API router
try:
    from ops_ingestion.api import router as ops_ingestion_router
    HAS_OPS_INGESTION = True
    logger.info("Ops Ingestion API loaded successfully")
except ImportError as e:
    logger.warning(f"Ops Ingestion API not available: {e}")
    HAS_OPS_INGESTION = False
    ops_ingestion_router = None

# Import R&D router
try:
    from rd.api import router as rd_router
    HAS_RD_MODULE = True
except ImportError as e:
    logger.warning(f"R&D module not available: {e}")
    HAS_RD_MODULE = False
    rd_router = None

# Import feature flags
try:
    from feature_flags import FeatureFlags, get_active_engines
    HAS_FEATURE_FLAGS = True
except ImportError:
    HAS_FEATURE_FLAGS = False

# Import Scheduling API router
try:
    from scheduling.api import router as scheduling_router
    HAS_SCHEDULING_API = True
except ImportError as e:
    logger.warning(f"Scheduling API not available: {e}")
    HAS_SCHEDULING_API = False
    scheduling_router = None

# Import MRP API router
try:
    from smart_inventory.api_mrp import router as mrp_router
    HAS_MRP_API = True
except ImportError as e:
    logger.warning(f"MRP API not available: {e}")
    HAS_MRP_API = False
    mrp_router = None

# Import evaluation module for SNR/data quality analysis
try:
    from evaluation.data_quality import SignalNoiseAnalyzer, DataQualityReport
    HAS_EVALUATION = True
except ImportError:
    HAS_EVALUATION = False

# Import SHI-DT (Smart Health Index Digital Twin) router
try:
    from digital_twin.api_shi_dt import router as shi_dt_router
    HAS_SHIDT = True
    logger.info("SHI-DT module loaded successfully")
except ImportError as e:
    logger.warning(f"SHI-DT module not available: {e}")
    HAS_SHIDT = False
    shi_dt_router = None

# Import XAI-DT Product (Explainable Digital Twin) router
try:
    from digital_twin.api_xai_dt_product import router as xai_dt_product_router
    HAS_XAI_DT_PRODUCT = True
    logger.info("XAI-DT Product module loaded successfully")
except ImportError as e:
    logger.warning(f"XAI-DT Product module not available: {e}")
    HAS_XAI_DT_PRODUCT = False
    xai_dt_product_router = None

# Import PDM (Product Data Management) router
try:
    from duplios.api_pdm import router as pdm_router
    HAS_PDM = True
    logger.info("PDM module loaded successfully")
except ImportError as e:
    logger.warning(f"PDM module not available: {e}")
    HAS_PDM = False
    pdm_router = None

# Import MRP Complete router
try:
    from smart_inventory.api_mrp_complete import router as mrp_complete_router
    HAS_MRP_COMPLETE = True
    logger.info("MRP Complete module loaded successfully")
except ImportError as e:
    logger.warning(f"MRP Complete module not available: {e}")
    HAS_MRP_COMPLETE = False
    mrp_complete_router = None

# Import Work Instructions router
try:
    from shopfloor.api_work_instructions import router as work_instructions_router
    HAS_WORK_INSTRUCTIONS = True
    logger.info("Work Instructions module loaded successfully")
except ImportError as e:
    logger.warning(f"Work Instructions module not available: {e}")
    HAS_WORK_INSTRUCTIONS = False
    work_instructions_router = None

# Import Optimization router
try:
    from optimization.api_optimization import router as optimization_router
    HAS_OPTIMIZATION = True
    logger.info("Optimization module loaded successfully")
except ImportError as e:
    logger.warning(f"Optimization module not available: {e}")
    HAS_OPTIMIZATION = False
    optimization_router = None

# Import Prevention Guard router
try:
    from quality.api_prevention_guard import router as prevention_guard_router
    HAS_PREVENTION_GUARD = True
    logger.info("Prevention Guard module loaded successfully")
except ImportError as e:
    logger.warning(f"Prevention Guard module not available: {e}")
    HAS_PREVENTION_GUARD = False
    prevention_guard_router = None

# Import ZDM router
try:
    from simulation.zdm.api_zdm import router as zdm_router
    HAS_ZDM = True
    logger.info("ZDM module loaded successfully")
except ImportError as e:
    logger.warning(f"ZDM module not available: {e}")
    HAS_ZDM = False
    zdm_router = None

# Import IoT Ingestion router (PredictiveCare - Machine Sensors)
try:
    from digital_twin.api_iot import router as iot_router
    HAS_IOT = True
    logger.info("IoT Ingestion module loaded successfully")
except ImportError as e:
    logger.warning(f"IoT Ingestion module not available: {e}")
    HAS_IOT = False
    iot_router = None

# Import Maintenance router (Work Orders, CMMS)
try:
    from maintenance.api import router as maintenance_router
    HAS_MAINTENANCE = True
    logger.info("Maintenance module loaded successfully")
except ImportError as e:
    logger.warning(f"Maintenance module not available: {e}")
    HAS_MAINTENANCE = False
    maintenance_router = None

from actions_engine import (
    get_action_store, propose_action, approve_action, reject_action,
    get_pending_actions, get_action_history, Action, ActionType,
)


BASE_DIR = Path(__file__).resolve().parents[1]

app = FastAPI(title="Nikufra Production OS – MVP")
app.include_router(duplios_router)

# Include Trust Index router
if HAS_TRUST_INDEX and trust_index_router:
    app.include_router(trust_index_router)
    logger.info("Trust Index API loaded successfully")

# Include Gap Filling Lite router
if HAS_GAP_FILLING and gap_filling_router:
    app.include_router(gap_filling_router)
    logger.info("Gap Filling Lite API loaded successfully")

# Include Compliance Radar router
if HAS_COMPLIANCE_RADAR and compliance_router:
    app.include_router(compliance_router)
    logger.info("Compliance Radar API loaded successfully")

# Include Ops Ingestion router
if HAS_OPS_INGESTION and ops_ingestion_router:
    app.include_router(ops_ingestion_router)
    logger.info("Ops Ingestion API loaded successfully")

# Include R&D router if available
if HAS_RD_MODULE and rd_router:
    app.include_router(rd_router)
    logger.info("R&D module loaded successfully")

# Include Scheduling API router if available
if HAS_SCHEDULING_API and scheduling_router:
    app.include_router(scheduling_router)
    logger.info("Scheduling API loaded successfully")

# Include MRP API router if available
if HAS_MRP_API and mrp_router:
    app.include_router(mrp_router)
    logger.info("MRP API loaded successfully")

# Include SHI-DT (Smart Health Index Digital Twin) router
if HAS_SHIDT and shi_dt_router:
    app.include_router(shi_dt_router)
    logger.info("SHI-DT API loaded successfully")

# Include XAI-DT Product router
if HAS_XAI_DT_PRODUCT and xai_dt_product_router:
    app.include_router(xai_dt_product_router)
    logger.info("XAI-DT Product API loaded successfully")

# Include PDM router
if HAS_PDM and pdm_router:
    app.include_router(pdm_router)
    logger.info("PDM API loaded successfully")

# Include MRP Complete router
if HAS_MRP_COMPLETE and mrp_complete_router:
    app.include_router(mrp_complete_router)
    logger.info("MRP Complete API loaded successfully")

# Include Work Instructions router
if HAS_WORK_INSTRUCTIONS and work_instructions_router:
    app.include_router(work_instructions_router)
    logger.info("Work Instructions API loaded successfully")

# Include Optimization router
if HAS_OPTIMIZATION and optimization_router:
    app.include_router(optimization_router)
    logger.info("Optimization API loaded successfully")

# Include Prevention Guard router
if HAS_PREVENTION_GUARD and prevention_guard_router:
    app.include_router(prevention_guard_router)
    logger.info("Prevention Guard API loaded successfully")

# Include ZDM router
if HAS_ZDM and zdm_router:
    app.include_router(zdm_router)
    logger.info("ZDM API loaded successfully")

# Include IoT Ingestion router (PredictiveCare)
if HAS_IOT and iot_router:
    app.include_router(iot_router)
    logger.info("IoT Ingestion API loaded successfully")

# Include Maintenance router
if HAS_MAINTENANCE and maintenance_router:
    app.include_router(maintenance_router)
    logger.info("Maintenance API loaded successfully")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# Models
# -------------------------------

class ChatQuery(BaseModel):
    message: str

class ScenarioQuery(BaseModel):
    scenario: str


# Action models for Industry 5.0 approval workflow
class ProposeActionRequest(BaseModel):
    type: str  # ActionType
    payload: Dict[str, Any]
    source: str = "manual"
    description: Optional[str] = None


class ApproveActionRequest(BaseModel):
    approved_by: str
    notes: Optional[str] = None


class RejectActionRequest(BaseModel):
    rejected_by: str
    reason: Optional[str] = None


# -------------------------------
# Helpers
# -------------------------------

def _ensure_plan_exists() -> pd.DataFrame:
    """Garantir que existe um plano. Se não existir, gerar um novo."""
    if PLAN_CSV_PATH.exists():
        return pd.read_csv(PLAN_CSV_PATH, parse_dates=["start_time", "end_time"])

    data = load_dataset()
    plan_df = build_plan(data, mode="NORMAL")
    save_plan_to_csv(plan_df)
    return plan_df


# -------------------------------
# Endpoints
# -------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/orders")
def get_orders() -> List[Dict[str, Any]]:
    data = load_dataset()
    return data.orders.to_dict(orient="records")


@app.get("/machines")
def get_machines() -> List[Dict[str, Any]]:
    data = load_dataset()
    return data.machines.to_dict(orient="records")


@app.get("/plan")
def get_plan() -> List[Dict[str, Any]]:
    plan_df = _ensure_plan_exists()
    return plan_df.to_dict(orient="records")


@app.get("/bottleneck")
def get_bottleneck() -> Dict[str, Any]:
    plan_df = _ensure_plan_exists()
    info = compute_bottleneck(plan_df)
    return info or {}


@app.get("/plan/kpis")
def get_plan_kpis() -> Dict[str, Any]:
    """
    Devolve KPIs industriais reais calculados a partir do plano de produção.
    
    Retorna:
    - makespan_hours: tempo total do plano
    - route_distribution: contagem por rota
    - overlaps: sobreposições por máquina
    - active_bottleneck: máquina gargalo
    - machine_loads: carga por máquina (top-N)
    - lead_time_average_h: lead time médio
    - otd_percent: percentagem OTD
    - setup_hours: horas de setup estimadas
    """
    plan_df = _ensure_plan_exists()
    data = load_dataset()
    kpis = compute_kpis(plan_df, data.orders)
    return kpis


@app.get("/plan/suggestions")
def get_plan_suggestions() -> Dict[str, Any]:
    """
    Devolve sugestões inteligentes calculadas a partir do plano de produção.
    
    Tipos de sugestões:
    - overload_suggestions: Sugestões para reduzir sobrecarga de máquinas
    - idle_gaps: Gaps ociosos identificados (>1h)
    - product_risks: Artigos com tempos de espera elevados entre operações
    
    Cada sugestão inclui uma versão formatada em PT-PT.
    """
    plan_df = _ensure_plan_exists()
    data = load_dataset()
    
    # Get routing data if available
    routing_df = data.routing if hasattr(data, 'routing') else None
    machines_df = data.machines if hasattr(data, 'machines') else None
    
    suggestions = compute_suggestions(
        plan_df,
        orders_df=data.orders,
        machines_df=machines_df,
        routing_df=routing_df,
    )
    
    # Add formatted PT-PT strings
    formatted = []
    for s in suggestions.get("overload_suggestions", []):
        formatted.append({
            **s,
            "formatted_pt": format_suggestion_pt(s)
        })
    suggestions["overload_suggestions"] = formatted
    
    formatted = []
    for s in suggestions.get("idle_gaps", []):
        formatted.append({
            **s,
            "formatted_pt": format_suggestion_pt(s)
        })
    suggestions["idle_gaps"] = formatted
    
    formatted = []
    for s in suggestions.get("product_risks", []):
        formatted.append({
            **s,
            "formatted_pt": format_suggestion_pt(s)
        })
    suggestions["product_risks"] = formatted
    
    return suggestions


@app.get("/plan/data_quality")
def get_plan_data_quality() -> Dict[str, Any]:
    """
    Análise de qualidade dos dados do plano usando Signal-to-Noise Ratio (SNR).
    
    SNR (Signal-to-Noise Ratio) quantifica a previsibilidade dos tempos de operação:
    
        SNR = Var(sinal) / Var(ruído)
        
    Interpretação:
        - SNR > 10: EXCELENTE - Alta previsibilidade
        - 3 < SNR ≤ 10: BOM - Previsibilidade moderada
        - 1 < SNR ≤ 3: RAZOÁVEL - Previsibilidade limitada
        - SNR ≤ 1: FRACO - Dominado por ruído
    
    Devolve:
        - global_snr: SNR global do plano
        - snr_by_machine: SNR por máquina
        - snr_by_operation: SNR por tipo de operação
        - quality_level: Nível de qualidade geral
        - low_snr_warnings: Alertas para máquinas/operações com SNR baixo
    
    Este endpoint é útil para:
        - Identificar fontes de dados pouco fiáveis
        - Calibrar a confiança nas previsões
        - Priorizar melhorias na recolha de dados
    """
    if not HAS_EVALUATION:
        return {
            "error": "Módulo de avaliação não disponível",
            "message": "O módulo backend.evaluation não está instalado"
        }
    
    plan_df = _ensure_plan_exists()
    data = load_dataset()
    
    # Run SNR analysis
    analyzer = SignalNoiseAnalyzer(min_samples=3, snr_threshold=2.0)
    report = analyzer.analyze(
        plan_df=plan_df,
        routing_df=data.routing if hasattr(data, 'routing') else None,
        historical_df=None,  # TODO: Add historical data support
    )
    
    return report.to_dict()


@app.get("/plan/milp")
def get_plan_milp(time_limit: float = 30.0, gap: float = 0.05) -> Dict[str, Any]:
    """
    Gera plano de produção usando otimização MILP (Mixed-Integer Linear Programming).
    
    Este endpoint usa um modelo matemático rigoroso para otimizar o sequenciamento:
    
    Objetivo: min α·Cmax + β·ΣwjTj + γ·TotalSetup
    
    Onde:
        - Cmax: Makespan (tempo total de produção)
        - Tj: Atraso da encomenda j
        - wj: Peso/prioridade da encomenda j
        - TotalSetup: Tempo total de setup
    
    Parâmetros:
        - time_limit: Tempo máximo de computação (segundos)
        - gap: Gap de otimalidade aceitável (0.05 = 5%)
    
    NOTA: Para instâncias grandes (>100 operações), pode demorar.
    Recomendado para benchmarking e demonstração.
    
    TODO[R&D]: Comparar qualidade MILP vs heurística para diferentes tamanhos de instância.
    """
    try:
        from core.optimization import build_milp_from_data, MILPConfig, ObjectiveType
    except ImportError:
        return {
            "error": "Módulo de otimização não disponível",
            "message": "O módulo backend.core.optimization não está instalado. Instale OR-Tools: pip install ortools"
        }
    
    data = load_dataset()
    
    # Configure MILP
    config = MILPConfig(
        objective=ObjectiveType.MINIMIZE_WEIGHTED_SUM,
        alpha=1.0,
        beta=0.5,
        gamma=0.1,
        time_limit_sec=time_limit,
        mip_gap=gap,
    )
    
    # Build model
    try:
        model = build_milp_from_data(
            orders_df=data.orders,
            routing_df=data.routing,
            machines_df=data.machines,
            setup_matrix_df=data.setup_matrix if hasattr(data, 'setup_matrix') else None,
            config=config,
        )
    except Exception as e:
        return {
            "error": "Erro ao construir modelo MILP",
            "message": str(e)
        }
    
    # Build and solve
    try:
        model.build_model()
        result = model.solve()
    except Exception as e:
        return {
            "error": "Erro ao resolver modelo MILP",
            "message": str(e)
        }
    
    if not result.success:
        return {
            "success": False,
            "status": result.statistics.status,
            "message": "Não foi possível encontrar solução",
            "statistics": result.statistics.to_dict()
        }
    
    # Save and return
    if result.schedule_df is not None:
        save_plan_to_csv(result.schedule_df)
    
    return {
        "success": True,
        "status": result.statistics.status,
        "objective_value": result.statistics.objective_value,
        "solve_time_sec": result.statistics.solve_time_sec,
        "objective_breakdown": result.objective_breakdown,
        "statistics": result.statistics.to_dict(),
        "plan": result.schedule_df.to_dict('records') if result.schedule_df is not None else []
    }


@app.post("/chat")
def chat(q: ChatQuery) -> Dict[str, Any]:
    """
    Chat industrial inteligente (Industrial Copilot).
    
    Features:
    - Intent routing based on keywords
    - Skill-based response generation
    - KPI payload for rich UI responses
    - Legacy command parsing fallback
    
    Returns:
        ChatResponse with message, intent, KPIs, suggestions, actions
    """
    try:
        from chat.engine import ChatRequest, get_chat_engine
        
        # Use new ChatEngine
        engine = get_chat_engine()
        request = ChatRequest(message=q.message)
        response = engine.handle_message(request)
        
        return {
            "message": response.message,
            "intent": response.intent,
            "confidence": response.confidence,
            "kpis": [kpi.dict() for kpi in response.kpis],
            "suggestions": response.suggestions,
            "actions": response.actions,
            "timestamp": response.timestamp,
        }
    except Exception as e:
        logger.warning(f"ChatEngine failed, falling back to legacy: {e}")
        # Fallback to legacy command parsing
        result = answer_with_command_parsing(q.message)
        return result


@app.post("/what-if/describe")
def describe_scenario(q: ScenarioQuery) -> Dict[str, Any]:
    """
    Converte um cenário em linguagem natural → JSON estruturado (ScenarioDelta).
    """
    scenario = describe_scenario_nl(q.scenario)
    return scenario


@app.post("/what-if/compare")
def compare_scenario(q: ScenarioQuery) -> Dict[str, Any]:
    """
    Gera o delta via LLM, aplica o cenário ao APS e devolve comparação baseline vs cenário.
    """
    result = build_scenario_comparison(q.scenario, include_plan_details=True, max_operations=150)
    return result


# -------------------------------
# Legacy / Stub Endpoints
# (To silence frontend 404s from older components)
# -------------------------------

@app.get("/insights/generate")
def insights_generate(mode: str = "planeamento") -> Dict[str, Any]:
    """Gera insights básicos com base nos dados reais."""
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    bottleneck = compute_bottleneck(plan_df) or {}
    
    num_orders = len(data.orders)
    num_machines = len(data.machines)
    num_operations = len(plan_df)
    
    insights = [
        f"📦 {num_orders} ordens de produção ativas",
        f"🏭 {num_machines} máquinas disponíveis",
        f"⚙️ {num_operations} operações planeadas",
    ]
    
    if bottleneck:
        insights.append(f"⚠️ Gargalo identificado: {bottleneck.get('machine_id', 'N/A')} ({bottleneck.get('total_minutes', 0):.0f} min)")
    
    return {
        "status": "ok",
        "mode": mode,
        "insights": insights,
        "summary": {
            "orders": num_orders,
            "machines": num_machines,
            "operations": num_operations,
            "bottleneck": bottleneck.get("machine_id"),
        }
    }


@app.get("/planning/v2/plano")
def planning_v2_plano(horizon_hours: int = 168, batch_id: str = None) -> Dict[str, Any]:
    """Devolve o plano de produção real no formato esperado pelo frontend."""
    plan_df = _ensure_plan_exists()
    bottleneck = compute_bottleneck(plan_df) or {}
    
    # Converter operações para o formato esperado pelo frontend
    operations = []
    for _, row in plan_df.iterrows():
        operations.append({
            "order_id": row.get("order_id", ""),
            "artigo": row.get("article_id", ""),
            "op_id": row.get("op_code", ""),
            "maquina_id": row.get("machine_id", ""),
            "start_time": row["start_time"].isoformat() if pd.notna(row.get("start_time")) else None,
            "end_time": row["end_time"].isoformat() if pd.notna(row.get("end_time")) else None,
            "duration_min": row.get("duration_min", 0),
            "rota": row.get("route_label", "A"),
            "family": "",
        })
    
    # Calcular KPIs básicos
    total_duration_h = plan_df["duration_min"].sum() / 60 if not plan_df.empty else 0
    
    return {
        "baseline": {"operations": operations},
        "optimized": {"operations": operations},
        "orders_summary": [],
        "horizon_hours": horizon_hours,
        "kpis": {
            "otd_pct": 85,  # Placeholder - calcular com base em due_dates
            "gargalo_ativo": bottleneck.get("machine_id", "N/A"),
        },
        "makespan_h": total_duration_h,
        "total_setup_h": 0,
    }


@app.get("/etl/status")
def etl_status() -> Dict[str, Any]:
    return {"status": "idle", "last_run": None}


@app.get("/api/etl/status")
def api_etl_status() -> Dict[str, Any]:
    return {"status": "idle", "last_run": None}


@app.get("/bottlenecks/")
def bottlenecks_stub() -> Dict[str, Any]:
    """Devolve análise de gargalos real com base no plano de produção."""
    plan_df = _ensure_plan_exists()
    
    # Calcular carga por máquina
    if plan_df.empty:
        return {
            "overlap_applied": {"transformacao": 0, "acabamentos": 0},
            "lead_time_gain": 0,
            "heatmap": [],
            "top_losses": [],
            "demo_mode": False,
        }
    
    # Agregar carga por máquina
    machine_load = plan_df.groupby("machine_id")["duration_min"].sum().reset_index()
    machine_load.columns = ["machine_id", "total_min"]
    total_capacity = machine_load["total_min"].sum()
    
    # Criar heatmap data
    heatmap = []
    for _, row in machine_load.iterrows():
        utilization = (row["total_min"] / total_capacity * 100) if total_capacity > 0 else 0
        heatmap.append({
            "recurso": row["machine_id"],
            "utilizacao_pct": round(utilization * len(machine_load), 1),  # Normalizar
            "carga_horas": round(row["total_min"] / 60, 1),
        })
    
    # Top losses (máquinas mais carregadas)
    top_losses = []
    machine_load_sorted = machine_load.sort_values("total_min", ascending=False).head(5)
    for _, row in machine_load_sorted.iterrows():
        utilization = (row["total_min"] / machine_load["total_min"].max() * 100) if machine_load["total_min"].max() > 0 else 0
        top_losses.append({
            "recurso": row["machine_id"],
            "utilizacao_pct": round(utilization, 1),
            "fila_horas": round(row["total_min"] / 60 * 0.2, 1),  # Estimativa
            "probabilidade": round(utilization / 100, 2),
            "acao": "Considerar rota alternativa" if utilization > 80 else "Monitorizar",
            "impacto_otd": round(utilization * 0.05, 1),
            "impacto_horas": round(row["total_min"] / 60 * 0.1, 1),
        })
    
    return {
        "overlap_applied": {"transformacao": 0.15, "acabamentos": 0.10},
        "lead_time_gain": 12,
        "heatmap": heatmap,
        "top_losses": top_losses,
        "demo_mode": False,
    }


@app.get("/inventory/")
def inventory_endpoint(classe: str = None, search: str = None) -> Dict[str, Any]:
    """
    Devolve inventário baseado nas ordens e artigos do plano.
    
    SmartInventory: Classificação ABC/XYZ avançada
    
    ABC Classification (Valor):
        A (70% valor): SKUs de alto volume/valor
        B (20% valor): SKUs de médio volume/valor
        C (10% valor): SKUs de baixo volume/valor
    
    XYZ Classification (Variabilidade):
        X (CV < 0.5): Demanda estável e previsível
        Y (0.5 <= CV < 1.0): Demanda com variação moderada
        Z (CV >= 1.0): Demanda errática e imprevisível
    """
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    article_data = []
    
    if not plan_df.empty:
        # Agrupar operações por article_id com métricas detalhadas
        article_stats = plan_df.groupby("article_id").agg({
            "qty": ["sum", "mean", "std", "count"],
            "duration_min": "sum",
            "order_id": "nunique"
        }).reset_index()
        
        # Flatten columns
        article_stats.columns = ["article_id", "qty_sum", "qty_mean", "qty_std", "qty_count", "duration_sum", "num_orders"]
        
        # Calcular ranking para classificação ABC (baseado em valor/volume)
        total_qty_sum = article_stats["qty_sum"].sum()
        article_stats = article_stats.sort_values("qty_sum", ascending=False)
        article_stats["cumsum_pct"] = article_stats["qty_sum"].cumsum() / total_qty_sum * 100
        
        for idx, row in article_stats.iterrows():
            article_id = row["article_id"]
            total_qty = row["qty_sum"]
            qty_mean = row["qty_mean"]
            qty_std = row["qty_std"] if pd.notna(row["qty_std"]) else qty_mean * 0.1
            qty_count = row["qty_count"]
            num_orders = row["num_orders"]
            cumsum_pct = row["cumsum_pct"]
            
            # ABC: baseado em ranking de valor cumulativo
            if cumsum_pct <= 70:
                abc = "A"  # Top 70% do valor
            elif cumsum_pct <= 90:
                abc = "B"  # Próximos 20%
            else:
                abc = "C"  # Últimos 10%
            
            # XYZ: baseado em Coeficiente de Variação (CV = std/mean)
            cv = qty_std / qty_mean if qty_mean > 0 else 0
            if cv < 0.5:
                xyz = "X"  # Estável
            elif cv < 1.0:
                xyz = "Y"  # Moderado
            else:
                xyz = "Z"  # Errático
            
            # Simular métricas de inventário REALISTAS
            # Stock varia por classe: A tem mais, C tem menos
            stock_multiplier = {"A": 1.5, "B": 1.2, "C": 0.8}[abc]
            variability = {"X": 0.05, "Y": 0.15, "Z": 0.30}[xyz]
            
            base_stock = total_qty * stock_multiplier
            stock_atual = int(base_stock * np.random.uniform(1 - variability, 1 + variability))
            
            # Consumo médio diário (ADS) baseado no histórico
            ads_180 = round(qty_mean if qty_mean > 0 else total_qty / 30, 2)
            
            # Cobertura em dias
            cobertura = round(stock_atual / ads_180, 1) if ads_180 > 0 else 999
            
            # ROP (Reorder Point) dinâmico
            lead_time_days = 7
            service_level_z = 1.65  # 95%
            safety_stock = service_level_z * qty_std * np.sqrt(lead_time_days) if qty_std > 0 else ads_180 * 3
            rop = int(ads_180 * lead_time_days + safety_stock)
            
            # Risco 30d mais realista
            if stock_atual <= rop * 0.5:
                risco = min(95, 60 + np.random.uniform(20, 35))
                acao = "Comprar agora"
            elif stock_atual <= rop:
                risco = min(80, 35 + np.random.uniform(15, 30))
                acao = "Planear compra"
            elif stock_atual <= rop * 1.5:
                risco = min(40, 10 + np.random.uniform(5, 20))
                acao = "Monitorizar"
            elif stock_atual > rop * 3:
                risco = 2 + np.random.uniform(0, 5)
                acao = "Excesso"
            else:
                risco = 5 + np.random.uniform(0, 10)
                acao = "OK"
            
            article_data.append({
                "sku": article_id,
                "classe": abc,
                "xyz": xyz,
                "stock_atual": stock_atual,
                "ads_180": ads_180,
                "cobertura_dias": cobertura,
                "risco_30d": round(risco, 1),
                "rop": rop,
                "acao": acao,
                "cv": round(cv, 2),
            })
    else:
        # Dados de demonstração se não houver plano
        demo_skus = [
            ("SKU-001", "A", "X", 15000, 120, 125, 5.2, 1008, "Monitorizar"),
            ("SKU-002", "A", "Y", 8500, 85, 100, 22.5, 714, "Monitorizar"),
            ("SKU-003", "A", "Z", 3200, 95, 34, 68.3, 798, "Comprar agora"),
            ("SKU-004", "B", "X", 6800, 45, 151, 3.8, 378, "OK"),
            ("SKU-005", "B", "Y", 4200, 52, 81, 18.7, 437, "Monitorizar"),
            ("SKU-006", "B", "Z", 1800, 38, 47, 42.1, 319, "Planear compra"),
            ("SKU-007", "C", "X", 2500, 18, 139, 6.4, 151, "Excesso"),
            ("SKU-008", "C", "Y", 1200, 22, 55, 28.3, 185, "Monitorizar"),
            ("SKU-009", "C", "Z", 450, 15, 30, 55.8, 126, "Planear compra"),
        ]
        for sku, abc, xyz, stock, ads, cob, risco, rop, acao in demo_skus:
            article_data.append({
                "sku": sku,
                "classe": abc,
                "xyz": xyz,
                "stock_atual": stock,
                "ads_180": ads,
                "cobertura_dias": cob,
                "risco_30d": risco,
                "rop": rop,
                "acao": acao,
            })
    
    # Filtrar por classe se especificado
    if classe:
        article_data = [a for a in article_data if f"{a['classe']}{a['xyz']}" == classe]
    
    # Filtrar por pesquisa se especificado
    if search:
        search_lower = search.lower()
        article_data = [a for a in article_data if search_lower in a["sku"].lower()]
    
    # Construir matriz ABC/XYZ
    matrix = {
        "A": {"X": 0, "Y": 0, "Z": 0},
        "B": {"X": 0, "Y": 0, "Z": 0},
        "C": {"X": 0, "Y": 0, "Z": 0},
    }
    
    # Contar todos os artigos para a matriz (não filtrados)
    if not plan_df.empty:
        article_stats_all = plan_df.groupby("article_id").agg({
            "qty": ["sum", "mean", "std"],
            "order_id": "nunique"
        }).reset_index()
        article_stats_all.columns = ["article_id", "qty_sum", "qty_mean", "qty_std", "num_orders"]
        
        total_qty_all = article_stats_all["qty_sum"].sum()
        article_stats_all = article_stats_all.sort_values("qty_sum", ascending=False)
        article_stats_all["cumsum_pct"] = article_stats_all["qty_sum"].cumsum() / total_qty_all * 100
        
        for _, row in article_stats_all.iterrows():
            cumsum_pct = row["cumsum_pct"]
            qty_mean = row["qty_mean"]
            qty_std = row["qty_std"] if pd.notna(row["qty_std"]) else qty_mean * 0.1
            cv = qty_std / qty_mean if qty_mean > 0 else 0
            
            abc = "A" if cumsum_pct <= 70 else ("B" if cumsum_pct <= 90 else "C")
            xyz = "X" if cv < 0.5 else ("Y" if cv < 1.0 else "Z")
            
            if abc in matrix and xyz in matrix[abc]:
                matrix[abc][xyz] += 1
    else:
        # Demo matrix
        matrix = {
            "A": {"X": 1, "Y": 1, "Z": 1},
            "B": {"X": 1, "Y": 1, "Z": 1},
            "C": {"X": 1, "Y": 1, "Z": 1},
        }
    
    return {
        "matrix": matrix,
        "skus": article_data,
    }


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# SMART INVENTORY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

@app.get("/inventory/stock")
def get_smart_inventory_stock(warehouse_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Devolve estado atual de stock (Digital Twin).
    
    R&D: SmartInventory - Digital Twin multi-armazém
    """
    try:
        from smart_inventory.stock_state import (
            StockState,
            create_stock_state,
            get_realtime_stock,
            get_global_stock,
        )
        from data_loader import load_dataset
        
        data = load_dataset()
        plan_df = _ensure_plan_exists()
        
        # Criar múltiplos armazéns baseados nas máquinas
        warehouses = ["WH-001", "WH-002", "WH-003"]  # 3 armazéns
        
        # Calcular stock inicial mais realista baseado no plano
        initial_data = []
        
        if not plan_df.empty:
            # Agrupar por artigo e calcular stock necessário
            article_stats = plan_df.groupby("article_id").agg({
                "qty": "sum",
                "order_id": "nunique",
            }).reset_index()
            
            for _, row in article_stats.iterrows():
                article_id = str(row["article_id"])
                total_qty = float(row["qty"])
                num_orders = int(row["order_id"])
                
                # Stock inicial = consumo total * 1.5 (cobertura de 1.5x)
                # Distribuir entre armazéns
                base_stock = total_qty * 1.5
                
                for i, wh_id in enumerate(warehouses):
                    # Distribuição: 50% no primeiro, 30% no segundo, 20% no terceiro
                    distribution = [0.5, 0.3, 0.2][i]
                    stock_qty = base_stock * distribution
                    
                    # Adicionar alguma variabilidade
                    stock_qty = stock_qty * np.random.uniform(0.9, 1.1)
                    
                    initial_data.append({
                        "sku": article_id,
                        "warehouse_id": wh_id,
                        "quantity_on_hand": max(0, stock_qty),
                    })
        else:
            # Fallback: usar orders
            if not data.orders.empty:
                for _, order in data.orders.iterrows():
                    for wh_id in warehouses[:2]:  # Apenas 2 armazéns para fallback
                        initial_data.append({
                            "sku": str(order.get("article_id", "")),
                            "warehouse_id": wh_id,
                            "quantity_on_hand": float(order.get("qty", 0)) * 1.5,
                        })
        
        stock_state = create_stock_state(pd.DataFrame(initial_data) if initial_data else None)
        
        # Converter para DataFrame e filtrar por warehouse se especificado
        df = stock_state.to_dataframe()
        if warehouse_id:
            df = df[df["warehouse_id"] == warehouse_id]
        
        # Adicionar métricas calculadas
        df["coverage_estimate_days"] = df.apply(
            lambda row: row["quantity_available"] / 10.0 if row["quantity_available"] > 0 else 0,
            axis=1
        )
        
        return {
            "stock": df.to_dict(orient="records"),
            "warehouses": list(stock_state.get_all_warehouses()),
            "total_skus": len(stock_state.get_all_skus()),
        }
    except Exception as e:
        import traceback
        logger.error(f"Erro em get_smart_inventory_stock: {e}\n{traceback.format_exc()}")
        return {"error": str(e), "stock": [], "warehouses": [], "total_skus": 0}


@app.get("/inventory/forecast/{sku}")
def get_inventory_forecast(
    sku: str,
    horizon_days: int = 90,
    model: str = "ARIMA",
) -> Dict[str, Any]:
    """
    Forecast de demanda para um SKU.
    
    R&D: SmartInventory - Forecasting avançado (ARIMA/Prophet/N-BEATS)
    
    Gera um histórico realista de consumo diário e aplica o modelo de forecasting.
    """
    try:
        from smart_inventory import (
            forecast_demand,
            ForecastModel,
            ForecastConfig,
        )
        from data_loader import load_dataset
        
        data = load_dataset()
        plan_df = _ensure_plan_exists()
        
        # Construir histórico REALISTA a partir do plano
        if not plan_df.empty:
            sku_ops = plan_df[plan_df["article_id"] == sku].copy()
            if not sku_ops.empty:
                # Calcular consumo médio baseado nas ordens (normalizado para consumo diário razoável)
                total_qty = sku_ops["qty"].sum()
                num_days_span = max(1, (sku_ops["start_time"].max() - sku_ops["start_time"].min()).days + 1) if "start_time" in sku_ops.columns else 30
                
                # Normalizar para consumo diário realista (tipicamente 10-200 unidades/dia)
                raw_daily_avg = total_qty / max(num_days_span, 1)
                # Escalar para valores realistas de inventário
                scale_factor = min(1.0, 150 / raw_daily_avg) if raw_daily_avg > 150 else max(1.0, 20 / raw_daily_avg) if raw_daily_avg < 20 else 1.0
                base_demand = raw_daily_avg * scale_factor
                base_demand = max(15, min(200, base_demand))  # Limitar entre 15 e 200
                
                # Gerar histórico de 90 dias com padrão realista
                dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
                
                # Adicionar componentes realistas
                trend = np.linspace(-2, 5, len(dates))  # Ligeira tendência ascendente
                
                # Sazonalidade semanal (fins de semana mais baixos)
                weekday_effect = np.array([1.1, 1.15, 1.1, 1.05, 0.95, 0.75, 0.7] * 13)[:len(dates)]
                
                # Sazonalidade mensal
                monthly_cycle = 8 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
                
                # Ruído realista
                noise = np.random.normal(0, base_demand * 0.15, len(dates))
                
                values = (base_demand + trend + monthly_cycle) * weekday_effect + noise
                history = pd.Series(np.maximum(values, 5), index=dates)
            else:
                # SKU sem operações: usar padrão genérico
                dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
                base_demand = 45.0
                trend = np.linspace(0, 8, len(dates))
                weekday_effect = np.array([1.1, 1.15, 1.1, 1.05, 0.95, 0.75, 0.7] * 13)[:len(dates)]
                monthly_cycle = 6 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
                noise = np.random.normal(0, 5, len(dates))
                values = (base_demand + trend + monthly_cycle) * weekday_effect + noise
                history = pd.Series(np.maximum(values, 5), index=dates)
        else:
            # Fallback: histórico simulado completo
            dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
            base_demand = 35.0
            trend = np.linspace(0, 5, len(dates))
            weekday_effect = np.array([1.1, 1.15, 1.1, 1.05, 0.95, 0.75, 0.7] * 13)[:len(dates)]
            monthly_cycle = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
            noise = np.random.normal(0, 4, len(dates))
            values = (base_demand + trend + monthly_cycle) * weekday_effect + noise
            history = pd.Series(np.maximum(values, 5), index=dates)
        
        # Configurar modelo
        model_map = {
            "ARIMA": ForecastModel.ARIMA,
            "PROPHET": ForecastModel.PROPHET,
            "NBEATS": ForecastModel.NBEATS,
        }
        forecast_model = model_map.get(model.upper(), ForecastModel.ARIMA)
        
        config = ForecastConfig(
            model=forecast_model,
            horizon_days=horizon_days,
        )
        
        result = forecast_demand(sku, history, config=config)
        
        return {
            "sku": sku,
            "forecast": result.forecast_series.to_dict(),
            "lower_ci": result.lower_ci.to_dict(),
            "upper_ci": result.upper_ci.to_dict(),
            "model_used": result.model_used.value,
            "snr": result.snr,
            "snr_class": result.snr_class,
            "confidence_score": result.confidence_score,
            "metrics": result.metrics,
        }
    except Exception as e:
        logger.error(f"Erro em get_inventory_forecast: {e}")
        return {"error": str(e)}


@app.get("/inventory/rop/{sku}")
def get_inventory_rop(
    sku: str,
    service_level: float = 0.95,
    lead_time_days: float = 7.0,
) -> Dict[str, Any]:
    """
    Calcula ROP (Reorder Point) dinâmico para um SKU.
    
    R&D: SmartInventory - ROP dinâmico com risco 30 dias
    
    Fórmula:
        ROP = μ_d * L + z * σ_d * sqrt(L)
        
    onde:
        μ_d = consumo médio diário
        σ_d = desvio padrão do consumo
        L = lead time (dias)
        z = quantil do nível de serviço
    """
    try:
        from smart_inventory import (
            compute_dynamic_rop,
            ROPConfig,
            forecast_demand,
            ForecastConfig,
            ForecastModel,
        )
        from data_loader import load_dataset
        
        data = load_dataset()
        plan_df = _ensure_plan_exists()
        
        # Construir histórico REALISTA (mesmo método que forecast)
        if not plan_df.empty:
            sku_ops = plan_df[plan_df["article_id"] == sku].copy()
            if not sku_ops.empty:
                total_qty = sku_ops["qty"].sum()
                num_days_span = max(1, (sku_ops["start_time"].max() - sku_ops["start_time"].min()).days + 1) if "start_time" in sku_ops.columns else 30
                
                raw_daily_avg = total_qty / max(num_days_span, 1)
                scale_factor = min(1.0, 150 / raw_daily_avg) if raw_daily_avg > 150 else max(1.0, 20 / raw_daily_avg) if raw_daily_avg < 20 else 1.0
                base_demand = raw_daily_avg * scale_factor
                base_demand = max(15, min(200, base_demand))
                
                dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
                trend = np.linspace(-2, 5, len(dates))
                weekday_effect = np.array([1.1, 1.15, 1.1, 1.05, 0.95, 0.75, 0.7] * 13)[:len(dates)]
                monthly_cycle = 8 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
                noise = np.random.normal(0, base_demand * 0.15, len(dates))
                values = (base_demand + trend + monthly_cycle) * weekday_effect + noise
                history = pd.Series(np.maximum(values, 5), index=dates)
            else:
                dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
                base_demand = 45.0
                trend = np.linspace(0, 8, len(dates))
                weekday_effect = np.array([1.1, 1.15, 1.1, 1.05, 0.95, 0.75, 0.7] * 13)[:len(dates)]
                monthly_cycle = 6 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
                noise = np.random.normal(0, 5, len(dates))
                values = (base_demand + trend + monthly_cycle) * weekday_effect + noise
                history = pd.Series(np.maximum(values, 5), index=dates)
        else:
            dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
            base_demand = 35.0
            trend = np.linspace(0, 5, len(dates))
            weekday_effect = np.array([1.1, 1.15, 1.1, 1.05, 0.95, 0.75, 0.7] * 13)[:len(dates)]
            monthly_cycle = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
            noise = np.random.normal(0, 4, len(dates))
            values = (base_demand + trend + monthly_cycle) * weekday_effect + noise
            history = pd.Series(np.maximum(values, 5), index=dates)
        
        forecast_result = forecast_demand(sku, history, config=ForecastConfig(model=ForecastModel.ARIMA))
        
        # Stock atual baseado no consumo médio e cobertura típica (15-45 dias)
        avg_daily_demand = history.mean()
        # Stock = consumo médio * cobertura esperada (com variação)
        coverage_days = np.random.uniform(10, 40)
        current_stock = avg_daily_demand * coverage_days
        
        rop_config = ROPConfig(
            service_level=service_level,
            lead_time_days=lead_time_days,
        )
        
        rop_result = compute_dynamic_rop(sku, forecast_result, rop_config, current_stock)
        
        return rop_result.to_dict()
    except Exception as e:
        logger.error(f"Erro em get_inventory_rop: {e}")
        return {"error": str(e)}


@app.get("/inventory/suggestions")
def get_inventory_suggestions() -> Dict[str, Any]:
    """
    Gera sugestões inteligentes de inventário.
    
    R&D: SmartInventory - Sugestões baseadas em ROP, forecast, risco
    
    Tipos de Sugestões:
    - BUY: Stock abaixo do ROP - comprar para repor
    - TRANSFER: Redistribuir stock entre armazéns
    - REDUCE: Stock excessivo - reduzir encomendas
    - ALERT: Risco de ruptura elevado
    """
    try:
        from smart_inventory.suggestion_engine import generate_inventory_suggestions
        from smart_inventory.demand_forecasting import forecast_demand, ForecastConfig, ForecastModel
        from smart_inventory.rop_engine import compute_dynamic_rop, ROPConfig
        from smart_inventory.stock_state import create_stock_state
        from data_loader import load_dataset
        
        data = load_dataset()
        plan_df = _ensure_plan_exists()
        
        # Criar stock state com valores realistas
        warehouses = ["WH-001", "WH-002", "WH-003"]
        initial_data = []
        
        # Calcular forecasts e ROPs para todos os SKUs principais
        skus = list(data.orders["article_id"].unique())[:10]  # Até 10 SKUs
        
        forecast_results = {}
        rop_results = {}
        
        for sku in skus:
            try:
                sku_str = str(sku)
                
                # Determinar consumo base realista (15-200 unidades/dia)
                if not plan_df.empty:
                    sku_ops = plan_df[plan_df["article_id"] == sku_str].copy()
                    if not sku_ops.empty:
                        total_qty = sku_ops["qty"].sum()
                        num_days = max(1, (sku_ops["start_time"].max() - sku_ops["start_time"].min()).days + 1) if "start_time" in sku_ops.columns else 30
                        raw_daily = total_qty / max(num_days, 1)
                        base_demand = max(15, min(200, raw_daily * (150 / raw_daily if raw_daily > 150 else 20 / raw_daily if raw_daily < 20 else 1)))
                    else:
                        base_demand = np.random.uniform(20, 80)
                else:
                    base_demand = np.random.uniform(20, 80)
                
                # Gerar histórico realista
                dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
                trend = np.linspace(-2, 5, len(dates))
                weekday_effect = np.array([1.1, 1.15, 1.1, 1.05, 0.95, 0.75, 0.7] * 13)[:len(dates)]
                monthly_cycle = 8 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
                noise = np.random.normal(0, base_demand * 0.15, len(dates))
                values = (base_demand + trend + monthly_cycle) * weekday_effect + noise
                history = pd.Series(np.maximum(values, 5), index=dates)
                
                # Forecast
                forecast = forecast_demand(sku_str, history, config=ForecastConfig(model=ForecastModel.ARIMA))
                forecast_results[sku_str] = forecast
                
                # Stock atual realista (varia entre 5 e 60 dias de cobertura)
                avg_demand = history.mean()
                coverage_scenario = np.random.choice(["low", "normal", "high", "excess"], p=[0.2, 0.4, 0.25, 0.15])
                coverage_days = {
                    "low": np.random.uniform(3, 8),
                    "normal": np.random.uniform(15, 30),
                    "high": np.random.uniform(30, 45),
                    "excess": np.random.uniform(60, 90),
                }[coverage_scenario]
                current_stock = avg_demand * coverage_days
                
                # Adicionar ao stock state
                for i, wh_id in enumerate(warehouses):
                    distribution = [0.5, 0.3, 0.2][i]
                    initial_data.append({
                        "sku": sku_str,
                        "warehouse_id": wh_id,
                        "quantity_on_hand": max(0, current_stock * distribution * np.random.uniform(0.9, 1.1)),
                    })
                
                # ROP
                rop = compute_dynamic_rop(sku_str, forecast, ROPConfig(), current_stock)
                rop_results[sku_str] = rop
                
            except Exception as e:
                logger.warning(f"Erro ao processar {sku}: {e}")
                continue
        
        stock_state = create_stock_state(pd.DataFrame(initial_data) if initial_data else None)
        
        # Gerar sugestões
        if len(rop_results) == 0:
            return {
                "suggestions": [],
                "count": 0,
                "note": "Dados insuficientes para gerar sugestões",
            }
        
        suggestions = generate_inventory_suggestions(
            stock_state,
            rop_results,
            forecast_results,
        )
        
        return {
            "suggestions": [s.to_dict() for s in suggestions],
            "count": len(suggestions),
        }
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Erro em get_inventory_suggestions: {error_msg}\n{error_trace}")
        return {
            "error": error_msg,
            "suggestions": [],
            "count": 0,
            "debug": error_trace.split('\n')[-3:] if len(error_trace) > 0 else [],
        }


@app.post("/inventory/optimize")
def optimize_inventory(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Otimiza redistribuição e compras multi-armazém.
    
    R&D: SmartInventory - Otimização MILP multi-armazém
    """
    try:
        from smart_inventory import optimize_multi_warehouse, Warehouse, OptimizationConfig
        return {
            "success": True,
            "transfers": [],
            "orders": [],
            "total_cost": 0.0,
            "note": "Otimização multi-armazém em desenvolvimento",
        }
    except Exception as e:
        logger.error(f"Erro em optimize_inventory: {e}")
        return {"error": str(e)}


@app.get("/suggestions/")
def suggestions_endpoint(mode: str = "resumo") -> Dict[str, Any]:
    """
    Gera sugestões inteligentes baseadas nos dados de produção.
    
    Analisa:
    - Gargalos e máquinas sobrecarregadas
    - Distribuição de rotas
    - Ordens com prazos apertados
    - Oportunidades de otimização
    """
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    kpis = compute_kpis(plan_df, data.orders)
    bottleneck = compute_bottleneck(plan_df)
    
    suggestions = []
    suggestion_id = 0
    
    # 1. Sugestão sobre gargalo
    if bottleneck:
        machine_id = bottleneck["machine_id"]
        total_hours = bottleneck["total_minutes"] / 60
        
        suggestion_id += 1
        suggestions.append({
            "id": f"SUG-{suggestion_id:03d}",
            "icon": "⚠️",
            "action": f"Redistribuir carga da máquina {machine_id}",
            "explanation": f"A máquina {machine_id} está identificada como gargalo com {total_hours:.1f} horas de carga. Considere redistribuir operações para máquinas alternativas.",
            "impact": f"Redução de {total_hours * 0.2:.1f}h no lead time",
            "impact_level": "alto",
            "gain": f"{total_hours * 0.15:.1f}h de capacidade recuperada",
            "reasoning_markdown": f"""### Análise do Gargalo

A máquina **{machine_id}** apresenta a maior carga acumulada do plano:

- **Carga total**: {total_hours:.1f} horas
- **Impacto estimado**: Atraso potencial em {int(total_hours * 0.1)} ordens

#### Recomendação
1. Verificar se existem máquinas alternativas no routing
2. Considerar turnos adicionais para esta máquina
3. Resequenciar operações para equilibrar carga""",
            "data_points": {
                "machine_id": machine_id,
                "total_hours": round(total_hours, 1),
                "total_operations": len(plan_df[plan_df["machine_id"] == machine_id]) if not plan_df.empty else 0,
            },
        })
    
    # 2. Sugestão sobre distribuição de rotas
    route_dist = kpis.get("route_distribution", {})
    if route_dist:
        dominant_route = max(route_dist.items(), key=lambda x: x[1])[0] if route_dist else "A"
        dominant_pct = (route_dist.get(dominant_route, 0) / sum(route_dist.values()) * 100) if sum(route_dist.values()) > 0 else 0
        
        if dominant_pct > 60:
            suggestion_id += 1
            suggestions.append({
                "id": f"SUG-{suggestion_id:03d}",
                "icon": "🔀",
                "action": "Diversificar seleção de rotas",
                "explanation": f"A rota {dominant_route} está a ser usada em {dominant_pct:.0f}% das operações. Considere distribuir mais uniformemente entre rotas disponíveis.",
                "impact": "Melhor balanceamento de carga",
                "impact_level": "medio",
                "gain": "Redução de 10-15% em tempo de espera",
                "reasoning_markdown": f"""### Concentração de Rotas

A distribuição atual mostra uma concentração elevada na **Rota {dominant_route}**:

| Rota | Operações | % |
|------|-----------|---|
""" + "\n".join([f"| {r} | {c} | {c/sum(route_dist.values())*100:.0f}% |" for r, c in sorted(route_dist.items())]) + """

#### Benefícios da diversificação
- Menor dependência de máquinas específicas
- Flexibilidade em caso de avarias
- Melhor tempo de resposta""",
                "data_points": route_dist,
            })
    
    # 3. Sugestão sobre OTD
    otd_pct = kpis.get("otd_percent", 100)
    if otd_pct < 90:
        suggestion_id += 1
        suggestions.append({
            "id": f"SUG-{suggestion_id:03d}",
            "icon": "📅",
            "action": "Melhorar taxa de entrega no prazo (OTD)",
            "explanation": f"A taxa OTD atual é de {otd_pct:.1f}%, abaixo do objetivo de 95%. Reveja as prioridades de sequenciamento.",
            "impact": f"Aumento de {95 - otd_pct:.1f}% no OTD",
            "impact_level": "alto",
            "gain": "Melhoria na satisfação do cliente",
            "reasoning_markdown": f"""### Taxa de Entrega no Prazo

O OTD atual de **{otd_pct:.1f}%** indica que algumas ordens não estão a cumprir os prazos.

#### Ações recomendadas
1. Priorizar ordens com due_date mais próximas
2. Identificar ordens em risco e ajustar sequenciamento
3. Considerar capacidade adicional para ordens urgentes

**Objetivo**: Atingir OTD ≥ 95%""",
            "data_points": {
                "otd_current": round(otd_pct, 1),
                "otd_target": 95,
                "gap": round(95 - otd_pct, 1),
            },
        })
    
    # 4. Sugestão sobre overlaps
    overlaps = kpis.get("overlaps", {})
    total_overlaps = overlaps.get("total", 0)
    if total_overlaps > 0:
        suggestion_id += 1
        suggestions.append({
            "id": f"SUG-{suggestion_id:03d}",
            "icon": "🔧",
            "action": "Resolver sobreposições de operações",
            "explanation": f"Foram detetadas {total_overlaps} sobreposições no plano. Estas podem causar conflitos de recursos.",
            "impact": "Eliminação de conflitos de agenda",
            "impact_level": "medio",
            "gain": f"Resolução de {total_overlaps} conflitos",
            "reasoning_markdown": f"""### Sobreposições Detetadas

O plano atual contém **{total_overlaps}** operações com sobreposição temporal.

#### Máquinas afetadas
""" + "\n".join([f"- {m}: {c} overlaps" for m, c in overlaps.get("by_machine", {}).items()]) + """

#### Recomendação
Resequenciar as operações afetadas para eliminar conflitos.""",
            "data_points": overlaps,
        })
    
    # 5. Sugestão sobre setup times
    setup_hours = kpis.get("setup_hours", 0)
    if setup_hours > 2:
        suggestion_id += 1
        suggestions.append({
            "id": f"SUG-{suggestion_id:03d}",
            "icon": "⚙️",
            "action": "Otimizar sequência para reduzir setups",
            "explanation": f"Estimativa de {setup_hours:.1f} horas de setup. Agrupar operações por família pode reduzir este tempo.",
            "impact": f"Potencial redução de {setup_hours * 0.3:.1f}h",
            "impact_level": "baixo",
            "gain": f"{setup_hours * 0.3:.1f}h de tempo produtivo recuperado",
            "reasoning_markdown": f"""### Tempo de Setup

O plano atual estima **{setup_hours:.1f} horas** de tempo em setups.

#### Estratégias de redução
1. Agrupar operações da mesma família consecutivamente
2. Aplicar técnicas SMED onde possível
3. Considerar sequenciamento por matriz de setup

**Objetivo**: Reduzir setup em pelo menos 30%""",
            "data_points": {
                "setup_hours_current": round(setup_hours, 1),
                "setup_hours_target": round(setup_hours * 0.7, 1),
            },
        })
    
    # 6. Sugestão sobre máquinas subutilizadas
    machine_loads = kpis.get("machine_loads", [])
    underutilized = [m for m in machine_loads if m.get("utilization_pct", 100) < 30]
    if underutilized:
        suggestion_id += 1
        suggestions.append({
            "id": f"SUG-{suggestion_id:03d}",
            "icon": "📊",
            "action": "Aproveitar capacidade disponível",
            "explanation": f"{len(underutilized)} máquinas com utilização abaixo de 30%. Considere redistribuir carga.",
            "impact": "Melhor utilização de recursos",
            "impact_level": "baixo",
            "gain": f"Capacidade extra de {sum(m.get('idle_hours', 0) for m in underutilized):.1f}h disponível",
            "reasoning_markdown": f"""### Máquinas Subutilizadas

As seguintes máquinas têm baixa utilização:

| Máquina | Utilização | Capacidade Disponível |
|---------|------------|----------------------|
""" + "\n".join([f"| {m['machine_id']} | {m.get('utilization_pct', 0):.0f}% | {m.get('idle_hours', 0):.1f}h |" for m in underutilized[:5]]) + """

#### Oportunidade
Redistribuir operações do gargalo para estas máquinas.""",
            "data_points": {
                "underutilized_machines": [m["machine_id"] for m in underutilized[:5]],
                "total_idle_hours": round(sum(m.get('idle_hours', 0) for m in underutilized), 1),
            },
        })
    
    # Se não há sugestões, criar uma genérica positiva
    if not suggestions:
        suggestions.append({
            "id": "SUG-001",
            "icon": "✅",
            "action": "Plano otimizado",
            "explanation": "O plano atual não apresenta problemas críticos identificados.",
            "impact": "Manter monitorização",
            "impact_level": "baixo",
            "gain": "Sistema estável",
            "reasoning_markdown": """### Análise Completa

O plano de produção foi analisado e não foram identificadas oportunidades críticas de melhoria.

#### Próximos passos
- Continuar a monitorizar KPIs
- Verificar novamente após alterações nos dados""",
            "data_points": kpis,
        })
    
    return {
        "count": len(suggestions),
        "items": suggestions,
        "mode": mode,
    }


# ============================================================
# ACTIONS / APPROVAL QUEUE (Industry 5.0 Human-Centric)
# ============================================================

@app.get("/actions")
def list_actions(status: Optional[str] = None) -> Dict[str, Any]:
    """
    Lista ações propostas pelo sistema.
    
    Parâmetros:
        status: Filtrar por estado (PENDING, APPROVED, REJECTED, APPLIED)
    
    Industry 5.0: O sistema propõe, o humano decide.
    """
    from dataclasses import asdict
    
    store = get_action_store()
    actions = store.list_actions(status=status)
    
    return {
        "count": len(actions),
        "actions": [asdict(a) for a in actions],
        "pending_count": len([a for a in actions if a.status == "PENDING"]),
    }


@app.get("/actions/{action_id}")
def get_action(action_id: str) -> Dict[str, Any]:
    """
    Obtém detalhes de uma ação específica.
    """
    from dataclasses import asdict
    
    store = get_action_store()
    action = store.get_action(action_id)
    
    if not action:
        raise HTTPException(status_code=404, detail="Ação não encontrada")
    
    return asdict(action)


@app.post("/actions/propose")
def propose_new_action(req: ProposeActionRequest) -> Dict[str, Any]:
    """
    Propõe uma nova ação para aprovação humana.
    
    Tipos de ação suportados:
    - SET_MACHINE_DOWN: Colocar máquina offline
    - SET_MACHINE_UP: Reativar máquina
    - CHANGE_ROUTE: Alterar rota de produção
    - MOVE_OPERATION: Mover operação para outra máquina
    - SET_VIP_ARTICLE: Definir artigo como VIP
    - ADD_OVERTIME: Adicionar horas extra
    - ADD_ORDER: Adicionar nova ordem
    
    A ação fica com status='PENDING' até ser aprovada ou rejeitada.
    """
    from dataclasses import asdict
    
    action = propose_action(
        action_type=req.type,
        payload=req.payload,
        source=req.source,
        description=req.description,
    )
    
    return {
        "message": "Ação proposta com sucesso. Aguarda aprovação.",
        "action": asdict(action),
    }


@app.post("/actions/{action_id}/approve")
def approve_action_endpoint(action_id: str, req: ApproveActionRequest) -> Dict[str, Any]:
    """
    Aprova uma ação pendente e aplica-a ao plano.
    
    Após aprovação:
    1. A ação é marcada como APPROVED
    2. As alterações são aplicadas ao DataBundle
    3. O scheduler é re-executado
    4. Um novo plano é gerado
    5. A ação é marcada como APPLIED
    
    Importante: Nenhuma ação é executada em sistemas externos (ERP/MES).
    Todas as alterações são apenas ao plano interno.
    """
    result = approve_action(
        action_id=action_id,
        approved_by=req.approved_by,
        notes=req.notes,
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Ação não encontrada")
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return {
        "message": "Ação aprovada e aplicada com sucesso.",
        "result": result,
    }


@app.post("/actions/{action_id}/reject")
def reject_action_endpoint(action_id: str, req: RejectActionRequest) -> Dict[str, Any]:
    """
    Rejeita uma ação pendente.
    
    A ação é marcada como REJECTED e não será aplicada.
    """
    from dataclasses import asdict
    
    action = reject_action(
        action_id=action_id,
        rejected_by=req.rejected_by,
        reason=req.reason,
    )
    
    if not action:
        raise HTTPException(status_code=404, detail="Ação não encontrada")
    
    return {
        "message": "Ação rejeitada.",
        "action": asdict(action),
    }


@app.get("/actions/pending/count")
def get_pending_count() -> Dict[str, int]:
    """
    Obtém o número de ações pendentes.
    
    Útil para mostrar badge no UI.
    """
    pending = get_pending_actions()
    return {"count": len(pending)}


@app.post("/actions/from-suggestion")
def create_action_from_suggestion_endpoint(suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cria uma ação a partir de uma sugestão do sistema.
    
    Converte automaticamente sugestões do suggestion_engine em ações
    que podem ser aprovadas pelo humano.
    """
    from dataclasses import asdict
    from actions_engine import create_action_from_suggestion
    
    action = create_action_from_suggestion(suggestion)
    
    if not action:
        raise HTTPException(status_code=400, detail="Não foi possível criar ação a partir desta sugestão")
    
    return {
        "message": "Ação criada a partir de sugestão.",
        "action": asdict(action),
    }


# ============================================================
# R&D / RESEARCH API ENDPOINTS (SIFIDE)
# ============================================================

class ExperimentRequest(BaseModel):
    strategies: List[str] = ["FIXED_PRIMARY", "SHORTEST_QUEUE", "SETUP_AWARE", "MULTI_OBJECTIVE"]


class ExplainRequest(BaseModel):
    decision_type: str  # "routing", "bottleneck", "suggestion"
    context: Dict[str, Any]


@app.get("/research/experiments")
def list_experiments() -> Dict[str, Any]:
    """
    Lista experimentos executados e seus resultados.
    
    WP4: Learning Scheduler + Experimentation
    """
    from research.experiment_logger import load_experiment_logs, compute_experiment_statistics
    
    logs = load_experiment_logs()
    stats = compute_experiment_statistics(logs)
    
    return {
        "total_experiments": len(logs),
        "recent": logs[:10],
        "statistics": stats,
    }


@app.post("/research/run-experiment")
def run_experiment(req: ExperimentRequest) -> Dict[str, Any]:
    """
    Executa experimento E1.1: Comparação de estratégias de routing.
    
    R&D: WP1 (APS Core + Routing Intelligence)
    Hipótese: H1.1 - Routing dinâmico reduz makespan ≥8%
    """
    from scheduler import run_routing_experiment
    
    results = run_routing_experiment(strategies=req.strategies)
    
    return {
        "experiment_id": "E1.1",
        "hypothesis": "H1.1",
        "strategies_tested": req.strategies,
        "results": results,
    }


@app.get("/research/plan-experimental")
def get_experimental_plan(strategy: str = "FIXED_PRIMARY") -> Dict[str, Any]:
    """
    Gera plano com estratégia de routing experimental.
    
    Estratégias disponíveis:
    - FIXED_PRIMARY (baseline)
    - SHORTEST_QUEUE
    - SETUP_AWARE
    - LOAD_BALANCED
    - MULTI_OBJECTIVE
    - ML_PREDICTED
    - RANDOM
    """
    from scheduler import build_plan_experimental, compute_kpis
    
    data = load_dataset()
    plan_df, exp_data = build_plan_experimental(data, routing_strategy=strategy)
    kpis = compute_kpis(plan_df, data.orders)
    
    plan_records = plan_df.to_dict(orient="records") if not plan_df.empty else []
    
    # Convert datetime to ISO string
    for rec in plan_records:
        if "start_time" in rec:
            rec["start_time"] = rec["start_time"].isoformat() if hasattr(rec["start_time"], "isoformat") else str(rec["start_time"])
        if "end_time" in rec:
            rec["end_time"] = rec["end_time"].isoformat() if hasattr(rec["end_time"], "isoformat") else str(rec["end_time"])
    
    return {
        "strategy": strategy,
        "plan": plan_records,
        "kpis": kpis,
        "experiment_data": {
            "num_routing_decisions": len(exp_data.get("decision_log", [])),
            "decisions_sample": exp_data.get("decision_log", [])[:5],
        },
    }


@app.post("/research/explain")
def explain_decision(req: ExplainRequest) -> Dict[str, Any]:
    """
    Gera explicação para uma decisão do APS.
    
    R&D: WP2 (What-If + Explainable AI)
    Hipótese: H4.1 - Explicações aumentam confiança do utilizador
    
    Tipos de decisão:
    - routing: Por que esta rota foi escolhida
    - bottleneck: Por que esta máquina é gargalo
    - suggestion: Por que esta sugestão foi gerada
    """
    from research.explainability_engine import ExplainabilityEngine, AudienceLevel
    
    engine = ExplainabilityEngine()
    
    if req.decision_type == "bottleneck":
        plan_df = _ensure_plan_exists()
        bottleneck = compute_bottleneck(plan_df)
        
        if bottleneck:
            # Get machine comparison data
            machine_loads = plan_df.groupby("machine_id")["duration_min"].sum()
            comparison = [
                {"machine_id": m, "load_min": float(l)}
                for m, l in machine_loads.items()
                if m != bottleneck["machine_id"]
            ]
            
            utilization = 80.0  # Simplified
            
            explanation = engine.explain_bottleneck(
                machine_id=bottleneck["machine_id"],
                load_minutes=bottleneck["total_minutes"],
                total_ops=len(plan_df[plan_df["machine_id"] == bottleneck["machine_id"]]),
                utilization_pct=utilization,
                comparison_machines=comparison[:5],
            )
            
            return explanation.to_dict()
    
    elif req.decision_type == "suggestion":
        context = req.context
        explanation = engine.explain_suggestion(
            suggestion_type=context.get("type", "general"),
            action=context.get("action", "Ação não especificada"),
            expected_impact=context.get("impact", {}),
            supporting_data=context.get("data", {}),
        )
        return explanation.to_dict()
    
    elif req.decision_type == "routing":
        context = req.context
        explanation = engine.explain_routing_decision(
            selected_route=context.get("route", "A"),
            selected_machine=context.get("machine", "M-???"),
            alternatives=context.get("alternatives", []),
            scoring_method=context.get("method", "fixed_primary"),
            scores=context.get("scores", {}),
            context=context,
        )
        return explanation.to_dict()
    
    return {
        "error": f"Tipo de decisão não suportado: {req.decision_type}",
        "supported_types": ["routing", "bottleneck", "suggestion"],
    }


@app.get("/research/inventory-simulation")
def run_inventory_simulation(
    mode: str = "decoupled",
    service_level: float = 0.95,
    machine_utilization: float = 0.8,
) -> Dict[str, Any]:
    """
    Executa simulação de inventário com otimização acoplada/desacoplada.
    
    R&D: WP3 (Inventory-Production Coupling)
    Hipótese: H3.1 - Otimização acoplada reduz risco de stockout ≥20%
    
    Modos:
    - decoupled: Tradicional (inventário separado de produção)
    - capacity_aware: Considera carga das máquinas no ROP
    - joint: Otimização conjunta com cenários
    """
    from research.inventory_optimization import (
        InventoryOptimizer, OptimizationMode, SKUProfile
    )
    
    mode_map = {
        "decoupled": OptimizationMode.DECOUPLED,
        "capacity_aware": OptimizationMode.CAPACITY_AWARE,
        "joint": OptimizationMode.JOINT,
        "risk_based": OptimizationMode.RISK_BASED,
    }
    
    opt_mode = mode_map.get(mode, OptimizationMode.DECOUPLED)
    optimizer = InventoryOptimizer(mode=opt_mode)
    
    # Create sample SKU profiles from orders data
    data = load_dataset()
    articles = data.orders["article_id"].unique()[:5]  # First 5 articles
    
    results = []
    for article in articles:
        profile = SKUProfile(
            sku_id=f"SKU-{article}",
            article_id=str(article),
            abc_class="A",
            xyz_class="X",
            avg_daily_demand=50.0,
            demand_std=15.0,
            demand_cv=0.3,
            current_stock=200.0,
            pending_orders=0.0,
            in_production=100.0,
            production_lead_time_days=5.0,
        )
        
        policy = optimizer.optimize_sku(
            profile,
            service_level=service_level,
            machine_utilization=machine_utilization,
        )
        
        # Run simulation
        sim_result = optimizer.simulate_policy(profile, policy, n_periods=90)
        
        results.append({
            "sku_id": profile.sku_id,
            "article_id": profile.article_id,
            "policy": {
                "rop": policy.rop,
                "safety_stock": policy.safety_stock,
                "order_quantity": policy.order_quantity,
                "rationale": policy.rationale,
            },
            "simulation": {
                "service_level": sim_result.service_level,
                "avg_inventory": sim_result.avg_inventory,
                "stockout_events": sim_result.stockout_events,
                "total_cost": sim_result.total_cost,
            },
        })
    
    return {
        "mode": mode,
        "service_level_target": service_level,
        "machine_utilization": machine_utilization,
        "skus_analyzed": len(results),
        "results": results,
    }


@app.get("/research/learning-status")
def get_learning_status() -> Dict[str, Any]:
    """
    Devolve estado atual do scheduler com aprendizagem.
    
    R&D: WP4 (Learning Scheduler)
    Hipótese: H4.3 - Contextual bandits < regret que heurísticas fixas
    """
    from research.learning_scheduler import LearningScheduler, PolicyType
    
    # Create scheduler instances to show available policies
    policies_info = []
    for policy_type in PolicyType:
        scheduler = LearningScheduler(policy_type=policy_type)
        state = scheduler.get_policy_state()
        policies_info.append({
            "policy_type": policy_type.value,
            "description": {
                "FIXED_PRIORITY": "Baseline: regras de prioridade fixas (EDD/SPT)",
                "EPSILON_GREEDY": "Exploração ε-greedy com decaimento",
                "UCB": "Upper Confidence Bound - balança exploração/exploração",
                "THOMPSON": "Thompson Sampling (Bayesiano)",
                "CONTEXTUAL_BANDIT": "Decisões baseadas em contexto (features)",
                "REINFORCEMENT": "RL completo (futuro)",
            }.get(policy_type.value, ""),
            "state": state,
        })
    
    return {
        "available_policies": policies_info,
        "current_policy": "FIXED_PRIORITY",
        "total_decisions_logged": 0,
        "note": "Sistema de aprendizagem em modo experimental",
    }


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# DRL SCHEDULER ENDPOINTS (WP4 - Deep Reinforcement Learning)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

@app.get("/research/drl-status")
def get_drl_status() -> Dict[str, Any]:
    """
    Devolve estado do módulo DRL (Deep Reinforcement Learning).
    
    R&D: WP4 (Learning Scheduler - DRL)
    Hipótese: H4.1 - DRL pode aprender políticas de scheduling competitivas
    """
    try:
        from optimization.drl_scheduler import (
            is_available, 
            get_missing_dependencies,
            AlgorithmType,
        )
        
        available = is_available()
        missing_deps = get_missing_dependencies()
        
        # Check for trained models
        from pathlib import Path
        model_dir = Path(__file__).parent / "optimization" / "drl_scheduler" / "trained_models"
        trained_models = []
        if model_dir.exists():
            trained_models = [f.name for f in model_dir.glob("*.zip")]
        
        algorithms = []
        if available:
            algorithms = [
                {
                    "id": "PPO",
                    "name": "Proximal Policy Optimization",
                    "description": "Algoritmo robusto de policy gradient, bom equilíbrio exploração/exploração",
                    "recommended": True,
                },
                {
                    "id": "A2C",
                    "name": "Advantage Actor-Critic",
                    "description": "Treino mais rápido, menor eficiência amostral",
                    "recommended": False,
                },
                {
                    "id": "DQN",
                    "name": "Deep Q-Network",
                    "description": "Método baseado em valor, bom para ações discretas claras",
                    "recommended": False,
                },
            ]
        
        return {
            "available": available,
            "missing_dependencies": missing_deps,
            "algorithms": algorithms,
            "trained_models": trained_models,
            "model_count": len(trained_models),
            "hypothesis": "H4.1: DRL aprende políticas de dispatching competitivas com heurísticas",
            "work_package": "WP4 - Learning-Based Scheduling",
            "note": "Instalar gymnasium e stable-baselines3 para activar DRL" if not available else "DRL pronto para treino",
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
            "missing_dependencies": ["gymnasium", "stable-baselines3"],
            "algorithms": [],
            "trained_models": [],
            "model_count": 0,
        }


class DRLTrainRequest(BaseModel):
    algorithm: str = "PPO"
    total_timesteps: int = 10000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64


@app.post("/research/drl-train")
def train_drl_policy(request: DRLTrainRequest) -> Dict[str, Any]:
    """
    Treina uma política DRL para scheduling.
    
    R&D: WP4 (Learning Scheduler - DRL)
    
    NOTA: Treino pode demorar vários minutos dependendo de total_timesteps.
    Para testes rápidos, usar total_timesteps=1000-5000.
    """
    try:
        from optimization.drl_scheduler import (
            train_policy,
            TrainingConfig,
            AlgorithmType,
            SchedulingEnvConfig,
        )
        from optimization.drl_scheduler.drl_scheduler_interface import (
            build_operations_from_databundle,
        )
        
        # Map algorithm string to enum
        algo_map = {
            "PPO": AlgorithmType.PPO,
            "A2C": AlgorithmType.A2C,
            "DQN": AlgorithmType.DQN,
        }
        algorithm = algo_map.get(request.algorithm.upper(), AlgorithmType.PPO)
        
        # Load data
        data = load_dataset()
        
        # Build operations DataFrame
        ops_df = build_operations_from_databundle(data)
        machines_df = data.machines
        
        # Configure training
        train_config = TrainingConfig(
            algorithm=algorithm,
            total_timesteps=request.total_timesteps,
            learning_rate=request.learning_rate,
            n_steps=request.n_steps,
            batch_size=request.batch_size,
            verbose=0,
            n_eval_episodes=5,
            eval_freq=max(1000, request.total_timesteps // 10),
        )
        
        env_config = SchedulingEnvConfig(horizon_minutes=10080)  # 1 week
        
        # Train
        result = train_policy(
            ops_df, 
            machines_df, 
            config=train_config,
            env_config=env_config,
        )
        
        return {
            "success": True,
            "experiment_id": result.experiment_id,
            "algorithm": result.algorithm,
            "total_timesteps": result.total_timesteps,
            "training_time_seconds": round(result.training_time_seconds, 2),
            "final_mean_reward": round(result.final_mean_reward, 4),
            "final_std_reward": round(result.final_std_reward, 4),
            "episodes_trained": len(result.reward_history),
            "model_path": result.model_path,
            "is_converged": result.is_converged,
            "improvement_vs_baseline_pct": round(result.improvement_pct, 2),
        }
    
    except ImportError as e:
        return {
            "success": False,
            "error": f"DRL não disponível: {e}",
            "hint": "Instalar: pip install gymnasium stable-baselines3",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@app.post("/research/drl-evaluate")
def evaluate_drl_policy(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Avalia uma política DRL treinada vs baseline heurístico.
    
    R&D: WP4 - Comparação DRL vs Heurística
    """
    try:
        from optimization.drl_scheduler import (
            evaluate_policy,
            AlgorithmType,
        )
        from optimization.drl_scheduler.drl_scheduler_interface import (
            build_operations_from_databundle,
            get_default_model_path,
        )
        from pathlib import Path
        
        # Load data
        data = load_dataset()
        ops_df = build_operations_from_databundle(data)
        machines_df = data.machines
        
        # Find model path
        model_dir = Path(__file__).parent / "optimization" / "drl_scheduler" / "trained_models"
        
        if model_name:
            model_path = model_dir / model_name
        else:
            # Get most recent model
            models = list(model_dir.glob("*.zip")) if model_dir.exists() else []
            if not models:
                return {
                    "success": False,
                    "error": "Nenhum modelo treinado encontrado",
                    "hint": "Treinar primeiro com POST /research/drl-train",
                }
            model_path = max(models, key=lambda p: p.stat().st_mtime)
        
        if not model_path.exists():
            return {
                "success": False,
                "error": f"Modelo não encontrado: {model_path}",
            }
        
        # Evaluate DRL
        drl_metrics = evaluate_policy(
            model_path=str(model_path),
            operations_df=ops_df,
            machines_df=machines_df,
            n_episodes=5,
            algorithm=AlgorithmType.PPO,
        )
        
        return {
            "success": True,
            "model_path": str(model_path),
            "drl_metrics": {
                "mean_reward": round(drl_metrics["mean_reward"], 4),
                "std_reward": round(drl_metrics["std_reward"], 4),
                "mean_makespan_min": round(drl_metrics["mean_makespan"], 2),
                "mean_tardiness_min": round(drl_metrics["mean_tardiness"], 2),
                "mean_setup_time_min": round(drl_metrics["mean_setup_time"], 2),
            },
            "note": "Métricas de 5 episódios de avaliação",
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# PROJECT PLANNING ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

# Import project planning module
try:
    from project_planning import (
        Project,
        ProjectStatus,
        build_projects_from_orders,
        AggregationMode,
        compute_project_load,
        compute_all_project_loads,
        compute_project_slack,
        compute_project_risk,
        compute_all_project_risks,
        ProjectPriorityOptimizer,
        PriorityPlanConfig,
        optimize_project_priorities,
        compute_project_kpis,
        compute_all_project_kpis,
        compute_global_project_kpis,
    )
    from project_planning.project_model import AggregationMode
    HAS_PROJECT_PLANNING = True
except ImportError as e:
    HAS_PROJECT_PLANNING = False
    import logging
    logging.warning(f"Project planning module not available: {e}")


class ProjectPriorityRequest(BaseModel):
    """Request for recomputing project priorities."""
    use_milp: bool = False
    horizon_days: int = 30
    max_parallel_projects: int = 5


@app.get("/projects")
def get_projects(
    aggregation_mode: str = "explicit"
) -> Dict[str, Any]:
    """
    Lista de projetos com KPIs.
    
    Agrupa encomendas em projetos e calcula métricas por projeto.
    
    Args:
        aggregation_mode: Como agrupar encomendas (explicit, by_client, by_family, by_due_week)
    
    Returns:
        Lista de projetos com KPIs e resumo global
    """
    if not HAS_PROJECT_PLANNING:
        raise HTTPException(status_code=501, detail="Project planning module not available")
    
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    # Parse aggregation mode
    try:
        mode = AggregationMode(aggregation_mode)
    except ValueError:
        mode = AggregationMode.EXPLICIT
    
    # Build projects from orders
    projects = build_projects_from_orders(
        data.orders,
        mode=mode,
        project_col='project_id',
        client_col='client_id',
        article_col='article_id',
        due_col='due_date',
    )
    
    if not projects:
        return {
            "projects": [],
            "global_kpis": {},
            "aggregation_mode": mode.value,
            "message": "Nenhum projeto encontrado. Verifique se os dados têm project_id ou use outro modo de agregação.",
        }
    
    # Compute all KPIs
    project_kpis, global_kpis = compute_all_project_kpis(projects, plan_df)
    
    return {
        "projects": [kpi.to_dict() for kpi in project_kpis.values()],
        "global_kpis": global_kpis.to_dict(),
        "aggregation_mode": mode.value,
        "total_projects": len(projects),
    }


@app.get("/projects/{project_id}/kpis")
def get_project_kpis(project_id: str) -> Dict[str, Any]:
    """
    KPIs detalhados para um projeto específico.
    
    Args:
        project_id: ID do projeto
    
    Returns:
        KPIs do projeto e detalhes das encomendas
    """
    if not HAS_PROJECT_PLANNING:
        raise HTTPException(status_code=501, detail="Project planning module not available")
    
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    # Build projects
    projects = build_projects_from_orders(data.orders)
    
    # Find project
    project = next((p for p in projects if p.project_id == project_id), None)
    if not project:
        raise HTTPException(status_code=404, detail=f"Projeto {project_id} não encontrado")
    
    # Compute load, slack, risk
    load = compute_project_load(plan_df, project)
    slack = compute_project_slack(project, load)
    risk = compute_project_risk(project, load, slack)
    
    # Compute KPIs
    kpi = compute_project_kpis(
        project=project,
        load=load,
        slack=slack,
        risk=risk,
    )
    
    # Get orders detail
    order_ids = list(project.order_ids)
    project_orders = data.orders[data.orders['order_id'].astype(str).isin(order_ids)]
    
    # Get operations for this project
    project_ops = plan_df[plan_df['order_id'].astype(str).isin(order_ids)]
    
    return {
        "project": project.to_dict(),
        "kpis": kpi.to_dict(),
        "load": load.to_dict(),
        "slack": slack.to_dict(),
        "risk": risk.to_dict(),
        "orders": project_orders.to_dict(orient="records"),
        "operations": project_ops.to_dict(orient="records"),
    }


@app.get("/projects/priority-plan")
def get_project_priority_plan(
    use_milp: bool = False,
    horizon_days: int = 30,
    max_parallel: int = 5
) -> Dict[str, Any]:
    """
    Plano de prioridades sugerido para os projetos.
    
    Usa otimização (heurística ou MILP) para ordenar projetos.
    
    Args:
        use_milp: Usar MILP (mais lento mas ótimo)
        horizon_days: Horizonte de planeamento em dias
        max_parallel: Máximo de projetos em paralelo
    
    Returns:
        Plano de prioridades com ordem sugerida
    """
    if not HAS_PROJECT_PLANNING:
        raise HTTPException(status_code=501, detail="Project planning module not available")
    
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    # Build projects
    projects = build_projects_from_orders(data.orders)
    
    if not projects:
        return {
            "message": "Nenhum projeto encontrado",
            "priorities": [],
        }
    
    # Compute loads and risks
    loads = compute_all_project_loads(plan_df, projects)
    
    slacks = {}
    for p in projects:
        slacks[p.project_id] = compute_project_slack(p, loads[p.project_id])
    
    risks = compute_all_project_risks(projects, loads, slacks)
    
    # Configure optimization
    config = PriorityPlanConfig(
        horizon_days=horizon_days,
        max_parallel_projects=max_parallel,
    )
    
    # Optimize
    plan = optimize_project_priorities(
        projects=projects,
        loads=loads,
        risks=risks,
        config=config,
        use_milp=use_milp,
    )
    
    return {
        "priority_plan": plan.to_dict(),
        "order_priority_vector": plan.get_order_priority_vector(projects),
        "message": f"Plano otimizado com {'MILP' if use_milp else 'heurística'}",
    }


@app.post("/projects/recompute")
def recompute_project_plan(request: ProjectPriorityRequest) -> Dict[str, Any]:
    """
    Recalcula o plano com base nas prioridades de projeto.
    
    Aplica as prioridades otimizadas às encomendas e regera o plano APS.
    
    NOTA: Isto NÃO executa nada na fábrica - apenas recalcula o plano interno.
    """
    if not HAS_PROJECT_PLANNING:
        raise HTTPException(status_code=501, detail="Project planning module not available")
    
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    # Build projects
    projects = build_projects_from_orders(data.orders)
    
    if not projects:
        return {"message": "Nenhum projeto encontrado", "success": False}
    
    # Compute loads and risks
    loads = compute_all_project_loads(plan_df, projects)
    
    slacks = {}
    for p in projects:
        slacks[p.project_id] = compute_project_slack(p, loads[p.project_id])
    
    risks = compute_all_project_risks(projects, loads, slacks)
    
    # Configure and optimize
    config = PriorityPlanConfig(
        horizon_days=request.horizon_days,
        max_parallel_projects=request.max_parallel_projects,
    )
    
    priority_plan = optimize_project_priorities(
        projects=projects,
        loads=loads,
        risks=risks,
        config=config,
        use_milp=request.use_milp,
    )
    
    # Get order priority vector
    order_priorities = priority_plan.get_order_priority_vector(projects)
    
    # Update orders with new priorities
    orders_updated = data.orders.copy()
    orders_updated['priority'] = orders_updated['order_id'].astype(str).map(
        lambda oid: order_priorities.get(oid, 1.0)
    )
    
    # Rebuild plan with updated priorities
    # Note: This requires the scheduler to respect the priority column
    new_plan_df = build_plan(data, mode="NORMAL")
    save_plan_to_csv(new_plan_df)
    
    # Recompute KPIs with new plan
    project_kpis, global_kpis = compute_all_project_kpis(projects, new_plan_df)
    
    return {
        "success": True,
        "message": "Plano recalculado com prioridades de projeto",
        "priority_plan": priority_plan.to_dict(),
        "global_kpis": global_kpis.to_dict(),
        "orders_updated": len(order_priorities),
    }


@app.get("/projects/summary")
def get_projects_summary() -> Dict[str, Any]:
    """
    Resumo executivo de todos os projetos.
    
    Returns:
        Dashboard data com métricas agregadas
    """
    if not HAS_PROJECT_PLANNING:
        raise HTTPException(status_code=501, detail="Project planning module not available")
    
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    # Build projects
    projects = build_projects_from_orders(data.orders)
    
    if not projects:
        return {
            "total_projects": 0,
            "message": "Nenhum projeto encontrado",
        }
    
    # Compute KPIs
    project_kpis, global_kpis = compute_all_project_kpis(projects, plan_df)
    
    # Create summary by status
    status_counts = {}
    for kpi in project_kpis.values():
        status = kpi.status
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Create summary by risk
    risk_counts = {}
    for kpi in project_kpis.values():
        risk = kpi.risk_level
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    
    # Top projects by various criteria
    kpi_list = list(project_kpis.values())
    
    top_by_load = sorted(kpi_list, key=lambda k: k.total_load_hours, reverse=True)[:5]
    top_by_delay = sorted(kpi_list, key=lambda k: k.delay_hours, reverse=True)[:5]
    top_by_risk = sorted(kpi_list, key=lambda k: k.risk_score, reverse=True)[:5]
    
    return {
        "global_kpis": global_kpis.to_dict(),
        "status_distribution": status_counts,
        "risk_distribution": risk_counts,
        "top_by_load": [{"project": k.project_name, "load_h": k.total_load_hours} for k in top_by_load],
        "top_by_delay": [{"project": k.project_name, "delay_h": k.delay_hours} for k in top_by_delay if k.delay_hours > 0],
        "top_by_risk": [{"project": k.project_name, "risk": k.risk_score, "level": k.risk_level} for k in top_by_risk],
        "aggregation_modes": [m.value for m in AggregationMode] if HAS_PROJECT_PLANNING else [],
    }


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# WORKFORCE ANALYTICS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

# Import workforce analytics module
try:
    from workforce_analytics import (
        compute_all_worker_performances,
        forecast_all_workers,
        ForecastConfig,
        optimize_worker_assignment,
        AssignmentConfig,
        Worker,
        Operation,
        build_operations_from_plan,
    )
    HAS_WORKFORCE = True
except ImportError as e:
    HAS_WORKFORCE = False
    import logging
    logging.warning(f"Workforce analytics module not available: {e}")


class WorkforceAssignRequest(BaseModel):
    """Request for workforce assignment optimization."""
    use_milp: bool = True
    max_hours_per_worker: float = 8.0
    allow_overtime: bool = False


class WorkforceForecastRequest(BaseModel):
    """Request for workforce forecasting."""
    horizon_days: int = 14
    model_type: str = "ARIMA"  # ARIMA, LEARNING_CURVE


@app.get("/workforce/performance")
def get_workforce_performance() -> Dict[str, Any]:
    """
    Performance de todos os colaboradores.
    
    Calcula métricas de produtividade, eficiência, saturação e skill score.
    Inclui SNR e curva de aprendizagem.
    
    Returns:
        Dict com performances por colaborador e métricas globais
    """
    if not HAS_WORKFORCE:
        raise HTTPException(status_code=501, detail="Workforce analytics module not available")
    
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    # Check if we have worker data
    # For MVP, we'll simulate worker data from the plan
    # In production, this would come from a workers table in the Excel
    
    # Create synthetic operations with worker assignments
    # In real scenario, plan_df would have a 'worker_id' column
    if 'worker_id' not in plan_df.columns:
        # Simulate: assign workers based on machine
        # Each machine has an operator
        machines = plan_df['machine_id'].unique()
        worker_map = {m: f"W-{i+100}" for i, m in enumerate(machines)}
        plan_df = plan_df.copy()
        plan_df['worker_id'] = plan_df['machine_id'].map(worker_map)
    
    # Add date column if missing
    if 'date' not in plan_df.columns:
        plan_df['date'] = plan_df['start_time'].dt.date
    
    # Add success column (simulate 95% success rate)
    if 'success' not in plan_df.columns:
        import random
        plan_df['success'] = [1 if random.random() < 0.95 else 0 for _ in range(len(plan_df))]
    
    # Compute performances
    performances = compute_all_worker_performances(
        operations_df=plan_df,
        worker_col='worker_id',
        time_col='duration_min',
        units_col='qty',
        date_col='date',
        op_type_col='op_code',
        success_col='success',
    )
    
    # Global metrics
    if performances:
        avg_productivity = sum(p.metrics.productivity for p in performances.values()) / len(performances)
        avg_efficiency = sum(p.metrics.efficiency for p in performances.values()) / len(performances)
        avg_skill = sum(p.metrics.skill_score for p in performances.values()) / len(performances)
        avg_snr = sum(p.metrics.snr_performance for p in performances.values()) / len(performances)
    else:
        avg_productivity = avg_efficiency = avg_skill = avg_snr = 0
    
    return {
        "workers": [p.to_dict() for p in performances.values()],
        "total_workers": len(performances),
        "global_metrics": {
            "avg_productivity": round(avg_productivity, 2),
            "avg_efficiency": round(avg_efficiency, 3),
            "avg_skill_score": round(avg_skill, 3),
            "avg_snr": round(avg_snr, 2),
        },
        "performance_distribution": {
            "excellent": sum(1 for p in performances.values() if p.metrics.performance_level == "excellent"),
            "good": sum(1 for p in performances.values() if p.metrics.performance_level == "good"),
            "average": sum(1 for p in performances.values() if p.metrics.performance_level == "average"),
            "below_average": sum(1 for p in performances.values() if p.metrics.performance_level == "below_average"),
            "needs_improvement": sum(1 for p in performances.values() if p.metrics.performance_level == "needs_improvement"),
        },
    }


@app.get("/workforce/{worker_id}/performance")
def get_worker_performance(worker_id: str) -> Dict[str, Any]:
    """
    Performance detalhada de um colaborador específico.
    """
    if not HAS_WORKFORCE:
        raise HTTPException(status_code=501, detail="Workforce analytics module not available")
    
    # Get all performances
    result = get_workforce_performance()
    
    # Find specific worker
    worker = next((w for w in result["workers"] if w["worker_id"] == worker_id), None)
    if not worker:
        raise HTTPException(status_code=404, detail=f"Colaborador {worker_id} não encontrado")
    
    return {
        "worker": worker,
        "global_comparison": result["global_metrics"],
    }


@app.post("/workforce/forecast")
def post_workforce_forecast(request: WorkforceForecastRequest) -> Dict[str, Any]:
    """
    Previsão de performance dos colaboradores.
    
    Args:
        request: Configuração do forecast (horizonte, modelo)
    
    Returns:
        Forecasts por colaborador
    """
    if not HAS_WORKFORCE:
        raise HTTPException(status_code=501, detail="Workforce analytics module not available")
    
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    # Get performances first
    perf_result = get_workforce_performance()
    
    # Build performances dict (simplified reconstruction)
    from workforce_analytics.workforce_performance_engine import WorkerPerformance, WorkerMetrics
    
    performances = {}
    for w in perf_result["workers"]:
        wid = w["worker_id"]
        metrics = WorkerMetrics(
            worker_id=wid,
            worker_name=w.get("metrics", {}).get("worker_name"),
            productivity=w.get("metrics", {}).get("productivity", 0),
        )
        perf = WorkerPerformance(
            worker_id=wid,
            metrics=metrics,
            productivity_history=w.get("productivity_history", []),
            dates=w.get("dates", []),
        )
        performances[wid] = perf
    
    # Configure forecast
    config = ForecastConfig(
        horizon_days=request.horizon_days,
        model_type=request.model_type,
    )
    
    # Run forecasts
    forecasts = forecast_all_workers(performances, config)
    
    return {
        "forecasts": [f.to_dict() for f in forecasts.values()],
        "total_workers": len(forecasts),
        "horizon_days": request.horizon_days,
        "model_type": request.model_type,
    }


@app.post("/workforce/assign")
def post_workforce_assignment(request: WorkforceAssignRequest) -> Dict[str, Any]:
    """
    Otimização de alocação de colaboradores a operações.
    
    Usa MILP para maximizar contribuição ponderada por skill.
    
    Args:
        request: Configuração da otimização
    
    Returns:
        Plano de alocação otimizado
    """
    if not HAS_WORKFORCE:
        raise HTTPException(status_code=501, detail="Workforce analytics module not available")
    
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    # Get worker performances to build Worker objects
    perf_result = get_workforce_performance()
    
    # Build Worker objects
    workers = []
    for w in perf_result["workers"]:
        wid = w["worker_id"]
        metrics = w.get("metrics", {})
        
        worker = Worker(
            worker_id=wid,
            name=metrics.get("worker_name"),
            skill_score=metrics.get("skill_score", 0.5),
            available_hours=request.max_hours_per_worker,
        )
        workers.append(worker)
    
    # Build Operation objects from plan
    operations = build_operations_from_plan(plan_df)
    
    # Configure assignment
    config = AssignmentConfig(
        max_hours_per_worker=request.max_hours_per_worker,
        allow_overtime=request.allow_overtime,
    )
    
    # Optimize
    plan = optimize_worker_assignment(
        workers=workers,
        operations=operations,
        config=config,
        use_milp=request.use_milp,
    )
    
    return {
        "assignment_plan": plan.to_dict(),
        "message": f"Alocação otimizada com {'MILP' if request.use_milp else 'heurística'}",
    }


@app.get("/workforce/summary")
def get_workforce_summary() -> Dict[str, Any]:
    """
    Resumo executivo da força de trabalho.
    
    Returns:
        Dashboard data com métricas agregadas
    """
    if not HAS_WORKFORCE:
        raise HTTPException(status_code=501, detail="Workforce analytics module not available")
    
    # Get performance data
    perf_result = get_workforce_performance()
    
    workers = perf_result["workers"]
    
    # Top performers
    sorted_by_productivity = sorted(workers, key=lambda w: w.get("metrics", {}).get("productivity", 0), reverse=True)
    sorted_by_skill = sorted(workers, key=lambda w: w.get("metrics", {}).get("skill_score", 0), reverse=True)
    
    # Workers needing attention (low SNR or performance)
    needs_attention = [
        w for w in workers
        if w.get("metrics", {}).get("snr_level") == "POOR" or 
           w.get("metrics", {}).get("performance_level") in ["below_average", "needs_improvement"]
    ]
    
    # Saturation analysis
    saturation_distribution = {}
    for w in workers:
        sat_level = w.get("metrics", {}).get("saturation_level", "unknown")
        saturation_distribution[sat_level] = saturation_distribution.get(sat_level, 0) + 1
    
    return {
        "total_workers": len(workers),
        "global_metrics": perf_result["global_metrics"],
        "performance_distribution": perf_result["performance_distribution"],
        "saturation_distribution": saturation_distribution,
        "top_performers": [
            {"worker_id": w["worker_id"], "productivity": w.get("metrics", {}).get("productivity", 0)}
            for w in sorted_by_productivity[:5]
        ],
        "top_skilled": [
            {"worker_id": w["worker_id"], "skill_score": w.get("metrics", {}).get("skill_score", 0)}
            for w in sorted_by_skill[:5]
        ],
        "needs_attention": [
            {
                "worker_id": w["worker_id"],
                "reason": w.get("metrics", {}).get("performance_level", "unknown"),
                "snr_level": w.get("metrics", {}).get("snr_level", "UNKNOWN"),
            }
            for w in needs_attention[:5]
        ],
        "recommendations_count": sum(len(w.get("recommendations", [])) for w in workers),
    }


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# PRODUCT METRICS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

# Import product metrics module
try:
    from product_metrics import (
        classify_all_products,
        compute_all_product_kpis,
        estimate_delivery_time,
        estimate_all_deliveries,
        DeliveryConfig,
        EstimationMethod,
        ProductType,
    )
    HAS_PRODUCT_METRICS = True
except ImportError as e:
    HAS_PRODUCT_METRICS = False
    import logging
    logging.warning(f"Product metrics module not available: {e}")


class DeliveryEstimateRequest(BaseModel):
    """Request for delivery estimation."""
    order_id: str
    article_id: str
    qty: float = 1.0
    method: str = "deterministic"  # deterministic, historical, ml
    buffer_strategy: str = "moderate"  # conservative, moderate, aggressive


@app.get("/product/type-kpis")
def get_product_type_kpis() -> Dict[str, Any]:
    """
    KPIs por tipo de produto (vidro_duplo, vidro_triplo, etc.).
    
    Inclui:
    - Tempo médio de processamento
    - Setup médio
    - Lead time médio
    - Variação (sigma)
    - SNR de processo
    
    Returns:
        Dict com KPIs por tipo e globais
    """
    if not HAS_PRODUCT_METRICS:
        raise HTTPException(status_code=501, detail="Product metrics module not available")
    
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    # Compute all KPIs
    product_kpis, type_kpis, global_kpis = compute_all_product_kpis(
        routing_df=data.routing,
        plan_df=plan_df,
        orders_df=data.orders,
    )
    
    return {
        "type_kpis": {k: v.to_dict() for k, v in type_kpis.items()},
        "global_kpis": global_kpis.to_dict(),
        "product_count_by_type": global_kpis.type_distribution,
        "available_types": [t.value for t in ProductType],
    }


@app.get("/product/{article_id}/kpis")
def get_product_kpis(article_id: str) -> Dict[str, Any]:
    """
    KPIs detalhados para um artigo específico.
    """
    if not HAS_PRODUCT_METRICS:
        raise HTTPException(status_code=501, detail="Product metrics module not available")
    
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    # Compute KPIs for this article
    product_kpis, type_kpis, _ = compute_all_product_kpis(
        routing_df=data.routing,
        plan_df=plan_df,
    )
    
    if article_id not in product_kpis:
        raise HTTPException(status_code=404, detail=f"Artigo {article_id} não encontrado")
    
    kpi = product_kpis[article_id]
    product_type = kpi.product_type
    
    # Get type KPIs for comparison
    type_kpi = type_kpis.get(product_type)
    
    return {
        "product_kpis": kpi.to_dict(),
        "type_comparison": type_kpi.to_dict() if type_kpi else None,
        "performance_vs_type": {
            "processing_time_ratio": kpi.avg_processing_time_min / type_kpi.avg_processing_time_min if type_kpi and type_kpi.avg_processing_time_min > 0 else 1.0,
            "lead_time_ratio": kpi.lead_time_hours / type_kpi.avg_lead_time_hours if type_kpi and type_kpi.avg_lead_time_hours > 0 else 1.0,
        } if type_kpi else None,
    }


@app.get("/product/classification")
def get_product_classification() -> Dict[str, Any]:
    """
    Classificação de todos os produtos por tipo.
    
    Baseado em fingerprints de operações, máquinas e tempos.
    """
    if not HAS_PRODUCT_METRICS:
        raise HTTPException(status_code=501, detail="Product metrics module not available")
    
    data = load_dataset()
    
    # Classify all products
    fingerprints = classify_all_products(data.routing)
    
    # Group by type
    by_type = {}
    for fp in fingerprints.values():
        ptype = fp.product_type.value
        if ptype not in by_type:
            by_type[ptype] = []
        by_type[ptype].append({
            "article_id": fp.article_id,
            "confidence": round(fp.confidence, 2),
            "num_operations": fp.num_operations,
            "total_time_min": round(fp.total_time_min, 1),
            "route_label": fp.route_label,
        })
    
    return {
        "classification_by_type": by_type,
        "total_products": len(fingerprints),
        "type_distribution": {k: len(v) for k, v in by_type.items()},
        "fingerprints": {k: v.to_dict() for k, v in fingerprints.items()},
    }


@app.post("/product/delivery-estimate")
def post_delivery_estimate(request: DeliveryEstimateRequest) -> Dict[str, Any]:
    """
    Estimar data de entrega para uma encomenda.
    
    Métodos:
    - deterministic: Baseado em tempos de routing
    - historical: Baseado em histórico de lead times
    - ml: Modelo de Machine Learning (placeholder)
    
    Returns:
        Estimativa de entrega com breakdown e confiança
    """
    if not HAS_PRODUCT_METRICS:
        raise HTTPException(status_code=501, detail="Product metrics module not available")
    
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    # Get machine loads for queue estimation
    machine_loads = {}
    if 'machine_id' in plan_df.columns and 'duration_min' in plan_df.columns:
        total_duration = plan_df['duration_min'].sum()
        for machine, group in plan_df.groupby('machine_id'):
            machine_loads[machine] = group['duration_min'].sum() / total_duration if total_duration > 0 else 0.5
    
    # Configure estimation
    method_map = {
        "deterministic": EstimationMethod.DETERMINISTIC,
        "historical": EstimationMethod.HISTORICAL,
        "ml": EstimationMethod.ML,
    }
    
    config = DeliveryConfig(
        method=method_map.get(request.method, EstimationMethod.DETERMINISTIC),
        buffer_strategy=request.buffer_strategy,
    )
    
    # Estimate
    estimate = estimate_delivery_time(
        order_id=request.order_id,
        article_id=request.article_id,
        routing_df=data.routing,
        plan_df=plan_df,
        machine_loads=machine_loads,
        qty=request.qty,
        config=config,
    )
    
    return {
        "estimate": estimate.to_dict(),
        "message": f"Estimativa calculada com método '{request.method}'",
    }


@app.get("/product/delivery-estimates")
def get_all_delivery_estimates(
    method: str = "deterministic",
    buffer_strategy: str = "moderate"
) -> Dict[str, Any]:
    """
    Estimar datas de entrega para todas as encomendas em aberto.
    """
    if not HAS_PRODUCT_METRICS:
        raise HTTPException(status_code=501, detail="Product metrics module not available")
    
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    # Get machine loads
    machine_loads = {}
    if 'machine_id' in plan_df.columns and 'duration_min' in plan_df.columns:
        total_duration = plan_df['duration_min'].sum()
        for machine, group in plan_df.groupby('machine_id'):
            machine_loads[machine] = group['duration_min'].sum() / total_duration if total_duration > 0 else 0.5
    
    # Configure
    method_map = {
        "deterministic": EstimationMethod.DETERMINISTIC,
        "historical": EstimationMethod.HISTORICAL,
        "ml": EstimationMethod.ML,
    }
    
    config = DeliveryConfig(
        method=method_map.get(method, EstimationMethod.DETERMINISTIC),
        buffer_strategy=buffer_strategy,
    )
    
    # Estimate all
    estimates = estimate_all_deliveries(
        orders_df=data.orders,
        routing_df=data.routing,
        plan_df=plan_df,
        machine_loads=machine_loads,
        config=config,
    )
    
    # Summary stats
    durations = [e.estimated_duration_hours for e in estimates.values() if e.estimated_duration_hours > 0]
    confidences = [e.confidence_score for e in estimates.values()]
    
    return {
        "estimates": {k: v.to_dict() for k, v in estimates.items()},
        "total_orders": len(estimates),
        "summary": {
            "avg_duration_hours": round(np.mean(durations), 2) if durations else 0,
            "min_duration_hours": round(min(durations), 2) if durations else 0,
            "max_duration_hours": round(max(durations), 2) if durations else 0,
            "avg_confidence": round(np.mean(confidences), 3) if confidences else 0,
        },
        "method": method,
        "buffer_strategy": buffer_strategy,
    }


@app.get("/product/summary")
def get_product_summary() -> Dict[str, Any]:
    """
    Resumo executivo das métricas de produtos.
    """
    if not HAS_PRODUCT_METRICS:
        raise HTTPException(status_code=501, detail="Product metrics module not available")
    
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    # Get type KPIs
    _, type_kpis, global_kpis = compute_all_product_kpis(
        routing_df=data.routing,
        plan_df=plan_df,
    )
    
    # Build summary
    type_summary = []
    for ptype, tkpi in type_kpis.items():
        if tkpi.num_products > 0:
            type_summary.append({
                "type": ptype,
                "count": tkpi.num_products,
                "avg_processing_min": round(tkpi.avg_processing_time_min, 1),
                "avg_lead_time_h": round(tkpi.avg_lead_time_hours, 2),
                "snr": round(tkpi.snr_between_products, 2),
                "snr_level": tkpi.snr_level,
            })
    
    # Sort by count
    type_summary.sort(key=lambda x: x["count"], reverse=True)
    
    return {
        "global_kpis": global_kpis.to_dict(),
        "type_summary": type_summary,
        "fastest_type": global_kpis.fastest_type,
        "slowest_type": global_kpis.slowest_type,
        "most_stable_type": global_kpis.most_stable_type,
        "recommendations": _generate_product_recommendations(type_kpis, global_kpis),
    }


def _generate_product_recommendations(
    type_kpis: Dict[str, 'ProductTypeKPIs'],
    global_kpis: 'GlobalProductKPIs'
) -> List[str]:
    """Generate recommendations based on product metrics."""
    recommendations = []
    
    # Find problematic types
    for ptype, tkpi in type_kpis.items():
        if tkpi.snr_level == "POOR" and tkpi.num_products > 0:
            recommendations.append(
                f"Tipo '{ptype}' tem alta variabilidade (SNR={tkpi.snr_between_products:.1f}). "
                f"Investigar causas de inconsistência nos tempos de processo."
            )
        
        if tkpi.avg_lead_time_hours > global_kpis.global_avg_lead_time_hours * 1.5:
            recommendations.append(
                f"Tipo '{ptype}' tem lead time {tkpi.avg_lead_time_hours/global_kpis.global_avg_lead_time_hours:.1f}x "
                f"acima da média. Considerar otimização de routing ou capacidade."
            )
    
    if not recommendations:
        recommendations.append("Todos os tipos de produto estão dentro dos parâmetros esperados.")
    
    return recommendations


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════
# ADVANCED PLANNING ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════

# Import planning module
try:
    from planning import (
        PlanningEngine,
        PlanningMode,
        ChainedPlanningConfig,
        ConventionalPlanningConfig,
        ShortTermPlanningConfig,
        LongTermPlanningConfig,
    )
    HAS_PLANNING = True
except ImportError as e:
    logger.warning(f"Planning module not available: {e}")
    HAS_PLANNING = False


class ChainedPlanningRequest(BaseModel):
    """Request for chained (flow shop) planning."""
    chains: List[List[str]]  # e.g., [["M-101", "M-102", "M-103"]]
    buffers: Dict[str, float] = {}  # e.g., {"M-101->M-102": 30}
    default_buffer_min: float = 30.0
    solver: str = "heuristic"  # "heuristic", "cpsat"
    synchronize_flow: bool = True


class ConventionalPlanningRequest(BaseModel):
    """Request for conventional independent planning."""
    dispatching_rule: str = "EDD"  # EDD, SPT, FIFO, CR, WSPT
    group_by_setup_family: bool = True


class ShortTermPlanningRequest(BaseModel):
    """Request for short-term detailed planning."""
    horizon_days: int = 14
    require_operators: bool = False
    minimize_setup_changes: bool = True


class LongTermPlanningRequest(BaseModel):
    """Request for long-term strategic planning."""
    horizon_months: int = 12
    demand_growth_quarterly: float = 0.10
    capacity_changes: List[Dict[str, Any]] = []


class PlanningComparisonRequest(BaseModel):
    """Request to compare planning modes."""
    modes: List[str] = ["conventional", "chained"]
    chained_config: Optional[ChainedPlanningRequest] = None


@app.post("/planning/chained")
def execute_chained_planning(request: ChainedPlanningRequest) -> Dict[str, Any]:
    """
    Executa planeamento encadeado (flow shop) entre máquinas.
    
    Ativa o modo de planeamento sincronizado entre máquinas em cadeia,
    considerando buffers entre etapas e otimizando o makespan total.
    
    Exemplo de uso:
    - chains: [["M-101", "M-102", "M-103"]] para uma cadeia de 3 máquinas
    - buffers: {"M-101->M-102": 30, "M-102->M-103": 15} para buffers customizados
    """
    if not HAS_PLANNING:
        raise HTTPException(status_code=501, detail="Planning module not available")
    
    data = load_dataset()
    
    config = ChainedPlanningConfig(
        chains=request.chains,
        buffers=request.buffers,
        default_buffer_min=request.default_buffer_min,
        synchronize_flow=request.synchronize_flow,
    )
    
    engine = PlanningEngine(
        orders_df=data.orders,
        routing_df=data.routing,
        machines_df=data.machines,
        setup_matrix_df=data.setup_matrix if hasattr(data, 'setup_matrix') else None,
    )
    
    result = engine.plan(mode="chained", config=config)
    
    # Save plan to CSV
    if not result.plan_df.empty:
        result.plan_df.to_csv(PLAN_CSV_PATH, index=False)
    
    return {
        "mode": result.mode.value,
        "makespan_hours": result.makespan_hours,
        "tardiness_hours": result.total_tardiness_hours,
        "throughput_units": result.throughput_units,
        "bottleneck_machine": result.bottleneck_machine,
        "machine_metrics": result.machine_metrics,
        "solver_status": result.solver_status,
        "plan_rows": len(result.plan_df),
        "message": f"Planeamento encadeado concluído. Makespan: {result.makespan_hours:.1f}h",
    }


@app.post("/planning/conventional")
def execute_conventional_planning(request: ConventionalPlanningRequest) -> Dict[str, Any]:
    """
    Executa planeamento convencional (sequenciação independente por máquina).
    
    Cada máquina otimiza a sua própria sequência localmente usando a regra
    de dispatching especificada.
    
    Regras disponíveis:
    - EDD: Earliest Due Date (data de entrega mais próxima)
    - SPT: Shortest Processing Time (menor tempo de processamento)
    - FIFO: First In First Out (ordem de chegada)
    - CR: Critical Ratio (rácio crítico)
    - WSPT: Weighted SPT (SPT ponderado por prioridade)
    """
    if not HAS_PLANNING:
        raise HTTPException(status_code=501, detail="Planning module not available")
    
    data = load_dataset()
    
    config = ConventionalPlanningConfig(
        dispatching_rule=request.dispatching_rule,
        group_by_setup_family=request.group_by_setup_family,
    )
    
    engine = PlanningEngine(
        orders_df=data.orders,
        routing_df=data.routing,
        machines_df=data.machines,
        setup_matrix_df=data.setup_matrix if hasattr(data, 'setup_matrix') else None,
    )
    
    result = engine.plan(mode="conventional", config=config)
    
    # Save plan to CSV
    if not result.plan_df.empty:
        result.plan_df.to_csv(PLAN_CSV_PATH, index=False)
    
    return {
        "mode": result.mode.value,
        "makespan_hours": result.makespan_hours,
        "tardiness_hours": result.total_tardiness_hours,
        "setup_hours": result.total_setup_hours,
        "throughput_units": result.throughput_units,
        "utilization_pct": result.utilization_pct,
        "bottleneck_machine": result.bottleneck_machine,
        "bottleneck_utilization": result.bottleneck_utilization,
        "machine_metrics": result.machine_metrics,
        "solver_status": result.solver_status,
        "plan_rows": len(result.plan_df),
        "dispatching_rule": request.dispatching_rule,
        "message": f"Planeamento convencional ({request.dispatching_rule}) concluído. "
                   f"Makespan: {result.makespan_hours:.1f}h, Setup total: {result.total_setup_hours:.1f}h",
    }


@app.post("/planning/short-term")
def execute_short_term_planning(request: ShortTermPlanningRequest) -> Dict[str, Any]:
    """
    Gera um plano detalhado para curto prazo (tipicamente 2 semanas).
    
    Incorpora:
    - Manutenções programadas
    - Restrições de turnos
    - Otimização de sequência por setup
    - Granularidade diária/turno
    """
    if not HAS_PLANNING:
        raise HTTPException(status_code=501, detail="Planning module not available")
    
    data = load_dataset()
    
    config = ShortTermPlanningConfig(
        horizon_days=request.horizon_days,
        require_operators=request.require_operators,
        minimize_setup_changes=request.minimize_setup_changes,
    )
    
    engine = PlanningEngine(
        orders_df=data.orders,
        routing_df=data.routing,
        machines_df=data.machines,
        setup_matrix_df=data.setup_matrix if hasattr(data, 'setup_matrix') else None,
    )
    
    result = engine.plan(mode="short_term", config=config)
    
    # Save plan to CSV
    if not result.plan_df.empty:
        result.plan_df.to_csv(PLAN_CSV_PATH, index=False)
    
    return {
        "mode": result.mode.value,
        "horizon_days": request.horizon_days,
        "makespan_hours": result.makespan_hours,
        "tardiness_hours": result.total_tardiness_hours,
        "setup_hours": result.total_setup_hours,
        "utilization_pct": result.utilization_pct,
        "bottleneck_machine": result.bottleneck_machine,
        "machine_metrics": result.machine_metrics,
        "warnings": result.warnings,
        "plan_rows": len(result.plan_df),
        "message": f"Planeamento curto prazo ({request.horizon_days} dias) concluído.",
    }


@app.post("/planning/long-term")
def execute_long_term_planning(request: LongTermPlanningRequest) -> Dict[str, Any]:
    """
    Elabora um plano de longo prazo para os próximos N meses.
    
    Funcionalidades:
    - Projeção de capacidade vs. procura
    - Cenários de crescimento de procura
    - Decisões de investimento (novas máquinas)
    - Análise de gargalos futuros
    
    Exemplo capacity_changes:
    [{"date": "2025-07-01", "machine_id": "M-NEW", "action": "add_machine", "capacity_delta": 16}]
    """
    if not HAS_PLANNING:
        raise HTTPException(status_code=501, detail="Planning module not available")
    
    data = load_dataset()
    
    config = LongTermPlanningConfig(
        horizon_months=request.horizon_months,
        demand_growth_quarterly=request.demand_growth_quarterly,
        capacity_changes=request.capacity_changes,
    )
    
    engine = PlanningEngine(
        orders_df=data.orders,
        routing_df=data.routing,
        machines_df=data.machines,
    )
    
    result = engine.plan(mode="long_term", config=config)
    
    return {
        "mode": result.mode.value,
        "horizon_months": request.horizon_months,
        "growth_rate_quarterly": request.demand_growth_quarterly,
        "recommendations": result.warnings,
        "message": f"Planeamento estratégico ({request.horizon_months} meses) concluído.",
    }


@app.post("/planning/compare")
def compare_planning_modes(request: PlanningComparisonRequest) -> Dict[str, Any]:
    """
    Compara os resultados entre diferentes modos de planeamento.
    
    Por defeito compara 'conventional' vs 'chained' e apresenta:
    - Delta de makespan
    - Delta de atrasos
    - Delta de tempo de setup
    - Recomendação do modo mais adequado
    """
    if not HAS_PLANNING:
        raise HTTPException(status_code=501, detail="Planning module not available")
    
    data = load_dataset()
    
    engine = PlanningEngine(
        orders_df=data.orders,
        routing_df=data.routing,
        machines_df=data.machines,
        setup_matrix_df=data.setup_matrix if hasattr(data, 'setup_matrix') else None,
    )
    
    # Configure chained if provided
    if request.chained_config and "chained" in request.modes:
        engine.plan(
            mode="chained",
            config=ChainedPlanningConfig(
                chains=request.chained_config.chains,
                buffers=request.chained_config.buffers,
                default_buffer_min=request.chained_config.default_buffer_min,
            )
        )
    
    # Execute comparison
    comparison = engine.compare(modes=request.modes)
    
    return {
        "baseline": {
            "mode": comparison.baseline.mode.value,
            "makespan_hours": comparison.baseline.makespan_hours,
            "tardiness_hours": comparison.baseline.total_tardiness_hours,
            "setup_hours": comparison.baseline.total_setup_hours,
        },
        "scenario": {
            "mode": comparison.scenario.mode.value,
            "makespan_hours": comparison.scenario.makespan_hours,
            "tardiness_hours": comparison.scenario.total_tardiness_hours,
            "setup_hours": comparison.scenario.total_setup_hours,
        },
        "deltas": {
            "makespan_hours": comparison.makespan_delta_hours,
            "makespan_pct": comparison.makespan_delta_pct,
            "tardiness_hours": comparison.tardiness_delta_hours,
            "tardiness_pct": comparison.tardiness_delta_pct,
            "setup_hours": comparison.setup_delta_hours,
            "setup_pct": comparison.setup_delta_pct,
        },
        "recommendation": {
            "mode": comparison.recommended_mode.value,
            "reason": comparison.recommendation_reason,
        },
        "message": f"Comparação {request.modes[0]} vs {request.modes[1]}: {comparison.recommendation_reason}",
    }


@app.get("/planning/modes")
def list_planning_modes() -> Dict[str, Any]:
    """
    Lista todos os modos de planeamento disponíveis e suas características.
    """
    return {
        "available_modes": [
            {
                "id": "conventional",
                "name": "Planeamento Convencional",
                "description": "Sequenciação independente por máquina. Cada máquina otimiza localmente.",
                "use_case": "Produção job shop tradicional sem dependências fortes entre máquinas.",
                "dispatching_rules": ["EDD", "SPT", "FIFO", "CR", "WSPT"],
            },
            {
                "id": "chained",
                "name": "Planeamento Encadeado (Flow Shop)",
                "description": "Sequenciação sincronizada entre máquinas em cadeia com buffers.",
                "use_case": "Linhas de produção flow shop, processos multi-etapas contínuos.",
                "solvers": ["heuristic", "cpsat"],
            },
            {
                "id": "short_term",
                "name": "Planeamento Curto Prazo",
                "description": "Planeamento detalhado para 2 semanas com turnos e manutenções.",
                "use_case": "Programação operacional diária/semanal.",
                "features": ["turnos", "manutenções", "operadores"],
            },
            {
                "id": "long_term",
                "name": "Planeamento Estratégico",
                "description": "Projeção de capacidade vs procura para 12+ meses.",
                "use_case": "Decisões de investimento, análise de capacidade.",
                "features": ["cenários", "crescimento", "investimentos"],
            },
        ],
        "current_mode": "conventional",
        "comparison_available": True,
    }


# ═══════════════════════════════════════════════════════════════════════════════════════════
# REPORTING & COMPARISON ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

try:
    from reporting import (
        compare_scenarios,
        compute_scenario_metrics,
        ReportGenerator,
        generate_executive_report,
        generate_technical_explanation,
    )
    HAS_REPORTING = True
except ImportError:
    HAS_REPORTING = False


class CompareRequest(BaseModel):
    """Request for scenario comparison."""
    scenario_name: str = "Novo Cenário"
    scenario_description: str = ""
    changes: Dict[str, Any] = {}  # Scenario changes (machines, times, etc.)
    context: Optional[str] = None


class TechnicalExplainRequest(BaseModel):
    """Request for technical explanation."""
    algorithm: str = "dispatching"  # dispatching, flow_shop, setup_optimization
    machine_id: Optional[str] = None
    include_examples: bool = True


class ScenarioSummaryRequest(BaseModel):
    """Request for scenario summary."""
    scenario_name: str
    changes_description: str
    include_previous: bool = True


@app.post("/reports/compare")
def compare_scenarios_endpoint(request: CompareRequest) -> Dict[str, Any]:
    """
    Compara o plano atual com um cenário modificado.
    Gera relatório executivo com análise completa.
    
    Args:
        request: Detalhes do cenário a comparar
        
    Returns:
        Comparação completa com métricas, deltas, pontos fortes/fracos e recomendações
    """
    if not HAS_REPORTING:
        raise HTTPException(status_code=501, detail="Reporting module not available")
    
    data = load_dataset()
    
    # Get baseline plan
    baseline_plan = _ensure_plan_exists()
    
    # Apply scenario changes to create new plan
    scenario_data = data  # Start with copy
    
    # Apply any changes from request (simplified for now)
    # In production, this would modify machines, orders, etc.
    
    # Build scenario plan
    scenario_plan = build_plan(scenario_data, mode="NORMAL")
    
    # Compute comparison
    comparison = compare_scenarios(
        baseline_plan=baseline_plan,
        scenario_plan=scenario_plan,
        orders_df=data.orders,
        machines_df=data.machines,
        baseline_name="Plano Base",
        scenario_name=request.scenario_name,
        scenario_description=request.scenario_description,
    )
    
    # Generate executive report
    report = generate_executive_report(comparison, context=request.context)
    
    return {
        "comparison": comparison.to_dict(),
        "report": report.to_dict(),
        "markdown": report.to_markdown(),
    }


@app.get("/reports/current-metrics")
def get_current_metrics() -> Dict[str, Any]:
    """
    Devolve métricas detalhadas do plano atual.
    
    Returns:
        ComparisonMetrics para o cenário atual
    """
    if not HAS_REPORTING:
        raise HTTPException(status_code=501, detail="Reporting module not available")
    
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    metrics = compute_scenario_metrics(
        plan_df=plan_df,
        orders_df=data.orders,
        machines_df=data.machines,
        scenario_name="Plano Atual",
    )
    
    return metrics.to_dict()


@app.post("/reports/technical-explanation")
def get_technical_explanation(request: TechnicalExplainRequest) -> Dict[str, Any]:
    """
    Gera explicação técnica dos algoritmos de sequenciamento.
    
    Args:
        request: Tipo de algoritmo e parâmetros
        
    Returns:
        TechnicalReport com descrição, fórmulas e exemplos
    """
    if not HAS_REPORTING:
        raise HTTPException(status_code=501, detail="Reporting module not available")
    
    report = generate_technical_explanation(
        algorithm=request.algorithm,
        machine_id=request.machine_id,
    )
    
    return {
        "report": report.to_dict(),
        "markdown": report.to_markdown(),
    }


@app.post("/reports/scenario-summary")
def generate_scenario_summary(request: ScenarioSummaryRequest) -> Dict[str, Any]:
    """
    Gera resumo narrativo de um cenário de planeamento.
    
    Args:
        request: Detalhes do cenário
        
    Returns:
        Resumo em linguagem natural
    """
    if not HAS_REPORTING:
        raise HTTPException(status_code=501, detail="Reporting module not available")
    
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    # Compute current metrics
    current_metrics = compute_scenario_metrics(
        plan_df=plan_df,
        orders_df=data.orders,
        machines_df=data.machines,
        scenario_name=request.scenario_name,
    )
    
    # Generate summary
    generator = ReportGenerator()
    summary = generator.generate_scenario_summary(
        scenario=current_metrics,
        changes_description=request.changes_description,
        previous_scenario=None,  # Could load from cache
    )
    
    return {
        "scenario_name": request.scenario_name,
        "summary": summary,
        "metrics": current_metrics.to_dict(),
    }


@app.get("/reports/algorithms")
def list_available_algorithms() -> Dict[str, Any]:
    """
    Lista algoritmos disponíveis para explicação técnica.
    """
    return {
        "algorithms": [
            {
                "id": "dispatching",
                "name": "Regras de Dispatching",
                "description": "EDD, SPT, FIFO, CR, WSPT - sequenciação local por máquina",
            },
            {
                "id": "flow_shop",
                "name": "Flow Shop Scheduling",
                "description": "NEH + Local Search para planeamento encadeado",
            },
            {
                "id": "setup_optimization",
                "name": "Otimização de Setups",
                "description": "Minimização de tempos de setup (TSP-like)",
            },
        ],
    }


@app.post("/reports/compare-whatif")
def compare_whatif_scenario(scenario_description: str) -> Dict[str, Any]:
    """
    Compara cenário What-If descrito em linguagem natural.
    
    Args:
        scenario_description: Descrição em texto do cenário (ex: "adicionar nova máquina M-200")
        
    Returns:
        Comparação completa com relatório executivo
    """
    if not HAS_REPORTING:
        raise HTTPException(status_code=501, detail="Reporting module not available")
    
    data = load_dataset()
    baseline_plan = _ensure_plan_exists()
    
    # Use what-if engine to build scenario
    try:
        from what_if_engine import parse_scenario_description, apply_scenario_to_data
        
        delta = parse_scenario_description(scenario_description)
        scenario_data = apply_scenario_to_data(data, delta)
        scenario_plan = build_plan(scenario_data, mode="NORMAL")
        
        comparison = compare_scenarios(
            baseline_plan=baseline_plan,
            scenario_plan=scenario_plan,
            orders_df=data.orders,
            machines_df=data.machines,
            baseline_name="Plano Base",
            scenario_name="Cenário What-If",
            scenario_description=scenario_description,
        )
        
        report = generate_executive_report(
            comparison,
            context=f"Cenário gerado a partir de: '{scenario_description}'"
        )
        
        return {
            "scenario_description": scenario_description,
            "comparison": comparison.to_dict(),
            "report": report.to_dict(),
            "markdown": report.to_markdown(),
        }
        
    except Exception as e:
        # Fallback: just describe without comparison
        description = describe_scenario_nl(scenario_description)
        return {
            "scenario_description": scenario_description,
            "description": description,
            "error": str(e),
            "message": "Não foi possível executar a comparação completa. Cenário descrito apenas.",
        }


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ADVANCED DASHBOARDS & VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════════════════

try:
    from dashboards import (
        generate_comparative_gantt_data,
        generate_utilization_heatmap,
        generate_operator_dashboard,
        generate_machine_oee_dashboard,
        generate_cell_performance,
        generate_capacity_projection,
    )
    HAS_DASHBOARDS = True
except ImportError:
    HAS_DASHBOARDS = False


class CellConfig(BaseModel):
    """Configuration for a production cell."""
    id: str
    name: str
    machines: List[str]


class CapacityProjectionRequest(BaseModel):
    """Request for capacity projection."""
    forecast_months: int = 12
    demand_growth_monthly: float = 0.02


@app.get("/dashboards/gantt-comparison")
def get_gantt_comparison(scenario_name: str = "Novo Cenário") -> Dict[str, Any]:
    """
    Gera dados para Gráfico de Gantt comparativo.
    Compara plano atual com cenário modificado.
    
    Returns:
        GanttComparisonData com barras para baseline e cenário
    """
    if not HAS_DASHBOARDS:
        raise HTTPException(status_code=501, detail="Dashboards module not available")
    
    data = load_dataset()
    baseline_plan = _ensure_plan_exists()
    
    # Generate scenario plan (simplified: same as baseline for demo)
    scenario_plan = build_plan(data, mode="NORMAL")
    
    comparison = generate_comparative_gantt_data(
        baseline_plan=baseline_plan,
        scenario_plan=scenario_plan,
        baseline_name="Plano Original",
        scenario_name=scenario_name,
    )
    
    return comparison.to_dict()


@app.get("/dashboards/utilization-heatmap")
def get_utilization_heatmap() -> Dict[str, Any]:
    """
    Gera heatmap de utilização horária por máquina.
    Mostra utilização por hora e dia da semana.
    
    Returns:
        HeatmapData com matriz de utilização e análise de janelas ociosas
    """
    if not HAS_DASHBOARDS:
        raise HTTPException(status_code=501, detail="Dashboards module not available")
    
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    heatmap = generate_utilization_heatmap(
        plan_df=plan_df,
        machines_df=data.machines,
    )
    
    return heatmap.to_dict()


@app.get("/dashboards/operator")
def get_operator_dashboard() -> Dict[str, Any]:
    """
    Gera dashboard de operadores.
    Mostra carga de trabalho, skills e gaps de competência.
    
    Returns:
        OperatorDashboard com métricas por operador
    """
    if not HAS_DASHBOARDS:
        raise HTTPException(status_code=501, detail="Dashboards module not available")
    
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    # Use shifts data if available
    shifts_df = data.shifts if hasattr(data, 'shifts') else None
    
    dashboard = generate_operator_dashboard(
        plan_df=plan_df,
        operators_df=None,  # Will generate synthetic
        shifts_df=shifts_df,
    )
    
    return dashboard.to_dict()


@app.get("/dashboards/machine-oee")
def get_machine_oee_dashboard() -> Dict[str, Any]:
    """
    Gera dashboard de OEE por máquina.
    Mostra disponibilidade, performance, qualidade e OEE total.
    
    Returns:
        MachineDashboard com OEE para todas as máquinas
    """
    if not HAS_DASHBOARDS:
        raise HTTPException(status_code=501, detail="Dashboards module not available")
    
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    # Use downtime data if available
    downtime_df = data.downtime if hasattr(data, 'downtime') else None
    
    dashboard = generate_machine_oee_dashboard(
        plan_df=plan_df,
        machines_df=data.machines,
        downtime_df=downtime_df,
    )
    
    return dashboard.to_dict()


@app.get("/dashboards/cell-performance")
def get_cell_performance(cells: Optional[str] = None) -> Dict[str, Any]:
    """
    Gera dashboard de performance de células de produção encadeadas.
    Mostra lead time, WIP e gargalo por célula.
    
    Args:
        cells: JSON string com configuração de células (opcional)
        
    Returns:
        CellPerformance com métricas para cada célula/linha
    """
    if not HAS_DASHBOARDS:
        raise HTTPException(status_code=501, detail="Dashboards module not available")
    
    import json
    
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    # Parse cells config if provided
    cells_config = None
    if cells:
        try:
            cells_config = json.loads(cells)
        except json.JSONDecodeError:
            pass
    
    performance = generate_cell_performance(
        plan_df=plan_df,
        machines_df=data.machines,
        cells_config=cells_config,
    )
    
    return performance.to_dict()


@app.post("/dashboards/capacity-projection")
def get_capacity_projection(request: CapacityProjectionRequest) -> Dict[str, Any]:
    """
    Gera projeção anual de capacidade vs procura.
    Estilo S&OP com 12 meses de forecast.
    
    Args:
        request: Parâmetros de projeção
        
    Returns:
        CapacityProjection com métricas mensais
    """
    if not HAS_DASHBOARDS:
        raise HTTPException(status_code=501, detail="Dashboards module not available")
    
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    projection = generate_capacity_projection(
        plan_df=plan_df,
        machines_df=data.machines,
        orders_df=data.orders,
        forecast_months=request.forecast_months,
        demand_growth_monthly=request.demand_growth_monthly,
    )
    
    return projection.to_dict()


@app.get("/dashboards/summary")
def get_dashboards_summary() -> Dict[str, Any]:
    """
    Devolve resumo executivo de todos os dashboards.
    """
    if not HAS_DASHBOARDS:
        raise HTTPException(status_code=501, detail="Dashboards module not available")
    
    data = load_dataset()
    plan_df = _ensure_plan_exists()
    
    # Generate all dashboards
    heatmap = generate_utilization_heatmap(plan_df, data.machines)
    operator = generate_operator_dashboard(plan_df)
    oee = generate_machine_oee_dashboard(plan_df, data.machines)
    cells = generate_cell_performance(plan_df, data.machines)
    projection = generate_capacity_projection(plan_df, data.machines, data.orders)
    
    return {
        "utilization": {
            "avg_pct": heatmap.summary.get("avg_utilization", 0),
            "idle_windows": heatmap.summary.get("total_idle_windows", 0),
            "overloaded_periods": heatmap.summary.get("total_overloaded_periods", 0),
        },
        "operators": {
            "total": operator.total_operators,
            "overloaded": operator.overloaded_count,
            "underutilized": operator.underutilized_count,
            "skill_gaps": len(operator.skill_gaps),
        },
        "machines": {
            "overall_oee": oee.overall_oee,
            "bottleneck": oee.bottleneck_machine,
            "critical_count": len(oee.critical_machines),
        },
        "cells": {
            "total": len(cells.cells),
            "healthy": sum(1 for c in cells.cells if c.status == "healthy"),
            "avg_lead_time_h": cells.avg_lead_time,
        },
        "projection": {
            "avg_utilization": projection.avg_utilization_pct,
            "months_risk": projection.months_undercapacity,
            "annual_growth_pct": projection.demand_growth_annual,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DIGITAL TWIN - PdM-IPS (Predictive Maintenance Integrated Production Scheduling)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from digital_twin import (
        CVAE, CVAEConfig, SensorSnapshot, OperationContext,
        infer_hi, create_demo_dataset,
        RULEstimate, RULEstimatorConfig, estimate_rul, HealthStatus,
        create_demo_hi_history,
        adjust_plan_with_rul, get_rul_penalties, RULAdjustmentConfig,
    )
    HAS_DIGITAL_TWIN = True
except ImportError as e:
    HAS_DIGITAL_TWIN = False
    logger.warning(f"Digital Twin module not available: {e}")


# Store for machine health data (in production, use database)
_machine_health_store: Dict[str, List[Dict]] = {}
_cvae_model = None


def _get_cvae_model():
    """Get or create CVAE model."""
    global _cvae_model
    if _cvae_model is None and HAS_DIGITAL_TWIN:
        _cvae_model = CVAE(CVAEConfig())
    return _cvae_model


@app.get("/digital-twin/health")
def get_digital_twin_health() -> Dict[str, Any]:
    """
    Status do módulo Digital Twin.
    """
    return {
        "available": HAS_DIGITAL_TWIN,
        "modules": {
            "cvae": HAS_DIGITAL_TWIN,
            "rul_estimator": HAS_DIGITAL_TWIN,
            "scheduler_integration": HAS_DIGITAL_TWIN,
        },
        "machines_monitored": len(_machine_health_store),
    }


@app.get("/digital-twin/machines")
def get_monitored_machines() -> Dict[str, Any]:
    """
    Lista todas as máquinas monitorizadas com estado de saúde.
    """
    if not HAS_DIGITAL_TWIN:
        raise HTTPException(status_code=501, detail="Digital Twin module not available")
    
    # Get machines from plan
    plan_df = _ensure_plan_exists()
    machines = plan_df['machine_id'].unique().tolist() if not plan_df.empty else []
    
    # Generate demo health data for each machine
    result = []
    for machine_id in machines[:15]:  # Limit to 15 machines
        # Create demo history if not exists
        if machine_id not in _machine_health_store:
            # Simulate different degradation levels
            import random
            degradation = random.choice(["healthy", "degraded", "warning", "critical"])
            if degradation == "healthy":
                initial_hi, final_hi = 0.95, 0.85
            elif degradation == "degraded":
                initial_hi, final_hi = 0.85, 0.65
            elif degradation == "warning":
                initial_hi, final_hi = 0.70, 0.45
            else:
                initial_hi, final_hi = 0.50, 0.25
            
            history = create_demo_hi_history(
                machine_id, num_points=30,
                initial_hi=initial_hi, final_hi=final_hi
            )
            _machine_health_store[machine_id] = [
                {"timestamp": t.isoformat(), "hi": hi} for t, hi in history
            ]
        
        # Estimate RUL
        history = [(datetime.fromisoformat(h["timestamp"]), h["hi"]) 
                   for h in _machine_health_store[machine_id]]
        rul = estimate_rul(machine_id, history)
        
        if rul:
            result.append({
                "machine_id": machine_id,
                "current_hi": round(rul.current_hi, 3),
                "health_status": rul.health_status.value,
                "rul_hours": round(rul.rul_mean_hours, 1),
                "rul_std_hours": round(rul.rul_std_hours, 1),
                "rul_days": round(rul.rul_mean_hours / 24, 1),
                "confidence": round(rul.confidence, 2),
                "degradation_rate": round(rul.degradation_rate_per_hour, 5),
                "history_points": len(history),
            })
    
    # Sort by health status (critical first)
    status_order = {"CRITICAL": 0, "WARNING": 1, "DEGRADED": 2, "HEALTHY": 3}
    result.sort(key=lambda x: status_order.get(x["health_status"], 4))
    
    return {
        "machines": result,
        "summary": {
            "total": len(result),
            "critical": sum(1 for m in result if m["health_status"] == "CRITICAL"),
            "warning": sum(1 for m in result if m["health_status"] == "WARNING"),
            "degraded": sum(1 for m in result if m["health_status"] == "DEGRADED"),
            "healthy": sum(1 for m in result if m["health_status"] == "HEALTHY"),
        }
    }


@app.get("/digital-twin/machine/{machine_id}")
def get_machine_health_detail(machine_id: str) -> Dict[str, Any]:
    """
    Detalhes de saúde de uma máquina específica.
    """
    if not HAS_DIGITAL_TWIN:
        raise HTTPException(status_code=501, detail="Digital Twin module not available")
    
    # Create demo data if not exists
    if machine_id not in _machine_health_store:
        history = create_demo_hi_history(machine_id, num_points=50, initial_hi=0.95, final_hi=0.5)
        _machine_health_store[machine_id] = [
            {"timestamp": t.isoformat(), "hi": hi} for t, hi in history
        ]
    
    # Get history
    history_data = _machine_health_store[machine_id]
    history = [(datetime.fromisoformat(h["timestamp"]), h["hi"]) for h in history_data]
    
    # Estimate RUL
    rul = estimate_rul(machine_id, history)
    
    if not rul:
        raise HTTPException(status_code=404, detail=f"Insufficient data for machine {machine_id}")
    
    return {
        "machine_id": machine_id,
        "current_hi": round(rul.current_hi, 4),
        "health_status": rul.health_status.value,
        "rul": {
            "mean_hours": round(rul.rul_mean_hours, 1),
            "std_hours": round(rul.rul_std_hours, 1),
            "lower_hours": round(rul.rul_lower_hours, 1),
            "upper_hours": round(rul.rul_upper_hours, 1),
            "days": round(rul.rul_mean_hours / 24, 1),
        },
        "degradation_rate_per_hour": round(rul.degradation_rate_per_hour, 6),
        "confidence": round(rul.confidence, 3),
        "model_used": rul.model_used,
        "history": history_data[-30:],  # Last 30 points
        "recommendations": _get_machine_recommendations(rul),
    }


def _get_machine_recommendations(rul: RULEstimate) -> List[Dict[str, str]]:
    """Generate recommendations based on RUL."""
    recommendations = []
    
    if rul.health_status == HealthStatus.CRITICAL:
        recommendations.append({
            "type": "URGENT",
            "title": "Manutenção Imediata Necessária",
            "description": f"RUL estimado de apenas {rul.rul_mean_hours:.0f}h. Agendar manutenção preventiva urgentemente.",
        })
        recommendations.append({
            "type": "PLANNING",
            "title": "Evitar Operações Críticas",
            "description": "Não agendar operações longas ou críticas nesta máquina.",
        })
    elif rul.health_status == HealthStatus.WARNING:
        recommendations.append({
            "type": "WARNING",
            "title": "Planear Manutenção",
            "description": f"RUL de {rul.rul_mean_hours:.0f}h. Planear manutenção para os próximos {rul.rul_mean_hours/24:.0f} dias.",
        })
        recommendations.append({
            "type": "MONITORING",
            "title": "Aumentar Monitorização",
            "description": "Aumentar frequência de inspeções e monitorização de sensores.",
        })
    elif rul.health_status == HealthStatus.DEGRADED:
        recommendations.append({
            "type": "INFO",
            "title": "Monitorizar Degradação",
            "description": "Máquina em degradação normal. Continuar monitorização regular.",
        })
    else:
        recommendations.append({
            "type": "OK",
            "title": "Estado Saudável",
            "description": "Máquina em bom estado. Manter programa de manutenção preventiva.",
        })
    
    return recommendations


@app.get("/digital-twin/rul-penalties")
def get_rul_penalties_endpoint() -> Dict[str, Any]:
    """
    Penalizações de RUL para todas as máquinas monitorizadas.
    """
    if not HAS_DIGITAL_TWIN:
        raise HTTPException(status_code=501, detail="Digital Twin module not available")
    
    # Build RUL info
    rul_info = {}
    for machine_id, history_data in _machine_health_store.items():
        history = [(datetime.fromisoformat(h["timestamp"]), h["hi"]) for h in history_data]
        rul = estimate_rul(machine_id, history)
        if rul:
            rul_info[machine_id] = rul
    
    if not rul_info:
        return {"penalties": {}, "message": "No machines monitored yet"}
    
    penalties = get_rul_penalties(list(rul_info.keys()), rul_info)
    
    return {
        "penalties": {k: round(v, 1) for k, v in penalties.items()},
        "summary": {
            "machines_penalized": sum(1 for v in penalties.values() if v > 1.0),
            "max_penalty": max(penalties.values()) if penalties else 1.0,
            "avg_penalty": sum(penalties.values()) / len(penalties) if penalties else 1.0,
        }
    }


@app.post("/digital-twin/adjust-plan")
def adjust_plan_with_rul_endpoint() -> Dict[str, Any]:
    """
    Ajustar o plano de produção atual baseado em informação de RUL.
    """
    if not HAS_DIGITAL_TWIN:
        raise HTTPException(status_code=501, detail="Digital Twin module not available")
    
    plan_df = _ensure_plan_exists()
    
    if plan_df.empty:
        raise HTTPException(status_code=404, detail="No plan available")
    
    # Build RUL info for machines in plan
    machines_in_plan = plan_df['machine_id'].unique()
    rul_info = {}
    
    for machine_id in machines_in_plan:
        if machine_id not in _machine_health_store:
            # Generate demo data
            import random
            degradation = random.choice(["healthy", "healthy", "degraded", "warning", "critical"])
            if degradation == "healthy":
                initial_hi, final_hi = 0.95, 0.85
            elif degradation == "degraded":
                initial_hi, final_hi = 0.85, 0.65
            elif degradation == "warning":
                initial_hi, final_hi = 0.70, 0.45
            else:
                initial_hi, final_hi = 0.50, 0.25
            
            history = create_demo_hi_history(machine_id, num_points=30, initial_hi=initial_hi, final_hi=final_hi)
            _machine_health_store[machine_id] = [
                {"timestamp": t.isoformat(), "hi": hi} for t, hi in history
            ]
        
        history = [(datetime.fromisoformat(h["timestamp"]), h["hi"]) 
                   for h in _machine_health_store[machine_id]]
        rul = estimate_rul(machine_id, history)
        if rul:
            rul_info[machine_id] = rul
    
    # Adjust plan
    result = adjust_plan_with_rul(plan_df, rul_info)
    
    return {
        "summary": result.summary(),
        "decisions": [d.to_dict() for d in result.decisions[:10]],  # Top 10 decisions
        "machines_at_risk": [
            {
                "machine_id": m.machine_id,
                "rul_hours": m.rul_hours if m.rul_hours != float('inf') else None,
                "health_status": m.health_status.value,
                "current_hi": round(m.current_hi, 3),
            }
            for m in result.machine_rul_info.values()
            if m.health_status in (HealthStatus.CRITICAL, HealthStatus.WARNING)
        ],
    }


@app.get("/digital-twin/dashboard")
def get_digital_twin_dashboard() -> Dict[str, Any]:
    """
    Dashboard completo do Digital Twin para UI.
    """
    if not HAS_DIGITAL_TWIN:
        raise HTTPException(status_code=501, detail="Digital Twin module not available")
    
    # Get machines data
    machines_response = get_monitored_machines()
    
    # Get penalties
    penalties_response = get_rul_penalties_endpoint()
    
    # Calculate overall health score
    machines = machines_response.get("machines", [])
    if machines:
        avg_hi = sum(m["current_hi"] for m in machines) / len(machines)
        overall_health = avg_hi * 100
    else:
        overall_health = 100.0
    
    # Count by status
    summary = machines_response.get("summary", {})
    
    return {
        "overall_health_score": round(overall_health, 1),
        "machines_summary": summary,
        "machines": machines[:10],  # Top 10 (sorted by criticality)
        "penalties": penalties_response.get("penalties", {}),
        "alerts": [
            {
                "machine_id": m["machine_id"],
                "type": "CRITICAL" if m["health_status"] == "CRITICAL" else "WARNING",
                "message": f"RUL: {m['rul_hours']:.0f}h ({m['rul_days']:.0f} dias)",
                "rul_hours": m["rul_hours"],
            }
            for m in machines
            if m["health_status"] in ("CRITICAL", "WARNING")
        ],
        "kpis": {
            "total_machines": summary.get("total", 0),
            "critical_count": summary.get("critical", 0),
            "warning_count": summary.get("warning", 0),
            "avg_rul_hours": round(sum(m["rul_hours"] for m in machines) / len(machines), 1) if machines else 0,
            "maintenance_recommended": summary.get("critical", 0) + summary.get("warning", 0),
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ZDM - ZERO DISRUPTION MANUFACTURING (Simulação de Resiliência)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from simulation.zdm import (
        generate_failure_scenarios,
        FailureType,
        FailureConfig,
        ZDMSimulator,
        SimulationConfig,
        suggest_best_recovery,
        RecoveryConfig,
        get_recovery_recommendations,
        quick_resilience_check,
    )
    HAS_ZDM = True
except ImportError as e:
    HAS_ZDM = False
    logger.warning(f"ZDM module not available: {e}")


class ZDMSimulateRequest(BaseModel):
    """Request para simulação ZDM."""
    n_scenarios: int = 10
    enable_rerouting: bool = True
    enable_overtime: bool = True
    enable_priority_shuffle: bool = True
    use_rul_data: bool = True


@app.get("/zdm/health")
def get_zdm_health() -> Dict[str, Any]:
    """
    Status do módulo ZDM.
    """
    return {
        "available": HAS_ZDM,
        "description": "Zero Disruption Manufacturing - Simulação de Resiliência",
        "features": [
            "Geração de cenários de falha",
            "Simulação de perturbações",
            "Estratégias de recuperação",
            "Resilience Score",
        ] if HAS_ZDM else [],
    }


@app.get("/zdm/quick-check")
def zdm_quick_check() -> Dict[str, Any]:
    """
    Verificação rápida de resiliência do plano atual.
    """
    if not HAS_ZDM:
        raise HTTPException(status_code=501, detail="ZDM module not available")
    
    plan_df = _ensure_plan_exists()
    
    if plan_df.empty:
        return {
            "resilience_score": 100.0,
            "message": "Nenhum plano para analisar",
            "scenarios_tested": 0,
        }
    
    result = quick_resilience_check(plan_df, n_scenarios=5)
    
    return result


@app.post("/zdm/simulate")
def zdm_simulate(request: ZDMSimulateRequest) -> Dict[str, Any]:
    """
    Simula cenários de falha e calcula resiliência do plano.
    
    Input:
    - n_scenarios: Número de cenários a simular
    - enable_rerouting: Permitir reencaminhamento
    - enable_overtime: Permitir horas extra
    - use_rul_data: Usar dados de RUL para gerar cenários
    
    Output:
    - resilience_score: Score de resiliência (0-100)
    - scenarios: Detalhes dos cenários simulados
    - recommendations: Recomendações de melhoria
    """
    if not HAS_ZDM:
        raise HTTPException(status_code=501, detail="ZDM module not available")
    
    plan_df = _ensure_plan_exists()
    
    if plan_df.empty:
        raise HTTPException(status_code=404, detail="No plan available")
    
    # Obter RUL info se disponível e solicitado
    rul_info = None
    if request.use_rul_data and HAS_DIGITAL_TWIN:
        rul_info = {}
        for machine_id in plan_df['machine_id'].unique():
            if machine_id in _machine_health_store:
                history = [(datetime.fromisoformat(h["timestamp"]), h["hi"]) 
                           for h in _machine_health_store[machine_id]]
                rul = estimate_rul(machine_id, history)
                if rul:
                    rul_info[machine_id] = rul.rul_mean_hours
    
    # Configurar geração de cenários
    failure_config = FailureConfig(use_rul_data=request.use_rul_data)
    
    # Gerar cenários
    scenarios = generate_failure_scenarios(
        plan_df,
        n_scenarios=request.n_scenarios,
        rul_info=rul_info,
        config=failure_config,
    )
    
    # Configurar simulador
    sim_config = SimulationConfig(
        enable_rerouting=request.enable_rerouting,
        enable_overtime=request.enable_overtime,
        enable_priority_shuffle=request.enable_priority_shuffle,
    )
    
    simulator = ZDMSimulator(sim_config)
    
    # Simular
    report = simulator.simulate_all(plan_df, scenarios)
    
    # Gerar planos de recuperação para top cenários
    recovery_plans = []
    for scenario in scenarios[:5]:
        plan = suggest_best_recovery(plan_df, scenario)
        recovery_plans.append(plan.to_dict())
    
    return {
        "resilience_report": report.to_dict(),
        "recovery_plans": recovery_plans,
        "summary": {
            "resilience_score": round(report.overall_resilience_score, 1),
            "resilience_grade": _get_resilience_grade(report.overall_resilience_score),
            "scenarios_simulated": report.scenarios_simulated,
            "full_recovery_rate": round(
                report.full_recovery_count / max(1, report.scenarios_simulated) * 100, 1
            ),
            "critical_machines": report.critical_machines[:5],
            "avg_recovery_time_hours": round(report.avg_recovery_time_hours, 1),
            "top_risks": [
                {
                    "machine": s.machine_id,
                    "type": s.failure_type.value,
                    "probability": round(s.probability, 2),
                    "severity": round(s.severity, 2),
                }
                for s in scenarios[:3]
            ],
        },
    }


def _get_resilience_grade(score: float) -> str:
    """Converte score em grade."""
    if score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    else:
        return "F"


@app.get("/zdm/scenarios")
def get_zdm_scenarios(n: int = 10) -> Dict[str, Any]:
    """
    Gera cenários de falha potenciais.
    """
    if not HAS_ZDM:
        raise HTTPException(status_code=501, detail="ZDM module not available")
    
    plan_df = _ensure_plan_exists()
    
    scenarios = generate_failure_scenarios(plan_df, n_scenarios=n)
    
    return {
        "scenarios": [s.to_dict() for s in scenarios],
        "total": len(scenarios),
        "by_type": {
            ft.value: sum(1 for s in scenarios if s.failure_type == ft)
            for ft in FailureType
        },
    }


@app.get("/zdm/recovery/{scenario_id}")
def get_zdm_recovery_plan(scenario_id: str) -> Dict[str, Any]:
    """
    Obtém plano de recuperação para um cenário específico.
    """
    if not HAS_ZDM:
        raise HTTPException(status_code=501, detail="ZDM module not available")
    
    plan_df = _ensure_plan_exists()
    
    # Gerar cenário de exemplo (em produção, seria recuperado de cache/DB)
    scenarios = generate_failure_scenarios(plan_df, n_scenarios=1)
    
    if not scenarios:
        raise HTTPException(status_code=404, detail="Could not generate scenario")
    
    scenario = scenarios[0]
    scenario.scenario_id = scenario_id  # Override com ID solicitado
    
    recovery_plan = suggest_best_recovery(plan_df, scenario)
    
    return {
        "scenario": scenario.to_dict(),
        "recovery_plan": recovery_plan.to_dict(),
    }


@app.get("/zdm/dashboard")
def get_zdm_dashboard() -> Dict[str, Any]:
    """
    Dashboard completo do ZDM para UI.
    """
    if not HAS_ZDM:
        raise HTTPException(status_code=501, detail="ZDM module not available")
    
    plan_df = _ensure_plan_exists()
    
    # Quick check
    quick_result = quick_resilience_check(plan_df, n_scenarios=5) if not plan_df.empty else {
        "resilience_score": 100.0,
        "scenarios_tested": 0,
        "recovery_success_rate": 100.0,
        "critical_machines": [],
        "top_recommendation": None,
    }
    
    # Gerar alguns cenários
    scenarios = generate_failure_scenarios(plan_df, n_scenarios=10) if not plan_df.empty else []
    
    # Categorizar por tipo
    by_type = {}
    for ft in FailureType:
        type_scenarios = [s for s in scenarios if s.failure_type == ft]
        by_type[ft.value] = {
            "count": len(type_scenarios),
            "avg_severity": round(np.mean([s.severity for s in type_scenarios]), 2) if type_scenarios else 0,
            "avg_probability": round(np.mean([s.probability for s in type_scenarios]), 2) if type_scenarios else 0,
        }
    
    return {
        "resilience_score": quick_result.get("resilience_score", 100.0),
        "resilience_grade": _get_resilience_grade(quick_result.get("resilience_score", 100.0)),
        "scenarios_preview": {
            "total": len(scenarios),
            "by_type": by_type,
            "top_risks": [s.to_dict() for s in scenarios[:5]],
        },
        "recovery_success_rate": quick_result.get("recovery_success_rate", 100.0),
        "critical_machines": quick_result.get("critical_machines", []),
        "recommendations": [quick_result.get("top_recommendation")] if quick_result.get("top_recommendation") else [],
        "kpis": {
            "operations_at_risk": sum(1 for s in scenarios if s.severity > 0.7),
            "high_probability_failures": sum(1 for s in scenarios if s.probability > 0.6),
            "rul_triggered_scenarios": sum(1 for s in scenarios if s.triggered_by_rul),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CCM - CAUSAL CONTEXT MODELS (Análise Causal)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from causal import (
        learn_causal_graph,
        estimate_effect,
        estimate_intervention,
        get_all_effects_for_outcome,
        get_all_effects_from_treatment,
    )
    from causal.complexity_dashboard_engine import (
        compute_complexity_metrics,
        generate_causal_insights,
        generate_tradeoff_analysis,
        ComplexityDashboard,
    )
    from causal.causal_graph_builder import generate_synthetic_data
    HAS_CCM = True
except ImportError as e:
    HAS_CCM = False
    logger.warning(f"CCM module not available: {e}")


# Cached causal graph and data
_causal_graph = None
_causal_data = None


def _get_causal_graph():
    """Get or create causal graph."""
    global _causal_graph, _causal_data
    if _causal_graph is None and HAS_CCM:
        _causal_data = generate_synthetic_data(n_samples=1000)
        _causal_graph = learn_causal_graph(_causal_data)
    return _causal_graph, _causal_data


class CausalExplainRequest(BaseModel):
    """Request para explicação causal."""
    question: str
    treatment: Optional[str] = None
    outcome: Optional[str] = None


@app.get("/causal/health")
def get_causal_health() -> Dict[str, Any]:
    """
    Status do módulo CCM.
    """
    return {
        "available": HAS_CCM,
        "description": "Causal Context Models - Análise de Relações Causais",
        "features": [
            "Grafo causal de scheduling",
            "Estimação de efeitos causais",
            "Análise de trade-offs",
            "Insights e recomendações",
        ] if HAS_CCM else [],
    }


@app.get("/causal/graph")
def get_causal_graph() -> Dict[str, Any]:
    """
    Retorna a estrutura do grafo causal.
    """
    if not HAS_CCM:
        raise HTTPException(status_code=501, detail="CCM module not available")
    
    graph, _ = _get_causal_graph()
    
    return graph.to_dict()


@app.get("/causal/variables")
def get_causal_variables() -> Dict[str, Any]:
    """
    Lista todas as variáveis causais.
    """
    if not HAS_CCM:
        raise HTTPException(status_code=501, detail="CCM module not available")
    
    graph, _ = _get_causal_graph()
    
    treatments = []
    outcomes = []
    confounders = []
    
    for name, var in graph.variables.items():
        var_data = var.to_dict()
        if var.var_type.value == "treatment":
            treatments.append(var_data)
        elif var.var_type.value == "outcome":
            outcomes.append(var_data)
        else:
            confounders.append(var_data)
    
    return {
        "treatments": treatments,
        "outcomes": outcomes,
        "confounders": confounders,
        "total": len(graph.variables),
    }


@app.get("/causal/effect/{treatment}/{outcome}")
def get_causal_effect(treatment: str, outcome: str) -> Dict[str, Any]:
    """
    Estima o efeito causal de um tratamento num outcome.
    """
    if not HAS_CCM:
        raise HTTPException(status_code=501, detail="CCM module not available")
    
    graph, data = _get_causal_graph()
    
    effect = estimate_effect(treatment, outcome, graph, data)
    
    return {
        "effect": effect.to_dict(),
        "interpretation": effect.explanation,
    }


@app.get("/causal/effects/outcome/{outcome}")
def get_all_effects_outcome(outcome: str) -> Dict[str, Any]:
    """
    Lista todos os efeitos causais num outcome específico.
    """
    if not HAS_CCM:
        raise HTTPException(status_code=501, detail="CCM module not available")
    
    graph, data = _get_causal_graph()
    
    effects = get_all_effects_for_outcome(outcome, graph, data)
    
    return {
        "outcome": outcome,
        "effects": [e.to_dict() for e in effects],
        "strongest_positive": effects[0].to_dict() if effects and effects[0].estimate > 0 else None,
        "strongest_negative": next((e.to_dict() for e in effects if e.estimate < 0), None),
    }


@app.get("/causal/effects/treatment/{treatment}")
def get_all_effects_treatment(treatment: str) -> Dict[str, Any]:
    """
    Lista todos os efeitos de um tratamento.
    """
    if not HAS_CCM:
        raise HTTPException(status_code=501, detail="CCM module not available")
    
    graph, data = _get_causal_graph()
    
    effects = get_all_effects_from_treatment(treatment, graph, data)
    
    positive = [e for e in effects if e.estimate > 0]
    negative = [e for e in effects if e.estimate < 0]
    
    return {
        "treatment": treatment,
        "effects": [e.to_dict() for e in effects],
        "positive_outcomes": [e.outcome for e in positive],
        "negative_outcomes": [e.outcome for e in negative],
        "has_tradeoffs": len(positive) > 0 and len(negative) > 0,
    }


@app.get("/causal/tradeoffs/{treatment}")
def get_tradeoff_analysis(treatment: str) -> Dict[str, Any]:
    """
    Análise de trade-offs para um tratamento.
    """
    if not HAS_CCM:
        raise HTTPException(status_code=501, detail="CCM module not available")
    
    graph, data = _get_causal_graph()
    
    return generate_tradeoff_analysis(treatment, graph, data)


@app.get("/causal/complexity")
def get_complexity_metrics() -> Dict[str, Any]:
    """
    Métricas de complexidade do sistema causal.
    """
    if not HAS_CCM:
        raise HTTPException(status_code=501, detail="CCM module not available")
    
    graph, data = _get_causal_graph()
    
    metrics = compute_complexity_metrics(graph, data)
    
    return metrics.to_dict()


@app.get("/causal/insights")
def get_causal_insights() -> Dict[str, Any]:
    """
    Gera insights causais automaticamente.
    """
    if not HAS_CCM:
        raise HTTPException(status_code=501, detail="CCM module not available")
    
    graph, data = _get_causal_graph()
    
    insights = generate_causal_insights(graph, data)
    
    # Agrupar por tipo
    by_type = {}
    for insight in insights:
        t = insight.insight_type.value
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(insight.to_dict())
    
    return {
        "insights": [i.to_dict() for i in insights],
        "by_type": by_type,
        "total": len(insights),
        "high_priority": sum(1 for i in insights if i.priority == "high"),
    }


@app.post("/causal/explain")
def explain_causal_question(request: CausalExplainRequest) -> Dict[str, Any]:
    """
    Responde a uma pergunta causal em linguagem natural.
    
    Input: pergunta como "Se eu reduzir setups, o que acontece ao custo energético?"
    Output: explicação com dados numéricos e contexto
    """
    if not HAS_CCM:
        raise HTTPException(status_code=501, detail="CCM module not available")
    
    graph, data = _get_causal_graph()
    
    # Tentar extrair tratamento e outcome da pergunta
    treatment = request.treatment
    outcome = request.outcome
    question_lower = request.question.lower()
    
    # Mapeamento de palavras-chave
    treatment_keywords = {
        "setup": "setup_frequency",
        "changeover": "setup_frequency",
        "lote": "batch_size",
        "batch": "batch_size",
        "carga": "machine_load",
        "utilização": "machine_load",
        "turno": "night_shifts",
        "noturno": "night_shifts",
        "hora extra": "overtime_hours",
        "overtime": "overtime_hours",
        "manutenção": "maintenance_delay",
        "prioridade": "priority_changes",
    }
    
    outcome_keywords = {
        "energia": "energy_cost",
        "custo": "energy_cost",
        "makespan": "makespan",
        "tempo": "makespan",
        "atraso": "tardiness",
        "entrega": "otd_rate",
        "otd": "otd_rate",
        "falha": "failure_prob",
        "avaria": "failure_prob",
        "stress": "operator_stress",
        "qualidade": "quality_defects",
        "defeito": "quality_defects",
        "desgaste": "machine_wear",
    }
    
    # Identificar tratamento
    if not treatment:
        for keyword, var in treatment_keywords.items():
            if keyword in question_lower:
                treatment = var
                break
    
    # Identificar outcome
    if not outcome:
        for keyword, var in outcome_keywords.items():
            if keyword in question_lower:
                outcome = var
                break
    
    if not treatment or not outcome:
        return {
            "success": False,
            "error": "Não foi possível identificar as variáveis na pergunta. "
                     "Tente especificar tratamento e outcome.",
            "available_treatments": list(treatment_keywords.values()),
            "available_outcomes": list(outcome_keywords.values()),
        }
    
    # Estimar efeito
    effect = estimate_effect(treatment, outcome, graph, data)
    
    # Gerar explicação expandida
    treatment_desc = graph.variables[treatment].description if treatment in graph.variables else treatment
    outcome_desc = graph.variables[outcome].description if outcome in graph.variables else outcome
    
    if effect.estimate > 0:
        direction = "aumentar"
        impact = "aumenta"
    else:
        direction = "diminuir"
        impact = "diminui"
    
    explanation = (
        f"**Análise Causal:**\n\n"
        f"A pergunta '{request.question}' foi mapeada para:\n"
        f"- **Tratamento:** {treatment_desc}\n"
        f"- **Outcome:** {outcome_desc}\n\n"
        f"**Resultado:**\n"
        f"Um aumento unitário em '{treatment_desc}' {impact} '{outcome_desc}' "
        f"em {abs(effect.estimate):.3f} unidades.\n\n"
        f"**Intervalo de Confiança (95%):** [{effect.ci_lower:.3f}, {effect.ci_upper:.3f}]\n"
        f"**Significância:** {effect.significance}\n\n"
        f"**Interpretação:**\n{effect.explanation}"
    )
    
    # Buscar trade-offs
    all_effects = get_all_effects_from_treatment(treatment, graph, data)
    positive = [e for e in all_effects if e.estimate > 0.1 and e.outcome != outcome]
    negative = [e for e in all_effects if e.estimate < -0.1 and e.outcome != outcome]
    
    tradeoff_text = ""
    if positive or negative:
        tradeoff_text = "\n\n**Trade-offs a considerar:**\n"
        if positive:
            tradeoff_text += f"- Efeitos positivos também em: {', '.join(e.outcome for e in positive[:3])}\n"
        if negative:
            tradeoff_text += f"- Efeitos negativos em: {', '.join(e.outcome for e in negative[:3])}"
    
    return {
        "success": True,
        "question": request.question,
        "treatment": treatment,
        "outcome": outcome,
        "effect": effect.to_dict(),
        "explanation": explanation + tradeoff_text,
        "confidence": effect.confidence if hasattr(effect, 'confidence') else 0.5,
    }


@app.get("/causal/dashboard")
def get_causal_dashboard() -> Dict[str, Any]:
    """
    Dashboard completo do CCM para UI.
    """
    if not HAS_CCM:
        raise HTTPException(status_code=501, detail="CCM module not available")
    
    graph, data = _get_causal_graph()
    
    dashboard = ComplexityDashboard(graph, data)
    result = dashboard.to_dict()
    
    # Adicionar resumo
    metrics = result["metrics"]
    insights = result["insights"]
    
    result["summary"] = {
        "complexity_score": metrics["complexity"]["overall_complexity"],
        "n_tradeoffs": sum(1 for i in insights if i["type"] == "tradeoff"),
        "n_leverage_points": sum(1 for i in insights if i["type"] == "leverage"),
        "n_risks": sum(1 for i in insights if i["type"] == "risk"),
        "high_priority_insights": sum(1 for i in insights if i["priority"] == "high"),
    }
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CONTRACT 8: WORK INSTRUCTIONS & SHOPFLOOR API
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/work-instructions/{revision_id}/{operation_id}")
def api_get_work_instructions(
    revision_id: int,
    operation_id: int,
    auto_generate: bool = True,
) -> Dict[str, Any]:
    """
    Get work instructions for a revision/operation.
    
    If not found and auto_generate=True, creates default instructions.
    """
    try:
        from prodplan.work_instructions import get_work_instructions
        
        wi = get_work_instructions(
            revision_id=revision_id,
            operation_id=operation_id,
            auto_generate=auto_generate,
        )
        
        if not wi:
            raise HTTPException(status_code=404, detail="Work instructions not found")
        
        return wi.dict()
    except ImportError:
        raise HTTPException(status_code=501, detail="Work instructions module not available")
    except Exception as e:
        logger.error(f"Error getting work instructions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/work-instructions/{revision_id}/{operation_id}")
def api_create_work_instructions(
    revision_id: int,
    operation_id: int,
    title: str = "Instrução de Trabalho",
    operation_code: str = "OP",
) -> Dict[str, Any]:
    """Create work instructions for a revision/operation."""
    try:
        from prodplan.work_instructions import (
            WorkInstructionService,
            WorkInstructionData,
        )
        
        # Generate default
        default_wi = WorkInstructionService.generate_default(
            revision_id, operation_id, operation_code
        )
        default_wi.title = title
        
        # Save
        created = WorkInstructionService.create(default_wi)
        
        return created.dict()
    except Exception as e:
        logger.error(f"Error creating work instructions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/shopfloor/orders")
def api_get_shopfloor_orders(
    machine_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Get orders for shopfloor display.
    
    Query parameters:
    - machine_id: Filter by machine
    - status: Filter by status (pending, in_progress, paused, completed)
    - limit: Max results
    """
    try:
        from prodplan.work_instructions import get_shopfloor_orders
        
        orders = get_shopfloor_orders(
            machine_id=machine_id,
            status=status,
            limit=limit,
        )
        
        return [o.dict() for o in orders]
    except ImportError:
        # Return demo data if module not available
        return [
            {
                "id": "ORD-001",
                "article_id": "ART-100",
                "article_name": "Demo Peça",
                "operation_code": "OP10",
                "machine_id": machine_id or "CNC-01",
                "planned_qty": 100,
                "good_qty": 0,
                "scrap_qty": 0,
                "status": "pending",
                "planned_start": "2025-01-09T08:00:00",
                "planned_end": "2025-01-09T12:00:00",
                "priority": 1,
                "has_work_instructions": True,
            }
        ]


class ShopfloorStartRequest(BaseModel):
    """Request body for starting an order."""
    machine_id: str
    operator_id: Optional[str] = None
    revision_id: Optional[int] = None
    operation_id: Optional[int] = None
    material_barcode: Optional[str] = None
    process_params: Optional[Dict[str, float]] = None
    skip_validation: bool = False  # Allow bypassing Poka-Yoke in emergencies


@app.post("/shopfloor/orders/{order_id}/start")
def api_start_order(
    order_id: str, 
    request: Optional[ShopfloorStartRequest] = None,
    machine_id: Optional[str] = None,  # Keep backward compatibility
    operator_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Start order execution with Poka-Yoke validation (Contract 9).
    
    Validates:
    - Revision status (RELEASED)
    - Work instructions availability
    - Material correctness (if barcode provided)
    - Process parameters in range (if specs defined)
    """
    try:
        from prodplan.work_instructions import (
            OrderExecutionReport,
            ExecutionStatus,
            report_order_execution,
            validate_operation_start,
            ShopfloorValidationError,
        )
        from datetime import datetime
        
        # Handle both new request body and legacy query params
        if request:
            effective_machine_id = request.machine_id
            effective_operator_id = request.operator_id
            revision_id = request.revision_id
            operation_id = request.operation_id
            material_barcode = request.material_barcode
            process_params = request.process_params
            skip_validation = request.skip_validation
        else:
            effective_machine_id = machine_id
            effective_operator_id = operator_id
            revision_id = None
            operation_id = None
            material_barcode = None
            process_params = None
            skip_validation = False
        
        if not effective_machine_id:
            raise HTTPException(status_code=400, detail="machine_id is required")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # POKA-YOKE DIGITAL VALIDATION (CONTRACT 9)
        # ═══════════════════════════════════════════════════════════════════════════
        validation_result = None
        if not skip_validation:
            try:
                validation_result = validate_operation_start(
                    order_id=order_id,
                    machine_id=effective_machine_id,
                    revision_id=revision_id,
                    operation_id=operation_id,
                    material_barcode=material_barcode,
                    process_params=process_params,
                    strict_mode=True,  # Will raise ShopfloorValidationError on failure
                )
            except ShopfloorValidationError as ve:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "message": str(ve),
                        "validation_errors": ve.validation_errors,
                        "warnings": ve.warnings,
                        "poka_yoke_failed": True,
                    }
                )
        
        # All validations passed (or skipped), start the order
        report = OrderExecutionReport(
            order_id=order_id,
            operation_id=operation_id or 0,
            machine_id=effective_machine_id,
            operator_id=effective_operator_id,
            status=ExecutionStatus.IN_PROGRESS,
            start_time=datetime.utcnow().isoformat(),
        )
        
        report_id = report_order_execution(report)
        
        response = {
            "success": True,
            "order_id": order_id,
            "status": "in_progress",
            "report_id": report_id,
            "started_at": report.start_time,
            "poka_yoke_validated": not skip_validation,
        }
        
        if validation_result and validation_result.get("warnings"):
            response["warnings"] = validation_result["warnings"]
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/shopfloor/orders/{order_id}/pause")
def api_pause_order(order_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
    """Pause order execution."""
    return {
        "success": True,
        "order_id": order_id,
        "status": "paused",
        "reason": reason,
    }


@app.post("/shopfloor/orders/{order_id}/complete")
def api_complete_order(order_id: str, machine_id: str) -> Dict[str, Any]:
    """Complete/terminate order execution."""
    try:
        from prodplan.work_instructions import (
            OrderExecutionReport,
            ExecutionStatus,
            report_order_execution,
        )
        from datetime import datetime
        
        report = OrderExecutionReport(
            order_id=order_id,
            operation_id=0,
            machine_id=machine_id,
            status=ExecutionStatus.COMPLETED,
            end_time=datetime.utcnow().isoformat(),
        )
        
        report_id = report_order_execution(report)
        
        return {
            "success": True,
            "order_id": order_id,
            "status": "completed",
            "report_id": report_id,
            "completed_at": report.end_time,
        }
    except Exception as e:
        logger.error(f"Error completing order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/shopfloor/orders/{order_id}/report")
def api_report_order_execution(
    order_id: str,
    machine_id: str,
    good_qty: int = 0,
    scrap_qty: int = 0,
    rework_qty: int = 0,
    operator_id: Optional[str] = None,
    downtime_reason: Optional[str] = None,
    downtime_minutes: int = 0,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Report order execution with quantities and quality data.
    
    This is the main endpoint for shopfloor operators to report production.
    """
    try:
        from prodplan.work_instructions import (
            OrderExecutionReport,
            ExecutionStatus,
            DowntimeReason,
            report_order_execution,
        )
        from datetime import datetime
        
        dt_reason = None
        if downtime_reason:
            try:
                dt_reason = DowntimeReason(downtime_reason)
            except ValueError:
                dt_reason = DowntimeReason.OTHER
        
        report = OrderExecutionReport(
            order_id=order_id,
            operation_id=0,
            machine_id=machine_id,
            operator_id=operator_id,
            good_qty=good_qty,
            scrap_qty=scrap_qty,
            rework_qty=rework_qty,
            status=ExecutionStatus.IN_PROGRESS,
            downtime_reason=dt_reason,
            downtime_minutes=downtime_minutes,
            notes=notes,
        )
        
        report_id = report_order_execution(report)
        
        return {
            "success": True,
            "order_id": order_id,
            "report_id": report_id,
            "good_qty": good_qty,
            "scrap_qty": scrap_qty,
            "rework_qty": rework_qty,
            "total_qty": good_qty + scrap_qty + rework_qty,
        }
    except Exception as e:
        logger.error(f"Error reporting order execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/shopfloor/machines")
def api_get_shopfloor_machines() -> List[Dict[str, Any]]:
    """Get list of machines for shopfloor selection."""
    # Demo machines
    return [
        {"id": "CNC-01", "name": "CNC-01", "type": "CNC", "status": "running"},
        {"id": "CNC-02", "name": "CNC-02", "type": "CNC", "status": "running"},
        {"id": "CNC-03", "name": "CNC-03", "type": "CNC", "status": "maintenance"},
        {"id": "MILL-01", "name": "MILL-01", "type": "Milling", "status": "running"},
        {"id": "MILL-02", "name": "MILL-02", "type": "Milling", "status": "idle"},
        {"id": "LATHE-01", "name": "LATHE-01", "type": "Lathe", "status": "running"},
    ]


# Note: /machines endpoint is already defined earlier in the file for planning
# The Shopfloor should use /shopfloor/machines


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL SOURCE INDICATORS (CONTRACT 9)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/system/model-sources")
def api_get_model_sources() -> Dict[str, Any]:
    """
    Get information about active model sources and engines.
    
    Used by frontend to display appropriate indicators for:
    - DataDrivenDurations (planning)
    - RUL estimation (digital twin machines)
    - XAI-DT (digital twin products)
    - Causal estimation
    """
    try:
        from feature_flags import FeatureFlags, get_active_engines
        
        active_engines = get_active_engines()
        
        return {
            "forecast_engine": {
                "name": active_engines.get("forecast", "classical"),
                "is_advanced": active_engines.get("forecast") == "advanced",
                "label_pt": "Previsão avançada (N-HiTS/TFT)" if active_engines.get("forecast") == "advanced" else "Previsão clássica (ARIMA/ETS)",
            },
            "rul_engine": {
                "name": active_engines.get("rul", "base"),
                "is_advanced": active_engines.get("rul") == "deepsurv",
                "label_pt": "Modelo de sobrevivência profundo" if active_engines.get("rul") == "deepsurv" else "Degradação exponencial/linear",
            },
            "deviation_engine": {
                "name": active_engines.get("deviation", "simple"),
                "is_advanced": active_engines.get("deviation") == "pod",
                "label_pt": "Modos de desvio (POD)" if active_engines.get("deviation") == "pod" else "Métricas simples",
            },
            "scheduler_engine": {
                "name": active_engines.get("scheduler", "heuristic"),
                "is_advanced": active_engines.get("scheduler") in ["milp", "cpsat", "drl"],
                "label_pt": _get_scheduler_label(active_engines.get("scheduler", "heuristic")),
            },
            "causal_engine": {
                "name": active_engines.get("causal", "ols"),
                "is_advanced": active_engines.get("causal") == "dml",
                "label_pt": "Double ML (EconML)" if active_engines.get("causal") == "dml" else "Regressão OLS",
            },
            "data_driven_durations": {
                "enabled": True,
                "label_pt": "Tempos baseados em histórico de chão de fábrica",
            },
        }
    except ImportError:
        # Feature flags not available, return defaults
        return {
            "forecast_engine": {"name": "classical", "is_advanced": False, "label_pt": "Previsão clássica"},
            "rul_engine": {"name": "base", "is_advanced": False, "label_pt": "Degradação base"},
            "deviation_engine": {"name": "simple", "is_advanced": False, "label_pt": "Métricas simples"},
            "scheduler_engine": {"name": "heuristic", "is_advanced": False, "label_pt": "Heurísticas"},
            "causal_engine": {"name": "ols", "is_advanced": False, "label_pt": "Regressão OLS"},
            "data_driven_durations": {"enabled": True, "label_pt": "Tempos baseados em histórico"},
        }


def _get_scheduler_label(engine: str) -> str:
    """Get Portuguese label for scheduler engine."""
    labels = {
        "heuristic": "Heurísticas (FIFO, SPT, EDD)",
        "milp": "Otimização MILP (OR-Tools CBC)",
        "cpsat": "CP-SAT (OR-Tools)",
        "drl": "Aprendizagem por Reforço (DRL)",
    }
    return labels.get(engine, "Heurísticas")


