"""
Simple APS scheduler for the Nikufra Production OS MVP.

- Usa os dados do Excel via DataBundle (data_loader).
- Para já, só modo NORMAL (planeamento clássico, sem lógica encadeada avançada).
- Escolhe route_label == "A" por artigo (se existir, senão usa a primeira rota).
- Sequencia operações por ordem e máquina, evitando overlaps.
- Produz data/production_plan.csv com o plano final.

Engine Support:
- HEURISTIC: Fast dispatching rule (default)
- MILP: Mathematical optimization (OR-Tools)
- CPSAT: Constraint programming (OR-Tools)
- DRL: Deep Reinforcement Learning (Stable-Baselines3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Dict, Literal, Optional, List, Any

import pandas as pd

from data_loader import load_dataset, DataBundle
from chains import MachineChain


# Diretório base: um nível acima de backend/
BASE_DIR = Path(__file__).resolve().parents[1]
PLAN_CSV_PATH = BASE_DIR / "data" / "production_plan.csv"

PlanningMode = Literal["NORMAL", "ENCADEADO"]
SchedulingEngine = Literal["HEURISTIC", "MILP", "CPSAT", "DRL"]

logger = logging.getLogger(__name__)


@dataclass
class PlanEntry:
    order_id: str
    article_id: str
    route_id: str
    route_label: str
    op_seq: int
    op_code: str
    machine_id: str
    qty: int
    start_time: datetime
    end_time: datetime
    duration_min: float


def _get_planning_start(orders: pd.DataFrame) -> datetime:
    """
    Define o instante inicial do plano.
    Regra simples: 1 dia antes da primeira due_date, às 06:00.
    """
    if "due_date" in orders.columns and not orders["due_date"].isna().all():
        first_due = orders["due_date"].min()
        base_date = first_due.date() if isinstance(first_due, datetime) else first_due
    else:
        base_date = datetime.now().date()

    return datetime.combine(base_date, time(6, 0)) - timedelta(days=1)


def _choose_route_for_article(
    routing: pd.DataFrame, 
    article_id: str, 
    preferred_route: str = None,
    order_index: int = 0
) -> pd.DataFrame:
    """
    Para um article_id:
      - Se preferred_route for especificada, tenta usá-la.
      - Caso contrário, distribui por round-robin entre as rotas disponíveis.
    Retorna as operações dessa rota, ordenadas por op_seq.
    """
    routes_art = routing[routing["article_id"] == article_id]
    if routes_art.empty:
        return routes_art

    # Obter rotas únicas disponíveis para este artigo
    available_routes = routes_art["route_label"].unique().tolist()
    available_routes.sort()  # A, B, C...
    
    if preferred_route and preferred_route in available_routes:
        chosen_label = preferred_route
    else:
        # Round-robin: distribuir ordens pelas rotas disponíveis
        chosen_label = available_routes[order_index % len(available_routes)]
    
    routes_chosen = routes_art[routes_art["route_label"] == chosen_label]
    if routes_chosen.empty:
        # Fallback para primeira rota
        chosen_route_id = routes_art.iloc[0]["route_id"]
    else:
        chosen_route_id = routes_chosen.iloc[0]["route_id"]

    ops_route = routes_art[routes_art["route_id"] == chosen_route_id].copy()
    ops_route = ops_route.sort_values("op_seq")
    return ops_route


def build_plan(
    data: Optional[DataBundle] = None,
    mode: PlanningMode = "NORMAL",
    chains: Optional[List[MachineChain]] = None,
    engine: SchedulingEngine = "HEURISTIC",
) -> pd.DataFrame:
    """
    Constrói o plano de produção para o MVP.

    :param data: DataBundle carregado via load_dataset(). Se None, carrega internamente.
    :param mode: 'NORMAL' ou 'ENCADEADO'. Para já, só NORMAL está implementado (encadeado faz fallback).
    :param chains: Lista de cadeias de máquinas (apenas relevante para o modo ENCADEADO futuro).
    :param engine: 'HEURISTIC' (default), 'MILP', ou 'CPSAT' para otimização matemática.
    :return: DataFrame com o plano (production_plan).
    
    PROJECT PLANNING INTEGRATION
    ═══════════════════════════
    
    O scheduler respeita a coluna 'priority' em data.orders.
    
    Para integrar prioridades por projeto:
    1. Use project_planning.build_projects_from_orders(orders)
    2. Calcule loads/risks com compute_all_project_loads/risks
    3. Obtenha prioridades: optimize_project_priorities(projects, loads, risks)
    4. Aplique: orders['priority'] = orders['order_id'].map(plan.get_order_priority_vector())
    5. Chame build_plan(data)
    
    O ordenamento de encomendas usa:
    - orders.sort_values(['priority', 'due_date'], ascending=[False, True])
    
    Assim, prioridades mais altas (projetos urgentes/críticos) são processadas primeiro.
    
    TODO[R&D]: Compare engines on different instance sizes:
    - HEURISTIC: O(n log n) complexity, always feasible
    - MILP: Optimal for small instances (<100 ops)
    - CPSAT: Better for larger instances, good at finding feasible solutions
    - DRL: Learning-based, requires training but can generalize
    Metrics: makespan, computation time, optimality gap, regret vs baseline.
    """
    if data is None:
        data = load_dataset()

    if mode not in ("NORMAL", "ENCADEADO"):
        raise ValueError(f"Modo de planeamento inválido: {mode}")

    if mode == "ENCADEADO":
        # TODO[PLANEAMENTO_ENCADEADO]:
        # - Utilizar a lista 'chains' para aplicar as regras de fluxo encadeado
        #   (ex.: ["M-305B","M-412","M-513"]).
        # - Sequenciar operações respeitando coerência upstream/downstream.
        # - Considerar buffers e tempos de transporte entre máquinas.
        # - Integrar lógicas específicas (ex.: transporte, operadores dedicados).
        # Por agora, fazemos fallback para o modo NORMAL.
        mode = "NORMAL"

    # Route to appropriate engine
    if engine == "DRL":
        try:
            return _build_plan_drl(data)
        except Exception as e:
            logger.warning(f"DRL engine failed: {e}. Falling back to HEURISTIC.")
            engine = "HEURISTIC"
    
    if engine in ("MILP", "CPSAT"):
        try:
            return _build_plan_optimization(data, engine)
        except Exception as e:
            logger.warning(f"Optimization engine {engine} failed: {e}. Falling back to HEURISTIC.")
            engine = "HEURISTIC"
    
    # Default: heuristic
    return _build_plan_heuristic(data)


def _build_plan_optimization(data: DataBundle, engine: SchedulingEngine) -> pd.DataFrame:
    """
    Build plan using mathematical optimization (MILP or CP-SAT).
    
    TODO[R&D]: Optimization research:
    - Q1.1: Can MILP reduce makespan by ≥10% vs heuristic?
    - Compare model formulations (time-indexed vs big-M)
    - Warm-start from heuristic solution
    """
    try:
        from optimization.scheduling_models import (
            Operation, Machine, MILPSchedulingModel, CPSATSchedulingModel, SchedulingConfig, ModelType
        )
        from optimization.solver_interface import SolverConfig
    except ImportError as e:
        logger.warning(f"Optimization modules not available: {e}")
        raise
    
    orders = data.orders.copy()
    routing = data.routing.copy()
    machines_df = data.machines.copy()
    
    # Prepare operations
    operations = []
    machine_ids = machines_df['machine_id'].unique().tolist()
    
    for order_idx, (_, order) in enumerate(orders.iterrows()):
        order_id = str(order["order_id"])
        article_id = str(order["article_id"])
        qty = int(order["qty"])
        
        # Get route for this order
        ops_route = _choose_route_for_article(routing, article_id, order_index=order_idx)
        if ops_route.empty:
            continue
        
        route_id = str(ops_route.iloc[0]["route_id"])
        route_label = str(ops_route.iloc[0]["route_label"])
        
        # Track predecessors for precedence constraints
        prev_op_id = None
        
        for _, op in ops_route.iterrows():
            op_seq = int(op["op_seq"])
            op_code = str(op["op_code"])
            machine_id = str(op["primary_machine_id"])
            base_time = float(op["base_time_per_unit_min"])
            duration = base_time * qty
            setup_family = op.get("setup_family", "default")
            
            # Get eligible machines
            alt_machines = op.get("alt_machine_ids", "")
            if isinstance(alt_machines, str) and alt_machines:
                eligible = [machine_id] + [m.strip() for m in alt_machines.split(",")]
            else:
                eligible = [machine_id]
            
            op_id = f"{order_id}_{op_seq}"
            
            operations.append(Operation(
                id=op_id,
                order_id=order_id,
                article_id=article_id,
                op_seq=op_seq,
                op_code=op_code,
                processing_time_min=duration,
                setup_family=setup_family if isinstance(setup_family, str) else "default",
                eligible_machines=eligible,
                primary_machine=machine_id,
                route_id=route_id,
                route_label=route_label,
                predecessors=[prev_op_id] if prev_op_id else [],
            ))
            
            prev_op_id = op_id
    
    if not operations:
        logger.warning("No operations to schedule")
        return pd.DataFrame(columns=['order_id', 'article_id', 'route_id', 'route_label', 
                                    'op_seq', 'op_code', 'machine_id', 'qty', 
                                    'start_time', 'end_time', 'duration_min'])
    
    # Prepare machines
    machines = []
    for _, m in machines_df.iterrows():
        machines.append(Machine(
            id=str(m["machine_id"]),
            work_center=str(m.get("work_center_group", "")),
            speed_factor=float(m.get("speed_factor", 1.0)),
        ))
    
    # Create and configure model
    config = SchedulingConfig(
        time_limit_sec=30.0,  # Reasonable time for MVP
        optimality_gap=0.05,
        num_workers=4,
    )
    
    if engine == "MILP":
        model = MILPSchedulingModel(config)
    else:
        model = CPSATSchedulingModel(config)
    
    model.set_operations(operations)
    model.set_machines(machines)
    
    # Solve
    logger.info(f"Building {engine} model with {len(operations)} operations, {len(machines)} machines...")
    model.build_model()
    
    logger.info(f"Solving {engine} model...")
    success = model.solve()
    
    if not success:
        logger.warning(f"{engine} solver failed to find solution")
        raise RuntimeError(f"{engine} solver failed")
    
    # Extract solution
    solution_df = model.get_solution()
    stats = model.get_statistics()
    
    logger.info(f"{engine} solution found: makespan={stats.get('objective_value', 'N/A')}")
    
    # Convert to expected format
    start_horizon = _get_planning_start(orders)
    
    result_records = []
    for _, row in solution_df.iterrows():
        # Find original operation data
        op_parts = row['operation_id'].split('_')
        order_id = '_'.join(op_parts[:-1])
        
        start_time = start_horizon + timedelta(minutes=row['start_min'])
        end_time = start_horizon + timedelta(minutes=row['end_min'])
        
        result_records.append({
            'order_id': row['order_id'],
            'article_id': row['article_id'],
            'route_id': row.get('route_id', ''),
            'route_label': row.get('route_label', 'A'),
            'op_seq': row['op_seq'],
            'op_code': row['op_code'],
            'machine_id': row['machine_id'],
            'qty': orders[orders['order_id'] == row['order_id']]['qty'].iloc[0] if not orders[orders['order_id'] == row['order_id']].empty else 0,
            'start_time': start_time,
            'end_time': end_time,
            'duration_min': row['duration_min'],
        })
    
    return pd.DataFrame(result_records)


def _build_plan_drl(data: DataBundle) -> pd.DataFrame:
    """
    Build plan using Deep Reinforcement Learning (DRL).
    
    Uses a trained policy network to make dispatching decisions.
    The policy is trained using Stable-Baselines3 (PPO/A2C/DQN).
    
    R&D Work Package: WP4 - Learning-Based Scheduling
    ═══════════════════════════════════════════════════
    
    Technical Hypothesis (H4.1):
        DRL can learn dispatching policies that outperform hand-crafted
        heuristics (EDD, SPT, FIFO) by discovering non-obvious patterns
        in the scheduling state space.
    
    Experiment Design:
        1. Train on synthetic problem instances
        2. Evaluate on held-out test instances
        3. Compare vs HEURISTIC baseline: makespan, tardiness, setup time
        4. Record regret curves for SIFIDE documentation
    
    MDP Formulation:
        State: (machine_states, operation_states, global_time)
        Action: operation_index to dispatch
        Reward: -tardiness - makespan_factor - setup_penalty + flow_bonus
    
    TODO[R&D]:
        - Multi-objective DRL with Pareto rewards
        - Transfer learning between production scenarios
        - Online learning with policy updates during execution
        - Attention-based policy networks for variable-size problems
    
    Args:
        data: DataBundle with orders, routing, machines
    
    Returns:
        DataFrame with production plan
    """
    try:
        from optimization.drl_scheduler import build_plan_drl, DRLSchedulerConfig
    except ImportError as e:
        logger.warning(f"DRL scheduler not available: {e}")
        raise ImportError(
            "DRL scheduling requires gymnasium and stable-baselines3. "
            "Install with: pip install gymnasium stable-baselines3"
        )
    
    config = DRLSchedulerConfig(
        fallback_to_heuristic=True,  # Use heuristic if no trained model
        validate_plan=True,
    )
    
    return build_plan_drl(data, config)


def _build_plan_heuristic(data: DataBundle) -> pd.DataFrame:
    """
    Build plan using heuristic dispatching rules.
    
    Original MVP implementation - fast and always produces a feasible solution.
    """

    orders = data.orders.copy()
    routing = data.routing.copy()

    # Garantir tipo numérico coerente para base_time_per_unit_min
    routing["base_time_per_unit_min"] = pd.to_numeric(
        routing["base_time_per_unit_min"], errors="coerce"
    ).fillna(0.0)

    # Ordenar ordens: prioridade (desc) e due_date (asc), se existirem
    if "priority" in orders.columns and "due_date" in orders.columns:
        orders_sorted = orders.sort_values(
            ["priority", "due_date"], ascending=[False, True]
        )
    else:
        orders_sorted = orders.copy()

    start_horizon = _get_planning_start(orders_sorted)

    # Estado interno: quando é que cada máquina fica livre e qual foi o fim da última operação de cada ordem
    machine_next_free: Dict[str, datetime] = {}
    order_last_finish: Dict[str, datetime] = {}

    entries: List[PlanEntry] = []

    for order_idx, (_, order) in enumerate(orders_sorted.iterrows()):
        order_id = str(order["order_id"])
        article_id = str(order["article_id"])
        qty = int(order["qty"])

        ops_route = _choose_route_for_article(routing, article_id, order_index=order_idx)
        if ops_route.empty:
            # Sem rota definida para este artigo → ignorar ordem neste MVP
            continue

        # Todas as operações desta ordem usam a mesma route_id/label
        route_id = str(ops_route.iloc[0]["route_id"])
        route_label = str(ops_route.iloc[0]["route_label"])

        last_finish = order_last_finish.get(order_id, start_horizon)

        for _, op in ops_route.iterrows():
            op_seq = int(op["op_seq"])
            op_code = str(op["op_code"])
            # No teu routing, a máquina principal vem em primary_machine_id
            machine_id = str(op["primary_machine_id"])

            base_time_per_unit = float(op["base_time_per_unit_min"])
            duration_min = base_time_per_unit * qty
            duration = timedelta(minutes=duration_min)

            # earliest start = max(livre da máquina, livre da ordem, última operação desta ordem)
            earliest_machine = machine_next_free.get(machine_id, start_horizon)
            earliest_order = order_last_finish.get(order_id, start_horizon)
            start_time = max(earliest_machine, earliest_order, last_finish)
            end_time = start_time + duration

            machine_next_free[machine_id] = end_time
            order_last_finish[order_id] = end_time
            last_finish = end_time

            entries.append(
                PlanEntry(
                    order_id=order_id,
                    article_id=article_id,
                    route_id=route_id,
                    route_label=route_label,
                    op_seq=op_seq,
                    op_code=op_code,
                    machine_id=machine_id,
                    qty=qty,
                    start_time=start_time,
                    end_time=end_time,
                    duration_min=duration_min,
                )
            )

    plan_df = pd.DataFrame([e.__dict__ for e in entries])

    if not plan_df.empty:
        plan_df = plan_df.sort_values("start_time").reset_index(drop=True)

    return plan_df


def compute_bottleneck(plan_df: pd.DataFrame) -> Optional[dict]:
    """
    Computa o gargalo como a máquina com maior carga agregada (duration_min).
    """
    if plan_df.empty:
        return None

    carga = (
        plan_df.groupby("machine_id")["duration_min"]
        .sum()
        .sort_values(ascending=False)
    )
    machine_id = carga.index[0]
    total_minutes = float(carga.iloc[0])

    return {"machine_id": machine_id, "total_minutes": total_minutes}


def compute_kpis(plan_df: pd.DataFrame, orders_df: pd.DataFrame) -> Dict[str, any]:
    """
    Computa KPIs industriais reais a partir do plano de produção.
    
    Returns:
        Dict com:
        - makespan_hours: tempo total do plano (max end - min start)
        - route_distribution: contagem por rota (A/B/C)
        - overlaps: número de sobreposições por máquina
        - active_bottleneck: máquina gargalo
        - machine_loads: carga e tempo ocioso por máquina
        - lead_time_average_h: tempo médio de processamento por artigo
        - otd_percent: percentagem de ordens entregues a tempo
        - setup_hours: estimativa de horas de setup
        - total_operations: total de operações
        - total_orders: total de ordens
    """
    if plan_df.empty:
        return {
            "makespan_hours": 0,
            "route_distribution": {},
            "overlaps": {"total": 0, "by_machine": {}},
            "active_bottleneck": None,
            "machine_loads": [],
            "lead_time_average_h": 0,
            "otd_percent": 100,
            "setup_hours": 0,
            "total_operations": 0,
            "total_orders": 0,
        }

    # Garantir que temos as colunas de tempo como datetime
    if not pd.api.types.is_datetime64_any_dtype(plan_df["start_time"]):
        plan_df = plan_df.copy()
        plan_df["start_time"] = pd.to_datetime(plan_df["start_time"])
        plan_df["end_time"] = pd.to_datetime(plan_df["end_time"])

    # 1. Makespan (horas)
    min_start = plan_df["start_time"].min()
    max_end = plan_df["end_time"].max()
    makespan_seconds = (max_end - min_start).total_seconds()
    makespan_hours = makespan_seconds / 3600

    # 2. Distribuição de rotas
    if "route_label" in plan_df.columns:
        route_counts = plan_df["route_label"].value_counts().to_dict()
        route_distribution = {str(k): int(v) for k, v in route_counts.items()}
    else:
        route_distribution = {"default": len(plan_df)}

    # 3. Overlaps (sobreposições) por máquina
    overlaps_by_machine = {}
    total_overlaps = 0
    
    for machine_id, machine_ops in plan_df.groupby("machine_id"):
        machine_ops_sorted = machine_ops.sort_values("start_time")
        machine_overlaps = 0
        
        prev_end = None
        for _, op in machine_ops_sorted.iterrows():
            if prev_end is not None and op["start_time"] < prev_end:
                machine_overlaps += 1
            prev_end = op["end_time"]
        
        if machine_overlaps > 0:
            overlaps_by_machine[str(machine_id)] = machine_overlaps
            total_overlaps += machine_overlaps

    # 4. Gargalo ativo
    bottleneck = compute_bottleneck(plan_df)

    # 5. Cargas por máquina (load e idle time)
    machine_loads = []
    
    for machine_id, machine_ops in plan_df.groupby("machine_id"):
        machine_ops_sorted = machine_ops.sort_values("start_time")
        
        # Carga total (tempo de execução)
        if "duration_min" in machine_ops_sorted.columns:
            load_min = machine_ops_sorted["duration_min"].sum()
        else:
            # Calcular a partir de start/end times
            load_min = sum((row["end_time"] - row["start_time"]).total_seconds() / 60 for _, row in machine_ops_sorted.iterrows())
        
        # Tempo ocioso (gaps entre operações)
        idle_min = 0
        prev_end = None
        
        for _, op in machine_ops_sorted.iterrows():
            if prev_end is not None:
                gap = (op["start_time"] - prev_end).total_seconds() / 60
                if gap > 0:
                    idle_min += gap
            prev_end = op["end_time"]
        
        # Tempo total disponível (do início ao fim das ops desta máquina)
        if len(machine_ops_sorted) > 0:
            first_start = machine_ops_sorted["start_time"].min()
            last_end = machine_ops_sorted["end_time"].max()
            span_min = (last_end - first_start).total_seconds() / 60
            utilization_pct = (load_min / span_min * 100) if span_min > 0 else 0
        else:
            span_min = 0
            utilization_pct = 0
        
        machine_loads.append({
            "machine_id": str(machine_id),
            "load_min": round(load_min, 1),
            "load_hours": round(load_min / 60, 2),
            "idle_min": round(idle_min, 1),
            "idle_hours": round(idle_min / 60, 2),
            "utilization_pct": round(utilization_pct, 1),
            "num_operations": len(machine_ops_sorted),
        })
    
    # Ordenar por carga descendente
    machine_loads.sort(key=lambda x: x["load_min"], reverse=True)

    # 6. Lead time médio por artigo (em horas)
    lead_times = []
    
    for article_id, article_ops in plan_df.groupby("article_id"):
        article_start = article_ops["start_time"].min()
        article_end = article_ops["end_time"].max()
        lead_time_h = (article_end - article_start).total_seconds() / 3600
        lead_times.append(lead_time_h)
    
    lead_time_average_h = sum(lead_times) / len(lead_times) if lead_times else 0

    # 7. OTD (On-Time Delivery) %
    # Cruzar plano com due_dates das ordens
    otd_count = 0
    total_orders_checked = 0
    
    if "due_date" in orders_df.columns:
        orders_df = orders_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(orders_df["due_date"]):
            orders_df["due_date"] = pd.to_datetime(orders_df["due_date"], errors="coerce")
        
        # Para cada ordem, verificar se termina antes da due_date
        for order_id in plan_df["order_id"].unique():
            order_ops = plan_df[plan_df["order_id"] == order_id]
            order_finish = order_ops["end_time"].max()
            
            # Encontrar due_date desta ordem
            order_info = orders_df[orders_df["order_id"] == order_id]
            if not order_info.empty and pd.notna(order_info["due_date"].iloc[0]):
                due_date = order_info["due_date"].iloc[0]
                total_orders_checked += 1
                if order_finish <= due_date:
                    otd_count += 1
    
    otd_percent = (otd_count / total_orders_checked * 100) if total_orders_checked > 0 else 100

    # 8. Estimativa de horas de setup
    # Calculamos setup como tempo estimado quando há mudança de família/operação
    setup_min = 0
    DEFAULT_SETUP_MIN = 15  # Setup padrão entre operações diferentes
    
    for machine_id, machine_ops in plan_df.groupby("machine_id"):
        machine_ops_sorted = machine_ops.sort_values("start_time")
        prev_op_code = None
        
        for _, op in machine_ops_sorted.iterrows():
            if prev_op_code is not None and op["op_code"] != prev_op_code:
                setup_min += DEFAULT_SETUP_MIN
            prev_op_code = op["op_code"]
    
    setup_hours = setup_min / 60

    # Estatísticas gerais
    total_operations = len(plan_df)
    total_orders = plan_df["order_id"].nunique()
    total_articles = plan_df["article_id"].nunique()
    total_machines = plan_df["machine_id"].nunique()

    return {
        "makespan_hours": round(makespan_hours, 2),
        "route_distribution": route_distribution,
        "overlaps": {
            "total": total_overlaps,
            "by_machine": overlaps_by_machine,
        },
        "active_bottleneck": bottleneck,
        "machine_loads": machine_loads,
        "lead_time_average_h": round(lead_time_average_h, 2),
        "otd_percent": round(otd_percent, 1),
        "setup_hours": round(setup_hours, 2),
        "total_operations": total_operations,
        "total_orders": total_orders,
        "total_articles": total_articles,
        "total_machines": total_machines,
        "plan_start": min_start.isoformat() if pd.notna(min_start) else None,
        "plan_end": max_end.isoformat() if pd.notna(max_end) else None,
    }


def save_plan_to_csv(plan_df: pd.DataFrame, path: Path = PLAN_CSV_PATH) -> None:
    """Guarda o plano no ficheiro CSV em data/."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plan_df.to_csv(path, index=False)


# ============================================================
# R&D EXPERIMENTAL SCHEDULER (WP1, WP4)
# ============================================================

def build_plan_experimental(
    data: Optional[DataBundle] = None,
    routing_strategy: str = "FIXED_PRIMARY",
    log_decisions: bool = True,
) -> tuple[pd.DataFrame, Dict]:
    """
    Experimental plan builder with pluggable routing strategies.
    
    R&D Module: WP1 (APS Core + Routing Intelligence)
    Hypothesis: H1.1 - Dynamic routing reduces makespan by ≥8%
    
    Args:
        data: DataBundle from Excel
        routing_strategy: One of FIXED_PRIMARY, SHORTEST_QUEUE, SETUP_AWARE,
                         LOAD_BALANCED, MULTI_OBJECTIVE, ML_PREDICTED, RANDOM
        log_decisions: Whether to log all routing decisions for analysis
    
    Returns:
        (plan_df, experiment_data) where experiment_data contains routing decisions
    """
    from research.routing_engine import (
        RoutingEngine, ScoringStrategy, RouteOption, RoutingContext,
        build_route_options_from_routing_df
    )
    
    if data is None:
        data = load_dataset()
    
    # Map string to enum
    strategy_map = {
        "FIXED_PRIMARY": ScoringStrategy.FIXED_PRIMARY,
        "SHORTEST_QUEUE": ScoringStrategy.SHORTEST_QUEUE,
        "SETUP_AWARE": ScoringStrategy.SETUP_AWARE,
        "LOAD_BALANCED": ScoringStrategy.LOAD_BALANCED,
        "MULTI_OBJECTIVE": ScoringStrategy.MULTI_OBJECTIVE,
        "ML_PREDICTED": ScoringStrategy.ML_PREDICTED,
        "RANDOM": ScoringStrategy.RANDOM,
    }
    
    strategy = strategy_map.get(routing_strategy, ScoringStrategy.FIXED_PRIMARY)
    
    # Load setup matrix if available
    setup_matrix = {}
    if hasattr(data, 'setup_matrix') and data.setup_matrix is not None:
        for _, row in data.setup_matrix.iterrows():
            key = (str(row.get("from_setup_family", "")), str(row.get("to_setup_family", "")))
            setup_matrix[key] = float(row.get("setup_time_min", 30))
    
    # Initialize routing engine
    engine = RoutingEngine(strategy=strategy, setup_matrix=setup_matrix)
    
    orders = data.orders.copy()
    routing = data.routing.copy()
    
    routing["base_time_per_unit_min"] = pd.to_numeric(
        routing["base_time_per_unit_min"], errors="coerce"
    ).fillna(0.0)
    
    if "priority" in orders.columns and "due_date" in orders.columns:
        orders_sorted = orders.sort_values(
            ["priority", "due_date"], ascending=[False, True]
        )
    else:
        orders_sorted = orders.copy()
    
    start_horizon = _get_planning_start(orders_sorted)
    
    machine_next_free: Dict[str, datetime] = {}
    machine_last_family: Dict[str, str] = {}
    order_last_finish: Dict[str, datetime] = {}
    
    entries: List[PlanEntry] = []
    
    for _, order in orders_sorted.iterrows():
        order_id = str(order["order_id"])
        article_id = str(order["article_id"])
        qty = int(order["qty"])
        due_date = order.get("due_date")
        if pd.notna(due_date):
            due_date = pd.to_datetime(due_date)
        else:
            due_date = None
        
        # Get all operations for this article (all routes)
        art_ops = routing[routing["article_id"] == article_id]
        if art_ops.empty:
            continue
        
        # Get unique operation sequences
        op_seqs = sorted(art_ops["op_seq"].unique())
        
        last_finish = order_last_finish.get(order_id, start_horizon)
        selected_route_id = None
        selected_route_label = None
        
        for op_seq in op_seqs:
            # Build route options for this operation
            route_options = build_route_options_from_routing_df(routing, article_id, op_seq)
            
            if not route_options:
                continue
            
            # Build context for routing decision
            machine_loads = {
                m: (machine_next_free.get(m, start_horizon) - start_horizon).total_seconds() / 60
                for m in machine_next_free
            }
            
            context = RoutingContext(
                article_id=article_id,
                op_code=route_options[0].setup_family if route_options else "",
                op_seq=op_seq,
                qty=qty,
                due_date=due_date,
                previous_machine=entries[-1].machine_id if entries else None,
                previous_setup_family=machine_last_family.get(
                    entries[-1].machine_id if entries else "", ""
                ),
                machine_loads=machine_loads,
                machine_last_family=machine_last_family,
                current_time=last_finish,
            )
            
            # Select best route using the routing engine
            selected_route = engine.select_route(route_options, context, log_decision=log_decisions)
            
            # Use the selected route for all operations in this order
            if selected_route_id is None:
                selected_route_id = selected_route.route_id
                selected_route_label = selected_route.route_label
            
            # Get operation details
            op_data = art_ops[
                (art_ops["op_seq"] == op_seq) & 
                (art_ops["route_id"] == selected_route_id)
            ]
            
            if op_data.empty:
                # Fallback to any route with this op_seq
                op_data = art_ops[art_ops["op_seq"] == op_seq].iloc[[0]]
            
            op = op_data.iloc[0]
            op_code = str(op["op_code"])
            machine_id = str(op["primary_machine_id"])
            setup_family = str(op.get("setup_family", ""))
            
            base_time_per_unit = float(op["base_time_per_unit_min"])
            duration_min = base_time_per_unit * qty
            duration = timedelta(minutes=duration_min)
            
            earliest_machine = machine_next_free.get(machine_id, start_horizon)
            earliest_order = order_last_finish.get(order_id, start_horizon)
            start_time = max(earliest_machine, earliest_order, last_finish)
            end_time = start_time + duration
            
            machine_next_free[machine_id] = end_time
            machine_last_family[machine_id] = setup_family
            order_last_finish[order_id] = end_time
            last_finish = end_time
            
            entries.append(
                PlanEntry(
                    order_id=order_id,
                    article_id=article_id,
                    route_id=selected_route_id or str(op["route_id"]),
                    route_label=selected_route_label or str(op["route_label"]),
                    op_seq=op_seq,
                    op_code=op_code,
                    machine_id=machine_id,
                    qty=qty,
                    start_time=start_time,
                    end_time=end_time,
                    duration_min=duration_min,
                )
            )
    
    plan_df = pd.DataFrame([e.__dict__ for e in entries])
    
    if not plan_df.empty:
        plan_df = plan_df.sort_values("start_time").reset_index(drop=True)
    
    # Collect experiment data
    experiment_data = {
        "routing_strategy": routing_strategy,
        "num_operations": len(plan_df),
        "num_orders": len(orders_sorted),
        "decision_log": engine.get_decision_log() if log_decisions else [],
    }
    
    return plan_df, experiment_data


def run_routing_experiment(
    strategies: List[str] = None,
    data: Optional[DataBundle] = None,
) -> Dict[str, any]:
    """
    Run experiment E1.1: Compare routing strategies.
    
    Returns comparison of makespan, setup hours, and OTD across strategies.
    """
    from research.experiment_logger import log_experiment
    
    if strategies is None:
        strategies = ["FIXED_PRIMARY", "SHORTEST_QUEUE", "SETUP_AWARE", "MULTI_OBJECTIVE"]
    
    if data is None:
        data = load_dataset()
    
    results = {}
    
    for strategy in strategies:
        with log_experiment(f"E1.1-{strategy}", hypothesis="H1.1") as exp:
            exp.set_config(routing_strategy=strategy)
            exp.set_inputs(
                num_orders=len(data.orders),
                num_machines=len(data.machines) if hasattr(data, 'machines') else 0,
            )
            
            plan_df, exp_data = build_plan_experimental(data, routing_strategy=strategy)
            kpis = compute_kpis(plan_df, data.orders)
            
            exp.set_outputs(
                makespan_h=kpis.get("makespan_hours", 0),
                setup_hours=kpis.get("setup_hours", 0),
                otd_pct=kpis.get("otd_percent", 0),
                num_operations=len(plan_df),
            )
            
            results[strategy] = {
                "kpis": kpis,
                "num_decisions": len(exp_data.get("decision_log", [])),
            }
    
    # Compute deltas vs baseline (FIXED_PRIMARY)
    if "FIXED_PRIMARY" in results:
        baseline = results["FIXED_PRIMARY"]["kpis"]
        for strategy, data in results.items():
            if strategy != "FIXED_PRIMARY":
                data["delta_vs_baseline"] = {
                    "makespan_pct": (
                        (data["kpis"]["makespan_hours"] - baseline["makespan_hours"]) 
                        / baseline["makespan_hours"] * 100
                    ) if baseline["makespan_hours"] > 0 else 0,
                    "setup_pct": (
                        (data["kpis"]["setup_hours"] - baseline["setup_hours"]) 
                        / baseline["setup_hours"] * 100
                    ) if baseline["setup_hours"] > 0 else 0,
                }
    
    return results


if __name__ == "__main__":
    # Entrada de linha de comando para gerar o plano completo
    bundle = load_dataset()
    plan = build_plan(bundle, mode="NORMAL")
    save_plan_to_csv(plan)
    bottleneck = compute_bottleneck(plan) or {}
    print(f"Plano gerado para {len(plan)} operações.")
    if bottleneck:
        print(
            f"Gargalo atual: {bottleneck['machine_id']} "
            f"com {bottleneck['total_minutes']:.2f} minutos."
        )
    print(f"Plano gravado em: {PLAN_CSV_PATH}")

