"""
ProdPlan 4.0 - Scheduling Types
===============================

Tipos comuns para os motores de scheduling (Heurístico, MILP, CP-SAT, DRL).

Estrutura:
- SchedulingInstance: Input para qualquer engine
- SchedulingResult: Output unificado de qualquer engine
- ScheduledOperation: Uma operação agendada
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class SchedulerEngine(str, Enum):
    """Engine de scheduling disponível."""
    HEURISTIC = "heuristic"
    MILP = "milp"
    CPSAT = "cpsat"
    DRL = "drl"


class DispatchRule(str, Enum):
    """Regras de dispatching para heurísticas."""
    FIFO = "fifo"           # First In, First Out
    SPT = "spt"             # Shortest Processing Time
    EDD = "edd"             # Earliest Due Date
    CR = "cr"               # Critical Ratio
    WSPT = "wspt"           # Weighted Shortest Processing Time
    SQ = "sq"               # Shortest Queue
    SETUP_AWARE = "setup_aware"  # Minimiza setups
    RANDOM = "random"       # Random (baseline)


class ObjectiveType(str, Enum):
    """Tipo de objetivo de otimização."""
    MAKESPAN = "makespan"
    TARDINESS = "tardiness"
    SETUP = "setup"
    MULTI_OBJECTIVE = "multi_objective"


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Operation:
    """Uma operação a ser agendada."""
    operation_id: str
    order_id: str
    article_id: str
    op_seq: int                    # Sequência dentro da ordem
    op_code: str                   # Código da operação
    duration_min: float            # Duração em minutos
    primary_machine_id: str        # Máquina principal
    alternative_machines: List[str] = field(default_factory=list)
    setup_family: str = ""         # Família para cálculo de setup
    due_date: Optional[datetime] = None
    release_date: Optional[datetime] = None
    priority: float = 1.0
    weight: float = 1.0
    predecessor_ops: List[str] = field(default_factory=list)


@dataclass
class Machine:
    """Uma máquina/recurso disponível."""
    machine_id: str
    name: str = ""
    capacity: int = 1              # Capacidade (normalmente 1)
    speed_factor: float = 1.0      # Fator de velocidade
    available_from: Optional[datetime] = None
    available_until: Optional[datetime] = None
    shifts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SetupMatrix:
    """Matriz de tempos de setup entre famílias."""
    from_family: str
    to_family: str
    setup_time_min: float


class SchedulingInstance(BaseModel):
    """
    Instância completa de scheduling.
    
    Contém todos os dados necessários para qualquer engine de scheduling.
    """
    # Identificação
    instance_id: str = Field(default="default")
    description: str = Field(default="")
    
    # Operações a agendar
    operations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Máquinas disponíveis
    machines: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Matriz de setups (opcional)
    setup_matrix: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Horizonte de planeamento
    horizon_start: Optional[datetime] = None
    horizon_end: Optional[datetime] = None
    
    # Configuração
    objective: ObjectiveType = ObjectiveType.MAKESPAN
    objective_weights: Dict[str, float] = Field(
        default_factory=lambda: {"makespan": 1.0, "tardiness": 0.5, "setup": 0.2}
    )
    
    # Data-driven durations
    use_data_driven_durations: bool = False
    
    class Config:
        use_enum_values = True
    
    def get_operations(self) -> List[Operation]:
        """Converte operations dict para dataclass."""
        result = []
        for op_dict in self.operations:
            result.append(Operation(
                operation_id=op_dict.get("operation_id", op_dict.get("id", "")),
                order_id=op_dict.get("order_id", ""),
                article_id=op_dict.get("article_id", ""),
                op_seq=op_dict.get("op_seq", 0),
                op_code=op_dict.get("op_code", ""),
                duration_min=op_dict.get("duration_min", op_dict.get("processing_time_min", 0)),
                primary_machine_id=op_dict.get("primary_machine_id", op_dict.get("machine_id", "")),
                alternative_machines=op_dict.get("alternative_machines", []),
                setup_family=op_dict.get("setup_family", ""),
                due_date=op_dict.get("due_date"),
                release_date=op_dict.get("release_date"),
                priority=op_dict.get("priority", 1.0),
                weight=op_dict.get("weight", 1.0),
                predecessor_ops=op_dict.get("predecessor_ops", []),
            ))
        return result
    
    def get_machines(self) -> List[Machine]:
        """Converte machines dict para dataclass."""
        result = []
        for m_dict in self.machines:
            result.append(Machine(
                machine_id=m_dict.get("machine_id", m_dict.get("id", "")),
                name=m_dict.get("name", ""),
                capacity=m_dict.get("capacity", 1),
                speed_factor=m_dict.get("speed_factor", 1.0),
            ))
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScheduledOperation:
    """Uma operação agendada no plano."""
    operation_id: str
    order_id: str
    article_id: str
    op_seq: int
    op_code: str
    machine_id: str
    start_time: datetime
    end_time: datetime
    duration_min: float
    setup_time_min: float = 0.0
    is_data_driven_duration: bool = False


class SchedulingKPIs(BaseModel):
    """KPIs de scheduling (herdado de models_common)."""
    makespan_hours: float = 0.0
    total_tardiness_hours: float = 0.0
    num_late_orders: int = 0
    total_setup_time_hours: float = 0.0
    avg_machine_utilization: float = 0.0
    otd_rate: float = 1.0
    total_operations: int = 0
    total_orders: int = 0


class SchedulingResult(BaseModel):
    """
    Resultado unificado de qualquer engine de scheduling.
    """
    # Status
    success: bool = True
    engine_used: str = "heuristic"
    rule_used: Optional[str] = None  # Para heurísticas
    solve_time_sec: float = 0.0
    status: str = "optimal"  # optimal, feasible, infeasible, timeout
    
    # Plano
    scheduled_operations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # KPIs
    kpis: SchedulingKPIs = Field(default_factory=SchedulingKPIs)
    
    # Detalhes por máquina
    machine_utilization: Dict[str, float] = Field(default_factory=dict)
    machine_makespan: Dict[str, float] = Field(default_factory=dict)
    
    # Metadados
    warnings: List[str] = Field(default_factory=list)
    data_driven_count: int = 0  # Quantas operações usaram durações data-driven
    
    class Config:
        arbitrary_types_allowed = True


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS (API)
# ═══════════════════════════════════════════════════════════════════════════════

class PlanRequest(BaseModel):
    """Request para gerar plano de produção."""
    engine: SchedulerEngine = SchedulerEngine.HEURISTIC
    rule: DispatchRule = DispatchRule.EDD
    use_data_driven_durations: bool = False
    time_limit_sec: float = 60.0
    gap_tolerance: float = 0.05
    objective: ObjectiveType = ObjectiveType.MAKESPAN
    
    class Config:
        use_enum_values = True


class PlanResponse(BaseModel):
    """Response do endpoint de plano."""
    success: bool
    engine_used: str
    rule_used: Optional[str] = None
    solve_time_sec: float
    status: str
    operations: List[Dict[str, Any]]
    kpis: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_instance_from_dataframes(
    orders_df,
    routing_df,
    machines_df,
    horizon_start: Optional[datetime] = None,
) -> SchedulingInstance:
    """
    Cria SchedulingInstance a partir de DataFrames.
    
    Útil para integrar com o sistema existente.
    """
    import pandas as pd
    
    horizon_start = horizon_start or datetime.now()
    
    # Converter operações
    operations = []
    if routing_df is not None and not routing_df.empty:
        for _, row in routing_df.iterrows():
            op = {
                "operation_id": f"{row.get('article_id', '')}_{row.get('op_seq', 0)}",
                "order_id": str(row.get("order_id", "")),
                "article_id": str(row.get("article_id", "")),
                "op_seq": int(row.get("op_seq", 0)),
                "op_code": str(row.get("op_code", "")),
                "duration_min": float(row.get("duration_min", row.get("time_min", 0))),
                "primary_machine_id": str(row.get("machine_id", row.get("resource_id", ""))),
                "setup_family": str(row.get("setup_family", "")),
            }
            operations.append(op)
    
    # Converter máquinas
    machines = []
    if machines_df is not None and not machines_df.empty:
        for _, row in machines_df.iterrows():
            m = {
                "machine_id": str(row.get("machine_id", row.get("resource_id", ""))),
                "name": str(row.get("name", row.get("machine_name", ""))),
                "capacity": int(row.get("capacity", 1)),
            }
            machines.append(m)
    
    return SchedulingInstance(
        operations=operations,
        machines=machines,
        horizon_start=horizon_start,
    )



