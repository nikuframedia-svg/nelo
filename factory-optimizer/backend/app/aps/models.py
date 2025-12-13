"""
Data models para APS encadeado - ProdPlan 4.0

Estrutura hierárquica:
- Order: Ordem de produção com sequência de operações
- OpRef: Referência de operação (etapa do roteiro) com alternativas
- OpAlternative: Alternativa de máquina para uma operação
- ScheduledOperation: Operação agendada no plano
- MachineState: Estado dinâmico de uma máquina
- Plan: Plano completo (baseline + optimized)
- PlanResult: Resultado de um plano (Antes ou Depois)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class OpAlternative:
    """Alternativa de máquina para uma operação."""
    maquina_id: str  # "016", "133", "244", ...
    ratio_pch: float  # Peças por hora
    pessoas: float  # 0.5, 1, 2, ...
    family: str  # "Corte Peça", "Laminar/ Aparar Chapa", ...
    setup_h: float = 0.5  # Tempo de setup em horas (calculado ou fixo)
    overlap_pct: float = 0.0  # % da anterior para iniciar (0-1)


@dataclass
class OpRef:
    """Referência de operação - etapa do roteiro com alternativas."""
    op_id: str  # "032", "030", "033", ...
    rota: str  # "A", "B", "C", ...
    stage_index: int  # 1, 2, 3, ... (posição na sequência)
    precedencias: List[int] = field(default_factory=list)  # stage_index das etapas anteriores
    operacao_logica: str = ""  # "Corte", "Brunir", "Polir"
    alternatives: List[OpAlternative] = field(default_factory=list)  # Máquinas alternativas (OR)


@dataclass
class Order:
    """Ordem de produção com sequência encadeada de operações."""
    id: str
    artigo: str
    quantidade: int
    prioridade: str  # "VIP" | "ALTA" | "NORMAL" | "BAIXA"
    due_date: Optional[datetime] = None
    data_entrada: datetime = field(default_factory=datetime.utcnow)
    operations: List[OpRef] = field(default_factory=list)  # Sequência encadeada (DAG)


@dataclass
class ScheduledOperation:
    """Operação agendada no plano."""
    order_id: str
    op_ref: OpRef
    alternative_chosen: OpAlternative
    start_time: datetime
    end_time: datetime
    quantidade: int
    duracao_h: float  # quantidade / ratio_pch
    
    @property
    def maquina_id(self) -> str:
        return self.alternative_chosen.maquina_id
    
    @property
    def family(self) -> str:
        return self.alternative_chosen.family


@dataclass
class TimeWindow:
    """Janela de tempo (para indisponibilidades)."""
    start: datetime
    end: datetime


@dataclass
class MachineState:
    """Estado dinâmico de uma máquina."""
    id: str
    operacoes_agendadas: List[ScheduledOperation] = field(default_factory=list)
    carga_acumulada_h: float = 0.0
    ultima_operacao_fim: Optional[datetime] = None
    ultima_familia: Optional[str] = None  # Para colagem
    indisponibilidades: List[TimeWindow] = field(default_factory=list)
    
    def is_available(self, start_time: datetime, end_time: datetime) -> bool:
        """Verifica se máquina está disponível no intervalo."""
        # Verificar indisponibilidades
        for unavail in self.indisponibilidades:
            if not (end_time <= unavail.start or start_time >= unavail.end):
                return False
        
        # Verificar sobreposição com operações agendadas
        for op in self.operacoes_agendadas:
            if not (end_time <= op.start_time or start_time >= op.end_time):
                return False
        
        return True
    
    def get_next_available_time(self, required_duration_h: float, earliest_start: datetime) -> datetime:
        """Retorna o próximo tempo disponível para agendar operação."""
        current_time = max(earliest_start, self.ultima_operacao_fim or earliest_start)
        
        # Verificar indisponibilidades
        for unavail in sorted(self.indisponibilidades, key=lambda x: x.start):
            if current_time < unavail.end:
                current_time = unavail.end
        
        # Verificar operações agendadas
        for op in sorted(self.operacoes_agendadas, key=lambda x: x.start_time):
            if current_time < op.end_time:
                current_time = op.end_time
        
        return current_time


@dataclass
class PlanResult:
    """Resultado de um plano - Antes ou Depois."""
    makespan_h: float
    total_setup_h: float
    kpis: Dict[str, float] = field(default_factory=dict)
    operations: List[ScheduledOperation] = field(default_factory=list)
    gantt_by_machine: Dict[str, List[ScheduledOperation]] = field(default_factory=dict)
    
    def build_gantt(self, all_machines: Optional[List[str]] = None):
        """
        Constrói gantt agrupado por máquina.
        
        Args:
            all_machines: Lista opcional de todas as máquinas que devem aparecer
                         (mesmo sem operações). Se None, apenas máquinas com operações são incluídas.
        """
        self.gantt_by_machine = {}
        for op in self.operations:
            machine_id = op.maquina_id
            if machine_id not in self.gantt_by_machine:
                self.gantt_by_machine[machine_id] = []
            self.gantt_by_machine[machine_id].append(op)
        
        # Se all_machines foi fornecido, garantir que todas aparecem (mesmo sem operações)
        if all_machines:
            for machine_id in all_machines:
                if machine_id not in self.gantt_by_machine:
                    self.gantt_by_machine[machine_id] = []
        
        # Ordenar por start_time em cada máquina
        for machine_id in self.gantt_by_machine:
            self.gantt_by_machine[machine_id].sort(key=lambda x: x.start_time)


@dataclass
class APSConfig:
    """Configuração do APS (alterável via Chat)."""
    objective: Dict[str, float] = field(default_factory=lambda: {
        "weight_lead_time": 0.5,
        "weight_setups": 0.3,
        "weight_bottleneck_balance": 0.2,
        "weight_otd": 0.4,
    })
    overlap: Dict[str, any] = field(default_factory=lambda: {
        "max_transformacao": 0.3,
        "max_acabamentos": 0.15,
        "max_embalagem": 0.25,
        "reduce_for_slow_ops": True,
    })
    routing_preferences: Dict[str, any] = field(default_factory=lambda: {
        "prefer_route": {},
        "avoid_machine": [],
        "prefer_fast_machines": True,
    })
    family_grouping: Dict[str, any] = field(default_factory=lambda: {
        "enabled": True,
        "setup_reduction_pct": 0.7,
        "min_family_size": 2,
    })
    priorities: Dict[str, any] = field(default_factory=lambda: {
        "vip_orders": [],
        "custom_priorities": {},
    })
    
    def to_dict(self) -> Dict:
        """Serializa config para dict."""
        return {
            "objective": self.objective,
            "overlap": self.overlap,
            "routing_preferences": self.routing_preferences,
            "family_grouping": self.family_grouping,
            "priorities": self.priorities,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "APSConfig":
        """Deserializa config de dict."""
        return cls(
            objective=data.get("objective", {}),
            overlap=data.get("overlap", {}),
            routing_preferences=data.get("routing_preferences", {}),
            family_grouping=data.get("family_grouping", {}),
            priorities=data.get("priorities", {}),
        )


@dataclass
class Plan:
    """Plano completo com baseline e optimized."""
    batch_id: str
    horizon_hours: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    baseline: Optional[PlanResult] = None
    optimized: Optional[PlanResult] = None
    config: APSConfig = field(default_factory=APSConfig)
    
    def to_dict(self, all_machines_from_engine: Optional[List[str]] = None) -> Dict:
        """
        Serializa plano para dict (para cache).
        
        Args:
            all_machines_from_engine: Lista opcional de TODAS as máquinas do engine
                                     (para garantir que máquinas sem operações também aparecem)
        """
        baseline_dict = self._serialize_plan_result(self.baseline, all_machines_from_engine) if self.baseline else None
        optimized_dict = self._serialize_plan_result(self.optimized, all_machines_from_engine) if self.optimized else None
        
        # Adicionar resumo de orders ao plano completo
        all_order_ids = set()
        for result in [self.baseline, self.optimized]:
            if result:
                for op in result.operations:
                    all_order_ids.add(op.order_id)
        
        orders_summary = {
            "total_orders": len(all_order_ids),
            "orders": sorted(list(all_order_ids)),
        }
        
        # Log para debug
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"📋 Plan.to_dict(): orders_summary.total_orders={orders_summary['total_orders']}, orders={orders_summary['orders']}")
        
        return {
            "batch_id": self.batch_id,
            "horizon_hours": self.horizon_hours,
            "created_at": self.created_at.isoformat(),
            "baseline": baseline_dict,
            "optimized": optimized_dict,
            "config": self.config.to_dict(),
            "orders_summary": orders_summary,
        }
    
    @staticmethod
    def _serialize_plan_result(result: PlanResult, all_machines_from_engine: Optional[List[str]] = None) -> Dict:
        """
        Serializa PlanResult para dict.
        
        Args:
            result: PlanResult a serializar
            all_machines_from_engine: Lista opcional de TODAS as máquinas do engine
                                     (para garantir que máquinas sem operações também aparecem)
        """
        # CRÍTICO: Usar all_machines_from_engine se fornecido, senão usar gantt_by_machine.keys()
        # Isto garante que máquinas sem operações também aparecem no JSON
        if all_machines_from_engine:
            all_machines_list = sorted(all_machines_from_engine)
            # Garantir que todas as máquinas do engine estão no gantt_by_machine
            for machine_id in all_machines_from_engine:
                if machine_id not in result.gantt_by_machine:
                    result.gantt_by_machine[machine_id] = []
        else:
            all_machines_list = sorted(list(result.gantt_by_machine.keys()))
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"📋 _serialize_plan_result: {len(all_machines_list)} máquinas: {all_machines_list}")
        
        # LOG CRÍTICO: Verificar rotas antes de serializar
        rotas_antes_serializar = [op.op_ref.rota for op in result.operations if op.op_ref]
        rotas_unicas = list(set(rotas_antes_serializar))
        logger.info(f"🔵 [SERIALIZE] Rotas ANTES de serializar: {rotas_unicas} (distribuição: {rotas_antes_serializar.count('A')} A, {rotas_antes_serializar.count('B')} B)")
        
        # Verificar se há operações sem rota
        ops_sem_rota = [op for op in result.operations if not op.op_ref or not op.op_ref.rota]
        if ops_sem_rota:
            logger.error(f"❌ [SERIALIZE] {len(ops_sem_rota)} operações SEM ROTA antes de serializar!")
        
        return {
            "makespan_h": result.makespan_h,
            "total_setup_h": result.total_setup_h,
            "kpis": result.kpis,
            "operations": [
                {
                    "order_id": op.order_id,
                    "artigo": op.order_id.replace("ORD-", ""),  # Adicionar artigo
                    "op_id": op.op_ref.op_id,
                    "rota": op.op_ref.rota,  # CRÍTICO: Ler rota diretamente do op_ref
                    "maquina_id": op.maquina_id,
                    "start_time": op.start_time.isoformat(),
                    "end_time": op.end_time.isoformat(),
                    "quantidade": op.quantidade,
                    "duracao_h": op.duracao_h,
                    "family": op.family,
                }
                for op in result.operations
            ],
            "orders_summary": {  # Adicionar resumo de orders
                "total_orders": len(set(op.order_id for op in result.operations)),
                "orders": sorted(list(set(op.order_id for op in result.operations))),
            },
            "gantt_by_machine": {
                machine_id: [
                    {
                        "order_id": op.order_id,
                        "op_id": op.op_ref.op_id,
                        "start_time": op.start_time.isoformat(),
                        "end_time": op.end_time.isoformat(),
                        "quantidade": op.quantidade,
                        "duracao_h": op.duracao_h,
                    }
                    for op in ops
                ]
                for machine_id, ops in result.gantt_by_machine.items()
            },
            "all_machines": all_machines_list,  # Lista completa de máquinas
        }

