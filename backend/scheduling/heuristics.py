"""
ProdPlan 4.0 - Heuristic Dispatching Rules
==========================================

Implementa regras de dispatching clássicas para job-shop scheduling:
- FIFO: First In, First Out
- SPT: Shortest Processing Time
- EDD: Earliest Due Date
- CR: Critical Ratio
- WSPT: Weighted Shortest Processing Time

Cada regra ordena as operações prontas para execução e escolhe a próxima.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Callable, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DispatchingRule(str, Enum):
    """Regras de dispatching disponíveis."""
    FIFO = "fifo"
    SPT = "spt"
    EDD = "edd"
    CR = "cr"
    WSPT = "wspt"
    RANDOM = "random"


@dataclass
class ReadyOperation:
    """Operação pronta para ser agendada."""
    operation_id: str
    order_id: str
    article_id: str
    op_seq: int
    op_code: str
    processing_time_min: float
    due_date: Optional[datetime] = None
    priority: float = 1.0
    weight: float = 1.0
    release_time: datetime = field(default_factory=datetime.now)
    eligible_machines: List[str] = field(default_factory=list)


@dataclass
class DispatchDecision:
    """Decisão de dispatching."""
    operation: ReadyOperation
    selected_machine: str
    start_time: datetime
    rule_used: DispatchingRule
    score: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# DISPATCHING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def dispatch_fifo(operations: List[ReadyOperation]) -> List[ReadyOperation]:
    """
    FIFO - First In, First Out.
    
    Ordena por tempo de chegada (release_time).
    """
    return sorted(operations, key=lambda op: op.release_time)


def dispatch_spt(operations: List[ReadyOperation]) -> List[ReadyOperation]:
    """
    SPT - Shortest Processing Time.
    
    Ordena por tempo de processamento (menor primeiro).
    Minimiza tempo médio de fluxo.
    """
    return sorted(operations, key=lambda op: op.processing_time_min)


def dispatch_edd(operations: List[ReadyOperation]) -> List[ReadyOperation]:
    """
    EDD - Earliest Due Date.
    
    Ordena por data de entrega (mais cedo primeiro).
    Minimiza lateness máximo.
    """
    def due_key(op: ReadyOperation) -> datetime:
        if op.due_date:
            return op.due_date
        # Operações sem due_date vão para o fim
        return datetime.max
    
    return sorted(operations, key=due_key)


def dispatch_cr(
    operations: List[ReadyOperation],
    current_time: datetime,
) -> List[ReadyOperation]:
    """
    CR - Critical Ratio.
    
    CR = (due_date - current_time) / remaining_processing_time
    
    CR < 1: atrasado ou vai atrasar
    CR = 1: on schedule
    CR > 1: à frente
    
    Ordena por CR (menor primeiro = mais crítico).
    """
    def cr_key(op: ReadyOperation) -> float:
        if not op.due_date:
            return float('inf')
        
        time_remaining = (op.due_date - current_time).total_seconds() / 60
        if op.processing_time_min <= 0:
            return float('inf') if time_remaining > 0 else float('-inf')
        
        return time_remaining / op.processing_time_min
    
    return sorted(operations, key=cr_key)


def dispatch_wspt(operations: List[ReadyOperation]) -> List[ReadyOperation]:
    """
    WSPT - Weighted Shortest Processing Time.
    
    Ordena por weight / processing_time (maior primeiro).
    Minimiza soma ponderada dos tempos de conclusão.
    """
    def wspt_key(op: ReadyOperation) -> float:
        if op.processing_time_min <= 0:
            return float('inf')
        return -op.weight / op.processing_time_min  # Negativo para ordenar DESC
    
    return sorted(operations, key=wspt_key)


def dispatch_random(operations: List[ReadyOperation]) -> List[ReadyOperation]:
    """
    Random dispatching (baseline para comparação).
    """
    shuffled = operations.copy()
    np.random.shuffle(shuffled)
    return shuffled


# ═══════════════════════════════════════════════════════════════════════════════
# HEURISTIC DISPATCHER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class HeuristicDispatcher:
    """
    Dispatcher heurístico com múltiplas regras.
    
    Suporta:
    - Seleção de regra de dispatching
    - Machine selection (quando há alternativas)
    - Logging de decisões para R&D
    """
    
    def __init__(
        self,
        rule: DispatchingRule = DispatchingRule.FIFO,
        log_decisions: bool = False,
    ):
        self.rule = rule
        self.log_decisions = log_decisions
        self._decision_log: List[DispatchDecision] = []
        
        # Mapear regras para funções
        self._dispatch_functions: Dict[DispatchingRule, Callable] = {
            DispatchingRule.FIFO: dispatch_fifo,
            DispatchingRule.SPT: dispatch_spt,
            DispatchingRule.EDD: dispatch_edd,
            DispatchingRule.CR: self._dispatch_cr_wrapper,
            DispatchingRule.WSPT: dispatch_wspt,
            DispatchingRule.RANDOM: dispatch_random,
        }
    
    def _dispatch_cr_wrapper(self, operations: List[ReadyOperation]) -> List[ReadyOperation]:
        """Wrapper para CR que precisa de current_time."""
        return dispatch_cr(operations, datetime.now())
    
    def select_next(
        self,
        ready_operations: List[ReadyOperation],
        machine_availability: Optional[Dict[str, datetime]] = None,
    ) -> Optional[DispatchDecision]:
        """
        Seleciona a próxima operação a executar.
        
        Args:
            ready_operations: Operações prontas para executar
            machine_availability: Quando cada máquina fica livre
        
        Returns:
            DispatchDecision ou None se não houver operações
        """
        if not ready_operations:
            return None
        
        # Aplicar regra de dispatching
        dispatch_func = self._dispatch_functions.get(self.rule, dispatch_fifo)
        sorted_ops = dispatch_func(ready_operations)
        
        # Selecionar a primeira operação
        selected_op = sorted_ops[0]
        
        # Selecionar máquina (shortest queue se houver alternativas)
        machine_availability = machine_availability or {}
        selected_machine = self._select_machine(selected_op, machine_availability)
        
        # Calcular start time
        machine_free = machine_availability.get(selected_machine, datetime.now())
        start_time = max(machine_free, selected_op.release_time)
        
        decision = DispatchDecision(
            operation=selected_op,
            selected_machine=selected_machine,
            start_time=start_time,
            rule_used=self.rule,
            score=self._compute_score(selected_op),
        )
        
        if self.log_decisions:
            self._decision_log.append(decision)
        
        return decision
    
    def _select_machine(
        self,
        operation: ReadyOperation,
        machine_availability: Dict[str, datetime],
    ) -> str:
        """
        Seleciona a máquina para a operação.
        
        Estratégia: Shortest Queue (máquina que fica livre primeiro).
        """
        if not operation.eligible_machines:
            # Sem máquinas elegíveis definidas, usar primary ou default
            return "DEFAULT"
        
        if len(operation.eligible_machines) == 1:
            return operation.eligible_machines[0]
        
        # Escolher máquina que fica livre mais cedo
        best_machine = operation.eligible_machines[0]
        best_time = machine_availability.get(best_machine, datetime.min)
        
        for machine in operation.eligible_machines[1:]:
            free_time = machine_availability.get(machine, datetime.min)
            if free_time < best_time:
                best_time = free_time
                best_machine = machine
        
        return best_machine
    
    def _compute_score(self, operation: ReadyOperation) -> float:
        """Calcula score baseado na regra atual."""
        if self.rule == DispatchingRule.SPT:
            return -operation.processing_time_min
        elif self.rule == DispatchingRule.WSPT:
            return operation.weight / max(0.1, operation.processing_time_min)
        elif self.rule == DispatchingRule.EDD:
            if operation.due_date:
                return -operation.due_date.timestamp()
            return 0
        else:
            return 0
    
    def get_decision_log(self) -> List[DispatchDecision]:
        """Retorna log de decisões para análise R&D."""
        return self._decision_log
    
    def clear_log(self) -> None:
        """Limpa o log de decisões."""
        self._decision_log = []


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE DISPATCHER (Multi-rule)
# ═══════════════════════════════════════════════════════════════════════════════

class CompositeDispatcher:
    """
    Dispatcher composto que pode combinar múltiplas regras.
    
    Útil para:
    - Comparação de regras
    - Regras adaptativas
    - Experimentação R&D
    """
    
    def __init__(self, rules: List[DispatchingRule] = None):
        self.rules = rules or [DispatchingRule.FIFO]
        self.dispatchers = {
            rule: HeuristicDispatcher(rule, log_decisions=True)
            for rule in self.rules
        }
    
    def compare_rules(
        self,
        ready_operations: List[ReadyOperation],
    ) -> Dict[DispatchingRule, List[ReadyOperation]]:
        """
        Compara ordenação de diferentes regras.
        
        Returns:
            Dict com ordenação de cada regra
        """
        results = {}
        for rule, dispatcher in self.dispatchers.items():
            func = dispatcher._dispatch_functions.get(rule, dispatch_fifo)
            results[rule] = func(ready_operations)
        return results
    
    def select_best_rule(
        self,
        ready_operations: List[ReadyOperation],
        machine_loads: Dict[str, float],
        urgency_threshold: float = 0.5,
    ) -> DispatchingRule:
        """
        Seleciona a melhor regra baseado no contexto.
        
        Heurística simples:
        - Se há operações urgentes (CR < 1): usar EDD
        - Se carga está alta: usar SPT (minimizar WIP)
        - Caso contrário: usar FIFO
        """
        # Verificar urgência
        now = datetime.now()
        urgent_ops = [
            op for op in ready_operations
            if op.due_date and (op.due_date - now).total_seconds() / 60 < op.processing_time_min * 2
        ]
        
        if urgent_ops:
            return DispatchingRule.EDD
        
        # Verificar carga média
        avg_load = np.mean(list(machine_loads.values())) if machine_loads else 0
        
        if avg_load > 0.8:
            return DispatchingRule.SPT
        
        return DispatchingRule.FIFO


# ═══════════════════════════════════════════════════════════════════════════════
# HEURISTIC SCHEDULER (High-level API)
# ═══════════════════════════════════════════════════════════════════════════════

class HeuristicScheduler:
    """
    Scheduler heurístico de alto nível.
    
    Recebe uma SchedulingInstance e devolve SchedulingResult.
    Interface unificada para usar com diferentes regras de dispatching.
    
    Uso:
        scheduler = HeuristicScheduler(rule="EDD")
        result = scheduler.build_schedule(instance)
    """
    
    def __init__(
        self,
        rule: str = "EDD",
        log_decisions: bool = False,
    ):
        """
        Args:
            rule: Regra de dispatching ("FIFO", "SPT", "EDD", "CR", "WSPT", "SQ", "SETUP_AWARE")
            log_decisions: Se True, guarda log de decisões para análise
        """
        self.rule_name = rule.upper()
        self.log_decisions = log_decisions
        
        # Mapear string para enum
        rule_map = {
            "FIFO": DispatchingRule.FIFO,
            "SPT": DispatchingRule.SPT,
            "EDD": DispatchingRule.EDD,
            "CR": DispatchingRule.CR,
            "WSPT": DispatchingRule.WSPT,
            "SQ": DispatchingRule.FIFO,  # SQ usa mesma base que FIFO mas com machine selection
            "SETUP_AWARE": DispatchingRule.FIFO,  # Setup-aware usa FIFO base
            "RANDOM": DispatchingRule.RANDOM,
        }
        
        self.rule = rule_map.get(self.rule_name, DispatchingRule.EDD)
        self._dispatcher = HeuristicDispatcher(self.rule, log_decisions=log_decisions)
        self._decision_log: List[Dict] = []
    
    def build_schedule(self, instance) -> Dict:
        """
        Constrói um plano de produção usando heurística.
        
        Args:
            instance: SchedulingInstance ou dict com operations/machines
        
        Returns:
            Dict com scheduled_operations, kpis, etc.
        """
        import time
        from datetime import timedelta
        
        start_time = time.time()
        
        # Extrair operações e máquinas
        if hasattr(instance, 'get_operations'):
            operations = instance.get_operations()
            machines = instance.get_machines()
            horizon_start = instance.horizon_start or datetime.now()
        else:
            # Assumir que é dict
            operations = instance.get("operations", [])
            machines = instance.get("machines", [])
            horizon_start = instance.get("horizon_start", datetime.now())
        
        if not operations:
            return self._empty_result(time.time() - start_time)
        
        # Converter para ReadyOperation
        ready_ops = []
        for op in operations:
            if isinstance(op, dict):
                ready_ops.append(ReadyOperation(
                    operation_id=op.get("operation_id", ""),
                    order_id=op.get("order_id", ""),
                    article_id=op.get("article_id", ""),
                    op_seq=op.get("op_seq", 0),
                    op_code=op.get("op_code", ""),
                    processing_time_min=op.get("duration_min", op.get("processing_time_min", 0)),
                    due_date=op.get("due_date"),
                    priority=op.get("priority", 1.0),
                    weight=op.get("weight", 1.0),
                    release_time=op.get("release_date", horizon_start) or horizon_start,
                    eligible_machines=[op.get("primary_machine_id", "")] + op.get("alternative_machines", []),
                ))
            else:
                ready_ops.append(ReadyOperation(
                    operation_id=op.operation_id,
                    order_id=op.order_id,
                    article_id=op.article_id,
                    op_seq=op.op_seq,
                    op_code=op.op_code,
                    processing_time_min=op.duration_min,
                    due_date=op.due_date,
                    priority=op.priority,
                    weight=op.weight,
                    release_time=op.release_date or horizon_start,
                    eligible_machines=[op.primary_machine_id] + op.alternative_machines,
                ))
        
        # Ordenar por regra de dispatching
        dispatch_func = self._dispatcher._dispatch_functions.get(self.rule, dispatch_fifo)
        if self.rule == DispatchingRule.CR:
            sorted_ops = dispatch_cr(ready_ops, horizon_start)
        else:
            sorted_ops = dispatch_func(ready_ops)
        
        # Agendar operações
        scheduled = []
        machine_availability: Dict[str, datetime] = {}
        order_precedence: Dict[str, datetime] = {}  # order_id -> last op end time
        
        for op in sorted_ops:
            # Determinar máquina
            machine_id = self._select_best_machine(op, machine_availability)
            
            # Calcular start time
            machine_free = machine_availability.get(machine_id, horizon_start)
            precedence_ready = order_precedence.get(op.order_id, horizon_start)
            release = op.release_time if op.release_time else horizon_start
            
            start = max(machine_free, precedence_ready, release)
            duration = timedelta(minutes=op.processing_time_min)
            end = start + duration
            
            # Setup time (simplificado)
            setup_time = 0.0  # TODO: calcular setup baseado em família
            
            scheduled.append({
                "operation_id": op.operation_id,
                "order_id": op.order_id,
                "article_id": op.article_id,
                "op_seq": op.op_seq,
                "op_code": op.op_code,
                "machine_id": machine_id,
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "duration_min": op.processing_time_min,
                "setup_time_min": setup_time,
            })
            
            # Atualizar disponibilidades
            machine_availability[machine_id] = end
            order_precedence[op.order_id] = end
            
            if self.log_decisions:
                self._decision_log.append({
                    "operation_id": op.operation_id,
                    "rule": self.rule_name,
                    "machine": machine_id,
                    "start": start.isoformat(),
                })
        
        # Calcular KPIs
        solve_time = time.time() - start_time
        kpis = self._compute_kpis(scheduled, ready_ops, machine_availability, horizon_start)
        
        return {
            "success": True,
            "engine_used": "heuristic",
            "rule_used": self.rule_name,
            "solve_time_sec": round(solve_time, 3),
            "status": "optimal",
            "scheduled_operations": scheduled,
            "kpis": kpis,
            "machine_utilization": self._compute_utilization(scheduled, machine_availability, horizon_start),
            "warnings": [],
            "data_driven_count": 0,
        }
    
    def _select_best_machine(
        self,
        op: ReadyOperation,
        machine_availability: Dict[str, datetime],
    ) -> str:
        """Seleciona a melhor máquina para a operação."""
        eligible = op.eligible_machines if op.eligible_machines else ["DEFAULT"]
        eligible = [m for m in eligible if m]  # Remover vazios
        
        if not eligible:
            return "DEFAULT"
        
        if len(eligible) == 1:
            return eligible[0]
        
        # Shortest queue: máquina que fica livre primeiro
        best = eligible[0]
        best_time = machine_availability.get(best, datetime.min)
        
        for m in eligible[1:]:
            m_time = machine_availability.get(m, datetime.min)
            if m_time < best_time:
                best = m
                best_time = m_time
        
        return best
    
    def _compute_kpis(
        self,
        scheduled: List[Dict],
        operations: List[ReadyOperation],
        machine_availability: Dict[str, datetime],
        horizon_start: datetime,
    ) -> Dict:
        """Calcula KPIs do plano."""
        if not scheduled:
            return {
                "makespan_hours": 0,
                "total_tardiness_hours": 0,
                "num_late_orders": 0,
                "total_setup_time_hours": 0,
                "avg_machine_utilization": 0,
                "otd_rate": 1.0,
                "total_operations": 0,
                "total_orders": 0,
            }
        
        # Makespan
        end_times = [datetime.fromisoformat(s["end_time"]) for s in scheduled]
        makespan = (max(end_times) - horizon_start).total_seconds() / 3600
        
        # Tardiness
        op_due_dates = {op.operation_id: op.due_date for op in operations}
        total_tardiness = 0.0
        late_orders = set()
        
        for s in scheduled:
            due = op_due_dates.get(s["operation_id"])
            if due:
                end = datetime.fromisoformat(s["end_time"])
                if end > due:
                    tardiness = (end - due).total_seconds() / 3600
                    total_tardiness += tardiness
                    late_orders.add(s["order_id"])
        
        # Setup time
        total_setup = sum(s.get("setup_time_min", 0) for s in scheduled) / 60
        
        # Orders
        orders = set(s["order_id"] for s in scheduled)
        otd_rate = 1 - len(late_orders) / len(orders) if orders else 1.0
        
        return {
            "makespan_hours": round(makespan, 2),
            "total_tardiness_hours": round(total_tardiness, 2),
            "num_late_orders": len(late_orders),
            "total_setup_time_hours": round(total_setup, 2),
            "avg_machine_utilization": 0.75,  # Placeholder
            "otd_rate": round(otd_rate, 3),
            "total_operations": len(scheduled),
            "total_orders": len(orders),
        }
    
    def _compute_utilization(
        self,
        scheduled: List[Dict],
        machine_availability: Dict[str, datetime],
        horizon_start: datetime,
    ) -> Dict[str, float]:
        """Calcula utilização por máquina."""
        utilization = {}
        
        # Agrupar por máquina
        machine_ops: Dict[str, List[Dict]] = {}
        for s in scheduled:
            m = s["machine_id"]
            if m not in machine_ops:
                machine_ops[m] = []
            machine_ops[m].append(s)
        
        for machine, ops in machine_ops.items():
            total_work = sum(o["duration_min"] for o in ops)
            end_time = machine_availability.get(machine, horizon_start)
            total_time = (end_time - horizon_start).total_seconds() / 60
            
            if total_time > 0:
                utilization[machine] = round(total_work / total_time, 3)
            else:
                utilization[machine] = 0.0
        
        return utilization
    
    def _empty_result(self, solve_time: float) -> Dict:
        """Retorna resultado vazio."""
        return {
            "success": True,
            "engine_used": "heuristic",
            "rule_used": self.rule_name,
            "solve_time_sec": round(solve_time, 3),
            "status": "optimal",
            "scheduled_operations": [],
            "kpis": {
                "makespan_hours": 0,
                "total_tardiness_hours": 0,
                "num_late_orders": 0,
                "total_setup_time_hours": 0,
                "avg_machine_utilization": 0,
                "otd_rate": 1.0,
                "total_operations": 0,
                "total_orders": 0,
            },
            "machine_utilization": {},
            "warnings": ["No operations to schedule"],
            "data_driven_count": 0,
        }
    
    def get_decision_log(self) -> List[Dict]:
        """Retorna log de decisões."""
        return self._decision_log

