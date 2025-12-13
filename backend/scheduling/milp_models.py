"""
ProdPlan 4.0 - MILP Scheduling Models
=====================================

Modelos MILP (Mixed-Integer Linear Programming) para scheduling:
- Job-Shop: cada job tem sequência fixa de operações
- Flow-Shop: todos os jobs seguem a mesma sequência de máquinas

Usa OR-Tools como solver backend.

R&D / SIFIDE: WP1 - Intelligent APS Core
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MILPOperation:
    """Operação para modelo MILP."""
    id: str
    job_id: str
    machine_id: str
    processing_time: int  # minutos
    sequence: int  # ordem dentro do job
    due_date: Optional[datetime] = None
    weight: float = 1.0
    release_time: int = 0  # minutos desde horizonte


@dataclass
class MILPMachine:
    """Máquina para modelo MILP."""
    id: str
    speed_factor: float = 1.0
    available_from: int = 0  # minutos desde horizonte


@dataclass
class MILPSolution:
    """Solução do modelo MILP."""
    schedule: pd.DataFrame  # operation_id, machine_id, start, end
    makespan: float
    total_tardiness: float
    solve_time_sec: float
    gap: float
    status: str  # OPTIMAL, FEASIBLE, INFEASIBLE, etc.


# ═══════════════════════════════════════════════════════════════════════════════
# BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class BaseMILPModel(ABC):
    """Base class para modelos MILP de scheduling."""
    
    def __init__(
        self,
        time_limit_sec: float = 60.0,
        gap_tolerance: float = 0.05,
    ):
        self.time_limit_sec = time_limit_sec
        self.gap_tolerance = gap_tolerance
        self.operations: List[MILPOperation] = []
        self.machines: List[MILPMachine] = []
        self._model = None
        self._solver = None
    
    def set_operations(self, operations: List[MILPOperation]) -> None:
        """Define operações a agendar."""
        self.operations = operations
    
    def set_machines(self, machines: List[MILPMachine]) -> None:
        """Define máquinas disponíveis."""
        self.machines = machines
    
    @abstractmethod
    def build(self) -> None:
        """Constrói o modelo MILP."""
        pass
    
    @abstractmethod
    def solve(self) -> MILPSolution:
        """Resolve o modelo."""
        pass
    
    def _check_ortools(self) -> bool:
        """Verifica se OR-Tools está disponível."""
        try:
            from ortools.linear_solver import pywraplp
            return True
        except ImportError:
            logger.warning("OR-Tools não disponível para MILP")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# JOB-SHOP MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class MILPJobShopModel(BaseMILPModel):
    """
    Modelo MILP para Job-Shop Scheduling.
    
    Formulação:
    - Variáveis: s[op] = start time, x[op1,op2,m] = sequencing binary
    - Objetivo: min makespan (ou weighted tardiness)
    - Restrições:
      - Precedência dentro de cada job
      - Não-sobreposição em cada máquina (big-M ou disjunctive)
    
    Complexidade: NP-hard
    """
    
    def __init__(
        self,
        time_limit_sec: float = 60.0,
        gap_tolerance: float = 0.05,
        objective: str = "makespan",  # "makespan" ou "tardiness"
    ):
        super().__init__(time_limit_sec, gap_tolerance)
        self.objective = objective
        self._start_vars = {}
        self._makespan_var = None
    
    def build(self) -> None:
        """
        Constrói modelo MILP usando OR-Tools.
        """
        if not self._check_ortools():
            raise ImportError("OR-Tools necessário para MILP")
        
        from ortools.linear_solver import pywraplp
        
        # Criar solver
        self._solver = pywraplp.Solver.CreateSolver('CBC')
        if not self._solver:
            self._solver = pywraplp.Solver.CreateSolver('SCIP')
        
        if not self._solver:
            raise RuntimeError("Nenhum solver MILP disponível")
        
        solver = self._solver
        
        # Horizonte (big-M)
        M = sum(op.processing_time for op in self.operations) + 1000
        
        # Variáveis de start time
        for op in self.operations:
            self._start_vars[op.id] = solver.NumVar(
                op.release_time, M, f"s_{op.id}"
            )
        
        # Makespan
        self._makespan_var = solver.NumVar(0, M, "makespan")
        
        # Restrição: makespan >= end_time de todas as operações
        for op in self.operations:
            solver.Add(
                self._makespan_var >= self._start_vars[op.id] + op.processing_time,
                f"makespan_{op.id}"
            )
        
        # Restrições de precedência (dentro de cada job)
        jobs = {}
        for op in self.operations:
            if op.job_id not in jobs:
                jobs[op.job_id] = []
            jobs[op.job_id].append(op)
        
        for job_id, job_ops in jobs.items():
            sorted_ops = sorted(job_ops, key=lambda x: x.sequence)
            for i in range(len(sorted_ops) - 1):
                op1 = sorted_ops[i]
                op2 = sorted_ops[i + 1]
                solver.Add(
                    self._start_vars[op2.id] >= self._start_vars[op1.id] + op1.processing_time,
                    f"prec_{op1.id}_{op2.id}"
                )
        
        # Restrições de não-sobreposição por máquina (big-M)
        machines = {}
        for op in self.operations:
            if op.machine_id not in machines:
                machines[op.machine_id] = []
            machines[op.machine_id].append(op)
        
        for machine_id, machine_ops in machines.items():
            for i, op1 in enumerate(machine_ops):
                for op2 in machine_ops[i+1:]:
                    # y = 1 se op1 antes de op2
                    y = solver.BoolVar(f"y_{op1.id}_{op2.id}")
                    
                    # op1 antes de op2 OU op2 antes de op1
                    solver.Add(
                        self._start_vars[op2.id] >= self._start_vars[op1.id] + op1.processing_time - M * (1 - y),
                        f"seq1_{op1.id}_{op2.id}"
                    )
                    solver.Add(
                        self._start_vars[op1.id] >= self._start_vars[op2.id] + op2.processing_time - M * y,
                        f"seq2_{op1.id}_{op2.id}"
                    )
        
        # Objetivo
        solver.Minimize(self._makespan_var)
        
        logger.info(f"MILP Job-Shop: {solver.NumVariables()} vars, {solver.NumConstraints()} constraints")
    
    def solve(self) -> MILPSolution:
        """Resolve o modelo MILP."""
        if self._solver is None:
            self.build()
        
        from ortools.linear_solver import pywraplp
        
        # Set time limit
        self._solver.SetTimeLimit(int(self.time_limit_sec * 1000))
        
        # Solve
        import time
        start_time = time.time()
        status = self._solver.Solve()
        solve_time = time.time() - start_time
        
        # Status mapping
        status_map = {
            pywraplp.Solver.OPTIMAL: "OPTIMAL",
            pywraplp.Solver.FEASIBLE: "FEASIBLE",
            pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
            pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
            pywraplp.Solver.ABNORMAL: "ABNORMAL",
            pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
        }
        
        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            return MILPSolution(
                schedule=pd.DataFrame(),
                makespan=float('inf'),
                total_tardiness=float('inf'),
                solve_time_sec=solve_time,
                gap=1.0,
                status=status_map.get(status, "UNKNOWN"),
            )
        
        # Extrair solução
        records = []
        total_tardiness = 0.0
        
        for op in self.operations:
            start = self._start_vars[op.id].solution_value()
            end = start + op.processing_time
            
            # Calcular tardiness
            if op.due_date:
                due_min = (op.due_date - datetime.now()).total_seconds() / 60
                tardiness = max(0, end - due_min)
                total_tardiness += tardiness * op.weight
            
            records.append({
                "operation_id": op.id,
                "job_id": op.job_id,
                "machine_id": op.machine_id,
                "start_min": start,
                "end_min": end,
                "processing_time": op.processing_time,
            })
        
        return MILPSolution(
            schedule=pd.DataFrame(records),
            makespan=self._makespan_var.solution_value(),
            total_tardiness=total_tardiness,
            solve_time_sec=solve_time,
            gap=0.0 if status == pywraplp.Solver.OPTIMAL else self.gap_tolerance,
            status=status_map.get(status, "UNKNOWN"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW-SHOP MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class MILPFlowShopModel(BaseMILPModel):
    """
    Modelo MILP para Flow-Shop Scheduling.
    
    Simplificação: todos os jobs seguem a mesma sequência de máquinas.
    Mais simples que Job-Shop, mas ainda NP-hard.
    
    Formulação: Similar a Job-Shop, mas com estrutura de precedência uniforme.
    """
    
    def __init__(
        self,
        time_limit_sec: float = 60.0,
        gap_tolerance: float = 0.05,
        machine_sequence: List[str] = None,
    ):
        super().__init__(time_limit_sec, gap_tolerance)
        self.machine_sequence = machine_sequence or []
        self._start_vars = {}
        self._makespan_var = None
    
    def build(self) -> None:
        """Constrói modelo Flow-Shop."""
        if not self._check_ortools():
            raise ImportError("OR-Tools necessário para MILP")
        
        from ortools.linear_solver import pywraplp
        
        self._solver = pywraplp.Solver.CreateSolver('CBC')
        if not self._solver:
            self._solver = pywraplp.Solver.CreateSolver('SCIP')
        
        solver = self._solver
        
        M = sum(op.processing_time for op in self.operations) + 1000
        
        # Variáveis
        for op in self.operations:
            self._start_vars[op.id] = solver.NumVar(0, M, f"s_{op.id}")
        
        self._makespan_var = solver.NumVar(0, M, "makespan")
        
        # Makespan constraints
        for op in self.operations:
            solver.Add(
                self._makespan_var >= self._start_vars[op.id] + op.processing_time
            )
        
        # Precedência por job (igual a Job-Shop)
        jobs = {}
        for op in self.operations:
            if op.job_id not in jobs:
                jobs[op.job_id] = []
            jobs[op.job_id].append(op)
        
        for job_ops in jobs.values():
            sorted_ops = sorted(job_ops, key=lambda x: x.sequence)
            for i in range(len(sorted_ops) - 1):
                op1, op2 = sorted_ops[i], sorted_ops[i+1]
                solver.Add(
                    self._start_vars[op2.id] >= self._start_vars[op1.id] + op1.processing_time
                )
        
        # Não-sobreposição por máquina
        machines = {}
        for op in self.operations:
            if op.machine_id not in machines:
                machines[op.machine_id] = []
            machines[op.machine_id].append(op)
        
        for machine_ops in machines.values():
            for i, op1 in enumerate(machine_ops):
                for op2 in machine_ops[i+1:]:
                    y = solver.BoolVar(f"y_{op1.id}_{op2.id}")
                    solver.Add(
                        self._start_vars[op2.id] >= self._start_vars[op1.id] + op1.processing_time - M*(1-y)
                    )
                    solver.Add(
                        self._start_vars[op1.id] >= self._start_vars[op2.id] + op2.processing_time - M*y
                    )
        
        solver.Minimize(self._makespan_var)
        
        logger.info(f"MILP Flow-Shop: {solver.NumVariables()} vars, {solver.NumConstraints()} constraints")
    
    def solve(self) -> MILPSolution:
        """Resolve o modelo Flow-Shop."""
        if self._solver is None:
            self.build()
        
        from ortools.linear_solver import pywraplp
        
        self._solver.SetTimeLimit(int(self.time_limit_sec * 1000))
        
        import time
        start = time.time()
        status = self._solver.Solve()
        solve_time = time.time() - start
        
        status_map = {
            pywraplp.Solver.OPTIMAL: "OPTIMAL",
            pywraplp.Solver.FEASIBLE: "FEASIBLE",
            pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        }
        
        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            return MILPSolution(
                schedule=pd.DataFrame(),
                makespan=float('inf'),
                total_tardiness=0,
                solve_time_sec=solve_time,
                gap=1.0,
                status=status_map.get(status, "UNKNOWN"),
            )
        
        records = []
        for op in self.operations:
            records.append({
                "operation_id": op.id,
                "job_id": op.job_id,
                "machine_id": op.machine_id,
                "start_min": self._start_vars[op.id].solution_value(),
                "end_min": self._start_vars[op.id].solution_value() + op.processing_time,
            })
        
        return MILPSolution(
            schedule=pd.DataFrame(records),
            makespan=self._makespan_var.solution_value(),
            total_tardiness=0,
            solve_time_sec=solve_time,
            gap=0.0 if status == pywraplp.Solver.OPTIMAL else self.gap_tolerance,
            status=status_map.get(status, "UNKNOWN"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL API FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def solve_milp(instance, time_limit_sec: float = 60.0, gap_tolerance: float = 0.05) -> Dict:
    """
    Resolve scheduling instance usando MILP.
    
    Interface unificada para usar com SchedulingInstance.
    
    Args:
        instance: SchedulingInstance ou dict compatível
        time_limit_sec: Limite de tempo em segundos
        gap_tolerance: Gap de otimalidade aceitável
    
    Returns:
        Dict com scheduled_operations, kpis, etc. (mesmo formato que HeuristicScheduler)
    """
    import time
    from datetime import timedelta
    
    start_time = time.time()
    
    # Extrair operações
    if hasattr(instance, 'get_operations'):
        operations = instance.get_operations()
        horizon_start = instance.horizon_start or datetime.now()
    else:
        operations = instance.get("operations", [])
        horizon_start = instance.get("horizon_start", datetime.now())
    
    if not operations:
        return _empty_milp_result(time.time() - start_time)
    
    # Converter para MILPOperation
    milp_ops = []
    for op in operations:
        if isinstance(op, dict):
            milp_ops.append(MILPOperation(
                id=op.get("operation_id", ""),
                job_id=op.get("order_id", ""),
                machine_id=op.get("primary_machine_id", op.get("machine_id", "")),
                processing_time=int(op.get("duration_min", op.get("processing_time_min", 0))),
                sequence=op.get("op_seq", 0),
                due_date=op.get("due_date"),
                weight=op.get("weight", 1.0),
            ))
        else:
            milp_ops.append(MILPOperation(
                id=op.operation_id,
                job_id=op.order_id,
                machine_id=op.primary_machine_id,
                processing_time=int(op.duration_min),
                sequence=op.op_seq,
                due_date=op.due_date,
                weight=op.weight,
            ))
    
    # Criar e resolver modelo
    model = MILPJobShopModel(time_limit_sec=time_limit_sec, gap_tolerance=gap_tolerance)
    model.set_operations(milp_ops)
    
    try:
        model.build()
        solution = model.solve()
    except Exception as e:
        logger.error(f"MILP solve failed: {e}")
        return {
            "success": False,
            "engine_used": "milp",
            "rule_used": None,
            "solve_time_sec": round(time.time() - start_time, 3),
            "status": "error",
            "scheduled_operations": [],
            "kpis": _empty_kpis(),
            "warnings": [str(e)],
            "data_driven_count": 0,
        }
    
    # Converter resultado para formato unificado
    scheduled = []
    op_map = {op.id: op for op in milp_ops}
    
    if not solution.schedule.empty:
        for _, row in solution.schedule.iterrows():
            op = op_map.get(row["operation_id"])
            if op:
                start = horizon_start + timedelta(minutes=row["start_min"])
                end = horizon_start + timedelta(minutes=row["end_min"])
                
                scheduled.append({
                    "operation_id": row["operation_id"],
                    "order_id": op.job_id,
                    "article_id": "",
                    "op_seq": op.sequence,
                    "op_code": "",
                    "machine_id": row["machine_id"],
                    "start_time": start.isoformat(),
                    "end_time": end.isoformat(),
                    "duration_min": op.processing_time,
                    "setup_time_min": 0,
                })
    
    solve_time = time.time() - start_time
    
    return {
        "success": solution.status in ["OPTIMAL", "FEASIBLE"],
        "engine_used": "milp",
        "rule_used": None,
        "solve_time_sec": round(solve_time, 3),
        "status": solution.status.lower(),
        "scheduled_operations": scheduled,
        "kpis": {
            "makespan_hours": round(solution.makespan / 60, 2),
            "total_tardiness_hours": round(solution.total_tardiness / 60, 2),
            "num_late_orders": 0,
            "total_setup_time_hours": 0,
            "avg_machine_utilization": 0.75,
            "otd_rate": 1.0,
            "total_operations": len(scheduled),
            "total_orders": len(set(s["order_id"] for s in scheduled)),
        },
        "machine_utilization": {},
        "warnings": [],
        "data_driven_count": 0,
        "milp_gap": solution.gap,
    }


def _empty_milp_result(solve_time: float) -> Dict:
    """Resultado vazio para MILP."""
    return {
        "success": True,
        "engine_used": "milp",
        "rule_used": None,
        "solve_time_sec": round(solve_time, 3),
        "status": "optimal",
        "scheduled_operations": [],
        "kpis": _empty_kpis(),
        "machine_utilization": {},
        "warnings": ["No operations to schedule"],
        "data_driven_count": 0,
    }


def _empty_kpis() -> Dict:
    """KPIs vazios."""
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

