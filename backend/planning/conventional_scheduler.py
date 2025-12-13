"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    CONVENTIONAL SCHEDULER — Independent Machine Scheduling
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Implements traditional independent scheduling where each machine optimizes locally.

Dispatching Rules:
- EDD (Earliest Due Date): Prioritize orders with earliest due dates
- SPT (Shortest Processing Time): Prioritize shortest operations first
- FIFO (First In First Out): Process in arrival order
- CR (Critical Ratio): d_j - t / p_j (remaining slack / processing time)
- WSPT (Weighted SPT): Priority / processing time

This serves as the baseline for comparison with chained planning.

═══════════════════════════════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd

from .planning_modes import (
    ConventionalPlanningConfig,
    PlanningResult,
    PlanningMode,
)

logger = logging.getLogger(__name__)


class DispatchingRule(str, Enum):
    """Available dispatching rules."""
    EDD = "EDD"      # Earliest Due Date
    SPT = "SPT"      # Shortest Processing Time
    FIFO = "FIFO"    # First In First Out
    CR = "CR"        # Critical Ratio
    WSPT = "WSPT"    # Weighted Shortest Processing Time
    SLACK = "SLACK"  # Minimum Slack


@dataclass
class Operation:
    """A single operation to be scheduled."""
    op_id: str
    order_id: str
    article_id: str
    op_code: str
    machine_id: str
    qty: float
    duration_min: float
    due_date: datetime
    priority: float = 1.0
    setup_family: str = "default"
    arrival_time: datetime = field(default_factory=datetime.now)
    
    # For tracking
    sequence_position: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class MachineSchedule:
    """Schedule for a single machine."""
    machine_id: str
    operations: List[Operation]
    
    total_processing_min: float = 0.0
    total_setup_min: float = 0.0
    total_idle_min: float = 0.0
    utilization_pct: float = 0.0
    makespan_min: float = 0.0


class ConventionalScheduler:
    """
    Conventional independent machine scheduler.
    
    Each machine schedules its operations independently using dispatching rules.
    """
    
    def __init__(self, config: ConventionalPlanningConfig):
        self.config = config
        self.rule = DispatchingRule(config.dispatching_rule)
    
    def schedule(
        self,
        operations: List[Operation],
        machines: List[str],
        start_time: datetime,
        setup_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, MachineSchedule]]:
        """
        Schedule operations across machines independently.
        
        Args:
            operations: List of operations to schedule
            machines: List of machine IDs
            start_time: Planning start time
            setup_matrix: Setup times between families (optional)
            
        Returns:
            (schedule_df, machine_schedules)
        """
        # Group operations by machine
        ops_by_machine: Dict[str, List[Operation]] = {m: [] for m in machines}
        
        for op in operations:
            if op.machine_id in ops_by_machine:
                ops_by_machine[op.machine_id].append(op)
        
        # Schedule each machine independently
        all_records = []
        machine_schedules = {}
        
        for machine_id, machine_ops in ops_by_machine.items():
            if not machine_ops:
                machine_schedules[machine_id] = MachineSchedule(machine_id=machine_id, operations=[])
                continue
            
            # Sort operations by dispatching rule
            sorted_ops = self._sort_operations(machine_ops, start_time)
            
            # Assign times
            current_time = start_time
            scheduled_ops = []
            total_setup = 0.0
            total_idle = 0.0
            prev_family = None
            
            for seq_pos, op in enumerate(sorted_ops):
                # Setup time
                setup_time = 0.0
                if setup_matrix and prev_family and op.setup_family != prev_family:
                    setup_time = setup_matrix.get(prev_family, {}).get(op.setup_family, 0)
                    total_setup += setup_time
                
                # Start after setup
                op_start = current_time + timedelta(minutes=setup_time)
                op_end = op_start + timedelta(minutes=op.duration_min)
                
                op.sequence_position = seq_pos + 1
                op.start_time = op_start
                op.end_time = op_end
                
                scheduled_ops.append(op)
                
                all_records.append({
                    "op_id": op.op_id,
                    "order_id": op.order_id,
                    "article_id": op.article_id,
                    "op_code": op.op_code,
                    "machine_id": machine_id,
                    "qty": op.qty,
                    "start_time": op_start,
                    "end_time": op_end,
                    "duration_min": op.duration_min,
                    "setup_min": setup_time,
                    "sequence_position": seq_pos + 1,
                })
                
                current_time = op_end
                prev_family = op.setup_family
            
            # Calculate machine metrics
            total_proc = sum(op.duration_min for op in scheduled_ops)
            makespan = (current_time - start_time).total_seconds() / 60 if scheduled_ops else 0
            utilization = (total_proc / makespan * 100) if makespan > 0 else 0
            
            machine_schedules[machine_id] = MachineSchedule(
                machine_id=machine_id,
                operations=scheduled_ops,
                total_processing_min=total_proc,
                total_setup_min=total_setup,
                total_idle_min=max(0, makespan - total_proc - total_setup),
                utilization_pct=utilization,
                makespan_min=makespan,
            )
        
        schedule_df = pd.DataFrame(all_records)
        return schedule_df, machine_schedules
    
    def _sort_operations(
        self,
        operations: List[Operation],
        current_time: datetime,
    ) -> List[Operation]:
        """Sort operations according to the dispatching rule."""
        
        if self.rule == DispatchingRule.EDD:
            # Earliest Due Date
            return sorted(operations, key=lambda op: op.due_date)
        
        elif self.rule == DispatchingRule.SPT:
            # Shortest Processing Time
            return sorted(operations, key=lambda op: op.duration_min)
        
        elif self.rule == DispatchingRule.FIFO:
            # First In First Out (by arrival time)
            return sorted(operations, key=lambda op: op.arrival_time)
        
        elif self.rule == DispatchingRule.CR:
            # Critical Ratio: (due_date - current_time) / processing_time
            def critical_ratio(op: Operation) -> float:
                slack = (op.due_date - current_time).total_seconds() / 60
                return slack / op.duration_min if op.duration_min > 0 else float('inf')
            return sorted(operations, key=critical_ratio)
        
        elif self.rule == DispatchingRule.WSPT:
            # Weighted Shortest Processing Time: priority / processing_time
            def wspt_score(op: Operation) -> float:
                return -op.priority / op.duration_min if op.duration_min > 0 else float('-inf')
            return sorted(operations, key=wspt_score)
        
        elif self.rule == DispatchingRule.SLACK:
            # Minimum Slack: due_date - current_time - processing_time
            def slack(op: Operation) -> float:
                return (op.due_date - current_time).total_seconds() / 60 - op.duration_min
            return sorted(operations, key=slack)
        
        else:
            # Default: FIFO
            return operations
    
    def schedule_with_setup_optimization(
        self,
        operations: List[Operation],
        machines: List[str],
        start_time: datetime,
        setup_matrix: Dict[str, Dict[str, float]],
    ) -> Tuple[pd.DataFrame, Dict[str, MachineSchedule]]:
        """
        Schedule with setup time minimization.
        
        Groups operations by setup family when possible without violating due dates.
        """
        # Group operations by machine
        ops_by_machine: Dict[str, List[Operation]] = {m: [] for m in machines}
        
        for op in operations:
            if op.machine_id in ops_by_machine:
                ops_by_machine[op.machine_id].append(op)
        
        all_records = []
        machine_schedules = {}
        
        for machine_id, machine_ops in ops_by_machine.items():
            if not machine_ops:
                machine_schedules[machine_id] = MachineSchedule(machine_id=machine_id, operations=[])
                continue
            
            # Group by setup family
            families = {}
            for op in machine_ops:
                if op.setup_family not in families:
                    families[op.setup_family] = []
                families[op.setup_family].append(op)
            
            # Sort within each family by due date
            for family in families.values():
                family.sort(key=lambda op: op.due_date)
            
            # Sequence families to minimize setup changes
            # Simple greedy: start with most urgent, then nearest neighbor
            sorted_ops = self._sequence_families_greedy(families, setup_matrix)
            
            # Assign times
            current_time = start_time
            scheduled_ops = []
            total_setup = 0.0
            prev_family = None
            
            for seq_pos, op in enumerate(sorted_ops):
                setup_time = 0.0
                if prev_family and op.setup_family != prev_family:
                    setup_time = setup_matrix.get(prev_family, {}).get(op.setup_family, 0)
                    total_setup += setup_time
                
                op_start = current_time + timedelta(minutes=setup_time)
                op_end = op_start + timedelta(minutes=op.duration_min)
                
                op.sequence_position = seq_pos + 1
                op.start_time = op_start
                op.end_time = op_end
                
                scheduled_ops.append(op)
                
                all_records.append({
                    "op_id": op.op_id,
                    "order_id": op.order_id,
                    "article_id": op.article_id,
                    "op_code": op.op_code,
                    "machine_id": machine_id,
                    "qty": op.qty,
                    "start_time": op_start,
                    "end_time": op_end,
                    "duration_min": op.duration_min,
                    "setup_min": setup_time,
                    "sequence_position": seq_pos + 1,
                })
                
                current_time = op_end
                prev_family = op.setup_family
            
            total_proc = sum(op.duration_min for op in scheduled_ops)
            makespan = (current_time - start_time).total_seconds() / 60 if scheduled_ops else 0
            utilization = (total_proc / makespan * 100) if makespan > 0 else 0
            
            machine_schedules[machine_id] = MachineSchedule(
                machine_id=machine_id,
                operations=scheduled_ops,
                total_processing_min=total_proc,
                total_setup_min=total_setup,
                total_idle_min=max(0, makespan - total_proc - total_setup),
                utilization_pct=utilization,
                makespan_min=makespan,
            )
        
        schedule_df = pd.DataFrame(all_records)
        return schedule_df, machine_schedules
    
    def _sequence_families_greedy(
        self,
        families: Dict[str, List[Operation]],
        setup_matrix: Dict[str, Dict[str, float]],
    ) -> List[Operation]:
        """
        Sequence operation families using greedy nearest neighbor.
        
        Starts with the family containing the most urgent operation,
        then selects the next family with minimum setup time.
        """
        if not families:
            return []
        
        # Find family with most urgent operation
        most_urgent_family = min(
            families.keys(),
            key=lambda f: min(op.due_date for op in families[f])
        )
        
        result = []
        remaining_families = set(families.keys())
        current_family = most_urgent_family
        
        while remaining_families:
            # Add all operations from current family
            result.extend(families[current_family])
            remaining_families.discard(current_family)
            
            if not remaining_families:
                break
            
            # Find nearest neighbor (minimum setup time)
            next_family = min(
                remaining_families,
                key=lambda f: setup_matrix.get(current_family, {}).get(f, float('inf'))
            )
            current_family = next_family
        
        return result


def build_operations_from_plan(
    orders_df: pd.DataFrame,
    routing_df: pd.DataFrame,
) -> List[Operation]:
    """
    Build Operation objects from orders and routing data.
    """
    operations = []
    
    for _, order in orders_df.iterrows():
        article_id = order['article_id']
        article_routing = routing_df[routing_df['article_id'] == article_id]
        
        for _, route_row in article_routing.iterrows():
            op_id = f"{order['order_id']}_{route_row['op_code']}"
            duration = route_row['base_time_per_unit_min'] * order['qty']
            
            due_date = pd.to_datetime(order['due_date']) if 'due_date' in order else datetime.now() + timedelta(days=30)
            
            operations.append(Operation(
                op_id=op_id,
                order_id=order['order_id'],
                article_id=article_id,
                op_code=route_row['op_code'],
                machine_id=route_row['primary_machine_id'],
                qty=order['qty'],
                duration_min=duration,
                due_date=due_date,
                priority=order.get('priority', 1.0),
                setup_family=route_row.get('setup_family', 'default'),
            ))
    
    return operations



