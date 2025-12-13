"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    OPERATOR ALLOCATOR — Skill-Based Resource Assignment
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Allocates operators to machines/operations based on skills and availability.

Mathematical Model:
─────────────────────────────────────────────────────────────────────────────────────────────────────

Sets:
    O = {1, ..., n}     : Operations
    W = {1, ..., m}     : Workers/Operators
    T = {1, ..., t}     : Time slots/shifts
    S_w ⊆ Machines      : Skills of worker w

Variables:
    x_{o,w,t} ∈ {0,1}   : 1 if operation o assigned to worker w in slot t

Constraints:
    (1) Assignment: Σ_w Σ_t x_{o,w,t} = 1                    ∀o (each op assigned once)
    (2) Skill: x_{o,w,t} ≤ skill_{w,m(o)}                    (worker must have skill)
    (3) Availability: Σ_o x_{o,w,t} ≤ available_{w,t}       (worker availability)
    (4) Capacity: Operations limited by worker availability

Objective:
    Maximize: Σ skill_score × assignment
    or
    Minimize: Unassigned operations + idle worker time

═══════════════════════════════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ShiftType(str, Enum):
    """Shift types."""
    MORNING = "morning"      # 06:00 - 14:00
    AFTERNOON = "afternoon"  # 14:00 - 22:00
    NIGHT = "night"          # 22:00 - 06:00
    FULL_DAY = "full_day"    # 08:00 - 17:00


@dataclass
class OperatorSkill:
    """
    Operator skill definition.
    """
    operator_id: str
    operator_name: str
    
    # Machines this operator can work on
    qualified_machines: Set[str] = field(default_factory=set)
    
    # Skill levels per machine (0-1)
    skill_levels: Dict[str, float] = field(default_factory=dict)
    
    # Operation types the operator can perform
    qualified_operations: Set[str] = field(default_factory=set)
    
    # Availability
    available_shifts: List[ShiftType] = field(default_factory=list)
    shift_schedule: Dict[str, List[ShiftType]] = field(default_factory=dict)  # date -> shifts
    
    # Productivity factor (1.0 = standard)
    productivity_factor: float = 1.0
    
    def is_qualified_for(self, machine_id: str, operation_code: Optional[str] = None) -> bool:
        """Check if operator is qualified for a machine/operation."""
        if machine_id not in self.qualified_machines:
            return False
        
        if operation_code and self.qualified_operations:
            return operation_code in self.qualified_operations
        
        return True
    
    def get_skill_level(self, machine_id: str) -> float:
        """Get skill level for a machine (0-1)."""
        return self.skill_levels.get(machine_id, 0.5)
    
    def is_available(self, date: str, shift: ShiftType) -> bool:
        """Check if operator is available for a specific shift."""
        if date in self.shift_schedule:
            return shift in self.shift_schedule[date]
        return shift in self.available_shifts


@dataclass
class OperatorAssignment:
    """
    Assignment of an operator to an operation.
    """
    operation_id: str
    operator_id: str
    machine_id: str
    shift: ShiftType
    date: str
    start_time: datetime
    end_time: datetime
    
    skill_level: float = 0.5
    productivity_factor: float = 1.0


@dataclass
class AllocationResult:
    """Result of operator allocation."""
    assignments: List[OperatorAssignment]
    unassigned_operations: List[str]
    idle_operators: Dict[str, float]  # operator_id -> idle hours
    
    utilization_by_operator: Dict[str, float] = field(default_factory=dict)
    coverage_pct: float = 0.0
    
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class OperatorAllocator:
    """
    Allocator for assigning operators to operations based on skills.
    """
    
    def __init__(self, operators: List[OperatorSkill]):
        self.operators = {op.operator_id: op for op in operators}
    
    def allocate(
        self,
        operations: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime,
        require_all_assigned: bool = False,
    ) -> AllocationResult:
        """
        Allocate operators to operations.
        
        Args:
            operations: List of operations with machine_id, duration_min, op_code
            start_date: Start of planning horizon
            end_date: End of planning horizon
            require_all_assigned: If True, fail if any operation unassigned
            
        Returns:
            AllocationResult
        """
        assignments = []
        unassigned = []
        operator_hours = {op_id: 0.0 for op_id in self.operators}
        
        # Sort operations by priority/due date
        sorted_ops = sorted(
            operations,
            key=lambda op: (
                -op.get('priority', 1.0),
                op.get('due_date', datetime.max)
            )
        )
        
        for op in sorted_ops:
            machine_id = op['machine_id']
            duration_min = op['duration_min']
            op_code = op.get('op_code')
            
            # Find qualified operators
            qualified = []
            for op_skill in self.operators.values():
                if op_skill.is_qualified_for(machine_id, op_code):
                    qualified.append(op_skill)
            
            if not qualified:
                unassigned.append(op['op_id'])
                continue
            
            # Sort by skill level (highest first)
            qualified.sort(
                key=lambda os: os.get_skill_level(machine_id),
                reverse=True
            )
            
            # Try to assign to best available operator
            assigned = False
            for op_skill in qualified:
                # Simple assignment (improve with shift/availability check)
                assignment = OperatorAssignment(
                    operation_id=op['op_id'],
                    operator_id=op_skill.operator_id,
                    machine_id=machine_id,
                    shift=ShiftType.FULL_DAY,
                    date=start_date.strftime("%Y-%m-%d"),
                    start_time=start_date,
                    end_time=start_date + timedelta(minutes=duration_min),
                    skill_level=op_skill.get_skill_level(machine_id),
                    productivity_factor=op_skill.productivity_factor,
                )
                
                assignments.append(assignment)
                operator_hours[op_skill.operator_id] += duration_min / 60
                assigned = True
                break
            
            if not assigned:
                unassigned.append(op['op_id'])
        
        # Calculate utilization
        total_hours = (end_date - start_date).total_seconds() / 3600
        utilization = {}
        idle = {}
        
        for op_id, hours in operator_hours.items():
            utilization[op_id] = (hours / total_hours * 100) if total_hours > 0 else 0
            idle[op_id] = max(0, total_hours - hours)
        
        # Coverage
        coverage = (len(assignments) / len(operations) * 100) if operations else 100
        
        # Generate warnings and recommendations
        warnings = []
        recommendations = []
        
        if unassigned:
            warnings.append(f"{len(unassigned)} operações sem operador qualificado atribuído.")
        
        # Check for bottleneck operators
        for op_id, util in utilization.items():
            if util > 90:
                warnings.append(f"Operador {op_id} com utilização >90% ({util:.0f}%).")
                recommendations.append(f"Considere formar mais operadores para as máquinas de {op_id}.")
        
        # Check for idle operators
        for op_id, idle_h in idle.items():
            if idle_h > total_hours * 0.5:
                recommendations.append(
                    f"Operador {op_id} com {idle_h:.1f}h de tempo livre. "
                    "Considere atribuir formação ou tarefas adicionais."
                )
        
        return AllocationResult(
            assignments=assignments,
            unassigned_operations=unassigned,
            idle_operators=idle,
            utilization_by_operator=utilization,
            coverage_pct=coverage,
            warnings=warnings,
            recommendations=recommendations,
        )
    
    def find_skill_gaps(
        self,
        operations: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """
        Identify machines/operations without qualified operators.
        
        Returns:
            Dict[machine_id] -> list of missing skills
        """
        gaps = {}
        
        machines_needed = set(op['machine_id'] for op in operations)
        op_codes_needed = set(op.get('op_code', '') for op in operations if op.get('op_code'))
        
        for machine in machines_needed:
            qualified_count = sum(
                1 for op in self.operators.values()
                if machine in op.qualified_machines
            )
            
            if qualified_count == 0:
                gaps[machine] = ["Nenhum operador qualificado"]
            elif qualified_count == 1:
                gaps[machine] = ["Apenas 1 operador qualificado (risco)"]
        
        return gaps
    
    def recommend_training(
        self,
        operations: List[Dict[str, Any]],
        max_recommendations: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Recommend training to improve coverage.
        
        Returns list of training recommendations.
        """
        recommendations = []
        
        # Count operations per machine
        machine_ops = {}
        for op in operations:
            m = op['machine_id']
            machine_ops[m] = machine_ops.get(m, 0) + 1
        
        # Find understaffed machines
        for machine, count in sorted(machine_ops.items(), key=lambda x: -x[1]):
            qualified = [
                op for op in self.operators.values()
                if machine in op.qualified_machines
            ]
            
            if len(qualified) < 2 and count > 10:
                # Find operators who could be trained
                unqualified = [
                    op for op in self.operators.values()
                    if machine not in op.qualified_machines
                ]
                
                if unqualified:
                    # Prefer operators with related skills
                    best_candidate = max(
                        unqualified,
                        key=lambda op: len(op.qualified_machines)
                    )
                    
                    recommendations.append({
                        "machine_id": machine,
                        "operator_id": best_candidate.operator_id,
                        "operator_name": best_candidate.operator_name,
                        "reason": f"Máquina {machine} tem {count} operações mas apenas {len(qualified)} operadores.",
                        "priority": "HIGH" if count > 20 else "MEDIUM",
                    })
            
            if len(recommendations) >= max_recommendations:
                break
        
        return recommendations


def build_operators_from_excel(
    operators_df,
    skills_df,
) -> List[OperatorSkill]:
    """
    Build OperatorSkill objects from Excel data.
    
    Args:
        operators_df: DataFrame with operator_id, operator_name
        skills_df: DataFrame with operator_id, machine_id, skill_level
        
    Returns:
        List of OperatorSkill objects
    """
    operators = {}
    
    # Create base operators
    for _, row in operators_df.iterrows():
        op_id = row['operator_id']
        operators[op_id] = OperatorSkill(
            operator_id=op_id,
            operator_name=row.get('operator_name', op_id),
            qualified_machines=set(),
            skill_levels={},
            available_shifts=[ShiftType.MORNING, ShiftType.AFTERNOON],
        )
    
    # Add skills
    for _, row in skills_df.iterrows():
        op_id = row['operator_id']
        if op_id in operators:
            machine_id = row['machine_id']
            skill_level = row.get('skill_level', 0.5)
            
            operators[op_id].qualified_machines.add(machine_id)
            operators[op_id].skill_levels[machine_id] = skill_level
    
    return list(operators.values())



