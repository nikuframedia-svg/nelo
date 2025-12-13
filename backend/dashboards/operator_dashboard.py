"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    OPERATOR WORKLOAD DASHBOARD GENERATOR
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Generates operator workload data:
- Hours allocated vs available per shift
- Utilization status (overloaded/underutilized)
- Skills distribution per machine
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
import pandas as pd
import numpy as np


@dataclass
class ShiftWorkload:
    """Workload for a specific shift."""
    shift_name: str  # Manhã, Tarde, Noite
    day_of_week: int
    hours_allocated: float
    hours_available: float
    utilization_pct: float
    status: str  # normal, overloaded, underutilized


@dataclass
class OperatorMetrics:
    """Complete metrics for an operator."""
    operator_id: str
    operator_name: str
    total_hours_allocated: float
    total_hours_available: float
    utilization_pct: float
    status: str  # normal, overloaded, underutilized
    skills: List[str]  # Machine IDs they can operate
    skill_levels: Dict[str, float]  # Machine ID -> proficiency 0-1
    weekly_workload: List[ShiftWorkload]
    daily_breakdown: Dict[str, float]  # Day -> hours
    trend: str  # increasing, stable, decreasing


@dataclass
class SkillGap:
    """Identified skill gap."""
    machine_id: str
    required_hours: float
    available_operators: int
    qualified_hours: float
    gap_hours: float
    severity: str  # low, medium, high, critical
    recommendation: str


@dataclass
class OperatorDashboard:
    """Complete operator dashboard data."""
    operators: List[OperatorMetrics]
    skill_distribution: Dict[str, List[str]]  # Machine -> operators
    skill_gaps: List[SkillGap]
    total_operators: int
    overloaded_count: int
    underutilized_count: int
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operators": [
                {
                    "operator_id": op.operator_id,
                    "name": op.operator_name,
                    "hours_allocated": round(float(op.total_hours_allocated), 1),
                    "hours_available": round(float(op.total_hours_available), 1),
                    "utilization_pct": round(float(op.utilization_pct), 1),
                    "status": op.status,
                    "skills": op.skills,
                    "skill_levels": {k: round(float(v), 2) for k, v in op.skill_levels.items()},
                    "weekly_workload": [
                        {
                            "shift": w.shift_name,
                            "day": int(w.day_of_week),
                            "hours_allocated": round(float(w.hours_allocated), 1),
                            "hours_available": round(float(w.hours_available), 1),
                            "utilization": round(float(w.utilization_pct), 1),
                            "status": w.status,
                        }
                        for w in op.weekly_workload
                    ],
                    "daily_breakdown": {k: round(float(v), 1) for k, v in op.daily_breakdown.items()},
                    "trend": op.trend,
                }
                for op in self.operators
            ],
            "skill_distribution": self.skill_distribution,
            "skill_gaps": [
                {
                    "machine_id": g.machine_id,
                    "required_hours": round(float(g.required_hours), 1),
                    "available_operators": int(g.available_operators),
                    "qualified_hours": round(float(g.qualified_hours), 1),
                    "gap_hours": round(float(g.gap_hours), 1),
                    "severity": g.severity,
                    "recommendation": g.recommendation,
                }
                for g in self.skill_gaps
            ],
            "summary": {
                "total_operators": int(self.total_operators),
                "overloaded_count": int(self.overloaded_count),
                "underutilized_count": int(self.underutilized_count),
                "avg_utilization": round(float(self.summary.get("avg_utilization", 0)), 1),
                "critical_gaps": int(sum(1 for g in self.skill_gaps if g.severity == "critical")),
            },
        }


def generate_operator_dashboard(
    plan_df: pd.DataFrame,
    operators_df: Optional[pd.DataFrame] = None,
    shifts_df: Optional[pd.DataFrame] = None,
    week_hours: float = 40.0,
) -> OperatorDashboard:
    """
    Generate operator workload dashboard.
    
    Args:
        plan_df: Production plan DataFrame
        operators_df: Operators DataFrame (optional, will generate synthetic if missing)
        shifts_df: Shifts configuration (optional)
        week_hours: Standard work week hours
        
    Returns:
        OperatorDashboard with complete visualization data
    """
    machines = plan_df['machine_id'].unique().tolist() if not plan_df.empty else []
    
    # Generate synthetic operators if not provided
    if operators_df is None or operators_df.empty:
        operators_df = _generate_synthetic_operators(machines)
    
    # Build operator skills lookup
    operator_skills: Dict[str, Set[str]] = {}
    operator_levels: Dict[str, Dict[str, float]] = {}
    
    for _, row in operators_df.iterrows():
        op_id = row.get('operator_id', row.get('worker_id', f"OP-{_}"))
        skills = row.get('skills', row.get('machines', machines[:2]))
        
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(',')]
        
        operator_skills[op_id] = set(skills)
        operator_levels[op_id] = {m: np.random.uniform(0.7, 1.0) for m in skills}
    
    # Calculate machine hours from plan
    machine_hours = {}
    if not plan_df.empty:
        for machine in machines:
            m_ops = plan_df[plan_df['machine_id'] == machine]
            machine_hours[machine] = m_ops['duration_min'].sum() / 60 if 'duration_min' in m_ops.columns else 0
    
    # Allocate work to operators
    operator_work: Dict[str, float] = {op_id: 0.0 for op_id in operator_skills}
    operator_daily: Dict[str, Dict[str, float]] = {op_id: {} for op_id in operator_skills}
    
    days = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    
    for machine, hours in machine_hours.items():
        # Find qualified operators
        qualified = [op for op, skills in operator_skills.items() if machine in skills]
        
        if qualified:
            # Distribute hours among qualified operators
            hours_per_op = hours / len(qualified)
            for op in qualified:
                operator_work[op] += hours_per_op
                
                # Distribute across days
                daily_hours = hours_per_op / 5  # Assume 5-day week
                for day in days[:5]:
                    operator_daily[op][day] = operator_daily[op].get(day, 0) + daily_hours
    
    # Build operator metrics
    operators_list = []
    overloaded = 0
    underutilized = 0
    
    for op_id in operator_skills:
        op_row = operators_df[operators_df.get('operator_id', operators_df.get('worker_id', 'x')) == op_id]
        name = op_row['name'].iloc[0] if not op_row.empty and 'name' in op_row.columns else f"Operador {op_id}"
        
        hours_allocated = operator_work.get(op_id, 0)
        hours_available = week_hours
        util_pct = (hours_allocated / hours_available * 100) if hours_available > 0 else 0
        
        if util_pct > 100:
            status = "overloaded"
            overloaded += 1
        elif util_pct < 50:
            status = "underutilized"
            underutilized += 1
        else:
            status = "normal"
        
        # Weekly workload (synthetic shifts)
        weekly = []
        shifts = ['Manhã', 'Tarde', 'Noite']
        for day_idx in range(5):
            shift_hours = hours_allocated / 5 / len(shifts) if hours_allocated > 0 else 0
            for shift in shifts[:2]:  # Only morning and afternoon typically
                weekly.append(ShiftWorkload(
                    shift_name=shift,
                    day_of_week=day_idx,
                    hours_allocated=shift_hours,
                    hours_available=4.0,  # 4 hours per shift
                    utilization_pct=(shift_hours / 4.0 * 100) if shift_hours else 0,
                    status="normal" if shift_hours <= 4.5 else "overloaded",
                ))
        
        operators_list.append(OperatorMetrics(
            operator_id=op_id,
            operator_name=name,
            total_hours_allocated=hours_allocated,
            total_hours_available=hours_available,
            utilization_pct=util_pct,
            status=status,
            skills=list(operator_skills.get(op_id, [])),
            skill_levels=operator_levels.get(op_id, {}),
            weekly_workload=weekly,
            daily_breakdown=operator_daily.get(op_id, {}),
            trend="stable",
        ))
    
    # Build skill distribution
    skill_dist: Dict[str, List[str]] = {}
    for machine in machines:
        skill_dist[machine] = [
            op_id for op_id, skills in operator_skills.items()
            if machine in skills
        ]
    
    # Identify skill gaps
    skill_gaps = []
    for machine in machines:
        required_hours = machine_hours.get(machine, 0)
        qualified_ops = skill_dist.get(machine, [])
        qualified_hours = len(qualified_ops) * week_hours
        gap = max(0, required_hours - qualified_hours)
        
        if gap > 0:
            if gap > required_hours * 0.5:
                severity = "critical"
            elif gap > required_hours * 0.25:
                severity = "high"
            elif gap > required_hours * 0.1:
                severity = "medium"
            else:
                severity = "low"
            
            skill_gaps.append(SkillGap(
                machine_id=machine,
                required_hours=required_hours,
                available_operators=len(qualified_ops),
                qualified_hours=qualified_hours,
                gap_hours=gap,
                severity=severity,
                recommendation=f"Formar {max(1, int(np.ceil(gap/week_hours)))} operador(es) adicional(is) para {machine}.",
            ))
    
    # Summary
    avg_util = np.mean([op.utilization_pct for op in operators_list]) if operators_list else 0
    
    return OperatorDashboard(
        operators=operators_list,
        skill_distribution=skill_dist,
        skill_gaps=skill_gaps,
        total_operators=len(operators_list),
        overloaded_count=overloaded,
        underutilized_count=underutilized,
        summary={
            "avg_utilization": avg_util,
            "total_hours_allocated": sum(op.total_hours_allocated for op in operators_list),
            "total_hours_available": sum(op.total_hours_available for op in operators_list),
        },
    )


def _generate_synthetic_operators(machines: List[str], count: int = 10) -> pd.DataFrame:
    """Generate synthetic operator data."""
    operators = []
    
    names = [
        "João Silva", "Maria Santos", "Pedro Costa", "Ana Oliveira", "Carlos Ferreira",
        "Sofia Pereira", "Miguel Rodrigues", "Inês Martins", "Rui Almeida", "Catarina Sousa",
        "André Gomes", "Beatriz Fernandes", "Tiago Ribeiro", "Marta Carvalho", "Luís Nunes",
    ]
    
    for i in range(min(count, len(names))):
        # Each operator qualified for 2-4 machines
        num_skills = np.random.randint(2, min(5, len(machines) + 1))
        skills = np.random.choice(machines, size=min(num_skills, len(machines)), replace=False).tolist()
        
        operators.append({
            "operator_id": f"OP-{i+1:03d}",
            "name": names[i],
            "skills": skills,
        })
    
    return pd.DataFrame(operators)

