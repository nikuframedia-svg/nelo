"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    COMPARATIVE GANTT CHART DATA GENERATOR
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Generates data for comparing two planning scenarios side-by-side in a Gantt chart.
Color coding:
- Green: Operation in original position
- Blue: Operation in new position  
- Red: Operation changed (sequence or machine)
- Orange: New operation
- Gray: Removed operation
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


@dataclass
class GanttBar:
    """Single bar in the Gantt chart."""
    op_id: str
    order_id: str
    article_id: str
    machine_id: str
    start_time: datetime
    end_time: datetime
    duration_min: float
    color: str  # green, blue, red, orange, gray
    status: str  # original, moved, changed_machine, new, removed
    tooltip: str
    # Comparison data
    original_machine: Optional[str] = None
    original_start: Optional[datetime] = None
    delta_hours: float = 0.0
    sequence_change: int = 0  # +/- positions


@dataclass
class MachineGantt:
    """All bars for a single machine."""
    machine_id: str
    baseline_bars: List[GanttBar] = field(default_factory=list)
    scenario_bars: List[GanttBar] = field(default_factory=list)


@dataclass
class GanttComparisonData:
    """Complete comparative Gantt data."""
    baseline_name: str
    scenario_name: str
    machines: List[MachineGantt] = field(default_factory=list)
    time_range: Tuple[datetime, datetime] = field(default_factory=lambda: (datetime.now(), datetime.now()))
    summary: Dict[str, Any] = field(default_factory=dict)
    legend: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_name": self.baseline_name,
            "scenario_name": self.scenario_name,
            "time_range": {
                "start": self.time_range[0].isoformat(),
                "end": self.time_range[1].isoformat(),
            },
            "machines": [
                {
                    "machine_id": m.machine_id,
                    "baseline_bars": [
                        {
                            "op_id": b.op_id,
                            "order_id": b.order_id,
                            "article_id": b.article_id,
                            "start": b.start_time.isoformat(),
                            "end": b.end_time.isoformat(),
                            "duration_min": b.duration_min,
                            "color": b.color,
                            "status": b.status,
                            "tooltip": b.tooltip,
                        }
                        for b in m.baseline_bars
                    ],
                    "scenario_bars": [
                        {
                            "op_id": b.op_id,
                            "order_id": b.order_id,
                            "article_id": b.article_id,
                            "start": b.start_time.isoformat(),
                            "end": b.end_time.isoformat(),
                            "duration_min": b.duration_min,
                            "color": b.color,
                            "status": b.status,
                            "tooltip": b.tooltip,
                            "original_machine": b.original_machine,
                            "delta_hours": b.delta_hours,
                            "sequence_change": b.sequence_change,
                        }
                        for b in m.scenario_bars
                    ],
                }
                for m in self.machines
            ],
            "summary": self.summary,
            "legend": self.legend,
        }


def generate_comparative_gantt_data(
    baseline_plan: pd.DataFrame,
    scenario_plan: pd.DataFrame,
    baseline_name: str = "Plano Original",
    scenario_name: str = "Novo Plano",
) -> GanttComparisonData:
    """
    Generate comparative Gantt chart data.
    
    Args:
        baseline_plan: Original plan DataFrame
        scenario_plan: New scenario plan DataFrame
        baseline_name: Name for baseline
        scenario_name: Name for scenario
        
    Returns:
        GanttComparisonData with all visualization data
    """
    # Parse dates
    for df in [baseline_plan, scenario_plan]:
        if 'start_time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['start_time']):
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['end_time'] = pd.to_datetime(df['end_time'])
    
    # Get all machines
    all_machines = sorted(set(baseline_plan['machine_id'].unique()) | set(scenario_plan['machine_id'].unique()))
    
    # Calculate time range
    min_time = min(
        baseline_plan['start_time'].min() if not baseline_plan.empty else datetime.now(),
        scenario_plan['start_time'].min() if not scenario_plan.empty else datetime.now()
    )
    max_time = max(
        baseline_plan['end_time'].max() if not baseline_plan.empty else datetime.now(),
        scenario_plan['end_time'].max() if not scenario_plan.empty else datetime.now()
    )
    
    # Build operation lookup from baseline
    baseline_ops = {}
    for idx, row in baseline_plan.iterrows():
        op_key = f"{row.get('order_id', idx)}_{row.get('op_seq', idx)}"
        baseline_ops[op_key] = {
            'machine_id': row['machine_id'],
            'start_time': row['start_time'],
            'end_time': row['end_time'],
            'sequence_idx': idx,
        }
    
    # Build operation lookup from scenario
    scenario_ops = {}
    for idx, row in scenario_plan.iterrows():
        op_key = f"{row.get('order_id', idx)}_{row.get('op_seq', idx)}"
        scenario_ops[op_key] = {
            'machine_id': row['machine_id'],
            'start_time': row['start_time'],
            'end_time': row['end_time'],
            'sequence_idx': idx,
        }
    
    # Track changes
    changed_ops = 0
    moved_ops = 0
    new_ops = 0
    removed_ops = 0
    advanced_ops = 0
    delayed_ops = 0
    
    machine_gantts = []
    
    for machine_id in all_machines:
        mg = MachineGantt(machine_id=machine_id)
        
        # Baseline bars
        baseline_machine_ops = baseline_plan[baseline_plan['machine_id'] == machine_id]
        for idx, row in baseline_machine_ops.iterrows():
            op_key = f"{row.get('order_id', idx)}_{row.get('op_seq', idx)}"
            
            bar = GanttBar(
                op_id=op_key,
                order_id=str(row.get('order_id', '')),
                article_id=str(row.get('article_id', '')),
                machine_id=machine_id,
                start_time=row['start_time'],
                end_time=row['end_time'],
                duration_min=float(row.get('duration_min', 0)),
                color='green',
                status='original',
                tooltip=f"{row.get('article_id', '')} - {row.get('order_id', '')}",
            )
            mg.baseline_bars.append(bar)
        
        # Scenario bars
        scenario_machine_ops = scenario_plan[scenario_plan['machine_id'] == machine_id]
        for idx, row in scenario_machine_ops.iterrows():
            op_key = f"{row.get('order_id', idx)}_{row.get('op_seq', idx)}"
            
            # Check if this op existed in baseline
            baseline_info = baseline_ops.get(op_key)
            
            if baseline_info is None:
                # New operation
                color = 'orange'
                status = 'new'
                original_machine = None
                delta_hours = 0
                sequence_change = 0
                new_ops += 1
            elif baseline_info['machine_id'] != machine_id:
                # Changed machine
                color = 'red'
                status = 'changed_machine'
                original_machine = baseline_info['machine_id']
                delta_hours = (row['start_time'] - baseline_info['start_time']).total_seconds() / 3600
                sequence_change = 0
                changed_ops += 1
                if delta_hours < 0:
                    advanced_ops += 1
                elif delta_hours > 0:
                    delayed_ops += 1
            else:
                # Same machine - check timing
                delta_hours = (row['start_time'] - baseline_info['start_time']).total_seconds() / 3600
                
                if abs(delta_hours) < 0.1:  # Less than 6 minutes difference
                    color = 'green'
                    status = 'original'
                else:
                    color = 'blue'
                    status = 'moved'
                    moved_ops += 1
                    if delta_hours < 0:
                        advanced_ops += 1
                    else:
                        delayed_ops += 1
                
                original_machine = None
                sequence_change = int(idx - baseline_info['sequence_idx'])
            
            bar = GanttBar(
                op_id=op_key,
                order_id=str(row.get('order_id', '')),
                article_id=str(row.get('article_id', '')),
                machine_id=machine_id,
                start_time=row['start_time'],
                end_time=row['end_time'],
                duration_min=float(row.get('duration_min', 0)),
                color=color,
                status=status,
                tooltip=f"{row.get('article_id', '')} - {row.get('order_id', '')}",
                original_machine=original_machine,
                delta_hours=round(delta_hours, 2),
                sequence_change=sequence_change,
            )
            mg.scenario_bars.append(bar)
        
        machine_gantts.append(mg)
    
    # Count removed operations
    for op_key in baseline_ops:
        if op_key not in scenario_ops:
            removed_ops += 1
    
    # Build summary
    summary = {
        "total_baseline_ops": len(baseline_plan),
        "total_scenario_ops": len(scenario_plan),
        "changed_machine": changed_ops,
        "moved_time": moved_ops,
        "new_ops": new_ops,
        "removed_ops": removed_ops,
        "advanced_ops": advanced_ops,
        "delayed_ops": delayed_ops,
        "unchanged_ops": len(baseline_plan) - changed_ops - moved_ops - removed_ops,
    }
    
    # Legend
    legend = [
        {"color": "green", "label": "Posição original (sem alteração)"},
        {"color": "blue", "label": "Reposicionado no tempo (mesma máquina)"},
        {"color": "red", "label": "Alterado de máquina"},
        {"color": "orange", "label": "Nova operação"},
        {"color": "gray", "label": "Operação removida"},
    ]
    
    return GanttComparisonData(
        baseline_name=baseline_name,
        scenario_name=scenario_name,
        machines=machine_gantts,
        time_range=(min_time, max_time),
        summary=summary,
        legend=legend,
    )



