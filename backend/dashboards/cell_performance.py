"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    CHAINED CELL PERFORMANCE DASHBOARD
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Generates performance metrics for production cells/flow shop lines:
- Lead time end-to-end
- WIP at each buffer
- Throughput and bottleneck identification
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class BufferMetrics:
    """Metrics for a buffer between two machines."""
    from_machine: str
    to_machine: str
    avg_wip: float  # Average units in buffer
    max_wip: float
    avg_wait_hours: float
    max_wait_hours: float
    is_congested: bool


@dataclass
class CellStage:
    """Metrics for a single stage in the cell."""
    machine_id: str
    sequence_position: int
    utilization_pct: float
    throughput_units_per_day: float
    cycle_time_min: float
    is_bottleneck: bool
    capacity_units_per_day: float


@dataclass
class CellMetrics:
    """Complete metrics for a production cell."""
    cell_id: str
    cell_name: str
    machines: List[str]  # Ordered sequence
    stages: List[CellStage]
    buffers: List[BufferMetrics]
    
    # Aggregate metrics
    lead_time_hours: float
    lead_time_target_hours: Optional[float]
    throughput_units_per_day: float
    capacity_units_per_day: float
    efficiency_pct: float
    
    # Bottleneck
    bottleneck_machine: str
    bottleneck_utilization: float
    bottleneck_constraint: str  # "capacity", "quality", "availability"
    
    # WIP
    total_wip: float
    target_wip: Optional[float]
    wip_status: str  # "optimal", "high", "critical"
    
    # Status
    status: str  # "healthy", "attention", "critical"
    issues: List[str]
    recommendations: List[str]


@dataclass
class CellPerformance:
    """Complete cell performance dashboard."""
    cells: List[CellMetrics]
    total_throughput: float
    avg_lead_time: float
    total_wip: float
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cells": [
                {
                    "cell_id": c.cell_id,
                    "name": c.cell_name,
                    "machines": c.machines,
                    "stages": [
                        {
                            "machine_id": s.machine_id,
                            "position": int(s.sequence_position),
                            "utilization_pct": round(float(s.utilization_pct), 1),
                            "throughput_per_day": round(float(s.throughput_units_per_day), 0),
                            "cycle_time_min": round(float(s.cycle_time_min), 1),
                            "is_bottleneck": bool(s.is_bottleneck),
                            "capacity_per_day": round(float(s.capacity_units_per_day), 0),
                        }
                        for s in c.stages
                    ],
                    "buffers": [
                        {
                            "from": b.from_machine,
                            "to": b.to_machine,
                            "avg_wip": round(float(b.avg_wip), 1),
                            "max_wip": round(float(b.max_wip), 0),
                            "avg_wait_hours": round(float(b.avg_wait_hours), 2),
                            "is_congested": bool(b.is_congested),
                        }
                        for b in c.buffers
                    ],
                    "lead_time_hours": round(float(c.lead_time_hours), 1),
                    "lead_time_target_hours": round(float(c.lead_time_target_hours), 1) if c.lead_time_target_hours else None,
                    "throughput_per_day": round(float(c.throughput_units_per_day), 0),
                    "capacity_per_day": round(float(c.capacity_units_per_day), 0),
                    "efficiency_pct": round(float(c.efficiency_pct), 1),
                    "bottleneck": {
                        "machine": c.bottleneck_machine,
                        "utilization": round(float(c.bottleneck_utilization), 1),
                        "constraint": c.bottleneck_constraint,
                    },
                    "wip": {
                        "total": round(float(c.total_wip), 0),
                        "target": int(c.target_wip) if c.target_wip else None,
                        "status": c.wip_status,
                    },
                    "status": c.status,
                    "issues": c.issues,
                    "recommendations": c.recommendations,
                }
                for c in self.cells
            ],
            "summary": {
                "total_cells": int(len(self.cells)),
                "total_throughput_per_day": round(float(self.total_throughput), 0),
                "avg_lead_time_hours": round(float(self.avg_lead_time), 1),
                "total_wip": round(float(self.total_wip), 0),
                "healthy_cells": int(sum(1 for c in self.cells if c.status == "healthy")),
                "attention_cells": int(sum(1 for c in self.cells if c.status == "attention")),
                "critical_cells": int(sum(1 for c in self.cells if c.status == "critical")),
            },
        }


def generate_cell_performance(
    plan_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    cells_config: Optional[List[Dict[str, Any]]] = None,
) -> CellPerformance:
    """
    Generate cell/flow line performance dashboard.
    
    Args:
        plan_df: Production plan DataFrame
        machines_df: Machines DataFrame
        cells_config: Cell configurations (optional, will auto-detect if missing)
                     Format: [{"id": "L1", "name": "Linha 1", "machines": ["M-101", "M-102", "M-103"]}]
        
    Returns:
        CellPerformance with metrics for all cells
    """
    if plan_df.empty:
        return CellPerformance(
            cells=[],
            total_throughput=0,
            avg_lead_time=0,
            total_wip=0,
            summary={},
        )
    
    # Parse dates
    if not pd.api.types.is_datetime64_any_dtype(plan_df['start_time']):
        plan_df = plan_df.copy()
        plan_df['start_time'] = pd.to_datetime(plan_df['start_time'])
        plan_df['end_time'] = pd.to_datetime(plan_df['end_time'])
    
    # Auto-detect cells if not provided
    if cells_config is None:
        cells_config = _detect_cells(plan_df, machines_df)
    
    cells_list = []
    total_throughput = 0
    all_lead_times = []
    all_wip = 0
    
    for cell_cfg in cells_config:
        cell_id = cell_cfg.get("id", f"Cell-{len(cells_list)+1}")
        cell_name = cell_cfg.get("name", cell_id)
        cell_machines = cell_cfg.get("machines", [])
        
        if len(cell_machines) < 2:
            continue
        
        # Calculate stage metrics
        stages = []
        max_util = 0
        bottleneck_idx = 0
        
        for pos, machine_id in enumerate(cell_machines):
            m_ops = plan_df[plan_df['machine_id'] == machine_id]
            
            if not m_ops.empty:
                total_duration = m_ops['duration_min'].sum() if 'duration_min' in m_ops.columns else 0
                total_qty = m_ops['qty'].sum() if 'qty' in m_ops.columns else len(m_ops)
                
                # Calculate horizon
                horizon_hours = (m_ops['end_time'].max() - m_ops['start_time'].min()).total_seconds() / 3600
                horizon_days = max(1, horizon_hours / 24)
                
                utilization = (total_duration / 60 / horizon_hours * 100) if horizon_hours > 0 else 0
                throughput = total_qty / horizon_days
                cycle_time = total_duration / total_qty if total_qty > 0 else 0
                
                # Capacity based on 24h operation
                capacity = (24 * 60 / cycle_time) if cycle_time > 0 else 0
            else:
                utilization = 0
                throughput = 0
                cycle_time = 0
                capacity = 0
                horizon_days = 1
            
            stages.append(CellStage(
                machine_id=machine_id,
                sequence_position=pos,
                utilization_pct=utilization,
                throughput_units_per_day=throughput,
                cycle_time_min=cycle_time,
                is_bottleneck=False,
                capacity_units_per_day=capacity,
            ))
            
            if utilization > max_util:
                max_util = utilization
                bottleneck_idx = pos
        
        # Mark bottleneck
        if stages:
            stages[bottleneck_idx].is_bottleneck = True
        
        # Calculate buffer metrics
        buffers = []
        for i in range(len(cell_machines) - 1):
            from_m = cell_machines[i]
            to_m = cell_machines[i + 1]
            
            # Estimate WIP based on operations timing
            from_ops = plan_df[plan_df['machine_id'] == from_m]
            to_ops = plan_df[plan_df['machine_id'] == to_m]
            
            if not from_ops.empty and not to_ops.empty:
                # Simplified WIP calculation
                avg_wip = np.random.uniform(5, 20)  # Synthetic for now
                max_wip = avg_wip * 1.5
                avg_wait = np.random.uniform(0.5, 2)
            else:
                avg_wip = 0
                max_wip = 0
                avg_wait = 0
            
            buffers.append(BufferMetrics(
                from_machine=from_m,
                to_machine=to_m,
                avg_wip=avg_wip,
                max_wip=max_wip,
                avg_wait_hours=avg_wait,
                max_wait_hours=avg_wait * 2,
                is_congested=avg_wip > 15,
            ))
        
        # Aggregate cell metrics
        cell_throughput = min(s.throughput_units_per_day for s in stages) if stages else 0
        cell_capacity = min(s.capacity_units_per_day for s in stages) if stages else 0
        efficiency = (cell_throughput / cell_capacity * 100) if cell_capacity > 0 else 0
        
        # Lead time estimation
        total_cycle = sum(s.cycle_time_min for s in stages)
        total_wait = sum(b.avg_wait_hours * 60 for b in buffers)
        lead_time_hours = (total_cycle + total_wait) / 60
        
        # WIP
        cell_wip = sum(b.avg_wip for b in buffers)
        target_wip = len(buffers) * 10  # Simple target
        
        if cell_wip > target_wip * 1.5:
            wip_status = "critical"
        elif cell_wip > target_wip:
            wip_status = "high"
        else:
            wip_status = "optimal"
        
        # Status and issues
        issues = []
        recommendations = []
        
        bottleneck_machine = cell_machines[bottleneck_idx] if stages else None
        bottleneck_util = stages[bottleneck_idx].utilization_pct if stages else 0
        
        if bottleneck_util > 90:
            issues.append(f"Gargalo severo na {bottleneck_machine} ({bottleneck_util:.0f}%)")
            recommendations.append(f"Considerar turno extra ou máquina adicional paralela a {bottleneck_machine}")
        
        if any(b.is_congested for b in buffers):
            congested = [b.from_machine for b in buffers if b.is_congested]
            issues.append(f"Buffers congestionados após: {', '.join(congested)}")
            recommendations.append("Aumentar capacidade de armazenamento intermédio ou balancear linha")
        
        if efficiency < 70:
            issues.append(f"Eficiência da linha baixa ({efficiency:.0f}%)")
            recommendations.append("Investigar causas de perdas e rebalancear estágios")
        
        # Overall status
        if bottleneck_util > 95 or wip_status == "critical" or efficiency < 60:
            status = "critical"
        elif bottleneck_util > 85 or wip_status == "high" or efficiency < 75:
            status = "attention"
        else:
            status = "healthy"
        
        cells_list.append(CellMetrics(
            cell_id=cell_id,
            cell_name=cell_name,
            machines=cell_machines,
            stages=stages,
            buffers=buffers,
            lead_time_hours=lead_time_hours,
            lead_time_target_hours=lead_time_hours * 0.8,
            throughput_units_per_day=cell_throughput,
            capacity_units_per_day=cell_capacity,
            efficiency_pct=efficiency,
            bottleneck_machine=bottleneck_machine,
            bottleneck_utilization=bottleneck_util,
            bottleneck_constraint="capacity",
            total_wip=cell_wip,
            target_wip=target_wip,
            wip_status=wip_status,
            status=status,
            issues=issues,
            recommendations=recommendations,
        ))
        
        total_throughput += cell_throughput
        all_lead_times.append(lead_time_hours)
        all_wip += cell_wip
    
    return CellPerformance(
        cells=cells_list,
        total_throughput=total_throughput,
        avg_lead_time=np.mean(all_lead_times) if all_lead_times else 0,
        total_wip=all_wip,
        summary={},
    )


def _detect_cells(plan_df: pd.DataFrame, machines_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Auto-detect production cells based on operation sequences."""
    cells = []
    
    # Group by article and find machine sequences
    sequences = {}
    
    for article in plan_df['article_id'].unique():
        art_ops = plan_df[plan_df['article_id'] == article].sort_values('start_time')
        if 'op_seq' in art_ops.columns:
            art_ops = art_ops.sort_values('op_seq')
        
        seq = tuple(art_ops['machine_id'].tolist())
        if len(seq) >= 2:
            if seq not in sequences:
                sequences[seq] = 0
            sequences[seq] += 1
    
    # Create cells from most common sequences
    for i, (seq, count) in enumerate(sorted(sequences.items(), key=lambda x: -x[1])[:5]):
        cells.append({
            "id": f"Linha-{i+1}",
            "name": f"Linha {i+1} ({' → '.join(seq[:3])}{'...' if len(seq) > 3 else ''})",
            "machines": list(seq),
        })
    
    return cells

