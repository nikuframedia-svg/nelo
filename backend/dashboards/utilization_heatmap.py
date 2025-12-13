"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    HOURLY UTILIZATION HEATMAP GENERATOR
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Generates heatmap data showing machine utilization by hour and day of week.
Color scale: Blue (idle) → Yellow (50%) → Red (100% utilized)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


@dataclass
class HeatmapCell:
    """Single cell in the heatmap."""
    machine_id: str
    day_of_week: int  # 0=Monday, 6=Sunday
    hour: int  # 0-23
    utilization_pct: float
    processing_min: float
    color: str  # hex color
    is_idle: bool
    is_overloaded: bool


@dataclass
class IdleWindow:
    """Identified idle window."""
    machine_id: str
    day_of_week: int
    start_hour: int
    end_hour: int
    avg_utilization: float
    suggestion: str


@dataclass
class HeatmapData:
    """Complete heatmap data."""
    machines: List[str]
    days: List[str]  # ['Segunda', 'Terça', ...]
    hours: List[int]  # 0-23
    cells: List[HeatmapCell]
    idle_windows: List[IdleWindow]
    overloaded_periods: List[Dict[str, Any]]
    color_scale: Dict[str, str]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        # Build matrix format for frontend
        matrix = {}
        for cell in self.cells:
            if cell.machine_id not in matrix:
                matrix[cell.machine_id] = {}
            key = f"{cell.day_of_week}_{cell.hour}"
            matrix[cell.machine_id][key] = {
                "utilization": round(float(cell.utilization_pct), 1),
                "color": cell.color,
                "is_idle": bool(cell.is_idle),
                "is_overloaded": bool(cell.is_overloaded),
            }
        
        return {
            "machines": self.machines,
            "days": self.days,
            "hours": self.hours,
            "matrix": matrix,
            "idle_windows": [
                {
                    "machine_id": w.machine_id,
                    "day": self.days[w.day_of_week],
                    "start_hour": int(w.start_hour),
                    "end_hour": int(w.end_hour),
                    "avg_utilization": round(float(w.avg_utilization), 1),
                    "suggestion": w.suggestion,
                }
                for w in self.idle_windows
            ],
            "overloaded_periods": [
                {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
                 for k, v in p.items()}
                for p in self.overloaded_periods
            ],
            "color_scale": self.color_scale,
            "summary": {k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v)
                       for k, v in self.summary.items()},
        }


def _get_color(utilization_pct: float) -> str:
    """Convert utilization percentage to color."""
    # Blue → Yellow → Red gradient
    if utilization_pct <= 0:
        return "#1e40af"  # Deep blue (idle)
    elif utilization_pct <= 30:
        return "#3b82f6"  # Blue (low)
    elif utilization_pct <= 50:
        return "#22c55e"  # Green (moderate)
    elif utilization_pct <= 70:
        return "#eab308"  # Yellow (medium-high)
    elif utilization_pct <= 85:
        return "#f97316"  # Orange (high)
    elif utilization_pct <= 95:
        return "#ef4444"  # Red (very high)
    else:
        return "#7f1d1d"  # Dark red (overloaded)


def generate_utilization_heatmap(
    plan_df: pd.DataFrame,
    machines_df: Optional[pd.DataFrame] = None,
    week_start: Optional[datetime] = None,
) -> HeatmapData:
    """
    Generate hourly utilization heatmap for machines.
    
    Args:
        plan_df: Production plan DataFrame
        machines_df: Machines DataFrame (optional, for availability)
        week_start: Start of the week to analyze (optional)
        
    Returns:
        HeatmapData with matrix and analysis
    """
    if plan_df.empty:
        return HeatmapData(
            machines=[],
            days=['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo'],
            hours=list(range(24)),
            cells=[],
            idle_windows=[],
            overloaded_periods=[],
            color_scale={"0": "#1e40af", "50": "#eab308", "100": "#7f1d1d"},
            summary={},
        )
    
    # Parse dates
    if not pd.api.types.is_datetime64_any_dtype(plan_df['start_time']):
        plan_df = plan_df.copy()
        plan_df['start_time'] = pd.to_datetime(plan_df['start_time'])
        plan_df['end_time'] = pd.to_datetime(plan_df['end_time'])
    
    # Determine week
    if week_start is None:
        week_start = plan_df['start_time'].min()
        # Align to Monday
        week_start = week_start - timedelta(days=week_start.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    
    week_end = week_start + timedelta(days=7)
    
    machines = sorted(plan_df['machine_id'].unique())
    days = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    hours = list(range(24))
    
    # Initialize utilization matrix
    utilization = {m: np.zeros((7, 24)) for m in machines}
    
    # Calculate utilization for each hour slot
    for _, row in plan_df.iterrows():
        machine = row['machine_id']
        start = row['start_time']
        end = row['end_time']
        
        # Iterate through each hour the operation spans
        current = start
        while current < end:
            if week_start <= current < week_end:
                day_idx = current.weekday()
                hour_idx = current.hour
                
                # Calculate minutes in this hour slot
                slot_start = current.replace(minute=0, second=0, microsecond=0)
                slot_end = slot_start + timedelta(hours=1)
                
                actual_start = max(start, slot_start)
                actual_end = min(end, slot_end)
                
                minutes_used = (actual_end - actual_start).total_seconds() / 60
                utilization[machine][day_idx][hour_idx] += minutes_used / 60 * 100  # Convert to %
            
            current = current.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    
    # Build cells
    cells = []
    for machine in machines:
        for day_idx in range(7):
            for hour_idx in range(24):
                util_pct = min(100, utilization[machine][day_idx][hour_idx])
                
                cell = HeatmapCell(
                    machine_id=machine,
                    day_of_week=day_idx,
                    hour=hour_idx,
                    utilization_pct=util_pct,
                    processing_min=util_pct * 0.6,  # Convert back to minutes
                    color=_get_color(util_pct),
                    is_idle=util_pct < 20,
                    is_overloaded=util_pct > 90,
                )
                cells.append(cell)
    
    # Identify idle windows
    idle_windows = []
    for machine in machines:
        for day_idx in range(7):
            # Look for consecutive idle hours
            idle_start = None
            for hour_idx in range(24):
                util = utilization[machine][day_idx][hour_idx]
                
                if util < 30:  # Idle threshold
                    if idle_start is None:
                        idle_start = hour_idx
                else:
                    if idle_start is not None and (hour_idx - idle_start) >= 2:
                        avg_util = np.mean(utilization[machine][day_idx][idle_start:hour_idx])
                        idle_windows.append(IdleWindow(
                            machine_id=machine,
                            day_of_week=day_idx,
                            start_hour=idle_start,
                            end_hour=hour_idx,
                            avg_utilization=avg_util,
                            suggestion=f"Janela de {hour_idx-idle_start}h disponível para realocar produção de máquinas sobrecarregadas.",
                        ))
                    idle_start = None
            
            # Check end of day
            if idle_start is not None and (24 - idle_start) >= 2:
                avg_util = np.mean(utilization[machine][day_idx][idle_start:24])
                idle_windows.append(IdleWindow(
                    machine_id=machine,
                    day_of_week=day_idx,
                    start_hour=idle_start,
                    end_hour=24,
                    avg_utilization=avg_util,
                    suggestion=f"Janela de {24-idle_start}h disponível no final do dia.",
                ))
    
    # Identify overloaded periods
    overloaded_periods = []
    for machine in machines:
        for day_idx in range(7):
            for hour_idx in range(24):
                if utilization[machine][day_idx][hour_idx] > 90:
                    overloaded_periods.append({
                        "machine_id": machine,
                        "day": days[day_idx],
                        "hour": hour_idx,
                        "utilization": round(utilization[machine][day_idx][hour_idx], 1),
                    })
    
    # Summary
    all_utils = []
    for m in machines:
        all_utils.extend(utilization[m].flatten())
    
    summary = {
        "avg_utilization": round(np.mean(all_utils), 1),
        "max_utilization": round(np.max(all_utils), 1),
        "idle_slots": sum(1 for u in all_utils if u < 20),
        "overloaded_slots": sum(1 for u in all_utils if u > 90),
        "total_idle_windows": len(idle_windows),
        "total_overloaded_periods": len(overloaded_periods),
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
    }
    
    return HeatmapData(
        machines=machines,
        days=days,
        hours=hours,
        cells=cells,
        idle_windows=idle_windows,
        overloaded_periods=overloaded_periods,
        color_scale={
            "0": "#1e40af",
            "30": "#3b82f6",
            "50": "#22c55e",
            "70": "#eab308",
            "85": "#f97316",
            "95": "#ef4444",
            "100": "#7f1d1d",
        },
        summary=summary,
    )

