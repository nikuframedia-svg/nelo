"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    ANNUAL CAPACITY vs DEMAND PROJECTION
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Generates S&OP style capacity vs demand projections:
- 12-month forecast
- Capacity utilization by month
- Gap analysis (over/under capacity)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from calendar import monthrange


@dataclass
class MonthlyProjection:
    """Projection for a single month."""
    month: int  # 1-12
    year: int
    month_name: str
    
    # Demand
    demand_units: float
    demand_hours: float
    demand_growth_pct: float  # vs previous month
    
    # Capacity
    capacity_units: float
    capacity_hours: float
    
    # Gap analysis
    gap_units: float  # negative = overcapacity
    gap_hours: float
    gap_pct: float  # vs capacity
    
    # Status
    status: str  # "balanced", "overcapacity", "undercapacity"
    utilization_pct: float
    
    # Risk level
    risk_level: str  # "low", "medium", "high"
    
    # Actions needed
    actions: List[str]


@dataclass
class CapacityProjection:
    """Complete 12-month projection."""
    projections: List[MonthlyProjection]
    
    # Aggregates
    total_demand_units: float
    total_capacity_units: float
    avg_utilization_pct: float
    
    # Gap summary
    months_overcapacity: int
    months_undercapacity: int
    max_gap_month: Optional[str]
    max_gap_pct: float
    
    # Growth
    demand_growth_annual: float
    
    # Summary
    summary: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "projections": [
                {
                    "month": int(p.month),
                    "year": int(p.year),
                    "month_name": p.month_name,
                    "demand": {
                        "units": round(float(p.demand_units), 0),
                        "hours": round(float(p.demand_hours), 0),
                        "growth_pct": round(float(p.demand_growth_pct), 1),
                    },
                    "capacity": {
                        "units": round(float(p.capacity_units), 0),
                        "hours": round(float(p.capacity_hours), 0),
                    },
                    "gap": {
                        "units": round(float(p.gap_units), 0),
                        "hours": round(float(p.gap_hours), 0),
                        "pct": round(float(p.gap_pct), 1),
                    },
                    "status": p.status,
                    "utilization_pct": round(float(p.utilization_pct), 1),
                    "risk_level": p.risk_level,
                    "actions": p.actions,
                }
                for p in self.projections
            ],
            "totals": {
                "demand_units": round(float(self.total_demand_units), 0),
                "capacity_units": round(float(self.total_capacity_units), 0),
                "avg_utilization_pct": round(float(self.avg_utilization_pct), 1),
            },
            "gaps": {
                "months_overcapacity": int(self.months_overcapacity),
                "months_undercapacity": int(self.months_undercapacity),
                "max_gap_month": self.max_gap_month,
                "max_gap_pct": round(float(self.max_gap_pct), 1),
            },
            "growth": {
                "annual_demand_growth_pct": round(float(self.demand_growth_annual), 1),
            },
            "summary": self.summary,
            "recommendations": self.recommendations,
        }


def generate_capacity_projection(
    plan_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    orders_df: Optional[pd.DataFrame] = None,
    forecast_months: int = 12,
    demand_growth_monthly: float = 0.02,  # 2% monthly growth default
    start_date: Optional[datetime] = None,
) -> CapacityProjection:
    """
    Generate 12-month capacity vs demand projection.
    
    Args:
        plan_df: Production plan DataFrame
        machines_df: Machines DataFrame
        orders_df: Historical orders (optional, for trend analysis)
        forecast_months: Number of months to project
        demand_growth_monthly: Monthly demand growth rate
        start_date: Projection start date (default: next month)
        
    Returns:
        CapacityProjection with monthly forecasts
    """
    # Determine base metrics from current plan
    if not plan_df.empty:
        if not pd.api.types.is_datetime64_any_dtype(plan_df['start_time']):
            plan_df = plan_df.copy()
            plan_df['start_time'] = pd.to_datetime(plan_df['start_time'])
            plan_df['end_time'] = pd.to_datetime(plan_df['end_time'])
        
        # Base demand from current plan
        total_qty = plan_df['qty'].sum() if 'qty' in plan_df.columns else len(plan_df)
        total_hours = plan_df['duration_min'].sum() / 60 if 'duration_min' in plan_df.columns else 0
        
        # Calculate monthly average from plan horizon
        horizon_days = (plan_df['end_time'].max() - plan_df['start_time'].min()).days
        horizon_months = max(1, horizon_days / 30)
        
        base_demand_units = total_qty / horizon_months
        base_demand_hours = total_hours / horizon_months
    else:
        base_demand_units = 10000
        base_demand_hours = 500
    
    # Calculate capacity from machines
    num_machines = len(machines_df) if not machines_df.empty else 10
    hours_per_machine_month = 22 * 8  # 22 working days, 8 hours
    
    # Adjust for efficiency (assume 85% effective)
    effective_hours = num_machines * hours_per_machine_month * 0.85
    
    # Units capacity (based on avg cycle time from plan)
    if not plan_df.empty and 'duration_min' in plan_df.columns:
        avg_cycle_min = plan_df['duration_min'].mean() if plan_df['duration_min'].mean() > 0 else 10
        capacity_units = effective_hours * 60 / avg_cycle_min
    else:
        capacity_units = base_demand_units * 1.2  # Assume 20% headroom
    
    # Start date
    if start_date is None:
        start_date = datetime.now().replace(day=1) + timedelta(days=32)
        start_date = start_date.replace(day=1)
    
    # Month names in Portuguese
    month_names = [
        'Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
        'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'
    ]
    
    # Generate projections
    projections = []
    current_demand = base_demand_units
    prev_demand = base_demand_units
    
    total_demand = 0
    total_capacity = 0
    over_months = 0
    under_months = 0
    max_gap = 0
    max_gap_month = None
    
    for i in range(forecast_months):
        # Calculate month
        proj_date = start_date + timedelta(days=30 * i)
        month = proj_date.month
        year = proj_date.year
        
        # Demand with growth and seasonality
        growth = (1 + demand_growth_monthly) ** i
        
        # Seasonal adjustment (higher in Q4, lower in Q1)
        seasonal = 1.0
        if month in [10, 11, 12]:
            seasonal = 1.15
        elif month in [1, 2]:
            seasonal = 0.85
        elif month in [7, 8]:
            seasonal = 0.90
        
        demand_units = base_demand_units * growth * seasonal
        demand_hours = base_demand_hours * growth * seasonal
        
        # Capacity (constant unless expansion)
        cap_units = capacity_units
        cap_hours = effective_hours
        
        # Add some variation
        cap_units *= (1 + np.random.uniform(-0.05, 0.05))
        
        # Gap calculation
        gap_units = demand_units - cap_units
        gap_hours = demand_hours - cap_hours
        gap_pct = (gap_units / cap_units * 100) if cap_units > 0 else 0
        
        # Status
        if gap_pct > 10:
            status = "undercapacity"
            under_months += 1
        elif gap_pct < -15:
            status = "overcapacity"
            over_months += 1
        else:
            status = "balanced"
        
        # Utilization
        utilization = (demand_units / cap_units * 100) if cap_units > 0 else 0
        
        # Risk level
        if utilization > 100 or utilization < 60:
            risk_level = "high"
        elif utilization > 90 or utilization < 70:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Actions
        actions = []
        if gap_pct > 20:
            actions.append("Avaliar turno extra ou subcontratação")
            actions.append("Considerar investimento em nova máquina")
        elif gap_pct > 10:
            actions.append("Planear horas extra")
            actions.append("Rever prioridades de produção")
        elif gap_pct < -20:
            actions.append("Oportunidade para manutenção preventiva")
            actions.append("Considerar antecipação de produção")
        elif gap_pct < -10:
            actions.append("Avaliar redução de turnos")
        
        # Track max gap
        if abs(gap_pct) > max_gap:
            max_gap = abs(gap_pct)
            max_gap_month = month_names[month - 1]
        
        projections.append(MonthlyProjection(
            month=month,
            year=year,
            month_name=month_names[month - 1],
            demand_units=demand_units,
            demand_hours=demand_hours,
            demand_growth_pct=((demand_units - prev_demand) / prev_demand * 100) if prev_demand > 0 else 0,
            capacity_units=cap_units,
            capacity_hours=cap_hours,
            gap_units=gap_units,
            gap_hours=gap_hours,
            gap_pct=gap_pct,
            status=status,
            utilization_pct=utilization,
            risk_level=risk_level,
            actions=actions,
        ))
        
        total_demand += demand_units
        total_capacity += cap_units
        prev_demand = demand_units
    
    # Generate recommendations
    recommendations = []
    
    if under_months > forecast_months * 0.3:
        recommendations.append(
            f"{under_months} meses com subcapacidade prevista. Planear expansão de capacidade até Q2."
        )
    
    if over_months > forecast_months * 0.3:
        recommendations.append(
            f"{over_months} meses com excesso de capacidade. Considerar diversificação ou redução de turnos."
        )
    
    if max_gap > 30:
        recommendations.append(
            f"Gap crítico de {max_gap:.0f}% previsto em {max_gap_month}. Requer ação urgente."
        )
    
    annual_growth = (projections[-1].demand_units / projections[0].demand_units - 1) * 100 if projections else 0
    
    if annual_growth > 20:
        recommendations.append(
            f"Crescimento anual de {annual_growth:.0f}% requer plano de investimento em capacidade."
        )
    
    # Summary
    avg_util = np.mean([p.utilization_pct for p in projections]) if projections else 0
    
    summary = {
        "projection_months": forecast_months,
        "demand_trend": "crescente" if annual_growth > 5 else "estável" if annual_growth > -5 else "decrescente",
        "capacity_status": "adequada" if 70 <= avg_util <= 90 else "insuficiente" if avg_util > 90 else "excesso",
        "risk_months": sum(1 for p in projections if p.risk_level == "high"),
    }
    
    return CapacityProjection(
        projections=projections,
        total_demand_units=total_demand,
        total_capacity_units=total_capacity,
        avg_utilization_pct=avg_util,
        months_overcapacity=over_months,
        months_undercapacity=under_months,
        max_gap_month=max_gap_month,
        max_gap_pct=max_gap,
        demand_growth_annual=annual_growth,
        summary=summary,
        recommendations=recommendations,
    )

