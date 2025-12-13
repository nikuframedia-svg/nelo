"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    CAPACITY PLANNER — Long-Term Strategic Planning
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Long-term capacity vs demand planning (12+ months).

Features:
- Aggregate capacity planning
- Demand growth scenarios
- Investment decisions (new machines)
- Bottleneck projection
- What-if scenario comparison

Mathematical Model:
─────────────────────────────────────────────────────────────────────────────────────────────────────

Parameters:
    D_t     : Demand in period t
    C_m,t   : Capacity of machine m in period t
    g       : Growth rate per period
    I_t     : Investment decision in period t

Balance Equation:
    Production_t ≤ min(D_t, Σ_m C_m,t)

Bottleneck Detection:
    If D_t > Σ_m C_m,t → capacity shortage
    Identify m* = argmax_m (Utilization_m,t)

═══════════════════════════════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AggregationLevel(str, Enum):
    """Aggregation level for capacity planning."""
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class CapacityChange:
    """A planned capacity change (add/remove machine, shift change)."""
    change_id: str
    effective_date: datetime
    change_type: str  # "add_machine", "remove_machine", "add_shift", "remove_shift"
    machine_id: str
    capacity_delta_hours_per_day: float = 0.0
    description: str = ""


@dataclass
class DemandForecast:
    """Demand forecast for planning."""
    period: str  # e.g., "2025-01" for month
    product_group: str
    demand_units: float
    demand_hours: float  # Converted to machine hours
    confidence: float = 0.8


@dataclass
class CapacityScenario:
    """A capacity planning scenario."""
    scenario_id: str
    scenario_name: str
    
    # Demand assumptions
    base_demand: float  # Annual demand
    growth_rate_quarterly: float = 0.10  # 10% per quarter
    
    # Capacity changes
    capacity_changes: List[CapacityChange] = field(default_factory=list)
    
    # Results (filled after simulation)
    periods: List[str] = field(default_factory=list)
    demand_by_period: Dict[str, float] = field(default_factory=dict)
    capacity_by_period: Dict[str, float] = field(default_factory=dict)
    gap_by_period: Dict[str, float] = field(default_factory=dict)  # Positive = shortage
    utilization_by_period: Dict[str, float] = field(default_factory=dict)
    bottleneck_by_period: Dict[str, str] = field(default_factory=dict)


@dataclass
class LongTermPlan:
    """Result of long-term capacity planning."""
    horizon_start: datetime
    horizon_end: datetime
    horizon_months: int
    aggregation: AggregationLevel
    
    # Scenarios
    scenarios: List[CapacityScenario] = field(default_factory=list)
    
    # Comparison metrics
    comparison: Optional[Dict[str, Any]] = None
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    investment_decisions: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon_start": self.horizon_start.isoformat(),
            "horizon_end": self.horizon_end.isoformat(),
            "horizon_months": self.horizon_months,
            "aggregation": self.aggregation.value,
            "scenarios": [
                {
                    "id": s.scenario_id,
                    "name": s.scenario_name,
                    "periods": s.periods,
                    "demand": s.demand_by_period,
                    "capacity": s.capacity_by_period,
                    "gap": s.gap_by_period,
                    "utilization": s.utilization_by_period,
                    "bottleneck": s.bottleneck_by_period,
                }
                for s in self.scenarios
            ],
            "comparison": self.comparison,
            "recommendations": self.recommendations,
            "investment_decisions": self.investment_decisions,
        }


class CapacityPlanner:
    """
    Long-term capacity planner for strategic decisions.
    """
    
    def __init__(
        self,
        machines_df: pd.DataFrame,
        base_capacity_hours_per_day: float = 16.0,  # 2 shifts
    ):
        """
        Initialize capacity planner.
        
        Args:
            machines_df: DataFrame with machine info
            base_capacity_hours_per_day: Default hours per machine per day
        """
        self.machines = machines_df['machine_id'].tolist()
        self.base_capacity = base_capacity_hours_per_day
        self.machine_capacity = {
            m: base_capacity_hours_per_day for m in self.machines
        }
    
    def plan(
        self,
        scenarios: List[CapacityScenario],
        horizon_months: int = 12,
        start_date: Optional[datetime] = None,
        aggregation: AggregationLevel = AggregationLevel.MONTH,
    ) -> LongTermPlan:
        """
        Generate long-term capacity plan.
        
        Args:
            scenarios: List of scenarios to simulate
            horizon_months: Planning horizon in months
            start_date: Start of planning horizon
            aggregation: Time bucket aggregation
            
        Returns:
            LongTermPlan with simulated scenarios
        """
        if start_date is None:
            start_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        end_date = start_date + relativedelta(months=horizon_months)
        
        # Generate periods
        periods = self._generate_periods(start_date, end_date, aggregation)
        
        # Simulate each scenario
        for scenario in scenarios:
            self._simulate_scenario(scenario, periods, start_date, aggregation)
        
        # Generate comparison if multiple scenarios
        comparison = None
        if len(scenarios) >= 2:
            comparison = self._compare_scenarios(scenarios)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(scenarios)
        investment_decisions = self._suggest_investments(scenarios, periods)
        
        return LongTermPlan(
            horizon_start=start_date,
            horizon_end=end_date,
            horizon_months=horizon_months,
            aggregation=aggregation,
            scenarios=scenarios,
            comparison=comparison,
            recommendations=recommendations,
            investment_decisions=investment_decisions,
        )
    
    def _generate_periods(
        self,
        start: datetime,
        end: datetime,
        aggregation: AggregationLevel,
    ) -> List[str]:
        """Generate list of period labels."""
        periods = []
        current = start
        
        while current < end:
            if aggregation == AggregationLevel.DAY:
                periods.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)
            elif aggregation == AggregationLevel.WEEK:
                periods.append(f"{current.year}-W{current.isocalendar()[1]:02d}")
                current += timedelta(weeks=1)
            else:  # MONTH
                periods.append(current.strftime("%Y-%m"))
                current += relativedelta(months=1)
        
        return periods
    
    def _simulate_scenario(
        self,
        scenario: CapacityScenario,
        periods: List[str],
        start_date: datetime,
        aggregation: AggregationLevel,
    ):
        """Simulate a capacity scenario."""
        scenario.periods = periods
        
        # Days per period
        if aggregation == AggregationLevel.DAY:
            days_per_period = 1
        elif aggregation == AggregationLevel.WEEK:
            days_per_period = 5  # Working days
        else:
            days_per_period = 22  # Working days per month
        
        # Track capacity changes
        capacity_changes_map = {}
        for change in scenario.capacity_changes:
            period_key = self._date_to_period(change.effective_date, aggregation)
            if period_key not in capacity_changes_map:
                capacity_changes_map[period_key] = []
            capacity_changes_map[period_key].append(change)
        
        # Current capacity (hours per period)
        current_capacity = len(self.machines) * self.base_capacity * days_per_period
        
        # Simulate each period
        quarterly_demand = scenario.base_demand / 4  # Base quarterly demand
        period_in_quarter = 0
        current_quarter = 0
        
        for i, period in enumerate(periods):
            # Apply capacity changes
            if period in capacity_changes_map:
                for change in capacity_changes_map[period]:
                    if change.change_type == "add_machine":
                        current_capacity += change.capacity_delta_hours_per_day * days_per_period
                    elif change.change_type == "remove_machine":
                        current_capacity -= change.capacity_delta_hours_per_day * days_per_period
                    elif change.change_type == "add_shift":
                        current_capacity += change.capacity_delta_hours_per_day * days_per_period
                    elif change.change_type == "remove_shift":
                        current_capacity -= change.capacity_delta_hours_per_day * days_per_period
            
            # Calculate demand with growth
            # Growth is quarterly
            if aggregation == AggregationLevel.MONTH:
                month_in_year = (i % 12)
                quarter = month_in_year // 3
                growth_factor = (1 + scenario.growth_rate_quarterly) ** quarter
                demand = (quarterly_demand / 3) * growth_factor
            elif aggregation == AggregationLevel.WEEK:
                week_in_year = (i % 52)
                quarter = week_in_year // 13
                growth_factor = (1 + scenario.growth_rate_quarterly) ** quarter
                demand = (quarterly_demand / 13) * growth_factor
            else:
                day_in_year = (i % 365)
                quarter = day_in_year // 91
                growth_factor = (1 + scenario.growth_rate_quarterly) ** quarter
                demand = (quarterly_demand / 91) * growth_factor
            
            # Store results
            scenario.demand_by_period[period] = demand
            scenario.capacity_by_period[period] = current_capacity
            scenario.gap_by_period[period] = demand - current_capacity
            scenario.utilization_by_period[period] = (demand / current_capacity * 100) if current_capacity > 0 else 0
            
            # Identify bottleneck (simplified: highest utilized machine)
            if scenario.gap_by_period[period] > 0:
                scenario.bottleneck_by_period[period] = self.machines[0] if self.machines else "Unknown"
            else:
                scenario.bottleneck_by_period[period] = None
    
    def _date_to_period(self, dt: datetime, aggregation: AggregationLevel) -> str:
        """Convert datetime to period string."""
        if aggregation == AggregationLevel.DAY:
            return dt.strftime("%Y-%m-%d")
        elif aggregation == AggregationLevel.WEEK:
            return f"{dt.year}-W{dt.isocalendar()[1]:02d}"
        else:
            return dt.strftime("%Y-%m")
    
    def _compare_scenarios(
        self,
        scenarios: List[CapacityScenario],
    ) -> Dict[str, Any]:
        """Compare multiple scenarios."""
        if len(scenarios) < 2:
            return {}
        
        baseline = scenarios[0]
        comparison = scenarios[1]
        
        # Total gap
        baseline_total_gap = sum(max(0, g) for g in baseline.gap_by_period.values())
        comparison_total_gap = sum(max(0, g) for g in comparison.gap_by_period.values())
        
        # Average utilization
        baseline_avg_util = np.mean(list(baseline.utilization_by_period.values()))
        comparison_avg_util = np.mean(list(comparison.utilization_by_period.values()))
        
        # Periods with shortage
        baseline_shortage_periods = sum(1 for g in baseline.gap_by_period.values() if g > 0)
        comparison_shortage_periods = sum(1 for g in comparison.gap_by_period.values() if g > 0)
        
        return {
            "baseline": {
                "scenario_id": baseline.scenario_id,
                "total_gap_hours": float(baseline_total_gap),
                "avg_utilization_pct": float(baseline_avg_util),
                "periods_with_shortage": baseline_shortage_periods,
            },
            "comparison": {
                "scenario_id": comparison.scenario_id,
                "total_gap_hours": float(comparison_total_gap),
                "avg_utilization_pct": float(comparison_avg_util),
                "periods_with_shortage": comparison_shortage_periods,
            },
            "delta": {
                "gap_reduction_hours": float(baseline_total_gap - comparison_total_gap),
                "gap_reduction_pct": float((baseline_total_gap - comparison_total_gap) / baseline_total_gap * 100) if baseline_total_gap > 0 else 0,
                "utilization_change_pct": float(comparison_avg_util - baseline_avg_util),
                "shortage_periods_reduced": baseline_shortage_periods - comparison_shortage_periods,
            },
            "recommendation": "with_expansion" if comparison_total_gap < baseline_total_gap else "baseline",
        }
    
    def _generate_recommendations(
        self,
        scenarios: List[CapacityScenario],
    ) -> List[str]:
        """Generate strategic recommendations."""
        recommendations = []
        
        for scenario in scenarios:
            # Check for capacity shortages
            shortage_periods = [p for p, g in scenario.gap_by_period.items() if g > 0]
            
            if shortage_periods:
                first_shortage = shortage_periods[0]
                recommendations.append(
                    f"Cenário '{scenario.scenario_name}': Défice de capacidade a partir de {first_shortage}. "
                    f"Total de {len(shortage_periods)} períodos com capacidade insuficiente."
                )
            
            # Check for high utilization (>85%)
            high_util_periods = [p for p, u in scenario.utilization_by_period.items() if u > 85]
            if high_util_periods and len(high_util_periods) > len(scenario.periods) * 0.3:
                recommendations.append(
                    f"Cenário '{scenario.scenario_name}': Utilização >85% em {len(high_util_periods)} períodos. "
                    "Considere aumentar capacidade para manter flexibilidade."
                )
        
        return recommendations
    
    def _suggest_investments(
        self,
        scenarios: List[CapacityScenario],
        periods: List[str],
    ) -> List[Dict[str, Any]]:
        """Suggest investment decisions."""
        investments = []
        
        for scenario in scenarios:
            # Find first period with sustained shortage
            shortage_streak = 0
            shortage_start = None
            
            for period in periods:
                gap = scenario.gap_by_period.get(period, 0)
                
                if gap > 0:
                    if shortage_streak == 0:
                        shortage_start = period
                    shortage_streak += 1
                else:
                    shortage_streak = 0
                
                # Suggest investment if 3+ consecutive shortage periods
                if shortage_streak >= 3 and shortage_start:
                    avg_gap = np.mean([
                        scenario.gap_by_period.get(p, 0)
                        for p in periods[periods.index(shortage_start):periods.index(shortage_start)+3]
                    ])
                    
                    investments.append({
                        "scenario": scenario.scenario_name,
                        "recommended_date": shortage_start,
                        "type": "add_machine" if avg_gap > self.base_capacity * 22 else "add_shift",
                        "reason": f"Défice médio de {avg_gap:.0f}h por período a partir de {shortage_start}.",
                        "estimated_roi_months": 12,
                    })
                    break
        
        return investments


def create_baseline_vs_expansion_scenarios(
    base_demand: float,
    growth_rate: float = 0.10,
    expansion_date: datetime = None,
    expansion_capacity: float = 16.0,
) -> Tuple[CapacityScenario, CapacityScenario]:
    """
    Create standard baseline vs expansion scenarios.
    
    Args:
        base_demand: Annual demand in hours
        growth_rate: Quarterly growth rate
        expansion_date: Date of capacity expansion
        expansion_capacity: Capacity added (hours/day)
        
    Returns:
        (baseline_scenario, expansion_scenario)
    """
    if expansion_date is None:
        expansion_date = datetime.now() + relativedelta(months=6)
    
    baseline = CapacityScenario(
        scenario_id="baseline",
        scenario_name="Sem expansão",
        base_demand=base_demand,
        growth_rate_quarterly=growth_rate,
        capacity_changes=[],
    )
    
    expansion = CapacityScenario(
        scenario_id="with_expansion",
        scenario_name="Com nova máquina",
        base_demand=base_demand,
        growth_rate_quarterly=growth_rate,
        capacity_changes=[
            CapacityChange(
                change_id="expand_1",
                effective_date=expansion_date,
                change_type="add_machine",
                machine_id="M-NEW",
                capacity_delta_hours_per_day=expansion_capacity,
                description="Adição de nova máquina",
            )
        ],
    )
    
    return baseline, expansion



