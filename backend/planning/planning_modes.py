"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PLANNING MODES — Configuration & Data Structures
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Defines the different planning modes and their configurations:
- CONVENTIONAL: Independent machine scheduling
- CHAINED: Multi-stage flow shop with synchronization
- SHORT_TERM: Detailed 2-week planning with shifts/maintenance
- LONG_TERM: Strategic 12-month capacity planning
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


class PlanningMode(str, Enum):
    """Available planning modes."""
    CONVENTIONAL = "conventional"      # Independent per-machine scheduling
    CHAINED = "chained"               # Multi-stage flow shop
    SHORT_TERM = "short_term"         # Detailed 2-week planning
    LONG_TERM = "long_term"           # Strategic 12-month planning
    HYBRID = "hybrid"                 # Combined approach


class ObjectivePriority(str, Enum):
    """Optimization objective priorities."""
    MINIMIZE_MAKESPAN = "min_makespan"
    MINIMIZE_TARDINESS = "min_tardiness"
    MINIMIZE_SETUP = "min_setup"
    MINIMIZE_WIP = "min_wip"
    MAXIMIZE_THROUGHPUT = "max_throughput"
    BALANCE_LOAD = "balance_load"


class SolverType(str, Enum):
    """Solver backend to use."""
    HEURISTIC = "heuristic"
    MILP = "milp"
    CPSAT = "cpsat"
    GENETIC = "genetic"


@dataclass
class PlanningConfig:
    """
    Base configuration for all planning modes.
    """
    mode: PlanningMode = PlanningMode.CONVENTIONAL
    solver: SolverType = SolverType.HEURISTIC
    
    # Time horizon
    horizon_start: Optional[datetime] = None
    horizon_end: Optional[datetime] = None
    
    # Objectives (in priority order)
    objectives: List[ObjectivePriority] = field(default_factory=lambda: [
        ObjectivePriority.MINIMIZE_TARDINESS,
        ObjectivePriority.MINIMIZE_MAKESPAN,
        ObjectivePriority.MINIMIZE_SETUP,
    ])
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        "tardiness": 1.0,
        "makespan": 0.5,
        "setup": 0.2,
    })
    
    # Solver parameters
    time_limit_sec: float = 60.0
    gap_tolerance: float = 0.05  # 5% optimality gap
    
    # Constraints
    respect_due_dates: bool = True
    respect_shifts: bool = True
    respect_maintenance: bool = True
    finite_capacity: bool = True
    
    # Product priorities (article_id -> weight)
    product_priorities: Dict[str, float] = field(default_factory=dict)


@dataclass
class ChainedPlanningConfig(PlanningConfig):
    """
    Configuration for chained (flow shop) planning.
    
    Mathematical Model:
    - Minimize: C_max (makespan) + Σ w_j * T_j (weighted tardiness)
    - Subject to:
        - Precedence: S_{j,k+1} ≥ C_{j,k} + buffer_{k,k+1}
        - Machine capacity: No overlap on same machine
        - Flow synchronization across chain
    """
    mode: PlanningMode = PlanningMode.CHAINED
    
    # Chain definition: list of machine sequences
    # e.g., [["M-101", "M-102", "M-103"], ["M-201", "M-202"]]
    chains: List[List[str]] = field(default_factory=list)
    
    # Buffer times between chain stages (minutes)
    # Key: "from_machine->to_machine", Value: buffer_minutes
    buffers: Dict[str, float] = field(default_factory=dict)
    default_buffer_min: float = 30.0
    
    # Flow synchronization
    synchronize_flow: bool = True
    balance_chain_load: bool = True
    
    # Lot splitting
    allow_lot_splitting: bool = False
    min_lot_size: int = 1


@dataclass
class ConventionalPlanningConfig(PlanningConfig):
    """
    Configuration for conventional independent scheduling.
    
    Each machine optimizes its own sequence locally.
    """
    mode: PlanningMode = PlanningMode.CONVENTIONAL
    
    # Per-machine optimization
    optimize_per_machine: bool = True
    
    # Dispatching rule for ties
    dispatching_rule: str = "EDD"  # EDD, SPT, FIFO, CR
    
    # Setup grouping
    group_by_setup_family: bool = True


@dataclass
class ShortTermPlanningConfig(PlanningConfig):
    """
    Configuration for short-term detailed planning (typically 2 weeks).
    
    Features:
    - Daily/shift-level granularity
    - Maintenance windows
    - Operator availability
    - Detailed setup sequences
    """
    mode: PlanningMode = PlanningMode.SHORT_TERM
    solver: SolverType = SolverType.CPSAT  # More precise solver
    
    # Horizon (default 2 weeks)
    horizon_days: int = 14
    
    # Granularity
    time_bucket_minutes: int = 15  # 15-minute slots
    
    # Shift constraints
    shift_calendar: Optional[Dict[str, Any]] = None  # Machine -> shift schedule
    
    # Maintenance windows
    maintenance_windows: List[Dict[str, Any]] = field(default_factory=list)
    
    # Operator constraints
    require_operators: bool = True
    operator_skills: Dict[str, List[str]] = field(default_factory=dict)  # Operator -> machines
    
    # Setup optimization
    minimize_setup_changes: bool = True
    max_setup_per_day: Optional[int] = None


@dataclass
class LongTermPlanningConfig(PlanningConfig):
    """
    Configuration for long-term strategic planning (12 months).
    
    Features:
    - Aggregate capacity planning
    - Demand growth scenarios
    - Investment decisions (new machines)
    - Bottleneck analysis
    """
    mode: PlanningMode = PlanningMode.LONG_TERM
    solver: SolverType = SolverType.HEURISTIC  # Fast approximation
    
    # Horizon
    horizon_months: int = 12
    
    # Demand growth
    demand_growth_quarterly: float = 0.10  # 10% per quarter
    
    # Capacity changes
    capacity_changes: List[Dict[str, Any]] = field(default_factory=list)
    # e.g., [{"date": "2025-07-01", "machine_id": "M-NEW", "action": "add"}]
    
    # Scenarios to simulate
    scenarios: List[str] = field(default_factory=lambda: ["baseline", "with_expansion"])
    
    # Aggregation level
    aggregate_by: str = "week"  # day, week, month


@dataclass
class PlanningResult:
    """
    Result of a planning execution.
    """
    mode: PlanningMode
    config: PlanningConfig
    
    # Plan data
    plan_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # KPIs
    makespan_hours: float = 0.0
    total_tardiness_hours: float = 0.0
    total_setup_hours: float = 0.0
    throughput_units: float = 0.0
    utilization_pct: float = 0.0
    
    # Per-machine metrics
    machine_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Bottleneck
    bottleneck_machine: Optional[str] = None
    bottleneck_utilization: float = 0.0
    
    # Solver info
    solver_status: str = "unknown"
    solve_time_sec: float = 0.0
    gap_achieved: float = 0.0
    
    # Warnings/issues
    warnings: List[str] = field(default_factory=list)
    infeasibilities: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "mode": self.mode.value,
            "makespan_hours": float(self.makespan_hours),
            "total_tardiness_hours": float(self.total_tardiness_hours),
            "total_setup_hours": float(self.total_setup_hours),
            "throughput_units": float(self.throughput_units),
            "utilization_pct": float(self.utilization_pct),
            "bottleneck_machine": self.bottleneck_machine,
            "bottleneck_utilization": float(self.bottleneck_utilization),
            "solver_status": self.solver_status,
            "solve_time_sec": float(self.solve_time_sec),
            "gap_achieved": float(self.gap_achieved),
            "warnings": self.warnings,
            "infeasibilities": self.infeasibilities,
            "machine_metrics": {
                k: {mk: float(mv) for mk, mv in v.items()}
                for k, v in self.machine_metrics.items()
            },
            "plan_rows": len(self.plan_df),
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class PlanningComparison:
    """
    Comparison between two planning results (e.g., chained vs conventional).
    """
    baseline: PlanningResult
    scenario: PlanningResult
    
    # Deltas
    makespan_delta_hours: float = 0.0
    makespan_delta_pct: float = 0.0
    tardiness_delta_hours: float = 0.0
    tardiness_delta_pct: float = 0.0
    setup_delta_hours: float = 0.0
    setup_delta_pct: float = 0.0
    throughput_delta_units: float = 0.0
    throughput_delta_pct: float = 0.0
    
    # Recommendations
    recommended_mode: PlanningMode = PlanningMode.CONVENTIONAL
    recommendation_reason: str = ""
    
    def compute_deltas(self):
        """Compute delta metrics between baseline and scenario."""
        b = self.baseline
        s = self.scenario
        
        self.makespan_delta_hours = s.makespan_hours - b.makespan_hours
        self.makespan_delta_pct = (self.makespan_delta_hours / b.makespan_hours * 100) if b.makespan_hours > 0 else 0
        
        self.tardiness_delta_hours = s.total_tardiness_hours - b.total_tardiness_hours
        self.tardiness_delta_pct = (self.tardiness_delta_hours / b.total_tardiness_hours * 100) if b.total_tardiness_hours > 0 else 0
        
        self.setup_delta_hours = s.total_setup_hours - b.total_setup_hours
        self.setup_delta_pct = (self.setup_delta_hours / b.total_setup_hours * 100) if b.total_setup_hours > 0 else 0
        
        self.throughput_delta_units = s.throughput_units - b.throughput_units
        self.throughput_delta_pct = (self.throughput_delta_units / b.throughput_units * 100) if b.throughput_units > 0 else 0
        
        # Determine recommendation
        if s.makespan_hours < b.makespan_hours and s.total_tardiness_hours <= b.total_tardiness_hours:
            self.recommended_mode = s.mode
            self.recommendation_reason = f"Redução de {abs(self.makespan_delta_pct):.1f}% no makespan sem aumentar atrasos."
        elif s.total_tardiness_hours < b.total_tardiness_hours:
            self.recommended_mode = s.mode
            self.recommendation_reason = f"Redução de {abs(self.tardiness_delta_pct):.1f}% nos atrasos."
        else:
            self.recommended_mode = b.mode
            self.recommendation_reason = "Modo baseline apresenta melhores resultados globais."
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "scenario": self.scenario.to_dict(),
            "deltas": {
                "makespan_hours": float(self.makespan_delta_hours),
                "makespan_pct": float(self.makespan_delta_pct),
                "tardiness_hours": float(self.tardiness_delta_hours),
                "tardiness_pct": float(self.tardiness_delta_pct),
                "setup_hours": float(self.setup_delta_hours),
                "setup_pct": float(self.setup_delta_pct),
                "throughput_units": float(self.throughput_delta_units),
                "throughput_pct": float(self.throughput_delta_pct),
            },
            "recommendation": {
                "mode": self.recommended_mode.value,
                "reason": self.recommendation_reason,
            }
        }



