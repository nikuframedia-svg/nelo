"""
Inventory Optimization — Joint Stock + Capacity Optimization

R&D Module for WP3: Inventory-Production Coupling

Research Question (Q3):
    Can we jointly optimize coverage, risk, and OTD instead of treating
    inventory and production as separate modules?

Hypotheses:
    H3.1: Joint inventory+capacity optimization reduces stockout risk
          by ≥20% vs decoupled approach
    H3.2: Coupled model achieves same OTD with 15% less safety stock
    H3.3: Per-SKU risk forecasts improve replenishment timing accuracy

Technical Uncertainty:
    - Multi-objective formulation with conflicting goals
    - Demand uncertainty propagation through capacity constraints
    - Computational tractability for real-time decisions
    - Interaction between production delays and stock policies

Approach:
    1. Enhanced ABC/XYZ classification with capacity awareness
    2. ROP calculation considering production lead times
    3. Joint simulation of demand + production scenarios
    4. Risk-based safety stock optimization

Usage:
    from backend.research.inventory_optimization import InventoryOptimizer
    
    optimizer = InventoryOptimizer()
    policy = optimizer.optimize_sku(
        sku_id="SKU-001",
        demand_forecast=demand_df,
        production_capacity=capacity_df,
        risk_tolerance=0.95
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import random

import pandas as pd
import numpy as np


class OptimizationMode(Enum):
    """Inventory optimization modes."""
    DECOUPLED = "decoupled"           # Traditional: separate inventory & production
    CAPACITY_AWARE = "capacity_aware" # Consider production capacity in ROP
    JOINT = "joint"                   # Full joint optimization
    RISK_BASED = "risk_based"         # Risk-driven safety stock


class RiskLevel(Enum):
    """SKU risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SKUProfile:
    """Profile of a single SKU for optimization."""
    sku_id: str
    article_id: str
    
    # Classification
    abc_class: str  # A, B, C
    xyz_class: str  # X, Y, Z
    
    # Demand metrics
    avg_daily_demand: float
    demand_std: float
    demand_cv: float  # Coefficient of variation
    
    # Current state
    current_stock: float
    pending_orders: float
    in_production: float
    
    # Costs
    holding_cost_per_unit: float = 1.0
    stockout_cost_per_unit: float = 10.0
    
    # Production
    production_lead_time_days: float = 5.0
    min_lot_size: float = 100.0


@dataclass
class InventoryPolicy:
    """Inventory policy for a SKU."""
    sku_id: str
    rop: float  # Reorder Point
    safety_stock: float
    order_quantity: float
    
    # Policy type
    policy_type: str  # "s,Q", "s,S", "R,s,S"
    
    # Risk metrics
    expected_stockout_prob: float
    expected_service_level: float
    
    # Explanation
    rationale: str


@dataclass
class SimulationResult:
    """Result of an inventory simulation."""
    n_periods: int
    total_demand: float
    total_stockouts: float
    avg_inventory: float
    service_level: float
    total_cost: float
    stockout_events: int


class InventoryOptimizer:
    """
    Main inventory optimizer with production coupling.
    """
    
    def __init__(self, mode: OptimizationMode = OptimizationMode.DECOUPLED):
        self.mode = mode
        self._optimization_log: List[Dict[str, Any]] = []
    
    def compute_rop_decoupled(
        self,
        profile: SKUProfile,
        service_level: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Traditional ROP calculation (no capacity awareness).
        
        ROP = d * L + z * σ_d * √L
        
        where:
            d = average daily demand
            L = lead time (days)
            z = service level z-score
            σ_d = demand standard deviation
        """
        # Z-score lookup (avoid scipy dependency)
        z_scores = {
            0.80: 0.842, 0.85: 1.036, 0.90: 1.282,
            0.95: 1.645, 0.98: 2.054, 0.99: 2.326,
        }
        z = z_scores.get(round(service_level, 2), 1.645)
        try:
            from scipy import stats
            z = stats.norm.ppf(service_level)
        except ImportError:
            pass
        
        cycle_stock = profile.avg_daily_demand * profile.production_lead_time_days
        safety_stock = z * profile.demand_std * np.sqrt(profile.production_lead_time_days)
        
        rop = cycle_stock + safety_stock
        
        return rop, safety_stock
    
    def compute_rop_capacity_aware(
        self,
        profile: SKUProfile,
        service_level: float = 0.95,
        machine_utilization: float = 0.8,
        bottleneck_buffer_days: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Capacity-aware ROP: adjusts lead time based on machine load.
        
        TODO[R&D]: Test hypothesis H3.1.
        
        Lead time adjustment:
        - If machine utilization > 80%, add buffer
        - If bottleneck detected, add extra buffer
        """
        # Z-score lookup (avoid scipy dependency)
        z_scores = {
            0.80: 0.842, 0.85: 1.036, 0.90: 1.282,
            0.95: 1.645, 0.98: 2.054, 0.99: 2.326,
        }
        z = z_scores.get(round(service_level, 2), 1.645)
        try:
            from scipy import stats
            z = stats.norm.ppf(service_level)
        except ImportError:
            pass
        
        # Adjust lead time based on capacity
        effective_lead_time = profile.production_lead_time_days
        
        if machine_utilization > 0.9:
            effective_lead_time *= 1.3  # 30% longer lead time
        elif machine_utilization > 0.8:
            effective_lead_time *= 1.15  # 15% longer
        
        # Add bottleneck buffer
        effective_lead_time += bottleneck_buffer_days
        
        cycle_stock = profile.avg_daily_demand * effective_lead_time
        safety_stock = z * profile.demand_std * np.sqrt(effective_lead_time)
        
        rop = cycle_stock + safety_stock
        
        return rop, safety_stock
    
    def compute_rop_joint(
        self,
        profile: SKUProfile,
        service_level: float = 0.95,
        capacity_scenarios: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[float, float]:
        """
        Joint optimization considering multiple capacity scenarios.
        
        TODO[R&D]: Test hypothesis H3.2.
        
        Approach:
        1. Simulate multiple demand + capacity scenarios
        2. Find ROP that achieves target service level across scenarios
        3. Balance safety stock cost vs stockout cost
        """
        if capacity_scenarios is None:
            # Default scenarios: normal, high load, bottleneck
            capacity_scenarios = [
                {"utilization": 0.7, "prob": 0.6},
                {"utilization": 0.85, "prob": 0.3},
                {"utilization": 0.95, "prob": 0.1},
            ]
        
        # Weighted ROP across scenarios
        total_rop = 0.0
        total_safety = 0.0
        
        for scenario in capacity_scenarios:
            rop, safety = self.compute_rop_capacity_aware(
                profile,
                service_level,
                machine_utilization=scenario["utilization"],
            )
            total_rop += rop * scenario["prob"]
            total_safety += safety * scenario["prob"]
        
        return total_rop, total_safety
    
    def optimize_sku(
        self,
        profile: SKUProfile,
        service_level: float = 0.95,
        machine_utilization: float = 0.8,
        bottleneck_buffer_days: float = 0.0,
    ) -> InventoryPolicy:
        """
        Optimize inventory policy for a single SKU.
        """
        if self.mode == OptimizationMode.DECOUPLED:
            rop, safety = self.compute_rop_decoupled(profile, service_level)
            rationale = "ROP tradicional sem considerar capacidade produtiva."
        elif self.mode == OptimizationMode.CAPACITY_AWARE:
            rop, safety = self.compute_rop_capacity_aware(
                profile, service_level, machine_utilization, bottleneck_buffer_days
            )
            rationale = f"ROP ajustado para utilização de {machine_utilization*100:.0f}%."
        elif self.mode == OptimizationMode.JOINT:
            rop, safety = self.compute_rop_joint(profile, service_level)
            rationale = "ROP otimizado conjuntamente com cenários de capacidade."
        else:  # RISK_BASED
            rop, safety = self._compute_risk_based(profile, service_level)
            rationale = f"ROP baseado em risco (ABC={profile.abc_class}, XYZ={profile.xyz_class})."
        
        # Economic Order Quantity (EOQ) for order quantity
        order_qty = self._compute_eoq(profile)
        
        # Expected service level (simplified)
        expected_sl = service_level * (1 - machine_utilization * 0.1)
        
        policy = InventoryPolicy(
            sku_id=profile.sku_id,
            rop=round(rop, 0),
            safety_stock=round(safety, 0),
            order_quantity=round(order_qty, 0),
            policy_type="s,Q",  # Reorder point, fixed quantity
            expected_stockout_prob=1 - expected_sl,
            expected_service_level=expected_sl,
            rationale=rationale,
        )
        
        self._log_optimization(profile, policy)
        return policy
    
    def _compute_risk_based(
        self,
        profile: SKUProfile,
        base_service_level: float,
    ) -> Tuple[float, float]:
        """
        Risk-based ROP: higher service level for critical SKUs.
        """
        # Adjust service level based on ABC/XYZ
        sl_adjustments = {
            ("A", "X"): 0.99,
            ("A", "Y"): 0.97,
            ("A", "Z"): 0.95,
            ("B", "X"): 0.95,
            ("B", "Y"): 0.93,
            ("B", "Z"): 0.90,
            ("C", "X"): 0.90,
            ("C", "Y"): 0.85,
            ("C", "Z"): 0.80,
        }
        
        adjusted_sl = sl_adjustments.get(
            (profile.abc_class, profile.xyz_class),
            base_service_level
        )
        
        # Z-score lookup (avoid scipy dependency)
        z_scores = {
            0.80: 0.842, 0.85: 1.036, 0.90: 1.282,
            0.93: 1.476, 0.95: 1.645, 0.97: 1.881,
            0.98: 2.054, 0.99: 2.326,
        }
        z = z_scores.get(round(adjusted_sl, 2), 1.645)
        try:
            from scipy import stats
            z = stats.norm.ppf(adjusted_sl)
        except ImportError:
            pass
        
        cycle_stock = profile.avg_daily_demand * profile.production_lead_time_days
        safety_stock = z * profile.demand_std * np.sqrt(profile.production_lead_time_days)
        
        return cycle_stock + safety_stock, safety_stock
    
    def _compute_eoq(self, profile: SKUProfile) -> float:
        """Economic Order Quantity calculation."""
        # Simplified EOQ
        # EOQ = sqrt(2 * D * S / H)
        # D = annual demand, S = ordering cost, H = holding cost
        
        annual_demand = profile.avg_daily_demand * 365
        ordering_cost = 100.0  # Assumed fixed ordering cost
        holding_cost = profile.holding_cost_per_unit * 365
        
        if holding_cost <= 0:
            return profile.min_lot_size
        
        eoq = np.sqrt(2 * annual_demand * ordering_cost / holding_cost)
        
        # Respect minimum lot size
        return max(eoq, profile.min_lot_size)
    
    def _log_optimization(self, profile: SKUProfile, policy: InventoryPolicy) -> None:
        """Log optimization for analysis."""
        self._optimization_log.append({
            "sku_id": profile.sku_id,
            "mode": self.mode.value,
            "rop": policy.rop,
            "safety_stock": policy.safety_stock,
            "expected_service_level": policy.expected_service_level,
            "rationale": policy.rationale,
        })
    
    def simulate_policy(
        self,
        profile: SKUProfile,
        policy: InventoryPolicy,
        n_periods: int = 365,
        demand_scenarios: Optional[np.ndarray] = None,
    ) -> SimulationResult:
        """
        Simulate inventory policy over time.
        
        TODO[R&D]: Use for experiment E3.1.
        """
        if demand_scenarios is None:
            # Generate random demand
            demand_scenarios = np.random.normal(
                profile.avg_daily_demand,
                profile.demand_std,
                n_periods
            )
            demand_scenarios = np.maximum(0, demand_scenarios)
        
        # Simulation state
        inventory = profile.current_stock
        total_demand = 0.0
        total_stockouts = 0.0
        inventory_sum = 0.0
        stockout_events = 0
        pending_orders: List[Tuple[int, float]] = []  # (arrival_day, quantity)
        
        for day in range(n_periods):
            # Receive pending orders
            arriving = [q for (d, q) in pending_orders if d == day]
            inventory += sum(arriving)
            pending_orders = [(d, q) for (d, q) in pending_orders if d > day]
            
            # Fulfill demand
            demand = demand_scenarios[day]
            total_demand += demand
            
            if inventory >= demand:
                inventory -= demand
            else:
                stockout = demand - inventory
                total_stockouts += stockout
                stockout_events += 1
                inventory = 0
            
            inventory_sum += inventory
            
            # Check reorder point
            if inventory <= policy.rop and not pending_orders:
                arrival_day = day + int(profile.production_lead_time_days)
                pending_orders.append((arrival_day, policy.order_quantity))
        
        service_level = 1 - (total_stockouts / total_demand) if total_demand > 0 else 1.0
        avg_inventory = inventory_sum / n_periods
        
        # Total cost
        holding_cost = avg_inventory * profile.holding_cost_per_unit * (n_periods / 365)
        stockout_cost = total_stockouts * profile.stockout_cost_per_unit
        total_cost = holding_cost + stockout_cost
        
        return SimulationResult(
            n_periods=n_periods,
            total_demand=total_demand,
            total_stockouts=total_stockouts,
            avg_inventory=avg_inventory,
            service_level=service_level,
            total_cost=total_cost,
            stockout_events=stockout_events,
        )
    
    def get_optimization_log(self) -> List[Dict[str, Any]]:
        """Return optimization log."""
        return self._optimization_log


# ============================================================
# EXPERIMENT SUPPORT
# ============================================================

def run_coupling_experiment(
    profiles: List[SKUProfile],
    n_simulations: int = 100,
    n_periods: int = 365,
) -> Dict[str, Any]:
    """
    Compare decoupled vs coupled optimization.
    
    TODO[R&D]: Entry point for experiment E3.1.
    """
    results = {
        "decoupled": {"service_levels": [], "costs": [], "stockouts": []},
        "capacity_aware": {"service_levels": [], "costs": [], "stockouts": []},
        "joint": {"service_levels": [], "costs": [], "stockouts": []},
    }
    
    for mode_name, mode in [
        ("decoupled", OptimizationMode.DECOUPLED),
        ("capacity_aware", OptimizationMode.CAPACITY_AWARE),
        ("joint", OptimizationMode.JOINT),
    ]:
        optimizer = InventoryOptimizer(mode=mode)
        
        for profile in profiles:
            # Optimize policy
            policy = optimizer.optimize_sku(
                profile,
                service_level=0.95,
                machine_utilization=0.8,
            )
            
            # Run multiple simulations
            for _ in range(n_simulations // len(profiles)):
                result = optimizer.simulate_policy(profile, policy, n_periods)
                results[mode_name]["service_levels"].append(result.service_level)
                results[mode_name]["costs"].append(result.total_cost)
                results[mode_name]["stockouts"].append(result.stockout_events)
    
    # Compute statistics
    summary = {}
    for mode_name, data in results.items():
        summary[mode_name] = {
            "avg_service_level": np.mean(data["service_levels"]),
            "avg_cost": np.mean(data["costs"]),
            "avg_stockouts": np.mean(data["stockouts"]),
            "std_service_level": np.std(data["service_levels"]),
        }
    
    return summary

