"""
ProdPlan 4.0 + SmartInventory - Inventory Engine

This module provides inventory optimization capabilities:
- ABC/XYZ classification (value × variability)
- Reorder Point (ROP) calculation
- Safety Stock optimization
- Risk assessment
- Policy simulation

Integration with Production:
- Uses demand forecasts from ml/forecasting
- Considers production lead times
- Couples with APS for capacity-aware decisions

R&D / SIFIDE: WP3 - SmartInventory Integration
Research Questions:
- Q3.1: Can dynamic ROP reduce stockouts by ≥20%?
- Q3.2: What service level is achievable with ML-based forecasting?
Metrics: stockout rate, inventory turns, service level, holding cost.

References:
- Silver et al. (2016). Inventory and Production Management in Supply Chains
- Syntetos et al. (2005). The accuracy of intermittent demand estimates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS AND CONFIG
# ============================================================

class ABCClass(Enum):
    """ABC classification (by value)."""
    A = "A"  # High value (top 20% items = 80% value)
    B = "B"  # Medium value (next 30% items = 15% value)
    C = "C"  # Low value (bottom 50% items = 5% value)


class XYZClass(Enum):
    """XYZ classification (by demand variability)."""
    X = "X"  # Low variability (CV < 0.5)
    Y = "Y"  # Medium variability (0.5 <= CV < 1.0)
    Z = "Z"  # High variability (CV >= 1.0)


class InventoryPolicy(Enum):
    """Inventory replenishment policies."""
    CONTINUOUS_REVIEW = "continuous_review"  # (s, Q) policy
    PERIODIC_REVIEW = "periodic_review"      # (R, S) policy
    KANBAN = "kanban"                        # Pull-based
    MRP = "mrp"                              # Push-based (demand-driven)


@dataclass
class InventoryConfig:
    """Configuration for inventory calculations."""
    # Service level targets
    target_service_level: float = 0.95  # 95%
    
    # ABC thresholds (cumulative percentage)
    abc_a_threshold: float = 0.80  # Top items contributing 80% value
    abc_b_threshold: float = 0.95  # Next items contributing 15% value
    
    # XYZ thresholds (coefficient of variation)
    xyz_x_threshold: float = 0.5
    xyz_y_threshold: float = 1.0
    
    # Lead time (default, in days)
    default_lead_time_days: float = 7.0
    lead_time_std_days: float = 2.0
    
    # Cost parameters
    holding_cost_rate: float = 0.25  # 25% of item value per year
    ordering_cost: float = 50.0      # Fixed cost per order
    stockout_cost_rate: float = 2.0  # Penalty relative to item value
    
    # Review period (for periodic review)
    review_period_days: int = 7
    
    # Safety stock method
    safety_stock_method: str = "normal"  # normal, bootstrap, simulation


@dataclass
class SKUMetrics:
    """Metrics for a single SKU (Stock Keeping Unit)."""
    sku_id: str
    
    # Current state
    current_stock: float
    
    # Classification
    abc_class: ABCClass
    xyz_class: XYZClass
    
    # Demand statistics
    avg_daily_demand: float
    std_daily_demand: float
    cv_demand: float  # Coefficient of variation
    
    # Inventory parameters
    reorder_point: float
    safety_stock: float
    order_quantity: float  # EOQ or fixed
    
    # Risk metrics
    days_of_supply: float
    stockout_risk: float  # Probability of stockout before next replenishment
    excess_risk: float    # Probability of obsolescence
    
    # Value metrics
    annual_value: float
    holding_cost: float
    
    # Recommendations
    action: str  # "order", "monitor", "reduce", "ok"
    priority: int  # 1 = highest
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sku_id': self.sku_id,
            'current_stock': round(self.current_stock, 1),
            'abc_class': self.abc_class.value,
            'xyz_class': self.xyz_class.value,
            'combined_class': f"{self.abc_class.value}{self.xyz_class.value}",
            'avg_daily_demand': round(self.avg_daily_demand, 2),
            'std_daily_demand': round(self.std_daily_demand, 2),
            'cv_demand': round(self.cv_demand, 3),
            'reorder_point': round(self.reorder_point, 1),
            'safety_stock': round(self.safety_stock, 1),
            'order_quantity': round(self.order_quantity, 1),
            'days_of_supply': round(self.days_of_supply, 1),
            'stockout_risk': round(self.stockout_risk, 4),
            'excess_risk': round(self.excess_risk, 4),
            'annual_value': round(self.annual_value, 2),
            'holding_cost': round(self.holding_cost, 2),
            'action': self.action,
            'priority': self.priority,
        }


# ============================================================
# ABC/XYZ CLASSIFICATION
# ============================================================

def compute_abc_xyz(
    stock_data: pd.DataFrame,
    config: Optional[InventoryConfig] = None
) -> pd.DataFrame:
    """
    Compute ABC/XYZ classification for inventory items.
    
    Args:
        stock_data: DataFrame with columns:
            - sku_id
            - unit_value (price per unit)
            - demand_history (list/array of historical demand)
            OR
            - avg_demand, std_demand (pre-computed)
        config: Inventory configuration
    
    Returns:
        DataFrame with ABC and XYZ classifications
    
    ABC: Based on annual value (demand × price)
    XYZ: Based on coefficient of variation of demand
    """
    config = config or InventoryConfig()
    df = stock_data.copy()
    
    # Compute demand statistics if not provided
    if 'avg_demand' not in df.columns:
        if 'demand_history' in df.columns:
            df['avg_demand'] = df['demand_history'].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
            df['std_demand'] = df['demand_history'].apply(lambda x: np.std(x) if len(x) > 1 else 0)
        else:
            df['avg_demand'] = df.get('quantity', 0)
            df['std_demand'] = df['avg_demand'] * 0.3  # Default CV of 0.3
    
    # Compute annual value
    if 'annual_value' not in df.columns:
        unit_value = df.get('unit_value', 1.0)
        df['annual_value'] = df['avg_demand'] * 365 * unit_value
    
    # ABC Classification
    df = df.sort_values('annual_value', ascending=False)
    df['cumulative_value'] = df['annual_value'].cumsum()
    total_value = df['annual_value'].sum()
    
    if total_value > 0:
        df['cumulative_pct'] = df['cumulative_value'] / total_value
    else:
        df['cumulative_pct'] = 0
    
    def assign_abc(pct):
        if pct <= config.abc_a_threshold:
            return ABCClass.A
        elif pct <= config.abc_b_threshold:
            return ABCClass.B
        else:
            return ABCClass.C
    
    df['abc_class'] = df['cumulative_pct'].apply(assign_abc)
    
    # XYZ Classification
    df['cv'] = df.apply(
        lambda row: row['std_demand'] / row['avg_demand'] if row['avg_demand'] > 0 else float('inf'),
        axis=1
    )
    
    def assign_xyz(cv):
        if cv < config.xyz_x_threshold:
            return XYZClass.X
        elif cv < config.xyz_y_threshold:
            return XYZClass.Y
        else:
            return XYZClass.Z
    
    df['xyz_class'] = df['cv'].apply(assign_xyz)
    
    # Combined class
    df['combined_class'] = df.apply(
        lambda row: f"{row['abc_class'].value}{row['xyz_class'].value}",
        axis=1
    )
    
    return df


def get_abc_xyz_matrix(classified_data: pd.DataFrame) -> Dict[str, int]:
    """
    Generate ABC/XYZ matrix counts.
    
    Returns dict with counts for each combination:
    AX, AY, AZ, BX, BY, BZ, CX, CY, CZ
    """
    matrix = {}
    for abc in ['A', 'B', 'C']:
        for xyz in ['X', 'Y', 'Z']:
            key = f"{abc}{xyz}"
            mask = (
                (classified_data['abc_class'].apply(lambda x: x.value if isinstance(x, ABCClass) else x) == abc) &
                (classified_data['xyz_class'].apply(lambda x: x.value if isinstance(x, XYZClass) else x) == xyz)
            )
            matrix[key] = int(mask.sum())
    
    return matrix


# ============================================================
# REORDER POINT AND SAFETY STOCK
# ============================================================

def compute_rop(
    avg_demand: float,
    std_demand: float,
    lead_time_days: float,
    lead_time_std_days: float,
    service_level: float = 0.95,
    method: str = "normal"
) -> Tuple[float, float]:
    """
    Compute Reorder Point and Safety Stock.
    
    ROP = μ_L + z × σ_L
    
    where:
    - μ_L = expected demand during lead time = avg_demand × lead_time
    - σ_L = std of demand during lead time
    - z = safety factor for target service level
    
    Args:
        avg_demand: Average daily demand
        std_demand: Standard deviation of daily demand
        lead_time_days: Average lead time in days
        lead_time_std_days: Std of lead time in days
        service_level: Target service level (e.g., 0.95)
        method: "normal" (assumes normal distribution) or "bootstrap"
    
    Returns:
        (reorder_point, safety_stock)
    
    TODO[R&D]: Compare ROP methods:
    - Normal approximation
    - Bootstrap from historical data
    - Simulation-based
    Metrics: achieved service level, inventory cost
    """
    try:
        from scipy import stats
        z = stats.norm.ppf(service_level)
    except ImportError:
        # Approximate z-score
        z_table = {0.90: 1.28, 0.95: 1.645, 0.99: 2.33}
        z = z_table.get(service_level, 1.645)
    
    # Expected demand during lead time
    mean_demand_lt = avg_demand * lead_time_days
    
    # Variance of demand during lead time
    # Var(D_L) = L × Var(D) + μ_D² × Var(L)
    # (assumes independent demand and lead time)
    var_demand_lt = (
        lead_time_days * (std_demand ** 2) +
        (avg_demand ** 2) * (lead_time_std_days ** 2)
    )
    std_demand_lt = np.sqrt(var_demand_lt)
    
    # Safety stock
    safety_stock = z * std_demand_lt
    
    # Reorder point
    rop = mean_demand_lt + safety_stock
    
    return (max(0, rop), max(0, safety_stock))


def compute_safety_stock(
    std_demand: float,
    lead_time_days: float,
    service_level: float = 0.95,
    review_period_days: float = 0
) -> float:
    """
    Compute safety stock for given parameters.
    
    For continuous review: SS = z × σ_D × √L
    For periodic review: SS = z × σ_D × √(L + R)
    
    where R is review period
    """
    try:
        from scipy import stats
        z = stats.norm.ppf(service_level)
    except ImportError:
        z = 1.645  # 95%
    
    if review_period_days > 0:
        # Periodic review
        protection_period = lead_time_days + review_period_days
    else:
        # Continuous review
        protection_period = lead_time_days
    
    safety_stock = z * std_demand * np.sqrt(protection_period)
    
    return max(0, safety_stock)


def compute_eoq(
    annual_demand: float,
    ordering_cost: float,
    holding_cost_rate: float,
    unit_cost: float
) -> float:
    """
    Compute Economic Order Quantity.
    
    EOQ = √(2 × D × K / h)
    
    where:
    - D = annual demand
    - K = ordering cost per order
    - h = holding cost per unit per year = unit_cost × holding_rate
    """
    if annual_demand <= 0 or ordering_cost <= 0:
        return 0
    
    holding_cost = unit_cost * holding_cost_rate
    
    if holding_cost <= 0:
        return annual_demand  # Order everything at once
    
    eoq = np.sqrt(2 * annual_demand * ordering_cost / holding_cost)
    
    return max(1, eoq)


# ============================================================
# RISK ASSESSMENT
# ============================================================

def compute_inventory_risk(
    current_stock: float,
    avg_demand: float,
    std_demand: float,
    lead_time_days: float,
    reorder_point: float
) -> Dict[str, float]:
    """
    Compute inventory risk metrics.
    
    Returns:
        - stockout_risk: P(stock runs out before replenishment)
        - days_of_supply: current_stock / avg_demand
        - excess_risk: P(stock becomes obsolete)
    """
    # Days of supply
    days_of_supply = current_stock / avg_demand if avg_demand > 0 else float('inf')
    
    # Stockout risk during lead time
    if current_stock < reorder_point:
        # Already below ROP - estimate risk
        demand_during_lt = avg_demand * lead_time_days
        std_during_lt = std_demand * np.sqrt(lead_time_days)
        
        if std_during_lt > 0:
            try:
                from scipy import stats
                # P(demand > current_stock)
                stockout_risk = 1 - stats.norm.cdf(
                    current_stock,
                    loc=demand_during_lt,
                    scale=std_during_lt
                )
            except ImportError:
                # Simple approximation
                z = (current_stock - demand_during_lt) / std_during_lt
                stockout_risk = 0.5 if z >= 0 else 0.5 + abs(z) * 0.3
        else:
            stockout_risk = 1.0 if current_stock < demand_during_lt else 0.0
    else:
        stockout_risk = 0.05  # Minimal risk when above ROP
    
    # Excess risk (for slow-moving items)
    # Simple heuristic: risk increases with days of supply
    if days_of_supply > 365:
        excess_risk = min(1.0, (days_of_supply - 365) / 365)
    elif days_of_supply > 180:
        excess_risk = 0.1
    else:
        excess_risk = 0.01
    
    return {
        'stockout_risk': min(1.0, max(0, stockout_risk)),
        'days_of_supply': days_of_supply,
        'excess_risk': min(1.0, max(0, excess_risk)),
    }


# ============================================================
# INVENTORY ENGINE (High-Level)
# ============================================================

class InventoryEngine:
    """
    High-level inventory management engine.
    
    Integrates:
    - ABC/XYZ classification
    - ROP calculation
    - Risk assessment
    - Policy recommendations
    
    TODO[R&D]: Coupling with production:
    - Use production lead times instead of fixed
    - Consider capacity constraints for replenishment
    - Multi-echelon inventory optimization
    """
    
    def __init__(self, config: Optional[InventoryConfig] = None):
        self.config = config or InventoryConfig()
        self._classified_data: Optional[pd.DataFrame] = None
        self._sku_metrics: Dict[str, SKUMetrics] = {}
    
    def analyze(
        self,
        stock_data: pd.DataFrame,
        demand_forecast: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Perform complete inventory analysis.
        
        Args:
            stock_data: Current inventory data
            demand_forecast: Optional demand forecasts from ML module
        
        Returns:
            Complete analysis with classifications, ROP, risks
        """
        # ABC/XYZ classification
        self._classified_data = compute_abc_xyz(stock_data, self.config)
        
        # Compute metrics for each SKU
        self._sku_metrics = {}
        
        for _, row in self._classified_data.iterrows():
            sku_id = row['sku_id']
            
            # Get demand stats
            avg_demand = row.get('avg_demand', 0)
            std_demand = row.get('std_demand', avg_demand * 0.3)
            
            # Use forecast if available
            if demand_forecast is not None and sku_id in demand_forecast['sku_id'].values:
                forecast_row = demand_forecast[demand_forecast['sku_id'] == sku_id].iloc[0]
                avg_demand = forecast_row.get('forecast_mean', avg_demand)
                std_demand = forecast_row.get('forecast_std', std_demand)
            
            # Compute ROP and safety stock
            lead_time = row.get('lead_time_days', self.config.default_lead_time_days)
            lead_time_std = row.get('lead_time_std', self.config.lead_time_std_days)
            
            rop, safety_stock = compute_rop(
                avg_demand,
                std_demand,
                lead_time,
                lead_time_std,
                self.config.target_service_level
            )
            
            # Compute EOQ
            annual_demand = avg_demand * 365
            unit_value = row.get('unit_value', 1.0)
            eoq = compute_eoq(
                annual_demand,
                self.config.ordering_cost,
                self.config.holding_cost_rate,
                unit_value
            )
            
            # Current stock and risk
            current_stock = row.get('current_stock', row.get('quantity', 0))
            risk = compute_inventory_risk(
                current_stock,
                avg_demand,
                std_demand,
                lead_time,
                rop
            )
            
            # Determine action
            if current_stock <= rop:
                action = "order"
                priority = 1 if row['abc_class'] == ABCClass.A else 2
            elif risk['stockout_risk'] > 0.2:
                action = "monitor"
                priority = 2
            elif risk['excess_risk'] > 0.3:
                action = "reduce"
                priority = 3
            else:
                action = "ok"
                priority = 4
            
            # Create metrics
            cv = std_demand / avg_demand if avg_demand > 0 else float('inf')
            
            self._sku_metrics[sku_id] = SKUMetrics(
                sku_id=sku_id,
                current_stock=current_stock,
                abc_class=row['abc_class'],
                xyz_class=row['xyz_class'],
                avg_daily_demand=avg_demand,
                std_daily_demand=std_demand,
                cv_demand=cv,
                reorder_point=rop,
                safety_stock=safety_stock,
                order_quantity=eoq,
                days_of_supply=risk['days_of_supply'],
                stockout_risk=risk['stockout_risk'],
                excess_risk=risk['excess_risk'],
                annual_value=annual_demand * unit_value,
                holding_cost=current_stock * unit_value * self.config.holding_cost_rate,
                action=action,
                priority=priority,
            )
        
        # Summary
        abc_xyz_matrix = get_abc_xyz_matrix(self._classified_data)
        
        items_to_order = [m for m in self._sku_metrics.values() if m.action == "order"]
        items_at_risk = [m for m in self._sku_metrics.values() if m.stockout_risk > 0.2]
        
        return {
            'total_skus': len(self._sku_metrics),
            'abc_xyz_matrix': abc_xyz_matrix,
            'items_to_order': len(items_to_order),
            'items_at_risk': len(items_at_risk),
            'total_stock_value': sum(m.current_stock * (m.annual_value / 365 / m.avg_daily_demand if m.avg_daily_demand > 0 else 0) for m in self._sku_metrics.values()),
            'total_holding_cost': sum(m.holding_cost for m in self._sku_metrics.values()),
            'avg_days_of_supply': np.mean([m.days_of_supply for m in self._sku_metrics.values() if m.days_of_supply < float('inf')]),
            'high_priority_actions': [m.to_dict() for m in sorted(items_to_order, key=lambda x: x.priority)[:10]],
        }
    
    def get_sku_metrics(self, sku_id: str) -> Optional[SKUMetrics]:
        """Get metrics for a specific SKU."""
        return self._sku_metrics.get(sku_id)
    
    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """Get all SKU metrics as list of dicts."""
        return [m.to_dict() for m in sorted(
            self._sku_metrics.values(),
            key=lambda x: (x.priority, -x.annual_value)
        )]
    
    def simulate_policy(
        self,
        sku_id: str,
        policy: InventoryPolicy,
        simulation_days: int = 365,
        n_simulations: int = 100
    ) -> Dict[str, float]:
        """
        Simulate inventory policy performance.
        
        TODO[R&D]: Policy simulation research:
        - Compare policies under different demand patterns
        - Impact of forecast accuracy on policy choice
        - Multi-item joint optimization
        
        Returns:
            Service level, avg inventory, stockouts, costs
        """
        metrics = self._sku_metrics.get(sku_id)
        if not metrics:
            return {}
        
        results = {
            'service_levels': [],
            'avg_inventories': [],
            'stockout_days': [],
            'order_counts': [],
        }
        
        for _ in range(n_simulations):
            # Simple simulation
            stock = metrics.current_stock
            stockouts = 0
            total_inventory = 0
            orders = 0
            
            for day in range(simulation_days):
                # Generate demand
                demand = max(0, np.random.normal(
                    metrics.avg_daily_demand,
                    metrics.std_daily_demand
                ))
                
                # Check stock
                if stock < demand:
                    stockouts += 1
                    stock = 0
                else:
                    stock -= demand
                
                total_inventory += stock
                
                # Reorder decision
                if policy == InventoryPolicy.CONTINUOUS_REVIEW:
                    if stock <= metrics.reorder_point:
                        stock += metrics.order_quantity
                        orders += 1
                
                elif policy == InventoryPolicy.PERIODIC_REVIEW:
                    if day % self.config.review_period_days == 0:
                        target = metrics.reorder_point + metrics.order_quantity
                        if stock < target:
                            stock += (target - stock)
                            orders += 1
            
            results['service_levels'].append(1 - stockouts / simulation_days)
            results['avg_inventories'].append(total_inventory / simulation_days)
            results['stockout_days'].append(stockouts)
            results['order_counts'].append(orders)
        
        return {
            'avg_service_level': np.mean(results['service_levels']),
            'std_service_level': np.std(results['service_levels']),
            'avg_inventory': np.mean(results['avg_inventories']),
            'avg_stockout_days': np.mean(results['stockout_days']),
            'avg_orders_per_year': np.mean(results['order_counts']),
        }


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def quick_analyze(stock_data: pd.DataFrame) -> Dict[str, Any]:
    """Quick inventory analysis with default config."""
    engine = InventoryEngine()
    return engine.analyze(stock_data)


def get_reorder_list(stock_data: pd.DataFrame) -> pd.DataFrame:
    """Get list of items needing reorder."""
    engine = InventoryEngine()
    engine.analyze(stock_data)
    
    items = [
        m.to_dict() for m in engine._sku_metrics.values()
        if m.action == "order"
    ]
    
    return pd.DataFrame(items).sort_values('priority')



