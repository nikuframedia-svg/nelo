"""
ProdPlan 4.0 - MRP Engine
=========================

Material Requirements Planning (MRP) engine.

Features:
- Net requirements calculation
- Planned order generation
- Time-phased scheduling
- Integration with BOM engine

R&D / SIFIDE: WP3 - Inventory & Capacity Optimization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from smart_inventory.bom_engine import BOMEngine, ExplodedRequirement

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class OrderType(str, Enum):
    """Type of MRP order."""
    PLANNED_ORDER = "planned_order"
    FIRM_PLANNED = "firm_planned"
    RELEASED = "released"


class RequirementSource(str, Enum):
    """Source of material requirement."""
    FORECAST = "forecast"
    CUSTOMER_ORDER = "customer_order"
    SAFETY_STOCK = "safety_stock"
    DEPENDENT_DEMAND = "dependent_demand"


@dataclass
class GrossRequirement:
    """Gross material requirement."""
    item_id: str
    period: datetime  # Period start
    quantity: float
    source: RequirementSource
    reference_id: Optional[str] = None  # Order ID, Forecast ID, etc.


@dataclass
class InventoryPosition:
    """Current inventory position for an item."""
    item_id: str
    on_hand: float
    on_order: float  # Scheduled receipts
    allocated: float = 0
    safety_stock: float = 0
    
    @property
    def available(self) -> float:
        return self.on_hand + self.on_order - self.allocated


@dataclass
class PlannedOrder:
    """MRP planned order."""
    item_id: str
    order_type: OrderType
    quantity: float
    start_date: datetime  # When to start production/order
    due_date: datetime  # When needed
    lead_time_days: float
    lot_size: Optional[float] = None
    
    # Pegging info
    parent_order_id: Optional[str] = None
    requirement_source: Optional[RequirementSource] = None


@dataclass
class MRPResult:
    """Result of MRP run for an item."""
    item_id: str
    periods: List[datetime]
    gross_requirements: List[float]
    scheduled_receipts: List[float]
    projected_on_hand: List[float]
    net_requirements: List[float]
    planned_order_receipts: List[float]
    planned_order_releases: List[float]
    planned_orders: List[PlannedOrder]


# ═══════════════════════════════════════════════════════════════════════════════
# MRP ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class MRPEngine:
    """
    Material Requirements Planning engine.
    
    Implements standard MRP logic:
    1. Explode BOM
    2. Netting (gross -> net requirements)
    3. Lot sizing
    4. Offsetting (lead time)
    """
    
    def __init__(
        self,
        bom_engine: Optional[BOMEngine] = None,
        planning_horizon_days: int = 90,
        period_days: int = 7,  # Weekly buckets
    ):
        self.bom_engine = bom_engine or BOMEngine()
        self.planning_horizon_days = planning_horizon_days
        self.period_days = period_days
        
        self.gross_requirements: Dict[str, List[GrossRequirement]] = {}
        self.inventory_positions: Dict[str, InventoryPosition] = {}
        self.lot_sizes: Dict[str, float] = {}  # item_id -> fixed lot size
        self.lead_times: Dict[str, float] = {}  # item_id -> lead time in days
    
    def set_inventory(self, item_id: str, position: InventoryPosition) -> None:
        """Set inventory position for an item."""
        self.inventory_positions[item_id] = position
    
    def add_gross_requirement(self, requirement: GrossRequirement) -> None:
        """Add gross requirement."""
        if requirement.item_id not in self.gross_requirements:
            self.gross_requirements[requirement.item_id] = []
        self.gross_requirements[requirement.item_id].append(requirement)
    
    def set_lot_size(self, item_id: str, lot_size: float) -> None:
        """Set fixed lot size for an item."""
        self.lot_sizes[item_id] = lot_size
    
    def set_lead_time(self, item_id: str, lead_time_days: float) -> None:
        """Set lead time for an item."""
        self.lead_times[item_id] = lead_time_days
    
    def _generate_periods(self, start_date: datetime) -> List[datetime]:
        """Generate planning periods."""
        periods = []
        current = start_date
        end_date = start_date + timedelta(days=self.planning_horizon_days)
        
        while current < end_date:
            periods.append(current)
            current += timedelta(days=self.period_days)
        
        return periods
    
    def _aggregate_requirements(
        self,
        item_id: str,
        periods: List[datetime],
    ) -> List[float]:
        """Aggregate gross requirements into periods."""
        reqs = [0.0] * len(periods)
        
        item_reqs = self.gross_requirements.get(item_id, [])
        
        for req in item_reqs:
            # Find the period
            for i, period in enumerate(periods):
                period_end = period + timedelta(days=self.period_days)
                if period <= req.period < period_end:
                    reqs[i] += req.quantity
                    break
        
        return reqs
    
    def _calculate_lot_quantity(self, item_id: str, net_req: float) -> float:
        """Calculate lot quantity based on lot sizing rule."""
        lot_size = self.lot_sizes.get(item_id)
        
        if lot_size:
            # Fixed lot size: round up to nearest lot
            return np.ceil(net_req / lot_size) * lot_size
        else:
            # Lot-for-lot
            return net_req
    
    def run_mrp(
        self,
        item_id: str,
        start_date: Optional[datetime] = None,
        explode_bom: bool = True,
    ) -> MRPResult:
        """
        Run MRP for an item.
        
        Args:
            item_id: Item to plan
            start_date: Start of planning horizon
            explode_bom: Whether to explode BOM for dependent items
        
        Returns:
            MRPResult with time-phased plan
        """
        start_date = start_date or datetime.now()
        periods = self._generate_periods(start_date)
        n_periods = len(periods)
        
        # Initialize arrays
        gross_reqs = self._aggregate_requirements(item_id, periods)
        scheduled_receipts = [0.0] * n_periods
        projected_on_hand = [0.0] * n_periods
        net_reqs = [0.0] * n_periods
        planned_receipts = [0.0] * n_periods
        planned_releases = [0.0] * n_periods
        planned_orders: List[PlannedOrder] = []
        
        # Get inventory position
        inv = self.inventory_positions.get(item_id, InventoryPosition(item_id, 0, 0))
        lead_time = self.lead_times.get(item_id, 0)
        lead_periods = int(np.ceil(lead_time / self.period_days))
        
        # MRP calculation
        prev_poh = inv.on_hand
        
        for t in range(n_periods):
            # Projected on-hand before receipts
            poh_before = prev_poh - gross_reqs[t] + scheduled_receipts[t] + planned_receipts[t]
            
            # Net requirements (considering safety stock)
            net_req = max(0, inv.safety_stock - poh_before)
            net_reqs[t] = net_req
            
            if net_req > 0:
                # Calculate lot quantity
                lot_qty = self._calculate_lot_quantity(item_id, net_req)
                planned_receipts[t] = lot_qty
                
                # Offset by lead time
                release_period = t - lead_periods
                if release_period >= 0:
                    planned_releases[release_period] = lot_qty
                
                # Create planned order
                due_date = periods[t]
                start_dt = due_date - timedelta(days=lead_time)
                
                planned_orders.append(PlannedOrder(
                    item_id=item_id,
                    order_type=OrderType.PLANNED_ORDER,
                    quantity=lot_qty,
                    start_date=start_dt,
                    due_date=due_date,
                    lead_time_days=lead_time,
                    lot_size=self.lot_sizes.get(item_id),
                ))
                
                poh_before += lot_qty
            
            projected_on_hand[t] = poh_before
            prev_poh = poh_before
        
        # Explode BOM if requested
        if explode_bom:
            self._explode_dependent_demand(item_id, planned_orders, start_date)
        
        return MRPResult(
            item_id=item_id,
            periods=periods,
            gross_requirements=gross_reqs,
            scheduled_receipts=scheduled_receipts,
            projected_on_hand=projected_on_hand,
            net_requirements=net_reqs,
            planned_order_receipts=planned_receipts,
            planned_order_releases=planned_releases,
            planned_orders=planned_orders,
        )
    
    def _explode_dependent_demand(
        self,
        item_id: str,
        planned_orders: List[PlannedOrder],
        start_date: datetime,
    ) -> None:
        """
        Explode BOM to create dependent demand (gross requirements) for child items.
        """
        for order in planned_orders:
            # Get BOM components
            children = self.bom_engine._parent_map.get(item_id, [])
            
            for child in children:
                child_qty = order.quantity * child.quantity_per * (1 + child.scrap_rate)
                
                # Create dependent demand requirement
                self.add_gross_requirement(GrossRequirement(
                    item_id=child.component_id,
                    period=order.start_date,  # Needed at start of parent production
                    quantity=child_qty,
                    source=RequirementSource.DEPENDENT_DEMAND,
                    reference_id=f"{item_id}:{order.due_date.isoformat()}",
                ))
    
    def run_full_mrp(
        self,
        top_level_items: List[str],
        start_date: Optional[datetime] = None,
    ) -> Dict[str, MRPResult]:
        """
        Run MRP for all items, processing level by level.
        
        Args:
            top_level_items: List of finished goods to plan
            start_date: Start of planning horizon
        
        Returns:
            Dict of item_id -> MRPResult
        """
        start_date = start_date or datetime.now()
        results: Dict[str, MRPResult] = {}
        
        # Get all items in BOM order (top-down)
        items_to_process = list(top_level_items)
        processed: set = set()
        
        while items_to_process:
            item_id = items_to_process.pop(0)
            
            if item_id in processed:
                continue
            
            # Run MRP for this item
            result = self.run_mrp(item_id, start_date, explode_bom=True)
            results[item_id] = result
            processed.add(item_id)
            
            # Add child items to process queue
            children = self.bom_engine._parent_map.get(item_id, [])
            for child in children:
                if child.component_id not in processed:
                    items_to_process.append(child.component_id)
        
        # Run MRP for child items (they now have dependent demand)
        for item_id in processed:
            if item_id not in top_level_items:
                result = self.run_mrp(item_id, start_date, explode_bom=False)
                results[item_id] = result
        
        return results
    
    def to_dataframe(self, result: MRPResult) -> pd.DataFrame:
        """Convert MRP result to DataFrame."""
        return pd.DataFrame({
            "period": result.periods,
            "gross_requirements": result.gross_requirements,
            "scheduled_receipts": result.scheduled_receipts,
            "projected_on_hand": result.projected_on_hand,
            "net_requirements": result.net_requirements,
            "planned_receipts": result.planned_order_receipts,
            "planned_releases": result.planned_order_releases,
        })


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_simple_mrp(
    item_id: str,
    demand_forecast: List[Tuple[datetime, float]],
    on_hand: float,
    lead_time_days: float = 7,
    safety_stock: float = 0,
    lot_size: Optional[float] = None,
) -> MRPResult:
    """
    Simple MRP run without BOM explosion.
    
    Args:
        item_id: Item to plan
        demand_forecast: List of (date, quantity) tuples
        on_hand: Current on-hand inventory
        lead_time_days: Lead time in days
        safety_stock: Safety stock quantity
        lot_size: Fixed lot size (None for lot-for-lot)
    
    Returns:
        MRPResult
    """
    engine = MRPEngine()
    
    # Set inventory
    engine.set_inventory(item_id, InventoryPosition(
        item_id=item_id,
        on_hand=on_hand,
        on_order=0,
        safety_stock=safety_stock,
    ))
    
    # Set lead time
    engine.set_lead_time(item_id, lead_time_days)
    
    # Set lot size
    if lot_size:
        engine.set_lot_size(item_id, lot_size)
    
    # Add requirements
    for date, qty in demand_forecast:
        engine.add_gross_requirement(GrossRequirement(
            item_id=item_id,
            period=date,
            quantity=qty,
            source=RequirementSource.FORECAST,
        ))
    
    # Run MRP
    start_date = min(date for date, _ in demand_forecast) if demand_forecast else datetime.now()
    return engine.run_mrp(item_id, start_date, explode_bom=False)


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL API: MRP FROM ORDERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Order:
    """Encomenda de cliente."""
    order_id: str
    product_id: str
    quantity: float
    due_date: datetime
    customer_id: str = ""
    priority: int = 1
    status: str = "open"


@dataclass
class PurchaseSuggestion:
    """Sugestão de compra."""
    component_id: str
    quantity: float
    due_date: datetime
    source_order_ids: List[str]
    lead_time_days: float = 0
    estimated_cost: float = 0
    supplier_id: Optional[str] = None


@dataclass
class InternalOrderSuggestion:
    """Sugestão de ordem de produção interna (semiacabado)."""
    item_id: str
    quantity: float
    due_date: datetime
    source_order_ids: List[str]
    lead_time_days: float = 0
    operation_hours: float = 0


@dataclass
class MRPParameters:
    """Parâmetros MRP por SKU."""
    sku_id: str
    min_stock: float = 0
    max_stock: float = float('inf')
    reorder_min_qty: float = 1  # MOQ - Minimum Order Quantity
    reorder_multiple: float = 1  # Múltiplo de encomenda
    scrap_rate: float = 0.0
    lead_time_days: float = 7


@dataclass
class MRPFromOrdersResult:
    """Resultado completo de MRP a partir de encomendas."""
    orders_processed: int
    components_analyzed: int
    purchase_suggestions: List[PurchaseSuggestion]
    internal_order_suggestions: List[InternalOrderSuggestion]
    shortages: List[Dict[str, Any]]  # Componentes em falta
    warnings: List[str]


class MRPFromOrdersEngine:
    """
    Motor MRP completo para processar encomendas.
    
    Processo:
    1. Explodir BOM de cada encomenda
    2. Consultar stock (on_hand, committed)
    3. Aplicar parâmetros MRP (min/max, MOQ, múltiplos, scrap)
    4. Gerar sugestões de compra e produção interna
    """
    
    def __init__(self):
        self.mrp_parameters: Dict[str, MRPParameters] = {}
        self.inventory: Dict[str, InventoryPosition] = {}
        self.bom_engine: Optional[BOMEngine] = None
    
    def set_bom_engine(self, engine: BOMEngine) -> None:
        """Define BOM engine a usar."""
        self.bom_engine = engine
    
    def set_mrp_parameter(self, params: MRPParameters) -> None:
        """Define parâmetros MRP para um SKU."""
        self.mrp_parameters[params.sku_id] = params
    
    def set_inventory(self, item_id: str, position: InventoryPosition) -> None:
        """Define posição de inventário."""
        self.inventory[item_id] = position
    
    def load_parameters_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Carrega parâmetros MRP de DataFrame.
        
        Expected columns: sku_id, min_stock, max_stock, reorder_min_qty, 
                         reorder_multiple, scrap_rate, lead_time_days
        """
        for _, row in df.iterrows():
            self.set_mrp_parameter(MRPParameters(
                sku_id=str(row.get("sku_id", row.get("item_id", ""))),
                min_stock=float(row.get("min_stock", 0)),
                max_stock=float(row.get("max_stock", float('inf'))),
                reorder_min_qty=float(row.get("reorder_min_qty", row.get("moq", 1))),
                reorder_multiple=float(row.get("reorder_multiple", 1)),
                scrap_rate=float(row.get("scrap_rate", 0)),
                lead_time_days=float(row.get("lead_time_days", 7)),
            ))
    
    def run_mrp(
        self,
        orders: List[Order],
        horizon_start: Optional[datetime] = None,
        horizon_end: Optional[datetime] = None,
    ) -> MRPFromOrdersResult:
        """
        Executa MRP para uma lista de encomendas.
        
        Args:
            orders: Lista de encomendas a processar
            horizon_start: Início do horizonte de planeamento
            horizon_end: Fim do horizonte de planeamento
        
        Returns:
            MRPFromOrdersResult com sugestões e alertas
        """
        from .bom_engine import explode_bom, ComponentRequirement
        
        horizon_start = horizon_start or datetime.now()
        horizon_end = horizon_end or horizon_start + timedelta(days=90)
        
        # Agregar requisitos de todas as encomendas
        all_requirements: Dict[str, Dict] = {}  # component_id -> {qty, due_dates, order_ids}
        
        for order in orders:
            # Explodir BOM
            requirements = explode_bom(
                product_id=order.product_id,
                qty=order.quantity,
                bom_engine=self.bom_engine,
            )
            
            for req in requirements:
                if req.component_id not in all_requirements:
                    all_requirements[req.component_id] = {
                        "qty_required": 0,
                        "due_dates": [],
                        "order_ids": [],
                        "is_purchased": req.is_purchased,
                        "is_manufactured": req.is_manufactured,
                        "lead_time_days": req.lead_time_days,
                        "scrap_rate": req.scrap_rate,
                    }
                
                # Adicionar requisito com scrap
                qty_with_scrap = req.qty_required * (1 + req.scrap_rate)
                all_requirements[req.component_id]["qty_required"] += qty_with_scrap
                all_requirements[req.component_id]["due_dates"].append(order.due_date)
                all_requirements[req.component_id]["order_ids"].append(order.order_id)
        
        # Processar requisitos e gerar sugestões
        purchase_suggestions: List[PurchaseSuggestion] = []
        internal_suggestions: List[InternalOrderSuggestion] = []
        shortages: List[Dict[str, Any]] = []
        warnings: List[str] = []
        
        for component_id, req_data in all_requirements.items():
            # Obter stock atual
            inv = self.inventory.get(component_id, InventoryPosition(component_id, 0, 0))
            
            # Obter parâmetros MRP
            params = self.mrp_parameters.get(component_id, MRPParameters(component_id))
            
            # Calcular requisito líquido
            gross_req = req_data["qty_required"]
            available = inv.on_hand + inv.on_order - inv.allocated
            safety_stock = max(inv.safety_stock, params.min_stock)
            
            net_req = gross_req + safety_stock - available
            
            if net_req <= 0:
                # Stock suficiente
                continue
            
            # Aplicar MOQ e múltiplos
            order_qty = net_req
            if order_qty < params.reorder_min_qty:
                order_qty = params.reorder_min_qty
            
            if params.reorder_multiple > 1:
                order_qty = np.ceil(order_qty / params.reorder_multiple) * params.reorder_multiple
            
            # Calcular due date (mais cedo - lead time)
            earliest_due = min(req_data["due_dates"])
            lead_time = params.lead_time_days or req_data["lead_time_days"]
            suggested_due = earliest_due - timedelta(days=lead_time)
            
            if suggested_due < horizon_start:
                warnings.append(f"Componente {component_id}: lead time insuficiente para due date")
                suggested_due = horizon_start
            
            # Criar sugestão
            if req_data["is_purchased"]:
                purchase_suggestions.append(PurchaseSuggestion(
                    component_id=component_id,
                    quantity=order_qty,
                    due_date=suggested_due,
                    source_order_ids=req_data["order_ids"],
                    lead_time_days=lead_time,
                ))
            else:
                internal_suggestions.append(InternalOrderSuggestion(
                    item_id=component_id,
                    quantity=order_qty,
                    due_date=suggested_due,
                    source_order_ids=req_data["order_ids"],
                    lead_time_days=lead_time,
                ))
            
            # Verificar shortage
            if available < 0:
                shortages.append({
                    "component_id": component_id,
                    "shortage_qty": abs(available),
                    "required_qty": gross_req,
                    "available_qty": max(0, inv.on_hand + inv.on_order),
                })
        
        return MRPFromOrdersResult(
            orders_processed=len(orders),
            components_analyzed=len(all_requirements),
            purchase_suggestions=purchase_suggestions,
            internal_order_suggestions=internal_suggestions,
            shortages=shortages,
            warnings=warnings,
        )


def run_mrp_from_orders(
    orders: List[Dict[str, Any]],
    inventory: Optional[Dict[str, Dict]] = None,
    mrp_parameters: Optional[List[Dict]] = None,
) -> MRPFromOrdersResult:
    """
    Função de alto nível para executar MRP a partir de encomendas.
    
    Args:
        orders: Lista de dicts com order_id, product_id, quantity, due_date
        inventory: Dict de {item_id: {on_hand, on_order, allocated}}
        mrp_parameters: Lista de dicts com parâmetros MRP
    
    Returns:
        MRPFromOrdersResult
    """
    engine = MRPFromOrdersEngine()
    
    # Carregar inventário
    if inventory:
        for item_id, inv_data in inventory.items():
            engine.set_inventory(item_id, InventoryPosition(
                item_id=item_id,
                on_hand=inv_data.get("on_hand", 0),
                on_order=inv_data.get("on_order", 0),
                allocated=inv_data.get("allocated", 0),
                safety_stock=inv_data.get("safety_stock", 0),
            ))
    
    # Carregar parâmetros
    if mrp_parameters:
        for params in mrp_parameters:
            engine.set_mrp_parameter(MRPParameters(
                sku_id=params.get("sku_id", params.get("item_id", "")),
                min_stock=params.get("min_stock", 0),
                max_stock=params.get("max_stock", float('inf')),
                reorder_min_qty=params.get("reorder_min_qty", 1),
                reorder_multiple=params.get("reorder_multiple", 1),
                scrap_rate=params.get("scrap_rate", 0),
                lead_time_days=params.get("lead_time_days", 7),
            ))
    
    # Converter orders
    order_list = []
    for o in orders:
        due_date = o.get("due_date")
        if isinstance(due_date, str):
            due_date = datetime.fromisoformat(due_date.replace("Z", "+00:00"))
        
        order_list.append(Order(
            order_id=o.get("order_id", ""),
            product_id=o.get("product_id", o.get("article_id", "")),
            quantity=o.get("quantity", o.get("qty", 1)),
            due_date=due_date or datetime.now() + timedelta(days=7),
            customer_id=o.get("customer_id", ""),
            priority=o.get("priority", 1),
        ))
    
    return engine.run_mrp(order_list)

