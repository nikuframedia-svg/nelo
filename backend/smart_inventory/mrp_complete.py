"""
════════════════════════════════════════════════════════════════════════════════════════════════════
MRP COMPLETE - Material Requirements Planning Integrado
════════════════════════════════════════════════════════════════════════════════════════════════════

Motor MRP completo integrado com PDM, Forecasting e verificação de capacidade.

Features:
- Explosão de BOM multi-nível via PDM
- Cálculo de necessidades líquidas com MOQ/scrap
- Geração de ordens planejadas (compra + fabricação)
- Integração com engine de forecasting
- Verificação de capacidade e alertas
- Projeção de estoque futuro

Modelo Matemático:
- Necessidade Bruta (GB) = Σ demanda independente + Σ demanda dependente
- Necessidade Líquida (NR) = max(0, GB + SS - (OH + SR))
- Quantidade a Pedir = round_up(NR × (1 + scrap) / múltiplo) × múltiplo
- Se QP < MOQ então QP = MOQ
- Data Início = Data Necessidade - Lead Time

R&D / SIFIDE: WP3 - Inventory & Capacity Optimization
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
import math

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MRPConfig:
    """Configuration for MRP engine."""
    
    # Planning Horizon
    horizon_days: int = 90
    period_days: int = 7  # Weekly buckets
    
    # Defaults
    default_lead_time_days: float = 7.0
    default_safety_stock: float = 0.0
    default_moq: float = 1.0
    default_multiple: float = 1.0
    default_scrap_rate: float = 0.0
    
    # Processing
    max_bom_levels: int = 20
    enable_forecast: bool = True
    enable_capacity_check: bool = True
    
    # Alerts
    alert_low_coverage_days: int = 14
    alert_high_coverage_days: int = 180


class ItemSource(str, Enum):
    """Source/procurement type for an item."""
    MANUFACTURED = "manufactured"  # Produzido internamente
    PURCHASED = "purchased"  # Comprado a fornecedor
    MIXED = "mixed"  # Pode ser ambos


class OrderSource(str, Enum):
    """Source of demand."""
    SALES_ORDER = "sales_order"
    FORECAST = "forecast"
    DEPENDENT_DEMAND = "dependent_demand"
    SAFETY_STOCK = "safety_stock"
    MANUAL = "manual"


class PlannedOrderStatus(str, Enum):
    """Status of a planned order."""
    PLANNED = "planned"
    FIRM = "firm"
    RELEASED = "released"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class PlannedOrderType(str, Enum):
    """Type of planned order."""
    PURCHASE = "purchase"
    MANUFACTURE = "manufacture"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ItemMRPParameters:
    """MRP parameters for an item/SKU."""
    item_id: int
    sku: str
    name: str = ""
    
    # Source
    source: ItemSource = ItemSource.MANUFACTURED
    
    # Stock Parameters
    safety_stock: float = 0.0
    min_stock: float = 0.0
    max_stock: float = float('inf')
    
    # Ordering Parameters
    moq: float = 1.0  # Minimum Order Quantity
    multiple: float = 1.0  # Order in multiples of this
    scrap_rate: float = 0.0  # Expected scrap/loss rate (0-1)
    
    # Timing
    lead_time_days: float = 7.0
    
    # Unit
    unit: str = "pcs"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "sku": self.sku,
            "name": self.name,
            "source": self.source.value,
            "safety_stock": float(self.safety_stock),
            "min_stock": float(self.min_stock),
            "max_stock": float(self.max_stock),
            "moq": float(self.moq),
            "multiple": float(self.multiple),
            "scrap_rate": float(self.scrap_rate),
            "lead_time_days": float(self.lead_time_days),
            "unit": self.unit,
        }


@dataclass
class DemandEntry:
    """A demand entry (order or forecast)."""
    demand_id: str
    item_id: int
    sku: str
    quantity: float
    due_date: datetime
    source: OrderSource
    priority: int = 1
    reference_id: Optional[str] = None  # Order ID, Forecast ID, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "demand_id": self.demand_id,
            "item_id": self.item_id,
            "sku": self.sku,
            "quantity": float(self.quantity),
            "due_date": self.due_date.isoformat(),
            "source": self.source.value,
            "priority": self.priority,
            "reference_id": self.reference_id,
        }


@dataclass
class InventoryPosition:
    """Current inventory position for an item."""
    item_id: int
    sku: str
    on_hand: float = 0.0
    allocated: float = 0.0  # Reserved for orders
    on_order: float = 0.0  # Scheduled receipts (open POs/WOs)
    
    @property
    def available(self) -> float:
        """Available inventory (on-hand - allocated + on-order)."""
        return self.on_hand - self.allocated + self.on_order
    
    @property
    def net_available(self) -> float:
        """Net available (on-hand - allocated)."""
        return self.on_hand - self.allocated
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "sku": self.sku,
            "on_hand": float(self.on_hand),
            "allocated": float(self.allocated),
            "on_order": float(self.on_order),
            "available": float(self.available),
        }


@dataclass
class BomComponent:
    """A BOM component requirement."""
    component_item_id: int
    component_sku: str
    component_name: str
    qty_per_unit: float
    scrap_rate: float = 0.0
    level: int = 0
    parent_item_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_item_id": self.component_item_id,
            "component_sku": self.component_sku,
            "component_name": self.component_name,
            "qty_per_unit": float(self.qty_per_unit),
            "scrap_rate": float(self.scrap_rate),
            "level": self.level,
        }


@dataclass
class PlannedOrder:
    """A planned order (purchase or manufacture)."""
    order_id: str
    item_id: int
    sku: str
    order_type: PlannedOrderType
    status: PlannedOrderStatus
    
    quantity: float
    start_date: datetime  # When to start (place PO or start production)
    due_date: datetime  # When needed
    
    # Calculations
    gross_requirement: float = 0.0
    net_requirement: float = 0.0
    
    # Traceability
    demand_sources: List[str] = field(default_factory=list)
    parent_order_id: Optional[str] = None
    
    # Parameters used
    moq_applied: float = 0.0
    scrap_applied: float = 0.0
    lead_time_days: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "item_id": self.item_id,
            "sku": self.sku,
            "order_type": self.order_type.value,
            "status": self.status.value,
            "quantity": float(self.quantity),
            "start_date": self.start_date.isoformat(),
            "due_date": self.due_date.isoformat(),
            "gross_requirement": float(self.gross_requirement),
            "net_requirement": float(self.net_requirement),
            "demand_sources": self.demand_sources,
            "lead_time_days": float(self.lead_time_days),
        }


@dataclass
class PeriodBucket:
    """Time-phased bucket for MRP calculations."""
    period_start: datetime
    period_end: datetime
    gross_requirements: float = 0.0
    scheduled_receipts: float = 0.0
    projected_on_hand: float = 0.0
    net_requirements: float = 0.0
    planned_receipts: float = 0.0
    planned_releases: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "gross_requirements": float(self.gross_requirements),
            "scheduled_receipts": float(self.scheduled_receipts),
            "projected_on_hand": float(self.projected_on_hand),
            "net_requirements": float(self.net_requirements),
            "planned_receipts": float(self.planned_receipts),
            "planned_releases": float(self.planned_releases),
        }


@dataclass
class ItemMRPPlan:
    """MRP plan for a single item."""
    item_id: int
    sku: str
    name: str
    
    periods: List[PeriodBucket] = field(default_factory=list)
    planned_orders: List[PlannedOrder] = field(default_factory=list)
    
    # Summary
    total_gross_requirement: float = 0.0
    total_net_requirement: float = 0.0
    total_planned_quantity: float = 0.0
    
    # Metrics
    coverage_days: float = 0.0
    stockout_risk: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "sku": self.sku,
            "name": self.name,
            "periods": [p.to_dict() for p in self.periods],
            "planned_orders": [o.to_dict() for o in self.planned_orders],
            "total_gross_requirement": float(self.total_gross_requirement),
            "total_net_requirement": float(self.total_net_requirement),
            "total_planned_quantity": float(self.total_planned_quantity),
            "coverage_days": float(self.coverage_days),
            "stockout_risk": self.stockout_risk,
        }


@dataclass
class CapacityAlert:
    """Capacity constraint alert."""
    work_center: str
    period_start: datetime
    required_hours: float
    available_hours: float
    overload_hours: float
    affected_orders: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "work_center": self.work_center,
            "period_start": self.period_start.isoformat(),
            "required_hours": float(self.required_hours),
            "available_hours": float(self.available_hours),
            "overload_hours": float(self.overload_hours),
            "affected_orders": self.affected_orders,
        }


@dataclass
class MRPRunResult:
    """Complete result of an MRP run."""
    run_id: str
    run_timestamp: datetime
    
    # Configuration
    horizon_start: datetime
    horizon_end: datetime
    period_days: int
    
    # Input counts
    items_processed: int
    demands_processed: int
    bom_levels_exploded: int
    
    # Results
    item_plans: Dict[str, ItemMRPPlan] = field(default_factory=dict)
    purchase_orders: List[PlannedOrder] = field(default_factory=list)
    manufacture_orders: List[PlannedOrder] = field(default_factory=list)
    
    # Alerts
    capacity_alerts: List[CapacityAlert] = field(default_factory=list)
    shortage_alerts: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_timestamp": self.run_timestamp.isoformat(),
            "horizon_start": self.horizon_start.isoformat(),
            "horizon_end": self.horizon_end.isoformat(),
            "period_days": self.period_days,
            "items_processed": self.items_processed,
            "demands_processed": self.demands_processed,
            "bom_levels_exploded": self.bom_levels_exploded,
            "item_plans": {k: v.to_dict() for k, v in self.item_plans.items()},
            "purchase_orders": [o.to_dict() for o in self.purchase_orders],
            "manufacture_orders": [o.to_dict() for o in self.manufacture_orders],
            "capacity_alerts": [a.to_dict() for a in self.capacity_alerts],
            "shortage_alerts": self.shortage_alerts,
            "warnings": self.warnings,
            "summary": {
                "total_purchase_orders": len(self.purchase_orders),
                "total_manufacture_orders": len(self.manufacture_orders),
                "total_capacity_alerts": len(self.capacity_alerts),
                "total_shortage_alerts": len(self.shortage_alerts),
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MRP ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class MRPCompleteEngine:
    """
    Complete MRP Engine with PDM integration.
    
    Process:
    1. Load demands (orders + forecast)
    2. Explode BOM multi-level
    3. Calculate gross requirements by period
    4. Calculate net requirements
    5. Apply lot sizing (MOQ, multiples, scrap)
    6. Generate planned orders
    7. Check capacity constraints
    8. Generate alerts
    
    Usage:
        engine = MRPCompleteEngine()
        engine.load_item_parameters(params_list)
        engine.load_inventory(inventory_dict)
        engine.add_demand(demand_entry)
        result = engine.run_mrp()
    """
    
    def __init__(self, config: Optional[MRPConfig] = None):
        self.config = config or MRPConfig()
        
        # Data stores
        self.item_parameters: Dict[int, ItemMRPParameters] = {}
        self.sku_to_item_id: Dict[str, int] = {}
        self.inventory: Dict[int, InventoryPosition] = {}
        self.demands: List[DemandEntry] = []
        self.bom_structure: Dict[int, List[BomComponent]] = {}  # parent_id -> children
        
        # State
        self._order_counter = 0
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"PO-{datetime.now().strftime('%Y%m%d')}-{self._order_counter:05d}"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def load_item_parameters(self, params: List[ItemMRPParameters]) -> None:
        """Load item MRP parameters."""
        for p in params:
            self.item_parameters[p.item_id] = p
            self.sku_to_item_id[p.sku] = p.item_id
    
    def set_item_parameter(self, param: ItemMRPParameters) -> None:
        """Set single item parameter."""
        self.item_parameters[param.item_id] = param
        self.sku_to_item_id[param.sku] = param.item_id
    
    def load_inventory(self, inventory: Dict[int, InventoryPosition]) -> None:
        """Load inventory positions."""
        self.inventory = inventory
    
    def set_inventory_position(self, position: InventoryPosition) -> None:
        """Set inventory for an item."""
        self.inventory[position.item_id] = position
    
    def add_demand(self, demand: DemandEntry) -> None:
        """Add a demand entry."""
        self.demands.append(demand)
    
    def load_demands(self, demands: List[DemandEntry]) -> None:
        """Load multiple demand entries."""
        self.demands.extend(demands)
    
    def clear_demands(self) -> None:
        """Clear all demands."""
        self.demands = []
    
    def load_bom_from_pdm(self, db_session) -> None:
        """
        Load BOM structure from PDM module.
        
        Requires PDM models to be available.
        """
        try:
            from duplios.pdm_models import Item, ItemRevision, BomLine, RevisionStatus
            
            # Get all released revisions with BOM
            released_revisions = db_session.query(ItemRevision).filter(
                ItemRevision.status == RevisionStatus.RELEASED
            ).all()
            
            for rev in released_revisions:
                item = db_session.query(Item).filter(Item.id == rev.item_id).first()
                if not item:
                    continue
                
                # Get BOM lines
                bom_lines = db_session.query(BomLine).filter(
                    BomLine.parent_revision_id == rev.id
                ).all()
                
                components = []
                for line in bom_lines:
                    comp_rev = db_session.query(ItemRevision).filter(
                        ItemRevision.id == line.component_revision_id
                    ).first()
                    if not comp_rev:
                        continue
                    
                    comp_item = db_session.query(Item).filter(
                        Item.id == comp_rev.item_id
                    ).first()
                    if not comp_item:
                        continue
                    
                    components.append(BomComponent(
                        component_item_id=comp_item.id,
                        component_sku=comp_item.sku,
                        component_name=comp_item.name,
                        qty_per_unit=line.qty_per_unit,
                        scrap_rate=line.scrap_rate,
                        parent_item_id=item.id,
                    ))
                
                if components:
                    self.bom_structure[item.id] = components
                
                # Also create item parameters if not exists
                if item.id not in self.item_parameters:
                    from duplios.pdm_models import ItemType
                    
                    source = ItemSource.MANUFACTURED
                    if item.type == ItemType.RAW_MATERIAL:
                        source = ItemSource.PURCHASED
                    
                    self.item_parameters[item.id] = ItemMRPParameters(
                        item_id=item.id,
                        sku=item.sku,
                        name=item.name,
                        source=source,
                        unit=item.unit,
                    )
                    self.sku_to_item_id[item.sku] = item.id
            
            logger.info(f"Loaded BOM for {len(self.bom_structure)} items from PDM")
            
        except ImportError as e:
            logger.warning(f"PDM module not available: {e}")
    
    def set_bom_structure(self, parent_id: int, components: List[BomComponent]) -> None:
        """Set BOM structure for an item."""
        self.bom_structure[parent_id] = components
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BOM EXPLOSION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def explode_bom(
        self,
        item_id: int,
        quantity: float,
        level: int = 0,
    ) -> List[Tuple[BomComponent, float]]:
        """
        Explode BOM recursively.
        
        Returns list of (component, required_qty) tuples.
        """
        if level > self.config.max_bom_levels:
            logger.warning(f"Max BOM levels exceeded for item {item_id}")
            return []
        
        result = []
        components = self.bom_structure.get(item_id, [])
        
        for comp in components:
            # Calculate required quantity with scrap
            qty_with_scrap = quantity * comp.qty_per_unit * (1 + comp.scrap_rate)
            
            # Update component level
            comp_with_level = BomComponent(
                component_item_id=comp.component_item_id,
                component_sku=comp.component_sku,
                component_name=comp.component_name,
                qty_per_unit=comp.qty_per_unit,
                scrap_rate=comp.scrap_rate,
                level=level + 1,
                parent_item_id=item_id,
            )
            
            result.append((comp_with_level, qty_with_scrap))
            
            # Recursively explode sub-components
            sub_components = self.explode_bom(
                comp.component_item_id,
                qty_with_scrap,
                level + 1,
            )
            result.extend(sub_components)
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LOT SIZING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def apply_lot_sizing(
        self,
        item_id: int,
        net_requirement: float,
    ) -> Tuple[float, float, float]:
        """
        Apply lot sizing rules.
        
        Args:
            item_id: Item ID
            net_requirement: Net requirement quantity
        
        Returns:
            (order_qty, moq_applied, scrap_applied)
        """
        params = self.item_parameters.get(item_id)
        if not params:
            return net_requirement, 0, 0
        
        # Apply scrap rate adjustment as specified:
        # NecessárioBrutoAjustado = NecessidadeBruta / (1 - r)
        # Where r is the scrap rate (0-1)
        if params.scrap_rate > 0 and params.scrap_rate < 1:
            qty_with_scrap = net_requirement / (1 - params.scrap_rate)
        else:
            # Fallback for edge cases
            qty_with_scrap = net_requirement * (1 + params.scrap_rate)
        scrap_applied = qty_with_scrap - net_requirement
        
        # Apply MOQ
        order_qty = qty_with_scrap
        if order_qty < params.moq:
            order_qty = params.moq
        
        moq_applied = max(0, params.moq - qty_with_scrap) if order_qty == params.moq else 0
        
        # Apply multiple
        if params.multiple > 1:
            order_qty = math.ceil(order_qty / params.multiple) * params.multiple
        
        return order_qty, moq_applied, scrap_applied
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PERIOD GENERATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_periods(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[PeriodBucket]:
        """Generate time periods (buckets) for MRP."""
        periods = []
        current = start_date
        
        while current < end_date:
            period_end = min(current + timedelta(days=self.config.period_days), end_date)
            periods.append(PeriodBucket(
                period_start=current,
                period_end=period_end,
            ))
            current = period_end
        
        return periods
    
    def find_period_index(
        self,
        date: datetime,
        periods: List[PeriodBucket],
    ) -> int:
        """Find which period a date falls into."""
        for i, period in enumerate(periods):
            if period.period_start <= date < period.period_end:
                return i
        
        # If after all periods, return last
        if date >= periods[-1].period_end:
            return len(periods) - 1
        
        # If before all periods, return 0
        return 0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MRP CALCULATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_item_mrp(
        self,
        item_id: int,
        gross_requirements: Dict[int, float],  # period_idx -> qty
        periods: List[PeriodBucket],
        dependent_demands: List[Tuple[datetime, float, str]] = None,
    ) -> ItemMRPPlan:
        """
        Calculate MRP for a single item.
        
        Returns ItemMRPPlan with time-phased data and planned orders.
        """
        params = self.item_parameters.get(item_id)
        if not params:
            # Create default parameters
            params = ItemMRPParameters(
                item_id=item_id,
                sku=f"ITEM-{item_id}",
                lead_time_days=self.config.default_lead_time_days,
                safety_stock=self.config.default_safety_stock,
                moq=self.config.default_moq,
                multiple=self.config.default_multiple,
                scrap_rate=self.config.default_scrap_rate,
            )
        
        # Get inventory position
        inv = self.inventory.get(item_id, InventoryPosition(item_id, params.sku))
        
        # Initialize plan
        plan = ItemMRPPlan(
            item_id=item_id,
            sku=params.sku,
            name=params.name,
            periods=[],
        )
        
        # Add dependent demands to gross requirements
        if dependent_demands:
            for due_date, qty, source_id in dependent_demands:
                period_idx = self.find_period_index(due_date, periods)
                gross_requirements[period_idx] = gross_requirements.get(period_idx, 0) + qty
        
        # Calculate MRP period by period
        projected_oh = inv.net_available
        lead_time_periods = int(math.ceil(params.lead_time_days / self.config.period_days))
        
        for t, period in enumerate(periods):
            bucket = PeriodBucket(
                period_start=period.period_start,
                period_end=period.period_end,
            )
            
            # Gross requirements
            bucket.gross_requirements = gross_requirements.get(t, 0)
            plan.total_gross_requirement += bucket.gross_requirements
            
            # Scheduled receipts (orders already placed)
            scheduled_receipts = bucket.scheduled_receipts
            
            # Available stock = current stock + scheduled receipts
            available_stock = projected_oh + scheduled_receipts
            
            # Net requirement calculation as specified:
            # NecessidadeLíquida = max(0, NecessidadeBruta + StockSegurança - (StockAtual + RecebimentosAgendados))
            net_req = max(0, bucket.gross_requirements + params.safety_stock - available_stock)
            bucket.net_requirements = net_req
            plan.total_net_requirement += net_req
            
            if net_req > 0:
                # Apply lot sizing (MOQ, multiples, scrap)
                order_qty, moq_applied, scrap_applied = self.apply_lot_sizing(item_id, net_req)
                bucket.planned_receipts = order_qty
                plan.total_planned_quantity += order_qty
                
                # Calculate release date (offset by lead time)
                # DataPlaneadaLiberação = DataNecessidade - LeadTime
                release_date = period.period_start - timedelta(days=params.lead_time_days)
                
                # Create planned order
                order_type = PlannedOrderType.PURCHASE if params.source == ItemSource.PURCHASED else PlannedOrderType.MANUFACTURE
                
                planned_order = PlannedOrder(
                    order_id=self._generate_order_id(),
                    item_id=item_id,
                    sku=params.sku,
                    order_type=order_type,
                    status=PlannedOrderStatus.PLANNED,
                    quantity=order_qty,
                    start_date=release_date,
                    due_date=period.period_start,
                    gross_requirement=bucket.gross_requirements,
                    net_requirement=net_req,
                    moq_applied=moq_applied,
                    scrap_applied=scrap_applied,
                    lead_time_days=params.lead_time_days,
                )
                
                plan.planned_orders.append(planned_order)
                # Update projected on-hand after planned receipt
                available_stock += order_qty
            
            # Projected on-hand at end of period
            projected_oh = available_stock - bucket.gross_requirements
            bucket.projected_on_hand = projected_oh
            plan.periods.append(bucket)
        
        # Calculate coverage days
        if plan.total_gross_requirement > 0:
            avg_daily_demand = plan.total_gross_requirement / self.config.horizon_days
            if avg_daily_demand > 0:
                plan.coverage_days = inv.net_available / avg_daily_demand
        
        # Check stockout risk
        plan.stockout_risk = any(p.projected_on_hand < 0 for p in plan.periods)
        
        return plan
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN MRP RUN
    # ═══════════════════════════════════════════════════════════════════════════
    
    def run_mrp(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> MRPRunResult:
        """
        Run complete MRP.
        
        Process:
        1. Aggregate independent demands by item and period
        2. Process top-level items (level 0)
        3. Explode BOM and generate dependent demands
        4. Process lower levels iteratively
        5. Generate capacity alerts
        
        Returns:
            MRPRunResult with all plans and orders
        """
        import uuid
        
        start_date = start_date or datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = end_date or start_date + timedelta(days=self.config.horizon_days)
        
        run_id = f"MRP-{uuid.uuid4().hex[:8]}"
        logger.info(f"Starting MRP run {run_id}")
        
        # Generate periods
        periods = self.generate_periods(start_date, end_date)
        
        # Initialize result
        result = MRPRunResult(
            run_id=run_id,
            run_timestamp=datetime.now(timezone.utc),
            horizon_start=start_date,
            horizon_end=end_date,
            period_days=self.config.period_days,
            items_processed=0,
            demands_processed=len(self.demands),
            bom_levels_exploded=0,
        )
        
        # Step 1: Aggregate independent demands by item and period
        independent_demands: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        
        for demand in self.demands:
            item_id = demand.item_id
            period_idx = self.find_period_index(demand.due_date, periods)
            independent_demands[item_id][period_idx] += demand.quantity
        
        # Step 2: Process items level by level
        # Determine processing order using BOM levels
        item_levels: Dict[int, int] = {}  # item_id -> BOM level
        
        # Items with independent demand start at level 0
        for item_id in independent_demands.keys():
            item_levels[item_id] = 0
        
        # Calculate levels based on BOM structure
        def calculate_level(item_id: int, visited: Set[int]) -> int:
            if item_id in visited:
                return 0  # Cycle detection
            visited.add(item_id)
            
            max_child_level = -1
            for child in self.bom_structure.get(item_id, []):
                child_level = calculate_level(child.component_item_id, visited)
                max_child_level = max(max_child_level, child_level)
            
            return max_child_level + 1
        
        for item_id in list(independent_demands.keys()):
            item_levels[item_id] = calculate_level(item_id, set())
        
        # Add all BOM components to levels
        for parent_id, components in self.bom_structure.items():
            for comp in components:
                if comp.component_item_id not in item_levels:
                    item_levels[comp.component_item_id] = item_levels.get(parent_id, 0) + 1
        
        # Sort by level (process higher levels first)
        sorted_items = sorted(item_levels.items(), key=lambda x: x[1])
        
        # Step 3: Process each item
        dependent_demands: Dict[int, List[Tuple[datetime, float, str]]] = defaultdict(list)
        
        for item_id, level in sorted_items:
            # Get gross requirements for this item
            gross_reqs = dict(independent_demands.get(item_id, {}))
            
            # Add dependent demands from parent items
            deps = dependent_demands.get(item_id, [])
            
            # Calculate MRP
            item_plan = self.calculate_item_mrp(
                item_id=item_id,
                gross_requirements=gross_reqs,
                periods=periods,
                dependent_demands=deps,
            )
            
            result.item_plans[item_plan.sku] = item_plan
            result.items_processed += 1
            
            # Classify planned orders
            for order in item_plan.planned_orders:
                if order.order_type == PlannedOrderType.PURCHASE:
                    result.purchase_orders.append(order)
                else:
                    result.manufacture_orders.append(order)
            
            # Explode BOM for manufacture orders -> dependent demands
            for order in item_plan.planned_orders:
                if order.order_type == PlannedOrderType.MANUFACTURE:
                    bom_explosion = self.explode_bom(item_id, order.quantity, level)
                    result.bom_levels_exploded = max(result.bom_levels_exploded, level + 1)
                    
                    for comp, comp_qty in bom_explosion:
                        # Dependent demand due when production starts
                        dependent_demands[comp.component_item_id].append(
                            (order.start_date, comp_qty, order.order_id)
                        )
            
            # Generate shortage alert if stockout risk
            if item_plan.stockout_risk:
                params = self.item_parameters.get(item_id)
                result.shortage_alerts.append({
                    "item_id": item_id,
                    "sku": item_plan.sku,
                    "name": item_plan.name,
                    "stockout_periods": [
                        p.period_start.isoformat()
                        for p in item_plan.periods
                        if p.projected_on_hand < 0
                    ],
                })
        
        # Step 4: Capacity check (simplified)
        if self.config.enable_capacity_check:
            self._check_capacity(result, periods)
        
        logger.info(f"MRP run {run_id} completed: {result.items_processed} items, "
                   f"{len(result.purchase_orders)} purchase orders, "
                   f"{len(result.manufacture_orders)} manufacture orders")
        
        return result
    
    def _check_capacity(
        self,
        result: MRPRunResult,
        periods: List[PeriodBucket],
    ) -> None:
        """
        Check capacity constraints (simplified).
        
        For full implementation, integrate with ProdPlan scheduling data.
        """
        # Group manufacture orders by period
        orders_by_period: Dict[int, List[PlannedOrder]] = defaultdict(list)
        
        for order in result.manufacture_orders:
            period_idx = self.find_period_index(order.start_date, periods)
            orders_by_period[period_idx].append(order)
        
        # Simple capacity check: count orders per period
        # In full implementation, use actual routing times and work center capacity
        for period_idx, orders in orders_by_period.items():
            if len(orders) > 10:  # Threshold for alert
                result.capacity_alerts.append(CapacityAlert(
                    work_center="GENERAL",
                    period_start=periods[period_idx].period_start,
                    required_hours=len(orders) * 8,  # Placeholder
                    available_hours=40,  # Placeholder
                    overload_hours=max(0, len(orders) * 8 - 40),
                    affected_orders=[o.order_id for o in orders],
                ))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FORECAST INTEGRATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def load_forecast(
        self,
        forecast_data: List[Dict[str, Any]],
    ) -> int:
        """
        Load forecast data as demands.
        
        Expected format:
        [{"item_id": 1, "sku": "X", "date": "2025-01-15", "quantity": 100}, ...]
        
        Returns number of forecast entries loaded.
        """
        count = 0
        
        for entry in forecast_data:
            item_id = entry.get("item_id")
            sku = entry.get("sku", "")
            
            if not item_id and sku:
                item_id = self.sku_to_item_id.get(sku)
            
            if not item_id:
                continue
            
            date_str = entry.get("date", entry.get("forecast_date"))
            if isinstance(date_str, str):
                forecast_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            else:
                forecast_date = date_str
            
            self.add_demand(DemandEntry(
                demand_id=f"FC-{item_id}-{forecast_date.strftime('%Y%m%d')}",
                item_id=item_id,
                sku=sku or self.item_parameters.get(item_id, ItemMRPParameters(item_id, "")).sku,
                quantity=float(entry.get("quantity", 0)),
                due_date=forecast_date,
                source=OrderSource.FORECAST,
            ))
            count += 1
        
        logger.info(f"Loaded {count} forecast entries")
        return count


# ═══════════════════════════════════════════════════════════════════════════════
# SERVICE FACADE
# ═══════════════════════════════════════════════════════════════════════════════

class MRPService:
    """
    MRP Service facade for high-level operations.
    
    Usage:
        service = MRPService()
        service.load_from_pdm(db_session)
        service.add_sales_order(...)
        result = service.run_mrp()
    """
    
    def __init__(self, config: Optional[MRPConfig] = None):
        self.config = config or MRPConfig()
        self.engine = MRPCompleteEngine(self.config)
    
    def load_from_pdm(self, db_session) -> None:
        """Load BOM and item data from PDM."""
        self.engine.load_bom_from_pdm(db_session)
    
    def add_sales_order(
        self,
        order_id: str,
        item_id: int,
        sku: str,
        quantity: float,
        due_date: datetime,
    ) -> None:
        """Add a sales order as demand."""
        self.engine.add_demand(DemandEntry(
            demand_id=order_id,
            item_id=item_id,
            sku=sku,
            quantity=quantity,
            due_date=due_date,
            source=OrderSource.SALES_ORDER,
        ))
    
    def load_forecast(self, forecast_data: List[Dict[str, Any]]) -> int:
        """Load forecast data."""
        return self.engine.load_forecast(forecast_data)
    
    def set_inventory(
        self,
        item_id: int,
        sku: str,
        on_hand: float,
        allocated: float = 0,
        on_order: float = 0,
    ) -> None:
        """Set inventory position."""
        self.engine.set_inventory_position(InventoryPosition(
            item_id=item_id,
            sku=sku,
            on_hand=on_hand,
            allocated=allocated,
            on_order=on_order,
        ))
    
    def set_item_parameters(
        self,
        item_id: int,
        sku: str,
        **kwargs,
    ) -> None:
        """Set item MRP parameters."""
        self.engine.set_item_parameter(ItemMRPParameters(
            item_id=item_id,
            sku=sku,
            **kwargs,
        ))
    
    def run_mrp(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> MRPRunResult:
        """Run MRP."""
        return self.engine.run_mrp(start_date, end_date)
    
    def clear(self) -> None:
        """Clear all data."""
        self.engine.demands = []


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_mrp_service_instance: Optional[MRPService] = None


def get_mrp_service() -> MRPService:
    """Get singleton MRP service."""
    global _mrp_service_instance
    if _mrp_service_instance is None:
        _mrp_service_instance = MRPService()
    return _mrp_service_instance


def reset_mrp_service() -> None:
    """Reset singleton."""
    global _mrp_service_instance
    _mrp_service_instance = None


