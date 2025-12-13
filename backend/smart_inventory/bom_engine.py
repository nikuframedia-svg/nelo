"""
ProdPlan 4.0 - BOM Engine
=========================

Bill of Materials (BOM) explosion and management.

Features:
- Multi-level BOM explosion
- Component requirements calculation
- BOM validation and consistency checks
- Lead time accumulation

R&D / SIFIDE: WP3 - Inventory & Capacity Optimization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class BOMItemType(str, Enum):
    """Type of BOM item."""
    FINISHED_GOOD = "finished_good"
    SEMI_FINISHED = "semi_finished"
    RAW_MATERIAL = "raw_material"
    PACKAGING = "packaging"
    CONSUMABLE = "consumable"


@dataclass
class BOMItem:
    """Single BOM item/component."""
    item_id: str
    name: str
    item_type: BOMItemType = BOMItemType.RAW_MATERIAL
    unit: str = "UN"
    lead_time_days: float = 0
    cost_per_unit: float = 0
    min_order_qty: float = 1
    safety_stock: float = 0


@dataclass
class BOMComponent:
    """Component relationship in BOM."""
    parent_id: str
    component_id: str
    quantity_per: float  # Quantity of component per unit of parent
    sequence: int = 0  # Operation sequence
    scrap_rate: float = 0.0  # Expected scrap rate
    effective_from: Optional[datetime] = None
    effective_to: Optional[datetime] = None


@dataclass
class ExplodedRequirement:
    """Result of BOM explosion for a single component."""
    component_id: str
    component_name: str
    required_qty: float
    level: int  # BOM level (0 = top, 1 = first level, etc.)
    parent_id: Optional[str] = None
    lead_time_days: float = 0
    cumulative_lead_time: float = 0
    due_date: Optional[datetime] = None


# ═══════════════════════════════════════════════════════════════════════════════
# BOM ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class BOMEngine:
    """
    Bill of Materials engine.
    
    Handles multi-level BOM explosion and component requirement calculations.
    """
    
    def __init__(self):
        self.items: Dict[str, BOMItem] = {}
        self.components: List[BOMComponent] = []
        self._parent_map: Dict[str, List[BOMComponent]] = {}
    
    def add_item(self, item: BOMItem) -> None:
        """Add item to BOM master data."""
        self.items[item.item_id] = item
        logger.debug(f"Added BOM item: {item.item_id}")
    
    def add_component(self, component: BOMComponent) -> None:
        """Add component relationship."""
        self.components.append(component)
        
        if component.parent_id not in self._parent_map:
            self._parent_map[component.parent_id] = []
        self._parent_map[component.parent_id].append(component)
        
        logger.debug(f"Added BOM component: {component.parent_id} -> {component.component_id}")
    
    def load_from_dataframe(self, items_df: pd.DataFrame, bom_df: pd.DataFrame) -> None:
        """
        Load BOM data from DataFrames.
        
        Expected columns:
        - items_df: item_id, name, item_type, unit, lead_time_days, cost_per_unit
        - bom_df: parent_id, component_id, quantity_per, sequence, scrap_rate
        """
        # Load items
        for _, row in items_df.iterrows():
            item = BOMItem(
                item_id=str(row["item_id"]),
                name=str(row.get("name", row["item_id"])),
                item_type=BOMItemType(row.get("item_type", "raw_material")),
                unit=str(row.get("unit", "UN")),
                lead_time_days=float(row.get("lead_time_days", 0)),
                cost_per_unit=float(row.get("cost_per_unit", 0)),
            )
            self.add_item(item)
        
        # Load components
        for _, row in bom_df.iterrows():
            comp = BOMComponent(
                parent_id=str(row["parent_id"]),
                component_id=str(row["component_id"]),
                quantity_per=float(row["quantity_per"]),
                sequence=int(row.get("sequence", 0)),
                scrap_rate=float(row.get("scrap_rate", 0)),
            )
            self.add_component(comp)
        
        logger.info(f"Loaded {len(self.items)} items and {len(self.components)} BOM relationships")
    
    def explode(
        self,
        item_id: str,
        quantity: float,
        as_of_date: Optional[datetime] = None,
        max_levels: int = 10,
    ) -> List[ExplodedRequirement]:
        """
        Explode BOM for an item.
        
        Args:
            item_id: Item to explode
            quantity: Required quantity
            as_of_date: Date for effectivity check
            max_levels: Maximum BOM levels to traverse
        
        Returns:
            List of ExplodedRequirement for all components
        """
        as_of_date = as_of_date or datetime.now()
        requirements: List[ExplodedRequirement] = []
        visited: Set[Tuple[str, int]] = set()  # (item_id, level) to avoid cycles
        
        def _explode_recursive(
            current_id: str,
            current_qty: float,
            level: int,
            parent_id: Optional[str],
            cumulative_lt: float,
        ):
            if level > max_levels:
                logger.warning(f"Max BOM levels ({max_levels}) exceeded for {current_id}")
                return
            
            if (current_id, level) in visited:
                logger.warning(f"Cycle detected in BOM: {current_id} at level {level}")
                return
            visited.add((current_id, level))
            
            # Get item info
            item = self.items.get(current_id)
            item_name = item.name if item else current_id
            item_lt = item.lead_time_days if item else 0
            
            # Add requirement
            requirements.append(ExplodedRequirement(
                component_id=current_id,
                component_name=item_name,
                required_qty=current_qty,
                level=level,
                parent_id=parent_id,
                lead_time_days=item_lt,
                cumulative_lead_time=cumulative_lt + item_lt,
            ))
            
            # Get children components
            children = self._parent_map.get(current_id, [])
            
            for child in children:
                # Check effectivity
                if child.effective_from and child.effective_from > as_of_date:
                    continue
                if child.effective_to and child.effective_to < as_of_date:
                    continue
                
                # Calculate child quantity (including scrap)
                child_qty = current_qty * child.quantity_per * (1 + child.scrap_rate)
                
                _explode_recursive(
                    child.component_id,
                    child_qty,
                    level + 1,
                    current_id,
                    cumulative_lt + item_lt,
                )
        
        _explode_recursive(item_id, quantity, 0, None, 0)
        
        return requirements
    
    def get_leaf_requirements(
        self,
        item_id: str,
        quantity: float,
    ) -> Dict[str, float]:
        """
        Get aggregated requirements for leaf items (raw materials).
        
        Returns dict of {component_id: total_quantity}
        """
        requirements = self.explode(item_id, quantity)
        
        # Find leaf items (items with no children)
        leaf_items: Dict[str, float] = {}
        
        for req in requirements:
            if req.component_id not in self._parent_map or not self._parent_map[req.component_id]:
                # This is a leaf item
                if req.component_id not in leaf_items:
                    leaf_items[req.component_id] = 0
                leaf_items[req.component_id] += req.required_qty
        
        return leaf_items
    
    def calculate_cost(self, item_id: str, quantity: float) -> float:
        """
        Calculate total material cost for an item.
        """
        leaf_reqs = self.get_leaf_requirements(item_id, quantity)
        
        total_cost = 0.0
        for comp_id, qty in leaf_reqs.items():
            item = self.items.get(comp_id)
            if item:
                total_cost += qty * item.cost_per_unit
        
        return total_cost
    
    def get_cumulative_lead_time(self, item_id: str) -> float:
        """
        Get cumulative lead time for an item (critical path).
        """
        requirements = self.explode(item_id, 1.0)
        
        if not requirements:
            return 0.0
        
        return max(req.cumulative_lead_time for req in requirements)
    
    def validate_bom(self, item_id: str) -> List[str]:
        """
        Validate BOM structure for an item.
        
        Returns list of validation errors/warnings.
        """
        errors: List[str] = []
        
        # Check if item exists
        if item_id not in self.items:
            errors.append(f"Item {item_id} not found in master data")
        
        # Check for missing components
        children = self._parent_map.get(item_id, [])
        for child in children:
            if child.component_id not in self.items:
                errors.append(f"Component {child.component_id} not found in master data")
            
            if child.quantity_per <= 0:
                errors.append(f"Invalid quantity_per ({child.quantity_per}) for {child.component_id}")
        
        # Check for cycles
        try:
            self.explode(item_id, 1.0)
        except RecursionError:
            errors.append(f"Cycle detected in BOM for {item_id}")
        
        return errors
    
    def to_dataframe(self, item_id: str, quantity: float = 1.0) -> pd.DataFrame:
        """
        Export exploded BOM to DataFrame.
        """
        requirements = self.explode(item_id, quantity)
        
        return pd.DataFrame([
            {
                "level": req.level,
                "component_id": req.component_id,
                "component_name": req.component_name,
                "required_qty": req.required_qty,
                "parent_id": req.parent_id,
                "lead_time_days": req.lead_time_days,
                "cumulative_lead_time": req.cumulative_lead_time,
            }
            for req in requirements
        ])


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_sample_bom() -> BOMEngine:
    """
    Create a sample BOM engine for testing/demo.
    """
    engine = BOMEngine()
    
    # Add items
    engine.add_item(BOMItem("FG-001", "Finished Product A", BOMItemType.FINISHED_GOOD, lead_time_days=0.5))
    engine.add_item(BOMItem("SF-001", "Sub-Assembly 1", BOMItemType.SEMI_FINISHED, lead_time_days=1))
    engine.add_item(BOMItem("SF-002", "Sub-Assembly 2", BOMItemType.SEMI_FINISHED, lead_time_days=1.5))
    engine.add_item(BOMItem("RM-001", "Raw Material A", BOMItemType.RAW_MATERIAL, lead_time_days=5, cost_per_unit=10))
    engine.add_item(BOMItem("RM-002", "Raw Material B", BOMItemType.RAW_MATERIAL, lead_time_days=3, cost_per_unit=5))
    engine.add_item(BOMItem("RM-003", "Raw Material C", BOMItemType.RAW_MATERIAL, lead_time_days=7, cost_per_unit=15))
    engine.add_item(BOMItem("PK-001", "Packaging", BOMItemType.PACKAGING, lead_time_days=2, cost_per_unit=0.5))
    
    # Add BOM relationships
    engine.add_component(BOMComponent("FG-001", "SF-001", 1.0))
    engine.add_component(BOMComponent("FG-001", "SF-002", 2.0))
    engine.add_component(BOMComponent("FG-001", "PK-001", 1.0))
    engine.add_component(BOMComponent("SF-001", "RM-001", 2.0))
    engine.add_component(BOMComponent("SF-001", "RM-002", 1.5))
    engine.add_component(BOMComponent("SF-002", "RM-002", 1.0))
    engine.add_component(BOMComponent("SF-002", "RM-003", 0.5, scrap_rate=0.05))
    
    return engine


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL API FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComponentRequirement:
    """
    Requisito de componente resultante de explosão BOM.
    
    Formato simplificado para integração com MRP.
    """
    component_id: str
    qty_required: float
    is_purchased: bool  # True = comprar, False = fabricar
    is_manufactured: bool  # True = semiacabado, False = matéria-prima
    parent_id: Optional[str] = None
    level: int = 0
    lead_time_days: float = 0
    scrap_rate: float = 0.0
    unit: str = "UN"


# Global BOM engine instance (lazy loaded)
_global_bom_engine: Optional[BOMEngine] = None


def get_bom_engine() -> BOMEngine:
    """Obtém instância global do BOM engine."""
    global _global_bom_engine
    if _global_bom_engine is None:
        _global_bom_engine = BOMEngine()
    return _global_bom_engine


def explode_bom(
    product_id: str,
    revision_id: Optional[str] = None,
    qty: float = 1.0,
    bom_engine: Optional[BOMEngine] = None,
) -> List[ComponentRequirement]:
    """
    Explode BOM de um produto para obter requisitos de componentes.
    
    Esta é a função principal para integração com MRP.
    
    Args:
        product_id: ID do produto final
        revision_id: ID da revisão (opcional, para versionamento)
        qty: Quantidade a produzir
        bom_engine: BOMEngine a usar (se None, usa global)
    
    Returns:
        Lista de ComponentRequirement com todos os componentes necessários
    """
    engine = bom_engine or get_bom_engine()
    
    # Verificar se produto existe
    if product_id not in engine.items:
        logger.warning(f"Product {product_id} not found in BOM. Creating demo BOM.")
        # Criar BOM de demonstração se não existir
        engine = create_sample_bom()
        if product_id not in engine.items:
            return []
    
    # Explodir BOM
    exploded = engine.explode(product_id, qty)
    
    # Converter para ComponentRequirement
    requirements: List[ComponentRequirement] = []
    
    for req in exploded:
        if req.level == 0:
            # Produto principal, ignorar
            continue
        
        item = engine.items.get(req.component_id)
        item_type = item.item_type if item else BOMItemType.RAW_MATERIAL
        
        # Determinar se é comprado ou fabricado
        is_manufactured = item_type in (BOMItemType.SEMI_FINISHED, BOMItemType.FINISHED_GOOD)
        is_purchased = item_type in (BOMItemType.RAW_MATERIAL, BOMItemType.PACKAGING, BOMItemType.CONSUMABLE)
        
        # Obter scrap rate do componente
        scrap_rate = 0.0
        for comp in engine.components:
            if comp.parent_id == req.parent_id and comp.component_id == req.component_id:
                scrap_rate = comp.scrap_rate
                break
        
        requirements.append(ComponentRequirement(
            component_id=req.component_id,
            qty_required=req.required_qty,
            is_purchased=is_purchased,
            is_manufactured=is_manufactured,
            parent_id=req.parent_id,
            level=req.level,
            lead_time_days=req.lead_time_days,
            scrap_rate=scrap_rate,
            unit=item.unit if item else "UN",
        ))
    
    return requirements


def load_bom_from_duplios(product_id: str) -> Optional[BOMEngine]:
    """
    Carrega BOM do módulo Duplios (PDM).
    
    TODO: Integrar com API Duplios para obter BOM real.
    Por agora, retorna BOM de demonstração.
    """
    logger.info(f"Loading BOM for {product_id} from Duplios (demo mode)")
    return create_sample_bom()


def aggregate_requirements(requirements: List[ComponentRequirement]) -> Dict[str, float]:
    """
    Agrega requisitos de componentes (soma quantidades por componente).
    
    Útil para consolidar requisitos de múltiplas ordens.
    """
    aggregated: Dict[str, float] = {}
    
    for req in requirements:
        if req.component_id not in aggregated:
            aggregated[req.component_id] = 0
        aggregated[req.component_id] += req.qty_required
    
    return aggregated

