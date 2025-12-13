"""
ProdPlan 4.0 + SmartInventory - Inventory Management Engine

This package provides intelligent inventory management:
- ABC/XYZ classification
- Reorder point (ROP) calculation
- Safety stock optimization
- Demand-supply coupling with production

R&D / SIFIDE: WP3 - SmartInventory Integration
Research Question: Can coupled inventory-production optimization
                   reduce stockouts while maintaining service levels?
"""

from inventory_engine import (
    InventoryEngine,
    compute_abc_xyz,
    compute_rop,
    compute_safety_stock,
    compute_inventory_risk,
)

__all__ = [
    "InventoryEngine",
    "compute_abc_xyz",
    "compute_rop",
    "compute_safety_stock",
    "compute_inventory_risk",
]
