"""
════════════════════════════════════════════════════════════════════════════════════════════════════
LCA ENGINE - Life Cycle Assessment for Duplios
════════════════════════════════════════════════════════════════════════════════════════════════════

Contract 5 Implementation: Simple LCA Calculation

Computes environmental impact metrics based on BOM structure and material factors.

Metrics calculated:
- Carbon footprint (kg CO2eq)
- Water usage (m³)
- Energy consumption (kWh)
- Recycled content (%)
- Recyclability (%)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MATERIAL FACTORS DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MaterialFactor:
    """Environmental impact factors for a material type."""
    material_id: str
    name: str
    carbon_kg_per_kg: float        # kg CO2eq per kg material
    energy_kwh_per_kg: float       # kWh per kg material
    water_m3_per_kg: float         # m³ water per kg material
    recycled_default_pct: float    # Default recycled content %
    recyclability_default_pct: float  # Default recyclability %


# Material impact factors (simplified database)
# Sources: EcoInvent, GREET, industry averages
MATERIAL_FACTORS: Dict[str, MaterialFactor] = {
    # Metals
    "STEEL": MaterialFactor(
        material_id="STEEL",
        name="Steel (carbon)",
        carbon_kg_per_kg=1.85,
        energy_kwh_per_kg=6.0,
        water_m3_per_kg=0.02,
        recycled_default_pct=30.0,
        recyclability_default_pct=90.0,
    ),
    "STAINLESS_STEEL": MaterialFactor(
        material_id="STAINLESS_STEEL",
        name="Stainless Steel",
        carbon_kg_per_kg=4.0,
        energy_kwh_per_kg=12.0,
        water_m3_per_kg=0.03,
        recycled_default_pct=25.0,
        recyclability_default_pct=85.0,
    ),
    "ALUMINUM": MaterialFactor(
        material_id="ALUMINUM",
        name="Aluminum",
        carbon_kg_per_kg=8.1,
        energy_kwh_per_kg=15.0,
        water_m3_per_kg=0.04,
        recycled_default_pct=35.0,
        recyclability_default_pct=95.0,
    ),
    "COPPER": MaterialFactor(
        material_id="COPPER",
        name="Copper",
        carbon_kg_per_kg=3.0,
        energy_kwh_per_kg=8.0,
        water_m3_per_kg=0.05,
        recycled_default_pct=40.0,
        recyclability_default_pct=90.0,
    ),
    
    # Plastics
    "PP": MaterialFactor(
        material_id="PP",
        name="Polypropylene",
        carbon_kg_per_kg=1.9,
        energy_kwh_per_kg=4.0,
        water_m3_per_kg=0.01,
        recycled_default_pct=5.0,
        recyclability_default_pct=60.0,
    ),
    "PE": MaterialFactor(
        material_id="PE",
        name="Polyethylene",
        carbon_kg_per_kg=2.0,
        energy_kwh_per_kg=4.5,
        water_m3_per_kg=0.01,
        recycled_default_pct=10.0,
        recyclability_default_pct=70.0,
    ),
    "PVC": MaterialFactor(
        material_id="PVC",
        name="PVC",
        carbon_kg_per_kg=2.5,
        energy_kwh_per_kg=5.0,
        water_m3_per_kg=0.02,
        recycled_default_pct=5.0,
        recyclability_default_pct=40.0,
    ),
    "ABS": MaterialFactor(
        material_id="ABS",
        name="ABS Plastic",
        carbon_kg_per_kg=3.0,
        energy_kwh_per_kg=6.0,
        water_m3_per_kg=0.02,
        recycled_default_pct=10.0,
        recyclability_default_pct=50.0,
    ),
    
    # Wood & Paper
    "WOOD": MaterialFactor(
        material_id="WOOD",
        name="Wood (processed)",
        carbon_kg_per_kg=0.3,
        energy_kwh_per_kg=1.0,
        water_m3_per_kg=0.005,
        recycled_default_pct=0.0,
        recyclability_default_pct=80.0,
    ),
    "CARDBOARD": MaterialFactor(
        material_id="CARDBOARD",
        name="Cardboard",
        carbon_kg_per_kg=0.8,
        energy_kwh_per_kg=2.0,
        water_m3_per_kg=0.015,
        recycled_default_pct=70.0,
        recyclability_default_pct=90.0,
    ),
    
    # Glass & Ceramics
    "GLASS": MaterialFactor(
        material_id="GLASS",
        name="Glass",
        carbon_kg_per_kg=0.9,
        energy_kwh_per_kg=3.5,
        water_m3_per_kg=0.01,
        recycled_default_pct=25.0,
        recyclability_default_pct=95.0,
    ),
    
    # Electronics
    "PCB": MaterialFactor(
        material_id="PCB",
        name="Printed Circuit Board",
        carbon_kg_per_kg=15.0,
        energy_kwh_per_kg=50.0,
        water_m3_per_kg=0.2,
        recycled_default_pct=5.0,
        recyclability_default_pct=30.0,
    ),
    "BATTERY_LIION": MaterialFactor(
        material_id="BATTERY_LIION",
        name="Li-ion Battery",
        carbon_kg_per_kg=12.0,
        energy_kwh_per_kg=40.0,
        water_m3_per_kg=0.15,
        recycled_default_pct=10.0,
        recyclability_default_pct=50.0,
    ),
    
    # Default (unknown materials)
    "UNKNOWN": MaterialFactor(
        material_id="UNKNOWN",
        name="Unknown Material",
        carbon_kg_per_kg=2.0,
        energy_kwh_per_kg=5.0,
        water_m3_per_kg=0.02,
        recycled_default_pct=10.0,
        recyclability_default_pct=50.0,
    ),
}

# Family to material mapping (for items without explicit material)
FAMILY_TO_MATERIAL: Dict[str, str] = {
    "METAL": "STEEL",
    "METALS": "STEEL",
    "STEEL": "STEEL",
    "ALUMINUM": "ALUMINUM",
    "ALUMINIUM": "ALUMINUM",
    "PLASTIC": "PP",
    "PLASTICS": "PP",
    "WOOD": "WOOD",
    "PAPER": "CARDBOARD",
    "PACKAGING": "CARDBOARD",
    "ELECTRONIC": "PCB",
    "ELECTRONICS": "PCB",
    "GLASS": "GLASS",
}


# ═══════════════════════════════════════════════════════════════════════════════
# LCA RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LCAResult:
    """Result of LCA calculation."""
    revision_id: int
    
    # Environmental impacts
    carbon_kg_co2eq: float = 0.0
    water_m3: float = 0.0
    energy_kwh: float = 0.0
    
    # Circularity
    recycled_content_pct: float = 0.0
    recyclability_pct: float = 0.0
    
    # Breakdown
    components_analyzed: int = 0
    total_mass_kg: float = 0.0
    
    # Component details
    component_breakdown: List[Dict[str, Any]] = field(default_factory=list)
    
    # Calculation metadata
    calculated_at: datetime = field(default_factory=datetime.utcnow)
    data_quality_score: float = 0.0  # 0-100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "carbon_kg_co2eq": round(self.carbon_kg_co2eq, 3),
            "water_m3": round(self.water_m3, 4),
            "energy_kwh": round(self.energy_kwh, 2),
            "recycled_content_pct": round(self.recycled_content_pct, 1),
            "recyclability_pct": round(self.recyclability_pct, 1),
            "components_analyzed": self.components_analyzed,
            "total_mass_kg": round(self.total_mass_kg, 3),
            "calculated_at": self.calculated_at.isoformat(),
            "data_quality_score": round(self.data_quality_score, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# LCA COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_material_factor(family: Optional[str], sku: Optional[str]) -> MaterialFactor:
    """
    Get material factor based on family or SKU.
    
    Priority:
    1. Direct match in MATERIAL_FACTORS (using family or SKU prefix)
    2. Family-to-material mapping
    3. Default (UNKNOWN)
    """
    # Try direct match with family
    if family:
        family_upper = family.upper()
        if family_upper in MATERIAL_FACTORS:
            return MATERIAL_FACTORS[family_upper]
        if family_upper in FAMILY_TO_MATERIAL:
            return MATERIAL_FACTORS[FAMILY_TO_MATERIAL[family_upper]]
    
    # Try match with SKU prefix (common pattern: STEEL-001, AL-002)
    if sku:
        sku_upper = sku.upper()
        for key in MATERIAL_FACTORS:
            if sku_upper.startswith(key) or key in sku_upper:
                return MATERIAL_FACTORS[key]
        for key, material in FAMILY_TO_MATERIAL.items():
            if sku_upper.startswith(key) or key in sku_upper:
                return MATERIAL_FACTORS[material]
    
    return MATERIAL_FACTORS["UNKNOWN"]


def compute_simple_lca(
    revision_id: int,
    db: Optional[Session] = None,
) -> LCAResult:
    """
    Compute simple LCA for an item revision based on its BOM.
    
    Algorithm:
    1. Load BOM for revision
    2. For each component:
       a. Get component's item (for weight, family)
       b. Calculate mass = weight_kg * qty_per_unit * (1 + scrap_rate)
       c. Get material factor
       d. Apply factors to get carbon, energy, water
    3. Sum all impacts
    4. Calculate weighted recycled content and recyclability
    
    Args:
        revision_id: ID of the ItemRevision to analyze
        db: Database session (creates new if None)
    
    Returns:
        LCAResult with calculated impacts
    """
    from duplios.pdm_models import ItemRevision, BomLine, Item
    from duplios.models import SessionLocal
    
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    
    try:
        result = LCAResult(revision_id=revision_id)
        
        # Get revision
        revision = db.query(ItemRevision).filter(ItemRevision.id == revision_id).first()
        if not revision:
            logger.warning(f"Revision {revision_id} not found")
            return result
        
        # Get parent item info
        parent_item = db.query(Item).filter(Item.id == revision.item_id).first()
        
        # Get BOM lines
        bom_lines = db.query(BomLine).filter(BomLine.parent_revision_id == revision_id).all()
        
        if not bom_lines:
            logger.info(f"No BOM found for revision {revision_id}, using parent item only")
            # Use parent item weight if no BOM
            if parent_item and parent_item.weight_kg:
                factor = get_material_factor(parent_item.family, parent_item.sku)
                mass = parent_item.weight_kg
                
                result.carbon_kg_co2eq = mass * factor.carbon_kg_per_kg
                result.energy_kwh = mass * factor.energy_kwh_per_kg
                result.water_m3 = mass * factor.water_m3_per_kg
                result.recycled_content_pct = factor.recycled_default_pct
                result.recyclability_pct = factor.recyclability_default_pct
                result.total_mass_kg = mass
                result.components_analyzed = 1
                result.data_quality_score = 50.0
            
            return result
        
        # Process each BOM line
        total_carbon = 0.0
        total_energy = 0.0
        total_water = 0.0
        total_mass = 0.0
        weighted_recycled = 0.0
        weighted_recyclability = 0.0
        components_with_weight = 0
        
        for bom_line in bom_lines:
            # Get component revision
            comp_revision = db.query(ItemRevision).filter(
                ItemRevision.id == bom_line.component_revision_id
            ).first()
            
            if not comp_revision:
                continue
            
            # Get component item
            comp_item = db.query(Item).filter(Item.id == comp_revision.item_id).first()
            if not comp_item:
                continue
            
            # Get weight
            weight = comp_item.weight_kg or 0.1  # Default 100g if unknown
            
            # Calculate mass with scrap
            qty = bom_line.qty_per_unit
            scrap = bom_line.scrap_rate or 0.0
            mass = weight * qty * (1 + scrap)
            
            # Get material factor
            factor = get_material_factor(comp_item.family, comp_item.sku)
            
            # Calculate impacts
            carbon = mass * factor.carbon_kg_per_kg
            energy = mass * factor.energy_kwh_per_kg
            water = mass * factor.water_m3_per_kg
            
            # Accumulate
            total_carbon += carbon
            total_energy += energy
            total_water += water
            total_mass += mass
            
            # Weight recycled content/recyclability by mass
            weighted_recycled += factor.recycled_default_pct * mass
            weighted_recyclability += factor.recyclability_default_pct * mass
            
            if comp_item.weight_kg:
                components_with_weight += 1
            
            # Track breakdown
            result.component_breakdown.append({
                "component_sku": comp_item.sku,
                "component_name": comp_item.name,
                "qty": qty,
                "mass_kg": round(mass, 4),
                "material": factor.name,
                "carbon_kg": round(carbon, 4),
                "energy_kwh": round(energy, 3),
                "water_m3": round(water, 5),
            })
            
            result.components_analyzed += 1
        
        # Set results
        result.carbon_kg_co2eq = total_carbon
        result.energy_kwh = total_energy
        result.water_m3 = total_water
        result.total_mass_kg = total_mass
        
        if total_mass > 0:
            result.recycled_content_pct = weighted_recycled / total_mass
            result.recyclability_pct = weighted_recyclability / total_mass
        
        # Data quality score (based on how complete the data is)
        if result.components_analyzed > 0:
            data_quality = (components_with_weight / result.components_analyzed) * 100
            result.data_quality_score = data_quality
        
        logger.info(
            f"LCA computed for revision {revision_id}: "
            f"{result.carbon_kg_co2eq:.2f} kg CO2eq, "
            f"{result.components_analyzed} components"
        )
        
        return result
        
    finally:
        if close_session:
            db.close()


def update_dpp_with_lca(
    revision_id: int,
    lca_result: LCAResult,
    db: Optional[Session] = None,
) -> Optional[Any]:
    """
    Update or create DppRecord with LCA results.
    
    Also recalculates trust_index and data_completeness_pct.
    """
    from duplios.dpp_models import DppRecord
    from duplios.pdm_models import ItemRevision, Item
    from duplios.models import SessionLocal
    
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    
    try:
        # Get or create DPP record
        dpp = db.query(DppRecord).filter(
            DppRecord.item_revision_id == revision_id
        ).first()
        
        if not dpp:
            # Create new DPP record
            revision = db.query(ItemRevision).filter(ItemRevision.id == revision_id).first()
            if not revision:
                return None
            
            item = db.query(Item).filter(Item.id == revision.item_id).first()
            if not item:
                return None
            
            dpp = DppRecord(
                item_revision_id=revision_id,
                product_name=item.name,
            )
            db.add(dpp)
        
        # Update LCA fields
        dpp.carbon_kg_co2eq = lca_result.carbon_kg_co2eq
        dpp.water_m3 = lca_result.water_m3
        dpp.energy_kwh = lca_result.energy_kwh
        dpp.recycled_content_pct = lca_result.recycled_content_pct
        dpp.recyclability_pct = lca_result.recyclability_pct
        dpp.last_lca_calculation_at = datetime.utcnow()
        
        # Recalculate trust index and completeness
        dpp.data_completeness_pct = dpp.compute_data_completeness()
        dpp.trust_index = dpp.compute_trust_index()
        
        db.commit()
        db.refresh(dpp)
        
        return dpp
        
    finally:
        if close_session:
            db.close()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_available_materials() -> List[Dict[str, Any]]:
    """Get list of available materials for UI selection."""
    return [
        {
            "id": factor.material_id,
            "name": factor.name,
            "carbon_factor": factor.carbon_kg_per_kg,
            "recyclability": factor.recyclability_default_pct,
        }
        for factor in MATERIAL_FACTORS.values()
        if factor.material_id != "UNKNOWN"
    ]


def estimate_impact_for_material(
    material_id: str,
    mass_kg: float,
) -> Dict[str, float]:
    """Quick estimate of impact for a given material and mass."""
    factor = MATERIAL_FACTORS.get(material_id, MATERIAL_FACTORS["UNKNOWN"])
    
    return {
        "carbon_kg_co2eq": mass_kg * factor.carbon_kg_per_kg,
        "energy_kwh": mass_kg * factor.energy_kwh_per_kg,
        "water_m3": mass_kg * factor.water_m3_per_kg,
        "recycled_content_pct": factor.recycled_default_pct,
        "recyclability_pct": factor.recyclability_default_pct,
    }

