"""
Carbon Calculator Service - Calculates carbon footprint based on product data.
Based on Duplius MVP carbon calculator with enhanced material factors.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# Material carbon factors (kg CO2e per kg of material)
MATERIAL_FACTORS = {
    # Metals
    "steel": 2.8,
    "aço": 2.8,
    "aluminum": 11.5,
    "alumínio": 11.5,
    "copper": 3.8,
    "cobre": 3.8,
    "iron": 1.9,
    "ferro": 1.9,
    "bronze": 4.2,
    "brass": 3.5,
    "latão": 3.5,
    "zinc": 3.1,
    "zinco": 3.1,
    "nickel": 12.0,
    "níquel": 12.0,
    "titanium": 35.0,
    "titânio": 35.0,
    "lead": 2.0,
    "chumbo": 2.0,
    # Plastics
    "plastic": 2.5,
    "plástico": 2.5,
    "pet": 2.3,
    "hdpe": 1.8,
    "pvc": 2.9,
    "pp": 1.9,
    "polypropylene": 1.9,
    "polipropileno": 1.9,
    "abs": 3.1,
    "nylon": 8.1,
    "polyester": 5.5,
    "poliéster": 5.5,
    "polycarbonate": 6.0,
    "policarbonato": 6.0,
    "epoxy": 6.5,
    "epóxi": 6.5,
    "resina epóxi": 6.5,
    # Natural materials
    "wood": 0.5,
    "madeira": 0.5,
    "mdf": 0.8,
    "paper": 0.8,
    "papel": 0.8,
    "cardboard": 0.9,
    "cartão": 0.9,
    "cotton": 5.0,
    "algodão": 5.0,
    "wool": 22.0,
    "lã": 22.0,
    "leather": 17.0,
    "couro": 17.0,
    "rubber": 3.2,
    "borracha": 3.2,
    "cork": 0.3,
    "cortiça": 0.3,
    "bamboo": 0.4,
    "bambu": 0.4,
    # Glass & Ceramics
    "glass": 1.2,
    "vidro": 1.2,
    "ceramic": 1.5,
    "cerâmica": 1.5,
    "porcelain": 2.0,
    "porcelana": 2.0,
    # Composites
    "composite": 3.0,
    "compósito": 3.0,
    "carbon_fiber": 25.0,
    "fibra de carbono": 25.0,
    "fiberglass": 2.5,
    "fibra de vidro": 2.5,
    # Electronics
    "silicon": 50.0,
    "silício": 50.0,
    "pcb": 8.0,
    "circuit_board": 8.0,
    # Battery materials
    "lithium": 15.0,
    "lítio": 15.0,
    "cobalt": 35.0,
    "cobalto": 35.0,
    "graphite": 4.5,
    "grafite": 4.5,
    # Textiles
    "textile": 5.0,
    "têxtil": 5.0,
    "aramida": 12.0,
    "kevlar": 12.0,
    "eva": 2.8,
    "ptfe": 9.0,
    "teflon": 9.0,
    "nbr": 3.0,
    "viton": 15.0,
    # Lubricants
    "oil": 0.5,
    "óleo": 0.5,
    "lubricant": 0.6,
    "lubrificante": 0.6,
}

# Transport carbon factors (kg CO2e per ton-km)
TRANSPORT_FACTORS = {
    "air": 0.602,
    "aéreo": 0.602,
    "road": 0.062,
    "rodoviário": 0.062,
    "truck": 0.062,
    "camião": 0.062,
    "rail": 0.022,
    "ferroviário": 0.022,
    "comboio": 0.022,
    "sea": 0.015,
    "marítimo": 0.015,
    "ship": 0.015,
    "navio": 0.015,
    "pipeline": 0.005,
    "oleoduto": 0.005,
}

# Energy carbon intensity (kg CO2e per kWh) by region
ENERGY_FACTORS = {
    "pt": 0.25,  # Portugal (high renewables)
    "es": 0.29,  # Spain
    "de": 0.38,  # Germany
    "fr": 0.06,  # France (nuclear)
    "it": 0.33,  # Italy
    "uk": 0.21,  # UK
    "us": 0.42,  # USA average
    "cn": 0.58,  # China
    "world": 0.45,  # Global average
    "default": 0.40,
}


def get_material_factor(material_type: str) -> float:
    """Get carbon factor for a material type."""
    if not material_type:
        return 2.0  # Default factor
    normalized = material_type.lower().strip()
    # Try exact match first
    if normalized in MATERIAL_FACTORS:
        return MATERIAL_FACTORS[normalized]
    # Try partial match
    for key, factor in MATERIAL_FACTORS.items():
        if key in normalized or normalized in key:
            return factor
    return 2.0  # Default factor


def get_transport_factor(mode: str) -> float:
    """Get carbon factor for transport mode (kg CO2e per ton-km)."""
    if not mode:
        return 0.062  # Default to road
    normalized = mode.lower().strip()
    if normalized in TRANSPORT_FACTORS:
        return TRANSPORT_FACTORS[normalized]
    return 0.062  # Default to road


def get_energy_factor(region: str = "default") -> float:
    """Get energy carbon intensity for a region."""
    if not region:
        return ENERGY_FACTORS["default"]
    normalized = region.lower().strip()[:2]
    return ENERGY_FACTORS.get(normalized, ENERGY_FACTORS["default"])


def calculate_materials_carbon(materials: List[Dict[str, Any]]) -> float:
    """Calculate carbon footprint from materials."""
    total = 0.0
    for mat in materials:
        mass = mat.get("mass_kg") or mat.get("quantity") or 0
        mat_type = mat.get("material_type") or mat.get("material_name") or mat.get("type") or ""
        factor = mat.get("carbonFactor") or get_material_factor(mat_type)
        total += mass * factor
    return total


def calculate_transport_carbon(transport: List[Dict[str, Any]], product_mass_kg: float = 1.0) -> float:
    """Calculate carbon footprint from transport."""
    total = 0.0
    for leg in transport:
        distance = leg.get("distance") or leg.get("distance_km") or 0
        mode = leg.get("mode") or leg.get("transport_mode") or "road"
        factor = get_transport_factor(mode)
        # Convert to kg CO2e (factor is per ton-km)
        total += (distance * factor * product_mass_kg) / 1000
    return total


def calculate_energy_carbon(energy_kwh: float, region: str = "default") -> float:
    """Calculate carbon footprint from energy consumption."""
    factor = get_energy_factor(region)
    return energy_kwh * factor


def calculate_carbon_footprint(dpp_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate total carbon footprint for a DPP.
    Returns breakdown by category and total.
    """
    breakdown = {
        "materials_kg_co2eq": 0.0,
        "manufacturing_kg_co2eq": 0.0,
        "transport_kg_co2eq": 0.0,
        "usage_kg_co2eq": 0.0,
        "end_of_life_kg_co2eq": 0.0,
    }
    
    # Materials carbon
    inputs = dpp_data.get("inputs") or {}
    materials = dpp_data.get("materials") or inputs.get("materials") or []
    if materials:
        breakdown["materials_kg_co2eq"] = calculate_materials_carbon(materials)
    
    # Energy/Manufacturing carbon
    manufacturing = dpp_data.get("manufacturing") or {}
    energy = dpp_data.get("energy_consumption_kwh") or inputs.get("energy_kwh") or 0
    region = dpp_data.get("country_of_origin") or manufacturing.get("location") or "default"
    if energy:
        breakdown["manufacturing_kg_co2eq"] = calculate_energy_carbon(energy, region)
    
    # Direct manufacturing carbon if provided
    manufacturing = dpp_data.get("manufacturing") or {}
    if manufacturing.get("carbonFootprint"):
        breakdown["manufacturing_kg_co2eq"] += manufacturing["carbonFootprint"]
    
    # Transport carbon
    transport = dpp_data.get("transport") or dpp_data.get("supply_chain") or []
    if transport:
        # Estimate product mass from materials
        total_mass = sum(m.get("mass_kg", 0) for m in materials) or 1.0
        breakdown["transport_kg_co2eq"] = calculate_transport_carbon(transport, total_mass)
    
    # Usage phase
    usage = dpp_data.get("usage") or {}
    if usage.get("carbonFootprint"):
        breakdown["usage_kg_co2eq"] = usage["carbonFootprint"]
    
    # End of life
    end_of_life = dpp_data.get("endOfLife") or {}
    if end_of_life.get("carbonFootprint"):
        breakdown["end_of_life_kg_co2eq"] = end_of_life["carbonFootprint"]
    
    # Calculate total
    total = sum(breakdown.values())
    
    return {
        "total_kg_co2eq": round(total, 2),
        "breakdown": {k: round(v, 2) for k, v in breakdown.items()},
    }

