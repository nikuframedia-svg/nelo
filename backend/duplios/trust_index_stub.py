"""
Trust Index Service - Enhanced calculation based on Duplius MVP.
Calculates trust index score for a DPP based on data completeness and verification.
"""
from __future__ import annotations

from typing import Any, Dict, List


def calculate_trust_index(dpp_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate comprehensive trust index for a DPP.
    Returns score (0-100) and detailed breakdown.
    """
    score = 0
    max_score = 0
    breakdown = {}
    
    # 1. Basic Information Completeness (30 points)
    basic_score = 0
    basic_max = 30
    max_score += basic_max
    
    if dpp_data.get("product_name") or dpp_data.get("name"):
        basic_score += 10
    if dpp_data.get("gtin"):
        basic_score += 10
    if dpp_data.get("manufacturer_name"):
        basic_score += 5
    if dpp_data.get("country_of_origin"):
        basic_score += 5
    
    breakdown["basic_info"] = {"score": basic_score, "max": basic_max}
    score += basic_score
    
    # 2. Material Information (20 points)
    material_score = 0
    material_max = 20
    max_score += material_max
    
    materials = dpp_data.get("materials", [])
    if materials and len(materials) > 0:
        material_score += 10
        # Check for complete material data
        complete_materials = sum(
            1 for m in materials
            if (m.get("material_name") or m.get("material_type"))
            and (m.get("percentage") is not None or m.get("mass_kg") is not None)
        )
        if complete_materials == len(materials):
            material_score += 10
        elif complete_materials > 0:
            material_score += 5
    
    breakdown["materials"] = {"score": material_score, "max": material_max}
    score += material_score
    
    # 3. Environmental Impact Data (15 points)
    env_score = 0
    env_max = 15
    max_score += env_max
    
    if dpp_data.get("carbon_footprint_kg_co2eq"):
        env_score += 8
    if dpp_data.get("water_consumption_m3"):
        env_score += 4
    if dpp_data.get("energy_consumption_kwh"):
        env_score += 3
    
    breakdown["environmental"] = {"score": env_score, "max": env_max}
    score += env_score
    
    # 4. Circularity Metrics (15 points)
    circ_score = 0
    circ_max = 15
    max_score += circ_max
    
    if dpp_data.get("recyclability_percent") is not None:
        circ_score += 5
    if dpp_data.get("recycled_content_percent") is not None:
        circ_score += 5
    if dpp_data.get("durability_score"):
        circ_score += 2.5
    if dpp_data.get("reparability_score"):
        circ_score += 2.5
    
    breakdown["circularity"] = {"score": circ_score, "max": circ_max}
    score += circ_score
    
    # 5. Verification & Certifications (15 points)
    verify_score = 0
    verify_max = 15
    max_score += verify_max
    
    certifications = dpp_data.get("certifications", [])
    audits = dpp_data.get("third_party_audits", [])
    
    if certifications and len(certifications) > 0:
        verify_score += min(8, len(certifications) * 2)
    if audits and len(audits) > 0:
        verify_score += min(7, len(audits) * 2)
    
    breakdown["verification"] = {"score": verify_score, "max": verify_max}
    score += verify_score
    
    # 6. Supply Chain & Traceability (5 points)
    supply_score = 0
    supply_max = 5
    max_score += supply_max
    
    components = dpp_data.get("components", [])
    supply_chain = dpp_data.get("supply_chain", [])
    
    if components and len(components) > 0:
        supply_score += 2.5
    if supply_chain and len(supply_chain) > 0:
        supply_score += 2.5
    elif dpp_data.get("manufacturing_site_id"):
        supply_score += 1.5
    
    breakdown["supply_chain"] = {"score": supply_score, "max": supply_max}
    score += supply_score
    
    # Calculate final percentage
    trust_index = round((score / max_score) * 100) if max_score > 0 else 0
    
    # Determine trust level
    if trust_index >= 80:
        trust_level = "high"
        trust_label = "Alto"
    elif trust_index >= 60:
        trust_level = "medium"
        trust_label = "Médio"
    elif trust_index >= 40:
        trust_level = "low"
        trust_label = "Baixo"
    else:
        trust_level = "very_low"
        trust_label = "Muito Baixo"
    
    return {
        "trust_index": trust_index,
        "trust_level": trust_level,
        "trust_label": trust_label,
        "score": round(score, 1),
        "max_score": max_score,
        "breakdown": breakdown,
    }


def compute_trust_index(data: Dict) -> float:
    """Simple interface - returns just the trust index number."""
    result = calculate_trust_index(data)
    return result["trust_index"]


def get_trust_level(trust_index: float) -> str:
    """Get trust level description."""
    if trust_index >= 80:
        return "High"
    if trust_index >= 60:
        return "Medium"
    if trust_index >= 40:
        return "Low"
    return "Very Low"


def calculate_data_completeness(dpp_data: Dict[str, Any]) -> float:
    """Calculate data completeness percentage."""
    all_fields = [
        "gtin", "product_name", "product_category", "manufacturer_name",
        "manufacturer_eori", "manufacturing_site_id", "country_of_origin",
        "serial_or_lot", "materials", "components", "carbon_footprint_kg_co2eq",
        "water_consumption_m3", "energy_consumption_kwh", "recycled_content_percent",
        "recyclability_percent", "durability_score", "reparability_score",
        "hazardous_substances", "certifications", "third_party_audits",
    ]
    
    filled = 0
    for field in all_fields:
        value = dpp_data.get(field)
        if value is not None:
            if isinstance(value, (list, dict)):
                if len(value) > 0:
                    filled += 1
            elif isinstance(value, str):
                if value.strip():
                    filled += 1
            else:
                filled += 1
    
    return round((filled / len(all_fields)) * 100, 1)
