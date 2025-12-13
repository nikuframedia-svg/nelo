"""
Compliance Engine - Evaluates DPP compliance with EU regulations.
ESPR, CBAM, CSRD compliance checking.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ComplianceResult:
    espr_compliant: bool
    cbam_compliant: bool
    csrd_compliant: bool
    missing_fields: List[str]
    compliance_score: float
    recommendations: List[str]


# Required fields for ESPR (Ecodesign for Sustainable Products Regulation)
ESPR_REQUIRED_FIELDS = [
    "gtin",
    "manufacturer_name",
    "product_category",
    "materials",
    "recyclability_percent",
    "recycled_content_percent",
    "durability_score",
    "reparability_score",
    "carbon_footprint_kg_co2eq",
]

# Required fields for CBAM (Carbon Border Adjustment Mechanism)
CBAM_REQUIRED_FIELDS = [
    "carbon_footprint_kg_co2eq",
    "country_of_origin",
    "manufacturing_site_id",
]

# Required fields for CSRD (Corporate Sustainability Reporting Directive)
CSRD_REQUIRED_FIELDS = [
    "certifications",
    "third_party_audits",
]


def _check_field_exists(data: Dict, field: str) -> bool:
    """Check if a field exists and has meaningful value."""
    value = data.get(field)
    if value is None:
        return False
    if isinstance(value, (list, dict)) and len(value) == 0:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (int, float)) and value == 0:
        # Some fields are valid at 0, others not
        if field in ["carbon_footprint_kg_co2eq"]:
            return True  # 0 carbon is valid (though unusual)
        return False
    return True


def evaluate_espr_compliance(data: Dict) -> tuple[bool, List[str]]:
    """Evaluate ESPR compliance."""
    missing = []
    for field in ESPR_REQUIRED_FIELDS:
        if not _check_field_exists(data, field):
            missing.append(field)
    
    # Additional ESPR checks
    materials = data.get("materials", [])
    if materials:
        # Check if materials have required info
        for i, mat in enumerate(materials):
            if not mat.get("material_name") and not mat.get("material_type"):
                missing.append(f"materials[{i}].material_name")
            if mat.get("percentage") is None:
                missing.append(f"materials[{i}].percentage")
    
    # Check hazardous substances declaration
    if not data.get("hazardous_substances"):
        # Not strictly missing but recommended
        pass
    
    compliant = len(missing) == 0
    return compliant, missing


def evaluate_cbam_compliance(data: Dict) -> tuple[bool, List[str]]:
    """Evaluate CBAM compliance."""
    missing = []
    for field in CBAM_REQUIRED_FIELDS:
        if not _check_field_exists(data, field):
            missing.append(field)
    
    # CBAM requires energy/transport data
    inputs = data.get("inputs") or {}
    has_energy = data.get("energy_consumption_kwh") or inputs.get("energy_kwh")
    has_transport = data.get("transport") or data.get("supply_chain")
    
    if not has_energy and not has_transport:
        missing.append("energy_or_transport_data")
    
    compliant = len(missing) == 0
    return compliant, missing


def evaluate_csrd_compliance(data: Dict) -> tuple[bool, List[str]]:
    """Evaluate CSRD compliance."""
    missing = []
    
    # CSRD requires certification or audit data
    has_certifications = data.get("certifications") and len(data.get("certifications", [])) > 0
    has_audits = data.get("third_party_audits") and len(data.get("third_party_audits", [])) > 0
    has_regulatory = data.get("regulatory_status") and len(data.get("regulatory_status", {})) > 0
    
    if not (has_certifications or has_audits or has_regulatory):
        missing.append("certifications_or_audits")
    
    compliant = len(missing) == 0
    return compliant, missing


def evaluate_compliance(dpp_data: Dict[str, Any]) -> ComplianceResult:
    """
    Evaluate DPP compliance with EU regulations.
    Returns comprehensive compliance result.
    """
    all_missing = []
    recommendations = []
    
    # ESPR compliance
    espr_compliant, espr_missing = evaluate_espr_compliance(dpp_data)
    all_missing.extend([f"ESPR:{f}" for f in espr_missing])
    if not espr_compliant:
        recommendations.append("Adicionar dados de reciclabilidade e composição de materiais para conformidade ESPR")
    
    # CBAM compliance
    cbam_compliant, cbam_missing = evaluate_cbam_compliance(dpp_data)
    all_missing.extend([f"CBAM:{f}" for f in cbam_missing])
    if not cbam_compliant:
        recommendations.append("Adicionar dados de pegada carbónica e origem para conformidade CBAM")
    
    # CSRD compliance
    csrd_compliant, csrd_missing = evaluate_csrd_compliance(dpp_data)
    all_missing.extend([f"CSRD:{f}" for f in csrd_missing])
    if not csrd_compliant:
        recommendations.append("Adicionar certificações ou auditorias para conformidade CSRD")
    
    # Calculate overall compliance score
    total_checks = 3
    passed_checks = sum([espr_compliant, cbam_compliant, csrd_compliant])
    compliance_score = (passed_checks / total_checks) * 100
    
    # Add general recommendations
    if dpp_data.get("trust_index", 0) < 60:
        recommendations.append("Trust Index baixo - completar mais campos do DPP")
    if not dpp_data.get("water_consumption_m3"):
        recommendations.append("Considerar adicionar dados de pegada hídrica")
    
    return ComplianceResult(
        espr_compliant=espr_compliant,
        cbam_compliant=cbam_compliant,
        csrd_compliant=csrd_compliant,
        missing_fields=all_missing,
        compliance_score=round(compliance_score, 1),
        recommendations=recommendations,
    )


def get_compliance_summary(dpp_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get a summary dict suitable for API responses."""
    result = evaluate_compliance(dpp_data)
    return {
        "espr_compliant": result.espr_compliant,
        "cbam_compliant": result.cbam_compliant,
        "csrd_compliant": result.csrd_compliant,
        "compliance_score": result.compliance_score,
        "missing_fields": result.missing_fields,
        "recommendations": result.recommendations,
        "compliant_count": sum([result.espr_compliant, result.cbam_compliant, result.csrd_compliant]),
        "total_regulations": 3,
    }

