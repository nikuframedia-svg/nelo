"""Pydantic schemas for Duplios DPPs - Extended with full ESPR/CBAM/CSRD support."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class Material(BaseModel):
    material_name: str
    material_type: Optional[str] = None
    percentage: float = Field(ge=0, le=100)
    mass_kg: Optional[float] = None
    recyclable: Optional[bool] = True
    recycled_content: Optional[float] = 0
    origin_country: Optional[str] = None
    supplier: Optional[str] = None


class Component(BaseModel):
    component_name: str
    supplier_name: Optional[str] = None
    weight_kg: Optional[float] = None
    dpp_reference_optional: Optional[str] = None
    origin_country: Optional[str] = None
    co2e_kg: Optional[float] = None


class SupplyChainLeg(BaseModel):
    """Supply chain transport leg."""
    from_location: Optional[str] = None
    to_location: Optional[str] = None
    transport_mode: str = "road"
    distance_km: Optional[float] = None
    co2e_kg: Optional[float] = None
    supplier_name: Optional[str] = None


class ImpactBreakdown(BaseModel):
    materials_kg_co2eq: Optional[float] = 0.0
    manufacturing_kg_co2eq: Optional[float] = 0.0
    distribution_kg_co2eq: Optional[float] = 0.0
    usage_kg_co2eq: Optional[float] = 0.0
    end_of_life_kg_co2eq: Optional[float] = 0.0


class HazardousSubstance(BaseModel):
    substance_name: str
    regulation: Optional[str] = None
    status: str = "below_limit"
    concentration_ppm: Optional[float] = None
    cas_number: Optional[str] = None


class Certification(BaseModel):
    scheme: str
    issuer: Optional[str] = None
    valid_until: Optional[datetime] = None
    certificate_number: Optional[str] = None
    scope: Optional[str] = None


class ThirdPartyAudit(BaseModel):
    auditor_name: str
    scope: str
    date: datetime
    result: str
    report_url: Optional[str] = None


class ManufacturingInfo(BaseModel):
    """Manufacturing site and process information."""
    site_id: Optional[str] = None
    site_name: Optional[str] = None
    location: Optional[str] = None
    country: Optional[str] = None
    production_date: Optional[datetime] = None
    batch_number: Optional[str] = None
    energy_source: Optional[str] = None
    renewable_energy_percent: Optional[float] = None


class EndOfLifeInfo(BaseModel):
    """End of life instructions and recycling info."""
    instructions: Optional[str] = None
    recycling_code: Optional[str] = None
    disassembly_instructions: Optional[str] = None
    hazardous_waste: Optional[bool] = False
    takeback_program: Optional[str] = None


class ComplianceStatus(BaseModel):
    """Regulatory compliance status."""
    espr_compliant: bool = False
    cbam_compliant: bool = False
    csrd_compliant: bool = False
    compliance_score: float = 0
    missing_fields: List[str] = []
    last_evaluated: Optional[datetime] = None


class DPPBase(BaseModel):
    """Base DPP model with all ESPR/CBAM/CSRD fields."""
    # Identification
    gtin: str
    serial_or_lot: Optional[str] = None
    product_name: str
    product_category: Optional[str] = None
    description: Optional[str] = None
    
    # Manufacturer
    manufacturer_name: Optional[str] = None
    manufacturer_eori: Optional[str] = None
    manufacturing_site_id: Optional[str] = None
    country_of_origin: Optional[str] = None
    manufacturing: Optional[ManufacturingInfo] = None
    
    # Composition
    materials: List[Material]
    components: List[Component] = []
    supply_chain: List[SupplyChainLeg] = []
    
    # Environmental Impact
    carbon_footprint_kg_co2eq: float
    impact_breakdown: ImpactBreakdown = ImpactBreakdown()
    water_consumption_m3: Optional[float] = None
    energy_consumption_kwh: Optional[float] = None
    
    # Circularity
    recycled_content_percent: Optional[float] = Field(default=0, ge=0, le=100)
    recyclability_percent: Optional[float] = Field(default=0, ge=0, le=100)
    durability_score: Optional[int] = Field(default=5, ge=1, le=10)
    reparability_score: Optional[int] = Field(default=5, ge=1, le=10)
    expected_lifetime_years: Optional[int] = None
    warranty_months: Optional[int] = None
    
    # Hazardous substances
    hazardous_substances: List[HazardousSubstance] = []
    
    # Verification
    certifications: List[Certification] = []
    third_party_audits: List[ThirdPartyAudit] = []
    
    # End of life
    end_of_life: Optional[EndOfLifeInfo] = None
    
    # Trust & Quality
    trust_index: Optional[float] = Field(default=60, ge=0, le=100)
    data_completeness_percent: Optional[float] = Field(default=0, ge=0, le=100)
    data_freshness_date: Optional[datetime] = None
    
    # Compliance
    compliance: Optional[ComplianceStatus] = None
    
    # Status & URLs
    status: Optional[str] = "draft"
    qr_public_url: Optional[str] = None

    @validator("materials")
    def validate_materials(cls, v):
        if not v:
            raise ValueError("Pelo menos um material é obrigatório")
        total = sum(mat.percentage for mat in v)
        if not 95 <= total <= 105:
            raise ValueError("Percentagens dos materiais devem somar aproximadamente 100%")
        return v


class DPPCreate(DPPBase):
    created_by_user_id: Optional[str] = None


class DPPUpdate(BaseModel):
    product_name: Optional[str] = None
    product_category: Optional[str] = None
    description: Optional[str] = None
    materials: Optional[List[Material]] = None
    components: Optional[List[Component]] = None
    supply_chain: Optional[List[SupplyChainLeg]] = None
    carbon_footprint_kg_co2eq: Optional[float] = None
    impact_breakdown: Optional[ImpactBreakdown] = None
    water_consumption_m3: Optional[float] = None
    energy_consumption_kwh: Optional[float] = None
    recycled_content_percent: Optional[float] = None
    recyclability_percent: Optional[float] = None
    durability_score: Optional[int] = None
    reparability_score: Optional[int] = None
    expected_lifetime_years: Optional[int] = None
    warranty_months: Optional[int] = None
    hazardous_substances: Optional[List[HazardousSubstance]] = None
    certifications: Optional[List[Certification]] = None
    third_party_audits: Optional[List[ThirdPartyAudit]] = None
    end_of_life: Optional[EndOfLifeInfo] = None
    trust_index: Optional[float] = None
    data_completeness_percent: Optional[float] = None
    status: Optional[str] = None


class DPPRead(DPPBase):
    dpp_id: str
    qr_slug: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DPPPublic(BaseModel):
    """Public view of DPP (limited fields for QR scan)."""
    dpp_id: str
    gtin: str
    product_name: str
    product_category: Optional[str] = None
    description: Optional[str] = None
    manufacturer_name: Optional[str] = None
    country_of_origin: Optional[str] = None
    materials: List[Material] = []
    carbon_footprint_kg_co2eq: float = 0
    water_consumption_m3: Optional[float] = None
    recycled_content_percent: Optional[float] = None
    recyclability_percent: Optional[float] = None
    durability_score: Optional[int] = None
    reparability_score: Optional[int] = None
    trust_index: Optional[float] = None
    certifications: List[Certification] = []
    end_of_life: Optional[EndOfLifeInfo] = None
    compliance: Optional[ComplianceStatus] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class DashboardMetrics(BaseModel):
    """Dashboard metrics response."""
    total_dpps: int
    published_dpps: int
    draft_dpps: int
    total_carbon_kg: float
    avg_trust_index: float
    avg_recyclability: float
    avg_recycled_content: float
    espr_compliant_count: int
    cbam_compliant_count: int
    csrd_compliant_count: int
    categories: Dict[str, int]
    recent_dpps: List[Dict[str, Any]]


class ExportRequest(BaseModel):
    """Export request for CSV/Excel."""
    dpp_ids: Optional[List[str]] = None
    format: str = "csv"
    include_fields: Optional[List[str]] = None
