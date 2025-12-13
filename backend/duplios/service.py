"""Business logic for Duplios DPPs - Extended service layer."""
from __future__ import annotations

import random
import string
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from duplios.models import DPPModel, SessionLocal
from duplios.schemas import DPPCreate, DPPRead, DPPUpdate, DashboardMetrics
from duplios.trust_index_stub import compute_trust_index, calculate_data_completeness
from duplios.carbon_calculator import calculate_carbon_footprint
from duplios.compliance_engine import get_compliance_summary

PUBLIC_BASE_URL = "https://duplios.local"


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _generate_slug(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _build_payload(dpp_in: DPPCreate) -> Dict:
    data = dpp_in.dict()
    return data


def _model_to_read(dpp: DPPModel) -> DPPRead:
    """Convert SQLAlchemy model to Pydantic DPPRead by merging dpp_data with model fields."""
    data = dpp.dpp_data.copy() if dpp.dpp_data else {}
    data.update({
        "dpp_id": dpp.dpp_id,
        "qr_slug": dpp.qr_slug,
        "gtin": dpp.gtin,
        "product_name": dpp.product_name,
        "product_category": dpp.product_category,
        "manufacturer_name": dpp.manufacturer_name,
        "country_of_origin": dpp.country_of_origin,
        "trust_index": dpp.trust_index,
        "carbon_footprint_kg_co2eq": dpp.carbon_footprint_kg_co2eq,
        "recyclability_percent": dpp.recyclability_percent,
        "data_completeness_percent": dpp.data_completeness_percent,
        "qr_public_url": dpp.qr_public_url,
        "status": dpp.status,
        "created_at": dpp.created_at,
        "updated_at": dpp.updated_at,
    })
    return DPPRead.model_validate(data)


def _calculate_metrics(dpp_data: Dict) -> Dict:
    """Calculate all metrics for a DPP."""
    # Carbon footprint
    carbon_result = calculate_carbon_footprint(dpp_data)
    
    # Trust index
    trust = compute_trust_index(dpp_data)
    
    # Data completeness
    completeness = calculate_data_completeness(dpp_data)
    
    # Compliance
    compliance = get_compliance_summary(dpp_data)
    
    return {
        "carbon_footprint_kg_co2eq": carbon_result["total_kg_co2eq"],
        "impact_breakdown": carbon_result["breakdown"],
        "trust_index": trust,
        "data_completeness_percent": completeness,
        "compliance": compliance,
    }


def create_dpp(dpp_in: DPPCreate, db: Session) -> DPPRead:
    """Create a new DPP with calculated metrics."""
    payload = _build_payload(dpp_in)
    
    # Calculate metrics
    metrics = _calculate_metrics(payload)
    
    slug = _generate_slug()
    qr_url = f"{PUBLIC_BASE_URL}/duplios/view/{slug}"
    
    # Update payload with calculated values
    payload.update({
        "carbon_footprint_kg_co2eq": metrics["carbon_footprint_kg_co2eq"],
        "impact_breakdown": metrics["impact_breakdown"],
        "trust_index": metrics["trust_index"],
        "data_completeness_percent": metrics["data_completeness_percent"],
        "compliance": metrics["compliance"],
    })
    
    dpp = DPPModel(
        dpp_id=f"{payload.get('gtin')}-{slug}",
        qr_slug=slug,
        gtin=dpp_in.gtin,
        product_name=dpp_in.product_name,
        product_category=dpp_in.product_category,
        manufacturer_name=dpp_in.manufacturer_name,
        country_of_origin=dpp_in.country_of_origin,
        trust_index=metrics["trust_index"],
        carbon_footprint_kg_co2eq=metrics["carbon_footprint_kg_co2eq"],
        recyclability_percent=dpp_in.recyclability_percent or 0,
        data_completeness_percent=metrics["data_completeness_percent"],
        qr_public_url=qr_url,
        dpp_data=payload,
        created_by_user_id=dpp_in.created_by_user_id,
    )
    db.add(dpp)
    db.commit()
    db.refresh(dpp)
    return _model_to_read(dpp)


def update_dpp(dpp_id: str, dpp_update: DPPUpdate, db: Session) -> DPPRead:
    """Update a DPP and recalculate metrics."""
    dpp = db.query(DPPModel).filter(DPPModel.dpp_id == dpp_id).first()
    if not dpp:
        raise HTTPException(status_code=404, detail="DPP não encontrado")
    
    data = dpp.dpp_data.copy() if dpp.dpp_data else {}
    update_data = dpp_update.dict(exclude_unset=True)
    data.update(update_data)
    
    # Update direct model fields
    for field, value in update_data.items():
        if hasattr(dpp, field) and value is not None:
            setattr(dpp, field, value)
    
    # Recalculate metrics
    metrics = _calculate_metrics(data)
    data.update({
        "carbon_footprint_kg_co2eq": metrics["carbon_footprint_kg_co2eq"],
        "trust_index": metrics["trust_index"],
        "data_completeness_percent": metrics["data_completeness_percent"],
        "compliance": metrics["compliance"],
    })
    
    dpp.dpp_data = data
    dpp.trust_index = metrics["trust_index"]
    dpp.carbon_footprint_kg_co2eq = metrics["carbon_footprint_kg_co2eq"]
    dpp.data_completeness_percent = metrics["data_completeness_percent"]
    dpp.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(dpp)
    return _model_to_read(dpp)


def get_dpp_by_id(dpp_id: str, db: Session) -> DPPRead:
    """Get DPP by ID."""
    dpp = db.query(DPPModel).filter(DPPModel.dpp_id == dpp_id).first()
    if not dpp:
        raise HTTPException(status_code=404, detail="DPP não encontrado")
    return _model_to_read(dpp)


def get_dpp_by_slug(slug: str, db: Session) -> Optional[DPPRead]:
    """Get DPP by QR slug."""
    dpp = db.query(DPPModel).filter(DPPModel.qr_slug == slug).first()
    if not dpp:
        return None
    return _model_to_read(dpp)


def get_dpp_by_gtin(gtin: str, db: Session) -> DPPRead:
    """Get DPP by GTIN."""
    dpp = db.query(DPPModel).filter(DPPModel.gtin == gtin).first()
    if not dpp:
        raise HTTPException(status_code=404, detail="DPP não encontrado")
    return _model_to_read(dpp)


def list_dpps(
    db: Session,
    gtin: Optional[str] = None,
    category: Optional[str] = None,
    status_filter: Optional[str] = None,
    manufacturer: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[DPPRead]:
    """List DPPs with optional filters."""
    query = db.query(DPPModel)
    
    if gtin:
        query = query.filter(DPPModel.gtin == gtin)
    if category:
        query = query.filter(DPPModel.product_category == category)
    if status_filter:
        query = query.filter(DPPModel.status == status_filter)
    if manufacturer:
        query = query.filter(DPPModel.manufacturer_name.ilike(f"%{manufacturer}%"))
    
    dpps = query.order_by(DPPModel.created_at.desc()).offset(offset).limit(limit).all()
    return [_model_to_read(item) for item in dpps]


def delete_dpp(dpp_id: str, db: Session) -> bool:
    """Delete a DPP."""
    dpp = db.query(DPPModel).filter(DPPModel.dpp_id == dpp_id).first()
    if not dpp:
        return False
    db.delete(dpp)
    db.commit()
    return True


def publish_dpp(dpp_id: str, db: Session) -> DPPRead:
    """Publish a DPP after validation."""
    dpp = db.query(DPPModel).filter(DPPModel.dpp_id == dpp_id).first()
    if not dpp:
        raise HTTPException(status_code=404, detail="DPP não encontrado")
    
    # Validate required fields
    mandatory_fields = ["materials", "carbon_footprint_kg_co2eq"]
    for field in mandatory_fields:
        if not dpp.dpp_data.get(field):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Campo obrigatório ausente: {field}"
            )
    
    dpp.status = "published"
    dpp.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(dpp)
    return _model_to_read(dpp)


def recalculate_dpp_metrics(dpp_id: str, db: Session) -> Optional[DPPRead]:
    """Recalculate all metrics for a DPP."""
    dpp = db.query(DPPModel).filter(DPPModel.dpp_id == dpp_id).first()
    if not dpp:
        return None
    
    data = dpp.dpp_data.copy() if dpp.dpp_data else {}
    metrics = _calculate_metrics(data)
    
    data.update({
        "carbon_footprint_kg_co2eq": metrics["carbon_footprint_kg_co2eq"],
        "trust_index": metrics["trust_index"],
        "data_completeness_percent": metrics["data_completeness_percent"],
        "compliance": metrics["compliance"],
    })
    
    dpp.dpp_data = data
    dpp.trust_index = metrics["trust_index"]
    dpp.carbon_footprint_kg_co2eq = metrics["carbon_footprint_kg_co2eq"]
    dpp.data_completeness_percent = metrics["data_completeness_percent"]
    dpp.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(dpp)
    return _model_to_read(dpp)


def get_dashboard_metrics(db: Session) -> DashboardMetrics:
    """Get dashboard metrics and KPIs."""
    dpps = db.query(DPPModel).all()
    
    total = len(dpps)
    published = sum(1 for d in dpps if d.status == "published")
    draft = total - published
    
    total_carbon = sum(d.carbon_footprint_kg_co2eq or 0 for d in dpps)
    avg_trust = sum(d.trust_index or 0 for d in dpps) / total if total > 0 else 0
    avg_recyclability = sum(d.recyclability_percent or 0 for d in dpps) / total if total > 0 else 0
    
    # Calculate average recycled content from dpp_data
    recycled_contents = []
    for d in dpps:
        if d.dpp_data and d.dpp_data.get("recycled_content_percent"):
            recycled_contents.append(d.dpp_data["recycled_content_percent"])
    avg_recycled = sum(recycled_contents) / len(recycled_contents) if recycled_contents else 0
    
    # Compliance counts
    espr_count = 0
    cbam_count = 0
    csrd_count = 0
    for d in dpps:
        if d.dpp_data and d.dpp_data.get("compliance"):
            compliance = d.dpp_data["compliance"]
            if compliance.get("espr_compliant"):
                espr_count += 1
            if compliance.get("cbam_compliant"):
                cbam_count += 1
            if compliance.get("csrd_compliant"):
                csrd_count += 1
    
    # Categories
    categories = {}
    for d in dpps:
        cat = d.product_category or "Sem categoria"
        categories[cat] = categories.get(cat, 0) + 1
    
    # Recent DPPs
    recent = sorted(dpps, key=lambda x: x.created_at or datetime.min, reverse=True)[:5]
    recent_data = [
        {
            "dpp_id": d.dpp_id,
            "product_name": d.product_name,
            "trust_index": d.trust_index,
            "carbon_footprint_kg_co2eq": d.carbon_footprint_kg_co2eq,
            "status": d.status,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in recent
    ]
    
    return DashboardMetrics(
        total_dpps=total,
        published_dpps=published,
        draft_dpps=draft,
        total_carbon_kg=round(total_carbon, 2),
        avg_trust_index=round(avg_trust, 1),
        avg_recyclability=round(avg_recyclability, 1),
        avg_recycled_content=round(avg_recycled, 1),
        espr_compliant_count=espr_count,
        cbam_compliant_count=cbam_count,
        csrd_compliant_count=csrd_count,
        categories=categories,
        recent_dpps=recent_data,
    )
