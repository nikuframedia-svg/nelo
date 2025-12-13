"""FastAPI router for Duplios DPPs - Extended with dashboard, analytics, exports."""
from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse

from duplios.schemas import (
    DPPCreate, DPPRead, DPPUpdate, DPPPublic, DashboardMetrics, ExportRequest
)
from duplios.service import (
    create_dpp, get_db, get_dpp_by_gtin, get_dpp_by_id, get_dpp_by_slug,
    list_dpps, publish_dpp, update_dpp, delete_dpp, get_dashboard_metrics,
    recalculate_dpp_metrics
)
from duplios.qrcode_service import get_qr_png_bytes
from duplios.compliance_engine import get_compliance_summary
from duplios.carbon_calculator import calculate_carbon_footprint
from duplios.trust_index_stub import calculate_trust_index, calculate_data_completeness

router = APIRouter(prefix="/duplios", tags=["Duplios"])


# =====================
# CRUD Endpoints
# =====================

@router.post("/dpp", response_model=DPPRead)
def api_create_dpp(payload: DPPCreate, db=Depends(get_db)):
    """Create a new Digital Product Passport."""
    return create_dpp(payload, db)


@router.get("/dpp", response_model=List[DPPRead])
def api_list_dpp(
    gtin: Optional[str] = None,
    category: Optional[str] = None,
    status_filter: Optional[str] = None,
    manufacturer: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db=Depends(get_db)
):
    """List all DPPs with optional filters."""
    return list_dpps(
        db, gtin=gtin, category=category, 
        status_filter=status_filter, manufacturer=manufacturer,
        limit=limit, offset=offset
    )


@router.get("/dpp/{dpp_id}", response_model=DPPRead)
def api_get_dpp(dpp_id: str, db=Depends(get_db)):
    """Get DPP by ID."""
    return get_dpp_by_id(dpp_id, db)


@router.get("/dpp/by-gtin/{gtin}", response_model=DPPRead)
def api_get_dpp_gtin(gtin: str, db=Depends(get_db)):
    """Get DPP by GTIN."""
    return get_dpp_by_gtin(gtin, db)


@router.patch("/dpp/{dpp_id}", response_model=DPPRead)
def api_update_dpp(dpp_id: str, payload: DPPUpdate, db=Depends(get_db)):
    """Update a DPP."""
    return update_dpp(dpp_id, payload, db)


@router.delete("/dpp/{dpp_id}")
def api_delete_dpp(dpp_id: str, db=Depends(get_db)):
    """Delete a DPP."""
    success = delete_dpp(dpp_id, db)
    if not success:
        raise HTTPException(status_code=404, detail="DPP não encontrado")
    return {"message": "DPP eliminado com sucesso", "dpp_id": dpp_id}


@router.post("/dpp/{dpp_id}/publish", response_model=DPPRead)
def api_publish_dpp(dpp_id: str, db=Depends(get_db)):
    """Publish a DPP (validate and set status to published)."""
    return publish_dpp(dpp_id, db)


# =====================
# QR Code Endpoints
# =====================

@router.get("/dpp/{dpp_id}/qrcode")
def api_get_qrcode(dpp_id: str):
    """Get QR code image (PNG) for a DPP."""
    try:
        content = get_qr_png_bytes(dpp_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return Response(content=content, media_type="image/png")


# =====================
# Public Endpoints (no auth required for QR scans)
# =====================

@router.get("/public/dpp/{slug}", response_model=DPPPublic)
def api_get_public_dpp(slug: str, db=Depends(get_db)):
    """Get public DPP view by QR slug (for public scanning)."""
    dpp = get_dpp_by_slug(slug, db)
    if not dpp:
        raise HTTPException(status_code=404, detail="DPP não encontrado")
    return dpp


@router.get("/view/{slug}")
def api_view_dpp_redirect(slug: str):
    """Redirect to public DPP viewer (for QR code URLs)."""
    # This would typically redirect to a frontend page
    # For now, return the DPP data
    return {"redirect": f"/duplios/public/dpp/{slug}"}


# =====================
# Analytics & Dashboard
# =====================

@router.get("/dashboard", response_model=DashboardMetrics)
def api_get_dashboard(db=Depends(get_db)):
    """Get dashboard metrics and KPIs."""
    return get_dashboard_metrics(db)


@router.get("/analytics/compliance")
def api_get_compliance_analytics(db=Depends(get_db)):
    """Get compliance analytics across all DPPs."""
    dpps = list_dpps(db)
    
    espr_count = sum(1 for d in dpps if d.compliance and d.compliance.get("espr_compliant"))
    cbam_count = sum(1 for d in dpps if d.compliance and d.compliance.get("cbam_compliant"))
    csrd_count = sum(1 for d in dpps if d.compliance and d.compliance.get("csrd_compliant"))
    
    # Common missing fields
    all_missing = []
    for d in dpps:
        if d.compliance and d.compliance.get("missing_fields"):
            all_missing.extend(d.compliance["missing_fields"])
    
    from collections import Counter
    missing_counts = Counter(all_missing)
    
    return {
        "total_dpps": len(dpps),
        "espr_compliant": espr_count,
        "cbam_compliant": cbam_count,
        "csrd_compliant": csrd_count,
        "espr_percent": round(espr_count / len(dpps) * 100, 1) if dpps else 0,
        "cbam_percent": round(cbam_count / len(dpps) * 100, 1) if dpps else 0,
        "csrd_percent": round(csrd_count / len(dpps) * 100, 1) if dpps else 0,
        "common_missing_fields": dict(missing_counts.most_common(10)),
    }


@router.get("/analytics/carbon")
def api_get_carbon_analytics(db=Depends(get_db)):
    """Get carbon footprint analytics."""
    dpps = list_dpps(db)
    
    total_carbon = sum(d.carbon_footprint_kg_co2eq or 0 for d in dpps)
    avg_carbon = total_carbon / len(dpps) if dpps else 0
    
    # Group by category
    by_category = {}
    for d in dpps:
        cat = d.product_category or "Sem categoria"
        if cat not in by_category:
            by_category[cat] = {"count": 0, "total_carbon": 0}
        by_category[cat]["count"] += 1
        by_category[cat]["total_carbon"] += d.carbon_footprint_kg_co2eq or 0
    
    return {
        "total_carbon_kg": round(total_carbon, 2),
        "average_carbon_kg": round(avg_carbon, 2),
        "by_category": by_category,
        "products_count": len(dpps),
    }


# =====================
# Recalculation Endpoints
# =====================

@router.post("/dpp/{dpp_id}/recalculate")
def api_recalculate_dpp(dpp_id: str, db=Depends(get_db)):
    """Recalculate carbon footprint, trust index, and compliance for a DPP."""
    dpp = recalculate_dpp_metrics(dpp_id, db)
    if not dpp:
        raise HTTPException(status_code=404, detail="DPP não encontrado")
    return {
        "message": "Métricas recalculadas",
        "dpp_id": dpp_id,
        "carbon_footprint_kg_co2eq": dpp.carbon_footprint_kg_co2eq,
        "trust_index": dpp.trust_index,
        "data_completeness_percent": dpp.data_completeness_percent,
    }


@router.get("/dpp/{dpp_id}/compliance")
def api_get_dpp_compliance(dpp_id: str, db=Depends(get_db)):
    """Get detailed compliance evaluation for a DPP."""
    dpp = get_dpp_by_id(dpp_id, db)
    if not dpp:
        raise HTTPException(status_code=404, detail="DPP não encontrado")
    
    dpp_dict = dpp.dict() if hasattr(dpp, 'dict') else dpp.__dict__
    return get_compliance_summary(dpp_dict)


@router.get("/dpp/{dpp_id}/carbon-breakdown")
def api_get_carbon_breakdown(dpp_id: str, db=Depends(get_db)):
    """Get detailed carbon footprint breakdown for a DPP."""
    dpp = get_dpp_by_id(dpp_id, db)
    if not dpp:
        raise HTTPException(status_code=404, detail="DPP não encontrado")
    
    dpp_dict = dpp.dict() if hasattr(dpp, 'dict') else dpp.__dict__
    return calculate_carbon_footprint(dpp_dict)


@router.get("/dpp/{dpp_id}/trust-breakdown")
def api_get_trust_breakdown(dpp_id: str, db=Depends(get_db)):
    """Get detailed trust index breakdown for a DPP."""
    dpp = get_dpp_by_id(dpp_id, db)
    if not dpp:
        raise HTTPException(status_code=404, detail="DPP não encontrado")
    
    dpp_dict = dpp.dict() if hasattr(dpp, 'dict') else dpp.__dict__
    return calculate_trust_index(dpp_dict)


# =====================
# Export Endpoints
# =====================

@router.get("/export/csv")
def api_export_csv(
    status_filter: Optional[str] = None,
    category: Optional[str] = None,
    db=Depends(get_db)
):
    """Export DPPs to CSV format."""
    dpps = list_dpps(db, status_filter=status_filter, category=category)
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    headers = [
        "DPP ID", "GTIN", "Nome", "Categoria", "Fabricante", "País",
        "Carbono (kg CO2e)", "Água (m³)", "Trust Index", "Reciclabilidade (%)",
        "Conteúdo Reciclado (%)", "Durabilidade", "Reparabilidade",
        "ESPR", "CBAM", "CSRD", "Status", "Criado"
    ]
    writer.writerow(headers)
    
    # Data rows
    for d in dpps:
        compliance = d.compliance or {}
        writer.writerow([
            d.dpp_id,
            d.gtin,
            d.product_name,
            d.product_category or "",
            d.manufacturer_name or "",
            d.country_of_origin or "",
            d.carbon_footprint_kg_co2eq or 0,
            d.water_consumption_m3 or 0,
            d.trust_index or 0,
            d.recyclability_percent or 0,
            d.recycled_content_percent or 0,
            d.durability_score or 0,
            d.reparability_score or 0,
            "Sim" if compliance.get("espr_compliant") else "Não",
            "Sim" if compliance.get("cbam_compliant") else "Não",
            "Sim" if compliance.get("csrd_compliant") else "Não",
            d.status,
            d.created_at.strftime("%Y-%m-%d") if d.created_at else "",
        ])
    
    output.seek(0)
    
    return StreamingResponse(
        iter(["\ufeff" + output.getvalue()]),  # BOM for Excel UTF-8
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=Duplios-Export-{datetime.now().strftime('%Y%m%d')}.csv"
        }
    )


@router.get("/export/json")
def api_export_json(
    status_filter: Optional[str] = None,
    category: Optional[str] = None,
    db=Depends(get_db)
):
    """Export DPPs to JSON format."""
    dpps = list_dpps(db, status_filter=status_filter, category=category)
    
    # Convert to serializable format
    data = []
    for d in dpps:
        dpp_dict = d.dict() if hasattr(d, 'dict') else d.__dict__
        # Convert datetime objects to strings
        for key, value in dpp_dict.items():
            if isinstance(value, datetime):
                dpp_dict[key] = value.isoformat()
        data.append(dpp_dict)
    
    return {
        "export_date": datetime.now().isoformat(),
        "total_count": len(data),
        "dpps": data,
    }


# =====================
# PDM LITE ENDPOINTS
# =====================

from duplios.pdm_service import (
    create_item_with_initial_revision,
    get_item, get_item_by_sku, list_items,
    get_revisions_for_item, create_new_revision_from_previous,
    release_revision, get_bom, get_routing,
    get_active_revision, add_bom_line, add_routing_operation
)
from duplios.lca_engine import compute_simple_lca, update_dpp_with_lca, get_available_materials
from duplios.identity_service import (
    ingest_identity, verify_identity, get_identities_for_revision
)

from pydantic import BaseModel
from typing import Optional as Opt


class ItemCreateRequest(BaseModel):
    sku: str
    name: str
    type: str = "FINISHED"
    unit: str = "pcs"
    family: Opt[str] = None
    weight_kg: Opt[float] = None


class BomLineRequest(BaseModel):
    component_revision_id: int
    qty_per_unit: float
    scrap_rate: float = 0.0
    position: Opt[str] = None
    notes: Opt[str] = None


class RoutingOpRequest(BaseModel):
    op_code: str
    sequence: int
    machine_group: Opt[str] = None
    nominal_setup_time: float = 0.0
    nominal_cycle_time: float = 0.0
    tool_id: Opt[str] = None
    description: Opt[str] = None


class IdentityIngestRequest(BaseModel):
    revision_id: int
    unique_item_id: str


class IdentityVerifyRequest(BaseModel):
    unique_item_id: str


@router.get("/items")
def api_list_items(
    item_type: Opt[str] = None,
    family: Opt[str] = None,
    limit: int = 100,
    offset: int = 0,
    db=Depends(get_db)
):
    """List items with active revisions."""
    return list_items(item_type=item_type, family=family, limit=limit, offset=offset, db=db)


@router.post("/items")
def api_create_item(payload: ItemCreateRequest, db=Depends(get_db)):
    """Create a new item with initial DRAFT revision."""
    try:
        return create_item_with_initial_revision(
            sku=payload.sku,
            name=payload.name,
            item_type=payload.type,
            unit=payload.unit,
            family=payload.family,
            weight_kg=payload.weight_kg,
            db=db,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/items/{item_id}")
def api_get_item(item_id: int, db=Depends(get_db)):
    """Get item by ID."""
    result = get_item(item_id, db)
    if not result:
        raise HTTPException(status_code=404, detail="Item not found")
    return result


@router.get("/items/by-sku/{sku}")
def api_get_item_by_sku(sku: str, db=Depends(get_db)):
    """Get item by SKU."""
    result = get_item_by_sku(sku, db)
    if not result:
        raise HTTPException(status_code=404, detail="Item not found")
    return result


@router.get("/items/{item_id}/revisions")
def api_get_revisions(item_id: int, db=Depends(get_db)):
    """Get all revisions for an item."""
    return get_revisions_for_item(item_id, db)


@router.post("/items/{item_id}/revisions")
def api_create_revision(
    item_id: int,
    base_revision_id: Opt[int] = None,
    db=Depends(get_db)
):
    """Create new revision based on active or specified revision."""
    try:
        return create_new_revision_from_previous(item_id, base_revision_id, db=db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/revisions/{revision_id}/release")
def api_release_revision(revision_id: int, db=Depends(get_db)):
    """Release a DRAFT revision to RELEASED status."""
    try:
        return release_revision(revision_id, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/revisions/{revision_id}/bom")
def api_get_bom(revision_id: int, db=Depends(get_db)):
    """Get BOM for a revision."""
    return get_bom(revision_id, db)


@router.post("/revisions/{revision_id}/bom")
def api_add_bom_line(revision_id: int, payload: BomLineRequest, db=Depends(get_db)):
    """Add a BOM line to a revision."""
    return add_bom_line(
        parent_revision_id=revision_id,
        component_revision_id=payload.component_revision_id,
        qty_per_unit=payload.qty_per_unit,
        scrap_rate=payload.scrap_rate,
        position=payload.position,
        notes=payload.notes,
        db=db,
    )


@router.get("/revisions/{revision_id}/routing")
def api_get_routing(revision_id: int, db=Depends(get_db)):
    """Get routing for a revision."""
    return get_routing(revision_id, db)


@router.post("/revisions/{revision_id}/routing")
def api_add_routing_op(revision_id: int, payload: RoutingOpRequest, db=Depends(get_db)):
    """Add a routing operation to a revision."""
    return add_routing_operation(
        revision_id=revision_id,
        op_code=payload.op_code,
        sequence=payload.sequence,
        machine_group=payload.machine_group,
        nominal_setup_time=payload.nominal_setup_time,
        nominal_cycle_time=payload.nominal_cycle_time,
        tool_id=payload.tool_id,
        description=payload.description,
        db=db,
    )


# =====================
# LCA ENDPOINTS
# =====================

@router.post("/revisions/{revision_id}/lca/recalculate")
def api_recalculate_lca(revision_id: int, db=Depends(get_db)):
    """Recalculate LCA for a revision and update DPP."""
    lca_result = compute_simple_lca(revision_id, db)
    dpp = update_dpp_with_lca(revision_id, lca_result, db)
    
    return {
        "lca": lca_result.to_dict(),
        "dpp_updated": dpp is not None,
        "dpp_trust_index": dpp.trust_index if dpp else None,
        "dpp_data_completeness": dpp.data_completeness_pct if dpp else None,
    }


@router.get("/revisions/{revision_id}/lca")
def api_get_lca(revision_id: int, db=Depends(get_db)):
    """Get LCA calculation for a revision."""
    lca_result = compute_simple_lca(revision_id, db)
    return lca_result.to_dict()


@router.get("/materials")
def api_get_materials():
    """Get available materials for LCA calculations."""
    return get_available_materials()


# =====================
# DPP (linked to revision) ENDPOINTS
# =====================

@router.get("/dpp-revision/{revision_id}")
def api_get_dpp_by_revision(revision_id: int, db=Depends(get_db)):
    """Get DPP record for a specific revision."""
    from duplios.dpp_models import DppRecord
    
    dpp = db.query(DppRecord).filter(
        DppRecord.item_revision_id == revision_id
    ).first()
    
    if not dpp:
        raise HTTPException(status_code=404, detail="DPP not found for this revision")
    
    return {
        "id": dpp.id,
        "revision_id": dpp.item_revision_id,
        "gtin": dpp.gtin,
        "product_name": dpp.product_name,
        "product_category": dpp.product_category,
        "manufacturer_name": dpp.manufacturer_name,
        "country_of_origin": dpp.country_of_origin,
        "carbon_kg_co2eq": dpp.carbon_kg_co2eq,
        "water_m3": dpp.water_m3,
        "energy_kwh": dpp.energy_kwh,
        "recycled_content_pct": dpp.recycled_content_pct,
        "recyclability_pct": dpp.recyclability_pct,
        "durability_score": dpp.durability_score,
        "reparability_score": dpp.reparability_score,
        "trust_index": dpp.trust_index,
        "data_completeness_pct": dpp.data_completeness_pct,
        "status": dpp.status.value,
        "qr_public_url": dpp.qr_public_url,
    }


# =====================
# DIGITAL IDENTITY ENDPOINTS
# =====================

@router.post("/identity/ingest")
def api_ingest_identity(payload: IdentityIngestRequest, db=Depends(get_db)):
    """Ingest a new digital identity (RFID/QR/serial)."""
    identity = ingest_identity(
        item_revision_id=payload.revision_id,
        unique_item_id=payload.unique_item_id,
        db=db,
    )
    return {
        "id": identity.id,
        "revision_id": identity.item_revision_id,
        "unique_item_id": identity.unique_item_id,
        "identity_hash": identity.identity_hash[:16] + "...",
        "status": identity.verification_status.value,
        "created_at": identity.created_at.isoformat() if identity.created_at else None,
    }


@router.post("/identity/verify")
def api_verify_identity(payload: IdentityVerifyRequest, db=Depends(get_db)):
    """Verify a digital identity."""
    return verify_identity(payload.unique_item_id, db)


@router.get("/revisions/{revision_id}/identities")
def api_get_identities(revision_id: int, db=Depends(get_db)):
    """Get all digital identities for a revision."""
    identities = get_identities_for_revision(revision_id, db)
    return [
        {
            "id": i.id,
            "unique_item_id": i.unique_item_id,
            "status": i.verification_status.value,
            "verification_count": i.verification_count or 0,
            "last_verified_at": i.last_verified_at.isoformat() if i.last_verified_at else None,
            "created_at": i.created_at.isoformat() if i.created_at else None,
        }
        for i in identities
    ]
