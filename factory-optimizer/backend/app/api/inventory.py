from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime, timedelta
from app.etl.loader import get_loader
from app.ml.inventory import InventoryPredictor

router = APIRouter()

EMPTY_MATRIX = {
    "A": {"X": 0, "Y": 0, "Z": 0},
    "B": {"X": 0, "Y": 0, "Z": 0},
    "C": {"X": 0, "Y": 0, "Z": 0},
}

@router.get("/")
async def get_inventory(
    classe: Optional[str] = Query(None, description="Filtrar por classe ABC/XYZ"),
    search: Optional[str] = Query(None, description="Pesquisar SKU"),
    recalculate_rop: bool = Query(False, description="Recalcular ROP completo (Monte Carlo) - mais lento mas mais preciso")
):
    """Retorna análise de inventário"""
    try:
        loader = get_loader()
        # Recalcular ROP completo se solicitado (quando user acede à página)
        insights = loader.get_inventory_insights(recalculate_rop=recalculate_rop)

        matrix = insights.get("matrix", EMPTY_MATRIX)
        skus = insights.get("skus", [])

        filtered = []
        for sku_entry in skus:
            if search and search.lower() not in sku_entry["sku"].lower():
                continue
            if classe and sku_entry.get("classe") != classe:
                continue
            filtered.append(sku_entry)

        return {
            "matrix": matrix,
            "skus": filtered,
            "kpis": insights.get("kpis", {}),
            "top_risks": insights.get("top_risks", []),
            "generated_at": insights.get("generated_at"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rop")
async def recalculate_rop(
    sku: str = Query(..., description="SKU"),
    service_level: float = Query(0.95, description="Nível de serviço"),
    lead_time: float = Query(7.0, description="Lead time em dias")
):
    """Recalcula ROP para um SKU"""
    try:
        predictor = InventoryPredictor()
        loader = get_loader()
        stocks_mov = loader.get_stocks_mov()
        
        # Calcular ADS
        if not stocks_mov.empty and sku in stocks_mov["SKU"].values:
            sku_mov_180d = stocks_mov[
                (stocks_mov["SKU"] == sku) & 
                (stocks_mov["data"] >= datetime.now() - timedelta(days=180))
            ]["saidas"].sum()
            ads_180 = sku_mov_180d / 180 if sku_mov_180d > 0 else 0.1
        else:
            ads_180 = 10.0
        
        predictor.update_demand(sku, ads_180)
        rop_result = predictor.calculate_rop(sku, lead_time=lead_time, service_level=service_level)
        
        return {
            "sku": sku,
            "rop": rop_result["rop"],
            "stockout_prob": rop_result["stockout_prob"],
            "coverage_days": rop_result["coverage_days"],
            "mu": rop_result["mu"],
            "sigma": rop_result["sigma"],
            "service_level": service_level,
            "lead_time": lead_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

