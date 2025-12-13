"""
ProdPlan 4.0 - MRP API
======================

Endpoints REST para Material Requirements Planning.

Endpoints:
- POST /inventory/mrp/run-from-orders    - Executar MRP a partir de encomendas
- GET  /inventory/mrp/parameters         - Lista parâmetros MRP
- GET  /inventory/mrp/orders-status      - Status de materiais por encomenda
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inventory/mrp", tags=["MRP"])


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class DateRange(BaseModel):
    """Período de datas."""
    start: datetime
    end: datetime


class MRPRunRequest(BaseModel):
    """Request para executar MRP."""
    order_ids: List[str] = Field(description="IDs das encomendas a processar")
    horizon: Optional[DateRange] = Field(default=None, description="Horizonte de planeamento")


class ComponentStatus(BaseModel):
    """Status de um componente."""
    component_id: str
    qty_required: float
    qty_available: float
    shortage: float
    status: str  # "OK", "SHORTAGE", "PARTIAL"


class OrderMaterialStatus(BaseModel):
    """Status de materiais de uma encomenda."""
    order_id: str
    product_id: str
    quantity: float
    due_date: datetime
    material_status: str  # "OK", "SHORTAGE", "PARTIAL"
    components: List[ComponentStatus]


class PurchaseSuggestionResponse(BaseModel):
    """Sugestão de compra."""
    component_id: str
    quantity: float
    due_date: datetime
    source_orders: List[str]
    lead_time_days: float


class InternalOrderSuggestionResponse(BaseModel):
    """Sugestão de ordem interna."""
    item_id: str
    quantity: float
    due_date: datetime
    source_orders: List[str]


class MRPRunResponse(BaseModel):
    """Response do MRP."""
    success: bool
    orders_processed: int
    components_analyzed: int
    purchase_suggestions: List[PurchaseSuggestionResponse]
    internal_order_suggestions: List[InternalOrderSuggestionResponse]
    shortages: List[Dict[str, Any]]
    warnings: List[str]


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/run-from-orders", response_model=MRPRunResponse)
async def run_mrp_from_orders(request: MRPRunRequest):
    """
    Executa MRP a partir de uma lista de encomendas.
    
    Processo:
    1. Explode BOM de cada encomenda
    2. Consulta stock disponível
    3. Aplica parâmetros MRP (MOQ, múltiplos, scrap)
    4. Gera sugestões de compra e produção interna
    
    **Nota:** Por agora usa dados de demonstração.
    """
    try:
        from .mrp_engine import run_mrp_from_orders
        from data_loader import load_dataset
        
        # Carregar encomendas
        data = load_dataset()
        orders_df = data.orders
        
        # Filtrar encomendas solicitadas
        if request.order_ids:
            orders_df = orders_df[orders_df["order_id"].isin(request.order_ids)]
        
        if orders_df.empty:
            return MRPRunResponse(
                success=True,
                orders_processed=0,
                components_analyzed=0,
                purchase_suggestions=[],
                internal_order_suggestions=[],
                shortages=[],
                warnings=["No orders found with provided IDs"],
            )
        
        # Converter para formato MRP
        orders = []
        for _, row in orders_df.iterrows():
            orders.append({
                "order_id": str(row.get("order_id", "")),
                "product_id": str(row.get("article_id", row.get("product_id", ""))),
                "quantity": float(row.get("qty", row.get("quantity", 1))),
                "due_date": row.get("due_date", datetime.now() + timedelta(days=7)),
            })
        
        # Executar MRP
        result = run_mrp_from_orders(orders)
        
        # Converter resultado
        purchase_suggestions = [
            PurchaseSuggestionResponse(
                component_id=s.component_id,
                quantity=s.quantity,
                due_date=s.due_date,
                source_orders=s.source_order_ids,
                lead_time_days=s.lead_time_days,
            )
            for s in result.purchase_suggestions
        ]
        
        internal_suggestions = [
            InternalOrderSuggestionResponse(
                item_id=s.item_id,
                quantity=s.quantity,
                due_date=s.due_date,
                source_orders=s.source_order_ids,
            )
            for s in result.internal_order_suggestions
        ]
        
        return MRPRunResponse(
            success=True,
            orders_processed=result.orders_processed,
            components_analyzed=result.components_analyzed,
            purchase_suggestions=purchase_suggestions,
            internal_order_suggestions=internal_suggestions,
            shortages=result.shortages,
            warnings=result.warnings,
        )
        
    except Exception as e:
        logger.error(f"MRP run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orders-status")
async def get_orders_material_status(
    limit: int = Query(default=50, description="Limite de encomendas"),
):
    """
    Obtém status de materiais para encomendas abertas.
    
    Status possíveis:
    - OK: Todos os materiais disponíveis
    - PARTIAL: Alguns materiais em falta
    - SHORTAGE: Materiais críticos em falta
    """
    try:
        from .mrp_engine import run_mrp_from_orders
        from data_loader import load_dataset
        
        # Carregar encomendas
        data = load_dataset()
        orders_df = data.orders.head(limit)
        
        # Converter para formato MRP
        orders = []
        for _, row in orders_df.iterrows():
            orders.append({
                "order_id": str(row.get("order_id", "")),
                "product_id": str(row.get("article_id", row.get("product_id", ""))),
                "quantity": float(row.get("qty", row.get("quantity", 1))),
                "due_date": row.get("due_date", datetime.now() + timedelta(days=7)),
            })
        
        if not orders:
            return {"orders": [], "total": 0}
        
        # Executar MRP
        result = run_mrp_from_orders(orders)
        
        # Criar mapa de shortages por ordem
        shortage_by_order: Dict[str, List[Dict]] = {}
        for shortage in result.shortages:
            # Este é simplificado - numa implementação real teríamos rastreamento completo
            pass
        
        # Construir resposta simplificada
        order_statuses = []
        for o in orders:
            # Status simplificado baseado em resultado global
            has_shortages = len(result.shortages) > 0
            
            order_statuses.append({
                "order_id": o["order_id"],
                "product_id": o["product_id"],
                "quantity": o["quantity"],
                "due_date": o["due_date"].isoformat() if hasattr(o["due_date"], 'isoformat') else str(o["due_date"]),
                "material_status": "PARTIAL" if has_shortages else "OK",
                "purchase_suggestions": len([s for s in result.purchase_suggestions if o["order_id"] in s.source_order_ids]),
            })
        
        return {
            "orders": order_statuses,
            "total": len(order_statuses),
            "with_shortages": len([o for o in order_statuses if o["material_status"] != "OK"]),
            "summary": {
                "total_purchase_suggestions": len(result.purchase_suggestions),
                "total_internal_suggestions": len(result.internal_order_suggestions),
                "total_shortages": len(result.shortages),
            },
        }
        
    except Exception as e:
        logger.error(f"Get orders status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/parameters")
async def get_mrp_parameters():
    """
    Obtém parâmetros MRP configurados.
    
    Parâmetros por SKU:
    - min_stock: Stock mínimo
    - max_stock: Stock máximo
    - reorder_min_qty: MOQ (Minimum Order Quantity)
    - reorder_multiple: Múltiplo de encomenda
    - scrap_rate: Taxa de refugo
    - lead_time_days: Lead time
    """
    # Por agora retorna parâmetros de demonstração
    return {
        "parameters": [
            {
                "sku_id": "RM-001",
                "name": "Raw Material A",
                "min_stock": 100,
                "max_stock": 1000,
                "reorder_min_qty": 50,
                "reorder_multiple": 10,
                "scrap_rate": 0.02,
                "lead_time_days": 5,
            },
            {
                "sku_id": "RM-002",
                "name": "Raw Material B",
                "min_stock": 50,
                "max_stock": 500,
                "reorder_min_qty": 25,
                "reorder_multiple": 5,
                "scrap_rate": 0.01,
                "lead_time_days": 3,
            },
            {
                "sku_id": "SF-001",
                "name": "Sub-Assembly 1",
                "min_stock": 20,
                "max_stock": 200,
                "reorder_min_qty": 10,
                "reorder_multiple": 1,
                "scrap_rate": 0.03,
                "lead_time_days": 2,
            },
        ],
        "note": "Parameters are demo data. Configure in ERP/MES for production use.",
    }


@router.get("/bom/{product_id}")
async def get_product_bom(product_id: str, quantity: float = Query(default=1.0)):
    """
    Obtém BOM explodida de um produto.
    
    Args:
        product_id: ID do produto
        quantity: Quantidade a produzir
    """
    try:
        from .bom_engine import explode_bom, create_sample_bom
        
        # Usar BOM de demonstração
        requirements = explode_bom(product_id, qty=quantity)
        
        return {
            "product_id": product_id,
            "quantity": quantity,
            "components": [
                {
                    "component_id": r.component_id,
                    "qty_required": round(r.qty_required, 3),
                    "is_purchased": r.is_purchased,
                    "is_manufactured": r.is_manufactured,
                    "lead_time_days": r.lead_time_days,
                    "level": r.level,
                }
                for r in requirements
            ],
            "total_components": len(requirements),
        }
        
    except Exception as e:
        logger.error(f"BOM explosion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


