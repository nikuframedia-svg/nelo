"""
════════════════════════════════════════════════════════════════════════════════
OPS INGESTION API - REST Endpoints para Importação de Dados Operacionais
════════════════════════════════════════════════════════════════════════════════

Contract 14: API endpoints para importação de Excel

Endpoints:
- POST /ops-ingestion/orders/excel
- POST /ops-ingestion/inventory-moves/excel
- POST /ops-ingestion/hr/excel
- POST /ops-ingestion/machines/excel
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from ops_ingestion.services import get_ops_ingestion_service, build_planning_instance_from_raw
from ops_ingestion.schemas import ImportResult

# Import get_db (standard dependency)
try:
    from duplios.service import get_db
except ImportError:
    try:
        from duplios.models import SessionLocal
        def get_db():
            db = SessionLocal()
            try:
                yield db
            finally:
                db.close()
    except ImportError:
        # Fallback: create minimal get_db
        from sqlalchemy.orm import Session
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        import os
        DATABASE_URL = os.getenv("DUPLIOS_DATABASE_URL", "sqlite:///duplios.db")
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        def get_db():
            db = SessionLocal()
            try:
                yield db
            finally:
                db.close()

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ops-ingestion", tags=["Operational Data Ingestion"])


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/orders/excel", response_model=ImportResult)
async def import_orders_excel(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Importa ordens de produção do Excel.
    
    As specified in Contract 14:
    - Recebe ficheiro Excel
    - Valida e grava em ops_raw_orders
    - Executa data quality checks
    - Retorna resultado com contagens e warnings/erros
    """
    if not file.filename or not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Ficheiro deve ser Excel (.xlsx ou .xls)")
    
    service = get_ops_ingestion_service()
    result = service.import_orders_from_excel(file, db)
    
    # Log to R&D
    _log_to_rd("WPX_DATA_INGESTION", {
        "type": "orders",
        "imported_count": result.imported_count,
        "failed_count": result.failed_count,
        "warnings_count": len(result.warnings),
        "errors_count": len(result.errors),
        "source_file": result.source_file,
    })
    
    return result


@router.post("/inventory-moves/excel", response_model=ImportResult)
async def import_inventory_moves_excel(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Importa movimentos de inventário do Excel.
    
    As specified in Contract 14:
    - Recebe ficheiro Excel
    - Valida e grava em ops_raw_inventory_moves
    - Executa data quality checks
    - Retorna resultado com contagens e warnings/erros
    """
    if not file.filename or not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Ficheiro deve ser Excel (.xlsx ou .xls)")
    
    service = get_ops_ingestion_service()
    result = service.import_inventory_moves_from_excel(file, db)
    
    # Log to R&D
    _log_to_rd("WPX_DATA_INGESTION", {
        "type": "inventory_moves",
        "imported_count": result.imported_count,
        "failed_count": result.failed_count,
        "warnings_count": len(result.warnings),
        "errors_count": len(result.errors),
        "source_file": result.source_file,
    })
    
    return result


@router.post("/hr/excel", response_model=ImportResult)
async def import_hr_excel(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Importa dados de RH do Excel.
    
    As specified in Contract 14:
    - Recebe ficheiro Excel
    - Valida e grava em ops_raw_hr
    - Executa data quality checks
    - Retorna resultado com contagens e warnings/erros
    """
    if not file.filename or not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Ficheiro deve ser Excel (.xlsx ou .xls)")
    
    service = get_ops_ingestion_service()
    result = service.import_hr_from_excel(file, db)
    
    # Log to R&D
    _log_to_rd("WPX_DATA_INGESTION", {
        "type": "hr",
        "imported_count": result.imported_count,
        "failed_count": result.failed_count,
        "warnings_count": len(result.warnings),
        "errors_count": len(result.errors),
        "source_file": result.source_file,
    })
    
    return result


@router.post("/machines/excel", response_model=ImportResult)
async def import_machines_excel(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Importa dados de máquinas do Excel.
    
    As specified in Contract 14:
    - Recebe ficheiro Excel
    - Valida e grava em ops_raw_machines
    - Executa data quality checks
    - Retorna resultado com contagens e warnings/erros
    """
    if not file.filename or not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Ficheiro deve ser Excel (.xlsx ou .xls)")
    
    service = get_ops_ingestion_service()
    result = service.import_machines_from_excel(file, db)
    
    # Log to R&D
    _log_to_rd("WPX_DATA_INGESTION", {
        "type": "machines",
        "imported_count": result.imported_count,
        "failed_count": result.failed_count,
        "warnings_count": len(result.warnings),
        "errors_count": len(result.errors),
        "source_file": result.source_file,
    })
    
    return result


@router.get("/planning-instance")
async def get_planning_instance_from_raw(
    horizon_days: int = 30,
    db: Session = Depends(get_db),
):
    """
    Constrói SchedulingInstance a partir de dados raw.
    
    As specified in Contract 14:
    - Cria jobs a partir de ops_raw_orders
    - Cria operations a partir de routing_json
    - Cria machines a partir de ops_raw_machines
    
    Returns:
        dict com estrutura compatível com SchedulingInstance
    """
    instance = build_planning_instance_from_raw(db, horizon_days=horizon_days)
    return instance


@router.get("/orders")
async def get_imported_orders(
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """
    Lista ordens importadas do Excel.
    
    Returns:
        Lista de ordens com informações básicas
    """
    from ops_ingestion.models import OpsRawOrder
    
    orders = db.query(OpsRawOrder).order_by(OpsRawOrder.imported_at.desc()).limit(limit).all()
    
    return {
        "orders": [
            {
                "id": o.id,
                "external_order_code": o.external_order_code,
                "product_code": o.product_code,
                "quantity": o.quantity,
                "due_date": o.due_date.isoformat() if o.due_date else None,
                "line_or_center": o.line_or_center,
                "imported_at": o.imported_at.isoformat() if o.imported_at else None,
                "source_file": o.source_file,
            }
            for o in orders
        ],
        "total": len(orders),
    }


@router.get("/wip-flow/{order_code}")
async def get_wip_flow_for_order(
    order_code: str,
    db: Session = Depends(get_db),
):
    """
    Obtém posição WIP atual para uma ordem específica.
    
    As specified in Contract 14 FASE 3.2:
    - Reconstrói estado WIP por ordem e por estação
    - Determina fase atual (último to_station)
    - Soma qty_good/qty_scrap acumuladas
    
    Returns:
        {
            "order_code": str,
            "current_station": str | null,
            "total_good": float,
            "total_scrap": float,
            "completion_percent": float,
            "movements": [...]
        }
    """
    from ops_ingestion.models import OpsRawInventoryMove, OpsRawOrder
    from datetime import datetime
    
    # Verificar se ordem existe
    order = db.query(OpsRawOrder).filter(
        OpsRawOrder.external_order_code == order_code
    ).first()
    
    if not order:
        raise HTTPException(status_code=404, detail=f"Ordem {order_code} não encontrada")
    
    # Buscar movimentos da ordem, ordenados por timestamp
    moves = db.query(OpsRawInventoryMove).filter(
        OpsRawInventoryMove.order_code == order_code
    ).order_by(OpsRawInventoryMove.timestamp.asc()).all()
    
    # Calcular estado WIP
    total_good = sum(m.quantity_good or 0 for m in moves)
    total_scrap = sum(m.quantity_scrap or 0 for m in moves)
    
    # Última estação (último to_station não nulo)
    current_station = None
    for move in reversed(moves):
        if move.to_station:
            current_station = move.to_station
            break
    
    # Percentagem de conclusão (baseado na quantidade total da ordem)
    completion_percent = 0.0
    if order.quantity > 0:
        completion_percent = min(100.0, ((total_good + total_scrap) / order.quantity) * 100.0)
    
    return {
        "order_code": order_code,
        "product_code": order.product_code,
        "order_quantity": order.quantity,
        "current_station": current_station,
        "total_good": total_good,
        "total_scrap": total_scrap,
        "completion_percent": completion_percent,
        "movements": [
            {
                "id": m.id,
                "from_station": m.from_station,
                "to_station": m.to_station,
                "movement_type": m.movement_type.value if hasattr(m.movement_type, 'value') else str(m.movement_type),
                "quantity_good": m.quantity_good,
                "quantity_scrap": m.quantity_scrap,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
            }
            for m in moves
        ],
        "movements_count": len(moves),
    }


@router.get("/wip-flow")
async def get_all_wip_flows(
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """
    Lista todas as ordens com posição WIP atual.
    
    Returns:
        Lista de ordens com resumo WIP
    """
    from ops_ingestion.models import OpsRawOrder, OpsRawInventoryMove
    from collections import defaultdict
    
    # Buscar ordens recentes
    orders = db.query(OpsRawOrder).order_by(
        OpsRawOrder.imported_at.desc()
    ).limit(limit).all()
    
    wip_flows = []
    
    for order in orders:
        # Buscar movimentos da ordem
        moves = db.query(OpsRawInventoryMove).filter(
            OpsRawInventoryMove.order_code == order.external_order_code
        ).order_by(OpsRawInventoryMove.timestamp.asc()).all()
        
        total_good = sum(m.quantity_good or 0 for m in moves)
        total_scrap = sum(m.quantity_scrap or 0 for m in moves)
        
        # Última estação
        current_station = None
        for move in reversed(moves):
            if move.to_station:
                current_station = move.to_station
                break
        
        completion_percent = 0.0
        if order.quantity > 0:
            completion_percent = min(100.0, ((total_good + total_scrap) / order.quantity) * 100.0)
        
        wip_flows.append({
            "order_code": order.external_order_code,
            "product_code": order.product_code,
            "order_quantity": order.quantity,
            "current_station": current_station,
            "total_good": total_good,
            "total_scrap": total_scrap,
            "completion_percent": completion_percent,
            "movements_count": len(moves),
        })
    
    return {
        "wip_flows": wip_flows,
        "total": len(wip_flows),
    }


@router.get("/stats")
async def get_import_stats(
    db: Session = Depends(get_db),
):
    """
    Estatísticas dos dados importados.
    
    Returns:
        Contagens e resumos dos dados importados
    """
    from ops_ingestion.models import (
        OpsRawOrder,
        OpsRawInventoryMove,
        OpsRawHR,
        OpsRawMachine,
    )
    
    orders_count = db.query(OpsRawOrder).count()
    moves_count = db.query(OpsRawInventoryMove).count()
    hr_count = db.query(OpsRawHR).count()
    machines_count = db.query(OpsRawMachine).count()
    
    # Última importação
    last_order = db.query(OpsRawOrder).order_by(OpsRawOrder.imported_at.desc()).first()
    last_move = db.query(OpsRawInventoryMove).order_by(OpsRawInventoryMove.imported_at.desc()).first()
    last_hr = db.query(OpsRawHR).order_by(OpsRawHR.imported_at.desc()).first()
    last_machine = db.query(OpsRawMachine).order_by(OpsRawMachine.imported_at.desc()).first()
    
    return {
        "orders": {
            "total": orders_count,
            "last_import": last_order.imported_at.isoformat() if last_order and last_order.imported_at else None,
            "last_file": last_order.source_file if last_order else None,
        },
        "inventory_moves": {
            "total": moves_count,
            "last_import": last_move.imported_at.isoformat() if last_move and last_move.imported_at else None,
            "last_file": last_move.source_file if last_move else None,
        },
        "hr": {
            "total": hr_count,
            "last_import": last_hr.imported_at.isoformat() if last_hr and last_hr.imported_at else None,
            "last_file": last_hr.source_file if last_hr else None,
        },
        "machines": {
            "total": machines_count,
            "last_import": last_machine.imported_at.isoformat() if last_machine and last_machine.imported_at else None,
            "last_file": last_machine.source_file if last_machine else None,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _log_to_rd(experiment_type: str, event_data: dict) -> None:
    """Log import event to R&D module."""
    try:
        from rd.experiments_core import log_experiment_event
        log_experiment_event(
            experiment_type=experiment_type,
            event_data=event_data,
        )
        logger.info(f"Logged {experiment_type} event to R&D")
    except ImportError:
        logger.debug("R&D module not available, skipping import logging")
    except Exception as e:
        logger.warning(f"Failed to log to R&D: {e}")

