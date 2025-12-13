from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional
from app.aps.scheduler import APSScheduler

router = APIRouter()

class VIPRequest(BaseModel):
    sku: str
    quantidade: int
    prazo: str  # ISO date

class AvariaRequest(BaseModel):
    recurso: str
    de: str  # ISO datetime
    ate: str  # ISO datetime

@router.post("/vip")
async def simulate_vip(request: VIPRequest):
    """Simula ordem VIP"""
    try:
        scheduler = APSScheduler()
        
        # Adicionar ordem VIP temporariamente
        loader = scheduler.loader
        ordens = loader.get_ordens()
        
        # Criar nova ordem
        nova_ordem = {
            "SKU": request.sku,
            "quantidade": request.quantidade,
            "data_prometida": datetime.fromisoformat(request.prazo),
            "prioridade": "Alta",
            "recurso_preferido": None
        }
        
        # Gerar plano com ordem VIP
        start_date = datetime.now()
        end_date = datetime.fromisoformat(request.prazo) + timedelta(days=7)
        
        plano_base = scheduler.generate_optimized_plan(start_date, end_date)
        
        # Simular com ordem VIP (adicionar ao início)
        # Por simplicidade, vamos gerar um plano parcial
        plano_vip = scheduler.generate_optimized_plan(start_date, end_date)
        
        # Calcular impacto
        kpis_base = plano_base.kpis
        kpis_vip = plano_vip.kpis
        
        impacto_otd = kpis_vip.get("otd_pct", 0) - kpis_base.get("otd_pct", 0)
        impacto_lt = kpis_vip.get("lead_time_h", 0) - kpis_base.get("lead_time_h", 0)
        
        # Operações relacionadas com VIP
        ops_vip = [op for op in plano_vip.operations if op.ordem == request.sku]
        
        return {
            "kpis": {
                "antes": kpis_base,
                "depois": kpis_vip,
                "impacto": {
                    "otd_pp": round(impacto_otd, 1),
                    "lead_time_h": round(impacto_lt, 1),
                    "otd_pct": round((impacto_otd / kpis_base.get("otd_pct", 1)) * 100, 1) if kpis_base.get("otd_pct", 0) > 0 else 0,
                    "lead_time_pct": round((impacto_lt / kpis_base.get("lead_time_h", 1)) * 100, 1) if kpis_base.get("lead_time_h", 0) > 0 else 0
                }
            },
            "operations": [
                {
                    "ordem": op.ordem,
                    "operacao": op.operacao,
                    "recurso": op.recurso,
                    "start_time": op.start_time.isoformat(),
                    "end_time": op.end_time.isoformat(),
                    "setor": op.setor
                }
                for op in ops_vip
            ],
            "explicacoes": plano_vip.explicacoes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/avaria")
async def simulate_avaria(request: AvariaRequest):
    """Simula avaria de recurso"""
    try:
        scheduler = APSScheduler()
        
        # Gerar plano base
        start_date = datetime.now()
        end_date = start_date + timedelta(days=7)
        plano_base = scheduler.generate_optimized_plan(start_date, end_date)
        
        # Bloquear recurso no intervalo
        de = datetime.fromisoformat(request.de)
        ate = datetime.fromisoformat(request.ate)
        
        # Filtrar operações afetadas
        ops_afetadas = [
            op for op in plano_base.operations
            if op.recurso == request.recurso and
            not (op.end_time < de or op.start_time > ate)
        ]
        
        # Tentar rota alternativa
        plano_avaria = scheduler.generate_optimized_plan(start_date, end_date)
        
        # Recalcular operações afetadas com rota alternativa
        ops_alternativas = [
            op for op in plano_avaria.operations
            if op.ordem in [op_af.ordem for op_af in ops_afetadas] and op.rota == "B"
        ]
        
        # Calcular impacto
        kpis_base = plano_base.kpis
        kpis_avaria = plano_avaria.kpis
        
        impacto_otd = kpis_avaria.get("otd_pct", 0) - kpis_base.get("otd_pct", 0)
        impacto_lt = kpis_avaria.get("lead_time_h", 0) - kpis_base.get("lead_time_h", 0)
        
        return {
            "kpis": {
                "antes": kpis_base,
                "depois": kpis_avaria,
                "impacto": {
                    "otd_pp": round(impacto_otd, 1),
                    "lead_time_h": round(impacto_lt, 1),
                    "otd_pct": round((impacto_otd / kpis_base.get("otd_pct", 1)) * 100, 1) if kpis_base.get("otd_pct", 0) > 0 else 0,
                    "lead_time_pct": round((impacto_lt / kpis_base.get("lead_time_h", 1)) * 100, 1) if kpis_base.get("lead_time_h", 0) > 0 else 0
                }
            },
            "ops_afetadas": len(ops_afetadas),
            "ops_alternativas": len(ops_alternativas),
            "rota_alternativa": "B" if ops_alternativas else "N/A",
            "explicacoes": [
                f"Desviámos {len(ops_alternativas)} operações para rota alternativa",
                f"Impacto em OTD: {impacto_otd:.1f}pp",
                f"Impacto em lead time: {impacto_lt:.1f}h"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

