import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

from app.aps.scheduler import APSScheduler
from app.ml.bottlenecks import BottleneckPredictor

router = APIRouter()


logger = logging.getLogger(__name__)

# Recursos para modo DEMO
DEMO_RESOURCES = ["27", "29", "248"]
DEMO_UTILIZATION = 0.92  # 92%
DEMO_QUEUE_HOURS = 40.0  # 40h de fila
DEMO_PROB_GARGALO = 0.95  # 95%


def apply_demo_overrides_to_bottlenecks(
    bottlenecks: List[Dict[str, Any]], 
    resources: List[Dict[str, Any]],
    heatmap: List[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Aplica overrides de DEMO aos recursos 27, 29 e 248.
    Apenas afeta visualização - não altera dados reais do APS.
    """
    # Aplicar overrides nos recursos
    for resource in resources:
        recurso_id = str(resource.get("recurso", ""))
        # Verificar se é um dos recursos DEMO (pode ser "M-27", "27", "Recurso 27", etc.)
        is_demo_resource = any(demo_id in recurso_id for demo_id in DEMO_RESOURCES)
        
        if is_demo_resource:
            resource["utilizacao_pct"] = DEMO_UTILIZATION * 100
            resource["fila_horas"] = DEMO_QUEUE_HOURS
            resource["demo_override"] = True
    
    # Aplicar overrides nos bottlenecks
    for bottleneck in bottlenecks:
        recurso_id = str(bottleneck.get("recurso", ""))
        is_demo_resource = any(demo_id in recurso_id for demo_id in DEMO_RESOURCES)
        
        if is_demo_resource:
            bottleneck["utilizacao_pct"] = DEMO_UTILIZATION * 100
            bottleneck["fila_horas"] = DEMO_QUEUE_HOURS
            bottleneck["probabilidade"] = DEMO_PROB_GARGALO
            bottleneck["demo_override"] = True
            # Garantir que está no top 5
            bottleneck["acao"] = "Mover para Alternativa"
            bottleneck["impacto_otd"] = round(DEMO_PROB_GARGALO * 5, 1)
            bottleneck["impacto_horas"] = round(DEMO_QUEUE_HOURS * 0.1, 1)
    
    # Aplicar overrides no heatmap
    for row in heatmap:
        recurso_id = str(row.get("recurso", ""))
        is_demo_resource = any(demo_id in recurso_id for demo_id in DEMO_RESOURCES)
        
        if is_demo_resource:
            # Atualizar todas as horas para mostrar utilização alta
            for slot in row["utilizacao"]:
                slot["utilizacao_pct"] = DEMO_UTILIZATION * 100
            row["demo_override"] = True
    
    return bottlenecks, resources, heatmap


def _empty_bottlenecks_payload() -> Dict[str, Any]:
    return {
        "ready": False,
        "resources": [],
        "top_losses": [],
        "bottlenecks": [],
        "heatmap": [],
        "overlap_applied": {
            "transformacao": 0.0,
            "acabamentos": 0.0,
            "embalagem": 0.0,
        },
        "lead_time_gain": 0.0,
    }

@router.get("/")
async def get_bottlenecks(demo: Optional[bool] = Query(False, description="Ativar modo DEMO")):
    """
    Retorna análise de gargalos.
    
    Se demo=true, aplica overrides nos recursos 27, 29 e 248 para simular saturação.
    """
    demo_mode = demo or os.getenv("DEMO_MODE", "0") == "1"
    
    try:
        scheduler = APSScheduler()
        predictor = BottleneckPredictor()

        start_date = datetime.utcnow()
        end_date = start_date + timedelta(days=7)
        plano = scheduler.generate_optimized_plan(start_date, end_date)
        baseline = scheduler.generate_baseline_plan(start_date, end_date)

        operations = plano.operations if plano else []
        if not operations:
            logger.info("Bottlenecks: plano vazio, devolvendo payload neutro.")
            return _empty_bottlenecks_payload()

        resource_usage: Dict[str, float] = {}
        resource_queue: Dict[str, float] = {}
        for op in operations:
            recurso = op.recurso or "DESCONHECIDO"
            duration = (op.end_time - op.start_time).total_seconds() / 3600
            resource_usage[recurso] = resource_usage.get(recurso, 0.0) + duration
            resource_queue[recurso] = resource_queue.get(recurso, 0.0) + duration

        # Calcular janela temporal REAL (como no scheduler)
        if operations:
            min_start = min(op.start_time for op in operations)
            max_end = max(op.end_time for op in operations)
            total_hours = (max_end - min_start).total_seconds() / 3600.0 if max_end > min_start else 168.0
        else:
            total_hours = 168.0

        bottlenecks: List[Dict[str, Any]] = []
        resources: List[Dict[str, Any]] = []
        for recurso, usage in resource_usage.items():
            # Usar janela temporal real em vez de 168h fixas
            utilization_pct = (usage / total_hours) * 100 if total_hours > 0 else 0.0
            queue_hours = resource_queue.get(recurso, 0.0)

            resources.append(
                {
                    "recurso": recurso,
                    "utilizacao_pct": round(utilization_pct, 1),
                    "fila_horas": round(queue_hours, 1),
                }
            )

            prob = predictor.predict_probability(
                utilizacao=utilization_pct,
                num_setups=10,
                staffing=15,
                fila_atual=queue_hours,
            )

            if prob > 0.5 or utilization_pct > 80:
                drivers = predictor.get_bottleneck_drivers(
                    utilizacao=utilization_pct,
                    num_setups=10,
                    staffing=15,
                    fila_atual=queue_hours,
                )

                bottlenecks.append(
                    {
                        "recurso": recurso,
                        "utilizacao_pct": round(utilization_pct, 1),
                        "fila_horas": round(queue_hours, 1),
                        "probabilidade": round(prob, 3),
                        "drivers": drivers,
                        "acao": "Mover para Alternativa" if prob > 0.7 else "Colar famílias",
                        "impacto_otd": round(prob * 5, 1),
                        "impacto_horas": round(queue_hours * 0.1, 1),
                    }
                )

        top_losses = sorted(bottlenecks, key=lambda x: x["probabilidade"], reverse=True)[:5]

        heatmap: List[Dict[str, Any]] = []
        # Usar janela temporal real para o heatmap (arredondar para múltiplos de 8h)
        heatmap_hours = int(total_hours)
        heatmap_hours = ((heatmap_hours // 8) + 1) * 8  # Arredondar para cima para múltiplo de 8
        hours = list(range(0, min(heatmap_hours, 168), 8))  # Máximo 168h (1 semana)
        
        for recurso in list(resource_usage.keys())[:10]:
            row = {"recurso": recurso, "utilizacao": []}
            # Usar janela temporal real
            base_usage = resource_usage.get(recurso, 0.0) / total_hours if total_hours > 0 else 0.0
            for hour in hours:
                usage_at_hour = base_usage * (1 + 0.2 * (hour % 24) / 24)
                row["utilizacao"].append(
                    {
                        "hora": hour,
                        "utilizacao_pct": round(min(usage_at_hour * 100, 100), 1),
                    }
                )
            heatmap.append(row)

        def _avg_overlap(prefixes: List[str]) -> float:
            overlaps = [
                float(op.overlap or 0.0)
                for op in operations
                if op.setor and any(op.setor.lower().startswith(prefix) for prefix in prefixes)
            ]
            return float(sum(overlaps) / len(overlaps)) if overlaps else 0.0

        overlap_applied = {
            "transformacao": _avg_overlap(["transform", "linha 1"]),
            "acabamentos": _avg_overlap(["acaba", "linha 2"]),
            "embalagem": _avg_overlap(["embala", "linha 3"]),
        }

        lead_time_gain = 0.0
        if baseline and baseline.kpis and plano and plano.kpis:
            base_lead = float(baseline.kpis.get("lead_time_h", 0.0) or 0.0)
            opt_lead = float(plano.kpis.get("lead_time_h", 0.0) or 0.0)
            if base_lead > 0:
                lead_time_gain = round(max(0.0, (base_lead - opt_lead) / base_lead * 100), 1)

        # Aplicar overrides de DEMO se ativado
        demo_overrides = {}
        if demo_mode:
            bottlenecks, resources, heatmap = apply_demo_overrides_to_bottlenecks(
                bottlenecks, resources, heatmap
            )
            # Garantir que recursos DEMO aparecem no top 5
            for demo_id in DEMO_RESOURCES:
                demo_overrides[demo_id] = True
                # Adicionar aos bottlenecks se não estiverem lá
                demo_bottleneck_exists = any(
                    any(demo_id in str(b.get("recurso", "")) for demo_id in DEMO_RESOURCES)
                    for b in bottlenecks
                )
                if not demo_bottleneck_exists:
                    bottlenecks.append({
                        "recurso": f"M-{demo_id}",
                        "utilizacao_pct": DEMO_UTILIZATION * 100,
                        "fila_horas": DEMO_QUEUE_HOURS,
                        "probabilidade": DEMO_PROB_GARGALO,
                        "drivers": ["Saturação artificial (modo DEMO)"],
                        "acao": "Mover para Alternativa",
                        "impacto_otd": round(DEMO_PROB_GARGALO * 5, 1),
                        "impacto_horas": round(DEMO_QUEUE_HOURS * 0.1, 1),
                        "demo_override": True,
                    })
            # Reordenar top_losses
            top_losses = sorted(bottlenecks, key=lambda x: x["probabilidade"], reverse=True)[:5]
            logger.info(f"Modo DEMO ativado: recursos {DEMO_RESOURCES} saturados artificialmente")

        payload = {
            "ready": True,
            "resources": resources,
            "top_losses": top_losses,
            "bottlenecks": bottlenecks,
            "heatmap": heatmap,
            "overlap_applied": overlap_applied,
            "lead_time_gain": lead_time_gain,
            "demo_mode": demo_mode,
            "demo_overrides": demo_overrides if demo_mode else {},
        }

        return payload
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Falha na análise de gargalos: %s", exc)
        return _empty_bottlenecks_payload()

