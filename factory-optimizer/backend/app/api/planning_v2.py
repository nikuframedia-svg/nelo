"""
API v2 para planeamento APS encadeado.

Endpoints:
- GET /api/planning/plano?batch_id=X&horizon_hours=4
- POST /api/planning/recalculate?batch_id=X&horizon_hours=4
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.aps.cache import get_plan_cache
from app.aps.engine import APSEngine
from app.aps.models import APSConfig, Order
from app.aps.parser import ProductionDataParser
from app.aps.planning_config import get_planning_config_store
from app.etl.loader import get_loader

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/plano")
async def get_plan(
    batch_id: Optional[str] = Query(None, description="Batch ID"),
    horizon_hours: int = Query(24, description="Horizonte de planeamento em horas"),  # Aumentado para 24h para agendar todas as operações
):
    """
    Retorna plano do cache (não recalcula).
    
    Se não existir no cache, retorna erro (utilizador deve clicar "Recalcular").
    """
    if not batch_id:
        # Tentar obter batch_id mais recente
        loader = get_loader()
        status = loader.get_status()
        batch_id = status.get("latest_batch_id") or status.get("batch_id") or "default"
    
    cache = get_plan_cache()
    plan = cache.get(batch_id, horizon_hours)
    
    if not plan:
        raise HTTPException(
            status_code=404,
            detail=f"Plano não encontrado para batch_id={batch_id}, horizon_hours={horizon_hours}. Clique em 'Recalcular plano'.",
        )
    
    plan_dict = plan.to_dict()
    
    # Log para debug
    baseline_ops = plan_dict.get("baseline", {}).get("operations", [])
    optimized_ops = plan_dict.get("optimized", {}).get("operations", [])
    baseline_articles = set(op.get("artigo", "") for op in baseline_ops)
    optimized_articles = set(op.get("artigo", "") for op in optimized_ops)
    logger.info(
        f"📤 GET /plano retornando: baseline={len(baseline_ops)} ops ({len(baseline_articles)} artigos: {sorted(baseline_articles)}), "
        f"optimized={len(optimized_ops)} ops ({len(optimized_articles)} artigos: {sorted(optimized_articles)})"
    )
    
    return plan_dict


@router.post("/recalculate")
async def recalculate_plan(
    batch_id: Optional[str] = Query(None, description="Batch ID"),
    horizon_hours: int = Query(24, description="Horizonte de planeamento em horas"),  # Aumentado para 24h para agendar todas as operações
):
    """
    Força recálculo do plano e atualiza cache.
    
    Processo:
    1. Lê Excel do batch_id
    2. Parse e constrói Orders
    3. Executa APS (baseline + optimized)
    4. Guarda no cache
    """
    if not batch_id:
        loader = get_loader()
        status = loader.get_status()
        batch_id = status.get("latest_batch_id") or status.get("batch_id") or "default"
    
    try:
        # Invalidar cache antigo antes de recalcular (força regeneração)
        cache = get_plan_cache()
        cache.invalidate(batch_id)  # Invalidar todos os horizon_hours deste batch_id
        logger.info(f"🗑️ Cache invalidado para batch_id={batch_id} (forçando recálculo completo)")
        
        # 1. Obter ficheiro Excel do batch
        loader = get_loader()
        data_dir = loader.data_dir
        
        # Procurar ficheiro de produção - PRIORIDADE: "Nikufra DadosProducao (2).xlsx"
        # Primeiro, tentar encontrar o ficheiro principal "(2)" (com dados atualizados)
        excel_file = None
        primary_file = data_dir / "Nikufra DadosProducao (2).xlsx"
        if primary_file.exists():
            excel_file = primary_file
            logger.info(f"✅ Usando ficheiro principal (2): {excel_file}")
        else:
            # Se não existir, procurar outros ficheiros Nikufra
            production_files = (
                list(data_dir.glob("**/Nikufra DadosProducao*.xlsx"))
                + list(data_dir.glob("**/dadosnikufra*.xlsx"))
                + list(data_dir.glob("**/DadosProducao*.xlsx"))
            )
            if not production_files:
                raise HTTPException(
                    status_code=404,
                    detail="Ficheiro de produção não encontrado. Faça upload primeiro (Nikufra DadosProducao (2).xlsx).",
                )
            
            # Filtrar para priorizar "(2)" se possível
            files_with_2 = [f for f in production_files if "(2)" in str(f)]
            if files_with_2:
                excel_file = max(files_with_2, key=lambda p: p.stat().st_mtime)
                logger.info(f"✅ Usando ficheiro (com '(2)'): {excel_file}")
            else:
                # Se não houver ficheiros com "(2)", usar o mais recente sem "(2)"
                excel_file = max(production_files, key=lambda p: p.stat().st_mtime)
                logger.warning(f"⚠️ Usando ficheiro alternativo (não encontrado '(2)'): {excel_file}")
        
        # 2. Parse Excel com cache (evita reparsear)
        from app.aps.parser_cache import get_parser_cache
        parser_cache = get_parser_cache()
        
        # Tentar obter do cache primeiro
        orders = parser_cache.get(str(excel_file))
        
        if not orders:
            # Cache miss - fazer parse
            logger.info(f"Cache miss - parseando Excel: {excel_file}")
            parser = ProductionDataParser()
            file_str = str(excel_file).lower()
            if any(keyword in file_str for keyword in ["dadosnikufra", "nikufra dadosproducao", "dadosproducao"]):
                orders = parser.parse_dadosnikufra2(str(excel_file))
            else:
                logger.warning(f"Ficheiro {excel_file} não reconhecido como Nikufra.")
                orders = parser.parse_excel(str(excel_file), sheet_name=None) if hasattr(parser, 'parse_excel') else []
            
            if not orders:
                raise HTTPException(
                    status_code=400,
                    detail="Nenhuma ordem encontrada no Excel. Verifique o formato do ficheiro e se há folhas válidas.",
                )
            
            # Guardar no cache
            parser_cache.set(str(excel_file), orders)
            logger.info(f"✅ Parse completo: {len(orders)} orders, {len(set(o.artigo for o in orders))} artigos")
        else:
            logger.info(f"✅ Cache hit: {len(orders)} orders do cache")
        
        # 3. Carregar configuração de planeamento (indisponibilidades, ordens manuais, etc.)
        config_store = get_planning_config_store()
        planning_config = config_store.get(batch_id)
        
        # 4. PONTO 5: Criar APSConfig limpo (sem preferências de rota)
        # CRÍTICO: Sempre criar APSConfig limpo para garantir que não há preferências de rota "presas"
        # CRÍTICO: Isto garante que "otimizar plano" não usa preferências de rota antigas
        aps_config = APSConfig()
        # Garantir explicitamente que prefer_route está vazio (sem preferências forçadas)
        aps_config.routing_preferences["prefer_route"] = {}
        
        # VALIDAÇÃO CRÍTICA: Limpar qualquer preferência global ou inválida
        # Rejeitar preferências como "*", "all", ou qualquer chave que não seja um artigo específico
        invalid_keys = []
        for key in list(aps_config.routing_preferences.get("prefer_route", {}).keys()):
            if key in ["*", "all", "ALL", ""] or not key or len(key) < 2:
                invalid_keys.append(key)
                logger.warning(f"⚠️ [AUDIT] Removendo preferência de rota inválida: '{key}'")
        
        for invalid_key in invalid_keys:
            aps_config.routing_preferences["prefer_route"].pop(invalid_key, None)
        
        # Garantir que está vazio após limpeza
        if aps_config.routing_preferences.get("prefer_route"):
            logger.warning(f"⚠️ [AUDIT] APSConfig tinha preferências de rota: {aps_config.routing_preferences.get('prefer_route')}")
            aps_config.routing_preferences["prefer_route"] = {}
        
        logger.info(f"✅ [AUDIT] APSConfig criado limpo. routing_preferences.prefer_route = {aps_config.routing_preferences.get('prefer_route')}")
        
        # 5. Executar APS (com configuração de planeamento aplicada e APSConfig limpo)
        engine = APSEngine(config=aps_config, planning_config=planning_config)
        start_time = datetime.utcnow()
        
        # Usar horizonte da configuração se definido
        effective_horizon = planning_config.horizon_hours if planning_config.horizon_hours else horizon_hours
        
        # Log de performance
        import time
        perf_start = time.time()
        plan = engine.build_schedule(orders, effective_horizon, start_time)
        perf_elapsed = time.time() - perf_start
        logger.info(f"⚡ APS executado em {perf_elapsed:.2f}s para {len(orders)} orders, horizon={effective_horizon}h")
        plan.batch_id = batch_id  # Usar batch_id fornecido
        
        # Log detalhado das operações por artigo
        baseline_articles = set(op.order_id for op in plan.baseline.operations)
        optimized_articles = set(op.order_id for op in plan.optimized.operations)
        logger.info(
            f"Plano calculado: baseline makespan={plan.baseline.makespan_h:.1f}h, "
            f"optimized makespan={plan.optimized.makespan_h:.1f}h, "
            f"baseline operations={len(plan.baseline.operations)}, "
            f"optimized operations={len(plan.optimized.operations)}"
        )
        logger.info(
            f"📊 Artigos processados - Baseline: {len(baseline_articles)} Orders ({sorted(baseline_articles)}), "
            f"Optimized: {len(optimized_articles)} Orders ({sorted(optimized_articles)})"
        )
        
        # 4. PONTO 5: Garantir que não há preferências globais antes de guardar
        # Limpar qualquer preferência global que possa ter sido introduzida
        if plan.config.routing_preferences.get("prefer_route"):
            prefer_route_dict = plan.config.routing_preferences["prefer_route"]
            invalid_keys = [k for k in prefer_route_dict.keys() if k in ["*", "all", "ALL", ""] or not k or len(str(k)) < 2]
            if invalid_keys:
                logger.warning(f"⚠️ [AUDIT] Plano guardado no cache tinha preferências inválidas: {invalid_keys}. Removendo.")
                for invalid_key in invalid_keys:
                    prefer_route_dict.pop(invalid_key, None)
        
        # 4. Guardar no cache (passar todas as máquinas para garantir serialização completa)
        all_machine_ids = sorted(list(engine.machines.keys()))
        cache.set(batch_id, horizon_hours, plan, all_machines=all_machine_ids)
        logger.info(f"✅ [APS] Plano guardado no cache com {len(all_machine_ids)} máquinas: {all_machine_ids}")
        
        return {
            "ok": True,
            "batch_id": batch_id,
            "horizon_hours": horizon_hours,
            "plan": plan.to_dict(),
            "message": f"Plano recalculado com sucesso. {len(orders)} ordens processadas.",
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Erro ao recalcular plano: {exc}")
        raise HTTPException(status_code=500, detail=f"Erro ao recalcular plano: {str(exc)}") from exc


@router.get("/diagnose-routes")
async def diagnose_routes_endpoint(batch_id: Optional[str] = Query(None)):
    """
    Endpoint de diagnóstico para verificar rotas no backend vs JSON.
    
    Retorna análise detalhada das rotas escolhidas vs serializadas.
    """
    try:
        if not batch_id:
            loader = get_loader()
            status = loader.get_status()
            batch_id = status.get("latest_batch_id") or status.get("batch_id")
        
        if not batch_id:
            raise HTTPException(status_code=400, detail="batch_id necessário")
        
        cache = get_plan_cache()
        plan = cache.get(batch_id, 8)
        
        if not plan or not plan.optimized:
            raise HTTPException(status_code=404, detail="Plano não encontrado. Recalcule primeiro.")
        
        # Analisar rotas no objeto Python
        rotas_por_artigo_python = {}
        for op in plan.optimized.operations:
            artigo = op.order_id.replace('ORD-', '')
            rota = op.op_ref.rota if op.op_ref else 'SEM_ROTA'
            
            if artigo not in rotas_por_artigo_python:
                rotas_por_artigo_python[artigo] = []
            rotas_por_artigo_python[artigo].append(rota)
        
        # Analisar rotas no JSON
        plan_dict = plan.to_dict()
        optimized_ops = plan_dict.get('optimized', {}).get('operations', [])
        
        rotas_por_artigo_json = {}
        for op in optimized_ops:
            artigo = op.get('artigo') or op.get('order_id', '').replace('ORD-', '')
            rota = op.get('rota')
            
            if artigo not in rotas_por_artigo_json:
                rotas_por_artigo_json[artigo] = []
            rotas_por_artigo_json[artigo].append(rota or 'MISSING')
        
        # Comparar
        problemas = []
        for artigo in set(list(rotas_por_artigo_python.keys()) + list(rotas_por_artigo_json.keys())):
            rotas_python = set(rotas_por_artigo_python.get(artigo, []))
            rotas_json = set(rotas_por_artigo_json.get(artigo, []))
            
            if rotas_python != rotas_json:
                problemas.append({
                    'artigo': artigo,
                    'python': list(rotas_python),
                    'json': list(rotas_json)
                })
        
        # Distribuição geral
        todas_rotas_python = [op.op_ref.rota for op in plan.optimized.operations if op.op_ref]
        todas_rotas_json = [op.get('rota') for op in optimized_ops if op.get('rota')]
        
        return {
            "ok": True,
            "batch_id": batch_id,
            "total_ops_python": len(plan.optimized.operations),
            "total_ops_json": len(optimized_ops),
            "distribuicao_python": {
                "A": todas_rotas_python.count('A'),
                "B": todas_rotas_python.count('B')
            },
            "distribuicao_json": {
                "A": todas_rotas_json.count('A'),
                "B": todas_rotas_json.count('B')
            },
            "rotas_por_artigo_python": {
                k: {
                    "rotas": list(set(v)),
                    "total": len(v),
                    "distribuicao": {"A": v.count('A'), "B": v.count('B')}
                }
                for k, v in rotas_por_artigo_python.items()
            },
            "rotas_por_artigo_json": {
                k: {
                    "rotas": list(set(v)),
                    "total": len(v),
                    "distribuicao": {"A": v.count('A'), "B": v.count('B')}
                }
                for k, v in rotas_por_artigo_json.items()
            },
            "problemas": problemas,
            "primeiras_10_python": [
                {
                    "artigo": op.order_id.replace('ORD-', ''),
                    "op_id": op.op_ref.op_id if op.op_ref else '?',
                    "rota": op.op_ref.rota if op.op_ref else 'SEM_ROTA',
                    "maquina": op.maquina_id
                }
                for op in plan.optimized.operations[:10]
            ],
            "primeiras_10_json": [
                {
                    "artigo": op.get('artigo') or op.get('order_id', '').replace('ORD-', ''),
                    "op_id": op.get('op_id', '?'),
                    "rota": op.get('rota', 'MISSING'),
                    "maquina": op.get('maquina_id', '?')
                }
                for op in optimized_ops[:10]
            ]
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Erro no diagnóstico: {exc}")
        raise HTTPException(status_code=500, detail=f"Erro no diagnóstico: {str(exc)}")


@router.post("/audit-routes")
async def audit_routes_endpoint():
    """
    PONTO 8: Endpoint para executar auditoria completa de rotas.
    
    Executa auditoria sistemática da pipeline de rotas A/B sem usar Chat/LLM.
    Retorna logs detalhados da auditoria.
    """
    try:
        import subprocess
        import sys
        from pathlib import Path
        
        # Executar script de auditoria
        backend_dir = Path(__file__).parent.parent.parent
        script_path = backend_dir / "app" / "aps" / "audit_routes.py"
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(backend_dir)
        )
        
        return {
            "ok": True,
            "logs": result.stdout + result.stderr,
            "returncode": result.returncode,
            "message": "Auditoria executada. Ver logs para detalhes."
        }
    except Exception as exc:
        logger.exception(f"Erro ao executar auditoria: {exc}")
        raise HTTPException(status_code=500, detail=f"Erro ao executar auditoria: {str(exc)}")


@router.get("/config")
async def get_config(batch_id: Optional[str] = Query(None)):
    """Retorna configuração APS atual."""
    if not batch_id:
        loader = get_loader()
        status = loader.get_status()
        batch_id = status.get("latest_batch_id") or status.get("batch_id") or "default"
    
    cache = get_plan_cache()
    plan = cache.get(batch_id, 4)  # Usar horizon_hours=4 como default
    
    if plan:
        return plan.config.to_dict()
    
    # Retornar config padrão se não houver plano
    return APSConfig().to_dict()


@router.post("/config")
async def update_config(
    config_data: dict,
    batch_id: Optional[str] = Query(None),
):
    """
    Atualiza configuração APS.
    
    Body: {
        "objective": {...},
        "overlap": {...},
        "routing_preferences": {...},
        ...
    }
    """
    if not batch_id:
        loader = get_loader()
        status = loader.get_status()
        batch_id = status.get("latest_batch_id") or status.get("batch_id") or "default"
    
    try:
        # Validar e criar config
        config = APSConfig.from_dict(config_data)
        
        # Validar pesos objetivos (soma ≈ 1.0)
        obj_weights = config.objective
        total_weight = sum(
            obj_weights.get(k, 0.0)
            for k in ["weight_lead_time", "weight_setups", "weight_bottleneck_balance", "weight_otd"]
        )
        if abs(total_weight - 1.0) > 0.1:
            raise HTTPException(
                status_code=400,
                detail=f"Pesos objetivos devem somar ≈ 1.0 (atual: {total_weight:.2f})",
            )
        
        # Invalidar planos otimizados (manter baseline se possível)
        cache = get_plan_cache()
        # Por agora, invalidar tudo - no futuro pode manter baseline
        cache.invalidate(batch_id)
        
        return {
            "ok": True,
            "config": config.to_dict(),
            "message": "Configuração atualizada. Recalcule o plano para aplicar mudanças.",
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Erro ao atualizar config: {exc}")
        raise HTTPException(status_code=500, detail=f"Erro ao atualizar config: {str(exc)}") from exc

