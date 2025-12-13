"""
AUDITORIA COMPLETA DA PIPELINE DE ROTAS A/B

Este módulo implementa uma auditoria sistemática para diagnosticar problemas
com a seleção de rotas no APS.

Uso:
    python -m app.aps.audit_routes
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set

from app.aps.engine import APSEngine
from app.aps.models import APSConfig, Order
from app.aps.parser import ProductionDataParser
from app.etl.loader import get_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def audit_parser(excel_path: str) -> List[Order]:
    """
    PONTO 2: Auditoria ao PARSER
    
    Parse o Excel e loga detalhes de cada Order:
    - Rotas encontradas
    - Operações por rota
    - Máquinas por operação
    """
    logger.info("=" * 80)
    logger.info("PONTO 2: AUDITORIA AO PARSER")
    logger.info("=" * 80)
    
    parser = ProductionDataParser()
    orders = parser.parse_dadosnikufra2(excel_path)
    
    logger.info(f"\n📋 Total de Orders parseadas: {len(orders)}\n")
    
    for order in orders:
        logger.info(f"[PARSER] Order {order.artigo} (ID: {order.id})")
        
        # Agrupar operações por rota
        by_route: Dict[str, List] = {}
        for op_ref in order.operations:
            rota = op_ref.rota.upper() if op_ref.rota else "SEM_ROTA"
            if rota not in by_route:
                by_route[rota] = []
            by_route[rota].append(op_ref)
        
        routes_found = sorted(list(by_route.keys()))
        logger.info(f"  [PARSER]   Rotas encontradas: {routes_found}")
        
        # Detalhar cada operação
        for op_ref in order.operations:
            rota = op_ref.rota.upper() if op_ref.rota else "SEM_ROTA"
            machines = [alt.maquina_id for alt in op_ref.alternatives]
            logger.info(
                f"  [PARSER]   Op {op_ref.op_id} rota={rota} "
                f"stage={op_ref.stage_index} máquinas={machines}"
            )
        
        # Verificar se tem rotas A e B
        has_a = any(r.upper() == "A" for r in routes_found)
        has_b = any(r.upper() == "B" for r in routes_found)
        
        if not has_a and has_b:
            logger.warning(f"  ⚠️ [PARSER] {order.artigo}: SÓ TEM ROTA B (sem A)")
        elif has_a and not has_b:
            logger.warning(f"  ⚠️ [PARSER] {order.artigo}: SÓ TEM ROTA A (sem B)")
        elif not has_a and not has_b:
            logger.error(f"  ❌ [PARSER] {order.artigo}: SEM ROTAS VÁLIDAS")
        else:
            logger.info(f"  ✅ [PARSER] {order.artigo}: Tem rotas A e B")
        
        logger.info("")
    
    return orders


def audit_get_all_available_routes(engine: APSEngine, orders: List[Order]):
    """
    PONTO 3: Auditoria ao _get_all_available_routes
    
    Para cada Order, verifica quais rotas estão disponíveis.
    """
    logger.info("=" * 80)
    logger.info("PONTO 3: AUDITORIA AO _get_all_available_routes")
    logger.info("=" * 80)
    
    for order in orders:
        available_routes = engine._get_all_available_routes(order)
        routes_list = sorted(list(available_routes.keys()))
        
        logger.info(f"[APS] Order {order.artigo}: rotas disponíveis = {routes_list}")
        
        # Verificar se tem A e B
        has_a = any(r.upper() == "A" for r in routes_list)
        has_b = any(r.upper() == "B" for r in routes_list)
        
        if not has_a and has_b:
            logger.warning(f"  ⚠️ [APS] {order.artigo}: _get_all_available_routes retornou SÓ B")
        elif has_a and not has_b:
            logger.warning(f"  ⚠️ [APS] {order.artigo}: _get_all_available_routes retornou SÓ A")
        elif not has_a and not has_b:
            logger.error(f"  ❌ [APS] {order.artigo}: _get_all_available_routes retornou NENHUMA ROTA")
        else:
            logger.info(f"  ✅ [APS] {order.artigo}: Tem rotas A e B disponíveis")
        
        # Detalhar operações por rota
        for route_name, route_ops in available_routes.items():
            logger.info(f"    Rota {route_name}: {len(route_ops)} operações")
            for op in route_ops:
                machines = [alt.maquina_id for alt in op.alternatives]
                logger.info(f"      Op {op.op_id} (stage {op.stage_index}): máquinas {machines}")
        
        logger.info("")


def audit_choose_best_route(
    engine: APSEngine,
    orders: List[Order],
    start_time: datetime,
    end_time: datetime,
):
    """
    PONTO 4: Auditoria ao _choose_best_route
    
    Para cada Order, simula a escolha de rota e loga:
    - Se há prefer_route forçada
    - Scores de cada rota
    - Rota escolhida
    """
    logger.info("=" * 80)
    logger.info("PONTO 4: AUDITORIA AO _choose_best_route")
    logger.info("=" * 80)
    
    # Identificar gargalos (usar baseline)
    baseline = engine._calculate_baseline(orders, start_time, end_time)
    bottleneck_machines = engine._identify_bottleneck_machines(
        baseline, start_time, end_time, utilization_threshold=0.85
    )
    logger.info(f"🔍 Gargalos identificados: {sorted(bottleneck_machines)}\n")
    
    for order in orders:
        logger.info(f"[APS] Order {order.artigo} ({order.id})")
        
        # Verificar prefer_route
        prefer_route_dict = engine.config.routing_preferences.get("prefer_route", {})
        prefer_route = prefer_route_dict.get(order.artigo)
        
        if prefer_route:
            logger.info(f"  [APS]   prefer_route = {prefer_route} (FORÇADA)")
            logger.info(f"  [APS]   Rota escolhida = {prefer_route} (forçada pela config)")
            logger.info("")
            continue
        
        # Obter rotas disponíveis
        available_routes = engine._get_all_available_routes(order)
        routes_list = sorted(list(available_routes.keys()))
        
        if not routes_list:
            logger.error(f"  ❌ [APS] {order.artigo}: Nenhuma rota disponível")
            logger.info("")
            continue
        
        logger.info(f"  [APS]   Rotas disponíveis: {routes_list}")
        
        # Calcular scores para cada rota
        scores = {}
        for route_name, route_ops in available_routes.items():
            try:
                score = engine._simulate_route_score(
                    order, route_ops, route_name, start_time, end_time, bottleneck_machines
                )
                scores[route_name] = score
                logger.info(f"  [APS]   rota {route_name} → score={score:.2f}")
            except Exception as exc:
                logger.error(f"  ❌ [APS] Erro ao simular rota {route_name}: {exc}")
                scores[route_name] = float('inf')
        
        # Escolher melhor rota
        if scores:
            best_route = min(scores.items(), key=lambda x: x[1])
            logger.info(f"  [APS]   rota escolhida = {best_route[0]} (score={best_route[1]:.2f})")
            
            # Verificar se há viés
            if "A" in scores and "B" in scores:
                score_a = scores["A"]
                score_b = scores["B"]
                diff_pct = abs(score_a - score_b) / max(score_a, score_b) * 100
                if score_b < score_a * 0.9:  # B é >10% melhor
                    logger.info(f"  ✅ [APS] Rota B é {diff_pct:.1f}% melhor que A (escolha correta)")
                elif score_a < score_b * 0.9:  # A é >10% melhor
                    logger.info(f"  ✅ [APS] Rota A é {diff_pct:.1f}% melhor que B (escolha correta)")
                else:
                    logger.warning(f"  ⚠️ [APS] Rotas muito próximas (diff={diff_pct:.1f}%) - pode haver viés")
        
        logger.info("")


def audit_routing_preferences(engine: APSEngine):
    """
    PONTO 5: Garantir que não há preferências globais
    
    Verifica se há preferências de rota inválidas ou globais.
    """
    logger.info("=" * 80)
    logger.info("PONTO 5: AUDITORIA DE PREFERÊNCIAS DE ROTA")
    logger.info("=" * 80)
    
    prefer_route_dict = engine.config.routing_preferences.get("prefer_route", {})
    
    if not prefer_route_dict:
        logger.info("✅ [AUDIT] prefer_route está vazio (correto)")
        return
    
    logger.info(f"📋 [AUDIT] prefer_route tem {len(prefer_route_dict)} entradas:")
    
    invalid_keys = []
    for key, value in prefer_route_dict.items():
        if key in ["*", "all", "ALL", ""] or not key or len(str(key)) < 2:
            invalid_keys.append(key)
            logger.error(f"  ❌ [AUDIT] Chave inválida/global: '{key}' = '{value}'")
        else:
            logger.info(f"  📋 [AUDIT] '{key}' = '{value}'")
    
    if invalid_keys:
        logger.error(f"❌ [AUDIT] Encontradas {len(invalid_keys)} preferências inválidas/globais")
    else:
        logger.info("✅ [AUDIT] Todas as preferências são válidas (por artigo específico)")


def run_full_audit():
    """
    Executa auditoria completa.
    """
    logger.info("=" * 80)
    logger.info("AUDITORIA COMPLETA DA PIPELINE DE ROTAS A/B")
    logger.info("=" * 80)
    logger.info("")
    
    # Obter Excel atual
    loader = get_loader()
    data_dir = loader.data_dir
    excel_file = data_dir / "Nikufra DadosProducao (2).xlsx"
    if not excel_file.exists():
        excel_file = data_dir / "Nikufra DadosProducao.xlsx"
    
    if not excel_file.exists():
        logger.error(f"❌ Excel não encontrado em {data_dir}")
        return
    
    logger.info(f"📁 Excel: {excel_file}\n")
    
    # PONTO 2: Auditoria ao PARSER
    orders = audit_parser(str(excel_file))
    
    if not orders:
        logger.error("❌ Nenhuma Order parseada. Abortando auditoria.")
        return
    
    # Criar engine limpo
    aps_config = APSConfig()
    aps_config.routing_preferences["prefer_route"] = {}  # Garantir limpo
    engine = APSEngine(config=aps_config, planning_config=None)
    
    # PONTO 5: Verificar preferências
    audit_routing_preferences(engine)
    logger.info("")
    
    # PONTO 3: Auditoria ao _get_all_available_routes
    audit_get_all_available_routes(engine, orders)
    
    # PONTO 4: Auditoria ao _choose_best_route
    start_time = datetime.utcnow()
    end_time = start_time + timedelta(hours=8)
    audit_choose_best_route(engine, orders, start_time, end_time)
    
    # PONTO 8: Teste manual completo
    logger.info("=" * 80)
    logger.info("PONTO 8: TESTE MANUAL COMPLETO (baseline + optimized)")
    logger.info("=" * 80)
    
    # Inicializar máquinas
    engine._initialize_machines(orders)
    
    # Baseline
    baseline = engine._calculate_baseline(orders, start_time, end_time)
    logger.info("\n📊 BASELINE:")
    baseline_routes = {}
    for op in baseline.operations:
        artigo = op.order_id.replace("ORD-", "")
        rota = op.op_ref.rota
        if artigo not in baseline_routes:
            baseline_routes[artigo] = rota
        elif baseline_routes[artigo] != rota:
            logger.warning(f"  ⚠️ {artigo}: múltiplas rotas no baseline")
    
    for artigo, rota in sorted(baseline_routes.items()):
        logger.info(f"  [TEST] {artigo} → baseline: {rota}")
    
    # Optimized
    optimized = engine._calculate_optimized(orders, start_time, end_time, baseline)
    logger.info("\n📊 OPTIMIZED:")
    optimized_routes = {}
    for op in optimized.operations:
        artigo = op.order_id.replace("ORD-", "")
        rota = op.op_ref.rota
        if artigo not in optimized_routes:
            optimized_routes[artigo] = rota
        elif optimized_routes[artigo] != rota:
            logger.warning(f"  ⚠️ {artigo}: múltiplas rotas no optimized")
    
    for artigo, rota in sorted(optimized_routes.items()):
        baseline_rota = baseline_routes.get(artigo, "?")
        logger.info(f"  [TEST] {artigo} → baseline: {baseline_rota}, optimized: {rota}")
    
    # Resumo
    logger.info("\n📊 RESUMO:")
    all_a_baseline = all(r == "A" for r in baseline_routes.values())
    all_b_baseline = all(r == "B" for r in baseline_routes.values())
    all_a_optimized = all(r == "A" for r in optimized_routes.values())
    all_b_optimized = all(r == "B" for r in optimized_routes.values())
    
    logger.info(f"  Baseline: {'TUDO A' if all_a_baseline else 'TUDO B' if all_b_baseline else 'MISTO'}")
    logger.info(f"  Optimized: {'TUDO A' if all_a_optimized else 'TUDO B' if all_b_optimized else 'MISTO'}")
    
    if all_b_optimized:
        logger.error("  ❌ PROBLEMA: Optimized tem TUDO em B")
    elif all_a_optimized:
        logger.warning("  ⚠️ Optimized tem TUDO em A (pode ser normal se A for sempre melhor)")
    else:
        logger.info("  ✅ Optimized tem mistura A/B (correto)")
    
    logger.info("\n" + "=" * 80)
    logger.info("AUDITORIA CONCLUÍDA")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_full_audit()

