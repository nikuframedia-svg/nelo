"""
Teste direto para verificar escolha de rotas.

Este script testa diretamente a escolha de rotas sem passar pelo cache ou frontend.
"""

import logging
from datetime import datetime, timedelta

from app.aps.engine import APSEngine
from app.aps.models import APSConfig
from app.aps.parser import ProductionDataParser
from app.etl.loader import get_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_route_selection():
    """Testa diretamente a escolha de rotas."""
    
    print("=" * 80)
    print("TESTE DIRETO DE ESCOLHA DE ROTAS")
    print("=" * 80)
    print()
    
    # Carregar Excel
    loader = get_loader()
    data_dir = loader.data_dir
    excel_file = data_dir / "Nikufra DadosProducao (2).xlsx"
    if not excel_file.exists():
        excel_file = data_dir / "Nikufra DadosProducao.xlsx"
    
    if not excel_file.exists():
        print("❌ Excel não encontrado")
        return
    
    # Parse
    parser = ProductionDataParser()
    orders = parser.parse_dadosnikufra2(str(excel_file))
    
    print(f"📋 Orders parseadas: {len(orders)}\n")
    
    # Criar engine limpo
    aps_config = APSConfig()
    aps_config.routing_preferences["prefer_route"] = {}
    engine = APSEngine(config=aps_config, planning_config=None)
    
    # Inicializar máquinas
    engine._initialize_machines(orders)
    
    start_time = datetime.utcnow()
    end_time = start_time + timedelta(hours=8)
    
    # Calcular baseline
    print("=" * 80)
    print("BASELINE")
    print("=" * 80)
    baseline = engine._calculate_baseline(orders, start_time, end_time)
    
    baseline_rotas = {}
    for op in baseline.operations:
        artigo = op.order_id.replace('ORD-', '')
        rota = op.op_ref.rota if op.op_ref else '?'
        if artigo not in baseline_rotas:
            baseline_rotas[artigo] = []
        baseline_rotas[artigo].append(rota)
    
    print("\n📊 Rotas no baseline:")
    for artigo in sorted(baseline_rotas.keys()):
        rotas = baseline_rotas[artigo]
        rotas_unicas = list(set(rotas))
        print(f"  {artigo}: {rotas_unicas} ({rotas.count('A')} A, {rotas.count('B')} B)")
    
    # Calcular optimized
    print("\n" + "=" * 80)
    print("OPTIMIZED")
    print("=" * 80)
    
    # Reset máquinas para optimized
    for machine in engine.machines.values():
        machine.operacoes_agendadas = []
        machine.carga_acumulada_h = 0.0
        machine.ultima_operacao_fim = start_time
        machine.ultima_familia = None
    
    optimized = engine._calculate_optimized(orders, start_time, end_time, baseline)
    
    optimized_rotas = {}
    for op in optimized.operations:
        artigo = op.order_id.replace('ORD-', '')
        rota = op.op_ref.rota if op.op_ref else '?'
        if artigo not in optimized_rotas:
            optimized_rotas[artigo] = []
        optimized_rotas[artigo].append(rota)
    
    print("\n📊 Rotas no optimized:")
    for artigo in sorted(optimized_rotas.keys()):
        rotas = optimized_rotas[artigo]
        rotas_unicas = list(set(rotas))
        print(f"  {artigo}: {rotas_unicas} ({rotas.count('A')} A, {rotas.count('B')} B)")
        
        # Comparar com baseline
        baseline_rota = list(set(baseline_rotas.get(artigo, [])))
        if baseline_rota != rotas_unicas:
            print(f"    ⚠️ MUDANÇA: Baseline={baseline_rota} → Optimized={rotas_unicas}")
        else:
            print(f"    ✅ Mantido: {rotas_unicas}")
    
    # Resumo
    print("\n" + "=" * 80)
    print("RESUMO")
    print("=" * 80)
    
    todas_baseline = [r for rotas in baseline_rotas.values() for r in rotas]
    todas_optimized = [r for rotas in optimized_rotas.values() for r in rotas]
    
    print(f"\nBaseline: {todas_baseline.count('A')} A, {todas_baseline.count('B')} B")
    print(f"Optimized: {todas_optimized.count('A')} A, {todas_optimized.count('B')} B")
    
    if todas_optimized.count('B') == len(todas_optimized):
        print("\n❌ PROBLEMA: Optimized tem TUDO em B")
    elif todas_optimized.count('A') == len(todas_optimized):
        print("\n⚠️ Optimized tem TUDO em A (pode ser normal se A for sempre mais rápida)")
    else:
        print("\n✅ Optimized tem mistura A/B (correto)")


if __name__ == "__main__":
    test_route_selection()

