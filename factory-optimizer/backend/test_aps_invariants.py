"""
Testes simples para verificar invariantes do APS.

Executar: python test_aps_invariants.py
"""

import logging
from pathlib import Path
from datetime import datetime

from app.aps.parser import parse_dadosnikufra2
from app.aps.engine import APSEngine

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_parser_invariants():
    """Testa invariantes do parser."""
    print("\n" + "="*80)
    print("TESTE 1: Invariantes do Parser")
    print("="*80)
    
    # Priorizar "Nikufra DadosProducao (2).xlsx" (com dados atualizados)
    file_path = 'app/data/Nikufra DadosProducao (2).xlsx'
    if not Path(file_path).exists():
        # Fallback para sem "(2)" se o principal não existir
        file_path = 'app/data/Nikufra DadosProducao.xlsx'
        if not Path(file_path).exists():
            print(f"❌ Ficheiro não encontrado: Nikufra DadosProducao (2).xlsx ou Nikufra DadosProducao.xlsx")
            return False
        else:
            print(f"⚠️ Usando ficheiro alternativo (sem '(2)'): {file_path}")
    else:
        print(f"✅ Usando ficheiro principal (2): {file_path}")
    
    orders = parse_dadosnikufra2(file_path)
    
    # Invariante 1: Deve ter pelo menos 6 Orders (GO Artigo 1-6)
    assert len(orders) >= 6, f"Esperava pelo menos 6 Orders, mas apenas {len(orders)} foram criadas"
    print(f"✅ Invariante 1: {len(orders)} Orders criadas (>= 6)")
    
    # Invariante 2: Cada Order deve ter operações
    for order in orders:
        assert len(order.operations) > 0, f"Order {order.id} não tem operações!"
        print(f"  ✅ {order.id}: {len(order.operations)} operações")
    
    # Invariante 3: Não deve haver duplicados de (stage_index, rota, op_id) dentro de cada Order
    for order in orders:
        op_keys = {}
        for op_ref in order.operations:
            key = (op_ref.stage_index, op_ref.rota, op_ref.op_id)
            assert key not in op_keys, f"Order {order.id}: OpRef duplicado {key}"
            op_keys[key] = op_ref
        print(f"  ✅ {order.id}: Sem duplicados de OpRef")
    
    # Invariante 4: Cada OpRef deve ter pelo menos uma alternativa
    for order in orders:
        for op_ref in order.operations:
            assert len(op_ref.alternatives) > 0, (
                f"Order {order.id}, OpRef {op_ref.op_id} (Rota {op_ref.rota}, "
                f"Stage {op_ref.stage_index}) sem alternativas!"
            )
        print(f"  ✅ {order.id}: Todas as operações têm alternativas")
    
    print("\n✅ TODOS OS TESTES DO PARSER PASSARAM!")
    return True


def test_engine_invariants():
    """Testa invariantes do engine."""
    print("\n" + "="*80)
    print("TESTE 2: Invariantes do Engine")
    print("="*80)
    
    # Priorizar "Nikufra DadosProducao (2).xlsx" (com dados atualizados)
    file_path = 'app/data/Nikufra DadosProducao (2).xlsx'
    if not Path(file_path).exists():
        # Fallback para sem "(2)" se o principal não existir
        file_path = 'app/data/Nikufra DadosProducao.xlsx'
        if not Path(file_path).exists():
            print(f"❌ Ficheiro não encontrado: Nikufra DadosProducao (2).xlsx ou Nikufra DadosProducao.xlsx")
            return False
        else:
            print(f"⚠️ Usando ficheiro alternativo (sem '(2)'): {file_path}")
    else:
        print(f"✅ Usando ficheiro principal (2): {file_path}")
    
    orders = parse_dadosnikufra2(file_path)
    engine = APSEngine()
    plan = engine.build_schedule(orders, horizon_hours=24, start_time=datetime.utcnow())
    
    # Invariante 1: Baseline deve processar todas as Orders
    baseline_ops = plan.baseline.operations
    baseline_orders = set(op.order_id for op in baseline_ops)
    assert len(baseline_orders) == len(orders), (
        f"Baseline processou {len(baseline_orders)} Orders, mas esperava {len(orders)}"
    )
    print(f"✅ Invariante 1 (Baseline): {len(baseline_orders)} Orders processadas")
    
    # Invariante 2: Optimized deve processar todas as Orders (ou pelo menos a maioria)
    optimized_ops = plan.optimized.operations
    optimized_orders = set(op.order_id for op in optimized_ops)
    assert len(optimized_orders) >= len(orders) * 0.8, (
        f"Optimized processou apenas {len(optimized_orders)} Orders de {len(orders)} "
        f"({len(optimized_orders)/len(orders)*100:.1f}%)"
    )
    print(f"✅ Invariante 2 (Optimized): {len(optimized_orders)} Orders processadas")
    
    # Invariante 3: Não deve haver duplicados (mesma operação em múltiplas máquinas)
    for result_name, operations in [("Baseline", baseline_ops), ("Optimized", optimized_ops)]:
        op_keys = {}
        duplicates = []
        for op in operations:
            key = (op.order_id, op.op_ref.op_id, op.op_ref.rota, op.op_ref.stage_index)
            if key in op_keys:
                duplicates.append((key, op_keys[key].maquina_id, op.maquina_id))
            op_keys[key] = op
        
        assert len(duplicates) == 0, (
            f"{result_name}: {len(duplicates)} duplicados encontrados! "
            f"Exemplo: {duplicates[0] if duplicates else 'N/A'}"
        )
        print(f"✅ Invariante 3 ({result_name}): Sem duplicados")
    
    # Invariante 4: Cada ScheduledOperation deve ter alternative_chosen
    for result_name, operations in [("Baseline", baseline_ops), ("Optimized", optimized_ops)]:
        ops_sem_alt = [op for op in operations if not op.alternative_chosen]
        assert len(ops_sem_alt) == 0, (
            f"{result_name}: {len(ops_sem_alt)} operações sem alternative_chosen!"
        )
        print(f"✅ Invariante 4 ({result_name}): Todas as operações têm alternative_chosen")
    
    # Invariante 5: Verificar que operações de diferentes artigos aparecem
    baseline_artigos = set(op.order_id.replace("ORD-", "") for op in baseline_ops)
    optimized_artigos = set(op.order_id.replace("ORD-", "") for op in optimized_ops)
    
    assert len(baseline_artigos) >= 6, (
        f"Baseline: Esperava pelo menos 6 artigos diferentes, mas apenas {len(baseline_artigos)} encontrados: {baseline_artigos}"
    )
    print(f"✅ Invariante 5 (Baseline): {len(baseline_artigos)} artigos diferentes: {sorted(baseline_artigos)}")
    
    assert len(optimized_artigos) >= 4, (
        f"Optimized: Esperava pelo menos 4 artigos diferentes, mas apenas {len(optimized_artigos)} encontrados: {optimized_artigos}"
    )
    print(f"✅ Invariante 5 (Optimized): {len(optimized_artigos)} artigos diferentes: {sorted(optimized_artigos)}")
    
    print("\n✅ TODOS OS TESTES DO ENGINE PASSARAM!")
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTES DE INVARIANTES DO APS - ProdPlan 4.0")
    print("="*80)
    
    try:
        parser_ok = test_parser_invariants()
        engine_ok = test_engine_invariants()
        
        if parser_ok and engine_ok:
            print("\n" + "="*80)
            print("✅ TODOS OS TESTES PASSARAM!")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("❌ ALGUNS TESTES FALHARAM")
            print("="*80)
    except AssertionError as e:
        print(f"\n❌ ERRO: {e}")
    except Exception as e:
        print(f"\n❌ ERRO INESPERADO: {e}")
        import traceback
        traceback.print_exc()

