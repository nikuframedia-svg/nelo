"""
Teste rigoroso do APS V2 - verifica todos os invariantes obrigatórios.
"""
import logging
from datetime import datetime
from pathlib import Path

from app.aps.engine import APSEngine
from app.aps.parser import parse_dadosnikufra2

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_parser_invariants():
    """Testa invariantes do parser."""
    print("\n" + "="*80)
    print("TESTE 1: Invariantes do Parser")
    print("="*80)
    
    file_path = 'app/data/Nikufra DadosProducao (2).xlsx'
    if not Path(file_path).exists():
        print(f"❌ Ficheiro não encontrado: {file_path}")
        return False
    
    orders = parse_dadosnikufra2(file_path)
    
    # Invariante 1: Deve ter pelo menos 6 Orders
    assert len(orders) >= 6, f"Esperava pelo menos 6 Orders, mas apenas {len(orders)} foram criadas"
    print(f"✅ Invariante 1: {len(orders)} Orders criadas (>= 6)")
    
    # Invariante 2: Cada Order deve ter operações
    for order in orders:
        assert len(order.operations) > 0, f"Order {order.id} não tem operações!"
        print(f"  ✅ {order.id}: {len(order.operations)} OpRefs")
        
        # Invariante 3: Não deve haver duplicados de (stage_index, rota, op_id)
        op_keys = set()
        for op_ref in order.operations:
            key = (op_ref.stage_index, op_ref.rota, op_ref.op_id)
            assert key not in op_keys, f"DUPLICADO no parser: Order {order.id}, OpRef: {key}"
            op_keys.add(key)
        
        # Invariante 4: Todas as operações devem ter alternativas
        ops_sem_alt = [op for op in order.operations if not op.alternatives]
        assert not ops_sem_alt, f"Order {order.id}: {len(ops_sem_alt)} operações sem alternativas!"
        print(f"    ✅ Sem duplicados, todas têm alternativas")
    
    print("\n✅ TODOS OS TESTES DO PARSER PASSARAM!")
    return True


def test_engine_invariants():
    """Testa invariantes do engine."""
    print("\n" + "="*80)
    print("TESTE 2: Invariantes do Engine")
    print("="*80)
    
    file_path = 'app/data/Nikufra DadosProducao (2).xlsx'
    if not Path(file_path).exists():
        print(f"❌ Ficheiro não encontrado: {file_path}")
        return False
    
    orders = parse_dadosnikufra2(file_path)
    engine = APSEngine()
    plan = engine.build_schedule(orders, horizon_hours=24, start_time=datetime.utcnow())
    
    # Invariante 1: Todas as Orders devem ser processadas no baseline
    baseline_orders_processed = set(op.order_id for op in plan.baseline.operations)
    assert len(baseline_orders_processed) == len(orders), \
        f"Baseline: Esperava {len(orders)} Orders, mas processou {len(baseline_orders_processed)}"
    print(f"✅ Invariante 1 (Baseline): {len(baseline_orders_processed)} Orders processadas")
    
    # Invariante 2: Todas as Orders devem ser processadas no optimized
    optimized_orders_processed = set(op.order_id for op in plan.optimized.operations)
    assert len(optimized_orders_processed) == len(orders), \
        f"Optimized: Esperava {len(orders)} Orders, mas processou {len(optimized_orders_processed)}"
    print(f"✅ Invariante 2 (Optimized): {len(optimized_orders_processed)} Orders processadas")
    
    # Invariante 3: Não deve haver duplicados em PlanResult.operations
    baseline_op_keys = {}
    for op in plan.baseline.operations:
        key = (op.order_id, op.op_ref.op_id, op.op_ref.rota, op.op_ref.stage_index)
        assert key not in baseline_op_keys, f"DUPLICADO no baseline: {key} (máquinas: {baseline_op_keys[key].maquina_id} e {op.maquina_id})"
        baseline_op_keys[key] = op
    print(f"✅ Invariante 3 (Baseline): Sem duplicados ({len(baseline_op_keys)} operações únicas)")
    
    optimized_op_keys = {}
    for op in plan.optimized.operations:
        key = (op.order_id, op.op_ref.op_id, op.op_ref.rota, op.op_ref.stage_index)
        assert key not in optimized_op_keys, f"DUPLICADO no optimized: {key} (máquinas: {optimized_op_keys[key].maquina_id} e {op.maquina_id})"
        optimized_op_keys[key] = op
    print(f"✅ Invariante 3 (Optimized): Sem duplicados ({len(optimized_op_keys)} operações únicas)")
    
    # Invariante 4: Cada operação agendada deve ter uma alternativa escolhida
    for op in plan.baseline.operations:
        assert op.alternative_chosen is not None, f"Baseline: Operação {op.order_id}/{op.op_ref.op_id} sem alternative_chosen!"
        assert op.maquina_id == op.alternative_chosen.maquina_id, f"Baseline: Inconsistência - op.maquina_id={op.maquina_id} != alternative_chosen.maquina_id={op.alternative_chosen.maquina_id}"
    print(f"✅ Invariante 4 (Baseline): Todas as operações têm alternative_chosen válido")
    
    for op in plan.optimized.operations:
        assert op.alternative_chosen is not None, f"Optimized: Operação {op.order_id}/{op.op_ref.op_id} sem alternative_chosen!"
        assert op.maquina_id == op.alternative_chosen.maquina_id, f"Optimized: Inconsistência - op.maquina_id={op.maquina_id} != alternative_chosen.maquina_id={op.alternative_chosen.maquina_id}"
    print(f"✅ Invariante 4 (Optimized): Todas as operações têm alternative_chosen válido")
    
    # Invariante 5: Verificar que todos os artigos estão presentes
    all_articles = set(o.artigo for o in orders)
    baseline_articles = set(op.order_id.replace("ORD-", "") for op in plan.baseline.operations)
    optimized_articles = set(op.order_id.replace("ORD-", "") for op in plan.optimized.operations)
    
    assert all_articles == baseline_articles, f"Baseline: Faltam artigos: {all_articles - baseline_articles}"
    print(f"✅ Invariante 5 (Baseline): {len(baseline_articles)} artigos diferentes: {sorted(list(baseline_articles))}")
    
    assert all_articles == optimized_articles, f"Optimized: Faltam artigos: {all_articles - optimized_articles}"
    print(f"✅ Invariante 5 (Optimized): {len(optimized_articles)} artigos diferentes: {sorted(list(optimized_articles))}")
    
    # Invariante 6: Verificar que cada OpRef foi agendado no máximo UMA vez por Order
    for order in orders:
        baseline_ops_for_order = [op for op in plan.baseline.operations if op.order_id == order.id]
        optimized_ops_for_order = [op for op in plan.optimized.operations if op.order_id == order.id]
        
        baseline_op_refs = set((op.op_ref.op_id, op.op_ref.rota, op.op_ref.stage_index) for op in baseline_ops_for_order)
        optimized_op_refs = set((op.op_ref.op_id, op.op_ref.rota, op.op_ref.stage_index) for op in optimized_ops_for_order)
        
        # Cada OpRef deve aparecer no máximo uma vez
        assert len(baseline_op_refs) == len(baseline_ops_for_order), \
            f"Baseline Order {order.id}: {len(baseline_ops_for_order)} operações mas apenas {len(baseline_op_refs)} OpRefs únicos (há duplicados!)"
        assert len(optimized_op_refs) == len(optimized_ops_for_order), \
            f"Optimized Order {order.id}: {len(optimized_ops_for_order)} operações mas apenas {len(optimized_op_refs)} OpRefs únicos (há duplicados!)"
    
    print(f"✅ Invariante 6: Cada OpRef foi agendado no máximo UMA vez por Order")
    
    print("\n✅ TODOS OS TESTES DO ENGINE PASSARAM!")
    return True


if __name__ == "__main__":
    all_tests_passed = True
    if not test_parser_invariants():
        all_tests_passed = False
    if not test_engine_invariants():
        all_tests_passed = False
    
    print("\n" + "="*80)
    if all_tests_passed:
        print("✅ TODOS OS TESTES PASSARAM!")
    else:
        print("❌ ALGUNS TESTES FALHARAM!")
    print("="*80)

