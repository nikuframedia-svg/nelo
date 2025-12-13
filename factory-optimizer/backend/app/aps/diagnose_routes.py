"""
Script de diagnóstico para verificar se o problema está no backend ou frontend.

Verifica:
1. O que o backend calcula (rotas escolhidas)
2. O que o backend serializa no JSON
3. Se há discrepâncias
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from app.aps.cache import get_plan_cache
from app.etl.loader import get_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_routes():
    """Diagnostica rotas no plano calculado vs JSON serializado."""
    
    print("=" * 80)
    print("DIAGNÓSTICO DE ROTAS: Backend vs JSON")
    print("=" * 80)
    print()
    
    # Obter batch_id
    loader = get_loader()
    status = loader.get_status()
    batch_id = status.get('latest_batch_id') or status.get('batch_id')
    
    if not batch_id:
        print("❌ Nenhum batch_id encontrado")
        return
    
    print(f"📋 Batch ID: {batch_id}\n")
    
    # Obter plano do cache
    cache = get_plan_cache()
    plan = cache.get(batch_id, 8)
    
    if not plan:
        print("❌ Plano não encontrado no cache. Execute recalcular primeiro.")
        return
    
    if not plan.optimized:
        print("❌ Plano optimized não encontrado")
        return
    
    print("=" * 80)
    print("1. VERIFICAR ROTAS NO PLANO OTIMIZADO (objeto Python)")
    print("=" * 80)
    
    # Analisar rotas no objeto Python
    rotas_por_artigo = {}
    for op in plan.optimized.operations:
        artigo = op.order_id.replace('ORD-', '')
        rota = op.op_ref.rota if op.op_ref else 'SEM_ROTA'
        
        if artigo not in rotas_por_artigo:
            rotas_por_artigo[artigo] = {
                'rotas': [],
                'ops': []
            }
        
        rotas_por_artigo[artigo]['rotas'].append(rota)
        rotas_por_artigo[artigo]['ops'].append({
            'op_id': op.op_ref.op_id if op.op_ref else '?',
            'rota': rota,
            'maquina': op.maquina_id
        })
    
    print("\n📊 Rotas por artigo (objeto Python):")
    for artigo in sorted(rotas_por_artigo.keys()):
        rotas = rotas_por_artigo[artigo]['rotas']
        rotas_unicas = list(set(rotas))
        print(f"  {artigo}:")
        print(f"    Rotas encontradas: {rotas_unicas}")
        print(f"    Total operações: {len(rotas)}")
        print(f"    Distribuição: {rotas.count('A')} A, {rotas.count('B')} B")
        
        if len(rotas_unicas) == 1 and rotas_unicas[0] == 'B':
            print(f"    ⚠️ PROBLEMA: Só tem rota B!")
        elif len(rotas_unicas) == 1 and rotas_unicas[0] == 'A':
            print(f"    ✅ Só tem rota A (pode ser normal)")
        else:
            print(f"    ✅ Tem mistura A/B")
    
    print("\n" + "=" * 80)
    print("2. VERIFICAR ROTAS NO JSON SERIALIZADO")
    print("=" * 80)
    
    # Serializar plano
    plan_dict = plan.to_dict()
    optimized_ops = plan_dict.get('optimized', {}).get('operations', [])
    
    print(f"\n📊 Total de operações no JSON: {len(optimized_ops)}")
    
    rotas_json_por_artigo = {}
    for op in optimized_ops:
        artigo = op.get('artigo') or op.get('order_id', '').replace('ORD-', '')
        rota = op.get('rota')
        
        if not rota:
            print(f"  ⚠️ ATENÇÃO: Operação {op.get('order_id')}/{op.get('op_id')} SEM ROTA no JSON!")
        
        if artigo not in rotas_json_por_artigo:
            rotas_json_por_artigo[artigo] = []
        
        rotas_json_por_artigo[artigo].append(rota or 'MISSING')
    
    print("\n📊 Rotas por artigo (JSON serializado):")
    for artigo in sorted(rotas_json_por_artigo.keys()):
        rotas = rotas_json_por_artigo[artigo]
        rotas_unicas = list(set(rotas))
        print(f"  {artigo}:")
        print(f"    Rotas encontradas: {rotas_unicas}")
        print(f"    Total operações: {len(rotas)}")
        print(f"    Distribuição: {rotas.count('A')} A, {rotas.count('B')} B, {rotas.count('MISSING')} MISSING")
        
        if 'MISSING' in rotas_unicas:
            print(f"    ❌ PROBLEMA: Algumas operações não têm rota no JSON!")
        
        if len(rotas_unicas) == 1 and rotas_unicas[0] == 'B':
            print(f"    ⚠️ PROBLEMA: Só tem rota B no JSON!")
        elif len(rotas_unicas) == 1 and rotas_unicas[0] == 'A':
            print(f"    ✅ Só tem rota A no JSON (pode ser normal)")
        else:
            print(f"    ✅ Tem mistura A/B no JSON")
    
    print("\n" + "=" * 80)
    print("3. COMPARAR OBJETO PYTHON vs JSON")
    print("=" * 80)
    
    # Comparar
    problemas = []
    for artigo in sorted(set(list(rotas_por_artigo.keys()) + list(rotas_json_por_artigo.keys()))):
        rotas_python = set(rotas_por_artigo.get(artigo, {}).get('rotas', []))
        rotas_json = set(rotas_json_por_artigo.get(artigo, []))
        
        if rotas_python != rotas_json:
            problemas.append({
                'artigo': artigo,
                'python': rotas_python,
                'json': rotas_json
            })
            print(f"  ❌ {artigo}: DISCREPÂNCIA!")
            print(f"     Python: {rotas_python}")
            print(f"     JSON: {rotas_json}")
        else:
            print(f"  ✅ {artigo}: Consistente (Python = JSON)")
    
    print("\n" + "=" * 80)
    print("4. VERIFICAR PRIMEIRAS 10 OPERAÇÕES (detalhe)")
    print("=" * 80)
    
    print("\n📊 Objeto Python (primeiras 10):")
    for i, op in enumerate(plan.optimized.operations[:10]):
        artigo = op.order_id.replace('ORD-', '')
        rota = op.op_ref.rota if op.op_ref else 'SEM_ROTA'
        print(f"  {i+1}. {artigo}/{op.op_ref.op_id if op.op_ref else '?'}: rota={rota}, máquina={op.maquina_id}")
    
    print("\n📊 JSON (primeiras 10):")
    for i, op in enumerate(optimized_ops[:10]):
        artigo = op.get('artigo') or op.get('order_id', '').replace('ORD-', '')
        rota = op.get('rota', 'MISSING')
        print(f"  {i+1}. {artigo}/{op.get('op_id', '?')}: rota={rota}, máquina={op.get('maquina_id', '?')}")
    
    print("\n" + "=" * 80)
    print("RESUMO")
    print("=" * 80)
    
    total_ops_python = len(plan.optimized.operations)
    total_ops_json = len(optimized_ops)
    
    print(f"Total operações Python: {total_ops_python}")
    print(f"Total operações JSON: {total_ops_json}")
    
    if total_ops_python != total_ops_json:
        print(f"  ❌ PROBLEMA: Número de operações diferente!")
    
    if problemas:
        print(f"\n❌ Encontradas {len(problemas)} discrepâncias entre Python e JSON")
        print("   Isto indica problema na SERIALIZAÇÃO (backend)")
    else:
        print("\n✅ Python e JSON são consistentes")
        print("   Se o frontend mostra errado, o problema está no FRONTEND")
    
    # Verificar distribuição geral
    todas_rotas_python = [op.op_ref.rota for op in plan.optimized.operations if op.op_ref]
    todas_rotas_json = [op.get('rota') for op in optimized_ops if op.get('rota')]
    
    print(f"\n📊 Distribuição geral:")
    print(f"  Python: {todas_rotas_python.count('A')} A, {todas_rotas_python.count('B')} B")
    print(f"  JSON: {todas_rotas_json.count('A')} A, {todas_rotas_json.count('B')} B")
    
    if todas_rotas_python.count('B') == len(todas_rotas_python):
        print(f"  ❌ PROBLEMA BACKEND: Todas as rotas são B no objeto Python!")
    elif todas_rotas_json.count('B') == len(todas_rotas_json):
        print(f"  ❌ PROBLEMA SERIALIZAÇÃO: Todas as rotas são B no JSON (mas não no Python)!")
    else:
        print(f"  ✅ Há mistura de rotas")


if __name__ == "__main__":
    diagnose_routes()

