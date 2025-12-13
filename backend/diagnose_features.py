#!/usr/bin/env python3
"""
Diagnóstico de Features - Verifica todas as features criadas
"""

import sys
import traceback
from pathlib import Path

print("=" * 80)
print("DIAGNÓSTICO DE FEATURES - Verificando Implementações")
print("=" * 80)
print()

# 1. OPS INGESTION
print("1. OPS INGESTION")
print("-" * 80)
try:
    from ops_ingestion.models import OpsRawOrder, OpsRawInventoryMove, OpsRawHR, OpsRawMachine
    print("✅ Models import OK")
except Exception as e:
    print(f"❌ Models: {e}")
    traceback.print_exc()

try:
    from ops_ingestion.schemas import OrderRowSchema, InventoryMoveRowSchema, HRRowSchema, MachineRowSchema
    print("✅ Schemas import OK")
except Exception as e:
    print(f"❌ Schemas: {e}")
    traceback.print_exc()

try:
    from ops_ingestion.excel_parser import parse_excel_orders
    print("✅ Excel parser import OK")
except Exception as e:
    print(f"❌ Excel parser: {e}")
    traceback.print_exc()

try:
    from ops_ingestion.services import get_ops_ingestion_service
    service = get_ops_ingestion_service()
    print("✅ Services import OK")
except Exception as e:
    print(f"❌ Services: {e}")
    traceback.print_exc()

try:
    from ops_ingestion.api import router
    print(f"✅ API router: {len(router.routes)} endpoints")
    for route in router.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            print(f"   - {list(route.methods)} {route.path}")
except Exception as e:
    print(f"❌ API: {e}")
    traceback.print_exc()

try:
    from ops_ingestion.data_quality import analyze_orders_quality
    print("✅ Data quality import OK")
except Exception as e:
    print(f"❌ Data quality: {e}")
    traceback.print_exc()

# Verificar dependências
try:
    import pandas as pd
    print("✅ pandas disponível")
except ImportError:
    print("❌ pandas NÃO disponível")

try:
    import openpyxl
    print("✅ openpyxl disponível")
except ImportError:
    print("❌ openpyxl NÃO disponível")

print()

# 2. COMPLIANCE RADAR
print("2. COMPLIANCE RADAR")
print("-" * 80)
try:
    from duplios.compliance_models import RegulationType, ComplianceStatus, ComplianceItemStatus, ComplianceRadarResult
    print("✅ Models import OK")
except Exception as e:
    print(f"❌ Models: {e}")
    traceback.print_exc()

try:
    from duplios.compliance_radar import get_compliance_radar_service
    service = get_compliance_radar_service()
    print("✅ Service import OK")
except Exception as e:
    print(f"❌ Service: {e}")
    traceback.print_exc()

try:
    from duplios.api_compliance import router
    print(f"✅ API router: {len(router.routes)} endpoints")
    for route in router.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            print(f"   - {list(route.methods)} {route.path}")
except Exception as e:
    print(f"❌ API: {e}")
    traceback.print_exc()

# Verificar compliance_rules.yaml
rules_path = Path("duplios/data/compliance_rules.yaml")
if rules_path.exists():
    print("✅ compliance_rules.yaml existe")
else:
    print("❌ compliance_rules.yaml NÃO existe")

print()

# 3. TRUST INDEX
print("3. TRUST INDEX")
print("-" * 80)
try:
    from duplios.trust_index_models import DataSourceType, FieldTrustMeta, DPPTrustResult
    print("✅ Models import OK")
except Exception as e:
    print(f"❌ Models: {e}")
    traceback.print_exc()

try:
    from duplios.trust_index_service import TrustIndexService
    print("✅ Service import OK")
except Exception as e:
    print(f"❌ Service: {e}")
    traceback.print_exc()

try:
    from duplios.api_trust_index import router
    print(f"✅ API router: {len(router.routes)} endpoints")
    for route in router.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            print(f"   - {list(route.methods)} {route.path}")
except Exception as e:
    print(f"❌ API: {e}")
    traceback.print_exc()

print()

# 4. GAP FILLING LITE
print("4. GAP FILLING LITE")
print("-" * 80)
try:
    from duplios.gap_filling_lite import GapFillingLiteService
    print("✅ Service import OK")
except Exception as e:
    print(f"❌ Service: {e}")
    traceback.print_exc()

try:
    from duplios.api_gap_filling import router
    print(f"✅ API router: {len(router.routes)} endpoints")
    for route in router.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            print(f"   - {list(route.methods)} {route.path}")
except Exception as e:
    print(f"❌ API: {e}")
    traceback.print_exc()

# Verificar gap_factors.yaml
factors_path = Path("duplios/data/gap_factors.yaml")
if factors_path.exists():
    print("✅ gap_factors.yaml existe")
else:
    print("❌ gap_factors.yaml NÃO existe")

print()

# 5. INTEGRAÇÃO NO API.PY
print("5. INTEGRAÇÃO NO API.PY")
print("-" * 80)
try:
    import api
    app = api.app
    
    # Verificar rotas registadas
    routes = [r.path for r in app.routes if hasattr(r, 'path')]
    
    ops_routes = [r for r in routes if 'ops-ingestion' in r]
    compliance_routes = [r for r in routes if 'compliance' in r]
    trust_routes = [r for r in routes if 'trust-index' in r]
    gap_routes = [r for r in routes if 'gap-fill' in r]
    
    print(f"✅ Ops Ingestion routes: {len(ops_routes)}")
    for r in ops_routes:
        print(f"   - {r}")
    
    print(f"✅ Compliance routes: {len(compliance_routes)}")
    for r in compliance_routes:
        print(f"   - {r}")
    
    print(f"✅ Trust Index routes: {len(trust_routes)}")
    for r in trust_routes:
        print(f"   - {r}")
    
    print(f"✅ Gap Filling routes: {len(gap_routes)}")
    for r in gap_routes:
        print(f"   - {r}")
        
except Exception as e:
    print(f"❌ Erro ao verificar api.py: {e}")
    traceback.print_exc()

print()

# 6. R&D INTEGRATION
print("6. R&D INTEGRATION")
print("-" * 80)
try:
    from rd.experiments_core import WorkPackage, log_experiment_event
    
    # Verificar se WPX_DATA_INGESTION existe
    if hasattr(WorkPackage, 'WPX_DATA_INGESTION'):
        print("✅ WPX_DATA_INGESTION definido")
    else:
        print("❌ WPX_DATA_INGESTION NÃO definido")
    
    if hasattr(WorkPackage, 'WPX_COMPLIANCE_EVOLUTION'):
        print("✅ WPX_COMPLIANCE_EVOLUTION definido")
    else:
        print("❌ WPX_COMPLIANCE_EVOLUTION NÃO definido")
        
    print("✅ R&D module import OK")
except Exception as e:
    print(f"❌ R&D: {e}")
    traceback.print_exc()

print()
print("=" * 80)
print("DIAGNÓSTICO COMPLETO")
print("=" * 80)


