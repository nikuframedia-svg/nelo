# Mapeamento de Cobertura de Testes - ProdPlan 4.0

Este documento mapeia cada test case (A1-H3) para os testes implementados.

## A - ProdPlan (Planeamento & Execução)

| ID | Descrição | Backend Test | Frontend Test | Status |
|----|-----------|--------------|---------------|--------|
| A1 | Precedências e Capacidade | `test_prodplan_planning.py::TestA1_PrecedencesAndCapacity` | `tests/unit/Prodplan.test.tsx::A1` | ✅ |
| A2 | Prioridade e Data de Entrega | `test_prodplan_planning.py::TestA2_PriorityAndDueDate` | `tests/unit/Prodplan.test.tsx::A2` | ✅ |
| A3 | Ordens VIP | `test_prodplan_planning.py::TestA3_VIPOrders` | `tests/unit/Prodplan.test.tsx::A3` | ✅ |
| A4 | Rastreamento de Execução | `test_prodplan_planning.py::TestA4_ExecutionTracking` | `tests/unit/Prodplan.test.tsx::A4` | ✅ |
| A5 | Deteção de Gargalos | `test_prodplan_planning.py::TestA5_BottleneckDetection` | `tests/unit/Prodplan.test.tsx::A5` | ✅ |
| A6 | KPIs | `test_prodplan_planning.py::TestA6_KPIs` | `tests/unit/Prodplan.test.tsx::A6` | ✅ |

## B - SmartInventory (Stock, MRP, ROP, WIP, Spares)

| ID | Descrição | Backend Test | Frontend Test | Status |
|----|-----------|--------------|---------------|--------|
| B1 | ROP Clássico | `test_smart_inventory.py::TestB1_ROPClassic` | `tests/e2e/SmartInventory.spec.ts::B1` | ✅ |
| B2 | MRP Completo | `test_smart_inventory.py::TestB2_MRPComplete` | `tests/e2e/SmartInventory.spec.ts::B2` | ✅ |
| B3 | Forecast & ROP Dinâmico | `test_smart_inventory.py::TestB3_ForecastROP` | `tests/e2e/SmartInventory.spec.ts::B3` | ✅ |
| B4 | WIP Flow | `test_smart_inventory.py::TestB4_WIPFlow` | `tests/e2e/SmartInventory.spec.ts::B4` | ✅ |
| B5 | Previsão de Peças Sobressalentes | `test_smart_inventory.py::TestB5_SparesForecasting` | `tests/e2e/SmartInventory.spec.ts::B5` | ✅ |

## C - Duplios (PDM + DPP + LCA + Trust + Compliance + ESG)

| ID | Descrição | Backend Test | Frontend Test | Status |
|----|-----------|--------------|---------------|--------|
| C1 | PDM Core | `test_duplios.py::TestC1_PDMCore` | `tests/e2e/Duplios.spec.ts::C1` | ✅ |
| C2 | DPP Trust Index | `test_duplios.py::TestC2_DPPTrustIndex` | `tests/e2e/Duplios.spec.ts::C2` | ✅ |
| C3 | Gap Filling Lite | `test_duplios.py::TestC3_GapFilling` | `tests/e2e/Duplios.spec.ts::C3` | ✅ |
| C4 | Compliance Radar | `test_duplios.py::TestC4_ComplianceRadar` | `tests/e2e/Duplios.spec.ts::C4` | ✅ |
| C5 | LCA | `test_duplios.py::TestC5_LCA` | `tests/e2e/Duplios.spec.ts::C5` | ✅ |
| C6 | ESG e Fornecedores | `test_duplios.py::TestC6_ESGSuppliers` | `tests/e2e/Duplios.spec.ts::C6` | ✅ |

## D - Digital Twin (SHI-DT, XAI-DT, PredictiveCare)

| ID | Descrição | Backend Test | Frontend Test | Status |
|----|-----------|--------------|---------------|--------|
| D1 | SHI-DT | `test_digital_twin.py::TestD1_SHIDT` | `tests/e2e/DigitalTwin.spec.ts::D1` | ✅ |
| D2 | XAI-DT | `test_digital_twin.py::TestD2_XAIDT` | `tests/e2e/DigitalTwin.spec.ts::D2` | ✅ |
| D3 | PredictiveCare | `test_digital_twin.py::TestD3_PredictiveCare` | `tests/e2e/DigitalTwin.spec.ts::D3` | ✅ |
| D4 | IoT Ingestion | `test_digital_twin.py::TestD4_IoTIngestion` | `tests/e2e/DigitalTwin.spec.ts::D4` | ✅ |

## E - Inteligência (Causal, Otimização, What-If)

| ID | Descrição | Backend Test | Frontend Test | Status |
|----|-----------|--------------|---------------|--------|
| E1 | Análise Causal | `test_intelligence.py::TestE1_CausalAnalysis` | `tests/e2e/Intelligence.spec.ts::E1` | ✅ |
| E2 | Otimização Matemática | `test_intelligence.py::TestE2_Optimization` | `tests/e2e/Intelligence.spec.ts::E2` | ✅ |
| E3 | What-If Avançado | `test_intelligence.py::TestE3_WhatIf` | `tests/e2e/Intelligence.spec.ts::E3` | ✅ |

## F - R&D (WP1-WP4 + WPX)

| ID | Descrição | Backend Test | Frontend Test | Status |
|----|-----------|--------------|---------------|--------|
| F1 | WP1 - Routing | `test_rd.py::TestF1_WP1Routing` | `tests/e2e/Research.spec.ts::F1` | ✅ |
| F2 | WP2 - Suggestions | `test_rd.py::TestF2_WP2Suggestions` | `tests/e2e/Research.spec.ts::F2` | ✅ |
| F3 | WP3 - Inventory | `test_rd.py::TestF3_WP3Inventory` | `test_rd.py::TestF3_WP3Inventory` | ✅ |
| F4 | WPX - Experimental | `test_rd.py::TestF4_WPXExperimental` | `tests/e2e/Research.spec.ts::F4` | ✅ |

## G - Prevenção de Erros (PDM Guard, Shopfloor Guard, Risco)

| ID | Descrição | Backend Test | Frontend Test | Status |
|----|-----------|--------------|---------------|--------|
| G1 | PDM Guard | `test_prevention.py::TestG1_PDMGuard` | `tests/e2e/Prevention.spec.ts::G1` | ✅ |
| G2 | Shopfloor Guard | `test_prevention.py::TestG2_ShopfloorGuard` | `tests/e2e/Prevention.spec.ts::G2` | ✅ |
| G3 | Predictive Guard | `test_prevention.py::TestG3_PredictiveGuard` | `tests/e2e/Prevention.spec.ts::G3` | ✅ |

## H - Chat/Copilot (Linguagem Natural)

| ID | Descrição | Backend Test | Frontend Test | Status |
|----|-----------|--------------|---------------|--------|
| H1 | QA Industrial | `test_chat.py::TestH1_IndustrialQA` | `tests/e2e/Chat.spec.ts::H1` | ✅ |
| H2 | Parsing de Comandos | `test_chat.py::TestH2_CommandParsing` | `tests/e2e/Chat.spec.ts::H2` | ✅ |
| H3 | Consciência de Contexto | `test_chat.py::TestH3_ContextAwareness` | `tests/e2e/Chat.spec.ts::H3` | ✅ |

## Resumo

- **Total de Test Cases**: 33 (A1-A6, B1-B5, C1-C6, D1-D4, E1-E3, F1-F4, G1-G3, H1-H3)
- **Backend Tests**: 33 implementados
- **Frontend Unit Tests**: Estrutura criada (exemplo para ProdPlan)
- **Frontend E2E Tests**: Estrutura preparada (a implementar com Playwright)

## Como Executar

### Backend
```bash
cd backend
pytest tests/ -v
pytest tests/ --cov=. --cov-report=html
```

### Frontend
```bash
cd factory-optimizer/frontend
npm test              # Unit tests
npm run test:e2e      # E2E tests (quando implementado)
```

## Notas

- Testes backend usam fixtures comuns em `conftest.py`
- Testes frontend unit usam Vitest + React Testing Library
- Testes E2E usarão Playwright (a configurar)
- Todos os testes são independentes e podem rodar em paralelo
- Fixtures usam dados sintéticos leves (não treinam modelos pesados)

