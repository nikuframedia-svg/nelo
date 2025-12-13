# PredictiveCare Gap Analysis Report

**Generated:** 2025-12-11  
**Module:** Digital Twin / PredictiveCare  
**Contract:** 17

---

## Executive Summary

O sistema ProdPlan 4.0 já tem uma base sólida de funcionalidades para manutenção preditiva através do Digital Twin (SHI-DT, RUL, PdM-IPS). Este relatório identifica gaps e extensões necessárias para um módulo PredictiveCare completo.

---

## Feature Checklist

### 1. Recolha de dados de sensores IoT

| Status | Componente | Localização | Notas |
|--------|------------|-------------|-------|
| ✅ COBERTO | SensorSnapshot model | `health_indicator_cvae.py` | Suporta vibração, temperatura, corrente, pressão |
| ⚠️ PARCIAL | Ingestão IoT | `smart_inventory/iot_ingestion.py` | Existe mas é para **stock events**, não sensores de máquina |
| ❌ EM FALTA | Tabela dt_sensor_readings | - | Precisa de persistência SQL |
| ❌ EM FALTA | API POST /digital-twin/iot/readings | - | Endpoint para ingestão |

**Ação:** Criar `digital_twin/iot_ingestion.py` com modelo, serviço e API específicos para sensores de máquina.

---

### 2. Modelos ML para degradação e RUL

| Status | Componente | Localização | Notas |
|--------|------------|-------------|-------|
| ✅ COBERTO | CVAE para Health Index | `health_indicator_cvae.py` | PyTorch, multi-sensor |
| ✅ COBERTO | Health Index calculation | `shi_dt.py` | H(t) = 100 * exp(-α * E_rec) |
| ✅ COBERTO | BaseRulEstimator | `rul_estimator.py` | Exponential/Linear + Monte Carlo |
| ✅ COBERTO | DeepSurvRulEstimator | `rul_estimator.py` | Deep Survival Analysis (stub) |
| ⚠️ PARCIAL | Anomaly detection | `health_indicator_cvae.py` | Implícito via reconstruction error, mas sem score dedicado |

**Ação:** Adicionar `anomaly_score` explícito no resultado do Health Index.

---

### 3. Integração com CMMS / Ordens de manutenção

| Status | Componente | Localização | Notas |
|--------|------------|-------------|-------|
| ❌ EM FALTA | Tabela maintenance_work_orders | - | Não existe |
| ❌ EM FALTA | MaintenanceWorkOrder model | - | Não existe |
| ❌ EM FALTA | PredictiveCareMaintenanceBridge | - | Geração automática de ordens |
| ❌ EM FALTA | API de manutenção | - | CRUD de ordens |
| ❌ EM FALTA | Sync com CMMS externo | - | Preparar stub para Odoo/SAP PM |

**Ação:** Criar `backend/maintenance/` com models, service e API.

---

### 4. Agendamento inteligente de manutenção

| Status | Componente | Localização | Notas |
|--------|------------|-------------|-------|
| ✅ COBERTO | Penalização RUL no scheduling | `rul_integration_scheduler.py` | PdM-IPS implementado |
| ✅ COBERTO | Redistribuição de carga | `rul_integration_scheduler.py` | Automático |
| ⚠️ PARCIAL | Sugestão de janelas | `rul_integration_scheduler.py` | Tem lógica mas não retorna janelas ótimas explícitas |
| ❌ EM FALTA | MaintenanceWindowSuggestion | - | Estrutura dedicada com impacto no plano |

**Ação:** Criar função `suggest_maintenance_window()` que retorna janelas ótimas.

---

### 5. Otimização de peças sobressalentes

| Status | Componente | Localização | Notas |
|--------|------------|-------------|-------|
| ❌ EM FALTA | Tabela spare_parts | - | Não existe |
| ❌ EM FALTA | SmartSpareForecastService | - | Previsão de substituição |
| ❌ EM FALTA | Integração MRP | - | Incluir peças sobressalentes na demanda MRP |

**Ação:** Criar `smart_inventory/spares_models.py` e serviço de forecast.

---

### 6. Priorização por risco

| Status | Componente | Localização | Notas |
|--------|------------|-------------|-------|
| ✅ COBERTO | HealthStatus enum | `rul_estimator.py` | HEALTHY/DEGRADED/WARNING/CRITICAL |
| ⚠️ PARCIAL | Risk scoring | `shi_dt.py` | Tem HI mas não probabilidade de falha a 7/30 dias |
| ❌ EM FALTA | MachinePredictiveState agregado | - | Estrutura única com SHI+RUL+Risk |

**Ação:** Criar `PredictiveCareService` com `MachinePredictiveState`.

---

### 7. Interface clara (UI)

| Status | Componente | Localização | Notas |
|--------|------------|-------------|-------|
| ✅ COBERTO | MachinesPanel | `frontend/components/MachinesPanel.tsx` | Contrato 16 criou aba Máquinas |
| ⚠️ PARCIAL | Mapa de máquinas | `MachinesPanel.tsx` | Mostra SHI/RUL mas usa mocks |
| ❌ EM FALTA | Ordens de manutenção na UI | - | Não implementado |
| ❌ EM FALTA | Peças sobressalentes na UI | - | Não implementado |
| ❌ EM FALTA | ROI metrics | - | Placeholder |

**Ação:** Enriquecer `MachinesPanel.tsx` com dados reais e novas secções.

---

## Resumo de Implementação

| Fase | Status | Prioridade |
|------|--------|------------|
| FASE 1: IoT Ingestion | ❌ A FAZER | ALTA |
| FASE 2: PredictiveCareService | ❌ A FAZER | ALTA |
| FASE 3: CMMS / Work Orders | ❌ A FAZER | ALTA |
| FASE 4: Maintenance Scheduling | ⚠️ ESTENDER | MÉDIA |
| FASE 5: Spare Parts | ❌ A FAZER | MÉDIA |
| FASE 6: UI Enhancements | ⚠️ ESTENDER | MÉDIA |
| FASE 7: R&D Integration | ❌ A FAZER | BAIXA |

---

## Arquitetura Proposta

```
backend/
├── digital_twin/
│   ├── iot_ingestion.py        # NOVO: Ingestão de sensores
│   ├── predictive_care.py      # NOVO: PredictiveCareService
│   └── (existentes: shi_dt, rul_estimator, rul_integration_scheduler)
│
├── maintenance/                 # NOVO MÓDULO
│   ├── __init__.py
│   ├── models.py               # SQLAlchemy: maintenance_work_orders
│   ├── predictivecare_bridge.py # Geração automática de ordens
│   └── api.py                  # CRUD endpoints
│
└── smart_inventory/
    └── spares_models.py        # NOVO: Peças sobressalentes
```

---

## Conclusão

O sistema já tem ~60% das funcionalidades PredictiveCare implementadas através do SHI-DT e RUL.
Os principais gaps são:
1. Persistência e API de dados de sensores
2. Módulo de ordens de manutenção
3. Previsão de peças sobressalentes
4. Integração completa na UI

Todas estas extensões podem ser feitas sem alterar a arquitetura base existente.


