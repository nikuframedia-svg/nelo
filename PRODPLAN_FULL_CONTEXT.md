# ProdPlan 4.0 - Contexto Completo para LLM

## 📋 Visão Geral

**ProdPlan 4.0** é um sistema avançado de **APS (Advanced Planning & Scheduling)** para indústria, desenvolvido com foco em **Indústria 5.0** (resiliência, human-centric, sustentabilidade). O sistema integra inteligência artificial, análise causal, digital twins, e passaportes digitais de produto (DPP).

**Stack Tecnológico:**
- **Backend:** Python 3.9+, FastAPI, SQLAlchemy, PyTorch
- **Frontend:** React 18, TypeScript, TailwindCSS, Framer Motion, React Query
- **Base de Dados:** SQLite (Duplios), CSV/JSON (dados de planeamento)

---

## 🏗️ ARQUITETURA DE MÓDULOS

### Estrutura de Navegação Principal

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRODPLAN 4.0                                  │
├─────────────────────────────────────────────────────────────────────┤
│  MÓDULOS PRINCIPAIS (3)            │  UTILITÁRIOS (4)               │
│  ├─ 🏭 Prodplan                    │  ├─ What-If + ZDM              │
│  ├─ 📦 SmartInventory              │  ├─ 🔗 Causal Analysis         │
│  └─ 📋 Duplios (DPP)               │  ├─ Chat (IA)                  │
│                                     │  └─ 🔬 R&D                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🏭 MÓDULO 1: PRODPLAN

**Descrição:** Cockpit central de planeamento e análise de produção.

### 1.1 Submódulo: Planeamento
**Ficheiros:** `backend/scheduler.py`, `frontend/src/pages/Planning.tsx`

**Funcionalidades:**
- **Gantt Chart Interativo:** Visualização de operações por máquina/artigo/rota
- **Modos de Scheduling:**
  - Baseline (heurística FIFO/SPT)
  - Chained (operações encadeadas)
  - MILP (otimização matemática - TODO)
  - DRL (Deep Reinforcement Learning - TODO)
- **Filtros Avançados:** Por recurso, produto, período
- **KPIs de Plano:**
  - Makespan (tempo total)
  - Tardiness (atrasos)
  - Utilização de máquinas
  - On-Time Delivery (OTD)

**API Endpoints:**
```
POST /plan           → Gera plano de produção
GET  /plan           → Retorna plano atual
GET  /plan/kpis      → KPIs do plano
GET  /plan/timeline  → Dados para timeline
```

### 1.2 Submódulo: Dashboards & Relatórios
**Ficheiros:** `backend/dashboards/`, `frontend/src/pages/Dashboards.tsx`

**Funcionalidades:**
- **Heatmap de Utilização:** Carga por máquina/hora
- **Dashboard de Operadores:** Performance, saturação, skills
- **OEE por Máquina:** Disponibilidade × Performance × Qualidade
- **Performance por Célula:** Throughput, WIP, lead time
- **Projeção de Capacidade:** 12 meses, gap analysis

**API Endpoints:**
```
GET /dashboards/utilization-heatmap
GET /dashboards/operator
GET /dashboards/machine-oee
GET /dashboards/cell-performance
GET /dashboards/capacity-projection
GET /dashboards/summary
```

### 1.3 Submódulo: Colaboradores (Workforce)
**Ficheiros:** `backend/workforce/`, `frontend/src/pages/WorkforcePerformance.tsx`

**Funcionalidades:**
- **Produtividade Individual:** Peças/hora, eficiência
- **Matriz de Competências:** Skills por operador
- **Análise de Saturação:** Carga de trabalho
- **Previsão de Performance:** ML para tendências
- **Recomendações de Alocação:** Sugestões de distribuição

### 1.4 Submódulo: Gargalos (Bottlenecks)
**Ficheiros:** `backend/scheduler.py`, `frontend/src/pages/Bottlenecks.tsx`

**Funcionalidades:**
- **Deteção Automática:** Máquina com maior carga
- **Ranking de Carga:** Todas as máquinas ordenadas
- **Histórico de Gargalos:** Evolução ao longo do tempo
- **Sugestões de Mitigação:** Redistribuição de carga

**API Endpoints:**
```
GET /bottleneck → Retorna máquina gargalo e estatísticas
```

### 1.5 Submódulo: Sugestões IA
**Ficheiros:** `backend/suggestions_engine.py`, `frontend/src/pages/Suggestions.tsx`

**Funcionalidades:**
- **Geração Automática de Sugestões:**
  - Otimização de sequência
  - Redistribuição de carga
  - Antecipação de manutenção
  - Ajuste de prioridades
- **Priorização:** Por impacto estimado
- **Aprovação/Rejeição:** Workflow de ações

**API Endpoints:**
```
GET  /suggestions          → Lista sugestões
POST /actions/propose      → Propõe ação
POST /actions/{id}/approve → Aprova ação
POST /actions/{id}/reject  → Rejeita ação
```

### 1.6 Submódulo: Digital Twin (PdM-IPS)
**Ficheiros:** `backend/digital_twin/`, `frontend/src/pages/DigitalTwin.tsx`

**Descrição:** Digital Twin para Prognóstico de Vida Útil Remanescente (RUL) por máquina.

**Componentes:**
- **CVAE (Conditional VAE):** Extração de Health Indicators
- **RUL Estimator:** Estimação de vida útil com incerteza
- **Scheduler Integration:** Penalização de máquinas críticas

**Funcionalidades:**
- **Health Index (HI):** Score 0-1 por máquina
- **RUL Estimation:** Horas restantes com intervalo de confiança
- **Status de Saúde:** HEALTHY, DEGRADED, WARNING, CRITICAL
- **Ajuste de Plano:** Evitar operações críticas em máquinas em risco
- **Recomendações:** Manutenção preventiva

**API Endpoints:**
```
GET  /digital-twin/health
GET  /digital-twin/dashboard
GET  /digital-twin/machines
GET  /digital-twin/machine/{id}
GET  /digital-twin/rul-penalties
POST /digital-twin/adjust-plan
```

---

## 📦 MÓDULO 2: SMARTINVENTORY

**Ficheiros:** `backend/smart_inventory/`, `frontend/src/pages/SmartInventory.tsx`

**Descrição:** Motor de inventário ultra-avançado com Digital Twin, previsão de procura, e sugestões inteligentes.

### 2.1 Digital Twin de Inventário
**Ficheiro:** `stock_state.py`

**Estrutura de Dados:**
```python
StockState:
  - quantities: Dict[warehouse_id, Dict[sku, WarehouseStock]]
  - WarehouseStock:
    - on_hand: float      # Em stock físico
    - committed: float    # Reservado para ordens
    - in_transit: float   # Em trânsito
    - available: float    # Disponível = on_hand - committed
```

### 2.2 Ingestão IoT
**Ficheiro:** `iot_ingestion.py`

**Tipos de Eventos:**
- RFID scan
- Vision (câmaras)
- Manual scan
- ERP sync

### 2.3 Previsão de Procura
**Ficheiro:** `demand_forecasting.py`

**Modelos:**
- **MVP:** ARIMA, Prophet
- **Avançados (stubs):** N-BEATS, Non-Stationary Transformers, D-Linear
- **Métricas:** MAPE, MAE, RMSE, SNR (Signal-to-Noise Ratio)

### 2.4 ROP Dinâmico
**Ficheiro:** `rop_engine.py`

**Cálculos:**
- **Reorder Point (ROP):** lead_time × avg_demand + safety_stock
- **Safety Stock:** z_score × std_dev × sqrt(lead_time)
- **Risco 30 dias:** Simulação Monte Carlo para probabilidade de ruptura
- **Cobertura:** Dias de stock disponível

### 2.5 Classificação ABC/XYZ
**Funcionalidade:** Matriz de classificação por valor e variabilidade

### 2.6 Sugestões de Inventário
**Ficheiro:** `suggestion_engine.py`

**Tipos:**
- `BUY`: Comprar mais stock
- `TRANSFER`: Transferir entre armazéns
- `REDUCE`: Reduzir stock excessivo
- `RISK_ALERT`: Alerta de ruptura iminente
- `PRICE_ALERT`: Oportunidade de preço

**API Endpoints:**
```
GET /inventory/stock
GET /inventory/forecast/{sku}
GET /inventory/rop/{sku}
GET /inventory/suggestions
POST /inventory/optimize
```

---

## 📋 MÓDULO 3: DUPLIOS (Digital Product Passport)

**Ficheiros:** `backend/duplios/`, `frontend/src/pages/Duplios.tsx`

**Descrição:** Sistema de Passaporte Digital de Produto (DPP) conforme ESPR (Ecodesign for Sustainable Products Regulation).

### 3.1 Modelo de Dados DPP

```python
DPP:
  # Identificação
  - dpp_id: UUID
  - gtin: str (obrigatório)
  - serial_or_lot: str
  - product_name: str
  - product_category: str
  - manufacturer_name: str
  - manufacturer_eori: str
  - manufacturing_site_id: str
  - country_of_origin: str
  
  # Composição
  - materials: List[Material]
    - material_name, material_type, percentage, mass_kg
  - components: List[Component]
  
  # Impacto Ambiental
  - carbon_footprint_kg_co2eq: float
  - water_consumption_m3: float
  - energy_consumption_kwh: float
  - manufacturing_kg_co2eq, distribution_kg_co2eq, end_of_life_kg_co2eq
  
  # Circularidade
  - recycled_content_percent: float
  - recyclability_percent: float
  - durability_score: int (1-10)
  - reparability_score: int (1-10)
  - hazardous_substances: List[HazardousSubstance]
  
  # Verificação
  - certifications: List[Certification]
  - third_party_audits: List[Audit]
  - trust_index: float (0-100)
  - data_completeness_percent: float
  
  # Metadata
  - status: draft | validated | published
  - qr_public_url: str
```

### 3.2 Serviços

- **Trust Index Calculator:** Score de confiabilidade baseado em completude e verificação
- **Carbon Calculator:** Cálculo de pegada de carbono por fase
- **Compliance Engine:** Avaliação de conformidade ESPR, CBAM, CSRD
- **QR Code Generator:** Geração de QR para acesso ao DPP

### 3.3 Funcionalidades UI

- **DPP Builder:** Wizard multi-step para criar DPPs
- **DPP Viewer:** Visualização completa do passaporte
- **DPP List:** Lista com filtros e pesquisa
- **Dashboard:** Métricas agregadas, compliance, impacto
- **Export:** CSV, JSON, PDF

**API Endpoints:**
```
POST /duplios/dpp              → Criar DPP
GET  /duplios/dpp              → Listar DPPs
GET  /duplios/dpp/{id}         → Obter DPP
PUT  /duplios/dpp/{id}         → Atualizar DPP
POST /duplios/dpp/{id}/publish → Publicar DPP
GET  /duplios/dpp/{id}/qrcode  → Obter QR Code
GET  /duplios/dashboard        → Métricas agregadas
GET  /duplios/compliance       → Análise de compliance
```

---

## 🧪 MÓDULO 4: WHAT-IF + ZDM (Zero Disruption Manufacturing)

### 4.1 What-If Scenarios
**Ficheiros:** `backend/what_if_engine.py`, `frontend/src/pages/WhatIf.tsx`

**Funcionalidades:**
- **Descrição de Cenários:** Input em linguagem natural
- **Preview:** Análise do cenário descrito
- **Comparação:** Before/After KPIs

**API Endpoints:**
```
POST /whatif/describe → Descreve cenário
POST /whatif/compare  → Compara com baseline
```

### 4.2 ZDM Simulator (Zero Disruption Manufacturing)
**Ficheiros:** `backend/simulation/zdm/`, `frontend/src/pages/ZDMSimulator.tsx`

**Descrição:** Simulador de resiliência que testa o plano face a falhas e perturbações.

**Componentes:**

#### Geração de Cenários de Falha
```python
FailureType:
  - SUDDEN: Falha súbita (máquina para)
  - GRADUAL: Degradação (tempo de ciclo aumenta)
  - QUALITY: Defeitos (retrabalho)
  - MATERIAL: Falta de material
  - OPERATOR: Ausência de operador
```

#### Simulação
- Aplica falhas no cronograma
- Tenta estratégias de auto-recuperação
- Calcula métricas de impacto

#### Estratégias de Recuperação
```python
RecoveryStrategy:
  - LOCAL_REPLAN: Replaneamento local
  - VIP_PRIORITY: Priorizar encomendas VIP
  - ADD_SHIFT: Adicionar turno extra
  - CUT_LOWPRIORITY: Cortar baixa prioridade
  - REROUTE: Reencaminhar para máquinas alternativas
  - OUTSOURCE: Subcontratar
  - MAINTENANCE_URGENT: Manutenção urgente
```

#### Métricas
- **Resilience Score:** 0-100 (quanto maior, mais resiliente)
- **Recovery Time:** Tempo para recuperar
- **Throughput Loss:** Perda de produção
- **OTD Impact:** Impacto em entregas

**API Endpoints:**
```
GET  /zdm/health
GET  /zdm/dashboard
GET  /zdm/quick-check
POST /zdm/simulate
GET  /zdm/scenarios
GET  /zdm/recovery/{scenario_id}
```

---

## 🔗 MÓDULO 5: CAUSAL ANALYSIS (CCM - Causal Context Models)

**Ficheiros:** `backend/causal/`, `frontend/src/pages/CausalAnalysis.tsx`

**Descrição:** Análise causal para compreender trade-offs entre decisões de curto prazo e objetivos de longo prazo.

### 5.1 Grafo Causal

**Variáveis de Decisão (Tratamentos):**
```
- setup_frequency: Frequência de setups
- batch_size: Tamanho de lotes
- machine_load: Carga das máquinas
- night_shifts: Turnos noturnos
- overtime_hours: Horas extra
- maintenance_delay: Adiamento de manutenção
- priority_changes: Alterações de prioridade
```

**Outcomes:**
```
- energy_cost: Custo energético
- makespan: Tempo total de produção
- tardiness: Atrasos
- otd_rate: Taxa de entregas a tempo
- machine_wear: Desgaste das máquinas
- failure_prob: Probabilidade de falha
- operator_stress: Stress dos operadores
- quality_defects: Defeitos de qualidade
- production_stability: Estabilidade do plano
```

**Confounders (Contexto):**
```
- demand_volume: Volume de procura
- product_mix: Diversidade de produtos
- seasonality: Sazonalidade
- machine_age: Idade do equipamento
- workforce_experience: Experiência dos operadores
```

### 5.2 Funcionalidades

- **Estimação de Efeitos Causais:** ATE (Average Treatment Effect)
- **Análise de Trade-offs:** Efeitos positivos vs negativos
- **Identificação de Confounders:** Backdoor criterion
- **Insights Automáticos:**
  - Trade-offs identificados
  - Pontos de alavancagem
  - Riscos
  - Oportunidades
  - Interações não-óbvias

### 5.3 Interface de Perguntas

Permite perguntas em linguagem natural:
- "Se eu reduzir setups, o que acontece ao custo energético?"
- "Qual o impacto de aumentar turnos noturnos no stress?"
- "Adiar manutenção afeta a probabilidade de falhas?"

**API Endpoints:**
```
GET  /causal/health
GET  /causal/graph
GET  /causal/variables
GET  /causal/effect/{treatment}/{outcome}
GET  /causal/effects/outcome/{outcome}
GET  /causal/effects/treatment/{treatment}
GET  /causal/tradeoffs/{treatment}
GET  /causal/complexity
GET  /causal/insights
POST /causal/explain
GET  /causal/dashboard
```

---

## 💬 MÓDULO 6: CHAT (Assistente IA)

**Ficheiros:** `backend/qa_engine.py`, `frontend/src/pages/Chat.tsx`

**Funcionalidades:**
- **Q&A em Linguagem Natural:** Perguntas sobre o plano
- **Command Parsing:** Interpretação de comandos
- **Contexto:** Usa dados do plano atual

**API Endpoints:**
```
POST /ask → Pergunta ao assistente
```

---

## 🔬 MÓDULO 7: R&D (Research & Development)

**Ficheiros:** `frontend/src/pages/Research.tsx`

**Descrição:** Área para funcionalidades experimentais e investigação.

---

## 📊 DADOS E MODELOS

### Estrutura de Dados de Entrada

```
DataBundle:
  - orders: List[Order]
    - order_id, article_id, qty, due_date, priority
  - machines: List[Machine]
    - machine_id, name, capacity_per_hour
  - routes: List[Route]
    - article_id, route_id, operations: List[Operation]
  - plan_df: DataFrame (plano gerado)
```

### Ficheiros de Dados
```
data/
├── orders.json      # Encomendas
├── machines.json    # Máquinas
├── routes.json      # Rotas de produção
└── production_plan.csv  # Plano gerado
```

---

## 🔧 CONFIGURAÇÃO E EXECUÇÃO

### Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd factory-optimizer/frontend
npm install
npm run dev  # Desenvolvimento
npm run build  # Produção
```

### Variáveis de Ambiente
```
VITE_API_URL=http://127.0.0.1:8000
```

---

## 📈 MÉTRICAS E KPIs PRINCIPAIS

### Planeamento
- **Makespan:** Tempo total de produção
- **Tardiness:** Soma dos atrasos
- **OTD Rate:** % de entregas a tempo
- **Utilização:** % de uso das máquinas

### Inventário
- **Stock Coverage:** Dias de cobertura
- **ROP:** Ponto de reordenação
- **Risk 30d:** Probabilidade de ruptura
- **ABC/XYZ:** Classificação de SKUs

### Sustentabilidade (Duplios)
- **Carbon Footprint:** kg CO2eq
- **Recycled Content:** %
- **Trust Index:** 0-100

### Resiliência (ZDM)
- **Resilience Score:** 0-100
- **Recovery Rate:** % de cenários recuperados
- **Avg Recovery Time:** Horas

### Complexidade Causal (CCM)
- **Complexity Score:** 0-100
- **Trade-offs:** Número identificados
- **Leverage Points:** Pontos de alto impacto

---

## 🏛️ ARQUITETURA DE FICHEIROS

```
mvp geral/
├── backend/
│   ├── api.py                    # FastAPI principal
│   ├── scheduler.py              # Motor de scheduling
│   ├── data_loader.py            # Carregamento de dados
│   ├── suggestions_engine.py     # Sugestões IA
│   ├── qa_engine.py              # Q&A engine
│   ├── what_if_engine.py         # What-If engine
│   ├── actions_engine.py         # Gestão de ações
│   │
│   ├── smart_inventory/          # SmartInventory
│   │   ├── stock_state.py        # Digital Twin inventário
│   │   ├── iot_ingestion.py      # Ingestão IoT
│   │   ├── demand_forecasting.py # Previsão de procura
│   │   ├── rop_engine.py         # ROP dinâmico
│   │   ├── suggestion_engine.py  # Sugestões inventário
│   │   └── external_signals.py   # Sinais externos
│   │
│   ├── digital_twin/             # Digital Twin PdM-IPS
│   │   ├── health_indicator_cvae.py  # CVAE para HI
│   │   ├── rul_estimator.py          # Estimação RUL
│   │   └── rul_integration_scheduler.py  # Integração APS
│   │
│   ├── simulation/               # Simulação
│   │   └── zdm/                  # Zero Disruption Manufacturing
│   │       ├── failure_scenario_generator.py
│   │       ├── zdm_simulator.py
│   │       └── recovery_strategy_engine.py
│   │
│   ├── causal/                   # Análise Causal
│   │   ├── causal_graph_builder.py
│   │   ├── causal_effect_estimator.py
│   │   └── complexity_dashboard_engine.py
│   │
│   ├── duplios/                  # Digital Product Passport
│   │   ├── models.py
│   │   ├── schemas.py
│   │   ├── service.py
│   │   ├── api_duplios.py
│   │   ├── qrcode_service.py
│   │   ├── trust_index_stub.py
│   │   ├── carbon_calculator.py
│   │   └── compliance_engine.py
│   │
│   ├── dashboards/               # Dashboards
│   ├── workforce/                # Workforce
│   ├── ml/                       # Machine Learning
│   └── evaluation/               # Avaliação
│
├── factory-optimizer/
│   └── frontend/
│       └── src/
│           ├── App.tsx           # Routing principal
│           ├── pages/            # Páginas
│           │   ├── Prodplan.tsx
│           │   ├── SmartInventory.tsx
│           │   ├── Duplios.tsx
│           │   ├── WhatIf.tsx
│           │   ├── ZDMSimulator.tsx
│           │   ├── CausalAnalysis.tsx
│           │   ├── DigitalTwin.tsx
│           │   └── ...
│           ├── components/       # Componentes reutilizáveis
│           └── services/         # API clients
│
└── data/                         # Dados de exemplo
```

---

## 🚀 FUNCIONALIDADES AVANÇADAS PLANEJADAS (TODO)

- **MILP Scheduling:** Otimização matemática completa
- **DRL Scheduling:** Deep Reinforcement Learning
- **N-BEATS/NST Forecasting:** Modelos avançados de previsão
- **Blockchain DPP:** Anchoring em blockchain
- **Gap Filling AI:** Preenchimento automático de dados DPP
- **External Signals:** Integração com APIs de preços/notícias

---

## 📝 NOTAS PARA LLM

1. **O sistema é modular** - cada módulo pode funcionar independentemente
2. **Dados sintéticos** - o sistema gera dados de demonstração quando necessário
3. **API REST** - todas as funcionalidades são expostas via API
4. **React Query** - frontend usa caching e refetch automático
5. **Português PT-PT** - interface e mensagens em português
6. **Indústria 5.0** - foco em resiliência, sustentabilidade e factor humano



