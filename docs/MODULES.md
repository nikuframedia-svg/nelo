# Módulos ProdPlan 4.0

## 1. ProdPlan - Planeamento & Produção

**Localização**: `backend/scheduling/`, `backend/planning/`, `frontend/src/pages/Prodplan.tsx`

**Funcionalidades**:
- APS/APS+ (Advanced Planning & Scheduling)
- Gantt interativo com drag & drop
- Gestão de ordens de produção
- Análise de gargalos em tempo real
- Workforce analytics (operadores, performance)
- Máquinas & Manutenção (integração com Digital Twin)

**Tecnologias**:
- OR-Tools (CP-SAT) para scheduling
- PyTorch para previsão de tempos
- React Query para gestão de estado

---

## 2. SmartInventory - Inventário Inteligente

**Localização**: `backend/smart_inventory/`, `frontend/src/pages/SmartInventory.tsx`

**Funcionalidades**:
- Stock em tempo real com alertas
- MRP completo (Material Requirements Planning)
- Forecast & ROP dinâmico (Re-order Point)
- Ingestão de dados operacionais (Excel)
- Analytics avançados (SNR, tendências)

**Tecnologias**:
- ARIMA, ETS, XGBoost para forecasting
- Lógica MRP multi-nível
- Pandas para processamento de dados

---

## 3. Duplios - Passaportes Digitais de Produto

**Localização**: `backend/duplios/`, `frontend/src/pages/Duplios.tsx`

**Funcionalidades**:
- PDM (Product Data Management): Items, Revisions, BOM, Routing
- DPP (Digital Product Passport): Identidade digital, QR codes
- LCA (Life Cycle Assessment): Cálculo de impacto ambiental
- Compliance Radar: ESPR, CBAM, CSRD
- Trust Index avançado (field-level, 0-100)
- Gap Filling Lite: Preenchimento automático de dados em falta

**Tecnologias**:
- SQLAlchemy para gestão de dados
- YAML para configuração de fatores
- QR code generation

---

## 4. Digital Twin - Gêmeos Digitais

**Localização**: `backend/digital_twin/`, `frontend/src/pages/DigitalTwin.tsx`

### 4.1. SHI-DT (Smart Health Index - Digital Twin)

**Funcionalidades**:
- CVAE (Convolutional Variational Autoencoder) para deteção de anomalias
- Health Index dinâmico (0-100)
- RUL (Remaining Useful Life) estimação
- Perfis operacionais dinâmicos
- IoT ingestion (sensores)

**Tecnologias**:
- PyTorch (CVAE, DeepSurv)
- OPC-UA, MQTT para IoT

### 4.2. XAI-DT (Explainable Digital Twin de Produto)

**Funcionalidades**:
- Alinhamento CAD vs Scan 3D (ICP)
- Campo de desvio geométrico
- Deviation Score global
- RCA (Root Cause Analysis) geométrica
- Sugestões de correção de processo

**Tecnologias**:
- Open3D para processamento 3D
- PyTorch para ML-based RCA
- PCA para análise de padrões

### 4.3. PredictiveCare

**Funcionalidades**:
- Integração com SHI-DT
- Criação automática de ordens de manutenção
- Agendamento inteligente (integração com ProdPlan)
- Previsão de peças sobressalentes
- Priorização por risco

---

## 5. Inteligência - IA & Otimização

**Localização**: `backend/optimization/`, `backend/causal/`, `frontend/src/pages/Intelligence.tsx`

### 5.1. Otimização Matemática

**Funcionalidades**:
- Previsão de tempos (setup, ciclo) via ML
- Modelos de capacidade real (OEE, eficiência)
- Golden Runs (identificação de performance ótima)
- Otimização de parâmetros (Bayesian, RL, GA)
- Scheduling otimizado (MILP, CP-SAT, heurísticas)
- What-If avançado (cenários)

**Tecnologias**:
- PyTorch (neural networks)
- OR-Tools (CP-SAT, MILP)
- BoTorch (Bayesian Optimization)
- NSGA-II (multi-objective)

### 5.2. Análise Causal

**Funcionalidades**:
- Construção de grafo causal
- Estimação de efeitos causais
- Identificação de causas raiz
- Dashboard de complexidade

**Tecnologias**:
- DoWhy, CausalML
- NetworkX para grafos

### 5.3. ZDM (Zero Disruption Manufacturing)

**Funcionalidades**:
- Simulação de cenários de falha
- Resilience Score
- Planos de recuperação
- Análise de riscos

---

## 6. R&D - Investigação

**Localização**: `backend/rd/`, `frontend/src/pages/Research.tsx`

**Funcionalidades**:
- Experimentos WP1-WP4 (work packages principais)
- Work Packages experimentais (WPX):
  - WPX_TRUST_EVOLUTION
  - WPX_GAP_FILLING
  - WPX_COMPLIANCE
  - WPX_PREDICTIVECARE
  - WPX_OPS_INGESTION
- Logging estruturado de eventos
- Análise de resultados

**Tecnologias**:
- SQLite para base de dados de experimentos
- JSON para logs estruturados

---

## 7. Work Instructions & Prevention Guard

**Localização**: `backend/shopfloor/`, `backend/quality/`

### 7.1. Work Instructions

**Funcionalidades**:
- Instruções passo-a-passo digitais
- Checklists integradas
- Visualização 3D (Three.js)
- Rastreabilidade de execução
- Suporte multilíngua

### 7.2. Prevention Guard

**Funcionalidades**:
- Validação PDM (BOM, Routing, Documentação)
- Shopfloor Guard (material, equipamento, parâmetros)
- Predictive Guard (ML para risco de defeito)
- Digital Poka-Yoke
- Exception Manager

**Tecnologias**:
- PyTorch (classificador de risco)
- Validação automática de regras

---

## Integrações

- **ERP/MES**: Preparado para conectores (SQL Server, APIs REST)
- **IoT**: OPC-UA, MQTT para sensores
- **CMMS**: Bridge para sistemas de manutenção externos
- **LCA Databases**: Preparado para Ecoinvent (futuro)

---

## Dependências Principais

### Backend
- FastAPI
- SQLAlchemy
- PyTorch
- pandas, numpy
- OR-Tools
- Pydantic

### Frontend
- React
- TypeScript
- Vite
- @tanstack/react-query
- Framer Motion
- Lucide React
