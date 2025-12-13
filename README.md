# ProdPlan 4.0

Sistema industrial avançado para planeamento, produção, inventário inteligente, gestão de produtos e manutenção preditiva.

## 🏗️ Arquitetura

ProdPlan 4.0 é um sistema modular composto por:

- **Backend**: FastAPI (Python) com modelos ML/PyTorch
- **Frontend**: React + TypeScript + Vite
- **Base de Dados**: SQLite (desenvolvimento) / PostgreSQL (produção)

## 📦 Módulos Principais

### 1. **ProdPlan** - Planeamento & Produção
- APS/APS+ (scheduling complexo)
- Gantt interativo
- Gestão de ordens de produção
- Análise de gargalos
- Workforce analytics
- Máquinas & Manutenção (SHI-DT, PredictiveCare)

### 2. **SmartInventory** - Inventário Inteligente
- Stock em tempo real
- MRP completo (Material Requirements Planning)
- Forecast & ROP dinâmico
- Dados operacionais (ingestão Excel)
- Analytics avançados

### 3. **Duplios** - Passaportes Digitais de Produto
- PDM (Product Data Management)
- DPP (Digital Product Passport)
- LCA (Life Cycle Assessment)
- Compliance Radar (ESPR, CBAM, CSRD)
- Trust Index avançado
- Gap Filling Lite

### 4. **Digital Twin** - Gêmeos Digitais
- **SHI-DT**: Smart Health Index para máquinas (CVAE, RUL)
- **XAI-DT**: Explainable Digital Twin de produto (qualidade geométrica)
- IoT ingestion
- PredictiveCare (manutenção preditiva)

### 5. **Inteligência** - IA & Otimização
- Otimização matemática (MILP, CP-SAT, heurísticas)
- Análise causal
- What-If avançado
- ZDM (Zero Disruption Manufacturing)

### 6. **R&D** - Investigação
- Experimentos WP1-WP4
- Work Packages experimentais (WPX)
- Logging estruturado

## 🚀 Início Rápido

### Pré-requisitos

- Python 3.10+
- Node.js 18+
- pip / npm

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configurar variáveis de ambiente (criar .env)
# OPENAI_API_KEY=... (opcional, para chat)
# DATABASE_URL=sqlite:///factory_optimizer.db

# Executar servidor
python run_server.py
# ou
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Backend disponível em: `http://127.0.0.1:8000`

### Frontend

```bash
cd factory-optimizer/frontend
npm install

# Configurar API URL (criar .env.local)
# VITE_API_URL=http://127.0.0.1:8000

npm run dev
```

Frontend disponível em: `http://localhost:5173`

## 📁 Estrutura do Projeto

```
.
├── backend/              # Backend FastAPI
│   ├── api.py           # API principal
│   ├── scheduling/      # Motor de scheduling
│   ├── smart_inventory/ # MRP, Forecast, ROP
│   ├── duplios/         # DPP, PDM, Compliance
│   ├── digital_twin/     # SHI-DT, XAI-DT
│   ├── optimization/    # Otimização matemática
│   ├── intelligence/    # Causal, What-If
│   ├── rd/              # R&D experiments
│   └── ...
├── factory-optimizer/
│   └── frontend/        # Frontend React
├── data/                # Dados de exemplo
├── docs/                # Documentação
└── README.md
```

## 🔧 Configuração

### Variáveis de Ambiente (Backend)

Criar `backend/.env`:

```env
# API Keys (opcional)
OPENAI_API_KEY=sk-...

# Database
DATABASE_URL=sqlite:///factory_optimizer.db

# Logging
LOG_LEVEL=INFO
```

### Variáveis de Ambiente (Frontend)

Criar `factory-optimizer/frontend/.env.local`:

```env
VITE_API_URL=http://127.0.0.1:8000
```

## 📚 Documentação

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Arquitetura do sistema
- [MODULES.md](docs/MODULES.md) - Descrição detalhada dos módulos

## 🔒 Segurança

- **NUNCA** commitar ficheiros `.env`, tokens, chaves ou credenciais
- Usar variáveis de ambiente para configuração sensível
- Verificar `.gitignore` antes de commits

## 📝 Licença

[Definir licença]

## 🤝 Contribuição

[Instruções de contribuição]

---

**ProdPlan 4.0** - Sistema Industrial Avançado
