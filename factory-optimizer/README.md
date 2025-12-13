# ProdPlan 4.0 + SmartInventory

Webapp de análise e otimização de produção fabril com previsões, otimização de produção e gestão de chão de fábrica.

## Estrutura do Projeto

```
factory_optimizer/
├── backend/          # FastAPI backend
│   ├── app/
│   │   ├── api/      # API endpoints
│   │   ├── aps/      # APS Scheduler
│   │   ├── etl/      # ETL para .xlsx
│   │   ├── llm/      # Explicações LLM
│   │   └── ml/       # Modelos ML
│   ├── data/         # Dados .xlsx
│   └── models/       # Modelos treinados
└── frontend/         # React + TypeScript
    └── src/
        ├── components/
        ├── pages/
        └── theme/
```

## Instalação

### Backend

1. Instalar dependências:
```bash
cd backend
pip install -r requirements.txt
```

2. Configurar variáveis de ambiente (opcional para LLM):
```bash
cp .env.example .env
# Editar .env com suas chaves API
```

3. Executar servidor:
```bash
python -m app.main
# ou
uvicorn app.main:app --reload
```

O servidor estará disponível em `http://localhost:8000`

### Frontend

1. Instalar dependências:
```bash
cd frontend
npm install
```

2. Executar servidor de desenvolvimento:
```bash
npm run dev
```

A aplicação estará disponível em `http://localhost:5173`

## Funcionalidades

### 1. Planeamento APS
- Comparação "Antes vs Depois" do planeamento
- KPIs: OTD %, Lead time, Gargalo ativo, Horas de setup
- Gantt charts sincronizados
- Decisões tomadas com explicações

### 2. Gargalos & Overlap
- Heatmap de utilização de recursos
- Top 5 perdas com ações recomendadas
- Overlap aplicado por etapa
- Ganho de lead time

### 3. Inventário
- Matriz ABC/XYZ
- Cobertura por SKU
- ROP (Reorder Point) com serviço configurável
- Ações: Comprar agora / Acompanhar / Excesso

### 4. What-If
- Simulação VIP (ordem prioritária)
- Simulação Avaria (recurso indisponível)
- Impacto em KPIs
- Explicações das decisões

## Dados

Os dados devem ser colocados em `backend/data/`:
- `Nikufra DadosProducao.xlsx` (sheets: Roteiros, Staffing, Ordens)
- `Nikufra Stocks.xlsx` (sheets: Stocks_mov, Stocks_snap)

Se os ficheiros não existirem, o sistema criará dados de exemplo automaticamente.

## Tecnologias

- **Backend**: FastAPI, Python, scikit-learn, XGBoost
- **Frontend**: React, TypeScript, Vite, React Query
- **ML**: Regressão quantílica, Bandit contextual, Croston/TSB
- **LLM**: OpenAI (opcional, para explicações)

## Licença

MIT

