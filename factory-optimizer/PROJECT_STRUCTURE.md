# Estrutura do Projeto ProdPlan 4.0

## Backend (FastAPI)

### Módulos Principais

#### `app/etl/loader.py`
- Carrega dados de produção e stocks de ficheiros .xlsx
- Cria dados de exemplo se os ficheiros não existirem
- Gerencia tabelas: Roteiros, Staffing, Ordens, Stocks_mov, Stocks_snap

#### `app/aps/scheduler.py`
- Gera planos baseline (Antes) e otimizados (Depois)
- Aplica overlap, colagem de famílias, rotas alternativas
- Calcula KPIs: OTD %, Lead time, Gargalo ativo, Horas de setup

#### `app/ml/`
- **cycle_time.py**: Regressão quantílica (P50/P90) para tempo de ciclo
- **setup_time.py**: Predição de tempo de setup por família
- **bottlenecks.py**: Classificação de gargalos
- **routing.py**: Bandit contextual para escolha de rotas
- **inventory.py**: Croston/TSB para procura intermitente e ROP

#### `app/llm/explanations.py`
- Gera explicações usando LLM (OpenAI) ou heurísticas
- Explica decisões com números e trade-offs

#### `app/api/`
- **planning.py**: Endpoint `/api/planning/plano`
- **bottlenecks.py**: Endpoint `/api/bottlenecks/`
- **inventory.py**: Endpoints `/api/inventory/` e `/api/inventory/rop`
- **whatif.py**: Endpoints `/api/whatif/vip` e `/api/whatif/avaria`

## Frontend (React + TypeScript)

### Tema
- **Cores**: Fundo #0B0B0B, Cards #121212/#161616, Primário #00E676
- **Tipografia**: Inter, tamanhos 28/24/20 (títulos), 14-16 (corpo)
- **Espaçamento**: 8/12/16/24/32px
- **Border radius**: 16px (cards), 24px (highlights)

### Componentes

#### `components/KPICard.tsx`
- Exibe KPIs com título, valor, delta e unidade
- Suporta formato number, percentage, text

#### `components/GanttChart.tsx`
- Visualiza operações em timeline
- Mostra overlap, rotas alternativas
- Cores por setor (Transformação/Acabamentos)

#### `components/Heatmap.tsx`
- Heatmap de utilização de recursos
- Cores: verde (baixo), amarelo (médio), vermelho (alto)

#### `components/DataTable.tsx`
- Tabela com sticky header, zebra striping
- Suporta renderização customizada de colunas

### Páginas

#### `pages/Planning.tsx`
- 4 KPIs principais
- Gantt "Antes vs Depois" lado a lado
- Card "Decisões tomadas"
- Filtros: intervalo temporal, célula/linha

#### `pages/Bottlenecks.tsx`
- Heatmap recursos × horas
- Top 5 perdas com ações recomendadas
- Mini-cards: overlap aplicado, ganho lead time

#### `pages/Inventory.tsx`
- Matriz ABC/XYZ
- Tabela por SKU: stock, ADS-180, cobertura, ROP, ação
- Filtros: classe, pesquisa SKU

#### `pages/WhatIf.tsx`
- Simulação VIP: SKU, quantidade, prazo
- Simulação Avaria: recurso, de/até
- Resultados com impacto em KPIs

## Fluxo de Dados

1. **Carregamento**: ETL lê .xlsx → cria tabelas internas
2. **Planeamento**: APS gera planos usando modelos ML
3. **Explicações**: LLM gera explicações das decisões
4. **API**: Endpoints retornam JSON
5. **Frontend**: React Query busca dados → renderiza componentes

## Dados Esperados

### Nikufra DadosProducao.xlsx
- **Roteiros**: artigo, ordem_grupo, grupo_operacao, alternativa, maquinas_elegiveis, racio_pc_h, overlap_pct, pessoas
- **Staffing**: setor, num_pessoas, horario
- **Ordens**: SKU, quantidade, data_prometida, prioridade, recurso_preferido

### Nikufra Stocks.xlsx
- **Stocks_mov**: SKU, data, tipo, entradas, saidas, saldo
- **Stocks_snap**: SKU, data, stock_atual

## Execução

### Backend
```bash
cd backend
pip install -r requirements.txt
python run.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Notas

- Modelos ML são treinados automaticamente na primeira execução (dados sintéticos)
- Dados de exemplo são criados automaticamente se .xlsx não existirem
- LLM é opcional (usa heurísticas se não configurado)
- Viewport desktop: 1200-1440px

