# Quick Start Guide

## Pré-requisitos

- Python 3.9+
- Node.js 18+
- npm ou yarn

## Instalação Rápida

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
python run.py
```

O servidor estará em `http://localhost:8000`

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

A aplicação estará em `http://localhost:5173`

## Primeira Execução

1. O sistema criará automaticamente dados de exemplo se os ficheiros .xlsx não existirem
2. Aceda a `http://localhost:5173` no navegador
3. Navegue pelas páginas:
   - **Planeamento**: Visualiza planos "Antes vs Depois"
   - **Gargalos**: Analisa utilização de recursos
   - **Inventário**: Gerencia stocks e ROP
   - **What-If**: Simula cenários VIP e avarias

## Estrutura de Dados

Coloque os ficheiros .xlsx em `backend/data/`:

### Nikufra DadosProducao.xlsx
- Sheet "Roteiros": artigo, ordem_grupo, grupo_operacao, alternativa, maquinas_elegiveis, racio_pc_h, overlap_pct, pessoas
- Sheet "Staffing": setor, num_pessoas, horario
- Sheet "Ordens": SKU, quantidade, data_prometida, prioridade, recurso_preferido

### Nikufra Stocks.xlsx
- Sheet "Stocks_mov": SKU, data, tipo, entradas, saidas, saldo
- Sheet "Stocks_snap": SKU, data, stock_atual

## Configuração LLM (Opcional)

Para explicações LLM, configure no `.env`:
```
OPENAI_API_KEY=your_key_here
```

Sem a chave, o sistema usa explicações heurísticas.

## Troubleshooting

### Erro: Module not found
- Certifique-se de estar no diretório correto
- Instale as dependências: `pip install -r requirements.txt`

### Erro: Port already in use
- Mude a porta no `run.py` ou `vite.config.ts`

### Dados não aparecem
- Verifique se os ficheiros .xlsx estão em `backend/data/`
- O sistema criará dados de exemplo automaticamente

