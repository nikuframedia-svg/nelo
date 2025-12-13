# Arquitetura ProdPlan 4.0

## Visão Geral

ProdPlan 4.0 é um sistema industrial modular construído com:

- **Backend**: FastAPI (Python 3.10+)
- **Frontend**: React + TypeScript + Vite
- **ML/AI**: PyTorch, scikit-learn, OR-Tools
- **Base de Dados**: SQLite (dev) / PostgreSQL (prod)

## Camadas da Aplicação

### 1. Camada de Apresentação (Frontend)
- **React + TypeScript**: Componentes modulares
- **React Query**: Gestão de estado e cache
- **Framer Motion**: Animações
- **Lucide React**: Ícones

### 2. Camada de API (Backend)
- **FastAPI**: Framework REST API
- **Pydantic**: Validação de dados
- **SQLAlchemy**: ORM para base de dados

### 3. Camada de Domínio (Backend)
- **Scheduling**: Motor APS/APS+
- **SmartInventory**: MRP, Forecast, ROP
- **Duplios**: DPP, PDM, Compliance
- **Digital Twin**: SHI-DT, XAI-DT
- **Optimization**: MILP, CP-SAT, heurísticas
- **Intelligence**: Causal, What-If, ZDM
- **R&D**: Experimentos e logging

### 4. Camada de Dados
- **SQLite**: Base de dados local (desenvolvimento)
- **PostgreSQL**: Base de dados produção (recomendado)
- **Ficheiros**: Excel, CSV (ingestão operacional)

### 5. Camada de ML/AI
- **PyTorch**: Deep Learning (CVAE, RUL, previsões)
- **scikit-learn**: Modelos clássicos
- **OR-Tools**: Otimização matemática
- **BoTorch**: Bayesian Optimization

## Princípios de Design

1. **Modularidade**: Cada módulo é independente e pode evoluir separadamente
2. **Separação de Responsabilidades**: Backend (lógica), Frontend (UI), ML (modelos)
3. **Fallback Robusto**: Modelos avançados têm versões BASE como fallback
4. **Extensibilidade**: Preparado para integração com ERP/MES externos
5. **R&D Integration**: Todos os módulos suportam logging estruturado para análise

## Fluxo de Dados

```
Frontend (React) 
    ↓ HTTP/REST
Backend API (FastAPI)
    ↓
Domain Services (Scheduling, Inventory, etc.)
    ↓
ML Engines (PyTorch, scikit-learn)
    ↓
Database (SQLite/PostgreSQL)
```

## Segurança

- Validação de inputs (Pydantic)
- Variáveis de ambiente para segredos
- Sem hardcoding de credenciais
- `.gitignore` configurado para excluir ficheiros sensíveis

## Performance

- **Backend**: Async/await para I/O não bloqueante
- **Frontend**: Code splitting e lazy loading
- **ML**: Modelos otimizados, cache de inferências
- **Database**: Índices apropriados, queries otimizadas

## Extensibilidade Futura

- Preparado para microserviços (cada módulo pode ser um serviço)
- API RESTful facilita integração
- Estrutura modular permite adicionar novos módulos
- R&D logging permite análise contínua e melhoria


