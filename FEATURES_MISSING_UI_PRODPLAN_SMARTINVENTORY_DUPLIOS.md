# 🔍 Análise: Features Backend sem UI - ProdPlan, SmartInventory, Duplios

## Resumo Executivo

Análise específica dos três módulos principais: **ProdPlan**, **SmartInventory** e **Duplios** para identificar funcionalidades implementadas no backend que ainda não têm interface de utilizador (UI) correspondente.

---

## 📊 PRODPLAN (Advanced Planning & Scheduling)

### ✅ Implementado na UI
- ✅ Planning modes (conventional, chained, short-term, long-term)
- ✅ Plan comparison
- ✅ Basic plan visualization
- ✅ What-If scenarios (básico)

### ❌ Faltando na UI

1. **Data Quality Analysis (SNR)**
   - Backend: `GET /plan/data_quality`
   - Frontend: Não implementado
   - **Falta**: 
     - Dashboard de qualidade de dados
     - SNR por máquina/operação
     - Alertas de baixa SNR
     - Recomendações de melhoria de dados
     - Visualização de variabilidade

2. **MILP Optimization**
   - Backend: `GET /plan/milp?time_limit=X&gap=Y`
   - Frontend: Não implementado
   - **Falta**: 
     - Interface para executar otimização MILP
     - Configuração de parâmetros (time_limit, gap, objetivos)
     - Comparação MILP vs heurística
     - Visualização de estatísticas de resolução

3. **Product Metrics & KPIs**
   - Backend: 
     - `GET /product/type-kpis`
     - `GET /product/{article_id}/kpis`
     - `GET /product/classification`
     - `POST /product/delivery-estimate`
     - `GET /product/delivery-estimates`
     - `GET /product/summary`
   - Frontend: Não implementado
   - **Falta**: 
     - Dashboard de KPIs por tipo de produto
     - KPIs detalhados por artigo
     - Classificação de produtos
     - Estimativas de entrega (deterministic, historical, ML)
     - Resumo executivo de produtos

4. **Workforce Analytics**
   - Backend:
     - `POST /workforce/forecast`
     - `POST /workforce/assign`
     - `GET /workforce/summary`
   - Frontend: Existe `ProdplanWorkforce.tsx` mas precisa verificar se usa todos os endpoints
   - **Falta**: 
     - Interface completa de forecast de workforce
     - Otimização de alocação (MILP vs heurística)
     - Resumo executivo de workforce

5. **Planning Instance Builder from Ops Ingestion**
   - Backend: `GET /ops-ingestion/planning-instance`
   - Frontend: Não integrado no ProdPlan
   - **Falta**: 
     - Botão/opção para construir plano a partir de dados operacionais importados
     - Preview de jobs/operations/machines antes de criar plano
     - Integração com Advanced Planning

6. **What-If Advanced Features**
   - Backend: `POST /what-if/describe`, `POST /what-if/compare`
   - Frontend: Existe `WhatIf.tsx` mas precisa verificar se usa todos os endpoints
   - **Falta**: 
     - Comparação avançada de cenários
     - Visualização de métricas lado a lado
     - Export de comparações

---

## 📦 SMARTINVENTORY

### ✅ Implementado na UI
- ✅ Stock real-time
- ✅ Forecast visualization
- ✅ ROP (Reorder Point)
- ✅ MRP Encomendas (básico)
- ✅ MRP Completo (tab completa com runs, orders, alerts)
- ✅ Dados Operacionais (WIP flow, ordens importadas)

### ❌ Faltando na UI

1. **MRP Parameters Management**
   - Backend: `GET /inventory/mrp/parameters`
   - Frontend: Não implementado
   - **Falta**: 
     - Interface para visualizar parâmetros MRP por SKU
     - Editar parâmetros (min_stock, max_stock, MOQ, múltiplo, scrap_rate, lead_time)
     - Validação de parâmetros
     - Import/export de parâmetros

2. **BOM Explosion Viewer**
   - Backend: `GET /inventory/mrp/bom/{product_id}?quantity=X`
   - Frontend: Não implementado
   - **Falta**: 
     - Visualização hierárquica de BOM explosion
     - Árvore de componentes com quantidades
     - Indicadores de lead time por nível
     - Identificação de componentes comprados vs fabricados

3. **MRP Forecast Integration**
   - Backend: `POST /mrp/forecast` (em api_mrp_complete.py)
   - Frontend: Não implementado
   - **Falta**: 
     - Interface para carregar dados de forecast
     - Upload de forecast (CSV/JSON)
     - Visualização de forecast vs demand real
     - Integração com MRP runs

4. **Item Plan Visualization**
   - Backend: `GET /mrp/runs/{run_id}/item-plan/{sku}`
   - Frontend: Não implementado (existe no MRP Completo mas pode melhorar)
   - **Falta**: 
     - Visualização detalhada de plano por item
     - Timeline de necessidades vs disponibilidade
     - Gráfico de stock projection
     - Alertas de ruptura visual

5. **MRP Reset/Clear**
   - Backend: `DELETE /mrp/reset`
   - Frontend: Não implementado
   - **Falta**: 
     - Botão para resetar serviço MRP
     - Confirmação antes de reset
     - Limpar histórico de runs

6. **Inventory Suggestions Advanced**
   - Backend: `GET /inventory/suggestions`
   - Frontend: Implementado mas pode melhorar
   - **Falta**: 
     - Filtros avançados de sugestões
     - Ações em batch (aprovar múltiplas sugestões)
     - Histórico de sugestões implementadas

---

## 🏷️ DUPLIOS

### ✅ Implementado na UI
- ✅ DPP CRUD (create, list, view, update, delete)
- ✅ DPP publish
- ✅ QR Code generation
- ✅ Trust Index (breakdown completo)
- ✅ Gap Filling Lite (execução e resultados)
- ✅ Compliance Radar (scores, gaps, actions)
- ✅ Export CSV/JSON
- ✅ Dashboard básico
- ✅ Analytics (compliance, carbon)

### ❌ Faltando na UI

1. **Dashboard Metrics Completo**
   - Backend: `GET /duplios/dashboard`
   - Frontend: Implementado mas pode expandir
   - **Falta**: 
     - Visualização mais rica de métricas
     - Gráficos de evolução temporal
     - Comparação entre categorias
     - Filtros avançados no dashboard

2. **Analytics Avançados**
   - Backend:
     - `GET /duplios/analytics/compliance`
     - `GET /duplios/analytics/carbon`
   - Frontend: Implementado mas básico
   - **Falta**: 
     - Gráficos interativos de compliance
     - Breakdown de carbono por categoria
     - Tendências ao longo do tempo
     - Comparação entre fabricantes

3. **Carbon Breakdown Detalhado**
   - Backend: `GET /duplios/dpp/{dpp_id}/carbon-breakdown`
   - Frontend: Não implementado
   - **Falta**: 
     - Visualização detalhada de breakdown de carbono
     - Contribuição por componente/material
     - Gráfico de Sankey ou similar
     - Comparação com benchmarks

4. **Trust Breakdown Detalhado**
   - Backend: `GET /duplios/dpp/{dpp_id}/trust-breakdown`
   - Frontend: Não implementado (Trust Index já mostra breakdown, mas este endpoint pode ter mais detalhes)
   - **Falta**: 
     - Visualização alternativa de trust breakdown
     - Histórico de evolução de trust
     - Comparação com outros DPPs

5. **DPP Recalculate UI**
   - Backend: `POST /duplios/dpp/{dpp_id}/recalculate`
   - Frontend: Não implementado (pode estar no DPPViewer mas não visível)
   - **Falta**: 
     - Botão claro para recalcular métricas
     - Feedback visual durante recálculo
     - Comparação antes/depois

6. **PDM Lite Integration (Items/Revisions)**
   - Backend:
     - `GET /duplios/items`
     - `POST /duplios/items`
     - `GET /duplios/items/{item_id}`
     - `GET /duplios/items/{item_id}/revisions`
     - `POST /duplios/items/{item_id}/revisions`
     - `POST /duplios/revisions/{revision_id}/release`
     - `GET /duplios/revisions/{revision_id}/bom`
     - `POST /duplios/revisions/{revision_id}/bom`
     - `GET /duplios/revisions/{revision_id}/routing`
     - `POST /duplios/revisions/{revision_id}/routing`
   - Frontend: Implementado em `Duplios.tsx` mas pode melhorar
   - **Falta**: 
     - Interface mais completa de gestão de items/revisions
     - Editor visual de BOM
     - Editor visual de Routing
     - Workflow visual de release

7. **Identity Service Integration**
   - Backend:
     - `POST /duplios/identity/ingest`
     - `POST /duplios/identity/verify`
     - `GET /duplios/identity/{revision_id}/list`
   - Frontend: Implementado parcialmente em `Duplios.tsx`
   - **Falta**: 
     - Interface mais completa de ingest/verify
     - Visualização de identidades por revisão
     - Histórico de verificações

8. **LCA Recalculate**
   - Backend: `POST /duplios/revisions/{revision_id}/lca/recalculate`
   - Frontend: Implementado em `Duplios.tsx` mas pode melhorar
   - **Falta**: 
     - Feedback visual durante recálculo
     - Comparação antes/depois de LCA

9. **Public DPP Viewer (QR Code)**
   - Backend: `GET /duplios/public/dpp/{slug}`, `GET /duplios/view/{slug}`
   - Frontend: Não implementado
   - **Falta**: 
     - Página pública para visualizar DPP via QR code
     - Design otimizado para mobile
     - Visualização simplificada para consumidores

10. **Export Filters Avançados**
    - Backend: `GET /duplios/export/csv`, `GET /duplios/export/json` (com filtros)
    - Frontend: Implementado mas básico
    - **Falta**: 
      - Interface para configurar filtros de export
      - Preview antes de exportar
      - Seleção de campos a exportar

---

## 🎯 Prioridades Sugeridas

### Alta Prioridade
1. **SmartInventory: MRP Parameters Management** - Crítico para configuração
2. **SmartInventory: BOM Explosion Viewer** - Core feature de MRP
3. **Duplios: Public DPP Viewer** - Necessário para QR codes
4. **ProdPlan: Data Quality Dashboard** - Importante para confiança nos dados

### Média Prioridade
5. **ProdPlan: Product Metrics Dashboard** - Útil para análise
6. **Duplios: Carbon Breakdown Detalhado** - Melhora transparência
7. **SmartInventory: Forecast Integration UI** - Melhora planeamento
8. **ProdPlan: MILP Optimization UI** - Avançado mas útil

### Baixa Prioridade
9. **Duplios: Analytics Avançados** - Nice to have
10. **SmartInventory: MRP Reset UI** - Operacional
11. **ProdPlan: Workforce Forecast UI** - Se já existe, melhorar

---

## 📝 Notas

- Alguns endpoints podem estar parcialmente implementados mas não totalmente integrados
- Verificar se há componentes que podem ser expandidos vs criar novos
- Considerar criar páginas dedicadas vs integrar em dashboards existentes
- Algumas features podem estar em páginas diferentes (ex: ProdplanWorkforce vs WorkforcePerformance)


