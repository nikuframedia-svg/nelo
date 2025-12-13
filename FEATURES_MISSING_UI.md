# 🔍 Análise: Features Backend sem UI Correspondente

## Resumo Executivo

Este documento identifica funcionalidades implementadas no backend que ainda não têm interface de utilizador (UI) completa ou parcial.

---

## 📦 PDM (Product Data Management)

### ✅ Implementado na UI
- ✅ Items CRUD (listar, criar, editar)
- ✅ Revisions workflow (criar, release, obsolete)
- ✅ BOM management (adicionar/remover linhas)
- ✅ Routing management (adicionar/remover operações)
- ✅ Validação antes de release
- ✅ ECR listagem básica

### ❌ Faltando na UI

1. **ECR/ECO Workflow Completo**
   - Backend: `POST /pdm/ecr`, `GET /pdm/ecr/{ecr_id}/impact`
   - Frontend: Apenas listagem básica
   - **Falta**: 
     - Criar ECR a partir da UI
     - Visualizar impacto de ECR
     - Aprovar/rejeitar ECR
     - Criar ECO a partir de ECR aprovado
     - Workflow visual de ECR → ECO → Nova Revisão

2. **Revision Comparison/Diff**
   - Backend: `GET /pdm/revisions/compare?from_revision_id=X&to_revision_id=Y`
   - Frontend: Não implementado
   - **Falta**: 
     - Interface para comparar duas revisões
     - Visualização de diferenças (BOM, Routing, etc.)
     - Highlight de mudanças

3. **Impact Analysis UI**
   - Backend: `GET /pdm/items/{item_id}/impact`
   - Frontend: Não implementado
   - **Falta**: 
     - Visualizar impacto de mudança de item/revisão
     - Lista de ordens afetadas
     - Lista de produtos dependentes
     - Estimativa de impacto em produção

4. **BOM Explosion Visual**
   - Backend: `GET /pdm/revisions/{revision_id}/bom/explode?qty=X`
   - Frontend: Não implementado
   - **Falta**: 
     - Visualização hierárquica de BOM explosion
     - Árvore de componentes
     - Quantidades calculadas por nível

5. **Attachments Management**
   - Backend: Modelo `Attachment` existe em `pdm_models.py`
   - Frontend: Não implementado
   - **Falta**: 
     - Upload de anexos (CAD, PDFs, instruções)
     - Visualização de anexos por revisão
     - Download de documentos

---

## 📋 Work Instructions

### ✅ Implementado na UI
- ✅ Executar instruções (operator interface)
- ✅ Completar steps
- ✅ Quality checks
- ✅ Visualização passo-a-passo

### ❌ Faltando na UI

1. **Admin Interface (Create/Edit Instructions)**
   - Backend: `POST /work-instructions`, `GET /work-instructions/{id}`
   - Frontend: Apenas execução
   - **Falta**: 
     - Interface para criar/editar instruções
     - Editor de steps com drag-and-drop
     - Upload de imagens/vídeos/3D models
     - Configuração de quality checks
     - Versionamento de instruções

2. **Execution History & Analytics**
   - Backend: `GET /work-instructions/executions`
   - Frontend: Não implementado
   - **Falta**: 
     - Lista de execuções históricas
     - Estatísticas de execução (tempo médio, taxa de erro)
     - Análise de conformidade
     - Relatórios de execução

3. **3D Model Viewer Integration**
   - Backend: Suporte para `model_3d_url` e `highlight_region`
   - Frontend: Não implementado
   - **Falta**: 
     - Visualizador 3D integrado (Three.js)
     - Highlight de regiões por step
     - Anotações 3D

---

## ⚡ Optimization

### ✅ Implementado na UI
- ✅ Time prediction demo
- ✅ Golden runs listagem
- ✅ Parameter optimization demo
- ✅ Scheduling demo

### ❌ Faltando na UI

1. **What-If Analysis UI**
   - Backend: `build_planning_instance_from_raw()` em `ops_ingestion/services.py`
   - Frontend: Não implementado
   - **Falta**: 
     - Interface para criar cenários what-if
     - Comparação de cenários (métricas lado a lado)
     - Visualização de impacto de mudanças

2. **Pareto Frontier Visualization**
   - Backend: `POST /optimization/pareto/optimize`, `POST /optimization/pareto/demo`
   - Frontend: Não implementado
   - **Falta**: 
     - Gráfico de Pareto (scatter plot multi-objetivo)
     - Seleção interativa de soluções
     - Comparação de trade-offs

3. **Schedule Comparison**
   - Backend: `POST /optimization/schedule/compare`
   - Frontend: Não implementado
   - **Falta**: 
     - Comparar múltiplos schedules
     - Métricas lado a lado
     - Visualização de diferenças

4. **Golden Runs Gap Analysis**
   - Backend: `POST /optimization/golden-runs/gap`
   - Frontend: Não implementado
   - **Falta**: 
     - Visualizar gap entre performance atual e golden
     - Recomendações visuais
     - Gráficos de performance vs golden

5. **Time Prediction Training Interface**
   - Backend: Modelo ML existe mas não há endpoint de treino via API
   - Frontend: Não implementado
   - **Falta**: 
     - Interface para adicionar dados de treino
     - Trigger de re-treino
     - Visualização de accuracy do modelo

---

## 🛡️ Prevention Guard

### ✅ Implementado na UI
- ✅ Status e estatísticas
- ✅ Validation demo
- ✅ Risk prediction demo
- ✅ Exception management (approve/reject)
- ✅ Rules listagem

### ❌ Faltando na UI

1. **Custom Rules Editor**
   - Backend: `POST /guard/rules`, `PATCH /guard/rules/{rule_id}/toggle`
   - Frontend: Apenas listagem
   - **Falta**: 
     - Editor visual de regras
     - Criar/editar regras customizadas
     - Testar regras antes de ativar
     - Validação de sintaxe de condições

2. **Training Data Management**
   - Backend: `POST /guard/training/add-data`, `POST /guard/training/train`
   - Frontend: Não implementado
   - **Falta**: 
     - Interface para adicionar dados históricos
     - Upload de dataset
     - Trigger de treino do modelo preditivo
     - Visualização de accuracy/confusion matrix

3. **Event Log Viewer**
   - Backend: `GET /guard/events`, `GET /guard/statistics`
   - Frontend: Não implementado
   - **Falta**: 
     - Visualização de eventos do guard
     - Filtros e busca
     - Timeline de eventos
     - Estatísticas detalhadas

---

## 🏥 SHI-DT (Smart Health Index Digital Twin)

### ✅ Implementado na UI
- ✅ Machine list
- ✅ Health index visualization
- ✅ RUL estimation
- ✅ Alerts listagem
- ✅ Metrics summary

### ❌ Faltando na UI

1. **Sensor Data Ingestion UI**
   - Backend: `POST /shi-dt/machines/{machine_id}/ingest`
   - Frontend: Não implementado
   - **Falta**: 
     - Interface para upload de dados de sensores
     - Upload de ficheiro CSV/JSON
     - Validação de dados antes de ingestão
     - Preview de dados

2. **Model Training/Retraining Interface**
   - Backend: Lógica de re-treino existe mas não exposta via API
   - Frontend: Não implementado
   - **Falta**: 
     - Trigger de re-treino manual
     - Visualização de performance do modelo
     - Configuração de parâmetros de treino
     - Histórico de versões do modelo

3. **Operational Profile Management**
   - Backend: Suporte para perfis operacionais
   - Frontend: Não implementado
   - **Falta**: 
     - Criar/editar perfis operacionais
     - Associar perfis a máquinas
     - Visualização de perfis ativos

4. **Sensor Contribution Analysis Detail**
   - Backend: `top_contributors` em status response
   - Frontend: Visualização básica
   - **Falta**: 
     - Gráfico detalhado de contribuição de sensores
     - Timeline de contribuições
     - Análise de tendências

---

## 🔬 XAI-DT Product (Explainable Digital Twin)

### ✅ Implementado na UI
- ✅ Analyze scan (básico)
- ✅ Deviation analysis

### ❌ Faltando na UI

1. **Heatmap Visualization**
   - Backend: `GET /xai-dt-product/analyses/{analysis_id}/heatmap`
   - Frontend: Não implementado
   - **Falta**: 
     - Visualização 3D do heatmap de desvios
     - Overlay no modelo CAD
     - Cores por magnitude de desvio
     - Zoom e rotação interativa

2. **Root Cause Analysis UI**
   - Backend: `GET /xai-dt-product/root-causes`
   - Frontend: Não implementado
   - **Falta**: 
     - Visualização de causas raiz identificadas
     - Probabilidade/confiança por causa
     - Sugestões de correção
     - Histórico de causas similares

3. **Pattern Detection Visualization**
   - Backend: `GET /xai-dt-product/patterns`
   - Frontend: Não implementado
   - **Falta**: 
     - Visualização de padrões detectados
     - Clustering de desvios similares
     - Análise de padrões recorrentes

4. **Analysis History**
   - Backend: `GET /xai-dt-product/analyses`
   - Frontend: Não implementado
   - **Falta**: 
     - Lista de análises históricas
     - Comparação entre análises
     - Tendências de qualidade ao longo do tempo

---

## 📊 Ops Ingestion (Contract 14)

### ✅ Implementado na UI
- ✅ Upload de Excel files (4 tipos)
- ✅ WIP Flow visualization
- ✅ Orders listagem
- ✅ Stats dashboard

### ❌ Faltando na UI

1. **Planning Instance Builder UI**
   - Backend: `GET /ops-ingestion/planning-instance`
   - Frontend: Não implementado
   - **Falta**: 
     - Interface para construir planning instance a partir de dados raw
     - Preview de jobs/operations/machines
     - Integração com ProdPlan/APS
     - Export para scheduling

2. **Data Quality Dashboard**
   - Backend: Data quality checks existem em `data_quality.py`
   - Frontend: Não implementado
   - **Falta**: 
     - Visualização de quality flags
     - Estatísticas de qualidade por tipo de dado
     - Anomalias detectadas
     - Ações corretivas sugeridas

---

## 🎯 Prioridades Sugeridas

### Alta Prioridade
1. **PDM ECR/ECO Workflow** - Crítico para gestão de mudanças
2. **Work Instructions Admin Interface** - Necessário para criar conteúdo
3. **XAI-DT Heatmap Visualization** - Core feature do módulo
4. **Optimization What-If UI** - Valor alto para planeamento

### Média Prioridade
5. **Prevention Guard Rules Editor** - Melhora flexibilidade
6. **SHI-DT Sensor Ingestion UI** - Facilita uso do módulo
7. **PDM Revision Comparison** - Útil para análise de mudanças
8. **Optimization Pareto Visualization** - Melhora decisões

### Baixa Prioridade
9. **Work Instructions 3D Viewer** - Nice to have
10. **SHI-DT Model Training UI** - Avançado, poucos users
11. **XAI-DT Pattern Detection** - Análise avançada

---

## 📝 Notas

- Alguns endpoints podem estar implementados mas não expostos na UI principal
- Verificar se há componentes parciais que podem ser expandidos
- Considerar criar páginas dedicadas vs. integrar em dashboards existentes


