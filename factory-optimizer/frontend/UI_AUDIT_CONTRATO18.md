# UI Audit - Contrato 18
## Reorganização Final Ultra Clean

**Data:** 2025-12-11  
**Objetivo:** Mapear todas as funcionalidades e propor reorganização mínima

---

## 1. ESTRUTURA ATUAL

### 1.1 Módulos de Topo (App.tsx)
```
┌─────────────────────────────────────────────────────────────────────────┐
│  ProdPlan │ SmartInventory │ Duplios │ Digital Twin │ Inteligência │ R&D │
└─────────────────────────────────────────────────────────────────────────┘
                                + Chat (botão flutuante)
```

### 1.2 ProdPlan - Tabs Internas (7 tabs)
```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Planeamento │ Dashboards │ Colaboradores │ Gargalos │ Sugestões │ Máquinas │ Ferramentas │
└──────────────────────────────────────────────────────────────────────────────┘

Ferramentas Sub-tabs:
└─ Digital Twin | Data Quality | MILP | Prevention | SHI Training
```

✅ **Estrutura boa** - 7 tabs principais + dropdown Ferramentas

### 1.3 SmartInventory - Tabs Internas (10 tabs) ⚠️ MUITO!
```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Stock Real-Time │ Matriz ABC/XYZ │ Forecast │ MRP │ MRP Completo │ Forecast IA │ Parâmetros │ BOM │ Dados Op. │ Work Instr. │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

❌ **Problemas identificados:**
- 10 tabs é excessivo - dificulta navegação
- Tabs MRP + MRP Completo + Forecast IA + Parâmetros são redundantes → **Agrupar em MRP único**
- Work Instructions não pertence a SmartInventory → **Mover para ProdPlan**

### 1.4 Duplios - Tabs Internas (6 tabs)
```
┌────────────────────────────────────────────────────────────────────────┐
│ Visão Geral │ PDM │ Impacto (LCA) │ Compliance │ Identidade │ Analytics │
└────────────────────────────────────────────────────────────────────────┘
```

✅ **Estrutura boa** - tabs claras e bem agrupadas

### 1.5 Digital Twin - Sub-navegação (2)
```
┌─────────────────────────────────────┐
│ Máquinas (SHI-DT) │ Produto (XAI-DT) │
└─────────────────────────────────────┘
```

✅ **Estrutura boa** - simples e clara

### 1.6 Inteligência - Sub-navegação (3)
```
┌──────────────────────────────────────────┐
│ Análise Causal │ Otimização │ What-If │
└──────────────────────────────────────────┘
```

✅ **Estrutura boa** - foca em análise avançada

### 1.7 R&D - Tabs Internas
```
┌─────────────────────────────────────────────────────┐
│ Overview │ WP1 │ WP2 │ WP3 │ WP4 │ WPX │ Relatórios │
└─────────────────────────────────────────────────────┘
```

✅ **Estrutura boa** - work packages claramente identificados

---

## 2. PÁGINAS LEGACY (a limpar/consolidar)

### Páginas que redirecionam para módulos principais:
- `AdvancedPlanning.tsx` → /prodplan
- `Bottlenecks.tsx` → /prodplan
- `Dashboards.tsx` → /prodplan
- `Planning.tsx` → /prodplan
- `Reports.tsx` → /prodplan
- `Suggestions.tsx` → /prodplan
- `WorkforcePerformance.tsx` → /prodplan
- `ProjectPlanning.tsx` → /prodplan

### Páginas subcomponentes de ProdPlan:
- `ProdplanPlanning.tsx` - usado
- `ProdplanDashboards.tsx` - usado
- `ProdplanWorkforce.tsx` - usado
- `ProdplanBottlenecks.tsx` - usado
- `ProdplanSuggestions.tsx` - usado
- `ProdplanDigitalTwin.tsx` - usado

### Páginas de Digital Twin:
- `DigitalTwin.tsx` - wrapper não usado diretamente
- `DigitalTwinMachines.tsx` - usado
- `DigitalTwinProduct.tsx` - duplicado? verificar
- `XAIDTProduct.tsx` - usado

### Páginas de Inventário:
- `MRPDashboard.tsx` - legacy, funcionalidade em SmartInventory

### Páginas de Shopfloor:
- `Shopfloor.tsx` - legacy mas acessível via rota
- `WorkInstructions.tsx` - legacy, funcionalidade em SmartInventory

### Páginas de Qualidade:
- `PreventionGuard.tsx` - legacy, funcionalidade em ProdPlan/Tools
- `ZDMSimulator.tsx` - onde usar?

### Outras:
- `PDMDashboard.tsx` - legacy, funcionalidade em Duplios
- `OptimizationDashboard.tsx` - usado em Inteligência

---

## 3. PROPOSTA DE REORGANIZAÇÃO

### 3.1 Módulos de Topo (manter 6)
```
ProdPlan │ SmartInventory │ Duplios │ Digital Twin │ Inteligência │ R&D + Chat
```
✅ **Sem alteração** - estrutura já está limpa

### 3.2 ProdPlan - MANTER (já otimizado)
```
Planeamento │ Dashboards │ Colaboradores │ Gargalos │ Sugestões │ Máquinas │ Ferramentas
```
**+ Adicionar:** Work Instructions nas Ferramentas ou como sub-tab de Máquinas

### 3.3 SmartInventory - CONSOLIDAR (10 → 5 tabs)

**ANTES (10 tabs):**
```
Stock Real-Time │ Matriz ABC/XYZ │ Forecast │ MRP │ MRP Completo │ Forecast IA │ Parâmetros │ BOM │ Dados Op. │ Work Instr.
```

**DEPOIS (5 tabs):**
```
┌────────────────────────────────────────────────────────────────────────────┐
│ Stock & ABC/XYZ │ Forecast & ROP │ MRP Completo │ BOM & Estrutura │ Dados Operacionais │
└────────────────────────────────────────────────────────────────────────────┘
```

**Mapeamento:**
- `Stock & ABC/XYZ` = realtime + matrix
- `Forecast & ROP` = forecast (inclui forecast IA, parâmetros ROP)
- `MRP Completo` = mrp + mrp-complete + mrp-forecast + mrp-parameters (UNIFICADO)
- `BOM & Estrutura` = bom-explosion
- `Dados Operacionais` = operational-data (ingestão excels)

**Remover:**
- `Work Instructions` → mover para ProdPlan > Ferramentas > Instruções

### 3.4 Duplios - MANTER (já otimizado)
```
Visão Geral │ PDM │ Impacto │ Compliance │ Identidade │ Analytics
```

### 3.5 Digital Twin - MANTER
```
Máquinas (SHI-DT) │ Produto (XAI-DT)
```

### 3.6 Inteligência - MANTER
```
Análise Causal │ Otimização │ What-If
```

### 3.7 R&D - MANTER
```
Overview │ WP1-WP4 │ WPX │ Relatórios
```

---

## 4. FUNCIONALIDADES - CHECKLIST

| Feature | Módulo | Tab | Status |
|---------|--------|-----|--------|
| Gantt/Timeline | ProdPlan | Planeamento | ✅ |
| Modos de Planeamento | ProdPlan | Planeamento | ✅ |
| Heatmaps OEE | ProdPlan | Dashboards | ✅ |
| Relatórios | ProdPlan | Dashboards | ✅ |
| Colaboradores | ProdPlan | Colaboradores | ✅ |
| Gargalos | ProdPlan | Gargalos | ✅ |
| Sugestões IA | ProdPlan | Sugestões | ✅ |
| Productive Care | ProdPlan | Máquinas | ✅ |
| Work Orders | ProdPlan | Máquinas | ✅ |
| Spare Parts | ProdPlan | Máquinas | ✅ |
| Data Quality SNR | ProdPlan | Ferramentas | ✅ |
| MILP Optimization | ProdPlan | Ferramentas | ✅ |
| Prevention Guard | ProdPlan | Ferramentas | ✅ |
| SHI Training | ProdPlan | Ferramentas | ✅ |
| Work Instructions | ProdPlan | **Ferramentas** | 🔄 Mover |
| Stock Real-Time | SmartInventory | Stock & ABC/XYZ | ✅ |
| ABC/XYZ Matrix | SmartInventory | Stock & ABC/XYZ | ✅ |
| Forecast AI | SmartInventory | Forecast & ROP | ✅ |
| ROP Dinâmico | SmartInventory | Forecast & ROP | ✅ |
| MRP Engine | SmartInventory | MRP Completo | ✅ |
| BOM Explosion | SmartInventory | BOM & Estrutura | ✅ |
| Excel Ingestion | SmartInventory | Dados Op. | ✅ |
| DPP CRUD | Duplios | Overview | ✅ |
| PDM Items | Duplios | PDM | ✅ |
| LCA Impact | Duplios | Impacto | ✅ |
| Trust Index | Duplios | Compliance | ✅ |
| Gap Filling | Duplios | Compliance | ✅ |
| ESPR/CBAM/CSRD | Duplios | Compliance | ✅ |
| Identity | Duplios | Identidade | ✅ |
| Analytics | Duplios | Analytics | ✅ |
| SHI-DT | Digital Twin | Máquinas | ✅ |
| RUL | Digital Twin | Máquinas | ✅ |
| XAI-DT Product | Digital Twin | Produto | ✅ |
| Causal Analysis | Inteligência | Causal | ✅ |
| Optimization | Inteligência | Otimização | ✅ |
| What-If | Inteligência | What-If | ✅ |
| R&D WP1-4 | R&D | WP1-4 | ✅ |
| R&D WPX | R&D | WPX | ✅ |
| Chat/Copilot | Flutuante | - | ✅ |
| Data Upload | Header | Modal | ✅ |

---

## 5. AÇÕES A EXECUTAR

### 5.1 SmartInventory - Consolidação de Tabs
1. Renomear `realtime` → `stock`
2. Criar tab unificada `mrp` que inclui sub-secções
3. Remover `work-instructions` → mover para ProdPlan

### 5.2 ProdPlan - Adicionar Work Instructions
1. Adicionar `work-instructions` ao dropdown Ferramentas

### 5.3 Limpar Páginas Legacy
1. Manter apenas redirecionamentos essenciais
2. Verificar se todas as funcionalidades estão acessíveis

### 5.4 Verificação Final
1. Testar todas as rotas
2. Verificar que nenhuma feature ficou inacessível
3. Build frontend


