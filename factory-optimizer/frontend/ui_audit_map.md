# UI Audit Map - Contract 16

## Navegação Principal (6 módulos + Chat)

### Barra de Topo
```
┌─────────────────────────────────────────────────────────────────────┐
│  ⚙️ ProdPlan 4.0                            [Carregar Dados]        │
├─────────────────────────────────────────────────────────────────────┤
│  🏭 ProdPlan  │  📦 SmartInventory  │  🌿 Duplios  │               │
│  💻 Digital Twin  │  🧠 Inteligência  │  🔬 R&D                     │
└─────────────────────────────────────────────────────────────────────┘
                                            💬 Latif AI (botão flutuante)
```

---

## 1. PRODPLAN 🏭

### Estrutura de Tabs (7 principais + dropdown Ferramentas)

| Tab | Descrição | Sub-tabs |
|-----|-----------|----------|
| **Planeamento** | Gantt, Timeline, Modos | Principal, Avançado, Projetos, Gantt & Timeline |
| **Dashboards** | Heatmaps, OEE, Projeções | Overview, Heatmap, Relatórios |
| **Colaboradores** | Performance, Saturação | Performance, Alocação, Competências |
| **Gargalos** | Deteção, Análise | Lista, Análise, Mitigação |
| **Sugestões IA** | Recomendações | — |
| **Máquinas** ⭐ | Productive Care (NOVO) | Mapa, Agenda, Paragens |
| **Ferramentas** ▼ | Dropdown | Digital Twin, Qualidade Dados, MILP, Prevention Guard, SHI-DT Training |

### Tab "Máquinas" (NOVA - Contrato 16)

```
┌─────────────────────────────────────────────────────────────────────┐
│  🖥️ Máquinas & Manutenção                          [Atualizar]      │
│  Estado, saúde, paragens e manutenção dos recursos produtivos       │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  │ Máquinas │  │ SHI Médio│  │ OEE Médio│  │Manutenções│  │ Paragens │
│  │   24     │  │  82.5%   │  │  78.3%   │  │    6     │  │   4.5h   │
│  │ 1 offline│  │ 1 crítica│  │          │  │ 2 atraso │  │   hoje   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
├─────────────────────────────────────────────────────────────────────┤
│  [Filtro Estado ▼] [Filtro Célula ▼]        [Mapa] [Agenda] [Paragens]│
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  📊 MAPA DE MÁQUINAS                                                │
│  ┌─────────┬────────┬──────┬──────┬───────────────┬─────────┬──────┐│
│  │ Máquina │ Estado │ RUL  │ OEE  │Próx Manutenção│Paragem  │ Ações││
│  ├─────────┼────────┼──────┼──────┼───────────────┼─────────┼──────┤│
│  │ CNC-001 │Saudável│75 d  │85.2% │ 2025-01-15    │ 0.5h    │ ➡️ 🔗││
│  │ PRESS-01│ Alerta │19 d  │72.5% │ 2024-12-15    │ 1.5h    │ ➡️ 🔗││
│  │ LATHE-01│Crítico │ 5 d  │58.3% │ 2024-12-01    │ 2.5h    │ ➡️ 🔗││
│  └─────────┴────────┴──────┴──────┴───────────────┴─────────┴──────┘│
│                                                                      │
│  📅 AGENDA DE MANUTENÇÃO          ⚠️ PARAGENS RECENTES              │
│  ┌────────────────────────────┐   ┌──────────────────────────────┐  │
│  │ ⚠️ Em Atraso (2)           │   │ LATHE-01 - Avaria - 90min    │  │
│  │ • PRESS-001 - Rolamentos   │   │ PRESS-01 - Setup - 45min     │  │
│  │ • LATHE-001 - Revisão SHI  │   │ LATHE-01 - Aquecimento - ⏱️  │  │
│  ├────────────────────────────┤   │ CNC-001 - Microparagem - 15m │  │
│  │ 📅 Planeadas (4)           │   └──────────────────────────────┘  │
│  │ • CNC-001 - Calibração     │                                     │
│  │ • WELD-001 - Limpeza       │                                     │
│  └────────────────────────────┘                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Funcionalidades Agregadas
- **SHI (Smart Health Index)** - do Digital Twin
- **RUL (Remaining Useful Life)** - do Digital Twin
- **Paragens e Alarmes** - do Shopfloor
- **Manutenções Planeadas** - do PdM-IPS
- **OEE por Máquina** - dos Dashboards
- **Integração com Plano** - impacto de manutenções

#### Botões de Ação
- `Atualizar` - Refresh dados SHI/RUL + paragens
- `Ver detalhes` → Abre modal com métricas completas
- `Abrir em Digital Twin` → Drill-down técnico

---

## 2. SMARTINVENTORY 📦

### Estrutura de Tabs (10)

| Tab | Funcionalidade |
|-----|----------------|
| Stock Real-Time | Quantidades por SKU/armazém |
| Matriz ABC/XYZ | Classificação de itens |
| Forecast & ROP | Previsões e reorder points |
| MRP Encomendas | Cálculo de necessidades |
| MRP Completo | Motor MRP avançado |
| MRP Forecast | Integração de previsões |
| Parâmetros MRP | Configuração por SKU |
| BOM Explosion | Explosão de estruturas |
| Dados Operacionais | Ingestão de 4 Excels |
| Work Instructions | Gestão de instruções |

---

## 3. DUPLIOS 🌿

### Estrutura de Tabs (6)

| Tab | Funcionalidade |
|-----|----------------|
| Visão Geral | Lista DPPs, KPIs |
| PDM | Items, Revisions, BOM, Routing, ECO/ECR |
| Impacto (LCA) | Cálculo de impacto ambiental |
| Compliance | ESPR, CBAM, CSRD Radar |
| Identidade | QR codes, identidade digital |
| Analytics | Trust Index, Carbono, Evolução |

---

## 4. DIGITAL TWIN 💻

### Estrutura de Sub-tabs (2)

| Sub-tab | Funcionalidade |
|---------|----------------|
| Máquinas (SHI-DT) | Health Index, RUL, CVAE, Perfis |
| Produto (XAI-DT) | Desvios geométricos, RCA |

---

## 5. INTELIGÊNCIA 🧠

### Estrutura de Sub-tabs (3)

| Sub-tab | Funcionalidade |
|---------|----------------|
| Análise Causal | OLS, DML, Trade-offs |
| Otimização | MILP, GA, Bayesian |
| What-If Avançado | Cenários macro |

---

## 6. R&D 🔬

### Estrutura

- Overview com KPIs
- WP1 Routing Experiments
- WP2 Suggestions Evaluation
- WP3 Inventory & Capacity
- WP4 Learning Scheduler
- Relatórios SIFIDE

---

## 7. CHAT (Botão Flutuante) 💬

- Latif AI Assistant
- Modal overlay
- Acessível de qualquer página

---

## Mapeamento Features → UI

| Feature | Módulo | Tab | Implementado |
|---------|--------|-----|--------------|
| APS/Flow Shop | ProdPlan | Planeamento | ✅ |
| Scheduling MILP | ProdPlan | Ferramentas > MILP | ✅ |
| Gargalos | ProdPlan | Gargalos | ✅ |
| Sugestões IA | ProdPlan | Sugestões IA | ✅ |
| SHI-DT Operacional | ProdPlan | **Máquinas** | ✅ |
| Manutenção/Paragens | ProdPlan | **Máquinas** | ✅ |
| OEE por Máquina | ProdPlan | **Máquinas** | ✅ |
| Digital Twin Técnico | Digital Twin | Máquinas | ✅ |
| MRP Completo | SmartInventory | MRP Completo | ✅ |
| Forecast | SmartInventory | Forecast & ROP | ✅ |
| ABC/XYZ | SmartInventory | Matriz ABC/XYZ | ✅ |
| 4 Excels | SmartInventory | Dados Operacionais | ✅ |
| PDM | Duplios | PDM | ✅ |
| DPP | Duplios | Visão Geral | ✅ |
| Trust Index | Duplios | Analytics | ✅ |
| Compliance Radar | Duplios | Compliance | ✅ |
| Prevention Guard | ProdPlan | Ferramentas > Prevention | ✅ |
| Causal Analysis | Inteligência | Análise Causal | ✅ |
| R&D WP1-WP4 | R&D | Overview + Tabs | ✅ |

---

*UI Audit Map - Contract 16 - Generated automatically*


