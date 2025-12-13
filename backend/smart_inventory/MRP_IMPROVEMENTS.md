# MRP (Material Requirements Planning Completo) - Melhorias Implementadas

## ✅ Requisitos Implementados

### 1. Parâmetros MRP por SKU/Item
- ✅ **ItemMRPParameters** com todos os parâmetros:
  - `safety_stock`: Nível de stock de segurança
  - `min_stock`: Stock mínimo
  - `max_stock`: Stock máximo
  - `moq`: Lote mínimo de compra/fabrico (Minimum Order Quantity)
  - `multiple`: Múltiplo de encomenda
  - `scrap_rate`: Percentagem de refugo esperada (0-1)
  - `lead_time_days`: Lead time de fornecimento (comprados) ou produção (fabricados)
  - `source`: Tipo (MANUFACTURED, PURCHASED, MIXED)

### 2. Explosão de BOM
- ✅ **Explosão multi-nível** recursiva
- ✅ Considera ordens de venda/produção confirmadas
- ✅ Considera previsões de venda (forecast integration)
- ✅ Calcula requisitos brutos por componente em cada período
- ✅ Suporta até N níveis (configurável, padrão: 20)

### 3. Cálculo de Necessidades Líquidas
- ✅ **Fórmula implementada exatamente como especificada**:
  ```
  NecessidadeLíquida = max(0, NecessidadeBruta + StockSegurança - (StockAtual + RecebimentosAgendados))
  ```
- ✅ Considera stock disponível atual
- ✅ Considera stock em trânsito/encomendado (scheduled receipts)
- ✅ Adiciona stock de segurança
- ✅ Calcula por período de tempo

### 4. Geração de Ordens Planeadas
- ✅ **Geração automática** de ordens de compra ou fabrico
- ✅ **Loteamento respeitando parâmetros**:
  - Ajuste por refugo: `NecessárioBrutoAjustado = NecessidadeBruta / (1 - r)`
  - Aplicação de MOQ: `QuantidadeOrdem = max(MOQ, QuantidadeAjustada)`
  - Aplicação de múltiplo: `QuantidadeOrdem = ceil(Quantidade / múltiplo) * múltiplo`
- ✅ Exemplo: necessidade = 120, MOQ=100, múltiplo=50 → gera 150 unidades

### 5. Offset Temporal (Lead Time)
- ✅ **Offset das ordens planeadas no tempo**:
  - `DataPlaneadaLiberação = DataNecessidade - LeadTime`
- ✅ Calcula datas de colocação de ordem
- ✅ Calcula datas de chegada/terminação
- ✅ Garante que materiais estão disponíveis justo a tempo

### 6. Relatórios e Alertas
- ✅ **Alertas de ruptura de stock**:
  - Identifica itens em risco de stockout
  - Lista períodos com stock negativo projetado
- ✅ **Alertas de ordens sugeridas**:
  - Lista ordens planeadas pendentes de aprovação
  - Status: PLANNED (pendente), FIRM (confirmada), RELEASED (liberada)
- ✅ **Alertas de capacidade**:
  - Sinaliza sobrecarga de capacidade em períodos
  - Lista ordens afetadas
- ✅ **Alertas de atrasos estimados**:
  - Calcula atrasos se materiais não chegarem a tempo
  - Considera lead time e disponibilidade

## 📊 Modelo Matemático

### Necessidade Líquida
```
NecessidadeLíquida = max(0, NecessidadeBruta + StockSegurança - (StockAtual + RecebimentosAgendados))
```

### Ajuste por Refugo
```
NecessárioBrutoAjustado = NecessidadeBruta / (1 - r)
```
Onde r é a taxa de refugo (0-1)

### Loteamento
```
Se NecessidadeLíquida = N, MOQ = Qmin, múltiplo = m:
  1. Ajustar por refugo: N_adj = N / (1 - r)
  2. Aplicar MOQ: Q = max(Qmin, N_adj)
  3. Aplicar múltiplo: Q_final = ceil(Q / m) * m
```

### Offset de Tempo
```
DataPlaneadaLiberação = DataNecessidade - LeadTime
```

## 🚀 Funcionalidades

### Explosão de BOM
```python
from smart_inventory.mrp_complete import MRPCompleteEngine, ItemMRPParameters

engine = MRPCompleteEngine()

# Carregar parâmetros
params = ItemMRPParameters(
    item_id=1,
    sku="PROD-001",
    safety_stock=50,
    moq=100,
    multiple=50,
    scrap_rate=0.05,
    lead_time_days=14,
)
engine.set_item_parameter(params)

# Explodir BOM
components = engine.explode_bom(item_id=1, quantity=100)
for comp, qty in components:
    print(f"{comp.component_sku}: {qty} unidades")
```

### Executar MRP
```python
# Adicionar demanda
from smart_inventory.mrp_complete import DemandEntry, OrderSource
from datetime import datetime, timedelta

demand = DemandEntry(
    item_id=1,
    sku="PROD-001",
    quantity=200,
    due_date=datetime.now() + timedelta(days=30),
    source=OrderSource.SALES_ORDER,
)
engine.add_demand(demand)

# Executar MRP
result = engine.run_mrp()

# Acessar resultados
for plan in result.item_plans.values():
    print(f"{plan.sku}: {len(plan.planned_orders)} ordens planeadas")
    for order in plan.planned_orders:
        print(f"  {order.order_id}: {order.quantity} unidades, início: {order.start_date}")
```

### Relatórios e Alertas
```python
# Alertas de ruptura
for alert in result.shortage_alerts:
    print(f"⚠️ {alert['sku']}: risco de stockout em {alert['stockout_periods']}")

# Alertas de capacidade
for alert in result.capacity_alerts:
    print(f"⚠️ {alert.work_center}: sobrecarga de {alert.overload_hours}h")

# Ordens pendentes de aprovação
pending_orders = [
    o for o in result.purchase_orders + result.manufacture_orders
    if o.status == PlannedOrderStatus.PLANNED
]
print(f"📋 {len(pending_orders)} ordens pendentes de aprovação")
```

## 🔄 Integração

### PDM (Product Data Management)
- ✅ Carrega BOM de revisões RELEASED
- ✅ Usa estrutura hierárquica do PDM
- ✅ Considera scrap_rate da BOM

### SmartInventory
- ✅ Sincroniza quantidades de stock atual
- ✅ Considera ordens de compra/fabrico em aberto
- ✅ Atualiza projeções de stock

### Forecasting
- ✅ Integra previsões de venda como demanda
- ✅ Permite ajuste manual de previsões
- ✅ Configurável (enable_forecast)

### ProdPlan (Futuro)
- ⚠️ Verificação básica de capacidade implementada
- ⚠️ Preparado para integração completa com APS

## 📈 Configuração

```python
from smart_inventory.mrp_complete import MRPConfig

config = MRPConfig(
    horizon_days=90,  # Horizonte de planeamento (3 meses)
    period_days=7,    # Granularidade semanal
    
    # Defaults
    default_lead_time_days=7.0,
    default_safety_stock=0.0,
    default_moq=1.0,
    default_multiple=1.0,
    default_scrap_rate=0.0,
    
    # Features
    enable_forecast=True,
    enable_capacity_check=True,
    
    # Alerts
    alert_low_coverage_days=14,
    alert_high_coverage_days=180,
)
```

## 📝 Melhorias Implementadas

1. ✅ **Cálculo de necessidades líquidas corrigido**:
   - Fórmula exata: `max(0, GB + SS - (OH + SR))`
   - Considera scheduled receipts corretamente

2. ✅ **Ajuste por refugo corrigido**:
   - Fórmula exata: `N / (1 - r)` em vez de aproximação
   - Mais preciso para taxas de refugo maiores

3. ✅ **Offset temporal melhorado**:
   - Cálculo direto: `DataLiberação = DataNecessidade - LeadTime`
   - Mais simples e correto

4. ✅ **Relatórios completos**:
   - Alertas de ruptura
   - Alertas de capacidade
   - Ordens pendentes
   - Projeções de stock

5. ✅ **Integração com PDM**:
   - Carrega BOM automaticamente
   - Usa revisões RELEASED
   - Considera scrap_rate da BOM

## 🔮 Extensões Futuras

1. **Otimização de Custos**:
   - Minimização: `∑(c_p * Q_p + c_h * StockMédio)`
   - Programação linear inteira (PLI)
   - Trade-off custo de encomenda vs. stock

2. **Previsão Estatística Avançada**:
   - ARIMA, LSTM, Transformer
   - Integração direta no MRP
   - Ajuste automático de previsões

3. **Otimização Multi-etapa**:
   - Nivelamento de carga
   - Ajuste de planos de produção
   - Balanceamento de recursos

4. **Capacidade Finita**:
   - Integração completa com ProdPlan
   - Restrições de capacidade por work center
   - Replaneamento automático

5. **Aprovação de Ordens**:
   - Workflow de aprovação
   - Notificações
   - Histórico de aprovações

## 📊 Estrutura de Dados

### ItemMRPParameters
- Parâmetros por item/SKU
- Stock, encomenda, timing

### DemandEntry
- Entrada de demanda
- Ordem de venda, previsão, manual

### PlannedOrder
- Ordem planeada
- Compra ou fabrico
- Status, quantidades, datas

### ItemMRPPlan
- Plano completo por item
- Períodos, requisitos, ordens
- Projeções de stock

### MRPRunResult
- Resultado completo do MRP
- Planos, ordens, alertas
- Métricas e estatísticas


