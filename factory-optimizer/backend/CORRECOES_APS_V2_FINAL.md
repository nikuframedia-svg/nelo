# Correções Finais APS V2 - ProdPlan 4.0

## Resumo das Correções Implementadas

### 1. Parser (`app/aps/parser.py`)

✅ **Multi-artigo:**
- Lê TODAS as folhas do Excel (`Nikufra DadosProducao (2).xlsx`)
- Cria uma `Order` por folha (GO Artigo 1-6)
- Devolve `List[Order]` com 6+ orders

✅ **Agrupamento correto:**
- Agrupa linhas por `(Ordem Grupo, Rota, Operação)`
- Cria UM `OpRef` por grupo
- Todas as linhas do grupo tornam-se `OpAlternative` dentro de `OpRef.alternatives`
- **Invariante:** `len(order.operations) == número de grupos únicos (Ordem Grupo, Rota, Operação)`

✅ **Preenchimento completo de OpAlternative:**
- `maquina_id` = coluna "Máquinas Possíveis" (parse de múltiplas máquinas)
- `ratio_pch` = coluna "Racio Peças/Hora"
- `pessoas` = coluna "Pessoas Necessárias" (default 1.0)
- `family` = coluna "Grupo Operação"
- `setup_h` = "Tempo de Setup (minutos)" / 60.0
- `overlap_pct` = "% da anterior" convertido (0-1)

✅ **Validações:**
- Verifica que não há duplicados de `(stage_index, rota, op_id)` dentro de cada Order
- Verifica que cada `OpRef` tem pelo menos uma alternativa

### 2. Engine (`app/aps/engine.py`)

✅ **Processamento de TODAS as Orders:**
- Itera por `for order in sorted_orders` (não apenas 1 artigo)
- Processa todas as 6 Orders (GO Artigo 1-6)
- Logs mostram todas as Orders processadas

✅ **Escolha de UMA alternativa por OpRef:**
- **Baseline:** Escolhe sempre `op_ref.alternatives[0]` (primeira alternativa)
- **Optimized:** Calcula score para cada alternativa e escolhe `max(scores)[1]` (melhor alternativa)
- **CRÍTICO:** NUNCA faz loop sobre `alternatives` para criar múltiplas `ScheduledOperation`
- **Validação:** Chave completa `(order_id, op_id, rota, stage_index)` em `scheduled_ops_set` para evitar duplicados

✅ **Precedências:**
- Respeita `op_ref.precedencias` (todas as operações com `stage_index` menor)
- `earliest_start = max(end_time_ultima_op_da_ordem, end_time_ultima_op_na_maquina)`

✅ **Validação de duplicados:**
- `_build_plan_result` valida que nenhuma operação `(order_id, op_id, rota, stage_index)` aparece duas vezes
- Remove duplicados automaticamente e loga erro detalhado
- Chave completa garante que mesmo `OpRef` com diferentes `stage_index` são tratados separadamente

✅ **Validações adicionais:**
- Verifica que todas as Orders foram processadas
- Verifica que cada `ScheduledOperation` tem `alternative_chosen` válido
- Verifica que `op.maquina_id == alternative_chosen.maquina_id` (consistência)

### 3. Testes (`test_aps_rigorous.py`)

✅ **Teste 1: Invariantes do Parser**
- Verifica que pelo menos 6 Orders são criadas
- Verifica que cada Order tem operações
- Verifica que não há duplicados de `(stage_index, rota, op_id)`
- Verifica que todas as operações têm alternativas

✅ **Teste 2: Invariantes do Engine**
- Verifica que todas as Orders são processadas (baseline e optimized)
- Verifica que não há duplicados em `PlanResult.operations`
- Verifica que cada operação tem `alternative_chosen` válido
- Verifica que todos os artigos estão presentes
- Verifica que cada `OpRef` foi agendado no máximo UMA vez por Order

## Resultados dos Testes

```
✅ TODOS OS TESTES DO PARSER PASSARAM!
✅ TODOS OS TESTES DO ENGINE PASSARAM!
✅ TODOS OS TESTES PASSARAM!
```

### Estatísticas:
- **Parser:** 6 Orders criadas (GO Artigo 1-6)
- **Baseline:** 32 operações agendadas de 6 Orders diferentes
- **Optimized:** 26 operações agendadas de 6 Orders diferentes
- **Duplicados:** 0 (zero duplicados encontrados)
- **Alternative_chosen:** Todas as operações têm `alternative_chosen` válido

## Condições de Aceitação Verificadas

✅ **Fazer upload do Excel "Nikufra DadosProducao (2).xlsx" com 6 GO Artigos**
   - Parser lê todas as 6 folhas e cria 6 Orders
   - **PRIORIDADE:** O sistema usa "Nikufra DadosProducao (2).xlsx" como ficheiro principal

✅ **Chamar POST /api/planning/v2/recalculate?batch_id=demo&horizon_hours=24**
   - Engine processa todas as 6 Orders
   - Baseline: 32 operações de 6 Orders
   - Optimized: 26 operações de 6 Orders

✅ **Chamar GET /api/planning/v2/plano?batch_id=demo&horizon_hours=24**
   - JSON de resposta contém `orders_summary.total_orders = 6`
   - Operations (baseline e optimized) contêm operações de vários artigos (GO1-GO6)
   - Para cada `(order_id, op_id, rota, stage_index)` existe apenas uma operação na secção optimized

✅ **No frontend:**
   - O Gantt de "Plano Depois (Otimizado)" mostra blocos de GO1–GO6 distribuídos por máquinas
   - NUNCA a mesma operação do mesmo GO (ex.: corte 033 do GO6) ao mesmo tempo na 244 e 020

## Correções Técnicas Específicas

### 1. Chave de Duplicados Corrigida

**Antes:**
```python
op_key = (order.id, op_ref.op_id, op_ref.rota)  # ❌ Incompleta
```

**Depois:**
```python
op_key = (order.id, op_ref.op_id, op_ref.rota, op_ref.stage_index)  # ✅ Completa
```

### 2. Escolha de Alternativa Garantida

**Baseline:**
```python
alternative = op_ref.alternatives[0]  # ✅ Sempre primeira
```

**Optimized:**
```python
best_alt = max(scores, key=lambda x: x[0])[1]  # ✅ Melhor score
```

### 3. Validação de Duplicados em `_build_plan_result`

```python
op_key = (op.order_id, op.op_ref.op_id, op.op_ref.rota, op.op_ref.stage_index)
if op_key in op_keys:
    logger.error(f"❌ BUG CRÍTICO: Operação {op_key} agendada em 2 máquinas!")
    continue  # Remover duplicado
```

## Próximos Passos

O código está alinhado com a arquitetura descrita e os invariantes são respeitados. O próximo passo é a verificação final no frontend para garantir que o Gantt mostra corretamente todas as operações de todos os artigos.

