# Debug APS V2 - Problema: Apenas GO Artigo 6 aparece

## Problema Identificado
- No frontend só aparece GO Artigo 6
- Botão "Recalcular plano" pode não estar visível

## Correções Implementadas

### 1. Cache Limpo
- ✅ Cache completamente limpo: `rm -rf app/data/plan_cache/*`
- ✅ Cache em memória vazio

### 2. Logs Adicionados

#### Backend (`app/api/planning_v2.py`):
- ✅ Log quando calcula plano: mostra artigos processados
- ✅ Log quando retorna plano: mostra artigos no baseline e optimized

#### Backend (`app/aps/models.py`):
- ✅ Log em `Plan.to_dict()`: mostra `orders_summary` completo

#### Frontend (`frontend/src/pages/Planning.tsx`):
- ✅ Log quando recebe plano: mostra artigos recebidos
- ✅ Log quando recalcula: mostra processo completo

### 3. Botão "Recalcular plano"
- ✅ Botão sempre visível (linha 342)
- ✅ Logs adicionados para debug
- ✅ Refetch automático após recalcular

### 4. Verificações

#### Parser:
- ✅ Lê todas as 6 folhas
- ✅ Cria 6 Orders (GO Artigo 1-6)

#### Engine:
- ✅ Processa todas as 6 Orders
- ✅ Sem duplicados

#### Serialização:
- ✅ `artigo` extraído de `order_id.replace("ORD-", "")`
- ✅ Todas as operações incluem `artigo`

## Como Verificar

### 1. Abrir DevTools do navegador (F12)
- Ver Console para logs:
  - `📥 Frontend recebeu plano:` - mostra artigos recebidos
  - `🔄 Iniciando recálculo do plano...` - quando clica em recalcular
  - `✅ Recálculo concluído:` - quando termina

### 2. Verificar Backend Logs
- Procurar por:
  - `📊 Artigos processados - Baseline:` - mostra Orders processadas
  - `📋 Plan.to_dict(): orders_summary.total_orders=` - mostra total de orders
  - `📤 GET /plano retornando:` - mostra artigos retornados

### 3. Verificar Cache
```bash
cd backend
python3 -c "
from app.aps.cache import get_plan_cache
cache = get_plan_cache()
print(f'Cache dir: {cache.cache_dir}')
print(f'Cache em memória: {len(cache._memory_cache)} entradas')
"
```

### 4. Testar Recalcular
1. Abrir frontend
2. Ir para página de Planeamento
3. Clicar em "🔄 Recalcular plano"
4. Verificar logs no Console
5. Verificar se aparecem todos os artigos no Gantt

## Possíveis Causas Restantes

1. **Cache antigo no navegador**: Limpar cache do navegador (Ctrl+Shift+Delete)
2. **React Query cache**: O frontend pode ter cache antigo
3. **Filtro no GanttChart**: Verificar se há filtro por artigo (não encontrado)
4. **Problema de renderização**: Verificar se todas as operações estão a ser renderizadas

## Próximos Passos

1. ✅ Cache limpo
2. ✅ Logs adicionados
3. ⏳ Testar no frontend
4. ⏳ Verificar logs no Console
5. ⏳ Verificar se botão aparece

