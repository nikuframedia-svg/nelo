# PDM (Product Data Management Core) - Melhorias Implementadas

## ✅ Requisitos Implementados

### 1. Items e Revisões
- ✅ **Item**: Master record para produtos, componentes, matérias-primas
  - SKU único, nome, tipo (FINISHED, SEMI_FINISHED, RAW_MATERIAL, TOOLING, PACKAGING)
  - Unidade, família, peso
- ✅ **ItemRevision**: Versões controladas de cada item
  - Código único (A, B, C, 01, 02, etc.)
  - Status: DRAFT, RELEASED, OBSOLETE
  - Datas de vigência (effective_from, effective_to)
  - Notas e metadados

### 2. BOM (Bill of Materials)
- ✅ **BomLine**: Estrutura hierárquica de componentes
  - Parent revision → Component revision
  - Quantidade por unidade (qty_per_unit)
  - Percentual de refugo (scrap_rate)
  - Posição e notas opcionais
- ✅ **Validação de BOM**:
  - Detecção de ciclos (DAG validation via DFS)
  - Verificação de componentes válidos
  - Verificação de status de componentes (não permitir Draft em Released BOM)
  - Verificação de quantidades válidas (> 0)
- ✅ **BOM Explosion**: Explosão recursiva multi-nível
- ✅ **Integridade referencial**: Componentes devem existir e estar válidos

### 3. Roteiro de Fabrico (Routing)
- ✅ **RoutingOperation**: Sequência de operações
  - Código de operação (op_code)
  - Sequência ordenada (10, 20, 30...)
  - Máquina/grupo responsável (machine_group)
  - Tempos: setup (nominal_setup_time), ciclo (nominal_cycle_time)
  - Ferramentas (tool_id)
  - Flags: is_critical, requires_inspection
- ✅ **Validação de Routing**:
  - Sequência completa e ordenada
  - Tempos preenchidos
  - Recursos existentes
  - Work instructions para operações críticas

### 4. Fluxo ECO/ECR
- ✅ **ECR (Engineering Change Request)**:
  - Título, descrição, motivo
  - Prioridade (LOW, MEDIUM, HIGH, CRITICAL)
  - Status: OPEN, CLOSED
  - Requestor e datas
- ✅ **ECO (Engineering Change Order)**:
  - Implementa ECR criando nova revisão
  - Transição: from_revision → to_revision
  - Aprovação e implementação
  - Histórico completo de mudanças
- ✅ **Impact Analysis**: Análise de impacto de mudanças
  - Itens afetados
  - Ordens de produção abertas
  - DPPs afetados

### 5. Anexos de Engenharia
- ✅ **Attachment**: Modelo para anexos
  - Nome do ficheiro, tipo (CAD, PDF, WORK_INSTRUCTION, QUALITY_PLAN, etc.)
  - Caminho/URL para armazenamento externo
  - Tamanho, MIME type
  - Metadados (descrição, uploader, data)
  - Relacionado com ItemRevision
- ✅ **Validação de Attachments**:
  - Verificação de anexos obrigatórios por tipo de item
  - Configurável por tipo (FINISHED requer CAD, etc.)

### 6. Workflow de Release
- ✅ **Validações Automáticas** antes de liberar:
  - BOM não contém ciclos (DAG)
  - BOM não contém itens inativos/obsoletos
  - BOM componentes são válidos (preferencialmente Released)
  - Routing está completo
  - Work instructions para operações críticas
  - Attachments obrigatórios presentes
- ✅ **Bloqueio de Edições**:
  - Revisões RELEASED não podem ser editadas diretamente
  - Requer nova revisão via ECO para mudanças
- ✅ **Auto-Obsoleção**:
  - Ao liberar nova revisão, revisões anteriores são automaticamente obsoletas
  - Configurável via `auto_obsolete_on_release`
- ✅ **Notificações e Sinalização**:
  - Notificação quando revisão é liberada
  - Sinalização de ordens abertas afetadas
  - Sinalização de stock em curso afetado
  - Evita uso inadvertido de versões antigas

## 📊 Modelo de Dados

### BOM como DAG (Grafo Direcionado Acíclico)
```
G = (V, E)
- V: Conjunto de revisões de item
- E: Arcos (u → v) indicando que revisão u possui v como componente
- Restrição: Não permitir ciclos (validação via DFS)
```

### Routing como Sequência Ordenada
```
R = [(Op1, recurso1, t1), (Op2, recurso2, t2), ...]
- Sequência linear com precedência implícita
- Operações ordenadas por sequence (10, 20, 30...)
```

### Integridade Referencial
- ∀ componente em BOM, deve existir revisão válida
- Componentes obsoletos não podem aparecer em BOM ativa
- Máximo uma revisão RELEASED por item (configurável)

## 🚀 Funcionalidades

### Validações Automáticas
```python
from duplios.pdm_core import PDMService, PDMConfig

service = PDMService()

# Validar antes de liberar
validation = service.validate_for_release(revision_id)
if validation.valid:
    success, revision, validation = service.release_revision(revision_id)
```

### BOM Explosion
```python
# Explodir BOM multi-nível
explosion = service.explode_bom(revision_id, max_depth=10)
for level, component in explosion.components:
    print(f"Level {level}: {component['sku']} x {component['qty']}")
```

### Detecção de Ciclos
```python
# Verificar se BOM tem ciclos
has_cycle, cycle_path = service.detect_cycle(revision_id)
if has_cycle:
    print(f"Cycle detected: {' → '.join(cycle_path)}")
```

### Utility Functions
```python
# Obter revisão atual
current_rev = service.get_current_revision(item_id)

# Obter BOM
bom = service.get_bom(item_id, revision_code="A")

# Obter Routing
routing = service.get_routing(item_id, revision_code="A")
```

### Impact Analysis
```python
# Analisar impacto de mudança
impact = service.analyze_ecr_impact(ecr_id)
print(f"Affected items: {len(impact.affected_items)}")
print(f"Open orders: {len(impact.open_production_orders)}")
```

## 🔄 Integração com Outros Módulos

### ProdPlan
- ✅ Usa apenas revisões RELEASED para planeamento
- ✅ `get_current_revision()` retorna revisão ativa
- ✅ `get_routing()` fornece sequência de operações

### SmartInventory / MRP
- ✅ Usa apenas revisões RELEASED para explosão de BOM
- ✅ `get_bom()` fornece estrutura hierárquica
- ✅ Sinalização de stock afetado por mudanças

### Duplios (DPP)
- ✅ Extrai informações da revisão RELEASED
- ✅ Digital Identity ligada à revisão
- ✅ DPP records atualizados com mudanças

### Operações (Shopfloor)
- ✅ Puxa instruções de trabalho da revisão correta
- ✅ Parâmetros corretos conforme revisão na ordem
- ✅ Work instructions versionadas por revisão

## 📝 Tabelas Principais

1. **pdm_items**: Master items
2. **pdm_item_revisions**: Versões de itens
3. **pdm_bom_lines**: Estrutura BOM
4. **pdm_routing_operations**: Operações de fabrico
5. **pdm_attachments**: Anexos de engenharia
6. **pdm_ecr**: Engineering Change Requests
7. **pdm_eco**: Engineering Change Orders
8. **pdm_work_instructions**: Instruções de trabalho

## 🔒 Segurança e Controle de Acesso

- ✅ **Validações automáticas** bloqueiam release com erros
- ✅ **Workflow controlado**: Draft → Released → Obsolete
- ✅ **Histórico completo**: Todas as mudanças registadas
- ✅ **Integridade garantida**: Transações ACID
- ⚠️ **Nota**: Controle de acesso por utilizador pode ser adicionado conforme necessário

## 📈 Melhorias Implementadas

1. ✅ **Modelo Attachment** adicionado
2. ✅ **Validação de attachments** no release
3. ✅ **Notificações de release** implementadas
4. ✅ **Sinalização de ordens afetadas** (pontos de integração)
5. ✅ **Auto-obsoleção** configurável
6. ✅ **Utility functions** (get_current_revision, get_bom, get_routing)
7. ✅ **Impact analysis** para ECR/ECO
8. ✅ **Revision diff** para comparar revisões

## 🔮 Extensões Futuras

1. **Componentes Alternativos**: Suporte para alternativas na BOM
2. **Versionamento de Attachments**: Histórico de versões de ficheiros
3. **Aprovação Multi-nível**: Workflow de aprovação configurável
4. **Notificações em Tempo Real**: WebSocket ou message queue
5. **Integração Completa**: Queries reais para ProdPlan e SmartInventory
6. **Audit Trail**: Log detalhado de todas as operações


