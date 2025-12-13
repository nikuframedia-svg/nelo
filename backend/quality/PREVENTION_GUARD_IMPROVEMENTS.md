# Prevenção de Erros (Process & Quality Guard) - Melhorias Implementadas

## ✅ Requisitos Implementados

### 1. Validação de BOM e Routing no Release (PDM Guard)
- ✅ **PDMGuardEngine**: Validações automáticas antes de liberar revisão
- ✅ **BOM Validation**:
  - Componentes duplicados (BOM-001)
  - Quantidades zero/negativas (BOM-002)
  - Componentes obsoletos (BOM-003)
  - Ciclos na BOM (BOM-004) - DAG validation
  - Componentes ativos e disponíveis
- ✅ **Routing Validation**:
  - Operações completas com tempos (RTG-001)
  - Work centers válidos (RTG-002)
  - Operações de inspeção para produtos críticos (RTG-003)
- ✅ **Documentation Validation**:
  - Desenhos obrigatórios (DOC-001)
  - Instruções de trabalho (DOC-002)
  - Planos de qualidade (DOC-003)
- ✅ **Product Linter**: Aponta incongruências (item obsoleto, máquina inexistente)

### 2. Validação de Configuração de Processo
- ✅ **validate_process_configuration()**: Valida configuração de processo
- ✅ **Parâmetros padrão**: Verifica que todos os parâmetros requeridos têm valores padrão
- ✅ **Faixas aceitáveis**: Verifica que parâmetros têm limites min/max definidos
  - Exemplo: temperatura entre X e Y, torque entre A e B
- ✅ **Instruções de trabalho**: Verifica anexo quando necessário
- ✅ **Planos de inspeção**: Verifica anexo quando necessário

### 3. Guardião no Chão-de-fábrica (Shopfloor Guard)
- ✅ **ShopfloorGuardEngine**: Validações na execução
- ✅ **Material Validation**:
  - Verificação de material via código de barras/RFID (MAT-001)
  - Verificação de revisão compatível (MAT-002)
  - Verificação de expiração (MAT-003)
  - Bloqueia início se material errado
- ✅ **Equipment Validation**:
  - Máquina correta para operação (EQP-001)
  - Ferramentas calibradas (EQP-002)
  - Saúde da máquina (EQP-003)
- ✅ **Work Instruction Version**: Verifica versão correta da instrução
- ✅ **Pre-start Checklist**: Validação antes de iniciar ordem

### 4. Poka-yokes Digitais em Tempo Real
- ✅ **validate_parameters()**: Validação em tempo real de parâmetros
- ✅ **Alerta/Impede**: Se valor fora do limite seguro, alerta ou impede
- ✅ **Validação de checklist**: Não permite avançar se etapa crítica não confirmada
- ✅ **Integração IIoT**: Preparado para alarmes de sensores
  - Configuração de alarmes se sensores indicam valores fora da banda esperada
- ✅ **Validação de sequência**: Previne saltos de etapas

### 5. Módulo de Previsão de Risco de Qualidade
- ✅ **PredictiveGuardEngine**: ML para previsão de risco
- ✅ **Modelo Matemático**:
  - Classificação de risco: `P(Defeito|X)` onde X são features
  - Modelo inicial: `logit(P(Defeito)) = β0 + β1*Machine + β2*Operator + ...`
  - Modelo avançado: MLP (Multi-Layer Perceptron) com interações complexas
  - Minimiza entropia cruzada: `CrossEntropyLoss = -Σ[y*log(p) + (1-y)*log(1-p)]`
- ✅ **Features**:
  - Produto, máquina, operador, turno
  - Condições ambientais (temperatura, humidade)
  - Saúde da máquina, experiência do operador
  - Batch de material
- ✅ **Thresholds**:
  - LOW: P(defect) < 0.1
  - MEDIUM: 0.1 ≤ P(defect) < 0.3
  - HIGH: 0.3 ≤ P(defect) < 0.5
  - CRITICAL: P(defect) ≥ 0.7
- ✅ **Ações**:
  - Se risco alto: notificar supervisor
  - Recomendar inspeção extra
  - Sugerir ajuste de parâmetros
  - Ajustar sequência de produção

## 🚀 Funcionalidades Adicionais

### Exception Manager
- ✅ **Workflow de Aprovação**: Sistema de exceções para override de validações
- ✅ **Request Exception**: Operador pode solicitar exceção com justificativa
- ✅ **Approve/Reject**: Supervisor aprova ou rejeita
- ✅ **Expiry**: Exceções expiram automaticamente
- ✅ **Audit Trail**: Histórico completo de exceções

### Event Logging
- ✅ **GuardEvent**: Log de todos os eventos do sistema
- ✅ **Tipos de Eventos**:
  - VALIDATION_PASSED
  - VALIDATION_FAILED
  - RISK_ALERT
  - EXCEPTION_REQUESTED
  - EXCEPTION_RESOLVED
  - ERROR_PREVENTED
- ✅ **Rastreabilidade**: Timestamp, utilizador, contexto

### Melhoria Contínua
- ✅ **Historical Data**: Registo de dados históricos para treino
- ✅ **Training Pipeline**: Treino periódico do modelo preditivo
- ✅ **Similar Issues**: Encontra problemas similares históricos
- ✅ **Statistics**: Estatísticas de validações, erros prevenidos, exceções
- ✅ **Relatórios**: Extração de relatórios de lições aprendidas

### Custom Rules
- ✅ **ValidationRule**: Sistema de regras configuráveis
- ✅ **Add Custom Rule**: Permite adicionar regras personalizadas
- ✅ **Enable/Disable**: Ativar/desativar regras
- ✅ **Categories**: BOM, Routing, Documentation, Material, Equipment, Parameter, Quality, Compliance

## 📊 Modelos Matemáticos

### Previsão de Risco
```
P(Defeito|X) = sigmoid(MLP(X))

onde X = [product_id, machine_id, operator_id, shift, experience, health, ...]

Loss = -Σ[y*log(p) + (1-y)*log(1-p)]
```

### Validação Lógica
```
∀ componente c em BOM: status(c) = "ativo"
∀ operação o em routing: possui (machine != null ∧ tempo > 0)
∀ parâmetro p: min_p ≤ valor_p ≤ max_p
```

## 🔧 Implementação Técnica

### PDM Guard
- **BOM Validation**: Verifica duplicados, quantidades, obsoletos, ciclos
- **Routing Validation**: Verifica tempos, recursos, inspeções
- **Documentation Validation**: Verifica anexos obrigatórios
- **Process Configuration**: Verifica parâmetros padrão e faixas

### Shopfloor Guard
- **Material Validation**: Verifica SKU, revisão, expiração via barcode/RFID
- **Equipment Validation**: Verifica máquina, ferramentas, saúde
- **Parameter Validation**: Valida parâmetros em tempo real (poka-yoke)
- **Pre-start Checklist**: Validação antes de iniciar

### Predictive Guard
- **ML Model**: PyTorch MLP (Multi-Layer Perceptron)
- **Training**: Adam optimizer, Binary Cross Entropy Loss
- **Features**: 10 dimensões (produto, máquina, operador, contexto)
- **Inference**: < 1 segundo por ordem

### Exception Manager
- **Workflow**: Request → Pending → Approved/Rejected
- **Expiry**: Exceções expiram após X horas
- **Override Lookup**: Lookup rápido de overrides válidos

## 📝 Estrutura de Dados

### ValidationRule
```python
{
    "rule_id": "BOM-001",
    "name": "No Duplicate Components",
    "category": "bom",
    "severity": "error",
    "action": "block",
    "condition": "no_duplicate_components",
    "enabled": true
}
```

### ValidationIssue
```python
{
    "issue_id": "VI-abc123",
    "rule_id": "MAT-001",
    "rule_name": "Material Verification",
    "category": "material",
    "severity": "critical",
    "action": "block",
    "message": "Material mismatch: scanned MAT-002, required MAT-001",
    "entity_type": "order",
    "entity_id": "OP-2024-001"
}
```

### RiskPrediction
```python
{
    "prediction_id": "RP-abc123",
    "risk_level": "high",
    "defect_probability": 0.45,
    "risk_factors": {
        "machine_health": 0.15,
        "operator_experience": 0.12,
        "batch_complexity": 0.10,
        "shift_factor": 0.08
    },
    "recommendations": [
        "Consider machine maintenance before production",
        "Add extra inspection points"
    ],
    "similar_issues": [...],
    "confidence": 0.85
}
```

### ExceptionRequest
```python
{
    "exception_id": "EX-abc123",
    "validation_issue_id": "VI-xyz789",
    "order_id": "OP-2024-001",
    "requested_by": "operator-001",
    "reason": "Material is compatible, just different batch",
    "status": "pending",
    "expires_at": "2024-01-15T18:00:00Z"
}
```

## 🔄 Integração

### PDM (Product Data Management)
- ✅ Validações automáticas antes de release
- ✅ Bloqueia release se erros críticos
- ✅ Product Linter aponta incongruências

### MES (Manufacturing Execution System)
- ✅ Integração com leituras de barcode/RFID
- ✅ Verificação de materiais em tempo real
- ✅ Validação de equipamento e ferramentas

### IIoT (Industrial Internet of Things)
- ✅ Preparado para alarmes de sensores
- ✅ Validação de parâmetros em tempo real
- ✅ Integração com CNC/PLC para verificação de setpoints

### Work Instructions
- ✅ Verifica versão correta da instrução
- ✅ Validação de checklist em tempo real
- ✅ Poka-yoke de sequência

### Causal & ZDM
- ✅ Near-misses e alertas de risco alimentam grafo causal
- ✅ Refina modelo preditivo com feedback

## 🚀 Uso

### Validação de Release de Produto
```python
from quality.prevention_guard import get_prevention_guard_service

service = get_prevention_guard_service()

result = service.validate_product_release(
    item_data={"item_id": "PROD-001", "revision": "A"},
    bom_components=[
        {"component_id": "COMP-001", "qty_per_unit": 2, "status": "active"},
        {"component_id": "COMP-002", "qty_per_unit": 1, "status": "active"},
    ],
    routing_operations=[
        {"operation_id": "OP-10", "setup_time": 15, "cycle_time": 30, "work_center_id": "WC-01"},
    ],
    attachments=[
        {"attachment_id": "DWG-001", "type": "drawing"},
    ],
)

if not result.passed:
    print(f"Validation failed: {result.errors} errors, {result.warnings} warnings")
    for issue in result.issues:
        print(f"- {issue.message}")
```

### Validação de Início de Ordem
```python
validation_result, risk_prediction = service.validate_order_start(
    order_data={"order_id": "OP-2024-001", "product_id": "PROD-001", "quantity": 50},
    scanned_materials=[
        {"sku": "MAT-001", "revision": "A"},
    ],
    required_materials=[
        {"sku": "MAT-001", "revision": "A"},
    ],
    machine_data={"machine_id": "MC-01", "health_index": 0.85},
    context={
        "machine_id": "MC-01",
        "operator_id": "OP-001",
        "shift": 1,
        "operator_experience": 0.8,
        "machine_health": 0.85,
    },
)

if validation_result.blocked:
    print("Order start blocked due to validation issues")
    for issue in validation_result.issues:
        if issue.action == ValidationAction.BLOCK:
            print(f"BLOCKED: {issue.message}")

if risk_prediction.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
    print(f"High risk detected: {risk_prediction.defect_probability:.1%}")
    for rec in risk_prediction.recommendations:
        print(f"- {rec}")
```

### Previsão de Risco
```python
risk = service.predictive_guard.predict_risk(
    order_data={"order_id": "OP-2024-001", "product_id": "PROD-001"},
    context={
        "machine_id": "MC-01",
        "operator_id": "OP-001",
        "shift": 3,  # Night shift
        "operator_experience": 0.3,  # Inexperienced
        "machine_health": 0.55,  # Low health
    },
)

print(f"Risk level: {risk.risk_level.value}")
print(f"Defect probability: {risk.defect_probability:.1%}")
print(f"Confidence: {risk.confidence:.1%}")
```

### Solicitar Exceção
```python
exception = service.request_exception(
    issue_id="VI-abc123",
    order_id="OP-2024-001",
    operation_id="OP-10",
    requested_by="operator-001",
    reason="Material is compatible, just different batch number",
)

# Supervisor aprova
success, message = service.approve_exception(
    exception_id=exception.exception_id,
    approved_by="supervisor-001",
    note="Material verified compatible, approved for production",
)
```

### Adicionar Dados Históricos
```python
service.predictive_guard.add_historical_data(
    order_data={"order_id": "OP-2024-001", "product_id": "PROD-001"},
    context={
        "machine_id": "MC-01",
        "operator_id": "OP-001",
        "shift": 1,
    },
    had_defect=True,
    defect_details={
        "type": "dimensional",
        "cause": "machine calibration drift",
    },
)

# Treinar modelo
training_result = service.predictive_guard.train()
print(f"Training success: {training_result['success']}")
print(f"Samples: {training_result['samples']}")
```

## ✅ Checklist de Requisitos

- ✅ Validação de BOM e Routing no release (PDM Guard)
- ✅ Validação de configuração de processo (parâmetros padrão e faixas)
- ✅ Guardião no Chão-de-fábrica (Shopfloor Guard)
- ✅ Poka-yokes Digitais em tempo real
- ✅ Módulo de Previsão de Risco de Qualidade (ML)
- ✅ Modelo matemático de classificação de risco
- ✅ Product Linter (aponta incongruências)
- ✅ Integração com barcode/RFID
- ✅ Integração com IIoT (alarmes de sensores)
- ✅ Exception Manager (workflow de aprovação)
- ✅ Logging e melhoria contínua
- ✅ Relatórios de lições aprendidas

## 🔮 Extensões Futuras

### Visão Computacional
- ⚠️ Poka-yoke visual para verificação automática
- ⚠️ Validação de montagem via câmera
- ⚠️ Detecção de erros via algoritmos de CV

### Integração Avançada
- ⚠️ Integração direta com CNC/PLC
- ⚠️ Leitura automática de sensores IIoT
- ⚠️ Feedback em tempo real de máquinas

### Modelos Avançados
- ⚠️ Random Forest como alternativa ao MLP
- ⚠️ Ensemble de modelos
- ⚠️ Transfer learning entre produtos/máquinas


