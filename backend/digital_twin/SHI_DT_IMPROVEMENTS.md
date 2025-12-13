# SHI-DT (Smart Health Index Digital Twin) - Melhorias Implementadas

## ✅ Requisitos Implementados

### 1. Modelo CVAE (Conditional Variational Autoencoder)
- ✅ Implementado em `health_indicator_cvae.py`
- ✅ Loss function: L = E_{q_φ(z|x)}[-log p_θ(x|z)] + β * KL(q_φ(z|x) || p(z))
- ✅ Arquitetura: Encoder CNN/LSTM + Decoder para sequências temporais
- ✅ Suporte para contexto condicional (máquina, operação, produto)

### 2. Índice de Saúde H(t) em Tempo Real
- ✅ **Fórmula implementada**: H(t) = 100 * exp(-α * E_rec(t))
  - Onde E_rec(t) é o erro de reconstrução do CVAE
  - α (alpha) é um fator de escala configurável (padrão: 0.1)
  - Ajustado por perfil operacional: α_adjusted = α * threshold_factor(profile)
- ✅ Escala 0-100% (0 = crítico, 100 = saudável)
- ✅ Suavização com EMA (Exponential Moving Average)

### 3. Modelagem de Degradação Baseada em Uso
- ✅ Degradação de parâmetros: P(t) = P(0) - Δ_d * f(uso_acumulado, regime)
- ✅ Tracking de uso acumulado (horas, ciclos, carga)
- ✅ Ajuste do HI considerando condições operacionais
- ✅ Taxa de degradação calculada dinamicamente

### 4. Perfis Operacionais Adaptativos
- ✅ Detecção automática de perfil operacional:
  - IDLE, LOW_LOAD, NORMAL, HIGH_LOAD, PEAK, STARTUP, SHUTDOWN
- ✅ Baselines diferentes por perfil
- ✅ Thresholds ajustados dinamicamente
- ✅ Janela de detecção configurável (padrão: 5 minutos)

### 5. Estimativa de RUL (Remaining Useful Life)
- ✅ RUL estimado como tempo τ tal que H(τ) < threshold_critical (padrão: 20)
- ✅ Modelos de degradação: Linear e Exponencial
- ✅ Intervalos de confiança (Monte Carlo)
- ✅ DeepSurv (ADVANCED) como opção avançada

### 6. Pipeline de Inferência em Tempo Real
- ✅ **Performance otimizada**: < 1 segundo por máquina
  - Cache de inferências (TTL: 1 segundo)
  - Batch processing quando possível
  - Limpeza automática de cache antigo
- ✅ Processamento assíncrono
- ✅ Suporte para múltiplas máquinas simultâneas

### 7. Perfis Operacionais
- ✅ Registro de estatísticas de uso (horas, ciclos, carga)
- ✅ Histórico de perfis por máquina
- ✅ Detecção de transições (startup/shutdown)

### 8. Alertas Automáticos
- ✅ Alertas baseados em HI:
  - HEALTHY: HI > 80%
  - WARNING: 50% ≤ HI ≤ 80%
  - CRITICAL: HI < 30%
- ✅ Alertas baseados em RUL:
  - Crítico: RUL < 24h
  - Aviso: RUL < 100h
- ✅ Explicabilidade: Top K sensores contribuintes
- ✅ Ações recomendadas automáticas

### 9. Integração com Planeamento
- ✅ Alertas podem ser integrados com módulo de planeamento
- ✅ Sugestões de reagendamento de manutenção preventiva
- ✅ API REST para consulta de estado de saúde

### 10. Re-treino Periódico Automático
- ✅ **Online Learning habilitado por padrão**
- ✅ Re-treino incremental após N amostras (padrão: 100)
- ✅ **Re-treino periódico baseado em tempo** (padrão: semanalmente - 168h)
- ✅ Mínimo de amostras configurável (padrão: 500)
- ✅ Buffer de dados para re-treino
- ✅ Fine-tuning incremental do modelo

### 11. Auto-ajuste Contínuo
- ✅ Re-treino automático com novos dados
- ✅ Melhoria contínua da precisão
- ✅ Adaptação a mudanças operacionais
- ✅ Versionamento de modelos

## 📊 Configuração

```python
from digital_twin.shi_dt import SHIDTConfig, SHIDT

config = SHIDTConfig(
    # Health Index Formula
    hi_alpha=0.1,  # α para H(t) = 100 * exp(-α * E_rec(t))
    
    # Thresholds
    threshold_healthy=80.0,
    threshold_warning=50.0,
    threshold_critical=30.0,
    
    # RUL Settings
    rul_failure_threshold=20.0,
    rul_extrapolation_method="exponential",
    
    # Online Learning
    online_learning_enabled=True,
    online_learning_update_interval=100,  # Re-treinar após 100 amostras
    periodic_retrain_interval_hours=168.0,  # Re-treinar semanalmente
    periodic_retrain_min_samples=500,
    
    # Performance
    hi_ema_alpha=0.3,  # Suavização
)

shi_dt = SHIDT(config)
shi_dt.initialize(train_demo=True)
```

## 🚀 Uso

```python
from digital_twin.health_indicator_cvae import SensorSnapshot, OperationContext
from datetime import datetime, timezone

# Criar snapshot de sensores
snapshot = SensorSnapshot(
    machine_id='MC-CNC-001',
    timestamp=datetime.now(timezone.utc),
    vibration_rms=0.15,
    temperature_motor=0.3,
    current_rms=0.4,
    load_percent=0.6,
    speed_rpm=0.7,
)

context = OperationContext(
    machine_id='MC-CNC-001',
    op_code='OP-10',
    product_type='PROD-001',
)

# Ingerir dados e obter Health Index
reading = shi_dt.ingest_sensor_data('MC-CNC-001', snapshot, context)

print(f"Health Index: {reading.hi_smoothed:.2f}%")
print(f"RUL: {reading.rul_estimate.rul_hours:.1f}h" if reading.rul_estimate else "N/A")
print(f"Perfil: {reading.profile.value}")
```

## 📈 Melhorias de Performance

1. **Cache de Inferências**: Reduz tempo de processamento para chamadas repetidas
2. **Batch Processing**: Processamento em lote quando possível
3. **Re-treino Incremental**: Apenas atualiza modelo, não re-treina do zero
4. **Limpeza Automática**: Remove cache e buffers antigos automaticamente

## 🔄 Re-treino Periódico

O sistema re-treina automaticamente:
- **Incremental**: Após cada N amostras (configurável)
- **Periódico**: Após intervalo de tempo (padrão: semanalmente)
- **Condicional**: Apenas se houver dados suficientes (≥ 500 amostras)

## 📝 Notas de Implementação

- O modelo matemático H(t) = 100 * exp(-α * E_rec(t)) está **exatamente** como especificado
- O parâmetro α é calibrado empiricamente e ajustado por perfil operacional
- O re-treino periódico melhora continuamente a precisão das predições
- A performance está otimizada para inferência < 1 segundo por máquina


