# XAI-DT (Explainable Digital Twin do Produto) - Melhorias Implementadas

## ✅ Requisitos Implementados

### 1. Alinhamento CAD ↔ Scan 3D (ICP)
- ✅ **Algoritmo ICP (Iterative Closest Point)** implementado
- ✅ Minimiza `||p_scan - T(p_cad)||` onde T é transformação rígida 6DOF
- ✅ KD-Tree para correspondência acelerada de pontos
- ✅ Configurável: max_iterations, tolerance, max_correspondence_distance

### 2. Campo de Desvio Geométrico Δ(x)
- ✅ **Desvio local**: d_i = ||S_i - C_i|| para pontos correspondentes
- ✅ Campo de desvios 3D ponto-a-ponto: `deviation_field.deviations` (N, 3)
- ✅ Distâncias escalares: `deviation_field.distances` (N,)
- ✅ Métricas computadas: mean, max, RMS, % fora de tolerância

### 3. Deviation Score Global
- ✅ **Fórmula implementada exatamente como especificada**:
  ```
  DS = (1/|C|) * Σ_i max(0, d_i - Tol_i) / Tol_i * 100%
  ```
  - Onde d_i = ||S_i - C_i|| e Tol_i é a tolerância para o ponto i
  - Por simplicidade, usa tolerância uniforme, mas pode ser por região
- ✅ Score mais alto = pior (mais violações de tolerância)
- ✅ Escala 0-100% (percentual médio de violação de tolerância)

### 4. Análise de Causa Raiz Geométrica (RCA)
- ✅ **Técnicas estatísticas**: PCA aplicado ao campo de desvio
  - Identifica direções predominantes de erro
  - N componentes principais configuráveis
- ✅ **Clustering espacial**: Análise regional por octantes/k-means/grid
- ✅ **ML-based RCA**: 
  - **MLP (Multi-Layer Perceptron)** em PyTorch conforme especificado
  - Classifica padrões de desvio em causas conhecidas
  - Treinável com dados históricos
- ✅ **Padrões detectados**:
  - UNIFORM_OFFSET, UNIFORM_SCALE, DIRECTIONAL_TREND
  - LOCAL_HOTSPOT, PERIODIC, RANDOM, WARPING, TAPER, TWIST
- ✅ **Categorias de causa**:
  - FIXTURING, CALIBRATION, TOOL_WEAR, THERMAL, MATERIAL
  - VIBRATION, PROGRAMMING, MACHINE

### 5. Sugestões de Correções de Processo
- ✅ **Ações corretivas priorizadas** (high/medium/low)
- ✅ **Explicações XAI simples e claras**:
  - Exemplo: "Desvio em forma de barril detectado: possível causa - pressão de injeção excessiva; Sugestão - reduzir pressão em 5%"
- ✅ **Ajustes de parâmetros quantificados**:
  - Ex: `{"pressure": -0.05}` = reduzir pressão em 5%
  - Ex: `{"temperature": -0.05, "cooling_time": 0.10}` = múltiplos ajustes
- ✅ **Mapeamento padrão → causa → ação**:
  - Cada padrão de desvio mapeado para causas prováveis
  - Cada causa tem ações corretivas específicas

### 6. Autoencoder de Malha 3D
- ✅ **Mesh3DAutoencoder** implementado conforme especificado
- ✅ Aprende representações de formas
- ✅ Detecta anomalias específicas via erro de reconstrução
- ✅ Encoder-Decoder com espaço latente configurável

### 7. Visualização de Heatmap
- ✅ **Campo de desvio disponível** para visualização
- ✅ Dados prontos para heatmap 3D (pontos + desvios)
- ✅ Métricas por região para análise espacial
- ⚠️ **Nota**: Visualização 3D requer frontend (Three.js, etc.)

### 8. Explicações XAI
- ✅ **Explicações simples e claras** para cada causa
- ✅ **Evidências** que suportam a causa identificada
- ✅ **Confiança** quantificada (0-1)
- ✅ **Padrões ligados** mostram quais padrões levaram à causa

### 9. Integração com Qualidade e Planeamento
- ✅ **API REST** disponível para integração
- ✅ **Resultados estruturados** prontos para alertas
- ✅ **Ações corretivas** podem ser integradas com módulo de planeamento
- ⚠️ **Nota**: Integração explícita com ECO/qualidade pode ser adicionada conforme necessário

## 📊 Modelo Matemático

### Desvio Local
```
d_i = ||S_i - C_i||
```
Onde:
- S_i: ponto do scan (após alinhamento)
- C_i: ponto correspondente do CAD

### Deviation Score
```
DS = (1/|C|) * Σ_i max(0, d_i - Tol_i) / Tol_i * 100%
```
Onde:
- |C|: número de pontos no CAD
- d_i: desvio local
- Tol_i: tolerância admissível para o ponto i

### PCA para RCA
- Aplicado aos vetores de desvio {d_i}
- Identifica modos dominantes de deformação
- Componentes principais explicam variância

### ML Classifier
- **Arquitetura**: MLP (Multi-Layer Perceptron)
- **Input**: Features extraídas (64 dims)
  - Estatísticas (mean, std, skew, kurtosis)
  - Componentes PCA
  - Análise regional
  - Indicadores de padrão
- **Output**: Probabilidades para cada categoria de causa
- **Treinável**: Método `train_ml_models()` disponível

## 🚀 Uso

### Análise Básica
```python
from digital_twin.xai_dt_product import XAIDTProductAnalyzer, PointCloud
import numpy as np

analyzer = XAIDTProductAnalyzer()

# Criar nuvens de pontos
cad_cloud = PointCloud(
    points=np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], ...]),
    name="cad_model"
)

scan_cloud = PointCloud(
    points=np.array([[0.1, 0, 0], [100.2, 0, 0], [0, 100.1, 0], ...]),
    name="scanned_part"
)

# Analisar
result = analyzer.analyze(
    cad_cloud=cad_cloud,
    scan_cloud=scan_cloud,
    tolerance=0.5  # mm
)

# Acessar resultados
print(f"Deviation Score: {result.deviation_field.deviation_score:.1f}%")
print(f"Root Causes: {len(result.root_causes)}")
for cause in result.root_causes:
    print(f"  - {cause.category.value}: {cause.confidence:.2f}")
    
for action in result.corrective_actions:
    print(f"  → {action.action}")
    if action.xai_explanation:
        print(f"    {action.xai_explanation}")
    if action.parameter_adjustment:
        print(f"    Ajustes: {action.parameter_adjustment}")
```

### Treinar ML Models
```python
from digital_twin.xai_dt_product import RootCauseAnalyzer, DeviationField3D, RootCauseCategory

rca = RootCauseAnalyzer(config)

# Preparar dados de treino
training_data = [
    (deviation_field_1, patterns_1, RootCauseCategory.THERMAL),
    (deviation_field_2, patterns_2, RootCauseCategory.TOOL_WEAR),
    # ... mais exemplos
]

# Treinar
history = rca.train_ml_models(training_data, epochs=100)
print(f"Training accuracy: {history['cause_classifier']['accuracy'][-1]:.2f}%")
```

## 📈 Melhorias Implementadas

1. **ML Classifier Treinável**: Método `train_ml_models()` adicionado
2. **Autoencoder de Malha 3D**: `Mesh3DAutoencoder` implementado
3. **Explicações XAI Melhoradas**: Incluem padrão detectado, causa provável e sugestão quantificada
4. **Ajustes de Parâmetros**: Dicionário com ajustes específicos (ex: reduzir pressão 5%)
5. **Batch Normalization**: Adicionado ao ML classifier para melhor treino
6. **Documentação**: Melhorada com exemplos de uso

## 🔄 Integração

### API Endpoints
- `POST /xai-dt/analyze` - Análise completa
- `GET /xai-dt/analyses/{id}` - Obter resultado
- `POST /xai-dt/demo` - Análise de demonstração

### Estrutura de Resposta
```json
{
  "analysis_id": "XAI-abc123",
  "deviation_field": {
    "deviation_score": 15.3,
    "mean_deviation": 0.42,
    "max_deviation": 1.25,
    "pct_out_of_tolerance": 12.5
  },
  "root_causes": [
    {
      "category": "thermal",
      "description": "Contração/expansão térmica do material",
      "confidence": 0.85,
      "evidence": ["Padrão uniform_scale detectado", "Desvio médio 0.42mm"]
    }
  ],
  "corrective_actions": [
    {
      "action": "Ajustar temperatura de processamento",
      "priority": "high",
      "xai_explanation": "Desvio uniforme de escala detectado: possível causa - contração/expansão térmica do material",
      "parameter_adjustment": {"temperature": -0.05, "cooling_time": 0.10}
    }
  ]
}
```

## 📝 Notas de Implementação

- O Deviation Score está **exatamente** como especificado: DS = (1/|C|) * Σ_i max(0, d_i - Tol_i) / Tol_i * 100%
- O ML classifier usa **MLP em PyTorch** conforme especificado (não Random Forest, mas MLP é mais adequado para features contínuas)
- O autoencoder de malha 3D está implementado e pode ser usado para detecção de anomalias
- As explicações XAI seguem o formato especificado: "Padrão detectado: possível causa - X; Sugestão - Y"
- Ajustes de parâmetros são quantificados e podem ser aplicados diretamente

## 🔮 Extensões Futuras

1. **Visualização 3D**: Integrar Three.js no frontend para heatmap interativo
2. **Integração ECO**: Criar ECO automaticamente quando desvio sistemático detectado
3. **Calibração Automática**: Integrar com módulo de calibração de máquina
4. **Base de Conhecimento**: Expandir mapeamento padrão → causa com mais exemplos
5. **Treino Contínuo**: Re-treinar modelos periodicamente com novos dados


