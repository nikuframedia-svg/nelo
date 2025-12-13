# Trust Index Avançado para Duplios - Melhorias Implementadas (Contrato D1)

## ✅ Requisitos Implementados

### 1. Modelo de Dados Trust Index
- ✅ **DataSourceType Enum**: MEDIDO, REPORTADO, ESTIMADO, DESCONHECIDO
- ✅ **FieldTrustMeta Model**: Metadados de confiança por campo
  - field_key, base_class, fractions (measured/reported/estimated/unknown)
  - recency_days, third_party_verified, uncertainty_relative
  - materiality_weight, consistency_zscore
- ✅ **DPPTrustResult Model**: Resultado completo com breakdown
  - overall_trust_index (0-100)
  - field_scores (dict[str, float])
  - field_metas (dict[str, FieldTrustMeta])
  - key_messages (list[str]) para UI

### 2. Serviço de Cálculo
- ✅ **TrustIndexService**: Classe principal com `calculate_for_dpp()`
- ✅ **Algoritmo de Cálculo**:
  - Base score por tipo: MEDIDO=100, REPORTADO=85, ESTIMADO=65, DESCONHECIDO=0
  - Fatores de ajuste:
    - Recência (A): f_A = 1.0 (<1 ano), 0.95 (1-2 anos), 0.9 (2-3 anos), 0.85 (>3 anos)
    - Verificação 3rd-party (B): f_B = 1.1 (auditado), 1.0 (sem auditoria), 0.8 (conflito)
    - Incerteza (C): f_C = 1.05 (<0.1), 1.0 (0.1-0.2), 0.9 (0.2-0.5), 0.75 (>0.5)
    - Consistência vs peers (E): f_E = 1.0 (|z|<1), 0.95 (1<|z|<2), 0.8 (|z|>=2)
  - Score de campo: `score_field_raw = base_score * f_A * f_B * f_C * f_E` (truncado 0-100)
  - Ponderação global: `overall_trust = Σ_i (score_field_i * w_i)` onde w_i é materiality_weight
- ✅ **Persistência**: Atualiza `trust_index` e `trust_meta_json` na tabela `dpp_records`
- ✅ **Inferência de DataSourceType**: 
  - Verifica metadados em `additional_data.trust_meta`
  - Heurística: valores None/0 = DESCONHECIDO, default = ESTIMADO

### 3. API
- ✅ **GET /duplios/dpp/{dpp_id}/trust-index**: Retorna `DPPTrustResult` completo
- ✅ **POST /duplios/dpp/{dpp_id}/trust-index/recalculate**: Força recálculo
- ✅ Integrado no `api.py` principal

### 4. Integração com R&D
- ✅ **WPX_TRUST_EVOLUTION**: Tipo de experimento definido em `WorkPackage` enum
- ✅ **Tabela específica**: `rd_wpx_trust_evolution` criada para armazenar evoluções
- ✅ **Logging automático**: Quando mudança > 5 pontos, registra:
  - dpp_id, trust_index_old, trust_index_new, change, cause, field_scores, timestamp
- ✅ **log_experiment_event()**: Melhorada para salvar na tabela específica

### 5. Frontend
- ✅ **DPPViewer**: 
  - Badge com `overall_trust_index` no topo (já existia, atualizado)
  - Secção "Trust Index - Breakdown" com:
    - Mensagens chave (key_messages)
    - Tabela compacta: Campo | Score | Tipo | Última Atualização
- ✅ **DPPList**:
  - Coluna "Trust Index" visível (já existia)
  - Ordenação por Trust Index (asc/desc)
  - Filtro por Trust Index (≥80, ≥60, ≥40)
- ✅ **UI Simples**: Não expõe detalhes matemáticos, apenas resultados claros

## 📊 Modelo Matemático

### Score de Campo
```
base_score = {
    MEDIDO: 100,
    REPORTADO: 85,
    ESTIMADO: 65,
    DESCONHECIDO: 0
}

f_A = {
    < 365 dias: 1.0,
    365-729 dias: 0.95,
    730-1094 dias: 0.9,
    ≥ 1095 dias: 0.85
}

f_B = {
    third_party_verified: 1.1,
    sem auditoria: 1.0,
    conflito: 0.8
}

f_C = {
    < 0.1: 1.05,
    0.1-0.2: 1.0,
    0.2-0.5: 0.9,
    ≥ 0.5: 0.75
}

f_E = {
    |z| < 1: 1.0,
    1 ≤ |z| < 2: 0.95,
    |z| ≥ 2: 0.8
}

score_field = clamp(base_score * f_A * f_B * f_C * f_E, 0, 100)
```

### Trust Index Global
```
overall_trust = Σ_i (score_field_i * w_i)

onde w_i são materiality_weights:
- carbon_footprint_kg_co2eq: 0.40
- water_m3: 0.25
- energy_kwh: 0.15
- recycled_content_pct: 0.10
- recyclability_pct: 0.10
```

## 🔧 Implementação Técnica

### Backend
- **trust_index_models.py**: Modelos Pydantic (DataSourceType, FieldTrustMeta, DPPTrustResult)
- **trust_index_service.py**: Lógica de cálculo e persistência
- **api_trust_index.py**: Endpoints REST
- **Integração R&D**: Logging automático para WPX_TRUST_EVOLUTION

### Frontend
- **DPPViewer.tsx**: Breakdown de Trust Index com tabela
- **DPPList.tsx**: Ordenação e filtro por Trust Index
- **API Integration**: Fetch de `/duplios/dpp/{dpp_id}/trust-index`

## 📝 Estrutura de Dados

### FieldTrustMeta
```python
{
    "field_key": "carbon_footprint_kg_co2eq",
    "base_class": "MEDIDO",
    "measured_fraction": 0.6,
    "reported_fraction": 0.4,
    "estimated_fraction": 0.0,
    "unknown_fraction": 0.0,
    "recency_days": 120,
    "third_party_verified": true,
    "uncertainty_relative": 0.15,
    "materiality_weight": 0.40,
    "consistency_zscore": 0.5,
    "field_score": 95.2,
    "last_updated": "2024-01-15T10:00:00Z"
}
```

### DPPTrustResult
```python
{
    "dpp_id": "uuid-here",
    "overall_trust_index": 82.5,
    "field_scores": {
        "carbon_footprint_kg_co2eq": 95.2,
        "water_m3": 78.5,
        "energy_kwh": 65.0,
        "recycled_content_pct": 70.0,
        "recyclability_pct": 75.0
    },
    "field_metas": {
        "carbon_footprint_kg_co2eq": { ... },
        ...
    },
    "key_messages": [
        "Carbono: base medido + auditado",
        "Água: base estimado",
        "Reciclabilidade: base reportado"
    ],
    "calculated_at": "2024-01-15T10:00:00Z",
    "calculation_version": "1.0"
}
```

## 🔄 Integração

### Duplios DPP
- ✅ Calcula Trust Index ao criar/atualizar DPP
- ✅ Persiste em `dpp_records.trust_index` e `additional_data.trust_meta_json`
- ✅ Recalcula automaticamente após edições

### R&D Module
- ✅ Logs evoluções significativas (>5 pontos) para análise
- ✅ Armazena em `rd_wpx_trust_evolution` table
- ✅ Permite análise de tendências e causas

### Frontend
- ✅ Exibe Trust Index em listagem e detalhe
- ✅ Permite ordenação e filtro por confiança
- ✅ Mostra breakdown campo-a-campo

## 🚀 Uso

### Backend
```python
from duplios.trust_index_service import get_trust_index_service
from duplios.dpp_models import DppRecord

service = get_trust_index_service()

# Calcular Trust Index
result = service.calculate_for_dpp(dpp, db_session=db)

print(f"Overall Trust Index: {result.overall_trust_index}")
print(f"Field Scores: {result.field_scores}")
for msg in result.key_messages:
    print(f"- {msg}")
```

### API
```bash
# Obter Trust Index
GET /duplios/dpp/123/trust-index

# Recalcular
POST /duplios/dpp/123/trust-index/recalculate
```

### Frontend
- Trust Index visível no badge no topo do DPP
- Breakdown em tabela com scores por campo
- Ordenação e filtro na listagem de DPPs

## ✅ Checklist de Requisitos

- ✅ Modelo de dados Trust Index (DataSourceType, FieldTrustMeta, DPPTrustResult)
- ✅ Serviço de cálculo com algoritmo especificado
- ✅ API endpoints (GET e POST)
- ✅ Integração com R&D (WPX_TRUST_EVOLUTION)
- ✅ Frontend: Badge e breakdown no DPPDetail
- ✅ Frontend: Ordenação e filtro na listagem
- ✅ Persistência em database
- ✅ Logging de evoluções para análise
- ✅ UI simples sem expor detalhes matemáticos

## 🔮 Extensões Futuras

### Multi-Tier
- ⚠️ Integração com supply chain multi-tier
- ⚠️ Trust Index agregado por fornecedor
- ⚠️ Rastreabilidade de dados upstream

### ZKP (Zero-Knowledge Proofs)
- ⚠️ Verificação criptográfica de dados
- ⚠️ Trust Index baseado em provas ZKP
- ⚠️ Privacidade preservada

### Machine Learning
- ⚠️ Predição de Trust Index baseada em padrões
- ⚠️ Detecção de anomalias em dados
- ⚠️ Sugestões automáticas de melhoria


