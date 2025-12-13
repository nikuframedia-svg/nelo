# Gap Filling Lite para Duplios - Melhorias Implementadas (Contrato D2)

## ✅ Requisitos Implementados

### 1. Tabela de Fatores Internos
- ✅ **gap_factors.yaml**: Base de dados YAML com fatores por material, país e idade tecnológica
  - Materiais: steel, aluminum, polypropylene, cotton, polyester, etc.
  - Países: PT, PL, DE, FR, ES, IT, UK, US, CN, IN, BR, DEFAULT
  - Tech Age: young (<5 anos), mid (5-15 anos), old (>15 anos)
- ✅ **Fallback**: Se YAML não disponível, usa fatores padrão hardcoded

### 2. Serviço GapFillingLite
- ✅ **GapFillingLiteService**: Classe principal com `fill_for_dpp()`
- ✅ **Algoritmo**:
  - Extrai composição do DPP (materiais + massas)
  - Para cada material: obtém fatores base, calcula CO2/água/energia
  - Aplica ajustes contextuais:
    - País: multiplica CO2 por `energy_co2_factor_vs_eu`
    - Tech Age: multiplica CO2 por fator de idade
  - Soma totais e calcula reciclabilidade (média ponderada)
  - Calcula incerteza: ±30% (ou maior se contexto fraco)
  - Atualiza DPP apenas se campos estiverem vazios (ou force=True)
  - Marca campos como `source = "estimated_lite"` em metadata
- ✅ **Integração Trust Index**: Recalcula automaticamente após gap fill
- ✅ **Integração R&D**: Logs para WPX_GAPFILL_LITE

### 3. API & Hooks
- ✅ **POST /duplios/dpp/{dpp_id}/gap-fill-lite**: Endpoint para preencher campos
  - Parâmetro `force`: se True, sobrescreve valores existentes
  - Retorna campos preenchidos, valores, incerteza, contexto
- ✅ **Hooks automáticos**: Preparado para integrar em create/update DPP
  - (Nota: Integração automática pode ser adicionada no service.py se necessário)

### 4. Integração com Trust Index
- ✅ **Metadata**: Campos preenchidos marcados como `ESTIMADO`
- ✅ **Uncertainty**: `uncertainty_relative = 0.3` (ou maior)
- ✅ **Recálculo**: Trust Index recalculado automaticamente após gap fill

### 5. Integração com R&D
- ✅ **WPX_GAPFILL_LITE**: Tipo de experimento definido em `WorkPackage` enum
- ✅ **Tabela específica**: `rd_wpx_gapfill_lite` criada
- ✅ **Logging**: Registra dpp_id, filled_fields, values, uncertainty, context, method

### 6. Frontend
- ✅ **DPPViewer**: 
  - Secção "Estimativas Automáticas"
  - Botão "Preencher automaticamente" se campos em falta
  - Mostra resultado (sucesso/erro) e campos preenchidos
  - Tooltip sobre incerteza (±30%)
- ✅ **UI Simples**: Card com valores e mensagem sobre precisão

## 📊 Modelo Matemático

### Cálculo por Material
```
co2_m = base_co2_kg_per_kg * mass_m_kg
water_m = base_water_m3_per_kg * mass_m_kg
energy_m = base_energy_kwh_per_kg * mass_m_kg
```

### Ajustes Contextuais
```
co2_total_adjusted = Σ(co2_m) * country_factor * tech_age_factor

onde:
- country_factor = energy_co2_factor_vs_eu (do país)
- tech_age_factor = 1.0 (young), 1.1 (mid), 1.3 (old)
```

### Reciclabilidade
```
recyclability_estimated = Σ(recyclability_m * mass_m) / Σ(mass_m)
```

### Incerteza
```
uncertainty = 0.3 (±30%) base
+ 0.1 se país = DEFAULT (desconhecido)
```

## 🔧 Implementação Técnica

### Backend
- **gap_factors.yaml**: Base de dados de fatores
- **gap_filling_lite.py**: Serviço principal
- **api_gap_filling.py**: Endpoints REST
- **Integração R&D**: Logging para WPX_GAPFILL_LITE

### Frontend
- **DPPViewer.tsx**: UI para gap filling
- **dupliosApi.ts**: Função `apiGapFillLite()`

## 📝 Estrutura de Dados

### Gap Fill Result
```python
{
    "success": True,
    "filled_fields": ["carbon_kg_co2eq", "water_m3"],
    "values": {
        "carbon_kg_co2eq": 12.5,
        "water_m3": 3.2
    },
    "uncertainty": {
        "carbon_kg_co2eq": 0.3,
        "water_m3": 0.3
    },
    "context": {
        "country": "PT",
        "country_factor": 0.6,
        "tech_age_factor": 1.0,
        "materials_used": ["steel", "polypropylene"],
        "total_mass_kg": 5.0
    },
    "message": "Filled 2 field(s): carbon_kg_co2eq, water_m3"
}
```

## 🔄 Integração

### Duplios DPP
- ✅ Preenche campos ambientais em falta
- ✅ Atualiza metadata com `source = "estimated_lite"`
- ✅ Recalcula Trust Index automaticamente
- ✅ Não sobrescreve valores medidos/reportados (a menos que force=True)

### Trust Index (Contrato D1)
- ✅ Campos preenchidos marcados como `ESTIMADO`
- ✅ `uncertainty_relative = 0.3` (ou maior)
- ✅ Trust Index recalculado após gap fill

### R&D Module
- ✅ Logs evoluções para análise
- ✅ Armazena em `rd_wpx_gapfill_lite` table
- ✅ Permite comparação futura com GapFillingFull (Ecoinvent)

### Frontend
- ✅ Botão para preencher campos em falta
- ✅ Mostra resultado e campos preenchidos
- ✅ Alerta sobre incerteza (±30%)

## 🚀 Uso

### Backend
```python
from duplios.gap_filling_lite import get_gap_filling_lite_service
from duplios.dpp_models import DppRecord

service = get_gap_filling_lite_service()

# Preencher campos em falta
result = service.fill_for_dpp(dpp, db_session=db, force=False)

print(f"Filled fields: {result['filled_fields']}")
print(f"Values: {result['values']}")
print(f"Uncertainty: {result['uncertainty']}")
```

### API
```bash
# Preencher campos em falta
POST /duplios/dpp/123/gap-fill-lite
{
  "force": false
}
```

### Frontend
- Botão "Preencher automaticamente" aparece se campos ambientais em falta
- Mostra resultado e campos preenchidos
- Alerta sobre incerteza (±30%)

## ✅ Checklist de Requisitos

- ✅ Tabela de fatores internos (YAML com fallback)
- ✅ Serviço GapFillingLite com algoritmo completo
- ✅ API endpoint (POST /gap-fill-lite)
- ✅ Integração com Trust Index (recalcula automaticamente)
- ✅ Integração com R&D (WPX_GAPFILL_LITE)
- ✅ Frontend: Botão e estado
- ✅ UI simples com mensagem sobre incerteza
- ✅ Não bloqueia utilizador se falhar
- ✅ Não sobrescreve valores medidos/reportados

## 🔮 Extensões Futuras

### Ecoinvent Integration
- ⚠️ Integração com bases LCA externas (Ecoinvent, EF)
- ⚠️ GapFillingFull com NLP e context adjustment
- ⚠️ Monte Carlo para incerteza

### Machine Learning
- ⚠️ Predição de fatores baseada em histórico
- ⚠️ Ajuste automático de fatores por setor
- ⚠️ Detecção de anomalias em composição

### Multi-Tier
- ⚠️ Gap filling para supply chain multi-tier
- ⚠️ Agregação de fatores upstream
- ⚠️ Rastreabilidade de estimativas


