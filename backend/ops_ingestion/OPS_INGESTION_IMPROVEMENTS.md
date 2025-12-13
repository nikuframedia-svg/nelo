# Operational Data Ingestion Engine - Melhorias Implementadas (Contrato 14)

## ✅ Requisitos Implementados

### Fase 1 – Modelos & Tabelas de Ingestão

#### 1.1. Estrutura de Pastas
- ✅ `backend/ops_ingestion/` criado
- ✅ `models.py`: Modelos SQLAlchemy
- ✅ `schemas.py`: Schemas Pydantic
- ✅ `excel_parser.py`: Parser de Excel
- ✅ `services.py`: Serviços de importação
- ✅ `api.py`: Endpoints REST
- ✅ `data_quality.py`: Checks e ML básico
- ✅ `data/column_aliases.yaml`: Mapeamento de colunas

#### 1.2. Modelos SQLAlchemy
- ✅ **ops_raw_orders**: Ordens de produção brutas
  - external_order_code, product_code, quantity, due_date
  - routing_json (JSON), line_or_center
  - source_file, imported_at, quality_flags
  
- ✅ **ops_raw_inventory_moves**: Movimentos internos
  - order_code, from_station, to_station
  - movement_type (enum), quantity_good, quantity_scrap
  - timestamp, source_file, imported_at, quality_flags
  
- ✅ **ops_raw_hr**: Recursos Humanos
  - technician_code, name, role
  - skills_json (JSON), shift_pattern, home_cell
  - source_file, imported_at, quality_flags
  
- ✅ **ops_raw_machines**: Máquinas e Linhas
  - machine_code, description, line
  - capacity_per_shift_hours, avg_setup_time_minutes
  - maintenance_windows_json (JSON)
  - source_file, imported_at, quality_flags

- ✅ **ops_data_quality_flags**: Flags de qualidade
  - table_name, record_id, flag_type, field_name
  - message, severity, detected_at, detected_by

### Fase 2 – Parsers de Excel e Mapeamento Flexível

#### 2.1. Schemas Pydantic
- ✅ **OrderRowSchema**: Validação de linha de ordem
- ✅ **InventoryMoveRowSchema**: Validação de linha de movimento
- ✅ **HRRowSchema**: Validação de linha de RH
- ✅ **MachineRowSchema**: Validação de linha de máquina

#### 2.2. Mapeamento de Colunas
- ✅ **column_aliases.yaml**: Base de dados de aliases
  - Aceita variações: "Produto", "Código Produto", "product_code"
  - Mapeamento por heurística (match exato e parcial)
  - Fallback para aliases padrão se YAML não disponível

#### 2.3. Parsers Excel
- ✅ **parse_excel_orders()**: Lê Excel de ordens
- ✅ **parse_excel_inventory_moves()**: Lê Excel de movimentos
- ✅ **parse_excel_hr()**: Lê Excel de RH
- ✅ **parse_excel_machines()**: Lê Excel de máquinas

### Fase 3 – Ligação aos Módulos Existentes

#### 3.1. Feeding ProdPlan
- ✅ **build_planning_instance_from_raw()**: 
  - Cria SchedulingInstance a partir de ops_raw_orders e ops_raw_machines
  - Jobs = ordens, Operations = routing_json, Machines = máquinas
  - Flag "source = excel_import" para distinguir origem

#### 3.2. Feeding SmartInventory
- ⚠️ **WIPFlowService**: Preparado para implementação futura
  - Estrutura de dados suporta reconstrução de estado WIP
  - Movimentos ordenados por timestamp permitem tracking

#### 3.3. Feeding Colaboradores
- ⚠️ **Mapeamento para collaborators**: Preparado para implementação futura
  - Estrutura de dados suporta merge inteligente (skills, shifts)

#### 3.4. Feeding Digital Twin
- ⚠️ **Mapeamento para machines**: Preparado para implementação futura
  - Estrutura de dados suporta integração com SHI-DT

### Fase 4 – Data Quality & ML Básico

#### 4.1. Serviço de Qualidade
- ✅ **analyze_orders_quality()**:
  - Quantidades negativas
  - Datas de entrega no passado extremo
  - Tempos padrão absurdos (0s quando devia ser > 10s)
  
- ✅ **analyze_inventory_moves_quality()**:
  - Movimentos sem order_code
  - Timestamps fora de ordem
  
- ✅ **analyze_hr_quality()**:
  - Skills fora de 0-1
  - Padrões de turno incoerentes
  
- ✅ **analyze_machines_quality()**:
  - Capacidade 0 ou negativa
  - Setup time > capacidade de turno

#### 4.2. ML para Deteção de Anomalias
- ✅ **detect_anomalies_ml_orders()**:
  - Autoencoder simples (PyTorch)
  - Treina em features: quantity, nr_operações, tempos
  - Detecta anomalias se reconstrução > threshold
  - Requer N >= 100 registos para treinar

### Fase 5 – API & UI

#### 5.1. API
- ✅ **POST /ops-ingestion/orders/excel**: Importa ordens
- ✅ **POST /ops-ingestion/inventory-moves/excel**: Importa movimentos
- ✅ **POST /ops-ingestion/hr/excel**: Importa RH
- ✅ **POST /ops-ingestion/machines/excel**: Importa máquinas
- ✅ **GET /ops-ingestion/planning-instance**: Constrói SchedulingInstance

Cada endpoint:
- Recebe ficheiro (multipart)
- Chama serviço de import
- Executa data quality checks
- Retorna ImportResult com contagens e warnings/erros

#### 5.2. UI
- ⚠️ **Modal "Carregar Dados"**: Preparado para implementação futura
  - 4 cards: Ordens, Movimentos, RH, Máquinas
  - Upload de ficheiro por card
  - Mostra último ficheiro carregado + estatuto

### Fase 6 – Ligação com R&D

#### 6.1. Logging para R&D
- ✅ **WPX_DATA_INGESTION**: Tipo de experimento definido
- ✅ **Logging automático**: Cada import cria registo em rd_experiments
  - type, imported_count, failed_count, warnings_count, errors_count, source_file

## 📊 Estrutura de Dados

### ImportResult
```python
{
    "success": true,
    "imported_count": 150,
    "failed_count": 2,
    "warnings": ["Ordem OP123: Data de entrega muito antiga"],
    "errors": [],
    "record_ids": [1, 2, 3, ...],
    "source_file": "orders.xlsx"
}
```

### SchedulingInstance (from raw)
```python
{
    "jobs": [
        {
            "job_id": "OP123",
            "product_code": "PROD001",
            "quantity": 100.0,
            "due_date": "2024-12-31",
            "operations": [
                {
                    "operation_id": "OP123_OP1",
                    "machine": "M1",
                    "time_minutes": 30,
                    "setup_minutes": 10
                }
            ],
            "source": "excel_import"
        }
    ],
    "machines": [
        {
            "machine_id": "M1",
            "description": "Máquina 1",
            "capacity_per_shift_hours": 8.0,
            "source": "excel_import"
        }
    ],
    "horizon_days": 30,
    "source": "ops_raw_excels"
}
```

## 🔧 Implementação Técnica

### Backend
- **models.py**: Modelos SQLAlchemy (4 tabelas raw + 1 tabela de flags)
- **schemas.py**: Schemas Pydantic para validação
- **excel_parser.py**: Parser com mapeamento flexível (column_aliases.yaml)
- **services.py**: OpsIngestionService com 4 métodos de import
- **data_quality.py**: Checks de qualidade + ML básico (autoencoder)
- **api.py**: Endpoints REST (4 POST + 1 GET)

### Integração
- **ProdPlan**: `build_planning_instance_from_raw()` cria SchedulingInstance
- **R&D**: Logging automático para WPX_DATA_INGESTION
- **Database**: Usa mesmo Base/engine de duplios.models

## 🚀 Uso

### Backend
```python
from ops_ingestion.services import get_ops_ingestion_service
from ops_ingestion.api import build_planning_instance_from_raw

service = get_ops_ingestion_service()

# Importar ordens
result = service.import_orders_from_excel(file, db)
print(f"Importadas: {result.imported_count}, Erros: {len(result.errors)}")

# Construir SchedulingInstance
instance = build_planning_instance_from_raw(db, horizon_days=30)
```

### API
```bash
# Importar ordens
curl -X POST "http://localhost:8000/ops-ingestion/orders/excel" \
  -F "file=@orders.xlsx"

# Obter SchedulingInstance
curl "http://localhost:8000/ops-ingestion/planning-instance?horizon_days=30"
```

## ✅ Checklist de Requisitos

- ✅ Estrutura de pastas (ops_ingestion/)
- ✅ 4 tabelas raw (orders, inventory_moves, hr, machines)
- ✅ Tabela de flags de qualidade
- ✅ Schemas Pydantic para validação
- ✅ Parser Excel com mapeamento flexível (column_aliases.yaml)
- ✅ Serviços de importação (4 métodos)
- ✅ Data quality checks (4 funções)
- ✅ ML básico para anomalias (autoencoder opcional)
- ✅ API endpoints (4 POST + 1 GET)
- ✅ Integração com R&D (WPX_DATA_INGESTION)
- ✅ Integração com ProdPlan (build_planning_instance_from_raw)
- ⚠️ UI modal (preparado, implementação futura)
- ⚠️ Integração completa SmartInventory/Colaboradores/Digital Twin (preparado)

## 🔮 Extensões Futuras

### Integração Completa
- ⚠️ WIPFlowService: Reconstrução de estado WIP por ordem/estação
- ⚠️ Mapeamento para collaborators: Merge inteligente de skills/shifts
- ⚠️ Mapeamento para machines: Integração com SHI-DT

### UI
- ⚠️ Modal "Carregar Dados" com 4 cards
- ⚠️ Dashboard de qualidade de dados
- ⚠️ Visualização de anomalias detectadas

### ML Avançado
- ⚠️ Autoencoder mais sofisticado (mais features)
- ⚠️ Deteção de anomalias para outros tipos de dados
- ⚠️ Predição de qualidade de dados antes de importar


