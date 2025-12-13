# Testes Backend - ProdPlan 4.0

## Estrutura

```
backend/tests/
├── conftest.py                    # Fixtures comuns
├── test_prodplan_planning.py     # A1-A6: ProdPlan
├── test_smart_inventory.py       # B1-B5: SmartInventory
├── test_duplios.py               # C1-C6: Duplios
├── test_digital_twin.py          # D1-D4: Digital Twin
├── test_intelligence.py          # E1-E3: Inteligência
├── test_rd.py                    # F1-F4: R&D
├── test_prevention.py            # G1-G3: Prevenção
└── test_chat.py                  # H1-H3: Chat
```

## Executar Testes

### Todos os testes
```bash
cd backend
pytest tests/ -v
```

### Teste específico
```bash
pytest tests/test_prodplan_planning.py::TestA1_PrecedencesAndCapacity -v
```

### Com cobertura
```bash
pytest tests/ --cov=. --cov-report=html
# Abrir htmlcov/index.html no browser
```

### Testes por módulo
```bash
pytest tests/test_prodplan_planning.py -v      # ProdPlan
pytest tests/test_smart_inventory.py -v         # SmartInventory
pytest tests/test_duplios.py -v                # Duplios
pytest tests/test_digital_twin.py -v           # Digital Twin
pytest tests/test_intelligence.py -v           # Inteligência
pytest tests/test_rd.py -v                     # R&D
pytest tests/test_prevention.py -v             # Prevenção
pytest tests/test_chat.py -v                   # Chat
```

## Fixtures

Fixtures comuns disponíveis em `conftest.py`:

- `test_client`: Cliente FastAPI para testes
- `sample_orders`: Ordens de produção de exemplo
- `sample_machines`: Máquinas de exemplo
- `sample_skus`: SKUs de inventário de exemplo
- `sample_dpp`: DPP de exemplo
- `sample_bom`: BOM de exemplo
- `sample_routing`: Roteiro de fabrico de exemplo
- `sample_sensor_readings`: Leituras de sensores de exemplo
- `sample_experiment_config`: Configuração de experimento R&D

## Notas Importantes

1. **Dados Sintéticos**: Todos os testes usam dados sintéticos leves
2. **Sem Treino de Modelos**: Modelos ML não são treinados durante testes (usam mocks ou pesos pré-treinados)
3. **Independência**: Testes são independentes e podem rodar em paralelo
4. **Fallback**: Testes validam comportamento de fallback quando engines avançados falham

## Mapeamento de Cobertura

Ver `docs/test_coverage_map.md` para mapeamento completo de test cases (A1-H3) para testes implementados.

