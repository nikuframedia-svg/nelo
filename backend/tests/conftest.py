"""
Fixtures comuns para todos os testes do backend.
"""
import pytest
import tempfile
import os
from pathlib import Path
from typing import Generator
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

# Importar app FastAPI
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from api import app
    HAS_APP = True
except ImportError:
    HAS_APP = False
    app = None


@pytest.fixture(scope="session")
def test_db_path():
    """Cria uma base de dados temporária para testes."""
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    yield db_path
    os.close(db_fd)
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture(scope="function")
def test_client():
    """Cliente de teste FastAPI."""
    if not HAS_APP:
        pytest.skip("FastAPI app not available")
    return TestClient(app)


@pytest.fixture
def sample_orders():
    """Ordens de produção de exemplo."""
    return [
        {
            "order_id": "OP001",
            "product_id": "PROD-A",
            "quantity": 100,
            "due_date": (datetime.now() + timedelta(days=7)).isoformat(),
            "priority": 5,
            "status": "PENDING"
        },
        {
            "order_id": "OP002",
            "product_id": "PROD-B",
            "quantity": 50,
            "due_date": (datetime.now() + timedelta(days=5)).isoformat(),
            "priority": 8,
            "status": "PENDING"
        },
        {
            "order_id": "OP003",
            "product_id": "PROD-A",
            "quantity": 200,
            "due_date": (datetime.now() + timedelta(days=10)).isoformat(),
            "priority": 3,
            "status": "PENDING"
        }
    ]


@pytest.fixture
def sample_machines():
    """Máquinas de exemplo."""
    return [
        {
            "machine_id": "M1",
            "name": "CNC-001",
            "type": "CNC",
            "capacity_per_hour": 10,
            "status": "AVAILABLE"
        },
        {
            "machine_id": "M2",
            "name": "PRESS-001",
            "type": "PRESS",
            "capacity_per_hour": 15,
            "status": "AVAILABLE"
        },
        {
            "machine_id": "M3",
            "name": "LATHE-001",
            "type": "LATHE",
            "capacity_per_hour": 12,
            "status": "AVAILABLE"
        }
    ]


@pytest.fixture
def sample_skus():
    """SKUs de inventário de exemplo."""
    return [
        {
            "sku_id": "SKU-001",
            "name": "Raw Material A",
            "current_stock": 500,
            "min_stock": 100,
            "max_stock": 1000,
            "lead_time_days": 5,
            "moq": 50,
            "unit_cost": 10.0
        },
        {
            "sku_id": "SKU-002",
            "name": "Component B",
            "current_stock": 200,
            "min_stock": 50,
            "max_stock": 500,
            "lead_time_days": 3,
            "moq": 25,
            "unit_cost": 25.0
        }
    ]


@pytest.fixture
def sample_dpp():
    """DPP (Digital Product Passport) de exemplo."""
    return {
        "dpp_id": "DPP-TEST-001",
        "product_name": "Test Product",
        "product_code": "PROD-TEST",
        "manufacturer": "Test Manufacturer",
        "carbon_footprint_kg_co2eq": 15.5,
        "water_consumption_m3": 2.3,
        "recyclability": 0.75,
        "composition": [
            {"material": "steel", "percentage": 60},
            {"material": "plastic", "percentage": 40}
        ],
        "country_of_origin": "PT"
    }


@pytest.fixture
def sample_bom():
    """BOM (Bill of Materials) de exemplo."""
    return {
        "item_id": "PROD-A",
        "revision": "A",
        "components": [
            {"item_id": "SKU-001", "quantity": 2, "unit": "kg"},
            {"item_id": "SKU-002", "quantity": 1, "unit": "pcs"}
        ]
    }


@pytest.fixture
def sample_routing():
    """Roteiro de fabrico de exemplo."""
    return {
        "item_id": "PROD-A",
        "revision": "A",
        "operations": [
            {
                "operation_id": "OP1",
                "name": "Cutting",
                "machine_id": "M1",
                "setup_time_min": 15,
                "cycle_time_min": 5,
                "sequence": 1
            },
            {
                "operation_id": "OP2",
                "name": "Assembly",
                "machine_id": "M2",
                "setup_time_min": 20,
                "cycle_time_min": 10,
                "sequence": 2
            }
        ]
    }


@pytest.fixture
def sample_sensor_readings():
    """Leituras de sensores de exemplo."""
    base_time = datetime.now()
    return [
        {
            "machine_id": "M1",
            "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
            "sensor_type": "vibration",
            "value": 0.5 + (i % 10) * 0.1,
            "unit": "g"
        }
        for i in range(20)
    ]


@pytest.fixture
def sample_experiment_config():
    """Configuração de experimento R&D de exemplo."""
    return {
        "experiment_id": "EXP-TEST-001",
        "wp": "WP1",
        "description": "Test experiment",
        "parameters": {
            "alpha": 0.05,
            "beta": 0.1
        }
    }

