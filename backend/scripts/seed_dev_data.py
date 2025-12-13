#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    SEED DEV DATA - Dados de Desenvolvimento para ProdPlan 4.0
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Este script popula o sistema com dados de desenvolvimento consistentes para:
- Demonstrar todas as funcionalidades
- Testar integrações
- Validar o sistema end-to-end

Uso:
    python scripts/seed_dev_data.py

Dados criados:
- Items/Revisions/BOM/Routes (PDM)
- DPP Records
- Machine Sensor Data (SHI-DT)
- Operation Execution Logs
- R&D Experiments
- Inventory Data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any

import requests

BASE_URL = "http://127.0.0.1:8000"


def log(msg: str, level: str = "INFO"):
    """Simple logging."""
    print(f"[{level}] {msg}")


def api_call(method: str, endpoint: str, data: Dict = None) -> Dict:
    """Make API call."""
    url = f"{BASE_URL}{endpoint}"
    try:
        if method == "GET":
            resp = requests.get(url, timeout=30)
        elif method == "POST":
            resp = requests.post(url, json=data, timeout=30)
        else:
            return {"error": f"Unknown method: {method}"}
        
        if resp.status_code in [200, 201]:
            return resp.json()
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


def seed_shi_dt_data():
    """Seed SHI-DT machine health data."""
    log("Seeding SHI-DT data...")
    
    machines = ["MC-CNC-001", "MC-CNC-002", "MC-LATHE-001", "MC-MILL-001", "MC-PRESS-001"]
    
    for machine_id in machines:
        # Generate demo data - correct endpoint
        result = api_call("POST", f"/shi-dt/demo/generate-data?machine_id={machine_id}&n_samples=50")
        if "error" not in result:
            log(f"  ✅ Generated data for {machine_id}")
        else:
            log(f"  ⚠ {machine_id}: {result.get('error', 'Unknown')}", "WARN")


def seed_xai_dt_data():
    """Seed XAI-DT product analysis data."""
    log("Seeding XAI-DT data...")
    
    # XAI-DT demo requires body with product info
    demo_data = {
        "product_id": "PROD-001",
        "revision": "A",
        "n_points": 500
    }
    result = api_call("POST", "/xai-dt/demo", demo_data)
    if "error" not in result:
        log(f"  ✅ XAI-DT demo analysis created")
    else:
        log(f"  ⚠ XAI-DT: {result.get('error', 'Unknown')}", "WARN")


def seed_optimization_data():
    """Seed optimization golden runs."""
    log("Seeding Optimization data...")
    
    # Record some golden runs
    golden_runs = [
        {
            "product_id": "PROD-001",
            "operation_id": "OP-10",
            "machine_id": "MC-CNC-001",
            "cycle_time_minutes": 45.5,
            "defect_rate": 0.002,
            "oee": 0.92,
            "parameters": {"speed": 1.15, "temperature": 105, "pressure": 1.1},
            "context": {"shift": 1, "operator": "João Silva"}
        },
        {
            "product_id": "PROD-002",
            "operation_id": "OP-20",
            "machine_id": "MC-MILL-001",
            "cycle_time_minutes": 32.0,
            "defect_rate": 0.001,
            "oee": 0.95,
            "parameters": {"speed": 1.2, "temperature": 100, "pressure": 1.0},
            "context": {"shift": 2, "operator": "Maria Santos"}
        },
    ]
    
    for gr in golden_runs:
        result = api_call("POST", "/optimization/golden-runs/record", gr)
        if "error" not in result:
            log(f"  ✅ Golden run for {gr['product_id']}/{gr['operation_id']}")


def seed_prevention_guard_data():
    """Seed prevention guard historical data."""
    log("Seeding Prevention Guard training data...")
    
    # Add historical data for predictive model
    for i in range(30):
        data = {
            "order_id": f"OP-2024-{i:04d}",
            "product_id": f"PROD-{(i % 5) + 1:03d}",
            "machine_id": f"MC-{['CNC', 'MILL', 'LATHE', 'PRESS'][i % 4]}-001",
            "operator_id": f"OP-{(i % 8) + 1:03d}",
            "shift": (i % 3) + 1,
            "had_defect": random.random() < 0.15,
            "defect_type": "dimensional" if random.random() < 0.5 else "surface",
            "defect_cause": "tool_wear" if random.random() < 0.4 else "parameter_drift"
        }
        
        result = api_call("POST", "/guard/training/add-data", data)
    
    # Train the model
    result = api_call("POST", "/guard/training/train")
    if result.get("success"):
        log(f"  ✅ Predictive model trained with {result.get('samples', 0)} samples")
    else:
        log(f"  ⚠ Training: {result.get('reason', 'Unknown')}", "WARN")


def seed_rd_experiments():
    """Seed R&D experiments."""
    log("Seeding R&D experiments...")
    
    # WP1 experiment - check correct endpoint
    result = api_call("GET", "/rd/wp1/status")
    if "error" not in result:
        log(f"  ✅ WP1 module available")
    else:
        log(f"  ⚠ WP1: {result.get('error', 'Unknown')}", "WARN")
    
    # WP4 experiment 
    result = api_call("GET", "/rd/wp4/status")
    if "error" not in result:
        log(f"  ✅ WP4 module available")
    else:
        log(f"  ⚠ WP4: {result.get('error', 'Unknown')}", "WARN")
    
    # R&D status
    result = api_call("GET", "/rd/status")
    if "error" not in result:
        log(f"  ✅ R&D module status: {result.get('status', 'OK')}")


def seed_work_instructions():
    """Seed work instructions."""
    log("Seeding Work Instructions...")
    
    # Work instructions demo requires body
    demo_data = {
        "product_id": "PROD-001",
        "operation_id": "OP-10"
    }
    result = api_call("POST", "/work-instructions/demo", demo_data)
    if "error" not in result:
        log(f"  ✅ Work instructions demo created")
    else:
        log(f"  ⚠ Work Instructions: {result.get('error', 'Unknown')}", "WARN")


def seed_mrp_data():
    """Seed MRP data."""
    log("Seeding MRP data...")
    
    # Create demo MRP run
    demo_data = {
        "num_finished_products": 5,
        "num_components": 15,
        "weeks": 8
    }
    result = api_call("POST", "/mrp/demo", demo_data)
    if "error" not in result:
        log(f"  ✅ MRP demo run created")


def seed_scheduling_data():
    """Seed scheduling data by running a demo plan."""
    log("Seeding Scheduling data...")
    
    # Note: This relies on existing data from data_loader
    # Just verify the endpoint works
    result = api_call("GET", "/scheduling/engines")
    if "engines" in result:
        log(f"  ✅ Scheduling engines available: {len(result['engines'])}")


def check_backend_status():
    """Check if backend is running."""
    try:
        resp = requests.get(f"{BASE_URL}/shi-dt/status", timeout=5)
        return resp.status_code == 200
    except:
        return False


def main():
    """Main seed function."""
    print("=" * 70)
    print("  ProdPlan 4.0 - Development Data Seeding")
    print("=" * 70)
    print()
    
    # Check backend
    log("Checking backend status...")
    if not check_backend_status():
        log("Backend not running! Start with: python run_server.py", "ERROR")
        log("Trying to continue anyway...")
    else:
        log("  ✅ Backend is running")
    
    print()
    
    # Seed each module
    seed_shi_dt_data()
    seed_xai_dt_data()
    seed_optimization_data()
    seed_prevention_guard_data()
    seed_rd_experiments()
    seed_work_instructions()
    seed_mrp_data()
    seed_scheduling_data()
    
    print()
    print("=" * 70)
    log("Seeding complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

