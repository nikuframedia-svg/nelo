"""
════════════════════════════════════════════════════════════════════════════════════════════════════
ZDM API - Zero Disruption Manufacturing REST Endpoints
════════════════════════════════════════════════════════════════════════════════════════════════════

Endpoints for ZDM simulation:
- GET /zdm/status - Module status
- POST /zdm/scenarios/generate - Generate failure scenarios
- POST /zdm/simulate - Run resilience simulation
- GET /zdm/strategies - List recovery strategies
- POST /zdm/demo - Run demo simulation

R&D / SIFIDE: WP4 - Zero Defect & Zero Disruption Manufacturing
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .failure_scenario_generator import (
    FailureScenario,
    FailureType,
    FailureConfig,
    generate_failure_scenarios,
)
from .recovery_strategy_engine import (
    RecoveryStrategy,
    get_recovery_recommendations,
)
from .zdm_simulator import (
    ZDMSimulator,
    SimulationConfig,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/zdm", tags=["ZDM"])


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ScenarioGenerateRequest(BaseModel):
    """Request for generating failure scenarios."""
    machine_ids: List[str] = Field(default_factory=lambda: ["MC-001", "MC-002"])
    num_scenarios: int = Field(5, ge=1, le=50)
    failure_types: Optional[List[str]] = None
    severity_range: tuple = (0.3, 0.8)


class SimulateRequest(BaseModel):
    """Request for simulation."""
    scenario_id: Optional[str] = None
    machine_id: str = "MC-001"
    failure_type: str = "sudden"
    duration_hours: float = Field(4.0, ge=0.5, le=48)
    severity: float = Field(0.5, ge=0.1, le=1.0)


class DemoRequest(BaseModel):
    """Demo request."""
    num_scenarios: int = Field(3, ge=1, le=10)


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/status")
async def get_zdm_status():
    """Get ZDM module status."""
    return {
        "service": "ZDM - Zero Disruption Manufacturing",
        "version": "1.0.0",
        "status": "operational",
        "features": [
            "failure_scenario_generation",
            "resilience_simulation",
            "recovery_strategies",
            "impact_analysis"
        ],
        "failure_types": [ft.value for ft in FailureType],
        "recovery_strategies": [rs.value for rs in RecoveryStrategy],
    }


@router.post("/scenarios/generate")
async def generate_scenarios(request: ScenarioGenerateRequest):
    """Generate failure scenarios for simulation."""
    try:
        config = FailureConfig(
            machine_ids=request.machine_ids,
            failure_types=[FailureType(ft) for ft in request.failure_types] if request.failure_types else None,
            min_severity=request.severity_range[0],
            max_severity=request.severity_range[1],
        )
        
        scenarios = generate_failure_scenarios(
            config=config,
            num_scenarios=request.num_scenarios,
        )
        
        return {
            "success": True,
            "num_scenarios": len(scenarios),
            "scenarios": [
                {
                    "scenario_id": s.scenario_id,
                    "failure_type": s.failure_type.value,
                    "machine_id": s.machine_id,
                    "duration_hours": s.duration_hours,
                    "severity": s.severity,
                    "start_time": s.start_time.isoformat() if s.start_time else None,
                }
                for s in scenarios
            ],
        }
    except Exception as e:
        logger.error(f"Scenario generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate")
async def run_simulation(request: SimulateRequest):
    """Run resilience simulation for a failure scenario."""
    try:
        # Create scenario
        scenario = FailureScenario(
            scenario_id=request.scenario_id or f"SIM-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            failure_type=FailureType(request.failure_type),
            machine_id=request.machine_id,
            start_time=datetime.now(),
            duration_hours=request.duration_hours,
            severity=request.severity,
        )
        
        # Run quick resilience check
        result = quick_resilience_check(scenario)
        
        # Get recovery recommendations
        recommendations = get_recovery_recommendations(scenario)
        
        return {
            "success": True,
            "scenario": {
                "scenario_id": scenario.scenario_id,
                "failure_type": scenario.failure_type.value,
                "machine_id": scenario.machine_id,
                "severity": scenario.severity,
            },
            "impact": {
                "estimated_downtime_hours": result.get("downtime_hours", request.duration_hours),
                "orders_at_risk": result.get("orders_at_risk", 0),
                "throughput_loss_pct": result.get("throughput_loss_pct", request.severity * 20),
            },
            "resilience_score": result.get("resilience_score", 1.0 - request.severity * 0.5),
            "recovery_recommendations": [
                {
                    "strategy": rec.strategy.value,
                    "priority": rec.priority,
                    "expected_recovery_time": rec.expected_recovery_time,
                    "cost_estimate": rec.cost_estimate,
                }
                for rec in recommendations[:3]
            ] if recommendations else [],
        }
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies")
async def list_recovery_strategies():
    """List available recovery strategies."""
    strategies = [
        {
            "id": "local_replan",
            "name": "Local Replan",
            "description": "Reorder operations locally to minimize impact",
            "applicability": ["sudden", "gradual"],
            "typical_recovery_hours": 1.0,
        },
        {
            "id": "vip_priority",
            "name": "VIP Priority",
            "description": "Prioritize VIP orders to maintain critical deliveries",
            "applicability": ["sudden", "material"],
            "typical_recovery_hours": 0.5,
        },
        {
            "id": "add_shift",
            "name": "Add Shift",
            "description": "Add extra shift to recover lost capacity",
            "applicability": ["sudden", "gradual"],
            "typical_recovery_hours": 8.0,
        },
        {
            "id": "reroute",
            "name": "Reroute",
            "description": "Redirect work to alternative machines",
            "applicability": ["sudden", "quality"],
            "typical_recovery_hours": 2.0,
        },
        {
            "id": "partial_batch",
            "name": "Partial Batch",
            "description": "Split batches to process partial quantities",
            "applicability": ["material", "quality"],
            "typical_recovery_hours": 1.5,
        },
        {
            "id": "outsource",
            "name": "Outsource",
            "description": "External subcontracting for urgent orders",
            "applicability": ["sudden", "gradual"],
            "typical_recovery_hours": 24.0,
        },
    ]
    
    return {
        "count": len(strategies),
        "strategies": strategies,
    }


@router.post("/demo")
async def run_demo(request: DemoRequest):
    """Run demo ZDM simulation."""
    results = []
    
    demo_scenarios = [
        {"type": "sudden", "machine": "MC-CNC-001", "severity": 0.7, "hours": 4},
        {"type": "gradual", "machine": "MC-MILL-001", "severity": 0.4, "hours": 8},
        {"type": "quality", "machine": "MC-LATHE-001", "severity": 0.5, "hours": 2},
    ]
    
    for i, demo in enumerate(demo_scenarios[:request.num_scenarios]):
        # Calculate simple resilience score based on severity
        # Real implementation would use full simulation
        base_resilience = 1.0 - demo["severity"] * 0.6
        recovery_factor = 0.9 if demo["type"] == "gradual" else 0.7
        resilience_score = base_resilience * recovery_factor
        
        results.append({
            "scenario_id": f"DEMO-{i+1:03d}",
            "failure_type": demo["type"],
            "machine_id": demo["machine"],
            "severity": demo["severity"],
            "resilience_score": round(resilience_score, 3),
            "estimated_recovery_hours": round(demo["hours"] * (1 - resilience_score + 0.5), 1),
        })
    
    # Overall summary
    avg_resilience = sum(r["resilience_score"] for r in results) / len(results) if results else 0
    
    return {
        "demo": True,
        "num_scenarios": len(results),
        "scenarios": results,
        "summary": {
            "average_resilience_score": round(avg_resilience, 3),
            "system_status": "resilient" if avg_resilience > 0.7 else "at_risk" if avg_resilience > 0.4 else "vulnerable",
            "recommendation": "Consider preventive maintenance" if avg_resilience < 0.7 else "System is well prepared",
        },
    }

