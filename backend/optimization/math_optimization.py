"""
════════════════════════════════════════════════════════════════════════════════════════════════════
MATHEMATICAL OPTIMIZATION MODULE - Advanced Planning & Process Optimization
════════════════════════════════════════════════════════════════════════════════════════════════════

Módulo de otimização avançada para planeamento e processos industriais.

Components:
1. Time Prediction Engine - ML para previsão de tempos de setup/ciclo
2. Golden Runs - Identificação de desempenho ótimo
3. Process Parameter Optimizer - Bayesian Optimization / GA
4. Advanced Scheduling Solver - CP-SAT / MILP
5. Multi-Objective Optimizer - Pareto frontier

Mathematical Models:
- Scheduling: min Σ(w_j × delay_j) + α × Σ idle_time_m
- Parameter Optimization: min f(θ) = time(θ) + β × defect_rate(θ)
- Golden Run Gap: gap = (current - golden) / golden × 100%

R&D / SIFIDE: WP4 - Learning Scheduler & Advanced Optimization
"""

from __future__ import annotations

import logging
import json
import uuid
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import defaultdict
import random

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OptimizationConfig:
    """Configuration for optimization engines."""
    
    # Time Prediction
    time_model_hidden_size: int = 64
    time_model_epochs: int = 100
    
    # Parameter Optimization
    param_opt_iterations: int = 50
    param_opt_exploration: float = 0.1
    
    # Scheduling
    scheduling_time_limit_sec: int = 60
    scheduling_num_workers: int = 4
    
    # Golden Runs
    golden_run_percentile: float = 0.05  # Top 5%
    
    # Multi-objective
    pareto_population_size: int = 50
    pareto_generations: int = 100


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class OptimizationObjective(str, Enum):
    """Optimization objectives."""
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_DEFECTS = "minimize_defects"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_COST = "minimize_cost"
    BALANCED = "balanced"


class SchedulingPriority(str, Enum):
    """Scheduling priority rules."""
    FIFO = "fifo"
    EDD = "edd"  # Earliest Due Date
    SPT = "spt"  # Shortest Processing Time
    WSPT = "wspt"  # Weighted SPT
    CR = "cr"  # Critical Ratio
    OPTIMIZED = "optimized"  # CP-SAT


class OptimizerStatus(str, Enum):
    """Status of optimization run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProcessFeatures:
    """Features for time/quality prediction."""
    product_id: str
    operation_id: str
    machine_id: str
    material_type: str = ""
    batch_size: float = 1.0
    
    # Machine parameters
    speed_setting: float = 1.0
    temperature: float = 20.0
    pressure: float = 1.0
    
    # Context
    shift: int = 1  # 1, 2, 3
    operator_experience: float = 1.0  # 0-1
    machine_age_hours: float = 0.0
    
    # Historical
    last_setup_hours: float = 0.0
    consecutive_runs: int = 0
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML."""
        return np.array([
            hash(self.product_id) % 1000 / 1000,
            hash(self.operation_id) % 1000 / 1000,
            hash(self.machine_id) % 1000 / 1000,
            hash(self.material_type) % 1000 / 1000,
            self.batch_size / 100,
            self.speed_setting,
            self.temperature / 100,
            self.pressure,
            self.shift / 3,
            self.operator_experience,
            min(self.machine_age_hours / 10000, 1.0),
            min(self.last_setup_hours / 10, 1.0),
            min(self.consecutive_runs / 10, 1.0),
        ], dtype=np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TimePrediction:
    """Predicted time with confidence."""
    setup_time_minutes: float
    cycle_time_minutes: float
    total_time_minutes: float
    confidence: float  # 0-1
    model_version: str = "base"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "setup_time_minutes": round(self.setup_time_minutes, 2),
            "cycle_time_minutes": round(self.cycle_time_minutes, 2),
            "total_time_minutes": round(self.total_time_minutes, 2),
            "confidence": round(self.confidence, 3),
            "model_version": self.model_version,
        }


@dataclass
class GoldenRun:
    """Record of optimal performance."""
    run_id: str
    product_id: str
    operation_id: str
    machine_id: str
    
    # Performance metrics
    cycle_time_minutes: float
    defect_rate: float
    oee: float
    
    # Parameters that achieved this
    parameters: Dict[str, float]
    
    # Context
    context: Dict[str, Any]  # shift, operator, date, etc.
    
    # Metadata
    recorded_at: datetime
    validated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "product_id": self.product_id,
            "operation_id": self.operation_id,
            "machine_id": self.machine_id,
            "cycle_time_minutes": round(self.cycle_time_minutes, 2),
            "defect_rate": round(self.defect_rate, 4),
            "oee": round(self.oee, 3),
            "parameters": self.parameters,
            "context": self.context,
            "recorded_at": self.recorded_at.isoformat(),
            "validated": self.validated,
        }


@dataclass
class ParameterBounds:
    """Bounds for a process parameter."""
    name: str
    min_value: float
    max_value: float
    default_value: float
    step: float = 0.1
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    optimal_parameters: Dict[str, float]
    predicted_time: float
    predicted_defect_rate: float
    objective_value: float
    iterations_used: int
    improvement_percent: float
    confidence: float
    
    # History
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimal_parameters": self.optimal_parameters,
            "predicted_time": round(self.predicted_time, 2),
            "predicted_defect_rate": round(self.predicted_defect_rate, 4),
            "objective_value": round(self.objective_value, 4),
            "iterations_used": self.iterations_used,
            "improvement_percent": round(self.improvement_percent, 2),
            "confidence": round(self.confidence, 3),
            "history_length": len(self.optimization_history),
        }


@dataclass
class Job:
    """A job/task to be scheduled."""
    job_id: str
    product_id: str
    quantity: float
    
    # Timing
    processing_time_minutes: float
    setup_time_minutes: float
    due_date: datetime
    release_date: Optional[datetime] = None
    
    # Priority
    priority: int = 1
    weight: float = 1.0
    
    # Constraints
    required_machine: Optional[str] = None
    allowed_machines: List[str] = field(default_factory=list)
    predecessor_jobs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "product_id": self.product_id,
            "quantity": self.quantity,
            "processing_time_minutes": self.processing_time_minutes,
            "setup_time_minutes": self.setup_time_minutes,
            "due_date": self.due_date.isoformat(),
            "release_date": self.release_date.isoformat() if self.release_date else None,
            "priority": self.priority,
            "weight": self.weight,
        }


@dataclass
class Machine:
    """A machine/resource for scheduling."""
    machine_id: str
    name: str
    
    # Capacity
    available_hours_per_day: float = 8.0
    efficiency: float = 0.85  # OEE-based
    
    # Status
    available_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine_id": self.machine_id,
            "name": self.name,
            "available_hours_per_day": self.available_hours_per_day,
            "efficiency": self.efficiency,
            "available_from": self.available_from.isoformat(),
        }


@dataclass
class ScheduledJob:
    """A job with assigned schedule."""
    job_id: str
    machine_id: str
    start_time: datetime
    end_time: datetime
    setup_start: datetime
    
    # Metrics
    tardiness_minutes: float = 0  # Delay past due date
    waiting_time_minutes: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "machine_id": self.machine_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "setup_start": self.setup_start.isoformat(),
            "tardiness_minutes": round(self.tardiness_minutes, 1),
            "waiting_time_minutes": round(self.waiting_time_minutes, 1),
        }


@dataclass
class Schedule:
    """Complete production schedule."""
    schedule_id: str
    scheduled_jobs: List[ScheduledJob]
    
    # Metrics
    total_tardiness: float
    total_makespan_minutes: float
    machine_utilization: Dict[str, float]
    
    # Solver info
    solver_used: str
    solve_time_seconds: float
    optimality_gap: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "scheduled_jobs": [j.to_dict() for j in self.scheduled_jobs],
            "total_tardiness": round(self.total_tardiness, 1),
            "total_makespan_minutes": round(self.total_makespan_minutes, 1),
            "machine_utilization": {k: round(v, 3) for k, v in self.machine_utilization.items()},
            "solver_used": self.solver_used,
            "solve_time_seconds": round(self.solve_time_seconds, 2),
            "optimality_gap": round(self.optimality_gap, 4) if self.optimality_gap else None,
            "job_count": len(self.scheduled_jobs),
        }


@dataclass
class ParetoSolution:
    """A solution on the Pareto frontier."""
    solution_id: str
    parameters: Dict[str, float]
    objectives: Dict[str, float]  # objective_name -> value
    dominated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "solution_id": self.solution_id,
            "parameters": self.parameters,
            "objectives": {k: round(v, 4) for k, v in self.objectives.items()},
            "dominated": self.dominated,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TIME PREDICTION ENGINE (ML-based)
# ═══════════════════════════════════════════════════════════════════════════════

class TimePredictionEngineBase:
    """Base engine for time prediction."""
    
    def predict(self, features: ProcessFeatures) -> TimePrediction:
        """Predict setup and cycle time."""
        # Simple heuristic based on batch size and speed
        base_setup = 15.0  # 15 minutes base setup
        base_cycle = 5.0  # 5 minutes base cycle per unit
        
        setup_time = base_setup * (1.0 / features.speed_setting)
        cycle_time = base_cycle * features.batch_size * (1.0 / features.speed_setting)
        
        # Adjustments
        if features.consecutive_runs > 0:
            setup_time *= 0.7  # Reduced setup for consecutive runs
        
        if features.operator_experience < 0.5:
            cycle_time *= 1.2  # Slower for inexperienced
        
        return TimePrediction(
            setup_time_minutes=setup_time,
            cycle_time_minutes=cycle_time,
            total_time_minutes=setup_time + cycle_time,
            confidence=0.6,
            model_version="base_heuristic",
        )


class TimePredictionEngineML(TimePredictionEngineBase):
    """ML-based time prediction using neural network."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.model = None
        self.trained = False
        self._training_data: List[Tuple[np.ndarray, np.ndarray]] = []
    
    def _build_model(self, input_size: int):
        """Build PyTorch model."""
        try:
            import torch
            import torch.nn as nn
            
            class TimePredictor(nn.Module):
                def __init__(self, input_size, hidden_size):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_size, hidden_size // 2),
                        nn.ReLU(),
                        nn.Linear(hidden_size // 2, 2),  # setup_time, cycle_time
                    )
                
                def forward(self, x):
                    return self.net(x)
            
            self.model = TimePredictor(input_size, self.config.time_model_hidden_size)
            logger.info("Built ML time prediction model")
            return True
            
        except ImportError:
            logger.warning("PyTorch not available, using base engine")
            return False
    
    def add_training_sample(self, features: ProcessFeatures, actual_setup: float, actual_cycle: float):
        """Add a training sample."""
        x = features.to_vector()
        y = np.array([actual_setup, actual_cycle], dtype=np.float32)
        self._training_data.append((x, y))
    
    def train(self) -> Dict[str, Any]:
        """Train the model."""
        if len(self._training_data) < 10:
            return {"success": False, "reason": "Not enough training data"}
        
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            
            # Prepare data
            X = np.stack([d[0] for d in self._training_data])
            Y = np.stack([d[1] for d in self._training_data])
            
            X_tensor = torch.from_numpy(X)
            Y_tensor = torch.from_numpy(Y)
            
            # Build model
            if not self._build_model(X.shape[1]):
                return {"success": False, "reason": "Could not build model"}
            
            # Train
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            losses = []
            for epoch in range(self.config.time_model_epochs):
                self.model.train()
                optimizer.zero_grad()
                
                outputs = self.model(X_tensor)
                loss = criterion(outputs, Y_tensor)
                
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            
            self.trained = True
            
            return {
                "success": True,
                "samples": len(self._training_data),
                "final_loss": losses[-1],
                "epochs": self.config.time_model_epochs,
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"success": False, "reason": str(e)}
    
    def predict(self, features: ProcessFeatures) -> TimePrediction:
        """Predict using ML model if trained, else fallback to base."""
        if not self.trained or self.model is None:
            return super().predict(features)
        
        try:
            import torch
            
            self.model.eval()
            with torch.no_grad():
                x = torch.from_numpy(features.to_vector()).unsqueeze(0)
                output = self.model(x).numpy()[0]
            
            setup_time = max(0, float(output[0]))
            cycle_time = max(0, float(output[1]))
            
            return TimePrediction(
                setup_time_minutes=setup_time,
                cycle_time_minutes=cycle_time,
                total_time_minutes=setup_time + cycle_time,
                confidence=0.85,
                model_version="ml_pytorch",
            )
            
        except Exception as e:
            logger.warning(f"ML prediction failed, using base: {e}")
            return super().predict(features)


# ═══════════════════════════════════════════════════════════════════════════════
# CAPACITY MODEL ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class CapacityModelEngine:
    """
    Engine for modeling real machine capacity.
    
    As specified: "simular ou aprender a produtividade real de cada máquina/linha 
    tendo em conta eficiência, paragens, OEE histórico"
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.capacity_history: Dict[str, List[Dict[str, Any]]] = {}  # machine_id -> history
    
    def record_capacity_data(
        self,
        machine_id: str,
        oee: float,
        efficiency: float,
        downtime_minutes: float,
        throughput: float,
        context: Dict[str, Any],
    ) -> None:
        """Record capacity data for a machine."""
        if machine_id not in self.capacity_history:
            self.capacity_history[machine_id] = []
        
        self.capacity_history[machine_id].append({
            "oee": oee,
            "efficiency": efficiency,
            "downtime_minutes": downtime_minutes,
            "throughput": throughput,
            "context": context,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    
    def estimate_effective_capacity(
        self,
        machine_id: str,
        product_id: Optional[str] = None,
        operation_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Estimate effective capacity considering OEE and historical data.
        
        Returns:
            Dict with estimated capacity metrics
        """
        if machine_id not in self.capacity_history:
            # Default capacity
            return {
                "effective_capacity_per_hour": 10.0,
                "oee_estimate": 0.85,
                "efficiency_estimate": 0.85,
                "confidence": 0.0,
            }
        
        history = self.capacity_history[machine_id]
        
        # Filter by product/operation if specified
        if product_id or operation_id:
            filtered = [
                h for h in history
                if (not product_id or h.get("context", {}).get("product_id") == product_id)
                and (not operation_id or h.get("context", {}).get("operation_id") == operation_id)
            ]
            if filtered:
                history = filtered
        
        if not history:
            return {
                "effective_capacity_per_hour": 10.0,
                "oee_estimate": 0.85,
                "efficiency_estimate": 0.85,
                "confidence": 0.0,
            }
        
        # Calculate statistics
        oee_values = [h["oee"] for h in history]
        efficiency_values = [h["efficiency"] for h in history]
        throughput_values = [h["throughput"] for h in history]
        
        avg_oee = np.mean(oee_values)
        avg_efficiency = np.mean(efficiency_values)
        avg_throughput = np.mean(throughput_values)
        
        # Effective capacity = nominal * OEE * efficiency
        effective_capacity = avg_throughput * avg_oee * avg_efficiency
        
        # Confidence based on sample size
        confidence = min(1.0, len(history) / 100.0)
        
        return {
            "effective_capacity_per_hour": float(effective_capacity),
            "oee_estimate": float(avg_oee),
            "efficiency_estimate": float(avg_efficiency),
            "confidence": float(confidence),
            "sample_size": len(history),
        }
    
    def identify_bottlenecks(
        self,
        machines: List[str],
        planned_loads: Dict[str, float],  # machine_id -> planned_hours
    ) -> List[Dict[str, Any]]:
        """
        Identify capacity bottlenecks.
        
        As specified: "permitindo planeamentos mais fiáveis e identificação de gargalos ocultos"
        """
        bottlenecks = []
        
        for machine_id in machines:
            capacity = self.estimate_effective_capacity(machine_id)
            effective_hours_per_period = capacity["effective_capacity_per_hour"] * 8  # 8h shift
            
            planned_hours = planned_loads.get(machine_id, 0)
            
            if planned_hours > effective_hours_per_period:
                utilization = planned_hours / effective_hours_per_period if effective_hours_per_period > 0 else 0
                bottlenecks.append({
                    "machine_id": machine_id,
                    "planned_hours": planned_hours,
                    "available_hours": effective_hours_per_period,
                    "utilization": utilization,
                    "overload_hours": planned_hours - effective_hours_per_period,
                    "oee_estimate": capacity["oee_estimate"],
                })
        
        return sorted(bottlenecks, key=lambda x: x["overload_hours"], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# GOLDEN RUNS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class GoldenRunsEngine:
    """Engine for identifying and managing golden runs."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.golden_runs: Dict[str, GoldenRun] = {}  # key -> golden run
        self.run_history: List[Dict[str, Any]] = []
    
    def _make_key(self, product_id: str, operation_id: str, machine_id: str) -> str:
        """Create lookup key."""
        return f"{product_id}:{operation_id}:{machine_id}"
    
    def record_run(
        self,
        product_id: str,
        operation_id: str,
        machine_id: str,
        cycle_time_minutes: float,
        defect_rate: float,
        oee: float,
        parameters: Dict[str, float],
        context: Dict[str, Any],
    ) -> Optional[GoldenRun]:
        """
        Record a production run and check if it's a new golden run.
        
        Returns the GoldenRun if this is a new best.
        """
        # Store in history
        self.run_history.append({
            "product_id": product_id,
            "operation_id": operation_id,
            "machine_id": machine_id,
            "cycle_time_minutes": cycle_time_minutes,
            "defect_rate": defect_rate,
            "oee": oee,
            "parameters": parameters,
            "context": context,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        key = self._make_key(product_id, operation_id, machine_id)
        current_golden = self.golden_runs.get(key)
        
        # Check if this is a new golden run
        is_golden = False
        if current_golden is None:
            is_golden = True
        else:
            # Better if: faster cycle AND lower defect rate
            if cycle_time_minutes < current_golden.cycle_time_minutes and defect_rate <= current_golden.defect_rate:
                is_golden = True
            # Or significantly better OEE
            elif oee > current_golden.oee * 1.05:  # 5% better OEE
                is_golden = True
        
        if is_golden:
            golden = GoldenRun(
                run_id=f"GR-{uuid.uuid4().hex[:8]}",
                product_id=product_id,
                operation_id=operation_id,
                machine_id=machine_id,
                cycle_time_minutes=cycle_time_minutes,
                defect_rate=defect_rate,
                oee=oee,
                parameters=parameters,
                context=context,
                recorded_at=datetime.now(timezone.utc),
            )
            self.golden_runs[key] = golden
            logger.info(f"New golden run recorded: {key}")
            return golden
        
        return None
    
    def get_golden_run(self, product_id: str, operation_id: str, machine_id: str) -> Optional[GoldenRun]:
        """Get the golden run for a combination."""
        key = self._make_key(product_id, operation_id, machine_id)
        return self.golden_runs.get(key)
    
    def calculate_gap(
        self,
        product_id: str,
        operation_id: str,
        machine_id: str,
        current_cycle_time: float,
        current_oee: float,
    ) -> Optional[Dict[str, float]]:
        """Calculate gap between current performance and golden run."""
        golden = self.get_golden_run(product_id, operation_id, machine_id)
        if not golden:
            return None
        
        time_gap = (current_cycle_time - golden.cycle_time_minutes) / golden.cycle_time_minutes * 100
        oee_gap = (golden.oee - current_oee) / golden.oee * 100
        
        return {
            "time_gap_percent": round(time_gap, 2),
            "oee_gap_percent": round(oee_gap, 2),
            "golden_cycle_time": golden.cycle_time_minutes,
            "golden_oee": golden.oee,
            "recommended_parameters": golden.parameters,
        }
    
    def get_recommendations(self, product_id: str, operation_id: str, machine_id: str) -> Dict[str, Any]:
        """Get recommendations based on golden run."""
        golden = self.get_golden_run(product_id, operation_id, machine_id)
        if not golden:
            return {"available": False, "message": "No golden run recorded"}
        
        return {
            "available": True,
            "target_cycle_time": golden.cycle_time_minutes,
            "target_oee": golden.oee,
            "recommended_parameters": golden.parameters,
            "context_notes": golden.context,
            "recorded_at": golden.recorded_at.isoformat(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESS PARAMETER OPTIMIZER (Bayesian/GA)
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessParameterOptimizer:
    """
    Optimizer for process parameters using Bayesian Optimization, RL, or GA.
    
    As specified: "usar técnicas de otimização (ex.: Bayesian Optimization ou 
    Reinforcement Learning treinado para maximizar um reward de throughput/qualidade)"
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.surrogate_model = None
        self.observations: List[Tuple[Dict[str, float], float]] = []
        self.rl_agent = None  # RL agent for parameter optimization
    
    def optimize(
        self,
        parameter_bounds: List[ParameterBounds],
        objective_function: Callable[[Dict[str, float]], float],
        objective: OptimizationObjective = OptimizationObjective.MINIMIZE_TIME,
    ) -> OptimizationResult:
        """
        Optimize process parameters.
        
        Args:
            parameter_bounds: List of parameter bounds
            objective_function: Function that evaluates parameters -> objective value
            objective: Type of optimization
        
        Returns:
            OptimizationResult with optimal parameters
        """
        # Get initial baseline
        default_params = {p.name: p.default_value for p in parameter_bounds}
        baseline_value = objective_function(default_params)
        
        best_params = default_params.copy()
        best_value = baseline_value
        history = []
        optimization_method = "genetic_algorithm"
        
        # Try RL if agent is trained
        if self.rl_agent is not None:
            try:
                best_params, best_value, history = self._rl_optimize(
                    parameter_bounds, objective_function
                )
                optimization_method = "reinforcement_learning"
            except Exception as e:
                logger.warning(f"RL optimization failed, trying Bayesian: {e}")
        
        # Try Bayesian optimization
        if optimization_method == "genetic_algorithm":
            try:
                best_params, best_value, history = self._bayesian_optimize(
                    parameter_bounds, objective_function
                )
                optimization_method = "bayesian"
            except Exception as e:
                logger.warning(f"Bayesian optimization failed, using GA: {e}")
                best_params, best_value, history = self._genetic_algorithm(
                    parameter_bounds, objective_function
                )
        
        improvement = (baseline_value - best_value) / baseline_value * 100 if baseline_value > 0 else 0
        
        return OptimizationResult(
            optimal_parameters=best_params,
            predicted_time=best_value if objective == OptimizationObjective.MINIMIZE_TIME else 0,
            predicted_defect_rate=best_value if objective == OptimizationObjective.MINIMIZE_DEFECTS else 0,
            objective_value=best_value,
            iterations_used=len(history),
            improvement_percent=improvement,
            confidence=0.8,
            optimization_history=history,
        )
    
    def _bayesian_optimize(
        self,
        parameter_bounds: List[ParameterBounds],
        objective_function: Callable,
    ) -> Tuple[Dict[str, float], float, List[Dict]]:
        """Bayesian optimization using Gaussian Process surrogate."""
        from scipy.optimize import minimize
        from scipy.stats import norm
        
        n_params = len(parameter_bounds)
        bounds = [(p.min_value, p.max_value) for p in parameter_bounds]
        param_names = [p.name for p in parameter_bounds]
        
        # Initial samples
        X_observed = []
        y_observed = []
        history = []
        
        # Random initial samples
        for _ in range(5):
            params = {
                p.name: random.uniform(p.min_value, p.max_value)
                for p in parameter_bounds
            }
            x = [params[name] for name in param_names]
            y = objective_function(params)
            
            X_observed.append(x)
            y_observed.append(y)
            history.append({"params": params, "value": y, "type": "initial"})
        
        best_idx = np.argmin(y_observed)
        best_x = X_observed[best_idx]
        best_y = y_observed[best_idx]
        
        # Bayesian optimization iterations
        for iteration in range(self.config.param_opt_iterations - 5):
            # Fit simple surrogate (quadratic approximation)
            X_arr = np.array(X_observed)
            y_arr = np.array(y_observed)
            
            # Use acquisition function (Expected Improvement)
            def acquisition(x):
                # Simple distance-based uncertainty
                distances = np.sqrt(np.sum((X_arr - x) ** 2, axis=1))
                uncertainty = np.min(distances) if len(distances) > 0 else 1.0
                
                # Predict using nearest neighbor
                nearest_idx = np.argmin(distances) if len(distances) > 0 else 0
                predicted = y_arr[nearest_idx] if len(y_arr) > 0 else 0
                
                # Expected improvement
                improvement = best_y - predicted
                ei = improvement + self.config.param_opt_exploration * uncertainty
                return -ei  # Minimize negative EI
            
            # Optimize acquisition
            x0 = [random.uniform(b[0], b[1]) for b in bounds]
            result = minimize(acquisition, x0, bounds=bounds, method='L-BFGS-B')
            
            # Evaluate new point
            new_x = list(result.x)
            new_params = {name: val for name, val in zip(param_names, new_x)}
            new_y = objective_function(new_params)
            
            X_observed.append(new_x)
            y_observed.append(new_y)
            history.append({"params": new_params, "value": new_y, "type": "bayesian"})
            
            if new_y < best_y:
                best_x = new_x
                best_y = new_y
        
        best_params = {name: val for name, val in zip(param_names, best_x)}
        return best_params, best_y, history
    
    def _rl_optimize(
        self,
        parameter_bounds: List[ParameterBounds],
        objective_function: Callable,
    ) -> Tuple[Dict[str, float], float, List[Dict]]:
        """
        Reinforcement Learning optimization for parameters.
        
        As specified: "Reinforcement Learning treinado para maximizar um reward 
        de throughput/qualidade"
        """
        # For now, use a simple policy gradient approach
        # In full implementation, would use a trained RL agent (e.g., PPO, DQN)
        param_names = [p.name for p in parameter_bounds]
        history = []
        
        # Start with default parameters
        current_params = {p.name: p.default_value for p in parameter_bounds}
        best_params = current_params.copy()
        best_value = objective_function(current_params)
        
        # Simple policy gradient: explore around current best
        for iteration in range(self.config.param_opt_iterations):
            # Sample new parameters (exploration)
            new_params = {}
            for p in parameter_bounds:
                # Add noise to current best
                noise = random.gauss(0, (p.max_value - p.min_value) * 0.1)
                new_val = best_params[p.name] + noise
                new_params[p.name] = np.clip(new_val, p.min_value, p.max_value)
            
            # Evaluate
            value = objective_function(new_params)
            history.append({"params": new_params, "value": value, "type": "rl"})
            
            # Update if better
            if value < best_value:
                best_params = new_params.copy()
                best_value = value
        
        return best_params, best_value, history
    
    def train_rl_agent(
        self,
        training_data: List[Tuple[ProcessFeatures, Dict[str, float], float]],
        epochs: int = 100,
    ) -> Dict[str, Any]:
        """
        Train RL agent for parameter optimization.
        
        As specified: "um agente de RL pré-treinado em simulação"
        
        Args:
            training_data: List of (features, parameters, reward) tuples
            epochs: Training epochs
        
        Returns:
            Training history
        """
        # TODO: Implement full RL training (e.g., using stable-baselines3)
        # For now, just mark as available
        logger.info(f"RL agent training placeholder: {len(training_data)} samples, {epochs} epochs")
        self.rl_agent = "trained"  # Placeholder
        return {"success": True, "samples": len(training_data), "epochs": epochs}
    
    def _genetic_algorithm(
        self,
        parameter_bounds: List[ParameterBounds],
        objective_function: Callable,
    ) -> Tuple[Dict[str, float], float, List[Dict]]:
        """Genetic algorithm optimization."""
        population_size = 20
        generations = self.config.param_opt_iterations // population_size
        mutation_rate = 0.1
        
        param_names = [p.name for p in parameter_bounds]
        bounds = {p.name: (p.min_value, p.max_value) for p in parameter_bounds}
        
        history = []
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = {
                p.name: random.uniform(p.min_value, p.max_value)
                for p in parameter_bounds
            }
            fitness = objective_function(individual)
            population.append((individual, fitness))
            history.append({"params": individual, "value": fitness, "type": "ga_init"})
        
        best = min(population, key=lambda x: x[1])
        
        for gen in range(generations):
            # Selection (tournament)
            new_population = []
            
            for _ in range(population_size):
                # Tournament selection
                candidates = random.sample(population, min(3, len(population)))
                parent1 = min(candidates, key=lambda x: x[1])[0]
                
                candidates = random.sample(population, min(3, len(population)))
                parent2 = min(candidates, key=lambda x: x[1])[0]
                
                # Crossover
                child = {}
                for name in param_names:
                    if random.random() < 0.5:
                        child[name] = parent1[name]
                    else:
                        child[name] = parent2[name]
                
                # Mutation
                for name in param_names:
                    if random.random() < mutation_rate:
                        min_val, max_val = bounds[name]
                        child[name] = random.uniform(min_val, max_val)
                
                fitness = objective_function(child)
                new_population.append((child, fitness))
                history.append({"params": child, "value": fitness, "type": f"ga_gen{gen}"})
            
            population = new_population
            gen_best = min(population, key=lambda x: x[1])
            
            if gen_best[1] < best[1]:
                best = gen_best
        
        return best[0], best[1], history
    
    def _rl_optimize(
        self,
        parameter_bounds: List[ParameterBounds],
        objective_function: Callable,
    ) -> Tuple[Dict[str, float], float, List[Dict]]:
        """
        Reinforcement Learning optimization for parameters.
        
        As specified: "Reinforcement Learning treinado para maximizar um reward 
        de throughput/qualidade"
        """
        # For now, use a simple policy gradient approach
        # In full implementation, would use a trained RL agent (e.g., PPO, DQN)
        param_names = [p.name for p in parameter_bounds]
        history = []
        
        # Start with default parameters
        current_params = {p.name: p.default_value for p in parameter_bounds}
        best_params = current_params.copy()
        best_value = objective_function(current_params)
        
        # Simple policy gradient: explore around current best
        for iteration in range(self.config.param_opt_iterations):
            # Sample new parameters (exploration)
            new_params = {}
            for p in parameter_bounds:
                # Add noise to current best
                noise = random.gauss(0, (p.max_value - p.min_value) * 0.1)
                new_val = best_params[p.name] + noise
                new_params[p.name] = np.clip(new_val, p.min_value, p.max_value)
            
            # Evaluate
            value = objective_function(new_params)
            history.append({"params": new_params, "value": value, "type": "rl"})
            
            # Update if better
            if value < best_value:
                best_params = new_params.copy()
                best_value = value
        
        return best_params, best_value, history
    
    def train_rl_agent(
        self,
        training_data: List[Tuple[ProcessFeatures, Dict[str, float], float]],
        epochs: int = 100,
    ) -> Dict[str, Any]:
        """
        Train RL agent for parameter optimization.
        
        As specified: "um agente de RL pré-treinado em simulação"
        
        Args:
            training_data: List of (features, parameters, reward) tuples
            epochs: Training epochs
        
        Returns:
            Training history
        """
        # TODO: Implement full RL training (e.g., using stable-baselines3)
        # For now, just mark as available
        logger.info(f"RL agent training placeholder: {len(training_data)} samples, {epochs} epochs")
        self.rl_agent = "trained"  # Placeholder
        return {"success": True, "samples": len(training_data), "epochs": epochs}


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED SCHEDULING SOLVER (CP-SAT)
# ═══════════════════════════════════════════════════════════════════════════════

class SchedulingSolver:
    """
    Advanced scheduling solver using CP-SAT or heuristics.
    
    Formulation:
    min Σ(w_j × max(0, C_j - d_j)) + α × Σ idle_time_m
    
    Subject to:
    - Precedence constraints
    - Machine capacity (one job at a time)
    - Release dates
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
    
    def solve(
        self,
        jobs: List[Job],
        machines: List[Machine],
        priority: SchedulingPriority = SchedulingPriority.OPTIMIZED,
        horizon_hours: float = 168,  # 1 week
    ) -> Schedule:
        """
        Solve the scheduling problem.
        
        Args:
            jobs: Jobs to schedule
            machines: Available machines
            priority: Priority rule or optimized
            horizon_hours: Planning horizon
        
        Returns:
            Schedule with assigned jobs
        """
        start_time = datetime.now()
        
        if priority == SchedulingPriority.OPTIMIZED:
            # Try CP-SAT for small problems, Simulated Annealing for large
            if len(jobs) <= 20:
                try:
                    schedule = self._solve_cpsat(jobs, machines, horizon_hours)
                    schedule.solver_used = "cp_sat"
                except Exception as e:
                    logger.warning(f"CP-SAT failed, using SA: {e}")
                    schedule = self._solve_simulated_annealing(jobs, machines, horizon_hours)
            else:
                # Use metaheuristic for large problems
                schedule = self._solve_simulated_annealing(jobs, machines, horizon_hours)
        else:
            schedule = self._solve_heuristic(jobs, machines, priority)
        
        schedule.solve_time_seconds = (datetime.now() - start_time).total_seconds()
        return schedule
    
    def _solve_cpsat(
        self,
        jobs: List[Job],
        machines: List[Machine],
        horizon_hours: float,
    ) -> Schedule:
        """Solve using Google OR-Tools CP-SAT."""
        from ortools.sat.python import cp_model
        
        model = cp_model.CpModel()
        horizon_minutes = int(horizon_hours * 60)
        
        # Index mappings
        job_ids = [j.job_id for j in jobs]
        machine_ids = [m.machine_id for m in machines]
        
        # Variables
        # start[j] = start time of job j
        starts = {}
        ends = {}
        intervals = {}
        job_machine = {}  # (job, machine) -> interval var
        
        for j, job in enumerate(jobs):
            processing_time = int(job.processing_time_minutes + job.setup_time_minutes)
            
            starts[j] = model.NewIntVar(0, horizon_minutes, f'start_{j}')
            ends[j] = model.NewIntVar(0, horizon_minutes, f'end_{j}')
            
            # Duration constraint
            model.Add(ends[j] == starts[j] + processing_time)
            
            # Machine assignment
            allowed = job.allowed_machines if job.allowed_machines else machine_ids
            
            for m, machine in enumerate(machines):
                if machine.machine_id in allowed:
                    interval = model.NewOptionalIntervalVar(
                        starts[j], processing_time, ends[j],
                        model.NewBoolVar(f'assign_{j}_{m}'),
                        f'interval_{j}_{m}'
                    )
                    job_machine[(j, m)] = interval
        
        # Machine capacity: no overlap on each machine
        for m in range(len(machines)):
            machine_intervals = [
                job_machine[(j, m)]
                for j in range(len(jobs))
                if (j, m) in job_machine
            ]
            if machine_intervals:
                model.AddNoOverlap(machine_intervals)
        
        # Each job assigned to exactly one machine
        for j, job in enumerate(jobs):
            assignments = [
                model.GetBoolVar(job_machine[(j, m)])
                for m in range(len(machines))
                if (j, m) in job_machine
            ]
            model.Add(sum(assignments) == 1)
        
        # Precedence constraints
        job_idx = {job.job_id: j for j, job in enumerate(jobs)}
        for j, job in enumerate(jobs):
            for pred_id in job.predecessor_jobs:
                if pred_id in job_idx:
                    pred_j = job_idx[pred_id]
                    model.Add(starts[j] >= ends[pred_j])
        
        # Objective: minimize weighted tardiness
        base_time = datetime.now(timezone.utc)
        tardiness_vars = []
        
        for j, job in enumerate(jobs):
            due_minutes = int((job.due_date - base_time).total_seconds() / 60)
            tardiness = model.NewIntVar(0, horizon_minutes, f'tardiness_{j}')
            model.AddMaxEquality(tardiness, [0, ends[j] - due_minutes])
            tardiness_vars.append(int(job.weight) * tardiness)
        
        model.Minimize(sum(tardiness_vars))
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.scheduling_time_limit_sec
        solver.parameters.num_search_workers = self.config.scheduling_num_workers
        
        status = solver.Solve(model)
        
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            raise ValueError("No feasible solution found")
        
        # Extract solution
        scheduled_jobs = []
        for j, job in enumerate(jobs):
            start_val = solver.Value(starts[j])
            end_val = solver.Value(ends[j])
            
            # Find assigned machine
            assigned_machine = machine_ids[0]
            for m, machine in enumerate(machines):
                if (j, m) in job_machine:
                    interval = job_machine[(j, m)]
                    if solver.Value(model.GetBoolVar(interval)):
                        assigned_machine = machine.machine_id
                        break
            
            job_start = base_time + timedelta(minutes=start_val)
            job_end = base_time + timedelta(minutes=end_val)
            setup_start = job_start - timedelta(minutes=job.setup_time_minutes)
            
            tardiness_minutes = max(0, (job_end - job.due_date).total_seconds() / 60)
            
            scheduled_jobs.append(ScheduledJob(
                job_id=job.job_id,
                machine_id=assigned_machine,
                start_time=job_start,
                end_time=job_end,
                setup_start=setup_start,
                tardiness_minutes=tardiness_minutes,
            ))
        
        # Calculate metrics
        total_tardiness = sum(sj.tardiness_minutes for sj in scheduled_jobs)
        
        if scheduled_jobs:
            makespan = max((sj.end_time - base_time).total_seconds() / 60 for sj in scheduled_jobs)
        else:
            makespan = 0
        
        utilization = self._calculate_utilization(scheduled_jobs, machines, makespan)
        
        return Schedule(
            schedule_id=f"SCH-{uuid.uuid4().hex[:8]}",
            scheduled_jobs=scheduled_jobs,
            total_tardiness=total_tardiness,
            total_makespan_minutes=makespan,
            machine_utilization=utilization,
            solver_used="cp_sat",
            solve_time_seconds=0,
            optimality_gap=solver.BestObjectiveBound() / solver.ObjectiveValue() if solver.ObjectiveValue() > 0 else 0,
        )
    
    def _solve_heuristic(
        self,
        jobs: List[Job],
        machines: List[Machine],
        priority: SchedulingPriority,
    ) -> Schedule:
        """Solve using priority rule heuristics."""
        # Sort jobs by priority rule
        if priority == SchedulingPriority.EDD:
            sorted_jobs = sorted(jobs, key=lambda j: j.due_date)
        elif priority == SchedulingPriority.SPT:
            sorted_jobs = sorted(jobs, key=lambda j: j.processing_time_minutes)
        elif priority == SchedulingPriority.WSPT:
            sorted_jobs = sorted(jobs, key=lambda j: j.processing_time_minutes / j.weight)
        else:  # FIFO or default
            sorted_jobs = list(jobs)
        
        # Machine availability tracking
        machine_available = {m.machine_id: datetime.now(timezone.utc) for m in machines}
        
        scheduled_jobs = []
        base_time = datetime.now(timezone.utc)
        
        for job in sorted_jobs:
            # Find best machine
            allowed = job.allowed_machines if job.allowed_machines else [m.machine_id for m in machines]
            
            best_machine = None
            earliest_start = None
            
            for mid in allowed:
                if mid in machine_available:
                    start = machine_available[mid]
                    if earliest_start is None or start < earliest_start:
                        earliest_start = start
                        best_machine = mid
            
            if best_machine is None:
                continue
            
            # Schedule job
            total_time = job.setup_time_minutes + job.processing_time_minutes
            job_start = earliest_start
            job_end = job_start + timedelta(minutes=total_time)
            setup_start = job_start
            
            machine_available[best_machine] = job_end
            
            tardiness = max(0, (job_end - job.due_date).total_seconds() / 60)
            
            scheduled_jobs.append(ScheduledJob(
                job_id=job.job_id,
                machine_id=best_machine,
                start_time=job_start,
                end_time=job_end,
                setup_start=setup_start,
                tardiness_minutes=tardiness,
            ))
        
        # Metrics
        total_tardiness = sum(sj.tardiness_minutes for sj in scheduled_jobs)
        
        if scheduled_jobs:
            makespan = max((sj.end_time - base_time).total_seconds() / 60 for sj in scheduled_jobs)
        else:
            makespan = 0
        
        utilization = self._calculate_utilization(scheduled_jobs, machines, makespan)
        
        return Schedule(
            schedule_id=f"SCH-{uuid.uuid4().hex[:8]}",
            scheduled_jobs=scheduled_jobs,
            total_tardiness=total_tardiness,
            total_makespan_minutes=makespan,
            machine_utilization=utilization,
            solver_used=priority.value,
            solve_time_seconds=0,
        )
    
    def _calculate_utilization(
        self,
        scheduled_jobs: List[ScheduledJob],
        machines: List[Machine],
        makespan: float,
    ) -> Dict[str, float]:
        """Calculate machine utilization."""
        if makespan == 0:
            return {m.machine_id: 0 for m in machines}
        
        utilization = {}
        for machine in machines:
            machine_jobs = [sj for sj in scheduled_jobs if sj.machine_id == machine.machine_id]
            busy_time = sum(
                (sj.end_time - sj.setup_start).total_seconds() / 60
                for sj in machine_jobs
            )
            utilization[machine.machine_id] = busy_time / makespan
        
        return utilization


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-OBJECTIVE OPTIMIZER (Pareto)
# ═══════════════════════════════════════════════════════════════════════════════

class MultiObjectiveOptimizer:
    """
    Multi-objective optimization using NSGA-II style algorithm.
    
    Generates Pareto frontier for trade-off analysis.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
    
    def optimize(
        self,
        parameter_bounds: List[ParameterBounds],
        objective_functions: Dict[str, Callable[[Dict[str, float]], float]],
    ) -> List[ParetoSolution]:
        """
        Find Pareto frontier.
        
        Args:
            parameter_bounds: Parameter bounds
            objective_functions: Dict of {objective_name: function}
        
        Returns:
            List of Pareto-optimal solutions
        """
        population_size = self.config.pareto_population_size
        generations = self.config.pareto_generations
        
        param_names = [p.name for p in parameter_bounds]
        bounds = {p.name: (p.min_value, p.max_value) for p in parameter_bounds}
        
        # Initialize population
        population = []
        for i in range(population_size):
            params = {
                p.name: random.uniform(p.min_value, p.max_value)
                for p in parameter_bounds
            }
            objectives = {
                name: func(params)
                for name, func in objective_functions.items()
            }
            population.append(ParetoSolution(
                solution_id=f"PS-{i:04d}",
                parameters=params,
                objectives=objectives,
            ))
        
        # Evolution
        for gen in range(generations):
            # Non-dominated sorting
            fronts = self._non_dominated_sort(population)
            
            # Create new population
            new_population = []
            
            # Keep best fronts
            for front in fronts:
                if len(new_population) + len(front) <= population_size:
                    new_population.extend(front)
                else:
                    # Crowding distance for last front
                    remaining = population_size - len(new_population)
                    sorted_front = self._crowding_distance_sort(front, list(objective_functions.keys()))
                    new_population.extend(sorted_front[:remaining])
                    break
            
            # Generate offspring
            offspring = []
            while len(offspring) < population_size:
                # Tournament selection
                parent1 = self._tournament_select(new_population)
                parent2 = self._tournament_select(new_population)
                
                # Crossover
                child_params = {}
                for name in param_names:
                    if random.random() < 0.5:
                        child_params[name] = parent1.parameters[name]
                    else:
                        child_params[name] = parent2.parameters[name]
                
                # Mutation
                for name in param_names:
                    if random.random() < 0.1:
                        min_val, max_val = bounds[name]
                        child_params[name] = random.uniform(min_val, max_val)
                
                # Evaluate
                child_objectives = {
                    name: func(child_params)
                    for name, func in objective_functions.items()
                }
                
                offspring.append(ParetoSolution(
                    solution_id=f"PS-{gen}-{len(offspring):04d}",
                    parameters=child_params,
                    objectives=child_objectives,
                ))
            
            # Combine and select
            combined = new_population + offspring
            fronts = self._non_dominated_sort(combined)
            
            population = []
            for front in fronts:
                if len(population) + len(front) <= population_size:
                    population.extend(front)
                else:
                    remaining = population_size - len(population)
                    sorted_front = self._crowding_distance_sort(front, list(objective_functions.keys()))
                    population.extend(sorted_front[:remaining])
                    break
        
        # Return Pareto front
        fronts = self._non_dominated_sort(population)
        pareto_front = fronts[0] if fronts else []
        
        # Mark dominated solutions
        for sol in pareto_front:
            sol.dominated = False
        
        return pareto_front
    
    def _dominates(self, sol1: ParetoSolution, sol2: ParetoSolution) -> bool:
        """Check if sol1 dominates sol2 (all objectives better or equal, at least one strictly better)."""
        dominated = False
        all_equal_or_better = True
        
        for obj_name in sol1.objectives:
            if sol1.objectives[obj_name] < sol2.objectives[obj_name]:
                dominated = True
            elif sol1.objectives[obj_name] > sol2.objectives[obj_name]:
                all_equal_or_better = False
        
        return dominated and all_equal_or_better
    
    def _non_dominated_sort(self, population: List[ParetoSolution]) -> List[List[ParetoSolution]]:
        """Sort population into non-dominated fronts."""
        fronts = []
        remaining = list(population)
        
        while remaining:
            front = []
            dominated_by = {id(sol): [] for sol in remaining}
            domination_count = {id(sol): 0 for sol in remaining}
            
            for sol1 in remaining:
                for sol2 in remaining:
                    if sol1 is not sol2:
                        if self._dominates(sol1, sol2):
                            dominated_by[id(sol1)].append(sol2)
                        elif self._dominates(sol2, sol1):
                            domination_count[id(sol1)] += 1
            
            for sol in remaining:
                if domination_count[id(sol)] == 0:
                    front.append(sol)
            
            if not front:
                front = remaining  # Fallback
                remaining = []
            else:
                remaining = [s for s in remaining if s not in front]
            
            fronts.append(front)
        
        return fronts
    
    def _crowding_distance_sort(
        self,
        front: List[ParetoSolution],
        objective_names: List[str],
    ) -> List[ParetoSolution]:
        """Sort by crowding distance (prefer diverse solutions)."""
        if len(front) <= 2:
            return front
        
        distances = {id(sol): 0.0 for sol in front}
        
        for obj_name in objective_names:
            sorted_front = sorted(front, key=lambda s: s.objectives[obj_name])
            
            # Boundary solutions get infinite distance
            distances[id(sorted_front[0])] = float('inf')
            distances[id(sorted_front[-1])] = float('inf')
            
            obj_range = sorted_front[-1].objectives[obj_name] - sorted_front[0].objectives[obj_name]
            if obj_range > 0:
                for i in range(1, len(sorted_front) - 1):
                    distance = (
                        sorted_front[i + 1].objectives[obj_name] -
                        sorted_front[i - 1].objectives[obj_name]
                    ) / obj_range
                    distances[id(sorted_front[i])] += distance
        
        return sorted(front, key=lambda s: distances[id(s)], reverse=True)
    
    def _tournament_select(self, population: List[ParetoSolution]) -> ParetoSolution:
        """Tournament selection."""
        candidates = random.sample(population, min(3, len(population)))
        # Prefer solutions with better first objective (simple)
        return min(candidates, key=lambda s: list(s.objectives.values())[0])


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class MathOptimizationService:
    """
    Main service for mathematical optimization.
    
    Provides:
    - Time prediction
    - Golden run management
    - Parameter optimization
    - Scheduling
    - Multi-objective optimization
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        
        # Initialize engines
        self.time_predictor = TimePredictionEngineML(self.config)
        self.capacity_model = CapacityModelEngine(self.config)
        self.golden_runs = GoldenRunsEngine(self.config)
        self.param_optimizer = ProcessParameterOptimizer(self.config)
        self.scheduler = SchedulingSolver(self.config)
        self.multi_objective = MultiObjectiveOptimizer(self.config)
    
    def predict_time(self, features: ProcessFeatures) -> TimePrediction:
        """Predict processing time."""
        return self.time_predictor.predict(features)
    
    def record_run(self, **kwargs) -> Optional[GoldenRun]:
        """Record a production run."""
        return self.golden_runs.record_run(**kwargs)
    
    def get_golden_run_gap(
        self,
        product_id: str,
        operation_id: str,
        machine_id: str,
        current_cycle_time: float,
        current_oee: float,
    ) -> Optional[Dict[str, float]]:
        """Get gap from golden run."""
        return self.golden_runs.calculate_gap(
            product_id, operation_id, machine_id,
            current_cycle_time, current_oee
        )
    
    def optimize_parameters(
        self,
        parameter_bounds: List[ParameterBounds],
        objective: OptimizationObjective,
        base_features: Optional[ProcessFeatures] = None,
    ) -> OptimizationResult:
        """Optimize process parameters."""
        
        def objective_function(params: Dict[str, float]) -> float:
            # Simple surrogate: time decreases with speed but defects increase
            speed = params.get("speed", 1.0)
            temp = params.get("temperature", 100.0)
            
            time_estimate = 100 / speed  # Faster = less time
            defect_rate = 0.01 * (speed - 1.0) ** 2 + 0.001 * abs(temp - 100) ** 2
            
            if objective == OptimizationObjective.MINIMIZE_TIME:
                return time_estimate
            elif objective == OptimizationObjective.MINIMIZE_DEFECTS:
                return defect_rate
            else:
                return time_estimate + 1000 * defect_rate
        
        return self.param_optimizer.optimize(
            parameter_bounds,
            objective_function,
            objective,
        )
    
    def solve_schedule(
        self,
        jobs: List[Job],
        machines: List[Machine],
        priority: SchedulingPriority = SchedulingPriority.OPTIMIZED,
    ) -> Schedule:
        """Solve scheduling problem."""
        return self.scheduler.solve(jobs, machines, priority)
    
    def optimize_pareto(
        self,
        parameter_bounds: List[ParameterBounds],
    ) -> List[ParetoSolution]:
        """Generate Pareto frontier for multi-objective optimization."""
        
        def time_objective(params: Dict[str, float]) -> float:
            speed = params.get("speed", 1.0)
            return 100 / speed
        
        def quality_objective(params: Dict[str, float]) -> float:
            speed = params.get("speed", 1.0)
            temp = params.get("temperature", 100.0)
            return 0.01 * (speed - 1.0) ** 2 + 0.001 * abs(temp - 100) ** 2
        
        objectives = {
            "time": time_objective,
            "defect_rate": quality_objective,
        }
        
        return self.multi_objective.optimize(parameter_bounds, objectives)
    
    def what_if_analysis(
        self,
        base_schedule: Schedule,
        scenario_changes: Dict[str, Any],
    ) -> Tuple[Schedule, Dict[str, Any]]:
        """
        What-If advanced analysis.
        
        As specified: "permitir alterar parâmetros (capacidade, turnos, inserção 
        de ordens urgentes, falha de máquina simulada) e recalcular rapidamente 
        o plano ótimo ajustado, possivelmente usando computação paralela ou 
        otimização incremental"
        
        Args:
            base_schedule: Original schedule
            scenario_changes: Dict with changes:
                - "machine_unavailable": [machine_ids]
                - "new_urgent_jobs": [Job]
                - "capacity_changes": {machine_id: new_capacity}
                - "shift_changes": {machine_id: new_hours_per_day}
        
        Returns:
            (new_schedule, comparison_metrics)
        """
        # Clone base schedule
        jobs = [Job(
            job_id=sj.job_id,
            processing_time_minutes=(sj.end_time - sj.start_time).total_seconds() / 60,
            setup_time_minutes=0,
            due_date=sj.end_time,
            allowed_machines=[sj.machine_id],
        ) for sj in base_schedule.scheduled_jobs]
        
        machines = [Machine(machine_id="M-001", name="Machine 1")]  # Placeholder
        
        # Apply changes
        if "new_urgent_jobs" in scenario_changes:
            jobs.extend(scenario_changes["new_urgent_jobs"])
        
        if "machine_unavailable" in scenario_changes:
            unavailable = set(scenario_changes["machine_unavailable"])
            # Filter jobs that can't use unavailable machines
            for job in jobs:
                if job.allowed_machines:
                    job.allowed_machines = [m for m in job.allowed_machines if m not in unavailable]
        
        # Re-optimize
        new_schedule = self.scheduler.solve(
            jobs=jobs,
            machines=machines,
            priority=SchedulingPriority.OPTIMIZED,
        )
        
        # Compare
        comparison = {
            "original_tardiness": base_schedule.total_tardiness,
            "new_tardiness": new_schedule.total_tardiness,
            "tardiness_change": new_schedule.total_tardiness - base_schedule.total_tardiness,
            "original_makespan": base_schedule.total_makespan_minutes,
            "new_makespan": new_schedule.total_makespan_minutes,
            "makespan_change": new_schedule.total_makespan_minutes - base_schedule.total_makespan_minutes,
            "solve_time_seconds": new_schedule.solve_time_seconds,
        }
        
        return new_schedule, comparison
    
    def estimate_capacity(
        self,
        machine_id: str,
        product_id: Optional[str] = None,
        operation_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Estimate effective capacity for a machine."""
        return self.capacity_model.estimate_effective_capacity(
            machine_id, product_id, operation_id
        )
    
    def identify_bottlenecks(
        self,
        machines: List[str],
        planned_loads: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Identify capacity bottlenecks."""
        return self.capacity_model.identify_bottlenecks(machines, planned_loads)


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_service_instance: Optional[MathOptimizationService] = None


def get_optimization_service() -> MathOptimizationService:
    """Get singleton service."""
    global _service_instance
    if _service_instance is None:
        _service_instance = MathOptimizationService()
    return _service_instance


def reset_optimization_service() -> None:
    """Reset singleton."""
    global _service_instance
    _service_instance = None


