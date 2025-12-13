"""
Experiment Logger for SIFIDE R&D Programme

Provides structured logging for all experiments, enabling:
- Reproducibility
- Statistical analysis
- SIFIDE audit compliance

Usage:
    from backend.research import log_experiment
    
    with log_experiment("E1.1", hypothesis="H1.1") as exp:
        exp.set_config(routing_mode="dynamic", scoring="setup_aware")
        exp.set_inputs(num_orders=50, num_machines=12)
        
        # Run experiment
        result = run_scheduler(...)
        
        exp.set_outputs(makespan_h=42.5, setup_hours=3.2, otd_pct=94.5)
        exp.set_baseline(makespan_h=48.1, setup_hours=5.8, otd_pct=91.2)
"""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Generator


# Default log directory
LOG_DIR = Path(__file__).resolve().parents[2] / "data" / "experiments"


@dataclass
class ExperimentLog:
    """Structured log entry for a single experiment run."""
    experiment_id: str
    hypothesis: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    config: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    baseline_outputs: Dict[str, Any] = field(default_factory=dict)
    delta_pct: Dict[str, float] = field(default_factory=dict)
    notes: str = ""
    status: str = "running"  # running, completed, failed
    duration_seconds: float = 0.0


class ExperimentLogger:
    """
    Context manager for logging experiments.
    
    Automatically computes deltas, saves to JSON, and handles errors.
    """
    
    def __init__(
        self,
        experiment_id: str,
        hypothesis: str,
        log_dir: Optional[Path] = None,
    ):
        self.log = ExperimentLog(experiment_id=experiment_id, hypothesis=hypothesis)
        self.log_dir = log_dir or LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._start_time: Optional[datetime] = None
    
    def set_config(self, **kwargs: Any) -> None:
        """Set experiment configuration parameters."""
        self.log.config.update(kwargs)
    
    def set_inputs(self, **kwargs: Any) -> None:
        """Set experiment input parameters."""
        self.log.inputs.update(kwargs)
    
    def set_outputs(self, **kwargs: Any) -> None:
        """Set experiment output metrics."""
        self.log.outputs.update(kwargs)
    
    def set_baseline(self, **kwargs: Any) -> None:
        """Set baseline metrics for comparison."""
        self.log.baseline_outputs.update(kwargs)
        self._compute_deltas()
    
    def add_note(self, note: str) -> None:
        """Add a note to the experiment log."""
        self.log.notes += f"\n{note}" if self.log.notes else note
    
    def _compute_deltas(self) -> None:
        """Compute percentage deltas between outputs and baseline."""
        for key in self.log.outputs:
            if key in self.log.baseline_outputs:
                baseline_val = self.log.baseline_outputs[key]
                output_val = self.log.outputs[key]
                if baseline_val != 0:
                    delta = ((output_val - baseline_val) / baseline_val) * 100
                    self.log.delta_pct[key] = round(delta, 2)
    
    def _save(self) -> Path:
        """Save log to JSON file."""
        filename = f"{self.log.experiment_id}_{self.log.run_id}_{self.log.timestamp[:10]}.json"
        filepath = self.log_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(self.log), f, indent=2, ensure_ascii=False)
        return filepath
    
    def __enter__(self) -> "ExperimentLogger":
        self._start_time = datetime.utcnow()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start_time:
            self.log.duration_seconds = (datetime.utcnow() - self._start_time).total_seconds()
        
        if exc_type is not None:
            self.log.status = "failed"
            self.add_note(f"Error: {exc_type.__name__}: {exc_val}")
        else:
            self.log.status = "completed"
        
        filepath = self._save()
        print(f"[R&D] Experiment logged: {filepath}")


@contextmanager
def log_experiment(
    experiment_id: str,
    hypothesis: str,
    log_dir: Optional[Path] = None,
) -> Generator[ExperimentLogger, None, None]:
    """
    Context manager for logging experiments.
    
    Example:
        with log_experiment("E1.1", "H1.1") as exp:
            exp.set_config(mode="dynamic")
            exp.set_outputs(makespan=42.5)
            exp.set_baseline(makespan=48.1)
    """
    logger = ExperimentLogger(experiment_id, hypothesis, log_dir)
    with logger:
        yield logger


def load_experiment_logs(
    experiment_id: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> list[Dict[str, Any]]:
    """
    Load experiment logs from disk.
    
    Args:
        experiment_id: Filter by experiment ID (e.g., "E1.1")
        log_dir: Directory containing logs
    
    Returns:
        List of experiment log dictionaries
    """
    log_dir = log_dir or LOG_DIR
    if not log_dir.exists():
        return []
    
    logs = []
    for filepath in log_dir.glob("*.json"):
        if experiment_id and not filepath.name.startswith(experiment_id):
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            logs.append(json.load(f))
    
    return sorted(logs, key=lambda x: x.get("timestamp", ""), reverse=True)


def compute_experiment_statistics(logs: list[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate statistics from experiment logs.
    
    Returns:
        Dict with mean, std, min, max for each output metric
    """
    if not logs:
        return {}
    
    # Collect all output keys
    all_keys = set()
    for log in logs:
        all_keys.update(log.get("outputs", {}).keys())
    
    stats = {}
    for key in all_keys:
        values = [log["outputs"][key] for log in logs if key in log.get("outputs", {})]
        if values:
            import statistics
            stats[key] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }
    
    return stats



