"""
════════════════════════════════════════════════════════════════════════════════════════════════════
Data-Driven Durations - Historical-based Duration Estimation
════════════════════════════════════════════════════════════════════════════════════════════════════

Contract 9 Implementation: Use shopfloor execution logs to estimate operation durations

Features:
- Uses OperationExecutionLog for real duration data
- Fallback to standard times when no history
- Context-aware estimation (machine, shift, product mix)
- Confidence intervals
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DurationEstimate:
    """Result of duration estimation."""
    operation_id: str
    machine_id: Optional[str]
    
    # Estimated duration
    duration_s: float
    setup_time_s: float
    cycle_time_per_piece_s: float
    
    # Confidence
    confidence: float  # 0-1
    n_samples: int
    std_dev_s: float
    
    # Bounds
    lower_bound_s: float  # 5th percentile
    upper_bound_s: float  # 95th percentile
    
    # Source
    source: str  # "historical", "standard", "hybrid"
    
    def total_for_qty(self, qty: int) -> float:
        """Get total duration for a quantity."""
        return self.setup_time_s + self.cycle_time_per_piece_s * qty


@dataclass
class EstimationContext:
    """Context for duration estimation."""
    machine_id: Optional[str] = None
    operator_id: Optional[str] = None
    shift: Optional[str] = None  # "day", "night"
    batch_size: Optional[int] = None
    days_lookback: int = 30
    min_samples: int = 5


class DataDrivenDurations:
    """
    Estimates operation durations based on historical execution logs.
    
    Falls back to standard times when insufficient historical data.
    """
    
    def __init__(
        self,
        standard_times: Optional[Dict[str, float]] = None,
        default_cycle_time_s: float = 60.0,
        default_setup_time_s: float = 300.0,
    ):
        """
        Initialize with optional standard times for fallback.
        
        Args:
            standard_times: Dict mapping operation_id to standard cycle time (seconds)
            default_cycle_time_s: Default cycle time when no data available
            default_setup_time_s: Default setup time when no data available
        """
        self.standard_times = standard_times or {}
        self.default_cycle_time_s = default_cycle_time_s
        self.default_setup_time_s = default_setup_time_s
        
        logger.info("DataDrivenDurations initialized")
    
    def estimate_duration(
        self,
        operation_id: str,
        qty: int = 1,
        context: Optional[EstimationContext] = None,
    ) -> DurationEstimate:
        """
        Estimate duration for an operation based on historical data.
        
        Args:
            operation_id: Operation code to estimate
            qty: Quantity to produce
            context: Optional context (machine, shift, etc.)
        
        Returns:
            DurationEstimate with duration and confidence
        """
        context = context or EstimationContext()
        
        # Try to get historical data
        logs = self._get_historical_logs(operation_id, context)
        
        if len(logs) >= context.min_samples:
            return self._estimate_from_logs(operation_id, qty, logs, context)
        elif len(logs) > 0:
            # Hybrid: combine with standard times
            return self._estimate_hybrid(operation_id, qty, logs, context)
        else:
            # Fallback to standard times
            return self._estimate_standard(operation_id, qty, context)
    
    def _get_historical_logs(
        self,
        operation_id: str,
        context: EstimationContext,
    ) -> List[Dict[str, Any]]:
        """Get historical execution logs for operation."""
        try:
            from prodplan.execution_log_models import (
                query_execution_logs,
                ExecutionLogQuery,
                ExecutionLogStatus,
            )
            
            query = ExecutionLogQuery(
                operation_id=operation_id,
                machine_id=context.machine_id,
                status=ExecutionLogStatus.COMPLETED,
                from_date=datetime.utcnow() - timedelta(days=context.days_lookback),
                limit=200,
            )
            
            logs = query_execution_logs(query)
            
            # Convert to dicts for processing
            return [
                {
                    "cycle_time_s": log.cycle_time_s,
                    "setup_time_s": log.setup_time_s,
                    "qty_good": log.qty_good,
                    "qty_scrap": log.qty_scrap,
                    "machine_id": log.machine_id,
                    "start_time": log.start_time,
                }
                for log in logs
                if log.cycle_time_s > 0 or log.setup_time_s > 0
            ]
        except ImportError:
            logger.warning("Execution log models not available")
            return []
        except Exception as e:
            logger.error(f"Error fetching execution logs: {e}")
            return []
    
    def _estimate_from_logs(
        self,
        operation_id: str,
        qty: int,
        logs: List[Dict[str, Any]],
        context: EstimationContext,
    ) -> DurationEstimate:
        """Estimate duration from historical logs."""
        cycle_times = [log["cycle_time_s"] for log in logs if log["cycle_time_s"] > 0]
        setup_times = [log["setup_time_s"] for log in logs if log["setup_time_s"] > 0]
        
        # Calculate statistics
        avg_cycle = float(np.mean(cycle_times)) if cycle_times else self.default_cycle_time_s
        std_cycle = float(np.std(cycle_times)) if len(cycle_times) > 1 else avg_cycle * 0.1
        avg_setup = float(np.mean(setup_times)) if setup_times else self.default_setup_time_s
        
        # Percentiles for bounds
        if len(cycle_times) >= 5:
            lower_cycle = float(np.percentile(cycle_times, 5))
            upper_cycle = float(np.percentile(cycle_times, 95))
        else:
            lower_cycle = avg_cycle - 2 * std_cycle
            upper_cycle = avg_cycle + 2 * std_cycle
        
        total_duration = avg_setup + avg_cycle * qty
        lower_bound = avg_setup + lower_cycle * qty
        upper_bound = avg_setup + upper_cycle * qty
        
        # Confidence based on sample size
        confidence = min(1.0, 0.5 + len(logs) / 100)
        
        return DurationEstimate(
            operation_id=operation_id,
            machine_id=context.machine_id,
            duration_s=total_duration,
            setup_time_s=avg_setup,
            cycle_time_per_piece_s=avg_cycle,
            confidence=confidence,
            n_samples=len(logs),
            std_dev_s=std_cycle,
            lower_bound_s=max(0, lower_bound),
            upper_bound_s=upper_bound,
            source="historical",
        )
    
    def _estimate_hybrid(
        self,
        operation_id: str,
        qty: int,
        logs: List[Dict[str, Any]],
        context: EstimationContext,
    ) -> DurationEstimate:
        """Hybrid estimation: combine historical with standard."""
        # Get historical averages
        cycle_times = [log["cycle_time_s"] for log in logs if log["cycle_time_s"] > 0]
        setup_times = [log["setup_time_s"] for log in logs if log["setup_time_s"] > 0]
        
        historical_cycle = float(np.mean(cycle_times)) if cycle_times else None
        historical_setup = float(np.mean(setup_times)) if setup_times else None
        
        # Get standard times
        standard_cycle = self.standard_times.get(operation_id, self.default_cycle_time_s)
        standard_setup = self.default_setup_time_s
        
        # Weighted average (weight historical more if more samples)
        weight = len(logs) / context.min_samples
        
        if historical_cycle is not None:
            avg_cycle = weight * historical_cycle + (1 - weight) * standard_cycle
        else:
            avg_cycle = standard_cycle
        
        if historical_setup is not None:
            avg_setup = weight * historical_setup + (1 - weight) * standard_setup
        else:
            avg_setup = standard_setup
        
        total_duration = avg_setup + avg_cycle * qty
        
        return DurationEstimate(
            operation_id=operation_id,
            machine_id=context.machine_id,
            duration_s=total_duration,
            setup_time_s=avg_setup,
            cycle_time_per_piece_s=avg_cycle,
            confidence=0.3 + 0.2 * len(logs),  # Low confidence for hybrid
            n_samples=len(logs),
            std_dev_s=avg_cycle * 0.2,  # Estimate 20% variability
            lower_bound_s=total_duration * 0.8,
            upper_bound_s=total_duration * 1.4,
            source="hybrid",
        )
    
    def _estimate_standard(
        self,
        operation_id: str,
        qty: int,
        context: EstimationContext,
    ) -> DurationEstimate:
        """Fallback to standard times."""
        cycle_time = self.standard_times.get(operation_id, self.default_cycle_time_s)
        setup_time = self.default_setup_time_s
        
        total_duration = setup_time + cycle_time * qty
        
        return DurationEstimate(
            operation_id=operation_id,
            machine_id=context.machine_id,
            duration_s=total_duration,
            setup_time_s=setup_time,
            cycle_time_per_piece_s=cycle_time,
            confidence=0.3,  # Low confidence for standard times
            n_samples=0,
            std_dev_s=cycle_time * 0.3,  # Assume 30% variability
            lower_bound_s=total_duration * 0.7,
            upper_bound_s=total_duration * 1.5,
            source="standard",
        )
    
    def get_all_operation_stats(
        self,
        days_lookback: int = 30,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all operations with historical data.
        
        Returns dict mapping operation_id to stats.
        """
        try:
            from prodplan.execution_log_models import get_execution_stats
            
            # Get unique operation IDs from recent logs
            logs = self._get_all_recent_logs(days_lookback)
            operation_ids = set(log.get("operation_id") for log in logs if log.get("operation_id"))
            
            stats = {}
            for op_id in operation_ids:
                op_stats = get_execution_stats(op_id, days=days_lookback)
                stats[op_id] = {
                    "avg_cycle_time_s": op_stats.avg_cycle_time_s,
                    "std_cycle_time_s": op_stats.std_cycle_time_s,
                    "avg_setup_time_s": op_stats.avg_setup_time_s,
                    "n_executions": op_stats.n_executions,
                    "avg_scrap_rate": op_stats.avg_scrap_rate,
                }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting operation stats: {e}")
            return {}
    
    def _get_all_recent_logs(self, days: int) -> List[Dict[str, Any]]:
        """Get all recent logs (for stats gathering)."""
        try:
            from prodplan.execution_log_models import query_execution_logs, ExecutionLogQuery
            
            query = ExecutionLogQuery(
                from_date=datetime.utcnow() - timedelta(days=days),
                limit=1000,
            )
            logs = query_execution_logs(query)
            return [{"operation_id": log.operation_id} for log in logs]
        except Exception:
            return []


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON / FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

_instance: Optional[DataDrivenDurations] = None


def get_data_driven_durations(
    standard_times: Optional[Dict[str, float]] = None,
) -> DataDrivenDurations:
    """Get singleton DataDrivenDurations instance."""
    global _instance
    if _instance is None:
        _instance = DataDrivenDurations(standard_times)
    return _instance


def estimate_operation_duration(
    operation_id: str,
    qty: int = 1,
    machine_id: Optional[str] = None,
    use_historical: bool = True,
) -> DurationEstimate:
    """
    Convenience function to estimate operation duration.
    
    Args:
        operation_id: Operation to estimate
        qty: Quantity to produce
        machine_id: Optional machine filter
        use_historical: Whether to use historical data
    
    Returns:
        DurationEstimate
    """
    engine = get_data_driven_durations()
    
    if not use_historical:
        return engine._estimate_standard(
            operation_id, qty, EstimationContext(machine_id=machine_id)
        )
    
    context = EstimationContext(machine_id=machine_id)
    return engine.estimate_duration(operation_id, qty, context)
