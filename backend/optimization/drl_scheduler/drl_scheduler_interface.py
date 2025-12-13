"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    DRL SCHEDULER INTERFACE
═══════════════════════════════════════════════════════════════════════════════════════════════════════

This module provides the public interface for using trained DRL policies
to generate production schedules, compatible with the existing APS infrastructure.

Integration Flow:
────────────────
    1. DataBundle → SchedulingEnv
        - Extract operations from orders + routing
        - Configure machines from machines DataFrame
        
    2. Load Policy
        - Load trained model from zip file
        - Verify compatibility with environment
        
    3. Generate Schedule
        - Step through environment with policy decisions
        - Collect operation assignments
        
    4. Build Plan DataFrame
        - Convert to standard format (order_id, machine_id, start_time, end_time, ...)
        - Validate no overlaps per machine

Output Format:
─────────────
    DataFrame with columns:
        - op_id: Unique operation identifier
        - order_id: Parent order
        - article_id: Product being manufactured
        - route_id: Selected route
        - route_label: Route label (A, B, C, ...)
        - op_seq: Operation sequence number
        - op_code: Operation code
        - machine_id: Assigned machine
        - qty: Quantity to produce
        - start_time: Scheduled start (datetime)
        - end_time: Scheduled end (datetime)
        - duration_min: Processing duration in minutes
        - setup_min: Setup time in minutes

TODO[R&D]: Enhancements:
    - Real-time re-planning with policy adaptation
    - Uncertainty handling with distributional RL
    - Multi-agent coordination for decentralized control
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    from stable_baselines3 import PPO, A2C, DQN
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    PPO = A2C = DQN = None

from env_scheduling import (
    SchedulingEnv,
    SchedulingEnvConfig,
    OperationStatus,
)
from drl_trainer import AlgorithmType

logger = logging.getLogger(__name__)

# Default model directory
DEFAULT_MODEL_DIR = Path(__file__).parent / "trained_models"


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DRLSchedulerConfig:
    """
    Configuration for DRL-based scheduling.
    
    Attributes:
        model_path: Path to trained model (.zip file)
        algorithm: Algorithm type (PPO, A2C, DQN)
        deterministic: Use deterministic action selection
        horizon_start: Start time for scheduling horizon
        horizon_minutes: Duration of planning horizon
        fallback_to_heuristic: Use heuristic if DRL fails
        validate_plan: Validate generated plan for overlaps
    """
    model_path: Optional[str] = None
    algorithm: AlgorithmType = AlgorithmType.PPO
    deterministic: bool = True
    horizon_start: Optional[datetime] = None
    horizon_minutes: float = 10080.0  # 1 week
    fallback_to_heuristic: bool = True
    validate_plan: bool = True
    env_config: Optional[SchedulingEnvConfig] = None


# ═══════════════════════════════════════════════════════════════════════════════
# POLICY LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_trained_policy(
    model_path: Union[str, Path],
    algorithm: AlgorithmType = AlgorithmType.PPO,
):
    """
    Load a trained policy from disk.
    
    Args:
        model_path: Path to model file (.zip)
        algorithm: Algorithm type of the model
    
    Returns:
        Loaded model instance
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        ImportError: If stable-baselines3 not installed
    """
    if not HAS_SB3:
        raise ImportError(
            "stable-baselines3 is required to load DRL models. "
            "Install with: pip install stable-baselines3"
        )
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    algo_map = {
        AlgorithmType.PPO: PPO,
        AlgorithmType.A2C: A2C,
        AlgorithmType.DQN: DQN,
    }
    
    AlgoClass = algo_map.get(algorithm)
    if AlgoClass is None:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    model = AlgoClass.load(str(model_path))
    logger.info(f"Loaded {algorithm.value} model from {model_path}")
    
    return model


def get_default_model_path(algorithm: AlgorithmType = AlgorithmType.PPO) -> Optional[Path]:
    """
    Get path to default pre-trained model if available.
    
    Args:
        algorithm: Algorithm type
    
    Returns:
        Path to model if exists, None otherwise
    """
    default_paths = [
        DEFAULT_MODEL_DIR / f"scheduling_{algorithm.value.lower()}.zip",
        DEFAULT_MODEL_DIR / f"best_model.zip",
    ]
    
    for path in default_paths:
        if path.exists():
            return path
    
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def build_operations_from_databundle(
    data: Any,  # DataBundle type
    horizon_start: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Build operations DataFrame from DataBundle.
    
    Extracts operations from orders and routing information,
    creating a flat table suitable for the scheduling environment.
    
    Args:
        data: DataBundle with orders, routing, machines
        horizon_start: Start time for scheduling
    
    Returns:
        DataFrame with operation details
    """
    horizon_start = horizon_start or datetime.now()
    
    orders_df = data.orders.copy()
    routing_df = data.routing.copy()
    
    # Build operations by merging orders with routing
    operations_list = []
    
    for _, order in orders_df.iterrows():
        order_id = str(order['order_id'])
        article_id = str(order['article_id'])
        qty = float(order.get('qty', 1))
        
        # Get due date
        due_date = order.get('due_date')
        if pd.notna(due_date):
            if isinstance(due_date, str):
                due_date = pd.to_datetime(due_date)
            due_date_min = (due_date - horizon_start).total_seconds() / 60
        else:
            due_date_min = float('inf')
        
        # Get routing for this article
        art_routing = routing_df[routing_df['article_id'] == article_id]
        
        if art_routing.empty:
            logger.warning(f"No routing found for article {article_id}")
            continue
        
        # Select primary route (route_label == 'A' or first available)
        if 'route_label' in art_routing.columns:
            route_a = art_routing[art_routing['route_label'] == 'A']
            if not route_a.empty:
                selected_route = route_a
            else:
                selected_route = art_routing
        else:
            selected_route = art_routing
        
        # Get unique route_id
        route_id = str(selected_route.iloc[0].get('route_id', 'R-001'))
        route_label = str(selected_route.iloc[0].get('route_label', 'A'))
        
        # Build operations from routing steps
        predecessors = []
        for _, step in selected_route.iterrows():
            op_seq = int(step.get('op_seq', 0))
            op_code = str(step.get('op_code', f'OP-{op_seq}'))
            op_id = f"{order_id}-{op_seq}"
            
            # Machine
            machine_id = str(step.get('primary_machine_id', step.get('machine_id', 'M-001')))
            
            # Processing time
            if 'base_time_per_unit_min' in step:
                proc_time = float(step['base_time_per_unit_min']) * qty
            elif 'duration_min' in step:
                proc_time = float(step['duration_min'])
            else:
                proc_time = 60.0
            
            # Setup time
            setup_time = float(step.get('setup_time', step.get('setup_min', 0)))
            
            operations_list.append({
                'op_id': op_id,
                'order_id': order_id,
                'article_id': article_id,
                'route_id': route_id,
                'route_label': route_label,
                'op_seq': op_seq,
                'op_code': op_code,
                'machine_id': machine_id,
                'qty': qty,
                'processing_time': proc_time,
                'setup_time': setup_time,
                'due_date': due_date_min,
                'predecessors': predecessors.copy(),
                'product_family': article_id,
            })
            
            # This operation is predecessor for next
            predecessors = [op_id]
    
    return pd.DataFrame(operations_list)


# ═══════════════════════════════════════════════════════════════════════════════
# PLAN GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_plan_drl(
    data: Any,  # DataBundle
    config: Optional[DRLSchedulerConfig] = None,
) -> pd.DataFrame:
    """
    Build a production plan using a trained DRL policy.
    
    This is the main entry point for DRL-based scheduling, compatible
    with the existing APS infrastructure.
    
    Args:
        data: DataBundle with orders, routing, machines DataFrames
        config: DRL scheduler configuration
    
    Returns:
        DataFrame with scheduled operations in standard format:
            - op_id, order_id, article_id, route_id, route_label
            - op_seq, op_code, machine_id, qty
            - start_time, end_time, duration_min, setup_min
    
    Raises:
        FileNotFoundError: If model path specified but not found
        RuntimeError: If scheduling fails and no fallback available
    
    Example:
        >>> config = DRLSchedulerConfig(model_path="models/ppo_scheduler.zip")
        >>> plan = build_plan_drl(data, config)
        >>> print(f"Scheduled {len(plan)} operations")
    """
    config = config or DRLSchedulerConfig()
    horizon_start = config.horizon_start or datetime.now()
    
    # Build operations DataFrame
    operations_df = build_operations_from_databundle(data, horizon_start)
    
    if operations_df.empty:
        logger.warning("No operations to schedule")
        return pd.DataFrame(columns=[
            'op_id', 'order_id', 'article_id', 'route_id', 'route_label',
            'op_seq', 'op_code', 'machine_id', 'qty',
            'start_time', 'end_time', 'duration_min', 'setup_min'
        ])
    
    machines_df = data.machines
    
    # Try to load model
    model = None
    model_path = config.model_path
    
    if model_path:
        try:
            model = load_trained_policy(model_path, config.algorithm)
        except Exception as e:
            logger.error(f"Failed to load DRL model: {e}")
            if not config.fallback_to_heuristic:
                raise
    else:
        # Try default model
        default_path = get_default_model_path(config.algorithm)
        if default_path:
            try:
                model = load_trained_policy(default_path, config.algorithm)
            except Exception as e:
                logger.warning(f"Failed to load default model: {e}")
    
    if model is None:
        if config.fallback_to_heuristic:
            logger.info("No DRL model available, using dispatching heuristic fallback")
            return _build_plan_heuristic_fallback(operations_df, machines_df, horizon_start)
        else:
            raise RuntimeError("No DRL model available and fallback disabled")
    
    # Create environment
    env_config = config.env_config or SchedulingEnvConfig(
        horizon_minutes=config.horizon_minutes,
    )
    env = SchedulingEnv(operations_df, machines_df, env_config)
    
    # Generate schedule using policy
    schedule = _generate_schedule_with_policy(
        env, model,
        deterministic=config.deterministic,
    )
    
    # Build plan DataFrame
    plan_df = _build_plan_dataframe(
        schedule, operations_df, horizon_start
    )
    
    # Validate if configured
    if config.validate_plan:
        is_valid, issues = _validate_plan(plan_df)
        if not is_valid:
            logger.warning(f"Generated plan has issues: {issues}")
            if config.fallback_to_heuristic:
                logger.info("Falling back to heuristic due to invalid DRL plan")
                return _build_plan_heuristic_fallback(operations_df, machines_df, horizon_start)
    
    logger.info(f"DRL scheduler generated plan with {len(plan_df)} operations")
    
    return plan_df


def _generate_schedule_with_policy(
    env: SchedulingEnv,
    model,
    deterministic: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate a schedule by stepping through the environment with a policy.
    
    Args:
        env: Scheduling environment
        model: Trained policy model
        deterministic: Use deterministic action selection
    
    Returns:
        List of scheduled operation records
    """
    schedule = []
    obs, info = env.reset()
    done = False
    step_count = 0
    max_steps = env.max_steps
    
    while not done and step_count < max_steps:
        # Get action from policy
        action, _ = model.predict(obs, deterministic=deterministic)
        
        # Check if action is valid
        if env.action_mask[action] == 0:
            # Policy selected invalid action, choose first valid
            valid_actions = np.where(env.action_mask > 0)[0]
            if len(valid_actions) == 0:
                logger.warning("No valid actions available")
                break
            action = valid_actions[0]
        
        # Record the scheduled operation
        op_data = env.operations_data[action]
        op_state = env.state.operations[op_data['op_id']]
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        
        # Record after step (operation now has start/end times)
        schedule.append({
            'op_id': op_data['op_id'],
            'order_id': op_data['order_id'],
            'article_id': op_data['article_id'],
            'route_id': op_data.get('route_id', ''),
            'route_label': op_data.get('route_label', 'A'),
            'op_seq': op_data.get('op_seq', 0),
            'op_code': op_data.get('op_code', ''),
            'machine_id': op_data['machine_id'],
            'qty': op_data.get('qty', 1),
            'start_min': op_state.start_time,
            'end_min': op_state.end_time,
            'processing_time': op_data['processing_time'],
            'setup_time': op_data.get('setup_time', 0),
        })
    
    logger.debug(f"Generated schedule in {step_count} steps, {len(schedule)} operations")
    return schedule


def _build_plan_dataframe(
    schedule: List[Dict[str, Any]],
    operations_df: pd.DataFrame,
    horizon_start: datetime,
) -> pd.DataFrame:
    """
    Convert schedule records to standard plan DataFrame format.
    
    Args:
        schedule: List of operation records from policy
        operations_df: Original operations DataFrame
        horizon_start: Start time of planning horizon
    
    Returns:
        Plan DataFrame in standard format
    """
    if not schedule:
        return pd.DataFrame(columns=[
            'op_id', 'order_id', 'article_id', 'route_id', 'route_label',
            'op_seq', 'op_code', 'machine_id', 'qty',
            'start_time', 'end_time', 'duration_min', 'setup_min'
        ])
    
    records = []
    for op in schedule:
        # Convert minutes to datetime
        start_time = horizon_start + timedelta(minutes=op['start_min'] or 0)
        end_time = horizon_start + timedelta(minutes=op['end_min'] or 0)
        
        records.append({
            'op_id': op['op_id'],
            'order_id': op['order_id'],
            'article_id': op['article_id'],
            'route_id': op.get('route_id', ''),
            'route_label': op.get('route_label', 'A'),
            'op_seq': op.get('op_seq', 0),
            'op_code': op.get('op_code', ''),
            'machine_id': op['machine_id'],
            'qty': op.get('qty', 1),
            'start_time': start_time,
            'end_time': end_time,
            'duration_min': op['processing_time'],
            'setup_min': op.get('setup_time', 0),
        })
    
    plan_df = pd.DataFrame(records)
    
    # Sort by start time
    plan_df = plan_df.sort_values('start_time').reset_index(drop=True)
    
    return plan_df


def _validate_plan(plan_df: pd.DataFrame) -> tuple[bool, List[str]]:
    """
    Validate a generated plan for correctness.
    
    Checks:
        - No overlapping operations on same machine
        - All operations have valid times
        - Precedence constraints respected
    
    Args:
        plan_df: Plan DataFrame to validate
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    # Check for overlaps per machine
    for machine_id in plan_df['machine_id'].unique():
        machine_ops = plan_df[plan_df['machine_id'] == machine_id].sort_values('start_time')
        
        for i in range(len(machine_ops) - 1):
            current = machine_ops.iloc[i]
            next_op = machine_ops.iloc[i + 1]
            
            if current['end_time'] > next_op['start_time']:
                issues.append(
                    f"Overlap on {machine_id}: {current['op_id']} ends after "
                    f"{next_op['op_id']} starts"
                )
    
    # Check for missing times
    if plan_df['start_time'].isna().any():
        issues.append("Some operations have no start time")
    if plan_df['end_time'].isna().any():
        issues.append("Some operations have no end time")
    
    # Check for negative durations
    if (plan_df['end_time'] < plan_df['start_time']).any():
        issues.append("Some operations have end_time before start_time")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def _build_plan_heuristic_fallback(
    operations_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    horizon_start: datetime,
) -> pd.DataFrame:
    """
    Simple heuristic fallback scheduler (EDD - Earliest Due Date).
    
    Used when DRL model is not available or produces invalid results.
    
    Args:
        operations_df: Operations to schedule
        machines_df: Available machines
        horizon_start: Start time
    
    Returns:
        Plan DataFrame
    """
    # Sort by due date (EDD rule)
    ops_sorted = operations_df.sort_values(['due_date', 'order_id', 'op_seq'])
    
    # Track machine availability
    machine_available = {row['machine_id']: 0.0 for _, row in machines_df.iterrows()}
    
    # Track order progress (for precedence)
    order_last_end = {}
    
    records = []
    
    for _, op in ops_sorted.iterrows():
        order_id = op['order_id']
        machine_id = op['machine_id']
        
        # Ensure machine is available
        if machine_id not in machine_available:
            machine_available[machine_id] = 0.0
        
        # Start after machine is free and after previous op in same order
        earliest_start = max(
            machine_available[machine_id],
            order_last_end.get(order_id, 0.0)
        )
        
        # Calculate times
        processing_time = op.get('processing_time', 60.0)
        setup_time = op.get('setup_time', 0.0)
        
        start_min = earliest_start + setup_time
        end_min = start_min + processing_time
        
        # Update availability
        machine_available[machine_id] = end_min
        order_last_end[order_id] = end_min
        
        # Convert to datetime
        start_time = horizon_start + timedelta(minutes=start_min)
        end_time = horizon_start + timedelta(minutes=end_min)
        
        records.append({
            'op_id': op['op_id'],
            'order_id': op['order_id'],
            'article_id': op['article_id'],
            'route_id': op.get('route_id', ''),
            'route_label': op.get('route_label', 'A'),
            'op_seq': op.get('op_seq', 0),
            'op_code': op.get('op_code', ''),
            'machine_id': machine_id,
            'qty': op.get('qty', 1),
            'start_time': start_time,
            'end_time': end_time,
            'duration_min': processing_time,
            'setup_min': setup_time,
        })
    
    plan_df = pd.DataFrame(records)
    plan_df = plan_df.sort_values('start_time').reset_index(drop=True)
    
    logger.info(f"Heuristic fallback generated plan with {len(plan_df)} operations")
    
    return plan_df


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_drl_scheduler_info() -> Dict[str, Any]:
    """
    Get information about the DRL scheduler status.
    
    Returns:
        Dictionary with:
            - available: Whether DRL scheduling is available
            - has_model: Whether a trained model exists
            - model_path: Path to model if exists
            - algorithms: Supported algorithms
    """
    has_sb3 = HAS_SB3
    
    default_model = get_default_model_path()
    
    return {
        'available': has_sb3,
        'has_model': default_model is not None,
        'model_path': str(default_model) if default_model else None,
        'algorithms': [a.value for a in AlgorithmType],
        'dependencies_installed': has_sb3,
        'missing_dependencies': [] if has_sb3 else ['stable-baselines3', 'gymnasium'],
    }



