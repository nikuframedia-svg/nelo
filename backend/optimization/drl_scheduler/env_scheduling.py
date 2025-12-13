"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    SCHEDULING ENVIRONMENT (Gymnasium)
═══════════════════════════════════════════════════════════════════════════════════════════════════════

This module implements a Gymnasium-compatible environment for production scheduling.
The environment models the job shop scheduling problem as a Markov Decision Process (MDP).

MDP Formulation:
────────────────

    State Space S ∈ ℝ^(n_machines × 4 + n_operations × 5 + 1)
    ─────────────────────────────────────────────────────────
    For each machine m ∈ M:
        - available_time: when machine becomes free
        - current_setup: product family currently set up
        - queue_length: number of operations waiting
        - utilization: cumulative utilization rate
    
    For each operation o ∈ O:
        - status: {0=waiting, 1=ready, 2=scheduled, 3=completed}
        - processing_time: duration in minutes
        - due_date_slack: (due_date - current_time) / horizon
        - machine_id: assigned machine (one-hot or index)
        - priority: order priority weight
    
    Global:
        - current_time: normalized ∈ [0, 1]

    Action Space A ∈ {0, 1, ..., n_operations - 1}
    ───────────────────────────────────────────────
    Action a_t selects operation to dispatch next.
    Invalid actions (already scheduled or not ready) are masked.

    Reward Function R: S × A × S' → ℝ
    ──────────────────────────────────
    R(s, a, s') = -α·τ(a) - β·Δ_makespan(a) - γ·σ(a) + δ·φ(a) + ε·c(a)
    
    where:
        τ(a) = max(0, C_a - d_a) / horizon        # normalized tardiness
        Δ_makespan(a) = (new_makespan - old_makespan) / horizon
        σ(a) = setup_time(a) / max_setup          # normalized setup penalty
        φ(a) = 1 if machine was idle, 0 otherwise # flow bonus
        c(a) = 1 if on-time delivery, 0 otherwise # completion bonus

    Termination:
    ────────────
    Episode ends when all operations are scheduled or max_steps reached.

TODO[R&D]: Advanced features for future work:
    - Partial observability (POMDP) for uncertain processing times
    - Multi-agent formulation for decentralized control
    - Continuous action space for fine-grained timing
    - Graph neural network state representation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    # Fallback for type hints
    class gym:
        class Env:
            pass
    class spaces:
        @staticmethod
        def Box(*args, **kwargs):
            pass
        @staticmethod
        def Discrete(*args, **kwargs):
            pass
        @staticmethod
        def Dict(*args, **kwargs):
            pass


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class OperationStatus(IntEnum):
    """Status of an operation in the scheduling process."""
    WAITING = 0      # Predecessors not completed
    READY = 1        # Can be scheduled (predecessors done)
    SCHEDULED = 2    # Assigned to machine, not yet completed
    COMPLETED = 3    # Finished processing


@dataclass
class MachineState:
    """
    State of a machine in the scheduling environment.
    
    Attributes:
        machine_id: Unique identifier
        available_time: Time when machine becomes free (minutes from horizon start)
        current_setup: Product family currently configured (for setup time calculation)
        queue: List of operation IDs waiting on this machine
        total_load: Cumulative processing time assigned
        speed_factor: Machine efficiency multiplier (1.0 = nominal)
    """
    machine_id: str
    available_time: float = 0.0
    current_setup: Optional[str] = None
    queue: List[str] = field(default_factory=list)
    total_load: float = 0.0
    speed_factor: float = 1.0
    
    def to_vector(self, max_time: float, max_queue: int = 10) -> np.ndarray:
        """Convert to normalized feature vector."""
        return np.array([
            self.available_time / max_time if max_time > 0 else 0.0,
            len(self.queue) / max_queue,
            self.total_load / max_time if max_time > 0 else 0.0,
            self.speed_factor,
        ], dtype=np.float32)


@dataclass
class OperationState:
    """
    State of an operation in the scheduling environment.
    
    Attributes:
        op_id: Unique operation identifier
        order_id: Parent order
        article_id: Product being manufactured
        machine_id: Assigned machine
        processing_time: Duration in minutes
        setup_time: Setup time if product family changes
        due_date: Order due date (minutes from horizon start)
        status: Current operation status
        predecessors: Operations that must complete before this one
        start_time: When operation starts (None if not scheduled)
        end_time: When operation ends (None if not scheduled)
        priority: Scheduling priority weight
        product_family: For setup time grouping
    """
    op_id: str
    order_id: str
    article_id: str
    machine_id: str
    processing_time: float
    setup_time: float = 0.0
    due_date: float = float('inf')
    status: OperationStatus = OperationStatus.WAITING
    predecessors: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    priority: float = 1.0
    product_family: Optional[str] = None
    
    def to_vector(self, max_time: float, n_machines: int, machine_idx: int) -> np.ndarray:
        """Convert to normalized feature vector."""
        due_slack = (self.due_date - (self.end_time or 0)) / max_time if max_time > 0 else 0.0
        return np.array([
            self.status / 3.0,  # Normalized status
            self.processing_time / max_time if max_time > 0 else 0.0,
            np.clip(due_slack, -1, 1),  # Clipped slack
            machine_idx / n_machines if n_machines > 0 else 0.0,
            self.priority,
        ], dtype=np.float32)


@dataclass
class EnvState:
    """
    Complete environment state for observation.
    
    Mathematical representation:
        s = (M, O, t)
        where M = {m_1, ..., m_|M|} are machine states
              O = {o_1, ..., o_|O|} are operation states
              t = current time
    """
    machines: Dict[str, MachineState]
    operations: Dict[str, OperationState]
    current_time: float
    horizon: float
    completed_count: int = 0
    total_tardiness: float = 0.0
    total_setup: float = 0.0
    makespan: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SchedulingEnvConfig:
    """
    Configuration for the scheduling environment.
    
    Reward weights follow the formulation:
        R = -α·tardiness - β·makespan - γ·setup + δ·flow + ε·completion
    
    Attributes:
        horizon_minutes: Planning horizon duration
        reward_tardiness_weight: α - penalty for late delivery
        reward_makespan_weight: β - penalty for makespan increase
        reward_setup_weight: γ - penalty for setup changes
        reward_flow_weight: δ - bonus for continuous machine utilization
        reward_completion_weight: ε - bonus for on-time completion
        enable_perturbations: Allow random machine failures/delays
        perturbation_probability: Probability of perturbation per step
        max_steps_factor: Max steps = factor × n_operations
        normalize_rewards: Scale rewards to [-1, 1] range
    """
    horizon_minutes: float = 10080.0  # 1 week default
    reward_tardiness_weight: float = 1.0      # α
    reward_makespan_weight: float = 0.5       # β
    reward_setup_weight: float = 0.2          # γ
    reward_flow_weight: float = 0.1           # δ
    reward_completion_weight: float = 0.3     # ε
    enable_perturbations: bool = False
    perturbation_probability: float = 0.01
    max_steps_factor: float = 2.0
    normalize_rewards: bool = True
    seed: Optional[int] = None


# ═══════════════════════════════════════════════════════════════════════════════
# GYMNASIUM ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

class SchedulingEnv(gym.Env if HAS_GYMNASIUM else object):
    """
    Gymnasium environment for production scheduling.
    
    This environment models a job shop scheduling problem where an agent
    must decide the order in which operations are dispatched to machines.
    
    The agent observes:
        - Machine states (availability, setup, queue)
        - Operation states (status, times, due dates)
        - Global time
    
    And takes actions:
        - Select an operation index to dispatch next
    
    Episode terminates when:
        - All operations are scheduled
        - Maximum steps reached
    
    Example:
        >>> config = SchedulingEnvConfig(horizon_minutes=1440)
        >>> env = SchedulingEnv(operations_df, machines_df, config)
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()  # Random action
        >>> obs, reward, terminated, truncated, info = env.step(action)
    
    TODO[R&D]: Improvements for research:
        - Hierarchical action space (machine selection + operation selection)
        - Attention-based observation encoding for variable-size problems
        - Curriculum learning with progressive complexity
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}
    
    def __init__(
        self,
        operations_df: pd.DataFrame,
        machines_df: pd.DataFrame,
        config: Optional[SchedulingEnvConfig] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the scheduling environment.
        
        Args:
            operations_df: DataFrame with columns:
                - op_id, order_id, article_id, machine_id
                - processing_time (or base_time_per_unit_min + qty)
                - due_date (optional)
                - predecessors (optional, list or comma-separated)
            machines_df: DataFrame with columns:
                - machine_id
                - speed_factor (optional)
            config: Environment configuration
            render_mode: Rendering mode ("human" or "ansi")
        """
        super().__init__()
        
        if not HAS_GYMNASIUM:
            raise ImportError(
                "gymnasium is required for DRL scheduling. "
                "Install with: pip install gymnasium"
            )
        
        self.config = config or SchedulingEnvConfig()
        self.render_mode = render_mode
        self.rng = np.random.default_rng(self.config.seed)
        
        # Parse input data
        self._parse_operations(operations_df)
        self._parse_machines(machines_df)
        
        # Environment dimensions
        self.n_operations = len(self.operations_data)
        self.n_machines = len(self.machines_data)
        
        # Define observation space
        # Machine features: 4 per machine
        # Operation features: 5 per operation
        # Global features: 1 (current time)
        machine_obs_dim = self.n_machines * 4
        operation_obs_dim = self.n_operations * 5
        global_obs_dim = 1
        total_obs_dim = machine_obs_dim + operation_obs_dim + global_obs_dim
        
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(total_obs_dim,),
            dtype=np.float32,
        )
        
        # Define action space: select operation index
        self.action_space = spaces.Discrete(self.n_operations)
        
        # Action mask for invalid actions
        self.action_mask = np.ones(self.n_operations, dtype=np.float32)
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = int(self.n_operations * self.config.max_steps_factor)
        
        # State
        self.state: Optional[EnvState] = None
        
        # Metrics for logging
        self.episode_reward = 0.0
        self.episode_info: Dict[str, Any] = {}
        
        logger.info(
            f"SchedulingEnv initialized: {self.n_operations} operations, "
            f"{self.n_machines} machines, horizon={self.config.horizon_minutes}min"
        )
    
    def _parse_operations(self, df: pd.DataFrame) -> None:
        """Parse operations DataFrame into internal format."""
        self.operations_data: List[Dict[str, Any]] = []
        self.op_id_to_idx: Dict[str, int] = {}
        
        for idx, row in df.iterrows():
            op_id = str(row.get('op_id', f'OP-{idx}'))
            
            # Calculate processing time
            if 'processing_time' in row:
                proc_time = float(row['processing_time'])
            elif 'duration_min' in row:
                proc_time = float(row['duration_min'])
            elif 'base_time_per_unit_min' in row and 'qty' in row:
                proc_time = float(row['base_time_per_unit_min']) * float(row.get('qty', 1))
            else:
                proc_time = 60.0  # Default 1 hour
            
            # Parse predecessors
            preds = row.get('predecessors', [])
            if isinstance(preds, str):
                preds = [p.strip() for p in preds.split(',') if p.strip()]
            elif preds is None or (isinstance(preds, float) and np.isnan(preds)):
                preds = []
            
            # Due date (convert to minutes from start if datetime)
            due_date = row.get('due_date', float('inf'))
            if pd.isna(due_date):
                due_date = self.config.horizon_minutes
            elif isinstance(due_date, (pd.Timestamp, np.datetime64)):
                # Assume due_date is absolute; we'll normalize in reset()
                due_date = self.config.horizon_minutes  # Placeholder
            else:
                due_date = float(due_date)
            
            op_data = {
                'op_id': op_id,
                'order_id': str(row.get('order_id', '')),
                'article_id': str(row.get('article_id', '')),
                'machine_id': str(row.get('machine_id', row.get('primary_machine_id', 'M-001'))),
                'processing_time': proc_time,
                'setup_time': float(row.get('setup_time', row.get('setup_min', 0))),
                'due_date': due_date,
                'predecessors': preds,
                'priority': float(row.get('priority', 1.0)),
                'product_family': str(row.get('product_family', row.get('article_id', ''))),
            }
            
            self.op_id_to_idx[op_id] = len(self.operations_data)
            self.operations_data.append(op_data)
    
    def _parse_machines(self, df: pd.DataFrame) -> None:
        """Parse machines DataFrame into internal format."""
        self.machines_data: List[Dict[str, Any]] = []
        self.machine_id_to_idx: Dict[str, int] = {}
        
        for idx, row in df.iterrows():
            machine_id = str(row.get('machine_id', f'M-{idx}'))
            
            machine_data = {
                'machine_id': machine_id,
                'speed_factor': float(row.get('speed_factor', 1.0)),
            }
            
            self.machine_id_to_idx[machine_id] = len(self.machines_data)
            self.machines_data.append(machine_data)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
        
        Returns:
            observation: Initial observation vector
            info: Additional information dict
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Initialize machine states
        machines: Dict[str, MachineState] = {}
        for m_data in self.machines_data:
            machines[m_data['machine_id']] = MachineState(
                machine_id=m_data['machine_id'],
                available_time=0.0,
                current_setup=None,
                queue=[],
                total_load=0.0,
                speed_factor=m_data['speed_factor'],
            )
        
        # Initialize operation states
        operations: Dict[str, OperationState] = {}
        for op_data in self.operations_data:
            # Determine initial status based on predecessors
            has_predecessors = len(op_data['predecessors']) > 0
            initial_status = OperationStatus.WAITING if has_predecessors else OperationStatus.READY
            
            operations[op_data['op_id']] = OperationState(
                op_id=op_data['op_id'],
                order_id=op_data['order_id'],
                article_id=op_data['article_id'],
                machine_id=op_data['machine_id'],
                processing_time=op_data['processing_time'],
                setup_time=op_data['setup_time'],
                due_date=op_data['due_date'],
                status=initial_status,
                predecessors=op_data['predecessors'].copy(),
                priority=op_data['priority'],
                product_family=op_data['product_family'],
            )
        
        # Create initial state
        self.state = EnvState(
            machines=machines,
            operations=operations,
            current_time=0.0,
            horizon=self.config.horizon_minutes,
            completed_count=0,
            total_tardiness=0.0,
            total_setup=0.0,
            makespan=0.0,
        )
        
        # Reset tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_info = {
            'scheduled_ops': 0,
            'total_tardiness': 0.0,
            'total_setup': 0.0,
            'makespan': 0.0,
        }
        
        # Update action mask
        self._update_action_mask()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Index of operation to dispatch
        
        Returns:
            observation: New observation after action
            reward: Reward for this step
            terminated: True if episode ended normally
            truncated: True if episode was cut short
            info: Additional information
        """
        assert self.state is not None, "Environment not reset"
        
        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = False
        
        # Validate action
        if action < 0 or action >= self.n_operations:
            # Invalid action index
            reward = -1.0
            logger.warning(f"Invalid action index: {action}")
        elif self.action_mask[action] == 0:
            # Operation not dispatchable
            reward = -0.5  # Penalty for invalid action
            logger.debug(f"Action {action} not dispatchable (masked)")
        else:
            # Execute valid action
            reward = self._execute_action(action)
        
        # Apply perturbations if enabled
        if self.config.enable_perturbations:
            reward += self._apply_perturbations()
        
        # Update action mask for next step
        self._update_action_mask()
        
        # Check termination
        if self.state.completed_count >= self.n_operations:
            terminated = True
            # Bonus for completing all operations
            reward += self.config.reward_completion_weight
        elif self.current_step >= self.max_steps:
            truncated = True
            # Penalty for not finishing
            remaining = self.n_operations - self.state.completed_count
            reward -= 0.1 * remaining
        elif np.sum(self.action_mask) == 0:
            # No valid actions (deadlock)
            truncated = True
            reward -= 1.0
            logger.warning("Environment deadlocked - no valid actions")
        
        # Normalize reward if configured
        if self.config.normalize_rewards:
            reward = np.clip(reward, -1.0, 1.0)
        
        self.episode_reward += reward
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, action: int) -> float:
        """
        Execute a scheduling action.
        
        Args:
            action: Operation index to dispatch
        
        Returns:
            reward: Immediate reward for this action
        """
        op_data = self.operations_data[action]
        op_id = op_data['op_id']
        op_state = self.state.operations[op_id]
        machine_id = op_state.machine_id
        machine_state = self.state.machines[machine_id]
        
        # Calculate start time (max of machine availability and current time)
        start_time = max(machine_state.available_time, self.state.current_time)
        
        # Check for setup time
        setup_time = 0.0
        if machine_state.current_setup != op_state.product_family:
            setup_time = op_state.setup_time
            self.state.total_setup += setup_time
        
        # Calculate processing time (adjusted by machine speed)
        actual_proc_time = op_state.processing_time / machine_state.speed_factor
        
        # Calculate end time
        end_time = start_time + setup_time + actual_proc_time
        
        # Update operation state
        op_state.status = OperationStatus.COMPLETED
        op_state.start_time = start_time
        op_state.end_time = end_time
        
        # Update machine state
        machine_state.available_time = end_time
        machine_state.current_setup = op_state.product_family
        machine_state.total_load += actual_proc_time
        
        # Update global state
        self.state.completed_count += 1
        old_makespan = self.state.makespan
        self.state.makespan = max(self.state.makespan, end_time)
        
        # Update dependencies - mark successors as ready if all predecessors done
        self._update_dependencies(op_id)
        
        # Advance global time to next event if needed
        self._advance_time()
        
        # Calculate reward components
        reward = 0.0
        
        # Tardiness penalty
        tardiness = max(0, end_time - op_state.due_date)
        if tardiness > 0:
            normalized_tardiness = tardiness / self.state.horizon
            reward -= self.config.reward_tardiness_weight * normalized_tardiness
            self.state.total_tardiness += tardiness
        else:
            # Bonus for on-time
            reward += self.config.reward_completion_weight * 0.1
        
        # Makespan penalty
        makespan_increase = self.state.makespan - old_makespan
        if makespan_increase > 0:
            normalized_makespan = makespan_increase / self.state.horizon
            reward -= self.config.reward_makespan_weight * normalized_makespan
        
        # Setup penalty
        if setup_time > 0:
            max_setup = max(op['setup_time'] for op in self.operations_data) or 1.0
            normalized_setup = setup_time / max_setup
            reward -= self.config.reward_setup_weight * normalized_setup
        
        # Flow bonus (machine wasn't idle for long)
        idle_time = start_time - (machine_state.available_time - actual_proc_time - setup_time)
        if idle_time < actual_proc_time * 0.1:  # Less than 10% idle
            reward += self.config.reward_flow_weight
        
        # Update episode info
        self.episode_info['scheduled_ops'] = self.state.completed_count
        self.episode_info['total_tardiness'] = self.state.total_tardiness
        self.episode_info['total_setup'] = self.state.total_setup
        self.episode_info['makespan'] = self.state.makespan
        
        return reward
    
    def _update_dependencies(self, completed_op_id: str) -> None:
        """Update operation statuses based on completed operation."""
        for op_id, op_state in self.state.operations.items():
            if op_state.status == OperationStatus.WAITING:
                if completed_op_id in op_state.predecessors:
                    # Remove from predecessors
                    remaining_preds = [
                        p for p in op_state.predecessors
                        if self.state.operations.get(p, op_state).status != OperationStatus.COMPLETED
                    ]
                    if len(remaining_preds) == 0:
                        # All predecessors done - operation is ready
                        op_state.status = OperationStatus.READY
    
    def _advance_time(self) -> None:
        """Advance global time to next event if no operations are immediately ready."""
        ready_ops = [
            op for op in self.state.operations.values()
            if op.status == OperationStatus.READY
        ]
        
        if not ready_ops:
            # No ready operations - advance time to next machine availability
            min_available = min(
                m.available_time for m in self.state.machines.values()
            )
            self.state.current_time = max(self.state.current_time, min_available)
    
    def _update_action_mask(self) -> None:
        """Update mask of valid actions."""
        self.action_mask = np.zeros(self.n_operations, dtype=np.float32)
        
        for idx, op_data in enumerate(self.operations_data):
            op_id = op_data['op_id']
            op_state = self.state.operations[op_id]
            
            # Only ready operations can be dispatched
            if op_state.status == OperationStatus.READY:
                # Check if machine is available (within reasonable time)
                machine_state = self.state.machines.get(op_state.machine_id)
                if machine_state:
                    self.action_mask[idx] = 1.0
    
    def _apply_perturbations(self) -> float:
        """Apply random perturbations (machine failures, delays)."""
        reward_adjustment = 0.0
        
        if self.rng.random() < self.config.perturbation_probability:
            # Random machine delay
            machine_id = self.rng.choice(list(self.state.machines.keys()))
            machine = self.state.machines[machine_id]
            delay = self.rng.uniform(10, 60)  # 10-60 minutes
            machine.available_time += delay
            reward_adjustment -= 0.1  # Penalty for disruption
            logger.debug(f"Perturbation: {machine_id} delayed by {delay:.1f} minutes")
        
        return reward_adjustment
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector from current state.
        
        Observation structure:
            [machine_features..., operation_features..., global_features]
        """
        obs_parts = []
        
        # Machine features (sorted by machine_id for consistency)
        for m_data in self.machines_data:
            machine = self.state.machines[m_data['machine_id']]
            obs_parts.append(machine.to_vector(self.state.horizon))
        
        # Operation features (sorted by original order)
        for idx, op_data in enumerate(self.operations_data):
            op_state = self.state.operations[op_data['op_id']]
            machine_idx = self.machine_id_to_idx.get(op_state.machine_id, 0)
            obs_parts.append(op_state.to_vector(
                self.state.horizon, self.n_machines, machine_idx
            ))
        
        # Global features
        global_features = np.array([
            self.state.current_time / self.state.horizon
        ], dtype=np.float32)
        obs_parts.append(global_features)
        
        observation = np.concatenate(obs_parts)
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dict for current state."""
        return {
            'step': self.current_step,
            'completed': self.state.completed_count,
            'total_operations': self.n_operations,
            'makespan': self.state.makespan,
            'tardiness': self.state.total_tardiness,
            'setup_time': self.state.total_setup,
            'current_time': self.state.current_time,
            'action_mask': self.action_mask.copy(),
            'episode_reward': self.episode_reward,
        }
    
    def render(self) -> Optional[str]:
        """Render the environment state."""
        if self.render_mode == "ansi" or self.render_mode == "human":
            lines = [
                "=" * 60,
                f"Step: {self.current_step} | Time: {self.state.current_time:.1f}min",
                f"Completed: {self.state.completed_count}/{self.n_operations}",
                f"Makespan: {self.state.makespan:.1f}min | Tardiness: {self.state.total_tardiness:.1f}min",
                "-" * 60,
                "Machines:",
            ]
            for m_id, m_state in self.state.machines.items():
                lines.append(
                    f"  {m_id}: available={m_state.available_time:.1f}, "
                    f"load={m_state.total_load:.1f}"
                )
            lines.append("-" * 60)
            lines.append(f"Ready operations: {int(np.sum(self.action_mask))}")
            lines.append("=" * 60)
            
            output = "\n".join(lines)
            if self.render_mode == "human":
                print(output)
            return output
        return None
    
    def close(self) -> None:
        """Clean up environment resources."""
        pass
    
    def get_action_mask(self) -> np.ndarray:
        """Get current action mask for policy networks that support masking."""
        return self.action_mask.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_env_from_databundle(
    data: Any,  # DataBundle
    config: Optional[SchedulingEnvConfig] = None,
) -> SchedulingEnv:
    """
    Create a SchedulingEnv from a DataBundle.
    
    Args:
        data: DataBundle with orders, routing, machines DataFrames
        config: Environment configuration
    
    Returns:
        Configured SchedulingEnv instance
    """
    # Build operations DataFrame from routing and orders
    routing_df = data.routing
    orders_df = data.orders
    machines_df = data.machines
    
    # Merge to get operations with order info
    ops_df = routing_df.merge(
        orders_df[['order_id', 'article_id', 'qty', 'due_date']],
        on='article_id',
        how='left'
    )
    
    # Create unique operation IDs
    ops_df['op_id'] = ops_df.apply(
        lambda r: f"{r['order_id']}-{r.get('op_seq', 0)}", axis=1
    )
    
    # Calculate processing time
    if 'base_time_per_unit_min' in ops_df.columns:
        ops_df['processing_time'] = ops_df['base_time_per_unit_min'] * ops_df['qty']
    elif 'duration_min' in ops_df.columns:
        ops_df['processing_time'] = ops_df['duration_min']
    else:
        ops_df['processing_time'] = 60.0
    
    # Get machine ID
    if 'primary_machine_id' in ops_df.columns:
        ops_df['machine_id'] = ops_df['primary_machine_id']
    
    return SchedulingEnv(ops_df, machines_df, config)



