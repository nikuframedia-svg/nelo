"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    DRL TRAINER (Stable-Baselines3)
═══════════════════════════════════════════════════════════════════════════════════════════════════════

This module provides training functionality for Deep Reinforcement Learning policies
using Stable-Baselines3 algorithms.

Supported Algorithms:
────────────────────
    PPO (Proximal Policy Optimization):
        - Default choice for complex environments
        - Good sample efficiency and stability
        - Supports both continuous and discrete action spaces
        
    A2C (Advantage Actor-Critic):
        - Faster training, lower sample efficiency
        - Good for simpler environments
        
    DQN (Deep Q-Network):
        - Value-based method for discrete actions
        - Good for environments with clear optimal actions

Training Pipeline:
──────────────────
    1. Environment Creation
        └─ SchedulingEnv with configuration
    2. Algorithm Selection
        └─ PPO/A2C/DQN with hyperparameters
    3. Training Loop
        └─ Episodes with reward logging
    4. Evaluation
        └─ Compare against baseline heuristics
    5. Model Saving
        └─ .zip file for deployment

Experiment Logging (SIFIDE R&D):
────────────────────────────────
    All training runs are logged in JSON format with:
    - Configuration parameters
    - Reward curves
    - Regret vs baseline
    - Convergence metrics
    - KPI comparisons (makespan, tardiness, etc.)

TODO[R&D]: Advanced training features:
    - Curriculum learning with progressive difficulty
    - Multi-objective reward shaping
    - Imitation learning from heuristic demonstrations
    - Meta-learning for fast adaptation to new scenarios
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.evaluation import evaluate_policy as sb3_evaluate
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    PPO = A2C = DQN = None

from env_scheduling import SchedulingEnv, SchedulingEnvConfig

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_MODEL_DIR = Path(__file__).parent / "trained_models"
DEFAULT_LOG_DIR = Path(__file__).parent / "training_logs"


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class AlgorithmType(str, Enum):
    """Supported DRL algorithms."""
    PPO = "PPO"
    A2C = "A2C"
    DQN = "DQN"


@dataclass
class TrainingConfig:
    """
    Configuration for DRL training.
    
    Hyperparameters:
    ───────────────
        learning_rate: Step size for gradient updates
            - PPO default: 3e-4
            - A2C default: 7e-4
            - DQN default: 1e-4
            
        n_steps: Steps to collect before update (PPO/A2C)
            - Higher = more stable but slower
            - PPO default: 2048
            
        batch_size: Minibatch size for PPO/DQN
            - PPO default: 64
            
        gamma: Discount factor for future rewards
            - 0.99 = long-horizon planning
            - 0.9 = short-horizon, immediate rewards
            
        gae_lambda: GAE lambda for advantage estimation (PPO/A2C)
            - 1.0 = no bias, high variance
            - 0.0 = high bias, low variance
            - 0.95 = good default
    
    Training Parameters:
    ───────────────────
        total_timesteps: Total environment steps for training
        eval_freq: Steps between evaluations
        n_eval_episodes: Episodes for evaluation
        
    Network Architecture:
    ────────────────────
        policy_type: "MlpPolicy" for fully connected
        net_arch: Hidden layer sizes, e.g., [256, 256]
    """
    # Algorithm
    algorithm: AlgorithmType = AlgorithmType.PPO
    
    # Hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2  # PPO only
    ent_coef: float = 0.01   # Entropy coefficient for exploration
    vf_coef: float = 0.5     # Value function coefficient
    max_grad_norm: float = 0.5
    
    # Training
    total_timesteps: int = 100_000
    eval_freq: int = 5000
    n_eval_episodes: int = 10
    save_freq: int = 10000
    
    # Network
    policy_type: str = "MlpPolicy"
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    
    # Environment
    n_envs: int = 1  # Number of parallel environments
    
    # Logging
    verbose: int = 1
    tensorboard_log: Optional[str] = None
    
    # Experiment
    experiment_name: str = "drl_scheduling"
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d['algorithm'] = self.algorithm.value
        return d


@dataclass
class TrainingResult:
    """
    Results from a training run.
    
    Includes metrics for SIFIDE R&D documentation.
    """
    experiment_id: str
    algorithm: str
    total_timesteps: int
    training_time_seconds: float
    
    # Reward metrics
    final_mean_reward: float
    final_std_reward: float
    reward_history: List[float] = field(default_factory=list)
    
    # KPI metrics
    final_makespan: float = 0.0
    final_tardiness: float = 0.0
    final_setup_time: float = 0.0
    
    # Comparison with baseline
    baseline_reward: float = 0.0
    improvement_pct: float = 0.0
    regret_curve: List[float] = field(default_factory=list)
    
    # Convergence
    convergence_step: Optional[int] = None
    is_converged: bool = False
    
    # Model path
    model_path: Optional[str] = None
    
    # Metadata
    config: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingLoggingCallback(BaseCallback if HAS_SB3 else object):
    """
    Callback for logging training metrics.
    
    Logs:
        - Episode rewards
        - Episode lengths
        - KPIs (makespan, tardiness, etc.)
        - Regret vs baseline
    """
    
    def __init__(
        self,
        log_freq: int = 1000,
        baseline_reward: float = 0.0,
        verbose: int = 0,
    ):
        if HAS_SB3:
            super().__init__(verbose)
        self.log_freq = log_freq
        self.baseline_reward = baseline_reward
        
        # Metrics storage
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_makespans: List[float] = []
        self.episode_tardiness: List[float] = []
        self.regret_values: List[float] = []
        
        self._episode_reward = 0.0
        self._episode_length = 0
        self._last_makespan = 0.0
        self._last_tardiness = 0.0
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # Accumulate episode metrics
        if len(self.locals.get('rewards', [])) > 0:
            self._episode_reward += self.locals['rewards'][0]
            self._episode_length += 1
        
        # Check for episode end
        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            self.episode_rewards.append(self._episode_reward)
            self.episode_lengths.append(self._episode_length)
            
            # Get KPIs from info
            infos = self.locals.get('infos', [{}])
            if infos and len(infos) > 0:
                info = infos[0]
                self._last_makespan = info.get('makespan', 0.0)
                self._last_tardiness = info.get('tardiness', 0.0)
                self.episode_makespans.append(self._last_makespan)
                self.episode_tardiness.append(self._last_tardiness)
            
            # Calculate regret
            regret = self.baseline_reward - self._episode_reward
            self.regret_values.append(regret)
            
            # Reset for next episode
            self._episode_reward = 0.0
            self._episode_length = 0
        
        # Periodic logging
        if self.n_calls % self.log_freq == 0 and self.verbose > 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_regret = np.mean(self.regret_values[-100:]) if self.regret_values else 0
                logger.info(
                    f"Step {self.n_calls}: mean_reward={mean_reward:.2f}, "
                    f"regret={mean_regret:.2f}, episodes={len(self.episode_rewards)}"
                )
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_makespans': self.episode_makespans,
            'episode_tardiness': self.episode_tardiness,
            'regret_values': self.regret_values,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class DRLTrainer:
    """
    Trainer for DRL scheduling policies.
    
    Example usage:
        >>> env = SchedulingEnv(operations_df, machines_df)
        >>> config = TrainingConfig(algorithm=AlgorithmType.PPO)
        >>> trainer = DRLTrainer(env, config)
        >>> result = trainer.train()
        >>> trainer.save_model("models/scheduling_ppo.zip")
    
    TODO[R&D]: Advanced training features:
        - Distributed training with Ray
        - Population-based training for hyperparameter search
        - Offline RL from historical scheduling data
    """
    
    def __init__(
        self,
        env: Union[SchedulingEnv, Callable[[], SchedulingEnv]],
        config: Optional[TrainingConfig] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            env: Environment instance or factory function
            config: Training configuration
        """
        if not HAS_SB3:
            raise ImportError(
                "stable-baselines3 is required for DRL training. "
                "Install with: pip install stable-baselines3"
            )
        
        self.config = config or TrainingConfig()
        self.env_factory = env if callable(env) else lambda: env
        
        # Create environments
        self.env = self._create_env()
        self.eval_env = self._create_env()
        
        # Create model
        self.model = self._create_model()
        
        # Training state
        self.is_trained = False
        self.training_result: Optional[TrainingResult] = None
        
        logger.info(
            f"DRLTrainer initialized: algorithm={self.config.algorithm.value}, "
            f"timesteps={self.config.total_timesteps}"
        )
    
    def _create_env(self) -> Union[DummyVecEnv, SubprocVecEnv]:
        """Create vectorized environment."""
        if self.config.n_envs > 1:
            # Parallel environments
            env = SubprocVecEnv([self.env_factory for _ in range(self.config.n_envs)])
        else:
            # Single environment
            env = DummyVecEnv([lambda: Monitor(self.env_factory())])
        return env
    
    def _create_model(self):
        """Create DRL model based on configuration."""
        # Select algorithm
        algo_map = {
            AlgorithmType.PPO: PPO,
            AlgorithmType.A2C: A2C,
            AlgorithmType.DQN: DQN,
        }
        
        AlgoClass = algo_map[self.config.algorithm]
        
        # Build policy kwargs
        policy_kwargs = {
            'net_arch': self.config.net_arch,
        }
        
        # Algorithm-specific parameters
        if self.config.algorithm == AlgorithmType.PPO:
            model = AlgoClass(
                policy=self.config.policy_type,
                env=self.env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                clip_range=self.config.clip_range,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                policy_kwargs=policy_kwargs,
                verbose=self.config.verbose,
                tensorboard_log=self.config.tensorboard_log,
                seed=self.config.seed,
            )
        elif self.config.algorithm == AlgorithmType.A2C:
            model = AlgoClass(
                policy=self.config.policy_type,
                env=self.env,
                learning_rate=self.config.learning_rate,
                n_steps=min(self.config.n_steps, 5),  # A2C uses smaller n_steps
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                policy_kwargs=policy_kwargs,
                verbose=self.config.verbose,
                tensorboard_log=self.config.tensorboard_log,
                seed=self.config.seed,
            )
        elif self.config.algorithm == AlgorithmType.DQN:
            model = AlgoClass(
                policy=self.config.policy_type,
                env=self.env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                policy_kwargs=policy_kwargs,
                verbose=self.config.verbose,
                tensorboard_log=self.config.tensorboard_log,
                seed=self.config.seed,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
        
        return model
    
    def train(
        self,
        baseline_reward: float = 0.0,
        progress_callback: Optional[Callable[[int, Dict], None]] = None,
    ) -> TrainingResult:
        """
        Train the DRL policy.
        
        Args:
            baseline_reward: Baseline reward for regret calculation
            progress_callback: Optional callback(step, metrics) for progress updates
        
        Returns:
            TrainingResult with metrics and model path
        """
        logger.info("Starting DRL training...")
        start_time = time.time()
        
        # Create callbacks
        logging_callback = TrainingLoggingCallback(
            log_freq=1000,
            baseline_reward=baseline_reward,
            verbose=self.config.verbose,
        )
        
        callbacks = [logging_callback]
        
        # Add evaluation callback
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=str(DEFAULT_MODEL_DIR / self.config.experiment_name),
            log_path=str(DEFAULT_LOG_DIR / self.config.experiment_name),
            eval_freq=self.config.eval_freq,
            n_eval_episodes=self.config.n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=self.config.verbose,
        )
        callbacks.append(eval_callback)
        
        # Train
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callbacks,
            progress_bar=self.config.verbose > 0,
        )
        
        training_time = time.time() - start_time
        
        # Final evaluation
        mean_reward, std_reward = self.evaluate(n_episodes=self.config.n_eval_episodes)
        
        # Get training metrics
        metrics = logging_callback.get_metrics()
        
        # Calculate improvement
        improvement = ((mean_reward - baseline_reward) / abs(baseline_reward) * 100
                      if baseline_reward != 0 else 0.0)
        
        # Check convergence (reward stabilized)
        convergence_step = self._detect_convergence(metrics['episode_rewards'])
        
        # Build result
        experiment_id = f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = str(DEFAULT_MODEL_DIR / f"{experiment_id}.zip")
        
        # Save model
        self.save_model(model_path)
        
        self.training_result = TrainingResult(
            experiment_id=experiment_id,
            algorithm=self.config.algorithm.value,
            total_timesteps=self.config.total_timesteps,
            training_time_seconds=training_time,
            final_mean_reward=mean_reward,
            final_std_reward=std_reward,
            reward_history=metrics['episode_rewards'],
            final_makespan=np.mean(metrics['episode_makespans'][-10:]) if metrics['episode_makespans'] else 0,
            final_tardiness=np.mean(metrics['episode_tardiness'][-10:]) if metrics['episode_tardiness'] else 0,
            baseline_reward=baseline_reward,
            improvement_pct=improvement,
            regret_curve=metrics['regret_values'],
            convergence_step=convergence_step,
            is_converged=convergence_step is not None,
            model_path=model_path,
            config=self.config.to_dict(),
        )
        
        # Save results log
        log_path = DEFAULT_LOG_DIR / f"{experiment_id}.json"
        self.training_result.save(log_path)
        
        self.is_trained = True
        
        logger.info(
            f"Training complete: reward={mean_reward:.2f}±{std_reward:.2f}, "
            f"improvement={improvement:.1f}%, time={training_time:.1f}s"
        )
        
        return self.training_result
    
    def evaluate(self, n_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate the current policy.
        
        Args:
            n_episodes: Number of evaluation episodes
        
        Returns:
            (mean_reward, std_reward)
        """
        mean_reward, std_reward = sb3_evaluate(
            self.model,
            self.eval_env,
            n_eval_episodes=n_episodes,
            deterministic=True,
        )
        return mean_reward, std_reward
    
    def _detect_convergence(
        self,
        rewards: List[float],
        window: int = 100,
        threshold: float = 0.01,
    ) -> Optional[int]:
        """
        Detect when training converged.
        
        Convergence is detected when the standard deviation of rewards
        within a rolling window falls below a threshold.
        
        Args:
            rewards: List of episode rewards
            window: Rolling window size
            threshold: Convergence threshold (relative to mean)
        
        Returns:
            Step number where convergence was detected, or None
        """
        if len(rewards) < window * 2:
            return None
        
        for i in range(window, len(rewards)):
            window_rewards = rewards[i-window:i]
            mean_r = np.mean(window_rewards)
            std_r = np.std(window_rewards)
            
            if mean_r != 0 and std_r / abs(mean_r) < threshold:
                return i
        
        return None
    
    def save_model(self, path: Union[str, Path]) -> None:
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Union[str, Path]) -> None:
        """Load model from file."""
        algo_map = {
            AlgorithmType.PPO: PPO,
            AlgorithmType.A2C: A2C,
            AlgorithmType.DQN: DQN,
        }
        AlgoClass = algo_map[self.config.algorithm]
        self.model = AlgoClass.load(str(path), env=self.env)
        self.is_trained = True
        logger.info(f"Model loaded from {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def train_policy(
    operations_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    config: Optional[TrainingConfig] = None,
    env_config: Optional[SchedulingEnvConfig] = None,
    baseline_reward: float = 0.0,
) -> TrainingResult:
    """
    Train a DRL scheduling policy.
    
    High-level function for training without manually creating trainer.
    
    Args:
        operations_df: Operations DataFrame
        machines_df: Machines DataFrame
        config: Training configuration
        env_config: Environment configuration
        baseline_reward: Baseline reward for comparison
    
    Returns:
        TrainingResult with metrics and model path
    
    Example:
        >>> result = train_policy(
        ...     operations_df, machines_df,
        ...     config=TrainingConfig(total_timesteps=50000),
        ... )
        >>> print(f"Final reward: {result.final_mean_reward}")
    """
    config = config or TrainingConfig()
    env_config = env_config or SchedulingEnvConfig()
    
    def env_factory():
        return SchedulingEnv(operations_df, machines_df, env_config)
    
    trainer = DRLTrainer(env_factory, config)
    return trainer.train(baseline_reward=baseline_reward)


def evaluate_policy(
    model_path: Union[str, Path],
    operations_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    n_episodes: int = 10,
    env_config: Optional[SchedulingEnvConfig] = None,
    algorithm: AlgorithmType = AlgorithmType.PPO,
) -> Dict[str, float]:
    """
    Evaluate a trained policy.
    
    Args:
        model_path: Path to saved model
        operations_df: Operations DataFrame
        machines_df: Machines DataFrame
        n_episodes: Number of evaluation episodes
        env_config: Environment configuration
        algorithm: Algorithm type of the saved model
    
    Returns:
        Dictionary with KPIs:
            - mean_reward
            - std_reward
            - mean_makespan
            - mean_tardiness
            - mean_setup_time
    """
    if not HAS_SB3:
        raise ImportError("stable-baselines3 is required")
    
    env_config = env_config or SchedulingEnvConfig()
    env = SchedulingEnv(operations_df, machines_df, env_config)
    
    # Load model
    algo_map = {
        AlgorithmType.PPO: PPO,
        AlgorithmType.A2C: A2C,
        AlgorithmType.DQN: DQN,
    }
    model = algo_map[algorithm].load(str(model_path))
    
    # Evaluate
    episode_rewards = []
    episode_makespans = []
    episode_tardiness = []
    episode_setups = []
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_makespans.append(info.get('makespan', 0))
        episode_tardiness.append(info.get('tardiness', 0))
        episode_setups.append(info.get('setup_time', 0))
    
    return {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_makespan': float(np.mean(episode_makespans)),
        'mean_tardiness': float(np.mean(episode_tardiness)),
        'mean_setup_time': float(np.mean(episode_setups)),
        'n_episodes': n_episodes,
    }


def compare_with_baseline(
    model_path: Union[str, Path],
    operations_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    baseline_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
    n_episodes: int = 10,
    algorithm: AlgorithmType = AlgorithmType.PPO,
) -> Dict[str, Any]:
    """
    Compare DRL policy with a baseline heuristic.
    
    Args:
        model_path: Path to trained model
        operations_df: Operations DataFrame
        machines_df: Machines DataFrame
        baseline_fn: Function that returns baseline reward/KPI
        n_episodes: Evaluation episodes
        algorithm: Algorithm type
    
    Returns:
        Comparison metrics including improvement percentage
    """
    # Get DRL metrics
    drl_metrics = evaluate_policy(
        model_path, operations_df, machines_df,
        n_episodes=n_episodes, algorithm=algorithm
    )
    
    # Get baseline metrics
    baseline_reward = baseline_fn(operations_df, machines_df)
    
    # Calculate improvement
    improvement = (
        (drl_metrics['mean_reward'] - baseline_reward) / abs(baseline_reward) * 100
        if baseline_reward != 0 else 0.0
    )
    
    return {
        'drl_metrics': drl_metrics,
        'baseline_reward': baseline_reward,
        'improvement_pct': improvement,
        'drl_is_better': drl_metrics['mean_reward'] > baseline_reward,
    }



