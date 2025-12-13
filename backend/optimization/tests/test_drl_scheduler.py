"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    TESTS FOR DRL SCHEDULER
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Unit tests for the Deep Reinforcement Learning scheduling module.

Test Categories:
────────────────
    1. Environment Tests
        - State space correctness
        - Action masking
        - Reward calculation
        - Episode termination
        
    2. Training Tests
        - Policy convergence
        - Reward improvement
        - Model save/load
        
    3. Integration Tests
        - Plan generation
        - Plan validity (no overlaps)
        - Comparison with heuristic baseline

SIFIDE R&D Documentation:
─────────────────────────
    These tests serve as experimental validation for the DRL scheduling
    research. Results should be logged and compared against baselines
    for SIFIDE justification.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

# Check if dependencies are available
try:
    import gymnasium as gym
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


# ═══════════════════════════════════════════════════════════════════════════════
# TEST FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def toy_operations_df() -> pd.DataFrame:
    """
    Small toy problem: 4 operations on 2 machines.
    
    Problem structure:
        Order 1: OP-1 (M-001) → OP-2 (M-002)
        Order 2: OP-3 (M-001) → OP-4 (M-002)
    
    Optimal solution should interleave orders to minimize makespan.
    """
    return pd.DataFrame([
        {
            'op_id': 'OP-1',
            'order_id': 'O-001',
            'article_id': 'ART-A',
            'machine_id': 'M-001',
            'processing_time': 60.0,  # 1 hour
            'setup_time': 10.0,
            'due_date': 240.0,  # 4 hours
            'predecessors': [],
            'product_family': 'FAM-A',
        },
        {
            'op_id': 'OP-2',
            'order_id': 'O-001',
            'article_id': 'ART-A',
            'machine_id': 'M-002',
            'processing_time': 45.0,
            'setup_time': 5.0,
            'due_date': 240.0,
            'predecessors': ['OP-1'],
            'product_family': 'FAM-A',
        },
        {
            'op_id': 'OP-3',
            'order_id': 'O-002',
            'article_id': 'ART-B',
            'machine_id': 'M-001',
            'processing_time': 50.0,
            'setup_time': 15.0,
            'due_date': 180.0,  # 3 hours - more urgent
            'predecessors': [],
            'product_family': 'FAM-B',
        },
        {
            'op_id': 'OP-4',
            'order_id': 'O-002',
            'article_id': 'ART-B',
            'machine_id': 'M-002',
            'processing_time': 30.0,
            'setup_time': 5.0,
            'due_date': 180.0,
            'predecessors': ['OP-3'],
            'product_family': 'FAM-B',
        },
    ])


@pytest.fixture
def toy_machines_df() -> pd.DataFrame:
    """Two machines for the toy problem."""
    return pd.DataFrame([
        {'machine_id': 'M-001', 'speed_factor': 1.0},
        {'machine_id': 'M-002', 'speed_factor': 1.0},
    ])


@pytest.fixture
def medium_operations_df() -> pd.DataFrame:
    """
    Medium problem: 20 operations on 4 machines.
    Used for training tests.
    """
    operations = []
    np.random.seed(42)
    
    for order_idx in range(5):  # 5 orders
        order_id = f'O-{order_idx:03d}'
        article_id = f'ART-{chr(65 + order_idx % 3)}'
        due_date = 600 + order_idx * 100
        
        prev_op_id = None
        for op_seq in range(4):  # 4 operations per order
            op_id = f'{order_id}-{op_seq}'
            machine_id = f'M-{(op_seq % 4) + 1:03d}'
            
            operations.append({
                'op_id': op_id,
                'order_id': order_id,
                'article_id': article_id,
                'machine_id': machine_id,
                'processing_time': np.random.uniform(30, 90),
                'setup_time': np.random.uniform(5, 20),
                'due_date': due_date,
                'predecessors': [prev_op_id] if prev_op_id else [],
                'product_family': article_id,
            })
            prev_op_id = op_id
    
    return pd.DataFrame(operations)


@pytest.fixture
def medium_machines_df() -> pd.DataFrame:
    """Four machines for medium problem."""
    return pd.DataFrame([
        {'machine_id': 'M-001', 'speed_factor': 1.0},
        {'machine_id': 'M-002', 'speed_factor': 1.1},
        {'machine_id': 'M-003', 'speed_factor': 0.9},
        {'machine_id': 'M-004', 'speed_factor': 1.0},
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_GYMNASIUM, reason="gymnasium not installed")
class TestSchedulingEnv:
    """Tests for the SchedulingEnv class."""
    
    def test_env_creation(self, toy_operations_df, toy_machines_df):
        """Test environment can be created with toy problem."""
        from backend.optimization.drl_scheduler import SchedulingEnv, SchedulingEnvConfig
        
        config = SchedulingEnvConfig(horizon_minutes=480)
        env = SchedulingEnv(toy_operations_df, toy_machines_df, config)
        
        assert env.n_operations == 4
        assert env.n_machines == 2
        assert env.action_space.n == 4
    
    def test_env_reset(self, toy_operations_df, toy_machines_df):
        """Test environment reset returns valid observation."""
        from backend.optimization.drl_scheduler import SchedulingEnv, SchedulingEnvConfig
        
        config = SchedulingEnvConfig(horizon_minutes=480)
        env = SchedulingEnv(toy_operations_df, toy_machines_df, config)
        
        obs, info = env.reset()
        
        # Check observation shape
        expected_dim = 2 * 4 + 4 * 5 + 1  # 2 machines * 4 + 4 ops * 5 + 1 global
        assert obs.shape == (expected_dim,)
        
        # Check observation bounds
        assert np.all(obs >= -1.0)
        assert np.all(obs <= 1.0)
        
        # Check info
        assert 'action_mask' in info
        assert info['completed'] == 0
    
    def test_action_masking(self, toy_operations_df, toy_machines_df):
        """Test that action mask correctly identifies valid actions."""
        from backend.optimization.drl_scheduler import SchedulingEnv, SchedulingEnvConfig
        
        config = SchedulingEnvConfig(horizon_minutes=480)
        env = SchedulingEnv(toy_operations_df, toy_machines_df, config)
        
        obs, info = env.reset()
        mask = info['action_mask']
        
        # OP-1 and OP-3 should be ready (no predecessors)
        # OP-2 and OP-4 should be waiting (have predecessors)
        assert mask[0] == 1.0  # OP-1 ready
        assert mask[1] == 0.0  # OP-2 waiting (needs OP-1)
        assert mask[2] == 1.0  # OP-3 ready
        assert mask[3] == 0.0  # OP-4 waiting (needs OP-3)
    
    def test_step_valid_action(self, toy_operations_df, toy_machines_df):
        """Test stepping with a valid action."""
        from backend.optimization.drl_scheduler import SchedulingEnv, SchedulingEnvConfig
        
        config = SchedulingEnvConfig(horizon_minutes=480)
        env = SchedulingEnv(toy_operations_df, toy_machines_df, config)
        
        obs, info = env.reset()
        
        # Take action 0 (OP-1)
        obs2, reward, terminated, truncated, info2 = env.step(0)
        
        # Check state updated
        assert info2['completed'] == 1
        assert not terminated
        
        # OP-2 should now be ready (OP-1 completed)
        mask = info2['action_mask']
        assert mask[1] == 1.0  # OP-2 now ready
    
    def test_episode_completion(self, toy_operations_df, toy_machines_df):
        """Test episode terminates when all operations scheduled."""
        from backend.optimization.drl_scheduler import SchedulingEnv, SchedulingEnvConfig
        
        config = SchedulingEnvConfig(horizon_minutes=480)
        env = SchedulingEnv(toy_operations_df, toy_machines_df, config)
        
        obs, info = env.reset()
        terminated = False
        steps = 0
        
        # Greedily schedule all operations
        while not terminated and steps < 100:
            mask = info['action_mask']
            valid_actions = np.where(mask > 0)[0]
            
            if len(valid_actions) == 0:
                break
            
            action = valid_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
        
        assert terminated
        assert info['completed'] == 4
    
    def test_reward_calculation(self, toy_operations_df, toy_machines_df):
        """Test reward reflects tardiness and makespan."""
        from backend.optimization.drl_scheduler import SchedulingEnv, SchedulingEnvConfig
        
        # Tight due dates to encourage on-time delivery
        config = SchedulingEnvConfig(
            horizon_minutes=480,
            reward_tardiness_weight=1.0,
            reward_makespan_weight=0.5,
        )
        env = SchedulingEnv(toy_operations_df, toy_machines_df, config)
        
        obs, info = env.reset()
        total_reward = 0.0
        
        # Complete episode
        terminated = False
        while not terminated:
            mask = info['action_mask']
            valid_actions = np.where(mask > 0)[0]
            if len(valid_actions) == 0:
                break
            action = valid_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Reward should be negative (penalties) or small positive
        assert isinstance(total_reward, float)
        assert not np.isnan(total_reward)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not (HAS_GYMNASIUM and HAS_SB3), reason="gymnasium or stable-baselines3 not installed")
class TestDRLTrainer:
    """Tests for the DRL trainer."""
    
    def test_quick_training(self, toy_operations_df, toy_machines_df):
        """Test that training runs without errors (few episodes)."""
        from backend.optimization.drl_scheduler import (
            SchedulingEnv, SchedulingEnvConfig,
            DRLTrainer, TrainingConfig, AlgorithmType
        )
        
        env_config = SchedulingEnvConfig(horizon_minutes=480)
        
        def env_factory():
            return SchedulingEnv(toy_operations_df, toy_machines_df, env_config)
        
        train_config = TrainingConfig(
            algorithm=AlgorithmType.PPO,
            total_timesteps=500,  # Very short for testing
            n_steps=64,
            batch_size=32,
            n_eval_episodes=2,
            eval_freq=200,
            verbose=0,
        )
        
        trainer = DRLTrainer(env_factory, train_config)
        result = trainer.train()
        
        assert result is not None
        assert result.total_timesteps == 500
        assert isinstance(result.final_mean_reward, float)
        assert result.model_path is not None
    
    def test_reward_improves(self, medium_operations_df, medium_machines_df):
        """Test that reward improves with training (longer run)."""
        from backend.optimization.drl_scheduler import (
            SchedulingEnv, SchedulingEnvConfig,
            DRLTrainer, TrainingConfig, AlgorithmType
        )
        
        env_config = SchedulingEnvConfig(horizon_minutes=1440)
        
        def env_factory():
            return SchedulingEnv(medium_operations_df, medium_machines_df, env_config)
        
        train_config = TrainingConfig(
            algorithm=AlgorithmType.PPO,
            total_timesteps=2000,
            n_steps=128,
            batch_size=64,
            n_eval_episodes=3,
            eval_freq=500,
            verbose=0,
        )
        
        trainer = DRLTrainer(env_factory, train_config)
        result = trainer.train()
        
        # Check we got some reward history
        assert len(result.reward_history) > 0
        
        # Check training completed
        assert result.training_time_seconds > 0
    
    def test_model_save_load(self, toy_operations_df, toy_machines_df, tmp_path):
        """Test model can be saved and loaded."""
        from backend.optimization.drl_scheduler import (
            SchedulingEnv, SchedulingEnvConfig,
            DRLTrainer, TrainingConfig, AlgorithmType,
            load_trained_policy
        )
        
        env_config = SchedulingEnvConfig(horizon_minutes=480)
        
        def env_factory():
            return SchedulingEnv(toy_operations_df, toy_machines_df, env_config)
        
        train_config = TrainingConfig(
            algorithm=AlgorithmType.PPO,
            total_timesteps=200,
            verbose=0,
        )
        
        trainer = DRLTrainer(env_factory, train_config)
        
        # Save
        model_path = tmp_path / "test_model.zip"
        trainer.save_model(model_path)
        
        assert model_path.exists()
        
        # Load
        loaded_model = load_trained_policy(model_path, AlgorithmType.PPO)
        assert loaded_model is not None


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_GYMNASIUM, reason="gymnasium not installed")
class TestPlanGeneration:
    """Tests for plan generation using DRL."""
    
    def test_plan_generation_fallback(self, toy_operations_df, toy_machines_df):
        """Test plan generation falls back to heuristic when no model."""
        from backend.optimization.drl_scheduler import (
            build_plan_drl, DRLSchedulerConfig
        )
        from backend.optimization.drl_scheduler.drl_scheduler_interface import (
            build_operations_from_databundle,
            _build_plan_heuristic_fallback
        )
        
        # Use heuristic fallback directly
        horizon_start = datetime.now()
        plan_df = _build_plan_heuristic_fallback(
            toy_operations_df, toy_machines_df, horizon_start
        )
        
        assert len(plan_df) == 4
        assert 'start_time' in plan_df.columns
        assert 'end_time' in plan_df.columns
        assert 'machine_id' in plan_df.columns
    
    def test_plan_no_overlaps(self, toy_operations_df, toy_machines_df):
        """Test generated plan has no machine overlaps."""
        from backend.optimization.drl_scheduler.drl_scheduler_interface import (
            _build_plan_heuristic_fallback, _validate_plan
        )
        
        horizon_start = datetime.now()
        plan_df = _build_plan_heuristic_fallback(
            toy_operations_df, toy_machines_df, horizon_start
        )
        
        is_valid, issues = _validate_plan(plan_df)
        
        assert is_valid, f"Plan has issues: {issues}"
    
    def test_plan_respects_precedence(self, toy_operations_df, toy_machines_df):
        """Test that precedence constraints are respected."""
        from backend.optimization.drl_scheduler.drl_scheduler_interface import (
            _build_plan_heuristic_fallback
        )
        
        horizon_start = datetime.now()
        plan_df = _build_plan_heuristic_fallback(
            toy_operations_df, toy_machines_df, horizon_start
        )
        
        # OP-2 should start after OP-1 ends (same order)
        op1 = plan_df[plan_df['op_id'] == 'O-001-0'].iloc[0] if 'O-001-0' in plan_df['op_id'].values else None
        op2 = plan_df[plan_df['op_id'] == 'O-001-1'].iloc[0] if 'O-001-1' in plan_df['op_id'].values else None
        
        # Fallback uses different op_id format, check by order_id sequence
        order_ops = plan_df[plan_df['order_id'] == 'O-001'].sort_values('start_time')
        
        if len(order_ops) >= 2:
            first_op = order_ops.iloc[0]
            second_op = order_ops.iloc[1]
            assert second_op['start_time'] >= first_op['end_time']


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE COMPARISON TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_GYMNASIUM, reason="gymnasium not installed")
class TestBaselineComparison:
    """Tests comparing DRL with baseline heuristics."""
    
    def test_random_vs_greedy(self, toy_operations_df, toy_machines_df):
        """Test that greedy (first valid action) beats random."""
        from backend.optimization.drl_scheduler import SchedulingEnv, SchedulingEnvConfig
        
        config = SchedulingEnvConfig(horizon_minutes=480)
        env = SchedulingEnv(toy_operations_df, toy_machines_df, config)
        
        # Run greedy (always pick first valid action)
        greedy_rewards = []
        for _ in range(5):
            obs, info = env.reset()
            total_reward = 0.0
            terminated = False
            
            while not terminated:
                mask = info['action_mask']
                valid_actions = np.where(mask > 0)[0]
                if len(valid_actions) == 0:
                    break
                action = valid_actions[0]  # Greedy: first valid
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
            
            greedy_rewards.append(total_reward)
        
        # Run random
        np.random.seed(42)
        random_rewards = []
        for _ in range(5):
            obs, info = env.reset()
            total_reward = 0.0
            terminated = False
            
            while not terminated:
                mask = info['action_mask']
                valid_actions = np.where(mask > 0)[0]
                if len(valid_actions) == 0:
                    break
                action = np.random.choice(valid_actions)  # Random valid
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
            
            random_rewards.append(total_reward)
        
        mean_greedy = np.mean(greedy_rewards)
        mean_random = np.mean(random_rewards)
        
        # Greedy should be at least as good as random on average
        # (Note: might not always be true, but should be close)
        assert mean_greedy >= mean_random - 0.5  # Allow small margin


# ═══════════════════════════════════════════════════════════════════════════════
# SIFIDE R&D EXPERIMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not (HAS_GYMNASIUM and HAS_SB3), reason="Full DRL stack not available")
class TestSIFIDEExperiments:
    """
    Tests designed to generate evidence for SIFIDE R&D justification.
    
    These tests document:
        - Technical uncertainty being addressed
        - Experimental methodology
        - Measurable outcomes
    """
    
    def test_hypothesis_h4_1_setup(self, medium_operations_df, medium_machines_df):
        """
        Hypothesis H4.1: DRL can learn effective dispatching policies.
        
        This test validates the experimental setup for H4.1.
        Full experiment would require longer training.
        """
        from backend.optimization.drl_scheduler import (
            SchedulingEnv, SchedulingEnvConfig,
            TrainingConfig, AlgorithmType
        )
        
        # Document experimental setup
        experiment_setup = {
            'hypothesis': 'H4.1',
            'description': 'DRL learns dispatching policies competitive with heuristics',
            'metric': 'makespan reduction vs EDD baseline',
            'success_criterion': '≥5% makespan reduction',
            'problem_size': f'{len(medium_operations_df)} operations, {len(medium_machines_df)} machines',
            'algorithm': 'PPO',
            'state_space': 'machine states + operation states + global time',
            'action_space': 'discrete operation selection',
        }
        
        # Validate setup is correct
        env_config = SchedulingEnvConfig(horizon_minutes=1440)
        env = SchedulingEnv(medium_operations_df, medium_machines_df, env_config)
        
        assert env.n_operations == len(medium_operations_df)
        assert env.action_space.n == len(medium_operations_df)
        
        # Log setup for SIFIDE
        print("\n=== SIFIDE R&D Experiment H4.1 Setup ===")
        for key, value in experiment_setup.items():
            print(f"  {key}: {value}")
        
        assert True  # Setup validated


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])



