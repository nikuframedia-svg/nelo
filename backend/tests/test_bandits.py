"""
═══════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — Bandit Policy Tests
═══════════════════════════════════════════════════════════════════════════════

Tests for multi-armed bandit and contextual bandit policies.

Run with: python -m pytest backend/tests/test_bandits.py -v
"""

import numpy as np
import pytest

from backend.optimization.learning_scheduler import (
    PolicyConfig,
    PolicyType,
    PolicyMetrics,
    create_policy,
    FixedPriorityPolicy,
    ShortestQueuePolicy,
    LoadBalancedPolicy,
    EpsilonGreedyPolicy,
    UCBPolicy,
    ThompsonSamplingPolicy,
    ContextualBanditPolicy,
    LearningScheduler,
)


class TestPolicyCreation:
    """Test policy factory."""
    
    def test_create_epsilon_greedy(self):
        config = PolicyConfig(policy_type=PolicyType.EPSILON_GREEDY, epsilon=0.1)
        policy = create_policy(config)
        assert isinstance(policy, EpsilonGreedyPolicy)
        assert policy.epsilon == 0.1
    
    def test_create_ucb(self):
        config = PolicyConfig(policy_type=PolicyType.UCB, ucb_c=2.0)
        policy = create_policy(config)
        assert isinstance(policy, UCBPolicy)
        assert policy.c == 2.0
    
    def test_create_thompson(self):
        config = PolicyConfig(policy_type=PolicyType.THOMPSON)
        policy = create_policy(config)
        assert isinstance(policy, ThompsonSamplingPolicy)
    
    def test_create_fixed_priority(self):
        config = PolicyConfig(policy_type=PolicyType.FIXED_PRIORITY)
        policy = create_policy(config)
        assert isinstance(policy, FixedPriorityPolicy)
    
    def test_create_contextual(self):
        config = PolicyConfig(policy_type=PolicyType.CONTEXTUAL_BANDIT, context_dim=5)
        policy = create_policy(config)
        assert isinstance(policy, ContextualBanditPolicy)


class TestFixedHeuristics:
    """Test fixed heuristic policies."""
    
    def test_fixed_priority_selects_highest(self):
        config = PolicyConfig(policy_type=PolicyType.FIXED_PRIORITY)
        policy = create_policy(config)
        
        state = {'priorities': {'M1': 1, 'M2': 3, 'M3': 2}}
        action = policy.select_action(state, ['M1', 'M2', 'M3'])
        
        assert action == 'M1'  # Lowest priority value = highest priority
    
    def test_shortest_queue_selects_shortest(self):
        config = PolicyConfig(policy_type=PolicyType.SHORTEST_QUEUE)
        policy = create_policy(config)
        
        state = {'queue_lengths': {'M1': 5, 'M2': 2, 'M3': 8}}
        action = policy.select_action(state, ['M1', 'M2', 'M3'])
        
        assert action == 'M2'  # Shortest queue
    
    def test_load_balanced_selects_least_loaded(self):
        config = PolicyConfig(policy_type=PolicyType.LOAD_BALANCED)
        policy = create_policy(config)
        
        state = {'machine_loads': {'M1': 0.8, 'M2': 0.3, 'M3': 0.6}}
        action = policy.select_action(state, ['M1', 'M2', 'M3'])
        
        assert action == 'M2'  # Lowest load


class TestEpsilonGreedy:
    """Test epsilon-greedy policy."""
    
    def test_exploration_rate(self):
        """Test that epsilon controls exploration rate."""
        config = PolicyConfig(
            policy_type=PolicyType.EPSILON_GREEDY,
            epsilon=0.3,
            seed=42
        )
        policy = create_policy(config)
        
        state = {}
        actions = ['A', 'B', 'C']
        
        # Update to make 'A' clearly best
        for _ in range(20):
            policy.update(state, 'A', 1.0)
            policy.update(state, 'B', 0.0)
            policy.update(state, 'C', 0.0)
        
        # Run many selections
        selections = [policy.select_action(state, actions) for _ in range(1000)]
        
        # Should mostly select 'A' but sometimes explore
        a_rate = sum(1 for s in selections if s == 'A') / len(selections)
        assert 0.6 < a_rate < 0.9  # Mostly exploit but also explore
    
    def test_value_update(self):
        """Test that values are updated correctly."""
        config = PolicyConfig(policy_type=PolicyType.EPSILON_GREEDY)
        policy = create_policy(config)
        
        state = {}
        
        # Update action 'A' with rewards
        policy.update(state, 'A', 1.0)
        policy.update(state, 'A', 1.0)
        policy.update(state, 'A', 1.0)
        
        assert policy.action_values['A'] == 1.0
        assert policy.action_counts['A'] == 3


class TestUCB:
    """Test Upper Confidence Bound policy."""
    
    def test_initial_exploration(self):
        """Test that UCB explores unseen actions first."""
        config = PolicyConfig(policy_type=PolicyType.UCB, seed=42)
        policy = create_policy(config)
        
        state = {}
        actions = ['A', 'B', 'C', 'D']
        
        # First 4 selections should explore each action
        first_selections = set()
        for _ in range(4):
            action = policy.select_action(state, actions)
            first_selections.add(action)
            policy.update(state, action, 0.5)
        
        assert len(first_selections) == 4  # All actions explored
    
    def test_ucb_exploration_bonus(self):
        """Test that UCB includes exploration bonus."""
        config = PolicyConfig(policy_type=PolicyType.UCB, ucb_c=1.0)
        policy = create_policy(config)
        
        state = {}
        actions = ['A', 'B']
        
        # Make 'A' seem slightly better but less explored
        policy.update(state, 'A', 0.6)
        for _ in range(10):
            policy.update(state, 'B', 0.5)
        
        # UCB values should include exploration bonus
        ucb_values = policy.get_ucb_values(actions)
        
        # 'A' should have higher UCB due to exploration bonus despite less data
        assert ucb_values['A'] > ucb_values['B']
    
    def test_ucb_converges_to_best(self):
        """Test that UCB eventually selects best arm."""
        config = PolicyConfig(policy_type=PolicyType.UCB, seed=42)
        policy = create_policy(config)
        
        state = {}
        actions = ['A', 'B', 'C']
        
        # Simulate: A=0.8, B=0.5, C=0.2 average rewards
        np.random.seed(42)
        for _ in range(100):
            action = policy.select_action(state, actions)
            if action == 'A':
                reward = 0.8 + np.random.randn() * 0.1
            elif action == 'B':
                reward = 0.5 + np.random.randn() * 0.1
            else:
                reward = 0.2 + np.random.randn() * 0.1
            policy.update(state, action, reward)
        
        # After learning, 'A' should have highest estimated value
        assert policy.action_values['A'] > policy.action_values['B']
        assert policy.action_values['B'] > policy.action_values['C']


class TestThompsonSampling:
    """Test Thompson Sampling policy."""
    
    def test_thompson_explores(self):
        """Test that Thompson sampling explores."""
        config = PolicyConfig(policy_type=PolicyType.THOMPSON, seed=42)
        policy = create_policy(config)
        
        state = {}
        actions = ['A', 'B', 'C']
        
        # Run many selections without updates
        selections = [policy.select_action(state, actions) for _ in range(100)]
        
        # Should select all actions due to uncertainty
        assert len(set(selections)) == 3
    
    def test_thompson_converges(self):
        """Test that Thompson sampling converges to best arm."""
        config = PolicyConfig(policy_type=PolicyType.THOMPSON, seed=42)
        policy = create_policy(config)
        
        state = {}
        actions = ['A', 'B']
        
        # Consistently reward 'A' more
        np.random.seed(42)
        for _ in range(50):
            action = policy.select_action(state, actions)
            if action == 'A':
                policy.update(state, action, 1.0)
            else:
                policy.update(state, action, 0.0)
        
        # After learning, should select 'A' more often
        final_selections = [policy.select_action(state, actions) for _ in range(100)]
        a_rate = sum(1 for s in final_selections if s == 'A') / len(final_selections)
        
        assert a_rate > 0.7


class TestContextualBandit:
    """Test contextual bandit policies."""
    
    def test_contextual_uses_context(self):
        """Test that contextual bandit uses context features."""
        config = PolicyConfig(
            policy_type=PolicyType.CONTEXTUAL_BANDIT,
            context_dim=3,
            seed=42
        )
        policy = create_policy(config)
        
        actions = ['A', 'B']
        
        # Train: when context[0] > 0, A is better; otherwise B is better
        np.random.seed(42)
        for _ in range(50):
            context = np.random.randn(3)
            state = {'context': context}
            action = policy.select_action(state, actions)
            
            if context[0] > 0:
                reward = 1.0 if action == 'A' else 0.0
            else:
                reward = 0.0 if action == 'A' else 1.0
            
            policy.update(state, action, reward)
        
        # Test: positive context should prefer A
        pos_selections = []
        for _ in range(50):
            context = np.array([1.0, 0.0, 0.0])
            action = policy.select_action({'context': context}, actions)
            pos_selections.append(action)
        
        a_rate = sum(1 for s in pos_selections if s == 'A') / len(pos_selections)
        assert a_rate > 0.5  # Should prefer A when context[0] > 0


class TestLearningScheduler:
    """Test the LearningScheduler wrapper."""
    
    def test_build_state(self):
        """Test state building."""
        scheduler = LearningScheduler(policy_type=PolicyType.UCB)
        
        operation = {
            'id': 'OP-001',
            'op_code': 'CUT',
            'processing_times': {'M1': 10, 'M2': 15},
        }
        machines = [
            {'machine_id': 'M1', 'queue_length': 3, 'current_load': 0.5},
            {'machine_id': 'M2', 'queue_length': 1, 'current_load': 0.8},
        ]
        
        state = scheduler.build_state(operation, machines)
        
        assert state['operation_id'] == 'OP-001'
        assert state['processing_times']['M1'] == 10
        assert state['queue_lengths']['M1'] == 3
        assert state['machine_loads']['M2'] == 0.8
    
    def test_select_machine(self):
        """Test machine selection."""
        scheduler = LearningScheduler(policy_type=PolicyType.SHORTEST_QUEUE)
        
        state = {'queue_lengths': {'M1': 5, 'M2': 2, 'M3': 8}}
        machine = scheduler.select_machine(state, ['M1', 'M2', 'M3'])
        
        assert machine == 'M2'
    
    def test_update_and_metrics(self):
        """Test update and metrics tracking."""
        scheduler = LearningScheduler(policy_type=PolicyType.UCB)
        
        state = {'processing_times': {'M1': 10, 'M2': 15}}
        
        # Simulate some decisions
        for _ in range(10):
            machine = scheduler.select_machine(state, ['M1', 'M2'])
            scheduler.update(state, machine, reward=0.5, optimal_reward=1.0)
        
        metrics = scheduler.get_metrics()
        
        assert metrics['total_steps'] == 10
        assert metrics['cumulative_reward'] == 5.0
        assert metrics['cumulative_regret'] == 5.0


class TestPolicyMetrics:
    """Test policy metrics tracking."""
    
    def test_metrics_update(self):
        """Test that metrics are updated correctly."""
        metrics = PolicyMetrics()
        
        metrics.update('A', 1.0, 0.0, False)
        metrics.update('B', 0.5, 0.5, True)
        metrics.update('A', 0.8, 0.2, False)
        
        assert metrics.total_steps == 3
        assert metrics.cumulative_reward == 2.3
        assert metrics.cumulative_regret == 0.7
        assert metrics.action_counts['A'] == 2
        assert metrics.action_counts['B'] == 1
    
    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = PolicyMetrics()
        metrics.update('A', 1.0, 0.0, False)
        
        d = metrics.to_dict()
        
        assert 'total_steps' in d
        assert 'cumulative_reward' in d
        assert 'average_reward' in d
        assert 'action_counts' in d


class TestRegretAnalysis:
    """Test regret analysis for bandit algorithms."""
    
    def test_ucb_sublinear_regret(self):
        """Test that UCB achieves sublinear regret."""
        config = PolicyConfig(policy_type=PolicyType.UCB, seed=42)
        policy = create_policy(config)
        
        state = {}
        actions = ['A', 'B', 'C']
        rewards = {'A': 0.8, 'B': 0.5, 'C': 0.2}  # True expected rewards
        optimal_reward = 0.8
        
        regrets = []
        np.random.seed(42)
        
        for t in range(500):
            action = policy.select_action(state, actions)
            reward = rewards[action] + np.random.randn() * 0.1
            policy.update(state, action, reward)
            
            regret = optimal_reward - rewards[action]
            regrets.append(regret)
        
        cumulative_regret = np.cumsum(regrets)
        
        # Regret should grow sublinearly: O(log T)
        # After 500 steps, should be much less than 500 * 0.6 = 300
        assert cumulative_regret[-1] < 100
    
    def test_epsilon_greedy_vs_ucb(self):
        """Test that UCB outperforms epsilon-greedy in the long run."""
        def run_policy(policy_type, seed):
            config = PolicyConfig(policy_type=policy_type, epsilon=0.1, seed=seed)
            policy = create_policy(config)
            
            state = {}
            actions = ['A', 'B', 'C']
            rewards = {'A': 0.8, 'B': 0.5, 'C': 0.2}
            
            total_reward = 0
            np.random.seed(seed)
            
            for _ in range(200):
                action = policy.select_action(state, actions)
                reward = rewards[action] + np.random.randn() * 0.1
                policy.update(state, action, reward)
                total_reward += reward
            
            return total_reward
        
        ucb_rewards = [run_policy(PolicyType.UCB, i) for i in range(5)]
        eg_rewards = [run_policy(PolicyType.EPSILON_GREEDY, i) for i in range(5)]
        
        # UCB should achieve higher or similar total reward
        assert np.mean(ucb_rewards) >= np.mean(eg_rewards) - 10  # Allow some variance


if __name__ == '__main__':
    pytest.main([__file__, '-v'])



