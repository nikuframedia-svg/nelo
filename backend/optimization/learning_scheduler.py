"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — LEARNING SCHEDULER (WP4)
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Adaptive scheduling policies using Multi-Armed Bandits, Contextual Bandits, and Reinforcement Learning.

MOTIVATION
══════════

Traditional APS uses fixed heuristics (SPT, EDD, etc.). However:
- No single heuristic is optimal for all scenarios
- Factory conditions change over time
- Learning from experience can improve decisions

The Learning Scheduler explores the policy space and learns which decisions lead to better outcomes.

MATHEMATICAL FRAMEWORK
══════════════════════

Multi-Armed Bandit (MAB):
─────────────────────────

At each decision point t:
1. Choose action aₜ ∈ A (e.g., which machine to assign)
2. Receive reward rₜ (e.g., negative tardiness)
3. Update policy based on (aₜ, rₜ)

Goal: Minimize cumulative regret

    Regret(T) = T · μ* - Σₜ rₜ

where μ* = max_a E[r|a] is the optimal expected reward.

Contextual Bandit:
──────────────────

At each decision point t:
1. Observe context xₜ ∈ X (e.g., machine loads, due dates)
2. Choose action aₜ based on policy π(xₜ)
3. Receive reward rₜ
4. Update policy

Goal: Learn mapping π: X → A that maximizes expected reward.

POLICIES IMPLEMENTED
════════════════════

1. FIXED HEURISTICS (baseline):
   - fixed_priority: Always prioritize by order priority
   - shortest_queue: Choose machine with shortest queue
   - shortest_processing: SPT - shortest processing time first
   - earliest_due_date: EDD - earliest due date first
   - load_balanced: Distribute load evenly across machines

2. EXPLORATION-EXPLOITATION ALGORITHMS:
   - epsilon_greedy(ε): Random with probability ε, best otherwise
   - ucb (Upper Confidence Bound): Balance mean + uncertainty
   - thompson: Sample from posterior distribution (Bayesian)

3. CONTEXTUAL METHODS:
   - contextual_bandit: Linear UCB with context features
   - contextual_thompson: Thompson sampling with context

4. REINFORCEMENT LEARNING (stubs):
   - dqn: Deep Q-Network (requires training)
   - ppo: Proximal Policy Optimization (requires training)

THEORY: UPPER CONFIDENCE BOUND (UCB)
────────────────────────────────────

UCB selects the action that maximizes:

    UCB(a) = μ̂(a) + c · √(ln(t) / n(a))

where:
    μ̂(a) = estimated mean reward for action a
    n(a) = number of times action a was selected
    t    = total number of decisions
    c    = exploration constant (default: √2)

UCB achieves O(log T) regret, which is optimal for MAB.

THEORY: THOMPSON SAMPLING
─────────────────────────

Thompson sampling maintains a posterior distribution over the reward parameters.

For Beta-Bernoulli bandits:
    1. For each action a, maintain Beta(αₐ, βₐ)
    2. Sample θₐ ~ Beta(αₐ, βₐ) for each action
    3. Select argmax_a θₐ
    4. Update: if reward=1, αₐ += 1; else βₐ += 1

For Gaussian rewards:
    1. Maintain Normal-Inverse-Gamma posterior for (μ, σ²)
    2. Sample from posterior
    3. Select argmax_a μₐ

Thompson sampling is Bayes-optimal and achieves O(log T) regret.

METRICS
═══════

- Cumulative Reward: Σₜ rₜ
- Cumulative Regret: T · μ* - Σₜ rₜ
- Average Reward: (Σₜ rₜ) / T
- Action Distribution: frequency of each action
- SNR Context Score: predictability of context-reward relationship

R&D / SIFIDE ALIGNMENT
──────────────────────
Work Package 4: Learning Scheduler
- Hypothesis H4.1: UCB outperforms fixed heuristics after N=100 decisions
- Hypothesis H4.2: Contextual bandits improve with machine load features
- Experiment E4.1: Compare regret curves across policies

REFERENCES
──────────
[1] Lattimore & Szepesvári (2020). Bandit Algorithms. Cambridge University Press.
[2] Auer, Cesa-Bianchi & Fischer (2002). Finite-time Analysis of the Multiarmed Bandit Problem.
[3] Russo et al. (2018). A Tutorial on Thompson Sampling. Foundations and Trends in ML.
"""

from __future__ import annotations

import json
import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Generic

import numpy as np

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# TYPE DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

State = Dict[str, Any]
Action = str
Reward = float
Context = np.ndarray


class PolicyType(str, Enum):
    """Available policy types."""
    # Fixed heuristics
    FIXED_PRIORITY = "fixed_priority"
    SHORTEST_QUEUE = "shortest_queue"
    SHORTEST_PROCESSING = "shortest_processing"
    EARLIEST_DUE_DATE = "earliest_due_date"
    LOAD_BALANCED = "load_balanced"
    
    # Exploration-exploitation
    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "ucb"
    THOMPSON = "thompson"
    
    # Contextual
    CONTEXTUAL_BANDIT = "contextual_bandit"
    CONTEXTUAL_THOMPSON = "contextual_thompson"
    
    # Reinforcement Learning
    DQN = "dqn"
    PPO = "ppo"


@dataclass
class PolicyConfig:
    """Configuration for a scheduling policy."""
    policy_type: PolicyType
    epsilon: float = 0.1  # For epsilon-greedy
    ucb_c: float = 1.414  # sqrt(2) for UCB
    learning_rate: float = 0.01  # For gradient methods
    discount_factor: float = 0.99  # For RL
    context_dim: int = 10  # For contextual bandits
    seed: Optional[int] = None


@dataclass
class DecisionRecord:
    """Record of a single decision for logging and analysis."""
    timestamp: str
    step: int
    state: Dict[str, Any]
    context: Optional[List[float]]
    available_actions: List[str]
    chosen_action: str
    reward: float
    regret: float
    policy_type: str
    exploration: bool  # Was this an exploration step?
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'step': self.step,
            'state': self.state,
            'context': self.context,
            'available_actions': self.available_actions,
            'chosen_action': self.chosen_action,
            'reward': round(self.reward, 4),
            'regret': round(self.regret, 4),
            'policy_type': self.policy_type,
            'exploration': self.exploration,
        }


@dataclass
class PolicyMetrics:
    """Metrics for evaluating a policy."""
    total_steps: int = 0
    cumulative_reward: float = 0.0
    cumulative_regret: float = 0.0
    action_counts: Dict[str, int] = field(default_factory=dict)
    reward_history: List[float] = field(default_factory=list)
    regret_history: List[float] = field(default_factory=list)
    exploration_rate: float = 0.0
    snr_context_score: float = 1.0
    
    def update(self, action: str, reward: float, regret: float, explored: bool):
        self.total_steps += 1
        self.cumulative_reward += reward
        self.cumulative_regret += regret
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        self.reward_history.append(reward)
        self.regret_history.append(self.cumulative_regret)
        
        # Update exploration rate (exponential moving average)
        alpha = 0.05
        self.exploration_rate = (1 - alpha) * self.exploration_rate + alpha * float(explored)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_steps': self.total_steps,
            'cumulative_reward': round(self.cumulative_reward, 4),
            'cumulative_regret': round(self.cumulative_regret, 4),
            'average_reward': round(self.cumulative_reward / max(1, self.total_steps), 4),
            'action_counts': self.action_counts,
            'exploration_rate': round(self.exploration_rate, 4),
            'snr_context_score': round(self.snr_context_score, 4),
        }


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# BASE POLICY CLASS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

class BasePolicy(ABC):
    """
    Abstract base class for scheduling policies.
    
    All policies implement:
        select_action(state, available_actions) -> action
        update(state, action, reward) -> None
    """
    
    def __init__(self, config: PolicyConfig):
        self.config = config
        self.metrics = PolicyMetrics()
        self.rng = np.random.default_rng(config.seed)
        self._step = 0
        self._last_exploration = False
    
    @property
    def name(self) -> str:
        return self.config.policy_type.value
    
    @abstractmethod
    def select_action(
        self,
        state: State,
        available_actions: List[Action]
    ) -> Action:
        """Select an action given current state."""
        pass
    
    @abstractmethod
    def update(
        self,
        state: State,
        action: Action,
        reward: Reward,
        next_state: Optional[State] = None
    ) -> None:
        """Update policy based on observed reward."""
        pass
    
    def record_decision(
        self,
        state: State,
        available_actions: List[Action],
        action: Action,
        reward: Reward,
        optimal_reward: Optional[Reward] = None
    ) -> DecisionRecord:
        """Record a decision for logging."""
        regret = (optimal_reward - reward) if optimal_reward is not None else 0.0
        
        self.metrics.update(action, reward, regret, self._last_exploration)
        
        # Extract context if available
        context = None
        if 'context' in state and state['context'] is not None:
            context = list(state['context'])
        
        return DecisionRecord(
            timestamp=datetime.now().isoformat(),
            step=self._step,
            state={k: v for k, v in state.items() if k != 'context'},
            context=context,
            available_actions=available_actions,
            chosen_action=action,
            reward=reward,
            regret=regret,
            policy_type=self.name,
            exploration=self._last_exploration,
        )
    
    def get_metrics(self) -> PolicyMetrics:
        return self.metrics
    
    def reset(self):
        """Reset policy state."""
        self.metrics = PolicyMetrics()
        self._step = 0


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# FIXED HEURISTIC POLICIES
# ════════════════════════════════════════════════════════════════════════════════════════════════════

class FixedPriorityPolicy(BasePolicy):
    """
    Always select action based on fixed priority ordering.
    
    Deterministic baseline - no learning.
    """
    
    def select_action(self, state: State, available_actions: List[Action]) -> Action:
        self._step += 1
        self._last_exploration = False
        
        # Use priority if available, otherwise first action
        priorities = state.get('priorities', {})
        
        if priorities:
            # Select action with highest priority (lowest value)
            best_action = min(available_actions, key=lambda a: priorities.get(a, 999))
        else:
            best_action = available_actions[0]
        
        return best_action
    
    def update(self, state: State, action: Action, reward: Reward, next_state: Optional[State] = None):
        # Fixed policy does not learn
        pass


class ShortestQueuePolicy(BasePolicy):
    """
    Select machine with shortest queue.
    
    Minimizes waiting time, good for balanced load.
    """
    
    def select_action(self, state: State, available_actions: List[Action]) -> Action:
        self._step += 1
        self._last_exploration = False
        
        queue_lengths = state.get('queue_lengths', {})
        
        if queue_lengths:
            best_action = min(available_actions, key=lambda a: queue_lengths.get(a, 0))
        else:
            best_action = available_actions[0]
        
        return best_action
    
    def update(self, state: State, action: Action, reward: Reward, next_state: Optional[State] = None):
        pass


class LoadBalancedPolicy(BasePolicy):
    """
    Balance load across machines.
    
    Select machine with lowest current load (total processing time).
    """
    
    def select_action(self, state: State, available_actions: List[Action]) -> Action:
        self._step += 1
        self._last_exploration = False
        
        machine_loads = state.get('machine_loads', {})
        
        if machine_loads:
            best_action = min(available_actions, key=lambda a: machine_loads.get(a, 0))
        else:
            best_action = available_actions[0]
        
        return best_action
    
    def update(self, state: State, action: Action, reward: Reward, next_state: Optional[State] = None):
        pass


class ShortestProcessingPolicy(BasePolicy):
    """
    Shortest Processing Time (SPT) first.
    
    Minimizes average completion time.
    """
    
    def select_action(self, state: State, available_actions: List[Action]) -> Action:
        self._step += 1
        self._last_exploration = False
        
        processing_times = state.get('processing_times', {})
        
        if processing_times:
            best_action = min(available_actions, key=lambda a: processing_times.get(a, float('inf')))
        else:
            best_action = available_actions[0]
        
        return best_action
    
    def update(self, state: State, action: Action, reward: Reward, next_state: Optional[State] = None):
        pass


class EarliestDueDatePolicy(BasePolicy):
    """
    Earliest Due Date (EDD) first.
    
    Minimizes maximum lateness (Moore's algorithm).
    """
    
    def select_action(self, state: State, available_actions: List[Action]) -> Action:
        self._step += 1
        self._last_exploration = False
        
        due_dates = state.get('due_dates', {})
        
        if due_dates:
            best_action = min(available_actions, key=lambda a: due_dates.get(a, float('inf')))
        else:
            best_action = available_actions[0]
        
        return best_action
    
    def update(self, state: State, action: Action, reward: Reward, next_state: Optional[State] = None):
        pass


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# EXPLORATION-EXPLOITATION POLICIES
# ════════════════════════════════════════════════════════════════════════════════════════════════════

class EpsilonGreedyPolicy(BasePolicy):
    """
    Epsilon-Greedy exploration strategy.
    
    With probability ε: explore (random action)
    With probability 1-ε: exploit (best estimated action)
    
    Simple but effective. ε can decay over time.
    """
    
    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.epsilon = config.epsilon
        self.action_values: Dict[Action, float] = {}  # Q(a) estimates
        self.action_counts: Dict[Action, int] = {}
    
    def select_action(self, state: State, available_actions: List[Action]) -> Action:
        self._step += 1
        
        # Exploration
        if self.rng.random() < self.epsilon:
            self._last_exploration = True
            return self.rng.choice(available_actions)
        
        # Exploitation
        self._last_exploration = False
        
        # Initialize unseen actions
        for a in available_actions:
            if a not in self.action_values:
                self.action_values[a] = 0.0
                self.action_counts[a] = 0
        
        # Select best action
        best_value = max(self.action_values.get(a, 0.0) for a in available_actions)
        best_actions = [a for a in available_actions if self.action_values.get(a, 0.0) == best_value]
        
        return self.rng.choice(best_actions)
    
    def update(self, state: State, action: Action, reward: Reward, next_state: Optional[State] = None):
        """
        Incremental mean update:
        
            Q(a) ← Q(a) + (1/n) · (r - Q(a))
        """
        if action not in self.action_counts:
            self.action_counts[action] = 0
            self.action_values[action] = 0.0
        
        self.action_counts[action] += 1
        n = self.action_counts[action]
        
        # Incremental update
        self.action_values[action] += (1.0 / n) * (reward - self.action_values[action])


class UCBPolicy(BasePolicy):
    """
    Upper Confidence Bound (UCB1) policy.
    
    Selects action maximizing:
    
        UCB(a) = Q̂(a) + c · √(ln(t) / n(a))
    
    where:
        Q̂(a) = estimated mean reward for action a
        n(a) = number of times action a was selected
        t    = total number of steps
        c    = exploration constant (default: √2)
    
    Achieves O(log T) regret, which is theoretically optimal.
    """
    
    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.c = config.ucb_c
        self.action_values: Dict[Action, float] = {}
        self.action_counts: Dict[Action, int] = {}
    
    def select_action(self, state: State, available_actions: List[Action]) -> Action:
        self._step += 1
        
        # Initialize unseen actions with optimistic value
        for a in available_actions:
            if a not in self.action_values:
                self.action_values[a] = float('inf')  # Optimistic initialization
                self.action_counts[a] = 0
        
        # Select action with highest UCB
        ucb_values = {}
        for a in available_actions:
            if self.action_counts[a] == 0:
                ucb_values[a] = float('inf')
            else:
                exploitation = self.action_values[a]
                exploration = self.c * math.sqrt(math.log(self._step) / self.action_counts[a])
                ucb_values[a] = exploitation + exploration
        
        best_ucb = max(ucb_values.values())
        best_actions = [a for a in available_actions if ucb_values[a] == best_ucb]
        
        chosen = self.rng.choice(best_actions)
        self._last_exploration = self.action_counts[chosen] < 5  # Consider first 5 as exploration
        
        return chosen
    
    def update(self, state: State, action: Action, reward: Reward, next_state: Optional[State] = None):
        if action not in self.action_counts:
            self.action_counts[action] = 0
            self.action_values[action] = 0.0
        
        self.action_counts[action] += 1
        n = self.action_counts[action]
        
        # Incremental update
        self.action_values[action] += (1.0 / n) * (reward - self.action_values[action])
    
    def get_ucb_values(self, available_actions: List[Action]) -> Dict[Action, float]:
        """Get current UCB values for debugging/visualization."""
        ucb_values = {}
        for a in available_actions:
            if self.action_counts.get(a, 0) == 0:
                ucb_values[a] = float('inf')
            else:
                exploitation = self.action_values.get(a, 0.0)
                exploration = self.c * math.sqrt(math.log(self._step) / self.action_counts[a])
                ucb_values[a] = exploitation + exploration
        return ucb_values


class ThompsonSamplingPolicy(BasePolicy):
    """
    Thompson Sampling (Posterior Sampling) policy.
    
    Maintains Beta(α, β) posterior for each action assuming Bernoulli rewards.
    
    Algorithm:
    1. For each action a, sample θ_a ~ Beta(α_a, β_a)
    2. Select action with highest sampled θ
    3. Observe reward r ∈ {0, 1}
    4. Update: α_a += r, β_a += (1 - r)
    
    For continuous rewards, we use Gaussian approximation:
        Sample θ_a ~ N(μ_a, σ_a²)
    
    Thompson sampling is Bayes-optimal and matches UCB's O(log T) regret.
    """
    
    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        # Beta distribution parameters for each action
        self.alpha: Dict[Action, float] = {}  # successes + 1
        self.beta: Dict[Action, float] = {}   # failures + 1
        
        # For continuous rewards: track mean and variance
        self.sum_rewards: Dict[Action, float] = {}
        self.sum_sq_rewards: Dict[Action, float] = {}
        self.action_counts: Dict[Action, int] = {}
    
    def select_action(self, state: State, available_actions: List[Action]) -> Action:
        self._step += 1
        
        # Initialize unseen actions with uniform prior
        for a in available_actions:
            if a not in self.alpha:
                self.alpha[a] = 1.0
                self.beta[a] = 1.0
                self.sum_rewards[a] = 0.0
                self.sum_sq_rewards[a] = 0.0
                self.action_counts[a] = 0
        
        # Sample from posterior for each action
        samples = {}
        for a in available_actions:
            if self.action_counts[a] < 5:
                # Use Beta sampling for initial exploration
                samples[a] = self.rng.beta(self.alpha[a], self.beta[a])
            else:
                # Use Gaussian approximation for continuous rewards
                n = self.action_counts[a]
                mean = self.sum_rewards[a] / n
                variance = max(0.01, (self.sum_sq_rewards[a] / n) - mean**2)
                std = math.sqrt(variance / n)  # Standard error of mean
                samples[a] = self.rng.normal(mean, std)
        
        best_sample = max(samples.values())
        best_actions = [a for a in available_actions if samples[a] == best_sample]
        
        chosen = self.rng.choice(best_actions)
        self._last_exploration = self.action_counts[chosen] < 5
        
        return chosen
    
    def update(self, state: State, action: Action, reward: Reward, next_state: Optional[State] = None):
        if action not in self.alpha:
            self.alpha[action] = 1.0
            self.beta[action] = 1.0
            self.sum_rewards[action] = 0.0
            self.sum_sq_rewards[action] = 0.0
            self.action_counts[action] = 0
        
        # Update Beta parameters (treating normalized reward as Bernoulli-like)
        # For continuous rewards, we map to [0, 1] using sigmoid
        normalized = 1.0 / (1.0 + math.exp(-reward))  # Sigmoid
        self.alpha[action] += normalized
        self.beta[action] += (1 - normalized)
        
        # Update for Gaussian approximation
        self.sum_rewards[action] += reward
        self.sum_sq_rewards[action] += reward ** 2
        self.action_counts[action] += 1


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# CONTEXTUAL BANDIT POLICIES
# ════════════════════════════════════════════════════════════════════════════════════════════════════

class ContextualBanditPolicy(BasePolicy):
    """
    Linear Upper Confidence Bound (LinUCB) for contextual bandits.
    
    For each action a, maintains a linear model:
    
        r = θ_a^T · x + ε
    
    where x is the context vector.
    
    Algorithm:
    1. For each action a, compute:
       - θ̂_a = A_a^{-1} · b_a  (ridge regression estimate)
       - UCB_a = θ̂_a^T · x + α · √(x^T · A_a^{-1} · x)
    
    2. Select action with highest UCB
    
    3. Update:
       - A_a ← A_a + x · x^T
       - b_a ← b_a + r · x
    
    References:
    Li et al. (2010). "A Contextual-Bandit Approach to Personalized News Article Recommendation"
    """
    
    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.dim = config.context_dim
        self.alpha = config.ucb_c
        
        # Per-action: A_a and b_a
        self.A: Dict[Action, np.ndarray] = {}
        self.b: Dict[Action, np.ndarray] = {}
    
    def _init_action(self, action: Action):
        if action not in self.A:
            self.A[action] = np.eye(self.dim)
            self.b[action] = np.zeros(self.dim)
    
    def select_action(self, state: State, available_actions: List[Action]) -> Action:
        self._step += 1
        
        # Get context
        context = state.get('context')
        if context is None:
            context = np.zeros(self.dim)
        else:
            context = np.asarray(context)[:self.dim]
            if len(context) < self.dim:
                context = np.pad(context, (0, self.dim - len(context)))
        
        # Initialize unseen actions
        for a in available_actions:
            self._init_action(a)
        
        # Compute UCB for each action
        ucb_values = {}
        for a in available_actions:
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            
            exploitation = theta @ context
            exploration = self.alpha * np.sqrt(context @ A_inv @ context)
            ucb_values[a] = exploitation + exploration
        
        best_ucb = max(ucb_values.values())
        best_actions = [a for a in available_actions if ucb_values[a] == best_ucb]
        
        chosen = self.rng.choice(best_actions)
        self._last_exploration = np.sqrt(context @ np.linalg.inv(self.A[chosen]) @ context) > 0.5
        
        return chosen
    
    def update(self, state: State, action: Action, reward: Reward, next_state: Optional[State] = None):
        context = state.get('context')
        if context is None:
            return
        
        context = np.asarray(context)[:self.dim]
        if len(context) < self.dim:
            context = np.pad(context, (0, self.dim - len(context)))
        
        self._init_action(action)
        
        # Update A and b
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context


class ContextualThompsonPolicy(BasePolicy):
    """
    Thompson Sampling for Contextual Bandits.
    
    Maintains Bayesian linear regression model for each action:
    
        r | x, θ ~ N(θ^T x, σ²)
        θ ~ N(μ, Σ)
    
    Algorithm:
    1. Sample θ̃_a ~ N(μ_a, Σ_a) for each action
    2. Select action maximizing θ̃_a^T · x
    3. Update posterior using Bayes' rule
    """
    
    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.dim = config.context_dim
        self.sigma_sq = 1.0  # Noise variance
        self.lambda_reg = 1.0  # Regularization
        
        # Per-action: mean and precision
        self.B: Dict[Action, np.ndarray] = {}  # Precision matrix
        self.mu: Dict[Action, np.ndarray] = {}  # Mean
        self.f: Dict[Action, np.ndarray] = {}   # For computing mean
    
    def _init_action(self, action: Action):
        if action not in self.B:
            self.B[action] = self.lambda_reg * np.eye(self.dim)
            self.mu[action] = np.zeros(self.dim)
            self.f[action] = np.zeros(self.dim)
    
    def select_action(self, state: State, available_actions: List[Action]) -> Action:
        self._step += 1
        
        context = state.get('context')
        if context is None:
            context = np.zeros(self.dim)
        else:
            context = np.asarray(context)[:self.dim]
            if len(context) < self.dim:
                context = np.pad(context, (0, self.dim - len(context)))
        
        for a in available_actions:
            self._init_action(a)
        
        # Sample from posterior and compute expected reward
        samples = {}
        for a in available_actions:
            B_inv = np.linalg.inv(self.B[a])
            mu = B_inv @ self.f[a]
            
            # Sample from posterior
            try:
                theta_sample = self.rng.multivariate_normal(mu, self.sigma_sq * B_inv)
            except np.linalg.LinAlgError:
                theta_sample = mu
            
            samples[a] = theta_sample @ context
        
        best_sample = max(samples.values())
        best_actions = [a for a in available_actions if samples[a] == best_sample]
        
        chosen = self.rng.choice(best_actions)
        self._last_exploration = True  # Thompson always explores
        
        return chosen
    
    def update(self, state: State, action: Action, reward: Reward, next_state: Optional[State] = None):
        context = state.get('context')
        if context is None:
            return
        
        context = np.asarray(context)[:self.dim]
        if len(context) < self.dim:
            context = np.pad(context, (0, self.dim - len(context)))
        
        self._init_action(action)
        
        # Update precision and mean parameters
        self.B[action] += np.outer(context, context) / self.sigma_sq
        self.f[action] += reward * context / self.sigma_sq


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# REINFORCEMENT LEARNING POLICIES (STUBS)
# ════════════════════════════════════════════════════════════════════════════════════════════════════

class DQNPolicy(BasePolicy):
    """
    Deep Q-Network policy stub.
    
    TODO[R&D]: Implement full DQN with:
    - Neural network Q(s, a; θ)
    - Experience replay buffer
    - Target network for stability
    - Epsilon-greedy exploration
    
    Reference: Mnih et al. (2015). Human-level control through deep reinforcement learning.
    """
    
    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.epsilon = config.epsilon
        logger.warning("DQN policy is a stub. Using epsilon-greedy fallback.")
    
    def select_action(self, state: State, available_actions: List[Action]) -> Action:
        self._step += 1
        
        # Fallback to epsilon-greedy with uniform Q-values
        if self.rng.random() < self.epsilon:
            self._last_exploration = True
            return self.rng.choice(available_actions)
        
        self._last_exploration = False
        return available_actions[0]
    
    def update(self, state: State, action: Action, reward: Reward, next_state: Optional[State] = None):
        # TODO: Implement experience replay and network update
        pass


class PPOPolicy(BasePolicy):
    """
    Proximal Policy Optimization stub.
    
    TODO[R&D]: Implement PPO with:
    - Actor-critic architecture
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    
    Reference: Schulman et al. (2017). Proximal Policy Optimization Algorithms.
    """
    
    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        logger.warning("PPO policy is a stub. Using random fallback.")
    
    def select_action(self, state: State, available_actions: List[Action]) -> Action:
        self._step += 1
        self._last_exploration = True
        return self.rng.choice(available_actions)
    
    def update(self, state: State, action: Action, reward: Reward, next_state: Optional[State] = None):
        # TODO: Implement policy gradient update
        pass


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# POLICY FACTORY
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def create_policy(config: PolicyConfig) -> BasePolicy:
    """
    Factory function to create a policy from configuration.
    
    Args:
        config: PolicyConfig with policy_type and parameters
    
    Returns:
        Instantiated policy
    """
    policy_classes = {
        PolicyType.FIXED_PRIORITY: FixedPriorityPolicy,
        PolicyType.SHORTEST_QUEUE: ShortestQueuePolicy,
        PolicyType.LOAD_BALANCED: LoadBalancedPolicy,
        PolicyType.SHORTEST_PROCESSING: ShortestProcessingPolicy,
        PolicyType.EARLIEST_DUE_DATE: EarliestDueDatePolicy,
        PolicyType.EPSILON_GREEDY: EpsilonGreedyPolicy,
        PolicyType.UCB: UCBPolicy,
        PolicyType.THOMPSON: ThompsonSamplingPolicy,
        PolicyType.CONTEXTUAL_BANDIT: ContextualBanditPolicy,
        PolicyType.CONTEXTUAL_THOMPSON: ContextualThompsonPolicy,
        PolicyType.DQN: DQNPolicy,
        PolicyType.PPO: PPOPolicy,
    }
    
    policy_class = policy_classes.get(config.policy_type)
    if policy_class is None:
        raise ValueError(f"Unknown policy type: {config.policy_type}")
    
    return policy_class(config)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# LEARNING SCHEDULER
# ════════════════════════════════════════════════════════════════════════════════════════════════════

class LearningScheduler:
    """
    Scheduler that uses learning policies to make scheduling decisions.
    
    Integrates with the APS system to:
    1. Convert scheduling problems to bandit/RL states
    2. Select actions using the configured policy
    3. Compute rewards from scheduling outcomes
    4. Update the policy
    
    Usage:
        scheduler = LearningScheduler(policy_type=PolicyType.UCB)
        
        for operation in operations:
            state = scheduler.build_state(operation, machines)
            machine = scheduler.select_machine(state, eligible_machines)
            
            # Execute scheduling decision
            reward = compute_reward(result)
            scheduler.update(state, machine, reward)
    """
    
    def __init__(
        self,
        policy_type: PolicyType = PolicyType.UCB,
        config: Optional[PolicyConfig] = None,
        log_dir: Optional[Path] = None
    ):
        """
        Initialize Learning Scheduler.
        
        Args:
            policy_type: Type of policy to use
            config: Optional custom configuration
            log_dir: Directory for logging decisions
        """
        if config is None:
            config = PolicyConfig(policy_type=policy_type)
        
        self.policy = create_policy(config)
        self.log_dir = log_dir
        self.decision_log: List[DecisionRecord] = []
    
    def build_state(
        self,
        operation: Dict[str, Any],
        machines: List[Dict[str, Any]],
        context_features: Optional[List[float]] = None
    ) -> State:
        """
        Build state representation from operation and machine info.
        
        Args:
            operation: Current operation to schedule
            machines: Available machines with their current state
            context_features: Optional additional context
        
        Returns:
            State dict for policy
        """
        state = {
            'operation_id': operation.get('id'),
            'op_code': operation.get('op_code'),
            'processing_times': {},
            'queue_lengths': {},
            'machine_loads': {},
            'due_dates': {},
            'priorities': {},
        }
        
        for m in machines:
            m_id = m.get('machine_id', m.get('id'))
            state['processing_times'][m_id] = operation.get('processing_times', {}).get(m_id, 0)
            state['queue_lengths'][m_id] = m.get('queue_length', 0)
            state['machine_loads'][m_id] = m.get('current_load', 0)
        
        if context_features is not None:
            state['context'] = np.array(context_features)
        
        return state
    
    def select_machine(
        self,
        state: State,
        available_machines: List[str]
    ) -> str:
        """
        Select machine for current operation.
        
        Args:
            state: Current state
            available_machines: List of eligible machine IDs
        
        Returns:
            Selected machine ID
        """
        return self.policy.select_action(state, available_machines)
    
    def update(
        self,
        state: State,
        machine: str,
        reward: float,
        optimal_reward: Optional[float] = None
    ):
        """
        Update policy after observing reward.
        
        Args:
            state: State when decision was made
            machine: Selected machine
            reward: Observed reward (e.g., negative tardiness)
            optimal_reward: Optional optimal reward for regret calculation
        """
        self.policy.update(state, machine, reward)
        
        # Log decision
        available = list(state.get('processing_times', {}).keys())
        record = self.policy.record_decision(state, available, machine, reward, optimal_reward)
        self.decision_log.append(record)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current policy metrics."""
        return self.policy.get_metrics().to_dict()
    
    def save_log(self, filepath: Optional[Path] = None):
        """Save decision log to JSON file."""
        if filepath is None and self.log_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.log_dir / f"decisions_{self.policy.name}_{timestamp}.json"
        
        if filepath:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'policy': self.policy.name,
                'metrics': self.get_metrics(),
                'decisions': [d.to_dict() for d in self.decision_log],
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved decision log to {filepath}")
    
    def reset(self):
        """Reset scheduler state."""
        self.policy.reset()
        self.decision_log = []



