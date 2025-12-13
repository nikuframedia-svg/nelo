"""
Learning Scheduler — Logged Decisions + Learned Policies

R&D Module for WP4: Learning Scheduler + Experimentation

Research Questions:
    Q1: Can learned policies (RL/bandits) outperform fixed heuristics?
    Q4: Can we build a scheduler that improves over time from data?

Hypotheses:
    H4.3: Contextual bandits achieve lower cumulative regret than fixed heuristics
    H4.4: Learned scheduling policies transfer across similar product families

Technical Uncertainty:
    - Will learned policies generalize to new orders/products?
    - How to handle non-stationarity (changing demand patterns)?
    - Cold start problem: how to bootstrap learning?
    - Computational cost of online learning in real-time scheduling

Architecture:
    ┌─────────────────┐      ┌──────────────────┐
    │  Scheduler      │ ───► │  Decision Logger │ ───► Experiment DB
    │  (deterministic)│      └──────────────────┘
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Policy Selector│ ← Fixed (baseline) OR Learned (experiment)
    └─────────────────┘

Usage:
    from backend.research.learning_scheduler import LearningScheduler
    
    scheduler = LearningScheduler(policy="contextual_bandit")
    plan = scheduler.build_plan(orders, machines, routing)
    
    # After execution
    scheduler.update_with_feedback(actual_outcomes)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import random

import pandas as pd
import numpy as np


class PolicyType(Enum):
    """Available scheduling policies."""
    FIXED_PRIORITY = "fixed_priority"       # Baseline: static priority rules
    EPSILON_GREEDY = "epsilon_greedy"       # Simple exploration
    UCB = "ucb"                             # Upper Confidence Bound
    THOMPSON = "thompson"                   # Thompson Sampling
    CONTEXTUAL_BANDIT = "contextual_bandit" # Feature-based decisions
    REINFORCEMENT = "reinforcement"         # Full RL (future)


@dataclass
class SchedulingDecision:
    """A single scheduling decision to be logged."""
    decision_id: str
    timestamp: str
    decision_type: str  # "priority", "machine", "route", "sequence"
    context: Dict[str, Any]  # Features at decision time
    action: str  # What was decided
    alternatives: List[str]  # What else could have been chosen
    policy_used: str
    confidence: float
    
    # Filled in after execution
    reward: Optional[float] = None
    actual_outcome: Optional[Dict[str, Any]] = None


@dataclass
class PolicyState:
    """State of a learned policy."""
    policy_type: PolicyType
    parameters: Dict[str, Any] = field(default_factory=dict)
    n_decisions: int = 0
    total_reward: float = 0.0
    last_updated: Optional[str] = None


class SchedulingPolicy(ABC):
    """Abstract base class for scheduling policies."""
    
    @abstractmethod
    def select_action(
        self,
        context: Dict[str, Any],
        available_actions: List[str],
    ) -> Tuple[str, float]:
        """
        Select an action given context.
        
        Returns:
            (action, confidence)
        """
        pass
    
    @abstractmethod
    def update(
        self,
        context: Dict[str, Any],
        action: str,
        reward: float,
    ) -> None:
        """Update policy with observed reward."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the policy."""
        pass
    
    @abstractmethod
    def get_state(self) -> PolicyState:
        """Get current policy state for persistence."""
        pass


class FixedPriorityPolicy(SchedulingPolicy):
    """
    Baseline policy: Fixed priority rules.
    
    Priority rules (in order):
    1. Due date (EDD - Earliest Due Date)
    2. Processing time (SPT - Shortest Processing Time)
    3. Order priority field
    """
    
    def __init__(self, rule: str = "EDD"):
        self.rule = rule
        self.n_decisions = 0
    
    def select_action(
        self,
        context: Dict[str, Any],
        available_actions: List[str],
    ) -> Tuple[str, float]:
        self.n_decisions += 1
        
        if not available_actions:
            return "", 0.0
        
        # Simple: return first action (already sorted by priority externally)
        return available_actions[0], 1.0
    
    def update(self, context: Dict[str, Any], action: str, reward: float) -> None:
        # Fixed policy doesn't learn
        pass
    
    @property
    def name(self) -> str:
        return f"fixed_priority_{self.rule}"
    
    def get_state(self) -> PolicyState:
        return PolicyState(
            policy_type=PolicyType.FIXED_PRIORITY,
            parameters={"rule": self.rule},
            n_decisions=self.n_decisions,
        )


class EpsilonGreedyPolicy(SchedulingPolicy):
    """
    Epsilon-greedy exploration policy.
    
    With probability epsilon, explore (random action).
    Otherwise, exploit (best action based on learned values).
    """
    
    def __init__(self, epsilon: float = 0.1, decay: float = 0.99):
        self.epsilon = epsilon
        self.decay = decay
        self.action_values: Dict[str, float] = {}
        self.action_counts: Dict[str, int] = {}
        self.n_decisions = 0
        self.total_reward = 0.0
    
    def select_action(
        self,
        context: Dict[str, Any],
        available_actions: List[str],
    ) -> Tuple[str, float]:
        self.n_decisions += 1
        
        if not available_actions:
            return "", 0.0
        
        # Exploration
        if random.random() < self.epsilon:
            action = random.choice(available_actions)
            return action, self.epsilon
        
        # Exploitation
        best_action = None
        best_value = float("-inf")
        for action in available_actions:
            value = self.action_values.get(action, 0.0)
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action or available_actions[0], 1.0 - self.epsilon
    
    def update(self, context: Dict[str, Any], action: str, reward: float) -> None:
        self.total_reward += reward
        
        # Update action value (incremental mean)
        count = self.action_counts.get(action, 0)
        old_value = self.action_values.get(action, 0.0)
        new_count = count + 1
        new_value = old_value + (reward - old_value) / new_count
        
        self.action_values[action] = new_value
        self.action_counts[action] = new_count
        
        # Decay epsilon
        self.epsilon *= self.decay
    
    @property
    def name(self) -> str:
        return f"epsilon_greedy_{self.epsilon:.2f}"
    
    def get_state(self) -> PolicyState:
        return PolicyState(
            policy_type=PolicyType.EPSILON_GREEDY,
            parameters={
                "epsilon": self.epsilon,
                "action_values": self.action_values,
            },
            n_decisions=self.n_decisions,
            total_reward=self.total_reward,
        )


class UCBPolicy(SchedulingPolicy):
    """
    Upper Confidence Bound (UCB) policy.
    
    Balances exploration and exploitation using confidence bounds.
    
    TODO[R&D]: Test UCB vs epsilon-greedy in E4.1.
    """
    
    def __init__(self, c: float = 2.0):
        self.c = c  # Exploration parameter
        self.action_values: Dict[str, float] = {}
        self.action_counts: Dict[str, int] = {}
        self.n_decisions = 0
        self.total_reward = 0.0
    
    def select_action(
        self,
        context: Dict[str, Any],
        available_actions: List[str],
    ) -> Tuple[str, float]:
        self.n_decisions += 1
        
        if not available_actions:
            return "", 0.0
        
        best_action = None
        best_ucb = float("-inf")
        
        for action in available_actions:
            count = self.action_counts.get(action, 0)
            
            if count == 0:
                # Never tried: infinite UCB (explore)
                return action, 0.0
            
            value = self.action_values.get(action, 0.0)
            exploration_bonus = self.c * np.sqrt(np.log(self.n_decisions) / count)
            ucb = value + exploration_bonus
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action
        
        return best_action or available_actions[0], 0.8
    
    def update(self, context: Dict[str, Any], action: str, reward: float) -> None:
        self.total_reward += reward
        
        count = self.action_counts.get(action, 0)
        old_value = self.action_values.get(action, 0.0)
        new_count = count + 1
        new_value = old_value + (reward - old_value) / new_count
        
        self.action_values[action] = new_value
        self.action_counts[action] = new_count
    
    @property
    def name(self) -> str:
        return f"ucb_c{self.c}"
    
    def get_state(self) -> PolicyState:
        return PolicyState(
            policy_type=PolicyType.UCB,
            parameters={"c": self.c, "action_values": self.action_values},
            n_decisions=self.n_decisions,
            total_reward=self.total_reward,
        )


class ContextualBanditPolicy(SchedulingPolicy):
    """
    Contextual bandit policy.
    
    Uses context features to make decisions.
    
    TODO[R&D]: Implement with linear or neural model.
    TODO[R&D]: Define context features for scheduling.
    
    Features to consider:
    - Machine current load
    - Order priority
    - Time until due date
    - Setup family of previous job
    - Day of week / shift
    """
    
    def __init__(self):
        self.n_decisions = 0
        self.total_reward = 0.0
        self._fallback = EpsilonGreedyPolicy()
    
    def select_action(
        self,
        context: Dict[str, Any],
        available_actions: List[str],
    ) -> Tuple[str, float]:
        self.n_decisions += 1
        
        # TODO[R&D]: Implement context-aware selection
        # For now, fallback to epsilon-greedy
        return self._fallback.select_action(context, available_actions)
    
    def update(self, context: Dict[str, Any], action: str, reward: float) -> None:
        self.total_reward += reward
        self._fallback.update(context, action, reward)
    
    @property
    def name(self) -> str:
        return "contextual_bandit"
    
    def get_state(self) -> PolicyState:
        return PolicyState(
            policy_type=PolicyType.CONTEXTUAL_BANDIT,
            n_decisions=self.n_decisions,
            total_reward=self.total_reward,
        )


class LearningScheduler:
    """
    Main learning scheduler that wraps deterministic scheduling
    with learning capabilities.
    
    Key principle: The core scheduling logic remains deterministic.
    Learning affects only parameter selection and priority ordering.
    """
    
    POLICIES: Dict[PolicyType, type] = {
        PolicyType.FIXED_PRIORITY: FixedPriorityPolicy,
        PolicyType.EPSILON_GREEDY: EpsilonGreedyPolicy,
        PolicyType.UCB: UCBPolicy,
        PolicyType.CONTEXTUAL_BANDIT: ContextualBanditPolicy,
    }
    
    def __init__(
        self,
        policy_type: PolicyType = PolicyType.FIXED_PRIORITY,
        policy_kwargs: Optional[Dict[str, Any]] = None,
    ):
        policy_class = self.POLICIES.get(policy_type, FixedPriorityPolicy)
        self.policy = policy_class(**(policy_kwargs or {}))
        
        self._decision_log: List[SchedulingDecision] = []
        self._pending_decisions: Dict[str, SchedulingDecision] = {}
    
    def log_decision(
        self,
        decision_type: str,
        context: Dict[str, Any],
        action: str,
        alternatives: List[str],
    ) -> str:
        """Log a scheduling decision."""
        decision_id = f"D-{len(self._decision_log):06d}"
        
        decision = SchedulingDecision(
            decision_id=decision_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            decision_type=decision_type,
            context=context,
            action=action,
            alternatives=alternatives,
            policy_used=self.policy.name,
            confidence=0.0,
        )
        
        self._decision_log.append(decision)
        self._pending_decisions[decision_id] = decision
        
        return decision_id
    
    def record_outcome(
        self,
        decision_id: str,
        reward: float,
        outcome: Dict[str, Any],
    ) -> None:
        """Record the outcome of a decision for learning."""
        if decision_id not in self._pending_decisions:
            return
        
        decision = self._pending_decisions[decision_id]
        decision.reward = reward
        decision.actual_outcome = outcome
        
        # Update policy
        self.policy.update(decision.context, decision.action, reward)
        
        del self._pending_decisions[decision_id]
    
    def get_decision_log(self) -> List[Dict[str, Any]]:
        """Get all logged decisions."""
        return [
            {
                "decision_id": d.decision_id,
                "timestamp": d.timestamp,
                "decision_type": d.decision_type,
                "action": d.action,
                "policy_used": d.policy_used,
                "reward": d.reward,
            }
            for d in self._decision_log
        ]
    
    def get_policy_state(self) -> Dict[str, Any]:
        """Get current policy state."""
        state = self.policy.get_state()
        return {
            "policy_type": state.policy_type.value,
            "parameters": state.parameters,
            "n_decisions": state.n_decisions,
            "total_reward": state.total_reward,
        }
    
    def compute_regret(self, oracle_rewards: List[float]) -> float:
        """
        Compute cumulative regret vs oracle.
        
        TODO[R&D]: Use for experiment E4.1.
        """
        actual_rewards = [
            d.reward for d in self._decision_log
            if d.reward is not None
        ]
        
        if len(actual_rewards) != len(oracle_rewards):
            return float("nan")
        
        return sum(oracle_rewards) - sum(actual_rewards)


# ============================================================
# EXPERIMENT SUPPORT
# ============================================================

def run_policy_comparison_experiment(
    n_decisions: int,
    environment_fn,  # Callable that generates (context, actions, rewards)
    policies: List[PolicyType],
) -> Dict[str, Any]:
    """
    Compare multiple policies on the same environment.
    
    TODO[R&D]: Entry point for experiment E4.1.
    """
    results = {}
    
    for policy_type in policies:
        scheduler = LearningScheduler(policy_type=policy_type)
        total_reward = 0.0
        regrets = []
        
        for i in range(n_decisions):
            context, actions, true_rewards = environment_fn(i)
            
            # Select action
            action, _ = scheduler.policy.select_action(context, actions)
            
            # Get reward
            action_idx = actions.index(action) if action in actions else 0
            reward = true_rewards[action_idx]
            total_reward += reward
            
            # Update policy
            scheduler.policy.update(context, action, reward)
            
            # Compute regret
            best_reward = max(true_rewards)
            regrets.append(best_reward - reward)
        
        results[policy_type.value] = {
            "total_reward": total_reward,
            "cumulative_regret": sum(regrets),
            "avg_regret": np.mean(regrets),
            "policy_state": scheduler.get_policy_state(),
        }
    
    return results



