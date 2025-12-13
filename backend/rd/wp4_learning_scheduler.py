"""
ProdPlan 4.0 - WP4 Learning Scheduler
=====================================

Work Package 4: Reinforcement Learning for Scheduling

Multi-armed bandit e DRL para seleção dinâmica de políticas:
- Epsilon-greedy / UCB para policy selection
- Learning from production feedback
- Regret minimization

R&D / SIFIDE: Aprendizagem adaptativa para scheduling.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from pydantic import BaseModel, Field

from .experiments_core import (
    WorkPackage,
    ExperimentStatus,
    create_experiment,
    update_experiment_status,
    RDExperimentCreate,
    RDExperiment,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class BanditType(str, Enum):
    """Tipo de algoritmo bandit."""
    EPSILON_GREEDY = "epsilon_greedy"
    UCB1 = "ucb1"
    THOMPSON_SAMPLING = "thompson_sampling"


class RewardType(str, Enum):
    """Tipo de reward function."""
    MAKESPAN = "makespan"          # -makespan (menor é melhor)
    OTD = "otd"                     # OTD rate (maior é melhor)
    COMBINED = "combined"           # -tardiness - α*makespan - β*setups


class PolicyStats(BaseModel):
    """Estatísticas de uma política."""
    policy: str
    num_pulls: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    std_reward: float = 0.0
    ucb_value: float = float('inf')
    thompson_alpha: float = 1.0
    thompson_beta: float = 1.0


class EpisodeResult(BaseModel):
    """Resultado de um episódio."""
    episode_num: int
    policy_selected: str
    reward: float
    regret: float
    kpis: Dict[str, float]
    baseline_reward: float


class WP4RunRequest(BaseModel):
    """Request para episódio WP4."""
    name: str = Field(description="Nome da experiência")
    policies: List[str] = Field(
        default=["FIFO", "SPT", "EDD"],
        description="Políticas para o bandit"
    )
    baseline_policy: str = Field(
        default="FIFO",
        description="Política baseline para calcular regret"
    )
    num_episodes: int = Field(
        default=10,
        description="Número de episódios"
    )
    bandit_type: str = Field(
        default="epsilon_greedy",
        description="Tipo de bandit (epsilon_greedy, ucb1, thompson_sampling)"
    )
    epsilon: float = Field(
        default=0.1,
        description="Epsilon para epsilon-greedy"
    )
    reward_type: str = Field(
        default="combined",
        description="Tipo de reward (makespan, otd, combined)"
    )
    reward_weights: Dict[str, float] = Field(
        default_factory=lambda: {"tardiness": 1.0, "makespan": 0.5, "setups": 0.2},
        description="Pesos para reward combinado"
    )
    context: Dict[str, Any] = Field(default_factory=dict)


class WP4ExperimentResult(BaseModel):
    """Resultado completo de experiência WP4."""
    experiment_id: int
    name: str
    status: str
    num_episodes: int
    policies: List[str]
    baseline_policy: str
    episodes: List[EpisodeResult]
    policy_stats: List[PolicyStats]
    best_policy: str
    avg_reward: float
    avg_regret: float
    cumulative_regret: float
    conclusion: str
    total_time_sec: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# BANDIT SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════

class BanditScheduler:
    """
    Multi-armed bandit para seleção de políticas de scheduling.
    
    Implementa:
    - Epsilon-greedy
    - UCB1 (Upper Confidence Bound)
    - Thompson Sampling (Beta-Bernoulli)
    """
    
    def __init__(
        self,
        policies: List[str],
        bandit_type: str = "epsilon_greedy",
        epsilon: float = 0.1,
    ):
        """
        Inicializa bandit scheduler.
        
        Args:
            policies: Lista de políticas disponíveis
            bandit_type: Tipo de algoritmo
            epsilon: Parâmetro para epsilon-greedy
        """
        self.policies = policies
        self.bandit_type = BanditType(bandit_type)
        self.epsilon = epsilon
        
        # Estatísticas por política
        self.num_pulls: Dict[str, int] = {p: 0 for p in policies}
        self.total_rewards: Dict[str, float] = {p: 0.0 for p in policies}
        self.rewards_squared: Dict[str, float] = {p: 0.0 for p in policies}
        
        # Para Thompson Sampling
        self.alpha: Dict[str, float] = {p: 1.0 for p in policies}
        self.beta: Dict[str, float] = {p: 1.0 for p in policies}
        
        # Total de pulls
        self.total_pulls = 0
        
        logger.info(f"BanditScheduler initialized with {len(policies)} policies, type={bandit_type}")
    
    def select_policy(self, context: Optional[Dict] = None) -> str:
        """
        Seleciona política com base no algoritmo.
        
        Args:
            context: Contexto opcional (para future contextual bandits)
        
        Returns:
            Nome da política selecionada
        """
        if self.bandit_type == BanditType.EPSILON_GREEDY:
            return self._select_epsilon_greedy()
        elif self.bandit_type == BanditType.UCB1:
            return self._select_ucb1()
        elif self.bandit_type == BanditType.THOMPSON_SAMPLING:
            return self._select_thompson_sampling()
        else:
            return random.choice(self.policies)
    
    def _select_epsilon_greedy(self) -> str:
        """Seleção epsilon-greedy."""
        # Explorar com probabilidade epsilon
        if random.random() < self.epsilon:
            return random.choice(self.policies)
        
        # Exploitar: escolher melhor política média
        best_policy = None
        best_avg = -float('inf')
        
        for policy in self.policies:
            if self.num_pulls[policy] == 0:
                return policy  # Ainda não testada
            
            avg = self.total_rewards[policy] / self.num_pulls[policy]
            if avg > best_avg:
                best_avg = avg
                best_policy = policy
        
        return best_policy or random.choice(self.policies)
    
    def _select_ucb1(self) -> str:
        """Seleção UCB1."""
        best_policy = None
        best_ucb = -float('inf')
        
        for policy in self.policies:
            if self.num_pulls[policy] == 0:
                return policy  # Ainda não testada
            
            # UCB = média + sqrt(2 * ln(t) / n)
            avg = self.total_rewards[policy] / self.num_pulls[policy]
            exploration = math.sqrt(2 * math.log(self.total_pulls + 1) / self.num_pulls[policy])
            ucb = avg + exploration
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_policy = policy
        
        return best_policy or random.choice(self.policies)
    
    def _select_thompson_sampling(self) -> str:
        """Seleção Thompson Sampling (Beta-Bernoulli)."""
        best_policy = None
        best_sample = -float('inf')
        
        for policy in self.policies:
            # Sample from Beta(alpha, beta)
            sample = np.random.beta(self.alpha[policy], self.beta[policy])
            
            if sample > best_sample:
                best_sample = sample
                best_policy = policy
        
        return best_policy or random.choice(self.policies)
    
    def update(self, policy: str, reward: float):
        """
        Atualiza estatísticas após observar reward.
        
        Args:
            policy: Política usada
            reward: Reward observado (normalizado para [0,1] para Thompson Sampling)
        """
        self.num_pulls[policy] += 1
        self.total_rewards[policy] += reward
        self.rewards_squared[policy] += reward ** 2
        self.total_pulls += 1
        
        # Para Thompson Sampling, assumir reward normalizado em [0,1]
        # Tratar como Bernoulli: sucesso se reward > 0.5
        if reward > 0.5:
            self.alpha[policy] += 1
        else:
            self.beta[policy] += 1
        
        logger.debug(f"Updated policy {policy}: pulls={self.num_pulls[policy]}, avg_reward={self.get_avg_reward(policy):.4f}")
    
    def get_avg_reward(self, policy: str) -> float:
        """Retorna média de reward para uma política."""
        if self.num_pulls[policy] == 0:
            return 0.0
        return self.total_rewards[policy] / self.num_pulls[policy]
    
    def get_std_reward(self, policy: str) -> float:
        """Retorna desvio padrão de reward para uma política."""
        n = self.num_pulls[policy]
        if n < 2:
            return 0.0
        avg = self.get_avg_reward(policy)
        variance = self.rewards_squared[policy] / n - avg ** 2
        return math.sqrt(max(0, variance))
    
    def get_ucb_value(self, policy: str) -> float:
        """Retorna valor UCB para uma política."""
        if self.num_pulls[policy] == 0:
            return float('inf')
        avg = self.get_avg_reward(policy)
        exploration = math.sqrt(2 * math.log(self.total_pulls + 1) / self.num_pulls[policy])
        return avg + exploration
    
    def get_stats(self) -> List[PolicyStats]:
        """Retorna estatísticas de todas as políticas."""
        stats = []
        for policy in self.policies:
            stats.append(PolicyStats(
                policy=policy,
                num_pulls=self.num_pulls[policy],
                total_reward=round(self.total_rewards[policy], 4),
                avg_reward=round(self.get_avg_reward(policy), 4),
                std_reward=round(self.get_std_reward(policy), 4),
                ucb_value=round(self.get_ucb_value(policy), 4),
                thompson_alpha=self.alpha[policy],
                thompson_beta=self.beta[policy],
            ))
        return sorted(stats, key=lambda s: s.avg_reward, reverse=True)
    
    def get_best_policy(self) -> str:
        """Retorna política com melhor média de reward."""
        best = None
        best_avg = -float('inf')
        for policy in self.policies:
            avg = self.get_avg_reward(policy)
            if avg > best_avg:
                best_avg = avg
                best = policy
        return best or self.policies[0]


# ═══════════════════════════════════════════════════════════════════════════════
# EPISODE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_reward(
    kpis: Dict[str, float],
    reward_type: str,
    weights: Dict[str, float],
) -> float:
    """
    Calcula reward com base nos KPIs.
    
    Args:
        kpis: KPIs do scheduling
        reward_type: Tipo de reward
        weights: Pesos para reward combinado
    
    Returns:
        Reward normalizado (quanto maior, melhor)
    """
    if reward_type == "makespan":
        # Reward = -makespan (normalizado)
        makespan = kpis.get("makespan_hours", 0)
        return -makespan / 100.0  # Normalizar para ~[-1, 0]
    
    elif reward_type == "otd":
        # Reward = OTD rate
        return kpis.get("otd_rate", 0.0)
    
    else:  # combined
        tardiness = kpis.get("total_tardiness_hours", 0)
        makespan = kpis.get("makespan_hours", 0)
        setups = kpis.get("total_setup_time_hours", 0)
        
        w_t = weights.get("tardiness", 1.0)
        w_m = weights.get("makespan", 0.5)
        w_s = weights.get("setups", 0.2)
        
        # Reward negativo (minimizar estes valores)
        reward = -(w_t * tardiness + w_m * makespan + w_s * setups)
        
        # Normalizar para range razoável
        return reward / 100.0


def run_scheduler_episode(
    instance,  # SchedulingInstance
    bandit: BanditScheduler,
    baseline_policy: str,
    reward_type: str = "combined",
    reward_weights: Optional[Dict[str, float]] = None,
) -> EpisodeResult:
    """
    Executa um episódio de learning scheduler.
    
    Processo:
    1. Bandit escolhe a política
    2. Corre scheduling com essa política
    3. Calcula reward
    4. Corre baseline_policy e calcula reward_base
    5. Calcula regret = reward_base - reward (se baseline for melhor)
    6. Atualiza bandit
    
    Args:
        instance: SchedulingInstance com operações
        bandit: BanditScheduler
        baseline_policy: Política para calcular regret
        reward_type: Tipo de reward
        reward_weights: Pesos para reward combinado
    
    Returns:
        EpisodeResult com detalhes do episódio
    """
    from scheduling import HeuristicScheduler, solve_milp, solve_cpsat
    
    reward_weights = reward_weights or {"tardiness": 1.0, "makespan": 0.5, "setups": 0.2}
    
    # 1. Bandit seleciona política
    selected_policy = bandit.select_policy()
    
    # 2. Executar scheduling com política selecionada
    try:
        if selected_policy.upper() == "MILP":
            result = solve_milp(instance, time_limit_sec=30.0)
        elif selected_policy.upper() == "CPSAT":
            result = solve_cpsat(instance, time_limit_sec=30.0)
        else:
            scheduler = HeuristicScheduler(rule=selected_policy)
            result = scheduler.build_schedule(instance)
        
        kpis = result.get("kpis", {})
    except Exception as e:
        logger.error(f"Policy {selected_policy} failed: {e}")
        kpis = {"makespan_hours": 1000, "total_tardiness_hours": 100, "otd_rate": 0}
    
    # 3. Calcular reward
    reward = compute_reward(kpis, reward_type, reward_weights)
    
    # 4. Executar baseline
    try:
        if baseline_policy.upper() == "MILP":
            baseline_result = solve_milp(instance, time_limit_sec=30.0)
        elif baseline_policy.upper() == "CPSAT":
            baseline_result = solve_cpsat(instance, time_limit_sec=30.0)
        else:
            baseline_scheduler = HeuristicScheduler(rule=baseline_policy)
            baseline_result = baseline_scheduler.build_schedule(instance)
        
        baseline_kpis = baseline_result.get("kpis", {})
    except Exception as e:
        logger.error(f"Baseline {baseline_policy} failed: {e}")
        baseline_kpis = {"makespan_hours": 1000, "total_tardiness_hours": 100, "otd_rate": 0}
    
    baseline_reward = compute_reward(baseline_kpis, reward_type, reward_weights)
    
    # 5. Calcular regret
    # Regret = max(0, baseline_reward - reward) se baseline é oráculo
    regret = max(0, baseline_reward - reward)
    
    # 6. Atualizar bandit (normalizar reward para ~[0,1] para Thompson Sampling)
    normalized_reward = (reward + 2) / 4  # Assumir reward in [-2, 2] -> [0, 1]
    normalized_reward = max(0, min(1, normalized_reward))
    bandit.update(selected_policy, normalized_reward)
    
    return EpisodeResult(
        episode_num=bandit.total_pulls,
        policy_selected=selected_policy,
        reward=round(reward, 4),
        regret=round(regret, 4),
        kpis={
            "makespan_hours": round(kpis.get("makespan_hours", 0), 2),
            "tardiness_hours": round(kpis.get("total_tardiness_hours", 0), 2),
            "otd_rate": round(kpis.get("otd_rate", 0), 3),
            "setup_hours": round(kpis.get("total_setup_time_hours", 0), 2),
        },
        baseline_reward=round(baseline_reward, 4),
    )


def run_learning_experiment(request: WP4RunRequest) -> WP4ExperimentResult:
    """
    Executa experiência completa de learning scheduler.
    
    Corre múltiplos episódios e calcula estatísticas.
    """
    from data_loader import load_dataset
    from scheduling import create_instance_from_dataframes
    
    start_time = time.time()
    
    logger.info(f"Starting WP4 Learning Experiment: {request.name}")
    logger.info(f"Policies: {request.policies}, Episodes: {request.num_episodes}")
    
    # Criar experiência R&D
    experiment = create_experiment(RDExperimentCreate(
        wp=WorkPackage.WP4_LEARNING,
        name=request.name,
        description=f"Learning scheduler com {request.num_episodes} episódios",
        parameters={
            "policies": request.policies,
            "baseline_policy": request.baseline_policy,
            "num_episodes": request.num_episodes,
            "bandit_type": request.bandit_type,
            "epsilon": request.epsilon,
            "reward_type": request.reward_type,
        },
    ))
    update_experiment_status(experiment.id, ExperimentStatus.RUNNING)
    
    # Inicializar bandit
    bandit = BanditScheduler(
        policies=request.policies,
        bandit_type=request.bandit_type,
        epsilon=request.epsilon,
    )
    
    # Carregar instância (ou criar demo)
    try:
        data = load_dataset()
        instance = create_instance_from_dataframes(
            orders_df=data.orders,
            routing_df=data.routing,
            machines_df=data.machines,
        )
    except Exception as e:
        logger.warning(f"Could not load data: {e}. Using demo instance.")
        instance = _create_demo_instance()
    
    # Executar episódios
    episodes: List[EpisodeResult] = []
    cumulative_regret = 0.0
    
    for ep in range(request.num_episodes):
        # Variar ligeiramente a instância para cada episódio (simulando dias diferentes)
        varied_instance = _vary_instance(instance, ep)
        
        episode_result = run_scheduler_episode(
            instance=varied_instance,
            bandit=bandit,
            baseline_policy=request.baseline_policy,
            reward_type=request.reward_type,
            reward_weights=request.reward_weights,
        )
        episode_result.episode_num = ep + 1
        episodes.append(episode_result)
        cumulative_regret += episode_result.regret
        
        if (ep + 1) % 5 == 0:
            logger.info(f"Episode {ep+1}/{request.num_episodes}: "
                       f"policy={episode_result.policy_selected}, "
                       f"reward={episode_result.reward:.4f}, "
                       f"cumulative_regret={cumulative_regret:.4f}")
    
    # Estatísticas finais
    policy_stats = bandit.get_stats()
    best_policy = bandit.get_best_policy()
    avg_reward = sum(e.reward for e in episodes) / len(episodes) if episodes else 0
    avg_regret = cumulative_regret / len(episodes) if episodes else 0
    
    total_time = time.time() - start_time
    
    # Gerar conclusão
    conclusion = f"Melhor política: {best_policy} (avg reward: {bandit.get_avg_reward(best_policy):.4f}). "
    conclusion += f"Regret cumulativo: {cumulative_regret:.4f} em {request.num_episodes} episódios."
    
    # Atualizar experiência
    update_experiment_status(
        experiment.id,
        ExperimentStatus.FINISHED,
        summary={
            "best_policy": best_policy,
            "avg_reward": round(avg_reward, 4),
            "cumulative_regret": round(cumulative_regret, 4),
            "policy_stats": [s.dict() for s in policy_stats],
        },
        kpis={
            "avg_reward": round(avg_reward, 4),
            "avg_regret": round(avg_regret, 4),
            "cumulative_regret": round(cumulative_regret, 4),
        },
        conclusion=conclusion,
    )
    
    logger.info(f"WP4 Experiment finished: {conclusion}")
    
    return WP4ExperimentResult(
        experiment_id=experiment.id,
        name=request.name,
        status="finished",
        num_episodes=request.num_episodes,
        policies=request.policies,
        baseline_policy=request.baseline_policy,
        episodes=episodes,
        policy_stats=policy_stats,
        best_policy=best_policy,
        avg_reward=round(avg_reward, 4),
        avg_regret=round(avg_regret, 4),
        cumulative_regret=round(cumulative_regret, 4),
        conclusion=conclusion,
        total_time_sec=round(total_time, 2),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _create_demo_instance():
    """Cria instância de demonstração."""
    from scheduling.types import SchedulingInstance
    
    operations = []
    for i in range(20):
        operations.append({
            "operation_id": f"OP_{i:03d}",
            "order_id": f"ORD_{i//3:03d}",
            "article_id": f"ART_{i//5:02d}",
            "op_seq": i % 3,
            "op_code": f"OP_{i % 5}",
            "duration_min": 30 + (i * 5),
            "primary_machine_id": f"M-{(i % 5) + 300}",
            "due_date": datetime.now() + timedelta(hours=24 + i * 2),
        })
    
    machines = [{"machine_id": f"M-{i + 300}", "name": f"Machine {i}"} for i in range(5)]
    
    return SchedulingInstance(
        operations=operations,
        machines=machines,
        horizon_start=datetime.now(),
    )


def _vary_instance(instance, seed: int):
    """
    Varia ligeiramente a instância para simular diferentes dias.
    
    Adiciona variação aleatória nas durações e due dates.
    """
    import copy
    
    random.seed(seed)
    varied = copy.deepcopy(instance)
    
    for op in varied.operations:
        if "duration_min" in op:
            # Variar duração ±10%
            op["duration_min"] = op["duration_min"] * random.uniform(0.9, 1.1)
        if "due_date" in op:
            # Variar due date ±2 horas
            delta = timedelta(hours=random.uniform(-2, 2))
            op["due_date"] = op["due_date"] + delta
    
    return varied
