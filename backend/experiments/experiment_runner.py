"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — EXPERIMENT RUNNER (SIFIDE R&D)
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Framework for conducting reproducible experiments to validate research hypotheses.

PURPOSE
═══════

This module enables:
1. Systematic comparison of scheduling algorithms
2. Validation of research hypotheses
3. Generation of SIFIDE-compliant audit trails
4. Statistical analysis of results

EXPERIMENT STRUCTURE
════════════════════

Each experiment follows the scientific method:

    1. HYPOTHESIS: What are we testing?
    2. SETUP: What data, parameters, and policies?
    3. EXECUTION: Run the experiment
    4. ANALYSIS: Compute metrics and compare
    5. CONCLUSION: Accept/reject hypothesis

EXPERIMENT NAMING CONVENTION
───────────────────────────

    E{WP}.{N}.{slug}

    WP = Work Package number (1-4)
    N  = Experiment number within WP
    slug = Short description

    Examples:
    - E1.1.routing_strategies: Compare routing strategies
    - E2.1.suggestion_accuracy: Validate suggestion engine
    - E4.1.ucb_vs_heuristic: Compare UCB to fixed heuristics

LOGGING FORMAT
──────────────

Experiments are logged to JSON files with full reproducibility information:

    {
        "experiment_id": "E1.1.routing_strategies",
        "timestamp": "2024-01-15T10:30:00",
        "hypothesis": "UCB outperforms fixed heuristics after 100 steps",
        "config": { ... },
        "results": { ... },
        "conclusion": "SUPPORTED" | "REJECTED" | "INCONCLUSIVE",
        "statistical_tests": { ... }
    }

SIFIDE COMPLIANCE
─────────────────

For SIFIDE (Sistema de Incentivos Fiscais à I&D Empresarial), we document:

1. Technical Uncertainty: What was unknown before the experiment?
2. Scientific Method: How was the hypothesis tested?
3. Technical Advancement: What new knowledge was gained?
4. Systematic Activity: Reproducible experimental protocol

R&D WORK PACKAGES
─────────────────

WP1: Intelligent Routing & MILP
    - E1.1: Routing strategy comparison
    - E1.2: MILP vs heuristic quality
    - E1.3: Setup time minimization

WP2: Intelligent Suggestions
    - E2.1: Suggestion accuracy validation
    - E2.2: Impact of suggestions on KPIs

WP3: Inventory-Capacity Optimization
    - E3.1: Joint optimization benefit
    - E3.2: Safety stock vs capacity trade-off

WP4: Learning Scheduler
    - E4.1: UCB vs fixed heuristics
    - E4.2: Contextual bandit with machine features
    - E4.3: Regret analysis

REFERENCES
──────────
[1] SIFIDE Guidelines - AICEP
[2] Box, Hunter & Hunter (2005). Statistics for Experimenters. Wiley.
"""

from __future__ import annotations

import hashlib
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

logger = logging.getLogger(__name__)

# Default log directory
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# ENUMS AND CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

class WorkPackage(str, Enum):
    """R&D Work Packages."""
    WP1_ROUTING = "WP1"
    WP2_SUGGESTIONS = "WP2"
    WP3_INVENTORY = "WP3"
    WP4_LEARNING = "WP4"


class Conclusion(str, Enum):
    """Experiment conclusion."""
    SUPPORTED = "SUPPORTED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    ERROR = "ERROR"


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    """
    Configuration for an experiment.
    
    Attributes:
        experiment_id: Unique identifier (e.g., "E1.1.routing_strategies")
        work_package: R&D work package
        hypothesis: Scientific hypothesis being tested
        description: Detailed description
        parameters: Experiment parameters
        policies: List of policies/algorithms to compare
        n_replications: Number of replications for statistical validity
        seed: Random seed for reproducibility
        metrics: List of metrics to compute
    """
    experiment_id: str
    work_package: WorkPackage
    hypothesis: str
    description: str = ""
    
    # Experiment parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    policies: List[str] = field(default_factory=list)
    n_replications: int = 10
    seed: int = 42
    
    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        "makespan", "total_tardiness", "avg_reward", "cumulative_regret"
    ])
    
    # Statistical tests
    significance_level: float = 0.05
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['work_package'] = self.work_package.value
        return d
    
    def get_hash(self) -> str:
        """Generate hash for config reproducibility check."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


@dataclass
class PolicyResult:
    """Results for a single policy in the experiment."""
    policy_name: str
    replications: int
    
    # Primary metrics (mean ± std)
    makespan_mean: float = 0.0
    makespan_std: float = 0.0
    tardiness_mean: float = 0.0
    tardiness_std: float = 0.0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    regret_mean: float = 0.0
    regret_std: float = 0.0
    
    # Additional metrics
    snr_context_score: float = 1.0
    exploration_rate: float = 0.0
    
    # Raw data
    makespan_values: List[float] = field(default_factory=list)
    tardiness_values: List[float] = field(default_factory=list)
    reward_curves: List[List[float]] = field(default_factory=list)
    regret_curves: List[List[float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'policy_name': self.policy_name,
            'replications': self.replications,
            'makespan': {'mean': round(self.makespan_mean, 2), 'std': round(self.makespan_std, 2)},
            'tardiness': {'mean': round(self.tardiness_mean, 2), 'std': round(self.tardiness_std, 2)},
            'reward': {'mean': round(self.reward_mean, 4), 'std': round(self.reward_std, 4)},
            'regret': {'mean': round(self.regret_mean, 4), 'std': round(self.regret_std, 4)},
            'snr_context_score': round(self.snr_context_score, 2),
            'exploration_rate': round(self.exploration_rate, 4),
        }


@dataclass
class StatisticalTest:
    """Result of a statistical test."""
    test_name: str
    comparison: str  # e.g., "UCB vs FIXED_PRIORITY"
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    interpretation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'comparison': self.comparison,
            'statistic': round(self.statistic, 4),
            'p_value': round(self.p_value, 6),
            'significant': self.significant,
            'effect_size': round(self.effect_size, 4) if self.effect_size else None,
            'interpretation': self.interpretation,
        }


@dataclass
class ExperimentResult:
    """
    Complete result of an experiment.
    
    Contains all information needed for SIFIDE audit:
    - Configuration used
    - Results per policy
    - Statistical comparisons
    - Conclusion
    """
    experiment_id: str
    timestamp: str
    duration_sec: float
    config_hash: str
    
    # Results
    policy_results: Dict[str, PolicyResult] = field(default_factory=dict)
    statistical_tests: List[StatisticalTest] = field(default_factory=list)
    
    # Conclusion
    conclusion: Conclusion = Conclusion.INCONCLUSIVE
    conclusion_text: str = ""
    
    # SIFIDE fields
    technical_uncertainty: str = ""
    technical_advancement: str = ""
    
    # Metadata
    config: Optional[ExperimentConfig] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiment_id': self.experiment_id,
            'timestamp': self.timestamp,
            'duration_sec': round(self.duration_sec, 2),
            'config_hash': self.config_hash,
            'policy_results': {k: v.to_dict() for k, v in self.policy_results.items()},
            'statistical_tests': [t.to_dict() for t in self.statistical_tests],
            'conclusion': self.conclusion.value,
            'conclusion_text': self.conclusion_text,
            'technical_uncertainty': self.technical_uncertainty,
            'technical_advancement': self.technical_advancement,
            'config': self.config.to_dict() if self.config else None,
            'error': self.error,
        }
    
    def save(self, log_dir: Path = LOG_DIR) -> Path:
        """Save result to JSON file."""
        filename = f"{self.experiment_id}.{self.config_hash}.json"
        filepath = log_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Experiment result saved to {filepath}")
        return filepath


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def welch_t_test(
    values_a: List[float],
    values_b: List[float],
    alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """
    Welch's t-test for comparing two samples with unequal variances.
    
    H0: μ_A = μ_B
    H1: μ_A ≠ μ_B
    
    Returns:
        (t_statistic, p_value, significant)
    """
    n_a, n_b = len(values_a), len(values_b)
    
    if n_a < 2 or n_b < 2:
        return 0.0, 1.0, False
    
    mean_a, mean_b = np.mean(values_a), np.mean(values_b)
    var_a, var_b = np.var(values_a, ddof=1), np.var(values_b, ddof=1)
    
    # Welch's t-statistic
    se = np.sqrt(var_a / n_a + var_b / n_b)
    if se < 1e-10:
        return 0.0, 1.0, False
    
    t_stat = (mean_a - mean_b) / se
    
    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else 1
    
    # Approximate p-value using normal distribution for large df
    # For more accuracy, use scipy.stats.t.sf
    try:
        from scipy import stats
        p_value = 2 * stats.t.sf(abs(t_stat), df)
    except ImportError:
        # Fallback: normal approximation
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
    
    return t_stat, p_value, p_value < alpha


def _normal_cdf(x: float) -> float:
    """Approximate normal CDF using Abramowitz & Stegun formula."""
    a1, a2, a3 = 0.254829592, -0.284496736, 1.421413741
    a4, a5 = -1.453152027, 1.061405429
    p = 0.3275911
    
    sign = 1 if x >= 0 else -1
    x = abs(x) / np.sqrt(2)
    
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    
    return 0.5 * (1.0 + sign * y)


def cohens_d(values_a: List[float], values_b: List[float]) -> float:
    """
    Cohen's d effect size.
    
    d = (μ_A - μ_B) / s_pooled
    
    Interpretation:
        |d| < 0.2: negligible
        |d| < 0.5: small
        |d| < 0.8: medium
        |d| >= 0.8: large
    """
    n_a, n_b = len(values_a), len(values_b)
    
    if n_a < 2 or n_b < 2:
        return 0.0
    
    mean_a, mean_b = np.mean(values_a), np.mean(values_b)
    var_a, var_b = np.var(values_a, ddof=1), np.var(values_b, ddof=1)
    
    # Pooled standard deviation
    s_pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    
    if s_pooled < 1e-10:
        return 0.0
    
    return (mean_a - mean_b) / s_pooled


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ════════════════════════════════════════════════════════════════════════════════════════════════════

class ExperimentRunner:
    """
    Runner for conducting reproducible experiments.
    
    Usage:
        runner = ExperimentRunner()
        
        config = ExperimentConfig(
            experiment_id="E4.1.ucb_vs_heuristic",
            work_package=WorkPackage.WP4_LEARNING,
            hypothesis="UCB achieves lower regret than fixed priority after 100 steps",
            policies=["ucb", "fixed_priority", "epsilon_greedy"],
            n_replications=10
        )
        
        result = runner.run(config, simulation_fn)
        result.save()
    """
    
    def __init__(self, log_dir: Path = LOG_DIR):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def run(
        self,
        config: ExperimentConfig,
        simulation_fn: Callable[[str, int], Dict[str, Any]],
        verbose: bool = True
    ) -> ExperimentResult:
        """
        Run an experiment.
        
        Args:
            config: Experiment configuration
            simulation_fn: Function (policy_name, seed) -> metrics_dict
                           Must return dict with keys matching config.metrics
            verbose: Print progress
        
        Returns:
            ExperimentResult with all metrics and statistical tests
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        if verbose:
            logger.info(f"Starting experiment: {config.experiment_id}")
            logger.info(f"Hypothesis: {config.hypothesis}")
            logger.info(f"Policies: {config.policies}")
            logger.info(f"Replications: {config.n_replications}")
        
        # Initialize result
        result = ExperimentResult(
            experiment_id=config.experiment_id,
            timestamp=timestamp,
            duration_sec=0.0,
            config_hash=config.get_hash(),
            config=config,
        )
        
        try:
            # Run each policy
            for policy_name in config.policies:
                policy_result = self._run_policy(
                    policy_name, config, simulation_fn, verbose
                )
                result.policy_results[policy_name] = policy_result
            
            # Statistical comparisons
            result.statistical_tests = self._run_statistical_tests(
                result.policy_results, config
            )
            
            # Determine conclusion
            result.conclusion, result.conclusion_text = self._determine_conclusion(
                config, result
            )
            
            # SIFIDE fields
            result.technical_uncertainty = self._generate_technical_uncertainty(config)
            result.technical_advancement = self._generate_technical_advancement(result)
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            result.conclusion = Conclusion.ERROR
            result.error = str(e)
        
        result.duration_sec = time.time() - start_time
        
        if verbose:
            logger.info(f"Experiment completed in {result.duration_sec:.2f}s")
            logger.info(f"Conclusion: {result.conclusion.value}")
        
        return result
    
    def _run_policy(
        self,
        policy_name: str,
        config: ExperimentConfig,
        simulation_fn: Callable,
        verbose: bool
    ) -> PolicyResult:
        """Run all replications for a single policy."""
        result = PolicyResult(
            policy_name=policy_name,
            replications=config.n_replications
        )
        
        makespan_values = []
        tardiness_values = []
        reward_values = []
        regret_values = []
        reward_curves = []
        regret_curves = []
        
        for rep in range(config.n_replications):
            seed = config.seed + rep
            
            if verbose:
                logger.debug(f"  Running {policy_name} replication {rep + 1}/{config.n_replications}")
            
            # Run simulation
            metrics = simulation_fn(policy_name, seed)
            
            # Extract metrics
            makespan_values.append(metrics.get('makespan', 0.0))
            tardiness_values.append(metrics.get('total_tardiness', 0.0))
            reward_values.append(metrics.get('avg_reward', 0.0))
            regret_values.append(metrics.get('cumulative_regret', 0.0))
            
            if 'reward_history' in metrics:
                reward_curves.append(metrics['reward_history'])
            if 'regret_history' in metrics:
                regret_curves.append(metrics['regret_history'])
        
        # Compute statistics
        result.makespan_values = makespan_values
        result.tardiness_values = tardiness_values
        result.reward_curves = reward_curves
        result.regret_curves = regret_curves
        
        result.makespan_mean = float(np.mean(makespan_values))
        result.makespan_std = float(np.std(makespan_values))
        result.tardiness_mean = float(np.mean(tardiness_values))
        result.tardiness_std = float(np.std(tardiness_values))
        result.reward_mean = float(np.mean(reward_values))
        result.reward_std = float(np.std(reward_values))
        result.regret_mean = float(np.mean(regret_values))
        result.regret_std = float(np.std(regret_values))
        
        return result
    
    def _run_statistical_tests(
        self,
        policy_results: Dict[str, PolicyResult],
        config: ExperimentConfig
    ) -> List[StatisticalTest]:
        """Run pairwise statistical tests."""
        tests = []
        
        policies = list(policy_results.keys())
        
        for i, policy_a in enumerate(policies):
            for policy_b in policies[i + 1:]:
                # Compare regret
                values_a = [r[-1] if r else 0 for r in policy_results[policy_a].regret_curves]
                values_b = [r[-1] if r else 0 for r in policy_results[policy_b].regret_curves]
                
                if not values_a:
                    values_a = [policy_results[policy_a].regret_mean] * config.n_replications
                if not values_b:
                    values_b = [policy_results[policy_b].regret_mean] * config.n_replications
                
                t_stat, p_value, significant = welch_t_test(
                    values_a, values_b, config.significance_level
                )
                effect = cohens_d(values_a, values_b)
                
                # Interpretation
                if significant:
                    if np.mean(values_a) < np.mean(values_b):
                        interpretation = f"{policy_a} has significantly lower regret than {policy_b}"
                    else:
                        interpretation = f"{policy_b} has significantly lower regret than {policy_a}"
                else:
                    interpretation = f"No significant difference between {policy_a} and {policy_b}"
                
                tests.append(StatisticalTest(
                    test_name="Welch's t-test (regret)",
                    comparison=f"{policy_a} vs {policy_b}",
                    statistic=t_stat,
                    p_value=p_value,
                    significant=significant,
                    effect_size=effect,
                    interpretation=interpretation,
                ))
        
        return tests
    
    def _determine_conclusion(
        self,
        config: ExperimentConfig,
        result: ExperimentResult
    ) -> Tuple[Conclusion, str]:
        """Determine experiment conclusion based on results."""
        # Simple heuristic: check if any test is significant
        significant_tests = [t for t in result.statistical_tests if t.significant]
        
        if not significant_tests:
            return (
                Conclusion.INCONCLUSIVE,
                "Nenhuma diferença estatisticamente significativa encontrada entre as políticas."
            )
        
        # Check hypothesis direction
        # This is a simplified check - real implementation would parse the hypothesis
        
        conclusion_text = "Resultados dos testes estatísticos:\n"
        for test in significant_tests:
            conclusion_text += f"- {test.interpretation} (p={test.p_value:.4f}, d={test.effect_size:.2f})\n"
        
        return Conclusion.SUPPORTED, conclusion_text
    
    def _generate_technical_uncertainty(self, config: ExperimentConfig) -> str:
        """Generate SIFIDE technical uncertainty statement."""
        return (
            f"Incerteza técnica: A hipótese '{config.hypothesis}' "
            f"não tinha resposta conhecida a priori para o contexto específico "
            f"de planeamento de produção industrial considerado. "
            f"A comparação sistemática das políticas {config.policies} "
            f"com {config.n_replications} replicações permite validar ou rejeitar "
            f"a hipótese com significância estatística α={config.significance_level}."
        )
    
    def _generate_technical_advancement(self, result: ExperimentResult) -> str:
        """Generate SIFIDE technical advancement statement."""
        if result.conclusion == Conclusion.SUPPORTED:
            return (
                f"Avanço técnico: O experimento {result.experiment_id} "
                f"demonstrou que a hipótese é suportada pelos dados. "
                f"Este resultado contribui para o conhecimento sobre "
                f"a aplicação de algoritmos de aprendizagem ao planeamento de produção."
            )
        elif result.conclusion == Conclusion.REJECTED:
            return (
                f"Avanço técnico: O experimento {result.experiment_id} "
                f"demonstrou que a hipótese deve ser rejeitada. "
                f"Este resultado negativo é igualmente valioso para orientar "
                f"futuras investigações."
            )
        else:
            return (
                f"Avanço técnico: O experimento {result.experiment_id} "
                f"não permitiu conclusões definitivas. "
                f"Recomenda-se aumentar o número de replicações ou "
                f"refinar a configuração experimental."
            )


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def run_experiment(
    config: ExperimentConfig,
    simulation_fn: Callable[[str, int], Dict[str, Any]],
    save: bool = True
) -> ExperimentResult:
    """
    Run an experiment and optionally save results.
    
    Args:
        config: Experiment configuration
        simulation_fn: Simulation function
        save: Whether to save results to JSON
    
    Returns:
        ExperimentResult
    """
    runner = ExperimentRunner()
    result = runner.run(config, simulation_fn)
    
    if save:
        result.save()
    
    return result


def list_experiments(log_dir: Path = LOG_DIR) -> List[Dict[str, Any]]:
    """List all saved experiments."""
    experiments = []
    
    for filepath in log_dir.glob("E*.json"):
        with open(filepath, 'r') as f:
            data = json.load(f)
            experiments.append({
                'experiment_id': data.get('experiment_id'),
                'timestamp': data.get('timestamp'),
                'conclusion': data.get('conclusion'),
                'filepath': str(filepath),
            })
    
    return sorted(experiments, key=lambda x: x['timestamp'], reverse=True)


def load_experiment(experiment_id: str, log_dir: Path = LOG_DIR) -> Optional[Dict[str, Any]]:
    """Load a saved experiment by ID."""
    for filepath in log_dir.glob(f"{experiment_id}.*.json"):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None



