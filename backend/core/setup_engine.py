"""
═══════════════════════════════════════════════════════════════════════════════
                   PRODPLAN 4.0 - SETUP TIME ENGINE
═══════════════════════════════════════════════════════════════════════════════

This module provides setup time computation and prediction with
Signal-to-Noise Ratio (SNR) based confidence estimation.

Mathematical Foundation
═══════════════════════

Setup Time Model:
─────────────────
The setup time between consecutive operations depends on:

    S(o₁, o₂) = f(family(o₁), family(o₂), machine, context)

For the sequence-dependent case:

    S_{i→j} = s_{f(i), f(j)}  where f(·) returns the setup family

Sequence Setup Minimization:
───────────────────────────
Given n operations to sequence on a machine, minimize:

    min_π  Σᵢ₌₁ⁿ⁻¹ S(π(i), π(i+1))

This is equivalent to the Traveling Salesman Problem (TSP) on the
setup time graph, which is NP-hard in general.

SNR for Setup Prediction:
────────────────────────
When predicting setup times from historical data:

    SNR = Var(E[S|X]) / Var(S - E[S|X])
        = Var(systematic) / Var(residual)

where X = (from_family, to_family, machine, ...) are predictors.

High SNR indicates that setup times are predictable from the family transition.
Low SNR indicates high variability - setup times depend on unmeasured factors.

Heuristics for Sequence Construction:
────────────────────────────────────
1. Nearest Neighbor: At each step, choose the next operation with
   minimum setup from the current one.
   
2. Savings Heuristic: Compute savings s_ij = S(i,depot) + S(depot,j) - S(i,j)
   and merge operations with highest savings.

3. Christofides (for metric TSP): Guarantees 3/2-approximation.

R&D / SIFIDE: WP1 - Setup Optimization
Research Questions:
- Q1.4: Can ML-based setup prediction reduce actual vs planned variance?
- Q1.5: Impact of batching on total setup time?

References:
[1] Allahverdi et al. (2008). A survey of scheduling problems with setup times.
[2] Gupta & Stafford (2006). Flowshop scheduling research after five decades.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SetupPrediction:
    """
    Result of a setup time prediction.
    
    Attributes:
        value: Predicted setup time in minutes
        snr: Signal-to-Noise Ratio for this prediction
        confidence: Confidence score derived from SNR
        source: How the prediction was made ('matrix', 'ml', 'default')
        features_used: Features used in prediction
    """
    value: float
    snr: float = 1.0
    confidence: float = 0.5
    source: str = "default"
    features_used: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'value_min': round(self.value, 2),
            'snr': round(self.snr, 2),
            'confidence': round(self.confidence, 3),
            'source': self.source,
            'features_used': self.features_used,
        }


@dataclass
class SequenceSetupResult:
    """
    Result of computing setup times for a sequence of operations.
    """
    total_setup_min: float
    setup_events: List[Dict[str, Any]]
    avg_setup_min: float
    setup_count: int
    avg_snr: float
    avg_confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_setup_min': round(self.total_setup_min, 2),
            'total_setup_hours': round(self.total_setup_min / 60, 2),
            'setup_count': self.setup_count,
            'avg_setup_min': round(self.avg_setup_min, 2),
            'avg_snr': round(self.avg_snr, 2),
            'avg_confidence': round(self.avg_confidence, 3),
        }


# ════════════════════════════════════════════════════════════════════════════
# SETUP TIME COMPUTATION
# ════════════════════════════════════════════════════════════════════════════

class SetupEngine:
    """
    Engine for computing and predicting setup times.
    
    Supports multiple prediction methods:
    1. Matrix-based: Direct lookup from setup_matrix[from_family, to_family]
    2. ML-based: Learned from historical data (TODO)
    3. Default: Fallback value when no data available
    
    SNR is computed when historical data is available to estimate
    the reliability of predictions.
    
    Usage:
        engine = SetupEngine(setup_matrix)
        pred = engine.compute_setup_time('FAM-A', 'FAM-B')
        print(f"Setup: {pred.value:.1f} min (SNR={pred.snr:.1f})")
    
    Mathematical Basis:
    ──────────────────
    The setup time S_{ij} from family i to family j is modeled as:
    
        S_{ij} = μ_{ij} + ε
    
    where:
        μ_{ij} = E[S|from=i, to=j] (systematic component)
        ε ~ N(0, σ²)                (random noise)
    
    The SNR is:
        SNR = Var(μ) / Var(ε)
    
    With only matrix data (no historical variance), we estimate SNR=1
    as a conservative default.
    """
    
    def __init__(
        self,
        setup_matrix: Optional[Dict[Tuple[str, str], float]] = None,
        default_setup_min: float = 15.0,
        same_family_setup_min: float = 5.0
    ):
        """
        Initialize SetupEngine.
        
        Args:
            setup_matrix: Dict (from_family, to_family) -> setup_time_min
            default_setup_min: Default setup when matrix entry missing
            same_family_setup_min: Setup time when families are identical
        """
        self.setup_matrix = setup_matrix or {}
        self.default_setup_min = default_setup_min
        self.same_family_setup_min = same_family_setup_min
        
        # Historical data for SNR estimation
        self._historical: Optional[pd.DataFrame] = None
        self._snr_by_transition: Dict[Tuple[str, str], float] = {}
        self._variance_by_transition: Dict[Tuple[str, str], float] = {}
    
    @classmethod
    def from_dataframe(
        cls,
        setup_df: pd.DataFrame,
        from_col: str = 'from_setup_family',
        to_col: str = 'to_setup_family',
        time_col: str = 'setup_time_min'
    ) -> 'SetupEngine':
        """
        Create SetupEngine from DataFrame.
        
        Args:
            setup_df: DataFrame with setup matrix entries
            from_col: Column name for source family
            to_col: Column name for destination family
            time_col: Column name for setup time
        
        Returns:
            Configured SetupEngine
        """
        setup_matrix = {}
        for _, row in setup_df.iterrows():
            key = (str(row[from_col]), str(row[to_col]))
            setup_matrix[key] = float(row[time_col])
        
        return cls(setup_matrix)
    
    def load_historical(self, historical_df: pd.DataFrame) -> None:
        """
        Load historical setup data for SNR estimation.
        
        Expected columns:
        - from_family: Setup family before changeover
        - to_family: Setup family after changeover  
        - actual_setup_min: Actual observed setup time
        - machine_id (optional): For machine-specific SNR
        
        This enables variance estimation and more accurate SNR.
        """
        self._historical = historical_df
        self._compute_snr_from_historical()
    
    def _compute_snr_from_historical(self) -> None:
        """
        Compute SNR for each family transition from historical data.
        
        SNR = Var(group_mean) / Var(residual)
            = MSB / MSW  (Between-group / Within-group mean squares)
        """
        if self._historical is None or self._historical.empty:
            return
        
        df = self._historical
        
        if 'from_family' not in df.columns or 'to_family' not in df.columns:
            logger.warning("Historical data missing required columns")
            return
        
        if 'actual_setup_min' not in df.columns:
            logger.warning("Historical data missing actual_setup_min column")
            return
        
        # Compute variance per transition
        grouped = df.groupby(['from_family', 'to_family'])['actual_setup_min']
        
        for (from_f, to_f), group in grouped:
            key = (from_f, to_f)
            
            if len(group) >= 3:
                mean = group.mean()
                variance = group.var()
                
                self._variance_by_transition[key] = variance
                
                # Simple SNR estimation using coefficient of variation
                # SNR ≈ (μ/σ)²
                if variance > 0:
                    self._snr_by_transition[key] = (mean ** 2) / variance
                else:
                    self._snr_by_transition[key] = float('inf')
        
        logger.info(
            f"Computed SNR for {len(self._snr_by_transition)} transitions from historical data"
        )
    
    def compute_setup_time(
        self,
        from_family: str,
        to_family: str,
        machine_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> SetupPrediction:
        """
        Compute setup time for a family transition.
        
        Mathematical Model:
        ──────────────────
        S(i→j) = {
            s_{same}           if i = j  (same family)
            s_{ij}             if (i,j) ∈ Matrix
            s_{default}        otherwise
        }
        
        SNR Estimation:
        ──────────────
        - If historical data: SNR from variance decomposition
        - If matrix only: SNR = 1.0 (conservative estimate)
        - If default: SNR = 0.5 (high uncertainty)
        
        Args:
            from_family: Source setup family
            to_family: Destination setup family
            machine_id: Machine identifier (for machine-specific prediction)
            context: Additional context features
        
        Returns:
            SetupPrediction with value, SNR, and confidence
        """
        features_used = ['from_family', 'to_family']
        
        # Case 1: Same family - minimal setup
        if from_family == to_family:
            return SetupPrediction(
                value=self.same_family_setup_min,
                snr=10.0,  # High confidence for same-family
                confidence=0.95,
                source='same_family',
                features_used=features_used,
            )
        
        key = (from_family, to_family)
        
        # Case 2: Matrix lookup
        if key in self.setup_matrix:
            value = self.setup_matrix[key]
            
            # Get SNR from historical if available
            snr = self._snr_by_transition.get(key, 1.0)
            confidence = snr / (1 + snr)  # Convert SNR to R²-like confidence
            
            return SetupPrediction(
                value=value,
                snr=snr,
                confidence=confidence,
                source='matrix',
                features_used=features_used,
            )
        
        # Case 3: Try reverse lookup (symmetric matrix)
        rev_key = (to_family, from_family)
        if rev_key in self.setup_matrix:
            value = self.setup_matrix[rev_key]
            snr = self._snr_by_transition.get(rev_key, 0.8)  # Slightly lower for reverse
            confidence = snr / (1 + snr)
            
            return SetupPrediction(
                value=value,
                snr=snr,
                confidence=confidence,
                source='matrix_reverse',
                features_used=features_used,
            )
        
        # Case 4: Default fallback
        return SetupPrediction(
            value=self.default_setup_min,
            snr=0.5,  # Low confidence for default
            confidence=0.33,
            source='default',
            features_used=features_used,
        )
    
    def compute_sequence_setup_time(
        self,
        operations: List[Dict[str, Any]],
        family_key: str = 'setup_family'
    ) -> SequenceSetupResult:
        """
        Compute total setup time for a sequence of operations.
        
        Mathematical Formulation:
        ────────────────────────
        Given sequence π = (o₁, o₂, ..., oₙ):
        
            TotalSetup(π) = Σᵢ₌₁ⁿ⁻¹ S(oᵢ, oᵢ₊₁)
        
        where S(oᵢ, oⱼ) = setup_time(family(oᵢ), family(oⱼ))
        
        Args:
            operations: List of operation dicts, each containing family_key
            family_key: Key in operation dict for setup family
        
        Returns:
            SequenceSetupResult with total and breakdown
        """
        if len(operations) < 2:
            return SequenceSetupResult(
                total_setup_min=0.0,
                setup_events=[],
                avg_setup_min=0.0,
                setup_count=0,
                avg_snr=1.0,
                avg_confidence=1.0,
            )
        
        setup_events = []
        total_setup = 0.0
        snr_sum = 0.0
        confidence_sum = 0.0
        
        for i in range(len(operations) - 1):
            from_op = operations[i]
            to_op = operations[i + 1]
            
            from_family = str(from_op.get(family_key, 'default'))
            to_family = str(to_op.get(family_key, 'default'))
            
            pred = self.compute_setup_time(from_family, to_family)
            
            if from_family != to_family:
                setup_events.append({
                    'from_op': from_op.get('id', f'op_{i}'),
                    'to_op': to_op.get('id', f'op_{i+1}'),
                    'from_family': from_family,
                    'to_family': to_family,
                    'setup_min': pred.value,
                    'snr': pred.snr,
                    'confidence': pred.confidence,
                })
                
                total_setup += pred.value
                snr_sum += pred.snr
                confidence_sum += pred.confidence
        
        n_setups = len(setup_events)
        
        return SequenceSetupResult(
            total_setup_min=total_setup,
            setup_events=setup_events,
            avg_setup_min=total_setup / n_setups if n_setups > 0 else 0.0,
            setup_count=n_setups,
            avg_snr=snr_sum / n_setups if n_setups > 0 else 1.0,
            avg_confidence=confidence_sum / n_setups if n_setups > 0 else 1.0,
        )
    
    def optimize_sequence_greedy(
        self,
        operations: List[Dict[str, Any]],
        family_key: str = 'setup_family'
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Optimize operation sequence using Nearest Neighbor heuristic.
        
        Algorithm:
        ─────────
        1. Start with first operation
        2. At each step, choose next operation with minimum setup time
        3. Continue until all operations scheduled
        
        Complexity: O(n²)
        Quality: Typically 20-30% above optimal for random instances
        
        Args:
            operations: List of operations to sequence
            family_key: Key for setup family
        
        Returns:
            (optimized_sequence, total_setup_time)
        
        TODO[R&D]: Implement 2-opt local search for improvement
        TODO[R&D]: Compare with Christofides algorithm for larger instances
        """
        if len(operations) <= 1:
            return operations.copy(), 0.0
        
        remaining = operations.copy()
        sequence = [remaining.pop(0)]  # Start with first operation
        total_setup = 0.0
        
        while remaining:
            current = sequence[-1]
            current_family = str(current.get(family_key, 'default'))
            
            # Find operation with minimum setup from current
            best_idx = 0
            best_setup = float('inf')
            
            for i, op in enumerate(remaining):
                op_family = str(op.get(family_key, 'default'))
                pred = self.compute_setup_time(current_family, op_family)
                
                if pred.value < best_setup:
                    best_setup = pred.value
                    best_idx = i
            
            # Add best operation to sequence
            next_op = remaining.pop(best_idx)
            sequence.append(next_op)
            total_setup += best_setup
        
        return sequence, total_setup
    
    def estimate_setup_savings(
        self,
        operations: List[Dict[str, Any]],
        family_key: str = 'setup_family'
    ) -> Dict[str, float]:
        """
        Estimate potential setup time savings from optimization.
        
        Compares:
        - Current sequence setup time
        - Greedy-optimized sequence setup time
        - Lower bound (if we could batch by family)
        
        Returns dict with savings analysis.
        
        TODO[R&D]: Add statistical confidence interval on savings
        """
        # Current sequence
        current_result = self.compute_sequence_setup_time(operations, family_key)
        
        # Optimized sequence
        opt_seq, opt_setup = self.optimize_sequence_greedy(operations, family_key)
        
        # Lower bound: count unique family transitions needed
        families = [str(op.get(family_key, 'default')) for op in operations]
        unique_families = list(set(families))
        lb_transitions = max(0, len(unique_families) - 1)
        lb_setup = lb_transitions * self.same_family_setup_min
        
        savings_vs_current = current_result.total_setup_min - opt_setup
        savings_pct = (savings_vs_current / current_result.total_setup_min * 100 
                      if current_result.total_setup_min > 0 else 0)
        
        return {
            'current_setup_min': current_result.total_setup_min,
            'optimized_setup_min': opt_setup,
            'lower_bound_min': lb_setup,
            'savings_min': savings_vs_current,
            'savings_pct': round(savings_pct, 1),
            'gap_to_lb_min': opt_setup - lb_setup,
        }


# ════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def compute_setup_time(
    from_family: str,
    to_family: str,
    setup_matrix: Dict[Tuple[str, str], float],
    default: float = 15.0
) -> float:
    """
    Simple setup time lookup.
    
    For SNR-aware predictions, use SetupEngine.
    """
    if from_family == to_family:
        return 5.0  # Minimal same-family setup
    
    return setup_matrix.get((from_family, to_family), default)


def compute_sequence_setup(
    families: List[str],
    setup_matrix: Dict[Tuple[str, str], float],
    default: float = 15.0
) -> float:
    """
    Compute total setup time for a sequence of families.
    
    Args:
        families: List of setup families in sequence order
        setup_matrix: Setup time matrix
        default: Default setup time
    
    Returns:
        Total setup time in minutes
    """
    total = 0.0
    for i in range(len(families) - 1):
        total += compute_setup_time(families[i], families[i+1], setup_matrix, default)
    return total



