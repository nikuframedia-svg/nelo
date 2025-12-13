"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    SETUP OPTIMIZER — Sequence-Dependent Setup Minimization
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Optimizes operation sequences to minimize setup/changeover times.

Mathematical Model (TSP-like):
─────────────────────────────────────────────────────────────────────────────────────────────────────

Given:
    - n operations with setup families
    - Setup matrix S[i][j] = time to change from family i to family j
    
Objective:
    Minimize total setup time while respecting due date constraints
    
    min Σ S[f(i), f(i+1)]  for consecutive operations i, i+1
    
Subject to:
    - Due date constraints (soft or hard)
    - Machine capacity

Algorithms:
    1. Greedy nearest neighbor
    2. 2-opt local search
    3. Genetic algorithm (for larger instances)

═══════════════════════════════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import logging
import random
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SetupMatrix:
    """
    Setup time matrix between product families/types.
    """
    families: List[str]
    matrix: Dict[str, Dict[str, float]]  # from_family -> to_family -> time
    
    def get_time(self, from_family: str, to_family: str) -> float:
        """Get setup time between two families."""
        if from_family == to_family:
            return 0.0
        return self.matrix.get(from_family, {}).get(to_family, 0.0)
    
    @classmethod
    def from_dataframe(cls, df) -> 'SetupMatrix':
        """
        Build SetupMatrix from DataFrame.
        Expected columns: from_setup_family, to_setup_family, setup_time_min
        """
        families = list(set(df['from_setup_family'].tolist() + df['to_setup_family'].tolist()))
        matrix = {}
        
        for _, row in df.iterrows():
            from_f = row['from_setup_family']
            to_f = row['to_setup_family']
            time = row['setup_time_min']
            
            if from_f not in matrix:
                matrix[from_f] = {}
            matrix[from_f][to_f] = time
        
        return cls(families=families, matrix=matrix)


@dataclass 
class SequenceResult:
    """Result of sequence optimization."""
    sequence: List[str]  # Ordered list of operation IDs
    total_setup_time: float
    setup_savings: float  # Compared to original
    iterations: int
    algorithm: str


class SetupOptimizer:
    """
    Optimizer for sequence-dependent setup times.
    """
    
    def __init__(self, setup_matrix: SetupMatrix):
        self.setup_matrix = setup_matrix
    
    def optimize_sequence(
        self,
        operations: List[Dict[str, Any]],
        respect_due_dates: bool = True,
        max_tardiness_min: float = 60.0,
        algorithm: str = "greedy_2opt",
    ) -> SequenceResult:
        """
        Optimize operation sequence to minimize setup times.
        
        Args:
            operations: List of operations with 'op_id', 'setup_family', 'due_date', 'duration_min'
            respect_due_dates: If True, penalize tardiness
            max_tardiness_min: Maximum allowed tardiness per operation
            algorithm: 'greedy', 'greedy_2opt', 'genetic'
            
        Returns:
            SequenceResult with optimized sequence
        """
        if not operations:
            return SequenceResult(sequence=[], total_setup_time=0, setup_savings=0, iterations=0, algorithm=algorithm)
        
        # Calculate original setup time
        original_setup = self._calculate_total_setup(
            [op['op_id'] for op in operations],
            {op['op_id']: op['setup_family'] for op in operations}
        )
        
        if algorithm == "greedy":
            result = self._optimize_greedy(operations)
        elif algorithm == "greedy_2opt":
            result = self._optimize_greedy_2opt(operations, respect_due_dates, max_tardiness_min)
        elif algorithm == "genetic":
            result = self._optimize_genetic(operations, respect_due_dates, max_tardiness_min)
        else:
            result = self._optimize_greedy(operations)
        
        result.setup_savings = original_setup - result.total_setup_time
        return result
    
    def _calculate_total_setup(
        self,
        sequence: List[str],
        family_map: Dict[str, str],
    ) -> float:
        """Calculate total setup time for a sequence."""
        if len(sequence) <= 1:
            return 0.0
        
        total = 0.0
        for i in range(len(sequence) - 1):
            from_family = family_map.get(sequence[i], "default")
            to_family = family_map.get(sequence[i + 1], "default")
            total += self.setup_matrix.get_time(from_family, to_family)
        
        return total
    
    def _optimize_greedy(self, operations: List[Dict[str, Any]]) -> SequenceResult:
        """
        Greedy nearest neighbor algorithm.
        
        Starting from first operation, always pick the next operation
        with minimum setup time.
        """
        if not operations:
            return SequenceResult(sequence=[], total_setup_time=0, setup_savings=0, iterations=0, algorithm="greedy")
        
        ops_map = {op['op_id']: op for op in operations}
        remaining = set(ops_map.keys())
        
        # Start with operation with earliest due date
        first_op = min(operations, key=lambda op: op.get('due_date', datetime.max))
        sequence = [first_op['op_id']]
        remaining.remove(first_op['op_id'])
        
        while remaining:
            current_family = ops_map[sequence[-1]]['setup_family']
            
            # Find operation with minimum setup from current
            next_op_id = min(
                remaining,
                key=lambda op_id: self.setup_matrix.get_time(
                    current_family,
                    ops_map[op_id]['setup_family']
                )
            )
            
            sequence.append(next_op_id)
            remaining.remove(next_op_id)
        
        family_map = {op['op_id']: op['setup_family'] for op in operations}
        total_setup = self._calculate_total_setup(sequence, family_map)
        
        return SequenceResult(
            sequence=sequence,
            total_setup_time=total_setup,
            setup_savings=0,
            iterations=len(operations),
            algorithm="greedy",
        )
    
    def _optimize_greedy_2opt(
        self,
        operations: List[Dict[str, Any]],
        respect_due_dates: bool,
        max_tardiness_min: float,
    ) -> SequenceResult:
        """
        Greedy + 2-opt local search.
        """
        # Start with greedy solution
        greedy_result = self._optimize_greedy(operations)
        sequence = greedy_result.sequence.copy()
        
        ops_map = {op['op_id']: op for op in operations}
        family_map = {op['op_id']: op['setup_family'] for op in operations}
        
        # 2-opt improvement
        improved = True
        iterations = 0
        max_iterations = len(operations) * 10
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for i in range(len(sequence) - 2):
                for j in range(i + 2, len(sequence)):
                    # Try reversing segment [i+1, j]
                    new_sequence = sequence[:i+1] + sequence[i+1:j+1][::-1] + sequence[j+1:]
                    
                    new_setup = self._calculate_total_setup(new_sequence, family_map)
                    current_setup = self._calculate_total_setup(sequence, family_map)
                    
                    if new_setup < current_setup:
                        # Check due date feasibility if required
                        if respect_due_dates:
                            max_tard = self._calculate_max_tardiness(new_sequence, ops_map)
                            if max_tard > max_tardiness_min:
                                continue
                        
                        sequence = new_sequence
                        improved = True
                        break
                
                if improved:
                    break
        
        total_setup = self._calculate_total_setup(sequence, family_map)
        
        return SequenceResult(
            sequence=sequence,
            total_setup_time=total_setup,
            setup_savings=0,
            iterations=iterations,
            algorithm="greedy_2opt",
        )
    
    def _optimize_genetic(
        self,
        operations: List[Dict[str, Any]],
        respect_due_dates: bool,
        max_tardiness_min: float,
        population_size: int = 50,
        generations: int = 100,
    ) -> SequenceResult:
        """
        Genetic algorithm for larger instances.
        """
        if len(operations) <= 10:
            # For small instances, 2-opt is faster
            return self._optimize_greedy_2opt(operations, respect_due_dates, max_tardiness_min)
        
        ops_map = {op['op_id']: op for op in operations}
        family_map = {op['op_id']: op['setup_family'] for op in operations}
        op_ids = list(ops_map.keys())
        
        def fitness(sequence: List[str]) -> float:
            setup = self._calculate_total_setup(sequence, family_map)
            if respect_due_dates:
                tard = self._calculate_max_tardiness(sequence, ops_map)
                return setup + tard * 10  # Penalize tardiness
            return setup
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = op_ids.copy()
            random.shuffle(individual)
            population.append(individual)
        
        # Also add greedy solution
        greedy = self._optimize_greedy(operations)
        population[0] = greedy.sequence
        
        # Evolution
        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = [(ind, fitness(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1])
            
            # Selection (top 50%)
            survivors = [ind for ind, _ in fitness_scores[:population_size // 2]]
            
            # Crossover
            children = []
            while len(children) < population_size - len(survivors):
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                child = self._order_crossover(parent1, parent2)
                children.append(child)
            
            # Mutation
            for child in children:
                if random.random() < 0.1:  # 10% mutation rate
                    i, j = random.sample(range(len(child)), 2)
                    child[i], child[j] = child[j], child[i]
            
            population = survivors + children
        
        # Return best
        best = min(population, key=fitness)
        total_setup = self._calculate_total_setup(best, family_map)
        
        return SequenceResult(
            sequence=best,
            total_setup_time=total_setup,
            setup_savings=0,
            iterations=generations,
            algorithm="genetic",
        )
    
    def _order_crossover(self, parent1: List[str], parent2: List[str]) -> List[str]:
        """Order crossover (OX) for permutation encoding."""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [None] * size
        child[start:end] = parent1[start:end]
        
        remaining = [x for x in parent2 if x not in child]
        
        pos = 0
        for i in range(size):
            if child[i] is None:
                child[i] = remaining[pos]
                pos += 1
        
        return child
    
    def _calculate_max_tardiness(
        self,
        sequence: List[str],
        ops_map: Dict[str, Dict[str, Any]],
        start_time: Optional[datetime] = None,
    ) -> float:
        """Calculate maximum tardiness for a sequence."""
        if not sequence:
            return 0.0
        
        if start_time is None:
            start_time = datetime.now()
        
        current_time = start_time
        max_tard = 0.0
        prev_family = None
        
        for op_id in sequence:
            op = ops_map[op_id]
            
            # Setup time
            setup = 0.0
            if prev_family and op['setup_family'] != prev_family:
                setup = self.setup_matrix.get_time(prev_family, op['setup_family'])
            
            current_time += timedelta(minutes=setup + op['duration_min'])
            
            due_date = op.get('due_date', datetime.max)
            if current_time > due_date:
                tard = (current_time - due_date).total_seconds() / 60
                max_tard = max(max_tard, tard)
            
            prev_family = op['setup_family']
        
        return max_tard



