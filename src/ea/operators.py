"""
Genetic Operators for EA-HRL
Selection, Crossover, and Mutation operators.
"""

import torch as th
import numpy as np
from abc import ABC, abstractmethod
from .genome import AllocationGenome


# ==================== Selection Operators ====================

class SelectionOperator(ABC):
    """Base class for selection operators."""

    @abstractmethod
    def select(self, population: list) -> AllocationGenome:
        """Select a genome from the population."""
        raise NotImplementedError


class TournamentSelection(SelectionOperator):
    """
    Tournament selection.
    Randomly samples k individuals and returns the best.
    """

    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    def select(self, population: list) -> AllocationGenome:
        # Sample tournament
        tournament_size = min(self.tournament_size, len(population))
        indices = np.random.choice(len(population), size=tournament_size, replace=False)
        tournament = [population[i] for i in indices]

        # Return best in tournament
        return max(tournament, key=lambda g: g.fitness)


class RankSelection(SelectionOperator):
    """Rank-based selection with configurable selection pressure."""

    def __init__(self, selection_pressure: float = 1.5):
        self.selection_pressure = selection_pressure

    def select(self, population: list) -> AllocationGenome:
        n = len(population)

        # Sort by fitness
        sorted_pop = sorted(population, key=lambda g: g.fitness)

        # Compute rank-based probabilities
        ranks = np.arange(1, n + 1)
        probs = ranks ** self.selection_pressure
        probs = probs / probs.sum()

        # Sample
        idx = np.random.choice(n, p=probs)
        return sorted_pop[idx]


# ==================== Crossover Operators ====================

class CrossoverOperator(ABC):
    """Base class for crossover operators."""

    @abstractmethod
    def apply(self, parent1: AllocationGenome, parent2: AllocationGenome) -> AllocationGenome:
        """Create offspring from two parents."""
        raise NotImplementedError


class LayerwiseCrossover(CrossoverOperator):
    """
    Layer-wise crossover.
    Randomly selects each layer from either parent.
    """

    def apply(self, parent1: AllocationGenome, parent2: AllocationGenome) -> AllocationGenome:
        child_params = []

        for p1, p2 in zip(parent1.params, parent2.params):
            # Randomly choose parent for each parameter
            if np.random.random() < 0.5:
                child_params.append(p1.clone())
            else:
                child_params.append(p2.clone())

        child = AllocationGenome(params=child_params)
        child.param_shapes = parent1.param_shapes.copy()
        return child


class UniformCrossover(CrossoverOperator):
    """
    Uniform crossover.
    Each parameter element is independently chosen from either parent.
    """

    def apply(self, parent1: AllocationGenome, parent2: AllocationGenome) -> AllocationGenome:
        child_params = []

        for p1, p2 in zip(parent1.params, parent2.params):
            mask = th.rand_like(p1) < 0.5
            child_p = th.where(mask, p1, p2)
            child_params.append(child_p.clone())

        child = AllocationGenome(params=child_params)
        child.param_shapes = parent1.param_shapes.copy()
        return child


class BlendCrossover(CrossoverOperator):
    """
    BLX-alpha crossover (Blend Crossover).
    Creates offspring by interpolating between parents with extension.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def apply(self, parent1: AllocationGenome, parent2: AllocationGenome) -> AllocationGenome:
        child_params = []

        for p1, p2 in zip(parent1.params, parent2.params):
            # Compute range
            diff = (p2 - p1).abs()
            extend = self.alpha * diff

            # Random interpolation with extension
            low = th.min(p1, p2) - extend
            high = th.max(p1, p2) + extend

            child_p = low + th.rand_like(p1) * (high - low)
            child_params.append(child_p.clone())

        child = AllocationGenome(params=child_params)
        child.param_shapes = parent1.param_shapes.copy()
        return child


class SoftCrossover(CrossoverOperator):
    """
    Soft crossover (EvoRainbow style).
    Uses soft update (exponential moving average) instead of hard selection.
    """

    def __init__(self, tau: float = 0.5):
        self.tau = tau

    def apply(self, parent1: AllocationGenome, parent2: AllocationGenome) -> AllocationGenome:
        child_params = []

        for p1, p2 in zip(parent1.params, parent2.params):
            # Soft blend
            child_p = self.tau * p1 + (1 - self.tau) * p2
            child_params.append(child_p.clone())

        child = AllocationGenome(params=child_params)
        child.param_shapes = parent1.param_shapes.copy()
        return child


# ==================== Mutation Operators ====================

class MutationOperator(ABC):
    """Base class for mutation operators."""

    @abstractmethod
    def apply(self, genome: AllocationGenome) -> None:
        """Apply mutation to genome in-place."""
        raise NotImplementedError


class GaussianMutation(MutationOperator):
    """
    Gaussian mutation.
    Adds Gaussian noise to all parameters.
    """

    def __init__(self, strength: float = 0.1, adaptive: bool = False):
        self.strength = strength
        self.adaptive = adaptive

    def apply(self, genome: AllocationGenome) -> None:
        for p in genome.params:
            if self.adaptive:
                # Adaptive: scale by parameter magnitude
                scale = p.abs().mean() + 1e-8
                noise = th.randn_like(p) * self.strength * scale
            else:
                noise = th.randn_like(p) * self.strength
            p.add_(noise)


class SafeMutation(MutationOperator):
    """
    Safe mutation following ERL-Re² design.
    Limits mutation magnitude to preserve learned behaviors.
    """

    def __init__(self, strength: float = 0.1, safe_threshold: float = 0.5):
        self.strength = strength
        self.safe_threshold = safe_threshold

    def apply(self, genome: AllocationGenome) -> None:
        for p in genome.params:
            noise = th.randn_like(p) * self.strength

            # Check if change is too large
            change_ratio = noise.norm() / (p.norm() + 1e-8)

            if change_ratio > self.safe_threshold:
                # Scale down noise
                noise = noise * (self.safe_threshold / change_ratio)

            p.add_(noise)
