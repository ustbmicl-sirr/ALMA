"""
Genome representation for EA-HRL
Encodes the task allocation policy parameters as a genome for evolutionary optimization.
"""

import torch as th
import torch.nn as nn
import numpy as np
import copy


class AllocationGenome:
    """
    Genome representing task allocation policy parameters.

    In ALMA context, this wraps the AutoregressiveAllocPolicy parameters.
    """

    def __init__(
        self,
        alloc_policy: nn.Module = None,
        params: list = None,
        genome_id: int = 0
    ):
        self.genome_id = genome_id
        self.fitness = float('-inf')
        self.age = 0

        if alloc_policy is not None:
            # Initialize from allocation policy network
            self.params = [p.data.clone() for p in alloc_policy.parameters()]
            self.param_shapes = [p.shape for p in self.params]
        elif params is not None:
            # Initialize from parameter list
            self.params = [p.clone() for p in params]
            self.param_shapes = [p.shape for p in self.params]
        else:
            self.params = []
            self.param_shapes = []

        self._flat_params = None  # Cached flattened params

    def flatten(self) -> th.Tensor:
        """Flatten all parameters into a single vector."""
        if self._flat_params is None:
            self._flat_params = th.cat([p.flatten() for p in self.params])
        return self._flat_params

    def unflatten(self, flat_params: th.Tensor) -> None:
        """Unflatten vector back to parameter tensors."""
        self._flat_params = flat_params.clone()

        idx = 0
        for i, shape in enumerate(self.param_shapes):
            numel = np.prod(shape)
            self.params[i] = flat_params[idx:idx + numel].view(shape)
            idx += numel

    @property
    def num_params(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.params)

    def mutate(self, strength: float = 0.1) -> None:
        """Apply Gaussian mutation to parameters."""
        for p in self.params:
            noise = th.randn_like(p) * strength
            p.add_(noise)
        self._flat_params = None  # Invalidate cache

    def clone(self) -> "AllocationGenome":
        """Create a deep copy of this genome."""
        new_genome = AllocationGenome(
            params=[p.clone() for p in self.params],
            genome_id=self.genome_id
        )
        new_genome.param_shapes = self.param_shapes.copy()
        new_genome.fitness = self.fitness
        new_genome.age = self.age
        return new_genome

    def copy_from(self, other: "AllocationGenome") -> None:
        """Copy parameters from another genome."""
        for p, other_p in zip(self.params, other.params):
            p.copy_(other_p)
        self._flat_params = None

    def apply_to(self, alloc_policy: nn.Module) -> None:
        """Apply genome parameters to allocation policy network."""
        for p, genome_p in zip(alloc_policy.parameters(), self.params):
            p.data.copy_(genome_p)

    def load_from(self, alloc_policy: nn.Module) -> None:
        """Load parameters from allocation policy network."""
        for genome_p, p in zip(self.params, alloc_policy.parameters()):
            genome_p.copy_(p.data)
        self._flat_params = None

    def distance(self, other: "AllocationGenome") -> float:
        """Compute L2 distance to another genome."""
        return (self.flatten() - other.flatten()).norm().item()

    def to(self, device) -> "AllocationGenome":
        """Move genome to device."""
        self.params = [p.to(device) for p in self.params]
        self._flat_params = None
        return self

    def __repr__(self):
        return f"AllocationGenome(id={self.genome_id}, fitness={self.fitness:.4f}, params={self.num_params})"


class Population:
    """
    Manages a population of genomes for evolutionary optimization.
    """

    def __init__(self, pop_size: int, elite_size: int = 2):
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.genomes = []
        self.generation = 0

    def initialize_from_policy(self, alloc_policy: nn.Module, noise_std: float = 0.1) -> None:
        """Initialize population from a base allocation policy."""
        self.genomes = []

        for i in range(self.pop_size):
            genome = AllocationGenome(alloc_policy=alloc_policy, genome_id=i)
            if i > 0:
                # Add noise to create diversity
                genome.mutate(strength=noise_std)
            self.genomes.append(genome)

    def sort_by_fitness(self, descending: bool = True) -> None:
        """Sort population by fitness."""
        self.genomes.sort(key=lambda g: g.fitness, reverse=descending)

    @property
    def best(self) -> AllocationGenome:
        """Get best genome in population."""
        return max(self.genomes, key=lambda g: g.fitness)

    @property
    def elites(self) -> list:
        """Get elite genomes."""
        self.sort_by_fitness()
        return self.genomes[:self.elite_size]

    @property
    def mean_fitness(self) -> float:
        """Get mean fitness of population."""
        return np.mean([g.fitness for g in self.genomes])

    @property
    def fitness_std(self) -> float:
        """Get fitness standard deviation."""
        return np.std([g.fitness for g in self.genomes])

    @property
    def diversity(self) -> float:
        """Compute population diversity as average pairwise distance."""
        if len(self.genomes) < 2:
            return 0.0

        distances = []
        for i, g1 in enumerate(self.genomes):
            for g2 in self.genomes[i + 1:]:
                distances.append(g1.distance(g2))

        return np.mean(distances)

    def to(self, device) -> "Population":
        """Move all genomes to device."""
        for g in self.genomes:
            g.to(device)
        return self

    def __len__(self) -> int:
        return len(self.genomes)

    def __iter__(self):
        return iter(self.genomes)

    def __getitem__(self, idx: int) -> AllocationGenome:
        return self.genomes[idx]
