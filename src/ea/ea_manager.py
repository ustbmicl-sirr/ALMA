"""
EA Manager for ALMA
Main interface for evolutionary optimization of task allocation policies.
"""

import torch as th
import torch.nn as nn
import numpy as np
import random
import logging

from .genome import AllocationGenome, Population
from .operators import (
    TournamentSelection,
    LayerwiseCrossover,
    GaussianMutation,
    RankSelection,
    BlendCrossover,
    SoftCrossover,
    SafeMutation,
)

logger = logging.getLogger(__name__)


class EAManager:
    """
    Evolutionary Algorithm Manager for optimizing task allocation policies.

    Features:
    - Population management
    - Genetic operators (selection, crossover, mutation)
    - Bidirectional synchronization with gradient-trained network
    - Elite preservation
    - Fitness evaluation using ALMA's episode runner
    """

    def __init__(self, args, mac, runner, logger=None):
        """
        Initialize EA Manager.

        Args:
            args: Configuration arguments
            mac: Multi-Agent Controller (contains allocation policy)
            runner: Episode runner for fitness evaluation
            logger: Logger instance
        """
        self.args = args
        self.mac = mac
        self.runner = runner
        self.console_logger = logger

        # EA parameters from config
        ea_args = args.ea
        self.pop_size = ea_args.get('pop_size', 5)
        self.elite_size = ea_args.get('elite_size', 2)
        self.mutation_rate = ea_args.get('mutation_rate', 0.1)
        self.mutation_strength = ea_args.get('mutation_strength', 0.1)
        self.crossover_rate = ea_args.get('crossover_rate', 0.8)
        self.tournament_size = ea_args.get('tournament_size', 3)
        self.eval_episodes = ea_args.get('eval_episodes', 5)
        self.sync_interval = ea_args.get('sync_interval', 1000)
        self.sync_threshold = ea_args.get('sync_threshold', 0.05)
        self.bidirectional_sync = ea_args.get('bidirectional_sync', True)

        # Get allocation policy from MAC
        self.alloc_policy = self._get_alloc_policy()

        # Initialize population
        self.population = Population(
            pop_size=self.pop_size,
            elite_size=self.elite_size
        )
        self.population.initialize_from_policy(
            self.alloc_policy,
            noise_std=self.mutation_strength
        )

        # Move to device
        if args.use_cuda:
            self.population.to("cuda")

        # Best genome tracking
        self.best_genome = None
        self.generation = 0

        # Genetic operators
        self.selection = self._create_selection_operator(ea_args)
        self.crossover = self._create_crossover_operator(ea_args)
        self.mutation = self._create_mutation_operator(ea_args)

        # Statistics
        self.stats_history = []

        if self.console_logger:
            self.console_logger.info(f"EA Manager initialized: pop_size={self.pop_size}, "
                                    f"elite_size={self.elite_size}")

    def _get_alloc_policy(self) -> nn.Module:
        """Get allocation policy from MAC."""
        # ALMA stores allocation policy in mac.alloc_policy
        if hasattr(self.mac, 'alloc_policy') and self.mac.alloc_policy is not None:
            return self.mac.alloc_policy
        else:
            raise ValueError("MAC does not have allocation policy (alloc_policy). "
                           "Make sure hier_agent.task_allocation is set to 'aql'.")

    def _create_selection_operator(self, ea_args):
        """Create selection operator based on config."""
        selection_type = ea_args.get('selection', 'tournament')

        if selection_type == 'tournament':
            return TournamentSelection(self.tournament_size)
        elif selection_type == 'rank':
            return RankSelection(ea_args.get('selection_pressure', 1.5))
        else:
            return TournamentSelection(3)

    def _create_crossover_operator(self, ea_args):
        """Create crossover operator based on config."""
        crossover_type = ea_args.get('crossover', 'layerwise')

        if crossover_type == 'layerwise':
            return LayerwiseCrossover()
        elif crossover_type == 'uniform':
            from .operators import UniformCrossover
            return UniformCrossover()
        elif crossover_type == 'blend':
            return BlendCrossover(ea_args.get('blend_alpha', 0.5))
        elif crossover_type == 'soft':
            return SoftCrossover(ea_args.get('soft_tau', 0.5))
        else:
            return LayerwiseCrossover()

    def _create_mutation_operator(self, ea_args):
        """Create mutation operator based on config."""
        mutation_type = ea_args.get('mutation', 'gaussian')

        if mutation_type == 'gaussian':
            return GaussianMutation(
                strength=self.mutation_strength,
                adaptive=ea_args.get('adaptive_mutation', False)
            )
        elif mutation_type == 'safe':
            return SafeMutation(
                strength=self.mutation_strength,
                safe_threshold=ea_args.get('safe_threshold', 0.5)
            )
        else:
            return GaussianMutation(strength=self.mutation_strength)

    def evaluate_population(self, t_env: int) -> dict:
        """
        Evaluate fitness of all genomes in population.

        Args:
            t_env: Current environment timestep

        Returns:
            stats: Evaluation statistics
        """
        # Store original policy parameters
        original_params = [p.data.clone() for p in self.alloc_policy.parameters()]

        fitnesses = []

        for genome in self.population:
            # Apply genome to policy
            genome.apply_to(self.alloc_policy)

            # Evaluate
            fitness = self._evaluate_genome()
            genome.fitness = fitness
            fitnesses.append(fitness)

        # Restore original parameters
        for p, orig_p in zip(self.alloc_policy.parameters(), original_params):
            p.data.copy_(orig_p)

        # Update best genome
        self.population.sort_by_fitness()
        current_best = self.population.best

        if self.best_genome is None or current_best.fitness > self.best_genome.fitness:
            self.best_genome = current_best.clone()

        # Compute stats
        stats = {
            'ea/best_fitness': self.best_genome.fitness,
            'ea/population_best': current_best.fitness,
            'ea/mean_fitness': np.mean(fitnesses),
            'ea/fitness_std': np.std(fitnesses),
            'ea/diversity': self.population.diversity,
            'ea/generation': self.generation,
        }

        self.stats_history.append(stats)

        return stats

    def _evaluate_genome(self) -> float:
        """Evaluate a single genome using episode runner."""
        returns = []

        for _ in range(self.eval_episodes):
            # Run test episode
            episode_batch, subtask_info = self.runner.run(test_mode=True)

            # Get episode return
            episode_return = episode_batch["reward"].sum().item()
            returns.append(episode_return)

        return np.mean(returns)

    def evolve(self) -> dict:
        """
        Perform one generation of evolution.

        Returns:
            stats: Evolution statistics
        """
        new_genomes = []

        # Elite preservation
        elites = self.population.elites
        for elite in elites:
            new_genomes.append(elite.clone())

        # Generate offspring
        while len(new_genomes) < self.pop_size:
            # Selection
            parent1 = self.selection.select(self.population.genomes)
            parent2 = self.selection.select(self.population.genomes)

            # Crossover
            if random.random() < self.crossover_rate:
                child = self.crossover.apply(parent1, parent2)
            else:
                child = parent1.clone()

            # Mutation
            if random.random() < self.mutation_rate:
                self.mutation.apply(child)

            child.fitness = float('-inf')  # Reset fitness
            child.genome_id = len(new_genomes)
            new_genomes.append(child)

        # Update population
        self.population.genomes = new_genomes
        self.population.generation += 1
        self.generation += 1

        stats = {
            'ea/generation': self.generation,
            'ea/population_size': len(self.population),
        }

        return stats

    def bidirectional_sync_step(self, main_fitness: float) -> str:
        """
        Perform bidirectional synchronization between EA and main network.

        If EA finds a better solution, update main network.
        If main network is better, inject into EA population.

        Args:
            main_fitness: Current fitness of main allocation policy

        Returns:
            sync_type: Type of sync performed
        """
        if not self.bidirectional_sync or self.best_genome is None:
            return "no_sync"

        best_ea_fitness = self.best_genome.fitness

        # EA → Main: EA found significantly better solution
        if best_ea_fitness > main_fitness * (1 + self.sync_threshold):
            if self.console_logger:
                self.console_logger.info(
                    f"EA→Main sync: EA fitness {best_ea_fitness:.4f} > "
                    f"Main fitness {main_fitness:.4f}"
                )
            self._sync_ea_to_main()
            return "ea_to_main"

        # Main → EA: Gradient learning found better solution
        elif main_fitness > best_ea_fitness * (1 + self.sync_threshold):
            if self.console_logger:
                self.console_logger.info(
                    f"Main→EA sync: Main fitness {main_fitness:.4f} > "
                    f"EA fitness {best_ea_fitness:.4f}"
                )
            self._sync_main_to_ea(main_fitness)
            return "main_to_ea"

        return "no_sync"

    def _sync_ea_to_main(self) -> None:
        """Apply best EA genome to main allocation policy."""
        self.best_genome.apply_to(self.alloc_policy)

    def _sync_main_to_ea(self, main_fitness: float) -> None:
        """Inject main allocation policy into EA population."""
        # Create genome from main policy
        main_genome = AllocationGenome(alloc_policy=self.alloc_policy)
        main_genome.fitness = main_fitness

        if self.args.use_cuda:
            main_genome.to("cuda")

        # Inject into population (replace worst)
        self.population.sort_by_fitness()
        main_genome.genome_id = self.population.genomes[-1].genome_id
        self.population.genomes[-1] = main_genome

        # Update best if necessary
        if main_fitness > self.best_genome.fitness:
            self.best_genome = main_genome.clone()

    def get_best_params(self) -> list:
        """Get parameters of best genome."""
        if self.best_genome is None:
            return None
        return self.best_genome.params

    def apply_best_to_policy(self) -> None:
        """Apply best genome parameters to allocation policy."""
        if self.best_genome is not None:
            self.best_genome.apply_to(self.alloc_policy)

    def evaluate_main_fitness(self) -> float:
        """Evaluate current main allocation policy fitness."""
        returns = []

        for _ in range(self.eval_episodes):
            episode_batch, _ = self.runner.run(test_mode=True)
            episode_return = episode_batch["reward"].sum().item()
            returns.append(episode_return)

        return np.mean(returns)

    def save_state(self) -> dict:
        """Save EA state for checkpointing."""
        state = {
            'generation': self.generation,
            'best_genome_params': [p.cpu() for p in self.best_genome.params] if self.best_genome else None,
            'best_genome_fitness': self.best_genome.fitness if self.best_genome else None,
            'population_params': [
                [p.cpu() for p in g.params] for g in self.population.genomes
            ],
            'population_fitness': [g.fitness for g in self.population.genomes],
            'stats_history': self.stats_history,
        }
        return state

    def load_state(self, state: dict, device: str = "cpu") -> None:
        """Load EA state from checkpoint."""
        self.generation = state['generation']
        self.population.generation = self.generation

        if state['best_genome_params'] is not None:
            params = [p.to(device) for p in state['best_genome_params']]
            self.best_genome = AllocationGenome(params=params)
            self.best_genome.fitness = state['best_genome_fitness']

        # Restore population
        for i, (params, fitness) in enumerate(zip(
            state['population_params'],
            state['population_fitness']
        )):
            params = [p.to(device) for p in params]
            self.population.genomes[i].params = params
            self.population.genomes[i].fitness = fitness

        self.stats_history = state.get('stats_history', [])

    def get_stats(self) -> dict:
        """Get current EA statistics."""
        return {
            'generation': self.generation,
            'best_fitness': self.best_genome.fitness if self.best_genome else float('-inf'),
            'mean_fitness': self.population.mean_fitness,
            'fitness_std': self.population.fitness_std,
            'diversity': self.population.diversity,
        }
