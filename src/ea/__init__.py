# EA Module for ALMA
# Evolutionary Algorithm optimization for task allocation

from .ea_manager import EAManager
from .genome import AllocationGenome
from .operators import TournamentSelection, LayerwiseCrossover, GaussianMutation

REGISTRY = {}

REGISTRY["ea_manager"] = EAManager
