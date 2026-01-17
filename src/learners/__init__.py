from .q_learner import QLearner
from .ea_q_learner import EAQLearner, EAOnlyQLearner

REGISTRY = {}
REGISTRY["q_learner"] = QLearner
REGISTRY["ea_q_learner"] = EAQLearner
REGISTRY["ea_only_q_learner"] = EAOnlyQLearner