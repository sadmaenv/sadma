from learners.ppo_learner import PPOLearner
from learners.q_learner import QLearner

learner_REGISTRY = {}
learner_REGISTRY["ppo_learner"] = PPOLearner
learner_REGISTRY["q_learner"] = QLearner
