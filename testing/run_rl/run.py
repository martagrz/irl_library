#!/Library/Frameworks/Python.framework/Versions/3.8/bin/python3.8
import matplotlib.pyplot as plt
import numpy as np
import pickle
from environments.dynamic_pricing.demand_models import linear_demand, exp_demand
from environments.dynamic_pricing.finite_horizon import FiniteHorizonEnv
from environments.dynamic_pricing.finite_stock_and_horizon import FiniteStockHorizonEnv
from environments.gridworld import GridWorld
from rl_algorithms.LinearMDP import LinearMDP
np.random.seed(0)
n_rows, n_cols = 5, 5
discount_rate = 0.4
random_shift = 0.2
train = True

demand_params = np.array([75, 5, 2])
demand_model = exp_demand
state_params = np.array([[75, 0, 10, 0.4], 10])
action_params = np.array([[70, 120, 1], [0, 20, 1]])


if train:
    env = FiniteStockHorizonEnv(demand_model, demand_params, state_params, action_params,
                                stochastic=False,
                                discount_rate=0.2,
                                pre_train_transition=None)
    pre_train_transition = np.array([[env.transition_probabilities, env.possible_next_states], env.sa_reward_matrix])
    pickle.dump(pre_train_transition, open("pickle_jar/pre_train_transition.p", "wb"))
else:
    pre_train_transition = pickle.load(open("pickle_jar/pre_train_transition.p", "rb"))


env = FiniteStockHorizonEnv(demand_model, demand_params, state_params, action_params, stochastic=False,
                            discount_rate=0.2, pre_train_transition=pre_train_transition)
# does not work when stochastic is true

algo = LinearMDP(env)
state_values, q_values, policy, log_policy = algo.linear_value_iteration(env.sa_reward_matrix)

opt_policy = np.zeros(env.n_states)
for i in np.arange(policy.shape[0]):
    row = policy[i]
    action = np.argmax(row)
    opt_policy[i] = action

env.evaluate_policy(opt_policy, max_steps=100)

