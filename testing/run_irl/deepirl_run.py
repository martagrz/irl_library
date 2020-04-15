import numpy as np

from irl_algorithms.DeepIRL.deepirl import DeepIRL
from rl_algorithms.LinearMDP import LinearMDP
from environments.gridworld import GridWorld
from testing.run_irl.utils import run_algo, run_optimal
from testing.run_irl.plot_functions import plot_quiver, plot_reward_surface

import os
print(os.getcwd())

# Set parameters
n_rows = 10
n_cols = 10
discount_rate = 0.4
random_shift = 0.3
goal_state_index = [12, 45]
n_steps = 10
n_demos = 100

# Initialise framework
env = GridWorld(n_rows, n_cols, discount_rate, random_shift=0, goal_state_index=goal_state_index)
mdp_solver = LinearMDP(env)

# Solve for optimal and get demonstrations
demonstrations = run_optimal(env, n_rows, n_cols, n_steps, n_demos, mdp_solver, show=False)

algo = DeepIRL(env, mdp_solver)
state_values, q_values, policy, log_policy, rewards_vector = algo.run(demonstrations, learning_rate=0.01, n_iters=150)

learned_policy = np.argmax(policy, axis=1)
path = './pickle_jar/' + 'DeepIRL'
plot_quiver(n_rows, n_cols, learned_policy, title=path + "/quiver_plot", show=True)
plot_reward_surface(n_rows, n_cols, rewards_vector, title=path + "/reward_plot", show=True)

