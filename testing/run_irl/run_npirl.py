import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from irl_algorithms.NPIRL.npirl import NPIRL
from rl_algorithms.LinearMDP import LinearMDP
from environments.gridworld import GridWorld
from testing.run_irl.utils import run_algo, run_optimal


# Set parameters
n_rows = 5
n_cols = 5
discount_rate = 0.8
random_shift = 0.3
goal_state_index = 12
n_steps = 100
n_demos = 20

# Initialise framework
env = GridWorld(n_rows, n_cols, discount_rate, random_shift=0, goal_state_index=goal_state_index)
mdp_solver = LinearMDP(env)
algo = NPIRL(env, mdp_solver, laplace_prior=None, verbose=False)

# Solve for optimal and get demonstrations
demonstrations = run_optimal(env, n_rows, n_cols, n_steps, n_demos, mdp_solver, title=None, show=False)

y_pred, mu, sigma = algo.run(demonstrations, batch_size=7, epochs=8)

