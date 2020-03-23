import numpy as np
import pickle
from algorithms.maxent import MaxEnt
from algorithms.LinearMDP import LinearMDP
from algorithms.gpirl import GPIRL
from environments.gridworld import GridWorld
from testing.utils import get_demonstrations, run_algo, run_optimal

# Set parameters
n_rows = 5
n_cols = 5
discount_rate = 0.8
random_shift = 0.3
goal_state_index = 12
n_steps = 100
n_demos = 1000

# Initialise framework
env = GridWorld(n_rows, n_cols, discount_rate, random_shift=0, goal_state_index=goal_state_index)
mdp_solver = LinearMDP(env)
algo_0 = MaxEnt(env, mdp_solver, laplace_prior=None, verbose=False)
algo_1 = GPIRL(env, mdp_solver, laplace_prior=None, verbose=False)

# Solve for optimal and get demonstrations
demonstrations = run_optimal(env, n_rows, n_cols, n_steps, n_demos, mdp_solver, show=False)

# Solve IRL methods
run_algo(demonstrations, n_rows, n_cols, algo_0, name='MaxEnt', train=False, show=False)
run_algo(demonstrations, n_rows, n_cols, algo_1, name='GPIRL', train=True, show=True)
