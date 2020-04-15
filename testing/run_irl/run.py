from irl_algorithms.MaxEntIRL.maxent import MaxEnt
from rl_algorithms.LinearMDP import LinearMDP
from irl_algorithms.GPIRL.gpirl import GPIRL
from environments.gridworld import GridWorld
from testing.run_irl.utils import run_algo, run_optimal

# Set parameters
n_rows = 10
n_cols = 10
discount_rate = 0.4
random_shift = 0.3
goal_state_index = [12, 47]
n_steps = 10
n_demos = 100

# Initialise framework
env = GridWorld(n_rows, n_cols, discount_rate, random_shift=0, goal_state_index=goal_state_index)
mdp_solver = LinearMDP(env)
algo_0 = MaxEnt(env, mdp_solver, laplace_prior=None, verbose=False)
algo_1 = GPIRL(env, mdp_solver, laplace_prior=None, verbose=False)

# Solve for optimal and get demonstrations
demonstrations = run_optimal(env, n_rows, n_cols, n_steps, n_demos, mdp_solver, show=True)

# Solve IRL methods
run_algo(demonstrations, n_rows, n_cols, algo_0, name='MaxEnt', train=True, show=True)
run_algo(demonstrations, n_rows, n_cols, algo_1, name='GPIRL', train=True, show=True)
