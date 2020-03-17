import numpy as np
import pickle
from algorithms.maxent import MaxEnt
from algorithms.LinearMDP import LinearMDP
from environments.gridworld import GridWorld
from testing.plot_functions import plot_quiver, plot_reward_surface
from testing.utils import get_demonstrations

# Set parameters
n_rows = 5
n_cols = 5
discount_rate = 0.8
random_shift = 0.3
goal_state_index = 12

# Initialise framework
env = GridWorld(n_rows, n_cols, discount_rate, random_shift=0, goal_state_index=goal_state_index)
mdp_solver = LinearMDP(env)
algo = MaxEnt(env, mdp_solver, laplace_prior=None, verbose=False)

# Solve for optimal and get demonstrations
print("Solving for optimal policy via value iteration.")
_, _, opt_policy, opt_log_policy = mdp_solver.solve(env.states_actions_rewards_matrix)
optimal_policy = np.argmax(opt_policy, axis=1)
opt_reward_vector = env.states_actions_rewards_matrix[:, 0]
plot_quiver(n_rows, n_cols, optimal_policy, show=True)
plot_reward_surface(n_rows, n_cols, opt_reward_vector, show=True)

print("Obtaining demonstrations.")
n_steps = 10
n_demos = 50
demonstrations = get_demonstrations(env, n_steps, n_demos, optimal_policy)

# Run IRL algorithm
train = True
print('Running IRL algorithm.')
if train:
    state_values, q_values, policy, log_policy, states_actions_rewards_matrix = algo.run(demonstrations)
    pickle.dump(state_values, open("./pickle_jar/state_values.p", "wb"))
    pickle.dump(q_values, open("./pickle_jar/q_values.p", "wb"))
    pickle.dump(policy, open("./pickle_jar/policy.p", "wb"))
    pickle.dump(log_policy, open("./pickle_jar/log_policy.p", "wb"))
    pickle.dump(states_actions_rewards_matrix, open("./pickle_jar/sa_rewards_matrix.p", "wb"))

else:
    state_values = pickle.load(open("./pickle_jar/state_values.p", "rb"))
    q_values = pickle.load(open("./pickle_jar/q_values.p", "rb"))
    policy = pickle.load(open("./pickle_jar/policy.p", "rb"))
    log_policy = pickle.load(open("./pickle_jar/log_policy.p", "rb"))
    states_actions_rewards_matrix = pickle.load(open("./pickle_jar/sa_rewards_matrix.p", "rb"))
    print(states_actions_rewards_matrix)

learned_policy = np.argmax(policy, axis=1)
reward_vector = states_actions_rewards_matrix
plot_quiver(n_rows, n_cols, learned_policy, show=True)
plot_reward_surface(n_rows, n_cols, states_actions_rewards_matrix, show=True)

print("Done.")


