import numpy as np
import pickle
from testing.run_irl.plot_functions import plot_quiver, plot_reward_surface


def run_optimal(env, n_rows, n_cols, n_steps, n_demos, mdp_solver, title=None, show=False):
    print("Solving for optimal policy via value iteration.")
    _, _, opt_policy, opt_log_policy = mdp_solver.solve(env.states_actions_rewards_matrix)
    optimal_policy = np.argmax(opt_policy, axis=1)
    opt_reward_vector = env.states_actions_rewards_matrix[:, 0]
    plot_quiver(n_rows, n_cols, optimal_policy, title=title,show=show)
    plot_reward_surface(n_rows, n_cols, opt_reward_vector, title=title, show=show)

    demonstrations = get_demonstrations(env, n_steps, n_demos, optimal_policy)
    return demonstrations


def run_algo(demonstrations, n_rows, n_cols, algo, name, train=True, show=True):
    print('Running IRL algorithm', name)
    path = './run_irl/pickle_jar/' + name
    if train:
        state_values, q_values, policy, log_policy, rewards_vector = algo.run(demonstrations)
        pickle.dump(state_values, open(path + "/state_values.p", "wb"))
        pickle.dump(q_values, open(path + "/q_values.p", "wb"))
        pickle.dump(policy, open(path + "/policy.p", "wb"))
        pickle.dump(log_policy, open(path + "/log_policy.p", "wb"))
        pickle.dump(rewards_vector, open(path + "/sa_rewards_matrix.p", "wb"))

    else:
        state_values = pickle.load(open(path + "/state_values.p", "rb"))
        q_values = pickle.load(open(path + "/q_values.p", "rb"))
        policy = pickle.load(open(path + "/policy.p", "rb"))
        log_policy = pickle.load(open(path + "/log_policy.p", "rb"))
        rewards_vector = pickle.load(open(path + "/sa_rewards_matrix.p", "rb"))

    learned_policy = np.argmax(policy, axis=1)
    plot_quiver(n_rows, n_cols, learned_policy, title=path+"/quiver_plot", show=show)
    plot_reward_surface(n_rows, n_cols, rewards_vector, title= path+"/reward_plot", show=show)
    print("Done.")


def get_demonstrations(env, n_steps, n_demos, optimal_policy):
    print("Obtaining demonstrations.")
    init_states_list = np.random.randint(0, env.n_states, n_demos)
    demonstrations = np.zeros((n_demos, n_steps, 2))
    for demo in np.arange(n_demos):
        t = 0
        state_index = init_states_list[demo]
        while t < n_steps:
            action = optimal_policy[state_index]
            demonstrations[demo, t] = np.array((state_index, action))
            next_state_index, _, _ = env.step(state_index, action)
            state_index = next_state_index
            t += 1
    return demonstrations
