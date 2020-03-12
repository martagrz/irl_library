import numpy as np


def get_demonstrations(env, n_steps, n_demos, optimal_policy):
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
