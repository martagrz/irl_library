import numpy as np

class LinearMDP:
    def __init__(self, env):

    def softmax(q_values):
        # Find maximum elements.
        maxx = np.max(q)
        # Compute safe softmax
        v = maxx + np.log(np.sum(np.exp(q - np.repeat(maxx)))) #CHECK THIS!
        return

    def linear_value_iteration(env, reward_function, init_state_values=None):
        convergence_value = 1e-4 #or 1e-10
        diff = 1.0
        q_values = np.zeros((env.n_states, env.n_actions))
        if init_state_values is not None:
            state_values = init_state_values
        else:
            state_values = np.zeros(env.n_states)
        while diff >= convergence_value:
            old_state_values = state_values
            for state in np.arange(env.n_states):
                for action in np.arange(env.n_actions):
                    reward = reward_function(state, action)
                    transition_probabilities, possible_next_states = env.get_transitions(state, action)
                    next_state_values = np.array(len(possible_next_states))
                    for i in np.arange(len(possible_next_states)):
                        next_state = possible_next_states[i]
                        next_state_values[i] = state_values[next_state]
                    q_values[state, action] = reward + env.discount_rate * np.dot(transition_probabilities, next_state_values)
            state_values = softmax(q_values)
            diff = np.max(np.abs(state_values - old_state_values))
        log_policy= q_values - np.repeat(state_values, 1, actions)
        policy = np.exp(log_policy)
        return state_values, q_values, policy, log_policy


    def linear_mdp_frequency(env, mdp_solution, init_state_visitation_count=None, previous_state_visitation_count=None):
        # Compute the occupancy measure of the linear MDP given a policy.
        states = env.n_states
        actions = env.n_actions
        transitions = env.transitions
        diff = 1.0
        convergence_value = 1e-4
        # convergence_value = 1e-10
        if previous_state_visitation_count is not None:
            state_visitation_count = previous_state_visitation_count
        else:
            state_visitation_count = np.zeros((states, 1))

        if init_state_visitation_count is not None:
            init_state_visitation_count = (1 / states) * np.ones(states, 1);

        while diff >= convergence_value:
            new_state_visitation_count = state_visitation_count
            Dpi = repmat(mdp_solution.p, [1 1 transitions]). * mdp_data.sa_p. * repmat(Dp, [1 actions transitions]) * env.discount
            state_visitation_count = init_state_visitation_count + sum(sparse(mdp_data.sa_s(:), 1: states * actions * transitions, Dpi(:),
                states, states * actions * transitions)*ones(states * actions * transitions, 1), 2)
            diff = max(abs(state_visitation_count - new_state_visitation_count))

        return state_visitation_count


    def solve_mdp(env, reward_function):
        return linear_value_iteration(env, reward_function)

