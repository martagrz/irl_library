import numpy as np


def get_statistics(env, demonstrations, features):
    """

    :param env: environment from the environments package (gym AI compatibility to come)
    :param demonstrations: the array of shape [N, T, 2], where N is number of demonstrations,
                            T is number of steps in each demonstration and the last axis refers to state/action
    :param features: the features of each state, shape [N_STATES, N_FEATURES]

    :return feature_expectations:
    :return state_action_count:
    :return init_state_dist:
    """
    transition_probabilities = env.transition_probabilities
    # demonstrations is shape [N, T, 2] - last dimension is state/action pair
    n_demos, n_steps, _ = demonstrations.shape

    # Count features
    _, n_features = features.shape

    # Compute feature expectations.
    feature_expectations = np.zeros(n_features)
    init_state_dist = np.zeros(env.n_states)
    state_action_count = np.zeros((env.n_states, env.n_actions))
    state_list = np.zeros((n_demos, n_steps))  # populated by state index
    action_list = np.zeros((n_demos, n_steps))  # populated by action index

    for i in np.arange(n_demos):
        for t in np.arange(n_steps):
            state = np.int(demonstrations[i, t, 0])
            action = np.int(demonstrations[i, t, 1])
            state_list[i, t] = state
            action_list[i, t] = action
            state_vec = np.zeros(env.n_states)
            state_vec[state] = 1
            feature_expectations = feature_expectations + features.T @ state_vec

            init_state_dist[state] += 1

    for i in np.arange(n_demos):
        for t in np.arange(n_steps):
            state = np.int(state_list[i, t])
            action = np.int(action_list[i, t])
            state_action_count[state, action] += 1
            for next_state in np.arange(env.n_states):
                next_state_probability = transition_probabilities[state, action, next_state]
                init_state_dist[next_state] = init_state_dist[next_state] - env.discount_rate * next_state_probability

    return feature_expectations, state_action_count, init_state_dist


def policy_propagation(P, n_states, n_actions, demonstrations, policy, deterministic=True):
    """

    :param P: transition probability matrix [N_STATES, N_ACTIONS, N_STATES]
    :param n_states: number of states in env
    :param n_actions: number of actions in env
    :param demonstrations: matrix of expert trajectories [N_DEMOS, N_STEPS, 2] where 2 is state/action axis
    :param policy: calculated policy values [N_STATES, N_ACTIONS]
    :param deterministic: Bool, whether actions taken are determinimistic
    :return:
    """
    print(demonstrations.shape)
    print('nstates', n_states)
    n_demos, n_steps, _ = demonstrations.shape
    demo_states = demonstrations[:, :, 0]
    # mu[s, t] is the prob of visiting state s at time t
    mu = np.zeros([n_states, n_steps])
    for demo in demonstrations:
        mu[np.int(demo_states[0, 0]), 0] += 1  # each starting state
    mu[:, 0] = mu[:, 0] / (n_demos * n_steps)

    for next_state in np.arange(n_states - 1):
        for step in np.arange(n_steps - 1):
            if deterministic:
                for state in np.arange(n_states - 1):
                    action = np.argmax(policy[state])  # needs to be an integer
                    print(state, next_state, action)
                    mu[next_state, step + 1] = np.sum(mu[state, step] *
                                                      P[state, next_state, action])
            else:
                mu[state, step + 1] = np.sum(
                    [np.sum([mu[pre_s, step] *
                             P[pre_s, state, a1] * policy[pre_s, a1]
                             for a1 in np.arange(n_actions)]) for pre_s in np.arange(n_states)])
    state_frequencies = np.sum(mu, axis=1)
    return state_frequencies
