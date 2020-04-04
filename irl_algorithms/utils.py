import numpy as np

def get_statistics(env, demonstrations, features):
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

    for i in np.arange(n_demos):
        for t in np.arange(n_steps):
            state = np.int(state_list[i, t])
            init_state_dist[state] += 1

    for i in np.arange(n_demos):
        for t in np.arange(n_steps):
            state = np.int(state_list[i, t])
            action = np.int(action_list[i, t])
            state_action_count[state, action] += 1
            transition_probabilities, possible_next_states = env.get_transitions(state, action)
            for s in np.arange(len(possible_next_states)):
                next_state = np.int(possible_next_states[s])
                next_state_probability = transition_probabilities[s]
                init_state_dist[next_state] = init_state_dist[next_state] - \
                                              env.discount_rate * next_state_probability

    return feature_expectations, state_action_count, init_state_dist
