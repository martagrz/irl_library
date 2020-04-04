import numpy as np

class LinearMDP:
    def __init__(self, env):
        self.env = env

    def compute_softmax(self, q_values):
        max_q_values = np.amax(q_values, axis=1) # get max along each row
        n_rows, n_cols = q_values.shape
        state_values = max_q_values + np.log(np.sum(np.exp(q_values - np.repeat(max_q_values[:, np.newaxis], n_cols, axis=1))))
        return state_values

    def linear_value_iteration(self, sa_reward_matrix, init_state_values=None):
        convergence_value = 1e-4 #or 1e-10
        diff = 1.0
        q_values = np.zeros((self.env.n_states, self.env.n_actions))

        if init_state_values is not None:
            state_values = init_state_values
        else:
            state_values = np.zeros(self.env.n_states)

        while diff >= convergence_value:
            old_state_values = state_values

            for state_index in np.arange(self.env.n_states):
                for action_index in np.arange(self.env.n_actions):
                    reward = sa_reward_matrix[state_index, action_index]
                    possible_next_states = self.env.possible_next_states[state_index][action_index]
                    transition_probabilities = self.env.transition_probabilities[state_index][action_index]

                    if isinstance(possible_next_states, float):
                        next_state_index = np.int(possible_next_states)
                        next_state_values = state_values[next_state_index]
                    else:
                        next_state_values = np.zeros(len(possible_next_states))
                        for i in np.arange(len(possible_next_states)):

                            next_state_index = np.int(possible_next_states[i])
                            next_state_values[i] = state_values[next_state_index]

                    update_value = np.sum(np.multiply(transition_probabilities, next_state_values))
                    q_values[state_index, action_index] = reward + self.env.discount_rate * update_value

            state_values = self.compute_softmax(q_values)
            diff = np.max(np.abs(state_values - old_state_values))

        log_policy = q_values - np.repeat(state_values[:, np.newaxis], self.env.n_actions, axis=1)
        policy = np.exp(log_policy)
        return state_values, q_values, policy, log_policy

    def linear_mdp_frequency(self, policy, init_state_distribution=None):
        # Compute the occupancy measure of the linear MDP given a policy.
        diff = 1.0
        convergence_value = 1e-4

        if init_state_distribution is None:
            state_distribution = (1 / self.env.n_states) * np.ones(self.env.n_states)
        else:
            state_distribution = init_state_distribution

        while diff >= convergence_value:
            old_state_distribution = state_distribution

            for state in np.arange(self.env.n_states):
                state_probability = state_distribution[state]
                for action in np.arange(self.env.n_actions):
                    policy_value = policy[state, action]
                    transition_probabilities, possible_next_states = self.env.get_transitions(state, action)
                    for i in np.arange(len(possible_next_states)):
                        next_state = np.int(possible_next_states[i])
                        probability = transition_probabilities[i]
                        update_value = policy_value * probability * state_probability * self.env.discount_rate
                        state_distribution[next_state] += update_value

            # state_dist is a VECTOR of length n_states
            diff = np.max(np.abs(state_distribution - old_state_distribution))
        return state_distribution

    def solve(self, states_actions_rewards_matrix):
        return self.linear_value_iteration(states_actions_rewards_matrix)

