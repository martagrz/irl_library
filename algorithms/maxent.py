import numpy as np
from scipy.optimize import minimize
np.random.seed(0)


class MaxEnt:
    def __init__(self, env, mdp_solver, laplace_prior=None, verbose=False):
        self.env = env
        self.verbose = verbose
        self.laplace_prior = laplace_prior
        self.mdp_solver = mdp_solver

    def compute_objective(self, features, feature_expectations, init_state_dist, state_action_count):
        laplace_prior = self.laplace_prior
        n_actions = self.env.n_actions

        def function(rewards_vector):
            weights = rewards_vector
            states_rewards_matrix = features * rewards_vector
            states_actions_rewards_matrix = np.repeat(states_rewards_matrix[0][..., np.newaxis], n_actions, axis=1)

            _, _, policy, log_policy = self.mdp_solver.linear_value_iteration(states_actions_rewards_matrix)

            sum_log_probabilities = np.sum(np.sum(np.multiply(log_policy, state_action_count)))

            if laplace_prior is not None:
                sum_log_probabilities = sum_log_probabilities - laplace_prior * np.sum(np.abs(weights))

            sum_log_probabilities = - sum_log_probabilities
            state_dist = self.mdp_solver.linear_mdp_frequency(policy, init_state_dist)

            # Gradient calculations
            gradient = feature_expectations - np.dot(features.T, state_dist)
            if laplace_prior is not None:
                gradient = gradient - laplace_prior * np.sign(weights)
            gradient = -gradient

            return sum_log_probabilities

        return function

    def run(self, demonstrations, features=None):
        # demonstrations is shape [N, T, 2] - last dimension is state/action pair
        n_demos, n_steps, _ = demonstrations.shape

        # Build feature membership matrix, shape: n_states by n_features
        if features is None:
            features = np.eye(self.env.n_states)

        # Count features
        _, n_features = features.shape
        assert features.shape[0] == self.env.n_states

        # Compute feature expectations.
        feature_expectations = np.zeros((n_features, 1))
        state_list = np.zeros((n_demos, n_steps))  # populated by state index
        action_list = np.zeros((n_demos, n_steps))  # populated by action index
        state_action_count = np.zeros((self.env.n_states, self.env.n_actions))

        for i in np.arange(n_demos):
            for t in np.arange(n_steps):
                state = np.int(demonstrations[i, t, 0])
                action = np.int(demonstrations[i, t, 1])
                state_list[i, t] = state
                action_list[i, t] = action
                state_action_count[state, action] += 1
                state_vec = np.zeros((self.env.n_states, 1))
                state_vec[state] = 1
                feature_expectations += np.dot(features.T, state_vec)
                # here simple matrix transpose is used assuming no imaginary components of features matrix

        # Generate initial state distribution. Gives number of times a state has been visited across all demos/time
        init_state_dist = np.zeros(self.env.n_states)
        for i in np.arange(n_demos):
            for t in np.arange(n_steps):
                state = np.int(state_list[i, t])
                init_state_dist[state] += 1

        for i in np.arange(n_demos):
            for t in np.arange(n_steps):
                state = state_list[i, t]
                action = action_list[i, t]
                transition_probabilities, possible_next_states = self.env.get_transitions(state, action)
                for s in np.arange(len(possible_next_states)):
                    next_state = np.int(possible_next_states[s])
                    next_state_probability = transition_probabilities[s]
                    init_state_dist[next_state] -= self.env.discount_rate * next_state_probability
                    # Isn't this assuming we know the transition probabilities a priori...?

        # Run unconstrainted non-linear optimization.
        init_rewards = np.random.uniform(0, 1, n_features)[..., np.newaxis]
        function = self.compute_objective(features, feature_expectations, init_state_dist, state_action_count)

        function_output = minimize(function,
                                   init_rewards,
                                   method='BFGS')

        rewards_vector = function_output.x

        # Convert to full tabulated reward.
        weights = rewards_vector
        states_rewards_matrix = features * rewards_vector
        # Return corresponding reward function.
        states_actions_rewards_matrix = np.repeat(states_rewards_matrix[0][..., np.newaxis], self.env.n_actions, axis=1)
        state_values, q_values, policy, log_policy = self.mdp_solver.solve(states_actions_rewards_matrix)
        return state_values, q_values, policy, log_policy, states_actions_rewards_matrix
