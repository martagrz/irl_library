import numpy as np
from scipy.optimize import minimize
from algorithms.utils import get_statistics
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
            states_rewards_matrix = features @ rewards_vector
            states_actions_rewards_matrix = np.repeat(states_rewards_matrix[..., np.newaxis], n_actions, axis=1)
            _, _, policy, log_policy = self.mdp_solver.linear_value_iteration(states_actions_rewards_matrix)

            sum_log_probabilities = np.sum((log_policy * state_action_count))
            if laplace_prior is not None:
                sum_log_probabilities = sum_log_probabilities - laplace_prior * np.sum(np.abs(weights))

            sum_log_probabilities = - sum_log_probabilities

            state_dist = self.mdp_solver.linear_mdp_frequency(policy, init_state_dist)

            # Gradient calculations
            gradient = feature_expectations - features.T @ state_dist
            if laplace_prior is not None:
                gradient = gradient - laplace_prior * np.sign(weights)
            gradient = -gradient

            print('Loss: ', sum_log_probabilities)

            return sum_log_probabilities, gradient

        return function

    def run(self, demonstrations, features=None):
        if features is None:
            features = np.eye(self.env.n_states)
        _, n_features = features.shape

        feature_expectations, state_action_count, init_state_dist = get_statistics(self.env, demonstrations, features)

        # Run unconstrainted non-linear optimization.
        init_rewards = np.repeat(-10, n_features)[..., np.newaxis]
        function = self.compute_objective(features, feature_expectations, init_state_dist, state_action_count)

        function_output = minimize(function,
                                   init_rewards,
                                   method='BFGS',
                                   jac=True,
                                   options={'disp': True})

        rewards_vector = function_output.x
        print('Learned rewards: ', rewards_vector.reshape((self.env.n_rows, self.env.n_cols)))

        # Convert to full tabulated reward.
        weights = rewards_vector
        states_rewards_matrix = features @ rewards_vector
        states_actions_rewards_matrix = np.repeat(states_rewards_matrix[..., np.newaxis], self.env.n_actions, axis=1)
        # Return corresponding reward function.
        state_values, q_values, policy, log_policy = self.mdp_solver.solve(states_actions_rewards_matrix)

        return state_values, q_values, policy, log_policy, rewards_vector
