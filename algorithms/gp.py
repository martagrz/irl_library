import numpy as np
from scipy.optimize import minimize
np.random.seed(0)


class GPIRL:
    def __init__(self, env, mdp_solver, laplace_prior=None, verbose=False):
        self.env = env
        self.verbose = verbose
        self.laplace_prior = laplace_prior
        self.mdp_solver = mdp_solver

    def compute_objective(self, features, feature_expectations, init_state_dist, state_action_count):
        laplace_prior = self.laplace_prior
        n_actions = self.env.n_actions

        def function(rewards_vector):
            sum_log_probabilities = rewards_vector
            return sum_log_probabilities

        return function

    def run(self, demonstrations, features=None):
        # demonstrations is shape [N, T, 2] - last dimension is state/action pair
        n_demos, n_steps, _ = demonstrations.shape

        # Build feature membership matrix, shape: n_states by n_features
        if features is None:
            features = np.eye(self.env.n_states)


        state_values, q_values, policy, log_policy = self.mdp_solver.solve(states_actions_rewards_matrix)
        return state_values, q_values, policy, log_policy, states_actions_rewards_matrix
