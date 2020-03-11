import numpy as np
from algorithms.LinearMDP import LinearMDP
np.random.seed(0)

class MaxEnt:
    def __init__(self, env, params, mdp_solver, laplace_prior=None, verbose=False):
        self.env = env
        self.params = params
        self.verbose = verbose
        self.laplace_prior = laplace_prior
        self.mdp_solver = mdp_solver

    def compute_objective(self, reward_function, features, feature_expectations, state_action_expectations):
        weights = reward_function
        reward_function = features*reward_function

        _, _, policy, log_policy = self.mdp_solver.linear_value_iteration()

        sum_log_probabilities = sum(sum(log_policy * state_action_expectations))

        if self.laplace_prior is not None:
            sum_log_probabilities = sum_log_probabilities - self.laplace_prior * sum(abs(weights))

        sum_log_probabilities = - sum_log_probabilities
        state_visitation_count = self.mdp_solver.linear_mdp_frequency()
        gradient = feature_expectations -  np.dot(features.T,state_visitation_count)

        if self.laplace_prior is not None:
            gradient = gradient - self.laplace_prior * np.sign(weights)

        gradient = -gradient
        return sum_log_probabilities, gradient

    def run(self, demonstrations, features_type='none'):
        states = self.env.n_states
        actions = self.env.n_actions
        transitions = self.env.transitons
        n_demos, n_steps = demonstrations.shape()

        # Build feature membership matrix. ?????????
        if features_type == 'true':
            features = feature_data.splittable
            # Add dummy feature.
            features = horzcat(F, ones(states, 1))

        elif features_type == 'learned':
            features = true_features

        else:
            features = np.eye(states)

        # Compute feature expectations.
        feature_expectations = np.zeros((features.shape(), 1))
        state_list = np.zeros((n_demos, n_steps))
        action_list = np.zeros((n_demos, n_steps))
        state_action_count = np.zeros((states, actions))
        for i in np.arange(n_demos):
            for t in np.arange(n_steps):
                state = demonstrations[i,t][0]
                action = demonstrations[i,t][1]
                state_list[i, t] = state
                action_list[i, t] = action
                state_action_count[state, action] +=1
                state_vec = np.zeros((states, 1))
                state_vec[state] = 1
                feature_expectations = feature_expectations + np.dot(features.T, state_vec)

        # Generate initial state distribution for infinite horizon. WHAT IS THIS?????????
        initial_state_dist = np.sum(
            np.sparse(state_list, 1: n_demos * n_steps, np.ones((n_demos * n_steps, 1)), states, n_demos * n_steps)
        * np.ones(n_demos * n_steps, 1), 2) #what is this though
        for i in np.arange(n_demos):
            for t in np.arange(n_steps):
                state = state_list[i, t]
                action = action_list[i, t]
                for transition in np.ranage(transitions): #WHAT IS TRANSITION??????
                    next_state = self.env.step(state, action, transition)
                    initial_state_dist[next_state] = initial_state_dist[next_state] - self.env.discount * self.env.sa_p(state, action, transition)

        function = self.compute_objective
        function_options = 1

        # Initialize reward.
        reward_function = np.random((features, 1))

        # Run unconstrainted non-linear optimization.
        reward_function, _  = minFunc(function, reward_function, function_options)

        # Print timing.
        time = toc
        if verbose != 0:
             fprintf(1, 'Optimization completed in %f seconds.\n', time);

        # Convert to full tabulated reward.
        weights = reward_function
        reward_function = np.dot(features,reward_function)

        # Return corresponding reward function.
        reward_function = repmat(r, 1, actions)
        value_function, q_function, policy = solve_mdp(self.env, reward_function)

        return reward_function, value_function, q_function, policy, time
