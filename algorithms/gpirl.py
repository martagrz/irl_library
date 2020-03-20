import numpy as np
from scipy.optimize import minimize
from algorithms.gp import GaussianProcess
from algorithms.utils import get_statistics
from algorithms.gpirl_functions import hyper_parameter_transform, pack_parameters, unpack_parameters, get_hp_gradient, get_hp_prior, get_kernel
np.random.seed(0)

class GPIRL:
    def __init__(self, env, mdp_solver, laplace_prior=None, verbose=False):
        self.env = env
        self.verbose = verbose
        self.laplace_prior = laplace_prior
        self.mdp_solver = mdp_solver

    def optimise(self, gaussian_process, state_action_count, init_state_dist):

        def function(x):
            # Set constants.
            x = x[..., np.newaxis]
            n_states, n_actions = state_action_count.shape
            n_samples = len(gaussian_process.inducing_points_index)
            # Unpack parameters.
            gp, u = unpack_parameters(gaussian_process, x)
            # Compute kernel and kernel matrix derivatives.
            Kstar, logDetAndPPrior, alpha, Kinv, dhp, dhpdr = get_kernel(gp, u)
            # Compute GP likelihood.
            gp_log_lik = -0.5 * u.T @ alpha - logDetAndPPrior - 0.5 * n_samples * np.log(2*np.pi)
            # Add hyperparameter priors.
            hp_log_lik = 0
            hp_log_lik = hp_log_lik + get_hp_prior(gp.inv_widths, gp.ard_prior,  gp.ard_prior_wt, gp.ard_xform, gp)
            hp_log_lik = hp_log_lik + get_hp_prior(gp.noise_var, gp.noise_prior,  gp.noise_prior_wt, gp.noise_xform, gp)
            hp_log_lik = hp_log_lik + get_hp_prior(gp.rbf_var, gp.rbf_prior, gp.rbf_prior_wt, gp.rbf_xform, gp)
            # Compute reward under deterministic conditional approximation.
            rewards = Kstar.T @ alpha
            states_actions_rewards_matrix = np.tile(rewards, (n_states, n_actions))
            # Run value iteration to get policy.
            # print(states_actions_rewards_matrix.shape)
            _, _, policy, log_policy = self.mdp_solver.linear_value_iteration(states_actions_rewards_matrix)
            # Compute value by adding up log example probabilities.
            data_log_lik = np.sum(np.sum(log_policy * state_action_count))
            # Compute total log likelihood and invert for descent.
            loss = data_log_lik + gp_log_lik + hp_log_lik
            loss = -loss[0]
            # Add hyperparameter prior gradients.
            dhp[:gp.inv_widths.shape[1]] = dhp[:gp.inv_widths.shape[1]] + get_hp_gradient(gp.inv_widths, gp.ard_prior, gp.ard_prior_wt, gp.ard_xform, gp)
            idx = gp.inv_widths.shape[1]
            if gp.learn_noise:
                dhp[idx] = dhp[idx] + get_hp_gradient(gp.noise_var, gp.noise_prior, gp.noise_prior_wt, gp.noise_xform, gp)
                idx = idx + 1
            if gp.learn_rbf:
                dhp[idx] = dhp[idx] + get_hp_gradient(gp.rbf_var, gp.rbf_prior, gp.rbf_prior_wt, gp.rbf_xform, gp)
                idx = idx + 1
            # Compute state visitation count D and reward gradient.
            state_dist = self.mdp_solver.linear_mdp_frequency(policy, init_state_dist)[..., np.newaxis]
            drew = np.sum(state_action_count, axis=1)[..., np.newaxis] - state_dist
            # Apply posterior Jacobian.
            dr = Kinv @ (Kstar @ drew)
            dhp = dhp + dhpdr @ drew
            # Add derivative of GP likelihood.
            dudr = np.eye(n_samples)
            dr = dr - dudr @ alpha
            # Combine and invert for descent.
            gradient = np.vstack((dr, dhp))
            gradient = - gradient.flatten()
            print('Loss :', loss)
            return loss
        return function

    def run(self, demonstrations, inducing_points='all', features=None):
        if features is None:
            features = np.eye(self.env.n_states)
        _, n_features = features.shape
        _, state_action_count, init_state_dist = get_statistics(self.env, demonstrations, features)

        # Initialise rewards and GP
        init_rewards = np.random.randint(-10, 0, (5, self.env.n_states, 1))
        gaussian_process = GaussianProcess(self.env, features, inducing_points='all')
        function = self.optimise(gaussian_process, state_action_count, init_state_dist)

        # Optimisation
        # 1. High convergence value
        print('Optimising with high tolerance.')
        tol = gaussian_process.restart_tolerance
        options = {'disp': True}
        best_nll = float("inf")
        for k in np.arange(len(init_rewards)):
            print('Run', k, ' / ', len(init_rewards)-1)
            init_values = pack_parameters(gaussian_process, init_rewards[k][gaussian_process.inducing_points_index])
            output = minimize(function, init_values, tol=tol, options=options) # jac=True,
            x = output.x
            nll = output.fun
            if nll < best_nll:
                best_nll = nll
                best_x = x

        best_x = best_x[..., np.newaxis]

        # 2. Learn kernel params
        print('Learning kernel parameters.')
        _, u = unpack_parameters(gaussian_process, best_x)
        output = minimize(function, pack_parameters(gaussian_process, u), options=options) # jac=True,
        x = output.x
        nll = output.fun
        if nll < best_nll:
            best_nll = nll
            best_x = x

        best_x = best_x[..., np.newaxis]

        # 3. Normal convergence value- Now re - run the best value with normal tolerance.
        print('Optimising with normal tolerance.')
        tol = 1e-4
        max_iter = 3000
        options = {'disp': True, 'maxiter': max_iter}
        init_values = best_x
        output = minimize(function, init_values, options=options, tol=tol) # jac=True,
        best_x = output.x
        best_nll, = output.fun

        best_x = best_x[..., np.newaxis]

        # Update gp values
        gp, u = unpack_parameters(gaussian_process, best_x)
        nll = best_nll

        # Return results
        gaussian_process.Y = u
        Kstar, _, alpha, _, _, _ = get_kernel(gp, u)
        rewards_vector = Kstar.T @ alpha
        states_actions_rewards_matrix = np.tile(rewards_vector, (1, self.env.n_actions))
        # Return corresponding reward function.
        state_values, q_values, policy, log_policy = self.mdp_solver.solve(states_actions_rewards_matrix)
        return state_values, q_values, policy, log_policy, rewards_vector
