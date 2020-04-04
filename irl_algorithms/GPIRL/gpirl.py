import numpy as np
from scipy.optimize import minimize
from irl_algorithms.GPIRL.gp import GaussianProcess
from irl_algorithms.utils import get_statistics
from irl_algorithms.GPIRL.gpirl_functions import invert_kernel, pack_parameters, unpack_parameters, get_hp_likelihood, get_kernel
np.random.seed(0)

class GPIRL:
    def __init__(self, env, mdp_solver, laplace_prior=None, verbose=False):
        self.env = env
        self.verbose = verbose
        self.laplace_prior = laplace_prior
        self.mdp_solver = mdp_solver

    def optimise(self, gaussian_process, state_action_count):

        def function(x):
            gp, r = unpack_parameters(gaussian_process, x)
            n = gp.n_features
            u = r[gp.X_u_index]

            # Get kernels
            K_uu = get_kernel(gp, gp.X_u, gp.X_u, regularise=True)
            K_ru = get_kernel(gp, gp.X_r, gp.X_u, regularise=False)
            #print('Kuu', K_uu.shape, K_uu)
            #print('Kru', K_ru.shape, K_ru)
            K_uu_inverse = invert_kernel(K_uu, gp.X_u)

            # Get likelihoods:
            # HP likelihood
            hp_log_likelihood = 0
            hp_log_likelihood += get_hp_likelihood(gp.feature_weights, gp.ard_prior, gp.feature_weights_prior_wt, gp, gp.ard_xform)
            hp_log_likelihood += get_hp_likelihood(gp.noise_var, gp.noise_prior, gp.noise_prior_wt, gp, gp.noise_xform)
            hp_log_likelihood += get_hp_likelihood(gp.rbf_var, gp.rbf_prior, gp.rbf_prior_wt, gp, gp.rbf_xform)

            # GP likelihood
            gp_log_likelihood = -0.5 * u.T @ K_uu_inverse @ u - 0.5 * np.log(np.linalg.det(K_uu)) - n/2 * np.log(2*np.pi)

            # IRL likelihood
            states_rewards_matrix = K_ru.T @ K_uu_inverse @ u
            sa_reward_matrix = np.repeat(states_rewards_matrix[..., np.newaxis], self.env.n_actions, axis=1)
            _, _, policy, log_policy = self.mdp_solver.linear_value_iteration(sa_reward_matrix)
            irl_log_likelihood = np.sum(np.sum(log_policy * state_action_count))

            neg_log_likelihood = - (irl_log_likelihood + gp_log_likelihood + hp_log_likelihood)

            print('Loss :', neg_log_likelihood)
            return neg_log_likelihood
        return function

    def run(self, demonstrations, features=None):
        if features is None:
            features = np.eye(self.env.n_states)
        _, state_action_count, init_state_dist = get_statistics(self.env, demonstrations, features)

        # Initialise rewards and GP
        init_rewards = np.random.randint(-10, 0, self.env.n_states)
        gaussian_process = GaussianProcess(self.env, features, inducing_points_selection='all')

        function = self.optimise(gaussian_process, state_action_count)

        # Optimisation
        # 1. High convergence value
        print('Optimising with high tolerance.')
        tol = gaussian_process.restart_tolerance
        max_iter = 20
        best_nll = 1e10000
        options = {'disp': True, 'maxiter': max_iter}
        init_values = pack_parameters(gaussian_process, init_rewards)
        output = minimize(function, init_values, tol=tol,
                          options=options)
        x = output.x
        nll = output.fun
        best_nll = best_nll
        best_x = x

        # 2. Learn kernel params
        print('Learning kernel parameters.')
        _, u = unpack_parameters(gaussian_process, best_x)
        init_values = pack_parameters(gaussian_process, u)
        output = minimize(function, init_values, options=options)
        x = output.x
        nll = output.fun
        if nll < best_nll:
            best_nll = nll
            best_x = x

        # 3. Normal convergence value- Now re - run the best value with normal tolerance.
        print('Optimising with normal tolerance.')
        tol = 1e-9
        options = {'disp': True, 'maxiter': max_iter}
        init_values = best_x
        output = minimize(function, init_values, options=options)
        x = output.x
        nll = output.fun

        # Update gp values
        gp, r = unpack_parameters(gaussian_process, x)
        u = r[gp.X_u_index]

        # Return results
        K_uu = get_kernel(gp, gp.X_u, gp.X_u)
        K_uu_inverse = invert_kernel(K_uu, gp.X_u)
        K_ru = get_kernel(gp, gp.X_r, gp.X_u)
        rewards_vector = K_ru.T @ K_uu_inverse @ u
        states_actions_rewards_matrix = np.repeat(rewards_vector, self.env.n_actions).reshape((self.env.n_states,self.env.n_actions))
        # Return corresponding reward function.
        state_values, q_values, policy, log_policy = self.mdp_solver.solve(states_actions_rewards_matrix)
        return state_values, q_values, policy, log_policy, rewards_vector
