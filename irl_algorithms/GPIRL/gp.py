import numpy as np
from irl_algorithms.GPIRL.gpirl_functions import hp_transform


class GaussianProcess:
    def __init__(self, env, features, inducing_points_selection='all'):
        self.restart_tolerance = 1e1
        self.env = env
        # Initialize X.
        self.X_r = features
        self.n_states, self.n_features = features.shape
        self.X_u_index, self.X_u = self.get_inducing_points(self.X_r, selection=inducing_points_selection)
        self.n_features = features.shape[1]

        # Copy transform and prior information from parameters.
        self.noise_init = np.array([1e-2])
        self.rbf_init = np.array([5])
        self.ard_init = np.array([1])
        self.ard_prior = 'logsparsity'
        self.noise_prior = 'g0'
        self.rbf_prior = 'none'
        self.feature_weights_prior_wt = np.repeat(1, self.n_features)
        self.noise_prior_wt = 1
        self.rbf_prior_wt = 1
        self.ard_xform = 'exp'
        self.noise_xform = 'exp'
        self.rbf_xform = 'exp'

        # Transform hp values
        self.rbf_var = hp_transform(self.rbf_init, self.rbf_xform, 1)
        self.noise_var = hp_transform(self.noise_init, self.noise_xform, 1)
        self.feature_weights = hp_transform(self.ard_init*np.ones(self.n_features), self.ard_xform, 1)

        # Specify which values to optimize and how to optimize them.
        self.learn_noise = 0
        self.learn_rbf = 1
        self.gamma_shape = 2

    def get_inducing_points(self, X, selection='all'):
        if type(selection) != int:
            if selection == 'all':
                inducing_points_index = np.arange(self.env.n_states)
                inducing_points = X[inducing_points_index]
        elif type(selection) == int:
            inducing_points_index = np.random.randint(0, self.env.n_states, selection)
            inducing_points = X[inducing_points_index]
        return inducing_points_index, inducing_points
