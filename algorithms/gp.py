import numpy as np
from algorithms.gpirl_functions import hyper_parameter_transform


class GaussianProcess:
    def __init__(self, env, features, inducing_points='all'):
        self.restart_tolerance = 1e-4
        # Initialize X.
        self.X = features
        self.env = env
        if inducing_points is not None:
            if inducing_points == 'all':
                inducing_points_index = np.arange(self.env.n_states)
                self.inducing_points_index = inducing_points_index
                self.inducing_points = self.X[inducing_points_index]
        else:
            inducing_points_index = np.random.randint(0, self.env.n_states, 10)
            self.inducing_points_index = inducing_points_index
            self.inducing_points = self.X[inducing_points_index]

        # Copy transform and prior information from parameters.
        self.noise_init = 1
        self.rbf_init = 1e-2
        self.ard_init = 5
        self.ard_xform = 'exp'
        self.noise_xform = 'exp'
        self.rbf_xform = 'exp'
        self.ard_prior = 'logsparsity'
        self.noise_prior = 'g0'
        self.rbf_prior = 'none'
        self.ard_prior_wt = 1
        self.noise_prior_wt = 1
        self.rbf_prior_wt = 1

        # Initialize hyperparameters.
        self.noise_var = hyper_parameter_transform(self.noise_init, [], self.noise_xform, 3)
        self.rbf_var = hyper_parameter_transform(self.rbf_init, [], self.ard_xform, 3)
        self.inv_widths = hyper_parameter_transform(self.ard_init * np.ones((1, features.shape[1])), [], self.ard_xform, 3)

        # Specify which values to optimize and how to optimize them.
        self.learn_noise = 0
        self.learn_rbf = 1
        self.gamma_shape = 2
