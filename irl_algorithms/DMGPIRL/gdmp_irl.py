import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K_bd
from itertools import product
import os

from gpflow.base import Module, Parameter
from gpflow.config import default_float, set_default_float
from gpflow.utilities import ops, positive
from irl_algorithms.utils import get_statistics
from irl_algorithms.DMGPIRL.dmgp_functions import get_demonstrations_spilt
from irl_algorithms.DMGPIRL.dmgp_model import DMGP_Regressor

class DMGPIRL:
    def __init__(self, env, mdp_solver):
        self.env = env
        self.mdp_solver = mdp_solver

    def get_loss(self, model, sa_count):
        model_dmgp = model
        state_action_count = sa_count
        def loss(x):
            r, u = x
            X_u = features(u) # observed states used to train
            X_r = features(r) # all states in env

            # Update DMGP model based on new u input.
            # Get likelihoods:
            # HP likelihood
            hp_log_likelihood = 0

            # GP likelihood
            gp_log_likelihood = 0.5*(model_dmgp.neg_log_likelihood().numpy() + u.size*np.log(2*np.pi))

            # IRL likelihood
            mean_zs, var_zs = model_dmgp.predict_y(X_r)
            states_rewards_matrix = mean_zs
            sa_reward_matrix = np.repeat(states_rewards_matrix[..., np.newaxis], self.env.n_actions, axis=1)
            _, _, policy, log_policy = self.mdp_solver.linear_value_iteration(sa_reward_matrix)
            irl_log_likelihood = np.sum(np.sum(log_policy * state_action_count))

            neg_log_likelihood = - (irl_log_likelihood + gp_log_likelihood + hp_log_likelihood)

            print('Loss :', neg_log_likelihood)
            return neg_log_likelihood
        return loss

    def run(self, demonstrations, use_actions=False, features=True):
        if features is None:
            features = np.eye(self.env.n_states) #one hot encode states - change for non-discrete features
        _, state_action_count, init_state_dist = get_statistics(self.env, demonstrations, features)

        reward_vector = np.ones(features.shape[0])
        reward_tensor = tf.convert_to_tensor(reward_vector)
        features_tensor = tf.convert_to_tensor(features)
        m_in = 20
        d_in = 1

        model_dmgp = DMGP_Regressor(data=(features_tensor, reward_tensor), m=m_in, d=d_in, simple_dnn=True)

        with tf.GradientTape as tape:
            loss = self.get_loss(model_dmgp, state_action_count)
            opt = gpflow.optimizers.Scipy()

            @tf.custom_gradient
            def grad(dr):


            return loss, grad

        # add reward vector to trainable variables

        opt_logs = opt.minimize(loss, [r, model_dmgp.trainable_variables], options=dict(maxiter=500))
        neg_log_lkl_opt = 0.5 * (model_dmgp.neg_log_likelihood().numpy() + u.size * np.log(2 * np.pi))

        mean_f, var_f = model_dmgp.predict_y(X_r)
        mean_f, var_f = mean_f.numpy(), var_f.numpy()

        rewards_vector =
        state_values, q_values, policy, log_policy = self.mdp_solver.solve(states_actions_rewards_matrix)
        return state_values, q_values, policy, log_policy, rewards_vector
