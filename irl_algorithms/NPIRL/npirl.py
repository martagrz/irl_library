import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from irl_algorithms.NPIRL.np import NPModel
from irl_algorithms.NPIRL.np_functions import get_demonstrations_spilt
import matplotlib.pyplot as plt

class NPIRL:
    def __init__(self, env, mdp_solver, laplace_prior=None, verbose=False):
        self.env = env
        self.mdp_solver = mdp_solver

    def run(self, demonstrations, batch_size, epochs):
        # demonstrations is shape [N, T, 2] - last dimension is state/action pair
        n, t, _ = demonstrations.shape
        [x_context, y_context], [x_target, y_target] = get_demonstrations_spilt(demonstrations)

        # pass through LSTM first since data is sequential - but in some environments we don't care about sequential
        # then pass through NP
        model = NPModel(x_context, y_context, x_target)
        optimizer = tf.keras.optimizers.Adam(1e-2)
        model.compile(optimizer=optimizer, loss=model.loss, metrics=[model.rms])
        model.fit(x_target, y_target, batch_size=batch_size, epochs=epochs)

        prediction = model.predict(x_target)
        params = prediction[:, :, :2]
        mu, log_sigma = np.split(params, 2, axis=-1)
        sigma = np.exp(log_sigma)
        y_pred = tfp.distributions.Normal(loc=mu, scale=sigma)

        plt.scatter(x_target[0, :, 0], mu[0, :, 0], label='mu')
        plt.show()
        plt.close()

        plt.scatter(x_target[0, :, 0], y_target[0, :, 0], label='True', color='tab:brown')
        plt.scatter(x_target[0, :, 0], mu[0, :, 0], label='mu')
        plt.fill_between(x_target[0, :, 0], mu[0, :, 0] + sigma[0, :, 0], mu[0, :, 0] - sigma[0, :, 0], alpha=0.2)
        plt.legend()
        plt.show()
        plt.close()


        # predict optimal action from every possible state - returns policy

        # extract the sa_rewards matrix from the neural process or otherwise

        #states_actions_rewards_matrix =
        #state_values, q_values, policy, log_policy = self.mdp_solver.solve(states_actions_rewards_matrix)
        return y_pred, mu, sigma