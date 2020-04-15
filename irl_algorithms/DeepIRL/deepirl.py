import numpy as np
import tensorflow as tf
from irl_algorithms.DeepIRL.deepirl_model import DeepIRLModel
from irl_algorithms.utils import get_statistics

tf.keras.backend.set_floatx('float64')

class DeepIRL:
    def __init__(self, env, mdp_solver):
        self.env = env
        self.mdp_solver = mdp_solver

    def run(self, demonstrations, learning_rate, n_iters, features=None):

        if features is None:
            features = np.eye(self.env.n_states)

        state_action_frequencies, state_frequencies = get_frequencies(demonstrations, self.env)
        model = DeepIRLModel(output_nodes=1)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        # Convert to tensors
        features_tensor = tf.convert_to_tensor(features, dtype='float64')

        for iteration in np.arange(n_iters):
            if iteration % (n_iters / 10) == 0:
                print('Iteration: {}'.format(iteration))

            @tf.custom_gradient
            def get_loss(rewards):
                sa_rewards = rewards.numpy()
                sa_rewards = np.repeat(sa_rewards[..., np.newaxis], self.env.n_actions, axis=1)
                policy = linear_value_iteration(sa_rewards, self.env)
                log_policy = np.log(policy)
                mu_E = policy_propagation(policy, self.env)
                loss = np.sum(log_policy * state_action_frequencies)

                def grad(dr):
                    gradient = state_frequencies - mu_E
                    return dr * gradient
                return loss, grad

            with tf.GradientTape() as tape:
                rewards_vector = model(features_tensor)
                loss = get_loss(rewards_vector)
                print('Loss: ', loss)

            gradient = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(gradient, model.variables))

        states_actions_rewards_matrix = np.repeat(rewards_vector, self.env.n_actions).reshape((self.env.n_states, self.env.n_actions))
        # Return corresponding reward function.
        state_values, q_values, policy, log_policy = self.mdp_solver.solve(states_actions_rewards_matrix)
        rewards_vector = rewards_vector.numpy()
        #rewards_vector /= np.sqrt(np.sum(rewards_vector**2))
        return state_values, q_values, policy, log_policy, rewards_vector


def get_frequencies(demonstrations, env):
    n_demos, n_steps, _ = demonstrations.shape
    state_action_frequencies = np.zeros((env.n_states, env.n_actions))
    for demo in np.arange(n_demos):
        for step in np.arange(n_steps):
            state, action = demonstrations[demo, step]
            state = np.int(state)
            action = np.int(action)
            state_action_frequencies[state, action] += 1
    state_frequencies = np.sum(state_action_frequencies, axis=1)
    return state_action_frequencies, state_frequencies[..., np.newaxis]


def policy_propagation(policy, env):
    """

    :param policy: [N_STATES, N_ACTIONS], values of s-a pairs
    :param env: environment used in IRL formulation
    :return: expected state visitation frequency under the given policy [N_STATES, 1]
    """
    expected_state_frequency = np.ones((env.n_states, 1))
    for state in np.arange(env.n_states):
        for action in np.arange(env.n_actions):
            for next_state in np.arange(env.n_states):
                transition_probability = env.transition_probabilities[state, action, next_state]
                policy_value = policy[state, action]
                value = transition_probability * policy_value * expected_state_frequency[next_state]
                expected_state_frequency[state] += value
    return expected_state_frequency


def linear_value_iteration(reward_matrix, env):
    diff = 1.0
    convergence_value = 1e-10
    state_values = np.zeros(env.n_states)
    q_values = np.zeros((env.n_states, env.n_actions))
    policy = np.zeros((env.n_states, env.n_actions))

    while diff >= convergence_value:
        old_state_values = state_values
        for state in np.arange(env.n_states):
            for action in np.arange(env.n_actions):
                q_values[state, action] += reward_matrix[state, action]
                for next_state in np.arange(env.n_states):
                    _value = state_values[next_state]
                    _probability = env.transition_probabilities[state, action, next_state]
                    q_values[state, action] += _value * _probability
            state_values[state] = compute_softmax(q_values[state])
            diff = np.max(np.abs(state_values - old_state_values))

    for state in np.arange(env.n_states):
        for action in np.arange(env.n_actions):
            policy[state, action] = np.exp(q_values[state, action] - state_values[state])
    policy += 1e-10
    return policy


def compute_softmax(q_values):
    max_q_value = np.amax(q_values)  # get max along each row
    state_values = max_q_value + np.log(np.sum(np.exp(q_values - max_q_value)))
    return state_values

