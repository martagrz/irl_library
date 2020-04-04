import numpy as np


class MWUA:
    def __init__(self, actions, states):
        self.actions = actions
        self.states = states
        self.n_actions = len(actions)
        self.n_states = len(states)
        self.weights = np.ones((self.n_states, self.n_actions))

    def choose_action(self, state):
        action_weights = self.weights[state]
        probabilities = action_weights/np.sum(action_weights)
        print(probabilities, 'probs')
        action = np.random.choice(self.actions, p=probabilities)
        return action

    def update(self, state, action, reward):
        self.weights[state][action] *= 1 + reward*0.2

