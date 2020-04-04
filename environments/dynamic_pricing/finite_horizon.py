# Environment with a finite selling horizon with infinite stock, where the quantity demanded increases over the horizon.
# The firm only chooses the price at which to sell.

import numpy as np


class FiniteHorizonEnv:
    def __init__(self, demand_model, demand_params, state_params, action_params, discount_rate=0.2):
        self.discount_rate = discount_rate
        self.time_steps = state_params
        self.states = np.arange(1, self.time_steps+1, 1)
        self.n_states = len(self.states)

        low_price = action_params[0]
        high_price = action_params[1]
        price_steps = action_params[2]
        self.actions = np.arange(low_price, high_price, price_steps)
        self.n_actions = len(self.actions)

        # Init demand variables
        self.demand_params = demand_params
        self.demand_model = demand_model(demand_params, stochastic=False)

        self.transition_probabilities = np.zeros((self.n_states, self.n_actions))
        self.possible_next_states = np.zeros((self.n_states, self.n_actions))

        for state_index in np.arange(self.n_states):
            probabilities, next_states = self.get_transitions(state_index, action_index=None)
            self.transition_probabilities[state_index] = np.array([probabilities])
            self.possible_next_states[state_index] = np.array([next_states])

    def get_transitions(self, state_index, action_index):
        state = self.states[state_index]
        next_state = np.array([state + 1])
        next_state_index = next_state - self.states[0]
        transition_probabilities = np.array([1])
        if next_state_index == self.n_states:
            next_state_index -= 1
        return transition_probabilities, next_state_index

    def reset(self):
        state = np.array([0])
        return state

    def step(self, state_index, price_index):
        state = self.states[state_index]
        price = self.actions[price_index]
        done = False
        next_state = state + 1
        demand = np.int(self.demand_model(price, state))
        reward = np.round(demand*price,0)
        if next_state == np.max(self.states):
            done = True
        next_state_index = next_state - self.states[0]
        return next_state_index, reward, done

