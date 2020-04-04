# Environment with a finite selling horizon with finite stock.
# The firm chooses the price at which to sell and how much stock to sell.
# Demand is stochastic, arriving as a Poisson process.
# state_params = [[init_stock, stock_min, stock_max, stock_cost], time_max]
# action_params = [[price_min, price_max, price_steps], [restock_min, restock_max, restock_steps]]
import numpy as np


class FiniteStockHorizonEnv:
    def __init__(self, demand_model, demand_params, state_params, action_params, stochastic=True, discount_rate=0.2, pre_train_transition=None):
        self.discount_rate = discount_rate
        # Get states
        self.init_stock = state_params[0][0]
        self.stock_min = state_params[0][1]
        self.stock_max = state_params[0][2]
        self.stock_holding_cost = state_params[0][3]
        self.stock_range = np.arange(self.stock_min, self.stock_max, 1)

        self.time_steps = state_params[1]
        self.time_range = np.arange(1, self.time_steps, 1)

        xx, yy = np.meshgrid(self.stock_range, self.time_range)
        xx_flat = np.reshape(xx, -1)
        yy_flat = np.reshape(yy, -1)
        self.states = np.stack((xx_flat, yy_flat), axis=1)  # all possible pairs of states
        self.n_states = len(self.states)
        self.states_index = np.arange(self.n_states)

        # Get actions
        price_min, price_max, price_steps = action_params[0]
        self.price_range = np.arange(price_min, price_max, price_steps)
        self.n_prices = len(self.price_range)

        restock_min, restock_max, restock_steps = action_params[1]
        self.restock_range = np.arange(restock_min, restock_max, restock_steps)
        self.n_restock_options = len(self.restock_range)

        xx, yy = np.meshgrid(self.price_range, self.restock_range)
        xx_flat = np.reshape(xx, -1)
        yy_flat = np.reshape(yy, -1)
        self.actions = np.stack((xx_flat, yy_flat), axis=1)  # all possible pairs of actions
        self.n_actions = len(self.actions)
        self.actions_index = np.arange(self.n_actions)

        # Init demand variables
        self.demand_params = demand_params
        self.demand_model = demand_model(demand_params, stochastic)

        if pre_train_transition is None:
            self.transition_probabilities, self.possible_next_states, self.sa_reward_matrix = self.get_transitions(runs=100)
        else:
            [self.transition_probabilities, self.possible_next_states], self.sa_reward_matrix = pre_train_transition

    def get_transitions(self, runs):
        print('Calculating transition probabilities.')
        # for each state and action, run this for enough times to get the eqm distribution over demand
        # want to get the probability of moving to any stock - all stock values are technically possible
        # but time is always +1
        transition_probabilities = np.zeros((self.n_states, self.n_actions, len(self.stock_range)))
        possible_next_states_index = np.zeros(transition_probabilities.shape)
        sa_reward_matrix = np.zeros((self.n_states, self.n_actions))

        for state_index in np.arange(self.n_states):
            stock, time = self.get_values_from_index(state_index, mode='state')
            new_time = time + 1
            if new_time == self.time_steps:
                new_time = time

            next_states = self.states[np.asarray(self.states[:, 1] == new_time).nonzero()]
            next_states_index = np.zeros(next_states.shape[0])

            for i in np.arange(len(next_states)):
                next_states_index[i] = self.get_index_from_values(next_states[i], mode='state')

            probabilities = np.zeros(next_states_index.shape)

            for action_index in np.arange(self.n_actions):
                for _ in np.arange(runs):
                    ns_index, reward, _ = self.step(state_index, action_index)
                    sa_reward_matrix[state_index, action_index] += reward

                    indx = np.asarray(ns_index == next_states_index).nonzero()[0]
                    if indx.ndim == 0:
                        print('state', state_index, 'action', action_index)
                        print(next_states_index)
                        print(ns_index)
                        print(indx)
                        raise ValueError('Index cannot be empty.')
                    probabilities[indx] += 1

                probabilities = probabilities/np.sum(probabilities)
                sa_reward_matrix[state_index, action_index] = sa_reward_matrix[state_index, action_index]/runs
                transition_probabilities[state_index, action_index] = probabilities
                possible_next_states_index[state_index, action_index] = next_states_index
        print('Transition probabilities obtained.')
        return transition_probabilities, possible_next_states_index, sa_reward_matrix

    def reset(self):
        init_state = np.array([self.init_stock, 0])
        return init_state

    def get_index_from_values(self, values, mode):
        a = values[0]  # state: stock; action: price
        b = values[1]  # state time; action: restock
        if mode == 'state':  # obtain state index
            n_a = len(self.stock_range)
            index = a + (b-1)*n_a # time range starts at 1
        elif mode == 'action':  # obtain action index
            n_a = self.n_prices
            index = a + b*n_a # is it b or b-1?
        return index

    def get_values_from_index(self, index, mode):
        if mode == 'state':  # obtain state values
            values = self.states[index]
        elif mode == 'action':  # obtain action values
            values = self.actions[index]
        return values

    def step(self, current_state_index, action_index):
        done = False
        current_state = self.get_values_from_index(current_state_index, mode='state')
        stock, time = current_state
        action = self.get_values_from_index(action_index, mode='action')
        price, reorder = action

        new_stock = stock + reorder

        demand = np.int(self.demand_model(price, time))
        if new_stock < demand:
            demand = new_stock

        reward = np.round(demand * price, 0)
        new_stock = new_stock - demand
        time = time + 1

        if time == self.time_steps:
            time = time - 1
            done = True

        penalty = 0
        if new_stock < 0:
            penalty += 10
            new_stock = 0
            done = True
        elif new_stock >= self.stock_max-1:
            new_stock = self.stock_max-1
            penalty += 10

        holding_stock_cost = new_stock * self.stock_holding_cost
        reward = np.round(reward - penalty - holding_stock_cost, 0)
        next_state = np.array([new_stock, time])
        next_state_index = self.get_index_from_values(next_state, mode='state')
        return next_state_index, reward, done

    def evaluate_policy(self, policy, max_steps):
        done = False
        t = 0
        state = self.reset()
        state_index = self.get_index_from_values(state, mode='state')
        cumulative_reward = 0
        reward_list = []
        cumulative_reward_list = []
        while not done:
            action_index = np.int(policy[state_index])
            next_state_index, reward, done = self.step(state_index, action_index)
            state_index = next_state_index
            cumulative_reward += reward
            reward_list.append(reward)
            cumulative_reward_list.append(cumulative_reward)
            t += 1
            if t == max_steps:
                done = True
        print('Total steps: ', t, '/', max_steps)
        print('Cumulative reward: ', cumulative_reward)
        return reward_list, cumulative_reward_list
