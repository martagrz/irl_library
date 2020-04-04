import numpy as np
from tqdm import tqdm


class NashQLearning:
    def __init__(self, env, alpha, gamma, epsilon, error_variance, error=False):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.env = env
        self.error = error
        self.error_variance = error_variance
        self.q_tables = np.ones((self.env.n_players, self.env.n_players, self.env.grid_size, len(self.env.actions)))

    def get_index(self, state):
        row = state[0]
        col = state[1]
        index = np.int(row * self.env.n_cols + col)
        return index

    def choose_action(self, player, current_state_index, greedy=False):
        if greedy:
            action = np.argmax(self.q_tables[player, player, current_state_index])
        else:
            if np.random.uniform(0, 1) < self.epsilon:
                action = np.random.choice(self.env.actions, 1)
            else:
                action = np.argmax(self.q_tables[player, player, current_state_index])
        return action

    def get_policies(self, player, state, greedy=False):
        policies = np.ones(len(self.env.actions))
        for other_player in range(self.env.n_players):
            state_index = self.get_index(state[other_player])
            greedy_action = np.argmax(self.q_tables[player, other_player, state_index])
            if not greedy:
                action_probability_distribution = np.ones(len(self.env.actions))
                action_probability_distribution *= self.epsilon / len(self.env.actions)
                action_probability_distribution[greedy_action] += 1 - self.epsilon
            else:
                action_probability_distribution = np.zeros(len(self.env.actions))
                action_probability_distribution[greedy_action] = 1
            policies = np.dot(policies, action_probability_distribution)
        return policies

    def get_optimal_policy(self, player):
        optimal_policy = np.zeros(self.env.grid_size)
        for state_index in range(self.env.grid_size):
            optimal_policy[state_index] = np.argmax(self.q_tables[player][player][state_index])
        return optimal_policy

    def train(self, episodes):
        number_steps = np.zeros(episodes)
        for _ in tqdm(range(episodes)):
            state = self.env.reset()
            actions = np.zeros(self.env.n_players, dtype=int)
            done = np.zeros(self.env.n_players)
            i = 0
            cumulative_rewards = np.zeros(self.env.n_players)
            while not done.all():

                for player in range(self.env.n_players):
                    current_state_index = self.get_index(state[player])
                    if not done[player]:
                        actions[player] = np.int(self.choose_action(player, current_state_index))
                    else:
                        actions[player] = 4  # Stays in the same place
                next_state, rewards, done = self.env.step(state, actions)

                for n in range(self.env.n_players):
                    policies = self.get_policies(n, next_state)
                    if not done[n]:
                        for m in range(self.env.n_players):
                            current_state_index = self.get_index(state[m])
                            next_state_index = self.get_index(next_state[m])
                            next_state_q = self.q_tables[n, m, next_state_index]
                            nash_q = np.dot(policies, next_state_q)
                            assert not np.isnan(nash_q)
                            if m != n:
                                reward = rewards[m] + np.random.normal(0, self.error_variance)
                                value = (1 - self.alpha) * self.q_tables[n, m, current_state_index, actions[m]] + \
                                        self.alpha * (reward + self.gamma * nash_q)

                            else:
                                value = (1 - self.alpha) * self.q_tables[n, m, current_state_index, actions[m]] + \
                                        self.alpha * (rewards[m] + self.gamma * nash_q)
                            self.q_tables[n, m, current_state_index, actions[m]] = value

                state = next_state
                cumulative_rewards += rewards
                i += 1

            number_steps[_] = i

        optimal_policies = np.zeros((self.env.n_players, self.env.grid_size))
        for player in range(self.env.n_players):
            optimal_policies[player, :] = self.get_optimal_policy(player)

        return self.q_tables, optimal_policies, number_steps

    def evaluate(self):
        state = self.env.reset()
        actions = np.zeros(self.env.n_players, dtype=int)
        action_path = []
        rewards_list = []
        done = np.zeros(self.env.n_players)
        while not np.all(done):
            for player in range(self.env.n_players):
                current_state_index = self.get_index(state[player])
                if not done[player]:
                    actions[player] = self.choose_action(player, current_state_index, greedy=True)
                else:
                    actions[player] = 4  # stays in the same place

            next_state, rewards, done = self.env.step(state, actions)

            state = next_state
            action_path.append(actions)
            rewards_list.append(rewards)

        return action_path, rewards_list
