import numpy as np


class LONR:
    def __init__(self, env, alpha, gamma):
        self.gamma = gamma
        self.env = env
        self.alpha = alpha
        self.q_table = np.zeros((self.env.n_players, self.env.n_states, self.env.n_actions))

    def train(self, no_regret_algo, episodes, max_steps):
        avg_probabilities = np.zeros((max_steps+1, self.env.n_states, self.env.n_actions))
        for episode in range(episodes):
            choice_algorithm = no_regret_algo(self.env.actions, self.env.states)
            rewards_list = []
            actions_list = []
            states_list = []
            probability_list = np.ones((max_steps+1, self.env.n_states, self.env.n_actions))
            print(f"Episode: {episode}")
            total_reward = 0
            state = self.env.reset
            t = 0
            while t < max_steps:
                t += 1

                action = choice_algorithm.choose_action(state)
                next_state, reward, done = self.env.step(state, action)
                total_reward += reward

                expected_value = 0
                probability = np.zeros(self.env.n_actions)
                for a in range(self.env.n_actions):
                    probability[a] = choice_algorithm.weights[next_state][a]/np.sum(choice_algorithm.weights[next_state])
                    expected_value += choice_algorithm.weights[next_state][a]*self.q_table[state][next_state][a]

                for a in range(self.env.n_actions):
                    if a == action:
                        self.q_table[state, state, a] = 1/probability[a] * (reward[state] + self.gamma*expected_value)
                    else:
                        self.q_table[state, state, a] = 0

                print('qtable', self.q_table)

                # update mwua
                probability_list[t][state] = probability
                no_regret_reward = reward[state] + self.gamma*expected_value
                choice_algorithm.update(state, action, no_regret_reward)

                print('s' , state, 'a', action, 'r', reward)

                rewards_list.append(reward)
                actions_list.append(action)
                states_list.append(state)
                state = next_state

            avg_probabilities += probability_list

        avg_probabilities = avg_probabilities/episodes

        return rewards_list, actions_list, states_list, avg_probabilities
