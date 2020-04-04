import numpy as np
import random
from tqdm import tqdm
from algos.utils import get_index


class QLearning:
    def __init__(self,env,alpha,gamma,epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = env
        self.q_table = np.zeros((env.observation_space_n, env.action_space_n))
       
    def get_q_table(self, episodes=100001):
     
        rewards_list = np.zeros(episodes)
        print('Q-learning')
        for i in tqdm(range(1, episodes)):
            state = self.env.reset
            epochs, penalties, reward = 0, 0, 0
            cum_rewards = 0
            done = False
            while not done:
                state_index = get_index(state, self.env.observation_space)
                if random.uniform(0, 1) < self.epsilon:
                    action = np.random.choice(self.env.action_space)  # Explore action space
                    action_index = np.where(self.env.action_space == action)
                else:
                    action_index = np.argmax(self.q_table[state_index])  # Exploit learned values
                    action = self.env.action_space[action_index]

                next_state, reward, done = self.env.step(state, action)
                old_value = self.q_table[state_index, action_index]
                next_state_index = get_index(next_state,self.env.observation_space)
                next_max = np.max(self.q_table[next_state_index])
                new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                self.q_table[state_index, action_index] = new_value
                cum_rewards += reward

                if reward == 0:
                    penalties += 1

                state = next_state
                epochs += 1
            rewards_list[i] = cum_rewards

        return self.q_table, rewards_list

    def evaluate_agent(self, test_episodes=100):

        total_epochs, total_penalties = 0, 0
        save_policies = np.zeros((test_episodes,self.env.timesteps-1))
                
        for _ in range(test_episodes):
            state = self.env.reset
            epochs, penalties, reward = 0, 0, 0
   
            reward_list = []
            policy = []
            stock_list = []
            done = False
    
            while not done:
                state_index = get_index(state,self.env.observation_space)
                action_index = np.argmax(self.q_table[state_index])
                action = self.env.action_space[action_index]
                state, reward, done = self.env.step(state,action)
                reward_list.append(reward)
                policy.append(action)
                #print((state))
                if isinstance(state, int) == False:
                    stock_list.append(state[1])
                
                if reward == 0:
                    penalties += 1
                    
                epochs += 1

            total_penalties += penalties
            total_epochs += epochs
            save_policies[_] = policy

        return reward_list, stock_list, policy
