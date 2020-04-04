import numpy as np
import gym
import random
from tqdm import tqdm
from algos.utils import get_index

class MonteCarlo:
    def __init__(self,env,alpha,gamma,epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = env
        self.q_table = np.zeros((env.observation_space_n, env.action_space_n))
    
    def get_q_table(self,episodes=100001):
     
        all_epochs = []
        all_penalties = []
        r_list = np.zeros(episodes)
        print('Monte Carlo')

        for i in tqdm(range(1, episodes)):
            state = self.env.reset
            state_list = []
            action_list = []
            reward_list= []

            cum_rewards = 0
            epochs, penalties, reward = 0, 0, 0
            done = False


            while not done:

                state_index = get_index(state,self.env.observation_space)

                if random.uniform(0, 1) < self.epsilon:
                    action = np.random.choice(self.env.action_space) # Explore action space
                    action_index = np.where(self.env.action_space == action)
                else:
                    action_index = np.argmax(self.q_table[state_index]) # Exploit learned values
                    action = self.env.action_space[action_index]

                state_list.append(state)
                action_list.append(action)
                
                next_state, reward, done = self.env.step(state,action)

                reward_list.append(reward)

                if reward == 0:
                    penalties += 1
                cum_rewards += reward

                state = next_state
                epochs += 1
            
            for k in range(len(state_list)):
                state = state_list[k]
                action = action_list[k]
                action_index = np.where(self.env.action_space == action)
                state_index = get_index(state,self.env.observation_space)
                reward = reward_list[k:]
                discount = [self.gamma**j for j in range(len(reward))]
                discounted_reward = np.dot(reward,discount)
                
                self.q_table[state_index,action_index] = (1-self.alpha)*self.q_table[state_index,action_index] + self.alpha*discounted_reward

            #if i % 1000 == 0:
            #    clear_output(wait=True)
            #    print(f"Episode: {i}")

            r_list[i] = cum_rewards

        #print("Training finished.\n")

        return self.q_table, r_list

    def evaluate_agent(self,episodes=1):

        total_epochs, total_penalties = 0, 0
        reward_list = []
                
        for _ in range(episodes):
            state = self.env.reset
            epochs, penalties, reward = 0, 0, 0
    
            done = False
    
            while not done:
                state_index = get_index(state,self.env.observation_space)
                action_index = np.argmax(self.q_table[state_index])
                action = self.env.action_space[action_index]
                state, reward, done = self.env.step(state,action)
                reward_list.append(reward)
                
                if reward == 0:
                    penalties += 1
                    
                epochs += 1

            total_penalties += penalties
            total_epochs += epochs

        return reward_list
