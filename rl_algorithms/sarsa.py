import numpy as np
import gym
import random
from tqdm import tqdm
from algos.utils import get_index

class SARSA:
    def __init__(self,env,alpha,gamma,epsilon):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((env.observation_space_n, env.action_space_n))

    def get_q_table(self,episodes=100):
        
        all_epochs = []
        all_penalties = []
        #print('before for loop')
        rewards_list = np.zeros(episodes)
        print("SARSA")

        for i in tqdm(range(1, episodes)):
            state = self.env.reset
            cum_rewards = 0
            epochs, penalties, reward = 0, 0, 0
            done = False #terminal state param

            while not done:

                state_index = get_index(state,self.env.observation_space)

                if random.uniform(0, 1) < self.epsilon:
                    action = np.random.choice(self.env.action_space) # Explore action space
                    action_index = np.where(self.env.action_space == action) 

                else:
                    action_index = np.argmax(self.q_table[state_index]) # Exploit learned values
                    action = self.env.action_space[action_index]

                next_state, reward, done = self.env.step(state,action)
 
                next_state_index = get_index(next_state,self.env.observation_space)

                next_action_index = np.argmax(self.q_table[next_state_index])
                next_action = self.env.action_space[next_action_index]
                #print(next_state_index,next_action_index)
                #print(state_index,action_index)
                delta = reward + self.gamma*self.q_table[next_state_index,next_action_index] - self.q_table[state_index,action_index]

                self.q_table[state_index,action_index] = self.q_table[state_index,action_index] + self.alpha * (delta)
                cum_rewards += reward
               
                
                state = next_state   #move to next state
                epochs += 1
                
            #if i % 1000 == 0:
            #    clear_output(wait=True)
            #    print(f"Episode: {i}")

            rewards_list[i] = cum_rewards

        #print('Training finished.')

        return self.q_table, rewards_list #, self.eligibility_trace

    def evaluate_agent(self,episodes=1):

        total_epochs, total_penalties = 0, 0
                
        for _ in range(episodes):
            rewards_list = []
            state = self.env.reset
            epochs, penalties, reward = 0, 0, 0            
            done = False
    
            while not done:
                state_index = get_index(state,self.env.observation_space)

                action_index = np.argmax(self.q_table[state_index])
                action = self.env.action_space[action_index]
                state, reward, done = self.env.step(state,action)
                rewards_list.append(reward)
                #print('state',state)
                if reward == 0:
                    penalties += 1
                     
                epochs += 1
                
            total_penalties += penalties
            total_epochs += epochs

        return rewards_list
