import numpy as np
import gym
import random
import tensorflow as tf

def dense_nn(act_size):
    hidden = tf.keras.models.Sequential([
         tf.keras.layers.Dense(32, activation='relu'),
         tf.keras.layers.Dense(32),           
         tf.keras.layers.Dense(32, activation='relu'),
         tf.keras.layers.Dense(act_size)
         ])
    return hidden

class ReplayBuffer:
    def __init__(self):
        self.collection = []

    def add(self,item):
        if len(self.collection) > 20000:
            index = random.randint(0,9999)
            self.collection[index] = item
            self.collection
        else:
            self.collection.append(item)

        return self.collection


class DQN:
    def __init__(self, env, optimizer, gamma=0.05, epsilon=0.01):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.optimizer = optimizer
        self.act_size = self.env.action_space.n
        self.actions = np.arange(1,self.act_size,1)
        self.encoded_actions = tf.one_hot(self.actions,self.act_size,1.0,0.0)
        self.n_states = self.env.observation_space.n
        self.transition_buffer = []
        self.q_target = dense_nn(self.act_size)
        self.q_primary = dense_nn(self.act_size)

    def loss(self,y_target,y_pred):
        loss = tf.reduce_mean(tf.math.square(y_pred - tf.stop_gradient(y_target)))
        return loss
                                   
    def train_step(self,state,action,reward,next_state):
        with tf.GradientTape() as tape:
            y_target = self.q_target(next_state) #returns a vector of q values for each action
            max_q_target = tf.reduce_max(y_target * self.encoded_actions)
            y_target = reward - self.gamma * max_q_target           
            
            y_pred = tf.reduce_max(self.q_primary(state)) 
            loss = self.loss(y_target,y_pred)

        gradients = tape.gradient(loss, self.q_primary.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_primary.trainable_variables))
        return loss

    def get_action(self,state):
        q_values = self.q_primary(state)
        #tf.print(q_values)
        if np.random.random() < self.epsilon:
            best_action = self.env.action_space.sample()
        else:
            best_action = tf.math.argmax(q_values,axis=-1)
            best_action = tf.reshape(best_action, [])
            best_action = best_action.numpy()
        #tf.print('best_action',best_action)
        return best_action
        
    def train(self,episodes):
        rewards_history = []
        steps_history = []
        replay_buffer = ReplayBuffer()
        loss_history = []
        for episode in range(episodes):
            rewards, steps, losses = 0, 0, 0
            penalties, successes = 0, 0 
            state = self.env.reset() #Reset the environment at each episode
            #one hot encode states
            done = False
            while not done:
                #tf.print('state',state)
                encoded_state = tf.one_hot(state,self.n_states,1.0,0.0)
                encoded_state = tf.expand_dims(encoded_state, 0) #add batch_size dim
                action = self.get_action(encoded_state) 
                #tf.print(action)
                #print(action)
                next_state, reward, done, info = self.env.step(action)
                #tf.print('next_state',next_state)
                encoded_next_state = tf.one_hot(next_state,self.n_states,1.0,0.0)
                encoded_next_state = tf.expand_dims(encoded_next_state, 0)
                steps += 1
                rewards += reward
                #print('reward',reward)

                if reward == -10: 
                    penalties += 1
                replay_buffer.add((encoded_state,action,reward,encoded_next_state))
                state = next_state

                sample = random.sample(replay_buffer.collection,1)#sample from transitions
                #tf.print('sample',sample[0])
                (encoded_state, action, reward, encoded_next_state) = sample[0]
                loss = self.train_step(encoded_state,action,reward,encoded_next_state)
                losses += loss

                if steps % 100 == 0: 
                    #update parameters of q_target
                    self.q_target = tf.keras.models.clone_model(self.q_primary)
                    tf.print('Episode:', episode, 'Steps: ', steps, 'Penalties:', penalties, 'Reward:', reward, 'Loss:', loss)
                    
                #if steps == 50000:
                #    break
                
            #replay_buffer.append(traj)
            loss_history.append(losses)
            rewards_history.append(rewards)
            steps_history.append(steps)

            #PRINT EPISODE RESULTS
            tf.print('Episode:', episode, 'Reward:',rewards, 'Steps:',steps,'Loss:',loss)
            
        #PRINT OVERALL RESULTS AND PLOT
    

