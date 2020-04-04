import gym
import tensorflow as tf
import numpy as np 
from model import DQN

tf.random.set_seed(1)
np.random.seed(1)


env = gym.make('Taxi-v2').env
episodes = 100
gamma = 0.6
epsilon = 0.01
optimizer = tf.optimizers.Adam(1e-3)

model = DQN(env,optimizer,gamma,epsilon)
model.train(episodes= episodes)





