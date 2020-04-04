from no_regret import MWUA
from lonr import LONR
from env import SendKeep
import numpy as np
import matplotlib.pyplot as plt


alpha = 0.2
gamma = 0.2
episodes = 10
max_steps = 10
env = SendKeep()

timestep_reward = np.zeros((1, episodes))
training_algo = LONR(env, alpha, gamma)

rewards_list, actions_list, states_list, avg_probabilities = training_algo.train(MWUA, episodes, max_steps)


p1_keep = []
p1_send = []
p2_keep = []
p2_send = []
for step in range(max_steps+1):
    p1_keep.append(avg_probabilities[step][0][0])
    p1_send.append(avg_probabilities[step][0][1])
    p2_keep.append(avg_probabilities[step][1][0])
    p2_send.append(avg_probabilities[step][1][1])

plt.plot(range(max_steps+1), p1_keep, label='Player 1, Keep')
plt.plot(range(max_steps+1), p1_send, label='Player 1, Send')
plt.plot(range(max_steps+1), p2_keep, label='Player 2, Keep')
plt.plot(range(max_steps+1), p2_keep, label='Player 2, Send')
plt.xlabel('Time step')
plt.ylabel('Probability')
plt.legend()
plt.show()
plt.close()
