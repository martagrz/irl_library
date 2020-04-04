import tensorflow as tf
import numpy as np
from irl_algorithms.NPIRL.np import NPModel
import tensorflow_probability as tfp
import matplotlib.pyplot as plt


def get_y(x):
    y = np.sin(2*x) + 3*np.cos(x/2) + np.random.normal(loc=0, scale=0.01)
    return y


N = 20
x_context = np.random.uniform(0, 1, N)
y_context = get_y(x_context)
y_context = np.reshape(y_context, (1, N, 1))
x_context = np.reshape(x_context, (1, N, 1))

x_target = np.linspace(0, 1, N)
y_target = get_y(x_target)
y_target = np.reshape(y_target, (1, N, 1))
x_target = np.reshape(x_target, (1, N, 1))

"""f, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(x_context, y_context)
ax1.set_title("Context data")
ax2.scatter(x_target, y_target, color='red')
ax2.set_title("Target data")
plt.show(f)
plt.close(f)"""

print(x_context.shape)
print(y_context.shape)
print(x_target.shape)
print(y_target.shape)

model = NPModel(x_context, y_context, x_target)

optimizer = tf.keras.optimizers.Adam(1e-2)

model.compile(optimizer=optimizer, loss=model.loss, metrics=[model.rms])

model.fit(x_target, y_target, batch_size=7, epochs=20000)

prediction = model.predict(x_target)
params = prediction[:, :, :2]
mu, log_sigma = np.split(params, 2, axis=-1)
sigma = np.exp(log_sigma)
y_pred = tfp.distributions.Normal(loc=mu, scale=sigma)

print(y_target.shape)
print(x_target.shape)
print(mu.shape)
plt.plot(x_target[0, :, 0], y_target[0, :, 0], label='True', color='tab:brown')
plt.plot(x_target[0, :, 0], mu[0, :, 0], label='mu')
plt.fill_between(x_target[0, :, 0], mu[0, :, 0]+sigma[0,:,0], mu[0, :, 0]-sigma[0,:,0], alpha=0.2)
plt.legend()
plt.show()
plt.close()


print('Done')

