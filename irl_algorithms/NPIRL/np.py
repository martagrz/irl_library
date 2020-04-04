import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def get_deterministic_encoder():
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(64, activation="relu"),
                                        tf.keras.layers.Dense(2, activation=None)])
    return model

def get_latent_encoder():
    encoder_input = tf.keras.Input(shape=(1000, 2))
    x = tf.keras.layers.Dense(64, activation='relu')(encoder_input)
    x = tf.keras.layers.Dense(64, activation=None)(x)
    x = tf.reduce_mean(x, axis=0, keepdims=True)
    mu = tf.keras.layers.Dense(1)(x)
    log_sigma = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=encoder_input, outputs=[mu, log_sigma])
    return model

def get_decoder():
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(64, activation="relu"),
                                        tf.keras.layers.Dense(2, activation=None)])
    return model



class NPModel(tf.keras.Model):
    def __init__(self, x_context, y_context, x_target):
        super(NPModel, self).__init__()
        self._latent_encoder = get_latent_encoder()
        self._deterministic_encoder = get_deterministic_encoder()
        self._decoder = get_decoder()
        self.x_context = x_context
        self.y_context = y_context
        self.num_targets = 20
        self.num_latent = 64

    def conglomerate(self, tensor):
        return tf.reduce_mean(tensor, axis=0, keepdims=True)

    def call(self, x_target):
        encoder_input = tf.concat([self.x_context, self.y_context], axis=-1)
        mu, log_sigma = self._latent_encoder(encoder_input)
        sigma = tf.exp(log_sigma)
        latent_rep = tf.random.normal(mu.shape) * sigma + mu  # Reparametarisation trick - allows for gradient flow
        deterministic_rep = self._deterministic_encoder(encoder_input)
        representation = tf.concat([deterministic_rep, latent_rep],
                                   axis=-1)  # Need to tile along 0 axis and concat 1 axis
        decoder_input = tf.concat([representation, x_target], axis=-1)
        params_decoder = self._decoder(decoder_input)
        return tf.concat([params_decoder, x_target], axis=-1)

    def rms(self, y_target, output):
        params_decoder = output[:, :, :2]
        mu, log_sigma = tf.split(params_decoder, 2, axis=-1)
        mse = tf.reduce_mean(tf.math.square(y_target - mu))
        return tf.math.sqrt(mse)

    def loss(self, y_target, output):
        params_decoder = output[:, :, :2]
        x_target = output[:, :, 2:]
        mu, log_sigma = tf.split(params_decoder, 2, axis=-1)
        sigma = tf.exp(log_sigma)
        dist = tfp.distributions.MultivariateNormalDiag(mu, sigma)
        log_p = dist.log_prob(y_target)

        encoder_input_context = tf.concat([self.x_context, self.y_context], axis=-1)
        mu_context, log_sigma_context = self._latent_encoder(encoder_input_context)
        sigma_context = tf.exp(log_sigma_context)
        prior = tfp.distributions.Normal(loc=mu_context, scale=sigma_context)

        encoder_input_target = tf.concat([x_target, y_target], axis=-1)
        mu_target, log_sigma_target = self._latent_encoder(encoder_input_target)
        sigma_target = tf.exp(log_sigma_target)
        posterior = tfp.distributions.Normal(loc=mu_target, scale=sigma_target)

        kl = tf.reduce_sum(tfp.distributions.kl_divergence(posterior, prior), axis=-1, keepdims=True)
        loss = - tf.reduce_mean(log_p - kl / tf.cast(100, tf.float32))
        return loss

