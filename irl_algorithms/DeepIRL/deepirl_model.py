# https://github.com/yrlu/irl-imitation/blob/master/deep_maxent_irl.py was a useful reference

import tensorflow as tf


class DeepIRLModel(tf.keras.Model):
    def __init__(self, output_nodes=1):
        super(DeepIRLModel, self).__init__()
        self.dense_1 = tf.keras.layers.Dense(100, activation=tf.nn.sigmoid)
        self.dense_2 = tf.keras.layers.Dense(100, activation=tf.nn.sigmoid)
        self.output_layer = tf.keras.layers.Dense(output_nodes)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        rewards = self.output_layer(x)
        return rewards

