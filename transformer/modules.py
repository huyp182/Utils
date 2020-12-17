import numpy as np
import tensorflow as tf


def layer_normalization(inputs, epsilon=1e-8):
    '''
    Applies layer normalization.
    :param inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    :param epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    :return: A tensor with the same shape and data dtype as `inputs`.
    '''
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer())
    gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer())
    normalized = (inputs - mean) / (variance + epsilon) ** 0.5
    outputs = gamma * normalized + beta

    return outputs
