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
    normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
    outputs = gamma * normalized + beta

    return outputs


def multihead_attention(queries, keys, values, key_masks, num_heads=8, dropout_rate=0, training=True, causality=False):
    '''
    Applies multihead attention.
    :param queries: A 3d tensor with shape of [N, T_q, d_model].
    :param keys: A 3d tensor with shape of [N, T_k, d_model].
    :param values: A 3d tensor with shape of [N, T_k, d_model].
    :param key_masks: A 2d tensor with shape of [N, key_seqlen].
    :param num_heads: An int. Number of heads.
    :param dropout_rate: A floating point number.
    :param training: Boolean. Controller of mechanism for dropout.
    :param causality: Boolean. If true, units that reference the future are masked.
    :return: A 3d tensor with shape of (N, T_q, C).
    '''
    d_model = queries.get_shape().as_list()[-1]

    # Linear projections
    Q = tf.layers.dense(queries, d_model, use_bias=True)  # (N, T_q, d_model)
    K = tf.layers.dense(keys, d_model, use_bias=True)  # (N, T_k, d_model)
    V = tf.layers.dense(values, d_model, use_bias=True)  # (N, T_k, d_model)

    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

    # Attention
    outputs = scaled_dot_product_attention(Q_, K_, V_, key_masks, causality, dropout_rate, training)

    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

    # Residual connection
    outputs += queries

    # Normalize
    outputs = layer_normalization(outputs)

    return outputs


def scaled_dot_product_attention(Q, K, V, key_masks, causality=False, dropout_rate=0., training=True):
    '''
    :param Q: Packed queries. 3d tensor. [N, T_q, d_k].
    :param K: Packed keys. 3d tensor. [N, T_k, d_k].
    :param V: Packed values. 3d tensor. [N, T_k, d_v].
    :param key_masks: A 2d tensor with shape of [N, key_seqlen]
    :param causality: If True, applies masking for future blinding
    :param dropout_rate: A floating point number of [0, 1].
    :param training: boolean for controlling dropout
    :return:
    '''
    d_k = Q.get_shape().as_list()[-1]

    # dot product
    outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

    # scale
    outputs /= d_k ** 0.5

    # key masking
    outputs = mask(outputs, key_masks=key_masks, type='key')

    # causality or future blinding masking
    if causality:
        outputs = mask(outputs, type='future')

    # softmax
    outputs = tf.nn.softmax(outputs)

    # dropout
    outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

    # weighted sum (context vectors)
    outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs


def mask(inputs, key_masks=None, type=None):
    '''
    Masks paddings to inputs
    :param inputs: 3d tensor. (h*N, T_q, T_k)
    :param key_masks: 3d tensor. (N, 1, T_k)
    :param type: string. 'key' | 'future'
    :return:
    '''
    padding_num = -2 ** 32 + 1
    if type == 'key':
        key_masks = tf.to_float(key_masks)
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1])  # (h*N, seqlen)
        key_masks = tf.expand_dims(key_masks, 1) # (h*N, 1, seqlen)
        outputs = inputs + key_masks * padding_num
    elif type == 'future':
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)
        paddings = tf.ones_like(future_masks) * padding_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
    else:
        print('Check if you entered type correctly!')

    return outputs
