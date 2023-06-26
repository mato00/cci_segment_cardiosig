import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy

import math

@tf.function
def lld_loss_func(z, mu, logvar):
    positive = -(mu - z) ** 2 / 2. / tf.exp(logvar)
    loss = -1 * tf.reduce_mean(tf.reduce_sum(positive, -1))

    return loss

@tf.function
def y_loss_func4qrs(logits, labels):
    loss = tf.reduce_mean(BinaryCrossentropy()(labels, logits))

    return loss

@tf.function
def y_loss_func4hs(logits, labels):
    loss = tf.reduce_mean(CategoricalCrossentropy()(labels, logits))

    return loss

@tf.function
def mi_upper_loss_func(z, mu, logvar):
    ###
    mu = tf.stop_gradient(mu)
    logvar = tf.stop_gradient(logvar)
    ###

    n_index = tf.random.shuffle(tf.range(0, K.shape(z)[0]))
    z_sf_n = tf.gather(z, n_index, axis=0)

    positive = -(mu - z) ** 2 / 2. / tf.exp(logvar)
    negative = -(mu - z_sf_n) ** 2 / 2. / tf.exp(logvar)

    loss = tf.reduce_mean(abs(tf.reduce_sum(positive, -1) - tf.reduce_sum(negative, -1)))

    return loss

@tf.function
def sim_loss_func(z1, z2):
    z1 = tf.stop_gradient(z1)
    z1 = tf.math.l2_normalize(z1, axis=-1)
    z2 = tf.math.l2_normalize(z2, axis=-1)

    loss = - tf.reduce_mean(tf.reduce_sum((z1*z2), axis=-1)) + 1.

    return loss

@tf.function
def ora_loss_func(z1, z2):
    z1 = tf.math.l2_normalize(z1, axis=-1)
    z2 = tf.math.l2_normalize(z2, axis=-1)

    loss = tf.reduce_mean(tf.reduce_sum((tf.math.abs(z1*z2)), axis=-1))

    return loss

"""    
def sim_loss_func(z, q1, q2):
    z = tf.stop_gradient(z)

    z = tf.math.l2_normalize(z, axis=-1)
    q1 = tf.math.l2_normalize(q1, axis=-1)
    q2 = tf.math.l2_normalize(q2, axis=-1)

    loss1 = - tf.reduce_mean(tf.reduce_sum((z*q1), axis=-1)) + 1.
    loss2 = - tf.reduce_mean(tf.reduce_sum((z*q2), axis=-1)) + 1.

    return (loss1 + loss2) / 2.


@tf.function
def sim_loss_func(p_score, n_score):

    loss = - tf.reduce_mean(tf.math.log(p_score + 1e-6) + tf.math.log(1. - n_score + 1e-6))

    return loss
"""

def opt_func(lr):

    # return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    return tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)