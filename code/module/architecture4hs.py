import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import tensorflow_addons as tfa
from tensorflow.keras import layers, activations
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import TimeDistributed
import tensorflow.keras.backend as K

from .densenet import DenseNet_hs, DenseNet_qrs
from .tcn import TCN, tcn_full_summary


def encoder():
    x_in = Input(shape=(4000, 1))
    # Feature Mapping
    ## I Part
    feat_1 = layers.Conv1D(filters=8,
                           kernel_size=11,
                           dilation_rate=1,
                           padding='same',
                           activation=tf.nn.leaky_relu)(x_in)
    feat_1 = layers.Conv1D(filters=8,
                            kernel_size=11,
                            dilation_rate=1,
                            padding='same',
                            activation=tf.nn.leaky_relu)(feat_1)
    feat_1 = layers.Conv1D(filters=8,
                            kernel_size=11,
                            dilation_rate=1,
                            padding='same',
                            activation=None)(feat_1)
    feat_1 = tfa.layers.InstanceNormalization()(feat_1)
    feat_1 = activations.relu(feat_1)
    feat_1 = layers.MaxPool1D()(feat_1)
    feat_1 = layers.Conv1D(filters=16,
                            kernel_size=7,
                            dilation_rate=1,
                            padding='same',
                            activation=tf.nn.leaky_relu)(feat_1)
    feat_1 = layers.Conv1D(filters=16,
                            kernel_size=7,
                            dilation_rate=1,
                            padding='same',
                            activation=tf.nn.leaky_relu)(feat_1)
    feat_1 = layers.Conv1D(filters=16,
                            kernel_size=7,
                            dilation_rate=1,
                            padding='same',
                            activation=None)(feat_1)
    feat_1 = tfa.layers.InstanceNormalization()(feat_1)
    feat_1 = activations.relu(feat_1)
    feat_1 = layers.MaxPool1D()(feat_1)
    feat_1 = layers.Conv1D(filters=32,
                            kernel_size=5,
                            dilation_rate=1,
                            padding='same',
                            activation=tf.nn.leaky_relu)(feat_1)
    feat_1 = layers.Conv1D(filters=32,
                            kernel_size=5,
                            dilation_rate=1,
                            padding='same',
                            activation=tf.nn.leaky_relu)(feat_1)
    feat_1 = layers.Conv1D(filters=32,
                            kernel_size=5,
                            dilation_rate=1,
                            padding='same',
                            activation=None)(feat_1)
    feat_1 = tfa.layers.InstanceNormalization()(feat_1)
    feat_1 = activations.relu(feat_1)
    feat_1 = layers.MaxPool1D()(feat_1)
    feat_1 = layers.Conv1D(filters=64,
                            kernel_size=3,
                            dilation_rate=1,
                            padding='same',
                            activation=tf.nn.leaky_relu)(feat_1)
    feat_1 = layers.Conv1D(filters=64,
                            kernel_size=3,
                            dilation_rate=1,
                            padding='same',
                            activation=tf.nn.leaky_relu)(feat_1)
    feat_1 = layers.Conv1D(filters=64,
                            kernel_size=3,
                            dilation_rate=1,
                            padding='same',
                            activation=None)(feat_1)
    feat_1 = tfa.layers.InstanceNormalization()(feat_1)
    feat_1 = activations.relu(feat_1)
    feat_1 = layers.MaxPool1D()(feat_1)
    

    ## II Part
    feat_2 = layers.Conv1D(filters=8,
                            kernel_size=11,
                            dilation_rate=2,
                            padding='same',
                            activation=tf.nn.leaky_relu)(x_in)
    feat_2 = layers.Conv1D(filters=8,
                            kernel_size=11,
                            dilation_rate=2,
                            padding='same',
                            activation=tf.nn.leaky_relu)(feat_2)
    feat_2 = layers.Conv1D(filters=8,
                            kernel_size=11,
                            dilation_rate=2,
                            padding='same',
                            activation=None)(feat_2)
    feat_2 = tfa.layers.InstanceNormalization()(feat_2)
    feat_2 = activations.relu(feat_2)
    feat_2 = layers.MaxPool1D()(feat_2)
    feat_2 = layers.Conv1D(filters=16,
                            kernel_size=7,
                            dilation_rate=4,
                            padding='same',
                            activation=tf.nn.leaky_relu)(feat_2)
    feat_2 = layers.Conv1D(filters=16,
                            kernel_size=7,
                            dilation_rate=4,
                            padding='same',
                            activation=tf.nn.leaky_relu)(feat_2)
    feat_2 = layers.Conv1D(filters=16,
                            kernel_size=7,
                            dilation_rate=4,
                            padding='same',
                            activation=None)(feat_2)
    feat_2 = tfa.layers.InstanceNormalization()(feat_2)
    feat_2 = activations.relu(feat_2)
    feat_2 = layers.MaxPool1D()(feat_2)
    feat_2 = layers.Conv1D(filters=32,
                            kernel_size=5,
                            dilation_rate=8,
                            padding='same',
                            activation=tf.nn.leaky_relu)(feat_2)
    feat_2 = layers.Conv1D(filters=32,
                            kernel_size=5,
                            dilation_rate=8,
                            padding='same',
                            activation=tf.nn.leaky_relu)(feat_2)
    feat_2 = layers.Conv1D(filters=32,
                            kernel_size=5,
                            dilation_rate=8,
                            padding='same',
                            activation=None)(feat_2)
    feat_2 = tfa.layers.InstanceNormalization()(feat_2)
    feat_2 = activations.relu(feat_2)
    feat_2 = layers.MaxPool1D()(feat_2)
    feat_2 = layers.Conv1D(filters=64,
                            kernel_size=3,
                            dilation_rate=16,
                            padding='same',
                            activation=tf.nn.leaky_relu)(feat_2)
    feat_2 = layers.Conv1D(filters=64,
                            kernel_size=3,
                            dilation_rate=16,
                            padding='same',
                            activation=tf.nn.leaky_relu)(feat_2)
    feat_2 = layers.Conv1D(filters=64,
                            kernel_size=3,
                            dilation_rate=16,
                            padding='same',
                            activation=None)(feat_2)
    feat_2 = tfa.layers.InstanceNormalization()(feat_2)
    feat_2 = activations.relu(feat_2)
    feat_2 = layers.MaxPool1D()(feat_2)

    ## III Part
    feat_3 = layers.Conv1D(filters=8,
                            kernel_size=11,
                            dilation_rate=4,
                            padding='same',
                            activation=tf.nn.leaky_relu)(x_in)
    feat_3 = layers.Conv1D(filters=8,
                            kernel_size=11,
                            dilation_rate=4,
                            padding='same',
                            activation=tf.nn.leaky_relu)(feat_3)
    feat_3 = layers.Conv1D(filters=8,
                            kernel_size=11,
                            dilation_rate=4,
                            padding='same',
                            activation=None)(feat_3)
    feat_3 = tfa.layers.InstanceNormalization()(feat_3)
    feat_3 = activations.relu(feat_3)
    feat_3 = layers.MaxPool1D()(feat_3)
    feat_3 = layers.Conv1D(filters=16,
                            kernel_size=7,
                            dilation_rate=8,
                            padding='same',
                            activation=tf.nn.leaky_relu)(feat_3)
    feat_3 = layers.Conv1D(filters=16,
                            kernel_size=7,
                            dilation_rate=8,
                            padding='same',
                            activation=tf.nn.leaky_relu)(feat_3)
    feat_3 = layers.Conv1D(filters=16,
                            kernel_size=7,
                            dilation_rate=8,
                            padding='same',
                            activation=None)(feat_3)
    feat_3 = tfa.layers.InstanceNormalization()(feat_3)
    feat_3 = activations.relu(feat_3)
    feat_3 = layers.MaxPool1D()(feat_3)
    feat_3 = layers.Conv1D(filters=32,
                            kernel_size=5,
                            dilation_rate=16,
                            padding='same',
                            activation=tf.nn.leaky_relu)(feat_3)
    feat_3 = layers.Conv1D(filters=32,
                            kernel_size=5,
                            dilation_rate=16,
                            padding='same',
                            activation=tf.nn.leaky_relu)(feat_3)
    feat_3 = layers.Conv1D(filters=32,
                            kernel_size=5,
                            dilation_rate=16,
                            padding='same',
                            activation=None)(feat_3)
    feat_3 = tfa.layers.InstanceNormalization()(feat_3)
    feat_3 = activations.relu(feat_3)
    feat_3 = layers.MaxPool1D()(feat_3)
    feat_3 = layers.Conv1D(filters=64,
                            kernel_size=3,
                            dilation_rate=32,
                            padding='same',
                            activation=tf.nn.leaky_relu)(feat_3)
    feat_3 = layers.Conv1D(filters=64,
                            kernel_size=3,
                            dilation_rate=32,
                            padding='same',
                            activation=tf.nn.leaky_relu)(feat_3)
    feat_3 = layers.Conv1D(filters=64,
                            kernel_size=3,
                            dilation_rate=32,
                            padding='same',
                            activation=None)(feat_3)
    feat_3 = tfa.layers.InstanceNormalization()(feat_3)
    feat_3 = activations.relu(feat_3)
    feat_3 = layers.MaxPool1D()(feat_3)

    ## Merge
    feat = layers.Concatenate(axis=-1)([feat_1, feat_2, feat_3])

    ## self Attention
    attention = layers.GlobalAveragePooling1D()(feat) # (None, 24)
    attention = layers.Dense(24, activation=tf.nn.leaky_relu)(attention)
    attention = layers.Dense(192, activation='tanh')(attention)
    attention = layers.Reshape((1, 192))(attention)
    feat = layers.Multiply()([attention, feat])

    feat = layers.Conv1D(filters=192,
                         kernel_size=1,
                         dilation_rate=1,
                         padding='same',
                         activation=None)(feat)

    return Model(x_in, feat, name='encoder')

def hs_densenet_encoder():
    dense_en = DenseNet_hs(num_init_features=16, growth_rate=4, block_layers=[3, 5, 8], compression_rate=0.5)
    
    return dense_en

def tcn_encoder():
    x_in = Input(shape=(4000, 1))

    feat = TCN(nb_filters=24,
               kernel_size=11,
               nb_stacks=1,
               dilations=(1, 2, 4),
               padding='same',
               return_sequences=True,
               use_instance_norm=True,
               activation='relu')(x_in)
    feat = layers.MaxPool1D()(feat)

    feat = TCN(nb_filters=48,
               kernel_size=7,
               nb_stacks=1,
               dilations=(2, 4, 8),
               padding='same',
               return_sequences=True,
               use_instance_norm=True,
               activation='relu')(feat)
    feat = layers.MaxPool1D()(feat)

    feat = TCN(nb_filters=96,
               kernel_size=5,
               nb_stacks=1,
               dilations=(4, 8, 16),
               padding='same',
               return_sequences=True,
               use_instance_norm=True,
               activation='relu')(feat)
    feat = layers.MaxPool1D()(feat)
    
    feat = TCN(nb_filters=96,
               kernel_size=3,
               nb_stacks=1,
               dilations=(4, 8, 16),
               padding='same',
               return_sequences=True,
               use_instance_norm=True,
               activation='relu')(feat)
    feat = layers.MaxPool1D()(feat)

    feat = layers.Conv1D(filters=192,
                         kernel_size=1,
                         dilation_rate=1,
                         padding='same',
                         activation=None)(feat)
    
    return Model(x_in, feat, name='tcn_encoder')

def se_encoder():
    x_in = Input(shape=(4000, 1))

    feat = TCN(nb_filters=24,
               kernel_size=11,
               nb_stacks=1,
               dilations=(1, 2, 4),
               padding='same',
               return_sequences=True,
               use_instance_norm=True,
               activation='relu')(x_in)
    feat = layers.MaxPool1D()(feat)

    feat = TCN(nb_filters=48,
               kernel_size=7,
               nb_stacks=1,
               dilations=(2, 4, 8),
               padding='same',
               return_sequences=True,
               use_instance_norm=True,
               activation='relu')(feat)
    feat = layers.MaxPool1D()(feat)

    feat = TCN(nb_filters=96,
               kernel_size=5,
               nb_stacks=1,
               dilations=(4, 8, 16),
               padding='same',
               return_sequences=True,
               use_instance_norm=True,
               activation='relu')(feat)
    feat = layers.MaxPool1D()(feat)
    
    feat = TCN(nb_filters=96,
               kernel_size=5,
               nb_stacks=1,
               dilations=(16, 32, 64),
               padding='same',
               return_sequences=True,
               use_instance_norm=True,
               activation='relu')(feat)
    feat = layers.MaxPool1D()(feat)
    ## self Attention
    attention = layers.GlobalAveragePooling1D()(feat) # (None, 24)
    attention = layers.Dense(24, activation='relu')(attention)
    attention = layers.Dense(96, activation='tanh')(attention)
    attention = layers.Reshape((1, 96))(attention)
    feat = layers.Multiply()([attention, feat])
    
    feat = layers.Conv1D(filters=192,
                         kernel_size=1,
                         dilation_rate=1,
                         padding='same',
                         activation=None)(feat)
    
    return Model(x_in, feat, name='se_encoder')

def decoder():
    x_in = Input(shape=(250, 192))
    feat = layers.Dense(96, activation=tf.nn.leaky_relu)(x_in)
    feat = layers.Dense(48, activation=tf.nn.leaky_relu)(feat)
    feat = layers.Dense(24, activation=tf.nn.leaky_relu)(feat)
    logits = layers.Dense(4, activation=None)(feat)
    logits = tf.nn.softmax(logits)

    return Model(x_in, logits, name='decoder')

def q_predictor():
    x_in = Input(shape=(250, 192))
    mu = layers.Dense(24, activation=tf.nn.leaky_relu)(x_in)
    mu = layers.Dense(192, activation=None)(mu)

    logvar = layers.Dense(24, activation=tf.nn.leaky_relu)(x_in)
    logvar = layers.Dense(192, activation='tanh')(logvar)

    return Model(x_in, [mu, logvar], name='q_predictor')

def t_predictor():
    x_in = Input(shape=(250, 192))
    mu = layers.Dense(24, activation=tf.nn.leaky_relu)(x_in)
    mu = layers.Dense(192, activation=None)(mu)

    logvar = layers.Dense(24, activation=tf.nn.leaky_relu)(x_in)
    logvar = layers.Dense(192, activation='tanh')(logvar)
    
    return Model(x_in, [mu, logvar], name='t_predictor')


"""
def decoder2():
    x_in = Input(shape=(250, 192))
    feat = layers.Dense(48, activation=tf.nn.leaky_relu)(x_in)
    logits = layers.Dense(4, activation=None)(feat)
    logits = tf.nn.softmax(logits)

    return Model(x_in, logits, name='decoder')
"""