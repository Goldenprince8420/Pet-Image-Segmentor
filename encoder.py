import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D
from keras.layers import BatchNormalization, Dropout, concatenate, Reshape
from keras.layers import Dense, Activation


# @title ****Encoder****


def encoder_block_layers(input_data, filters, kernel_size = (3, 3), initializer = 'he_normal'):
    x = input_data
    for _ in range(2):
        x = Conv2D(filters = filters, kernel_size = kernel_size, padding= 'same', kernel_initializer = initializer)(x)
        x = Activation('relu')(x)
    return x


def encoder_block(inputs, filters, pool_size = (2, 2), dropout_rate = 0.3):
    x = inputs
    x = encoder_block_layers(x, filters = filters)
    appended = x
    x = MaxPool2D()(x)
    x = Dropout(rate = dropout_rate)(x)
    encoded = x
    return appended, encoded


def encoder(inputs):
    x = inputs
    f1, x = encoder_block(x, filters = 64)
    f2, x = encoder_block(x, filters = 128)
    f3, x = encoder_block(x, filters = 256)
    f4, x = encoder_block(x, filters = 512)
    return (f1, f2, f3, f4), x

