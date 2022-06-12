import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D
from keras.layers import BatchNormalization, Dropout, concatenate, Reshape
from keras.layers import Dense, Activation
from keras.layers import Conv2DTranspose


# @title ****Decoder****
def decoder_block_layers(inputs, filters, kernel_size = (3, 3), initializer = 'he_normal'):
    x = inputs
    for _ in range(2):
        x = Conv2D(filters = filters, kernel_size = kernel_size, padding = 'same', kernel_initializer = initializer)(x)
        output = Activation('relu')(x)
    return output


def decoder_block(inputs, concat, filters, kernel_size = (3, 3), stride = (2, 2), dropout_rate = 0.3):
    conv = inputs
    conv = Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = stride, padding = 'same')(conv)
    x = concatenate([conv, concat])
    x = decoder_block_layers(x, filters = filters)
    output = Dropout(rate = dropout_rate)(x)
    return output


def decoder(inputs, concat, OUTPUT_CHANNELS = 3):
    x = inputs
    c1, c2, c3, c4 = concat
    x = decoder_block(x, c4, filters = 512)
    x = decoder_block(x, c3, filters = 256)
    x = decoder_block(x, c2, filters = 128)
    x = decoder_block(x, c1, filters = 64)
    output = Conv2D(filters = OUTPUT_CHANNELS, kernel_size = (1, 1), activation = 'softmax')(x)
    return output

