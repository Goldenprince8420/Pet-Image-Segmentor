# @title ****Bottleneck****
def bottleneck_layers(inputs, filters, kernel_size = (3, 3), initializer = 'he_normal'):
  x = inputs
  for _ in range(2):
    x = Conv2D(filters = filters, kernel_size = kernel_size, padding = 'same', kernel_initializer = initializer)(x)
    x = Activation('relu')(x)
  
  return x


def bottleneck_block(inputs, filters):
  x = inputs
  output = bottleneck_layers(x, filters = filters)
  return output


