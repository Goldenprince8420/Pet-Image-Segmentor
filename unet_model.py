# @title ****UNet Model****
def Unet(INPUT_SHAPE = [128, 128, 3]):
  inputs = Input(shape = INPUT_SHAPE)
  x = inputs
  concat, encoder_output = encoder(x)
  bottleneck_output = bottleneck_block(encoder_output, filters = 1024)
  decoder_output = decoder(bottleneck_output, concat)

  unet_model = Model(inputs = inputs, outputs = decoder_output)

  return unet_model
