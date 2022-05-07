def load_dataset():
  !python -m tensorflow_datasets.scripts.download_and_prepare --register_checksums --datasets=oxford_iiit_pet:3.1.0

  dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

  print(dataset.keys())
  print("______________________INFO______________________________")
  print(info)

  return dataset, info



def random_flip(input_image, input_mask):
  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  return input_image, input_mask


def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask


@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128), method='nearest')
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128), method='nearest')
  input_image, input_mask = random_flip(input_image, input_mask)
  input_image, input_mask = normalize(input_image, input_mask)
  
  return input_image, input_mask


def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128), method='nearest')
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128), method='nearest')
  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


  # @title Preprocessing function 
def get_train_test(dataset):
  train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  test = dataset['test'].map(load_image_test) 
  return train, test

def preprocess_dataset(train, test):
  BATCH_SIZE = 64
  BUFFER_SIZE = 1000

  train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

  train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  test_dataset = test.batch(BATCH_SIZE)
  return train_dataset, test_dataset

