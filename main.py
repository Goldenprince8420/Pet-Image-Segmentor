# @title Importing primary Libraries
import os
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime



# @title ****Mounting Drive****
from google.colab import drive
drive.mount('/content/drive', force_remount = True)



# @title Importing tensorflow latest version and Keras
try:
  %tensorflow_version 2.x
except:
  pass

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
print(tf.__version__)
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import Input



# @title Importing Specific functions and layers from keras API

from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import BatchNormalization, Dropout, concatenate, Reshape
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler, ModelCheckpoint, CSVLogger, ReduceLROnPlateau









# @title GPU Configuration
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"

strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))






## Dataset Import and Processing
dataset, info = load_dataset()
train, test = get_train_test(dataset)
train_df, test_df = preprocess_dataset(train, test)



## Data Visualization
show_image_from_dataset(train, id = 4)
show_image_from_dataset(test, id = 3)



## Model Creation
model = Unet()
model.summary()


## Current Working Directory
os.chdir("/content/drive/MyDrive/Educational Workspace/DL_Workspace/Projects/pet_segmentor/")
os.getcwd()


## Callback Utilities
!rm -rf logs

logdir = os.path.join("/content/drive/MyDrive/Educational Workspace/DL_Workspace/TFS/Projects/pet_segmentor/logs", datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = TensorBoard(log_dir=logdir)

checkpoint = ModelCheckpoint('saved_model', save_best_only = True, verbose = 1)

csvlogger = CSVLogger('train.csv')

def reducelr(epoch):
  learning_rate = 0.001
  drop = 0.975
  epochs_drop = 1
  learning_rate = learning_rate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
  return learning_rate

learning_decay = LearningRateScheduler(reducelr, verbose = 1)

reducelr_plateau = ReduceLROnPlateau(factor = 0.1, patience = 4, verbose = 1)




## Model Compiling
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
loss = 'sparse_categorical_crossentropy'
metrics = ['accuracy']

model.compile(optimizer = optimizer, loss = loss, metrics = metrics)



## Model Run and Hyperparameters
BATCH_SIZE = 64
TRAIN_LENGTH = info.splits['train'].num_examples
EPOCHS = 32
VAL_SUBSPLITS = 5
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_df, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_df,
                          callbacks = [checkpoint, tensorboard_callback, csvlogger])



## Model Results
integer_slider = 2485
show_results(integer_slider)



## Saving Model
# @title ****Getting the Current Datetime****
from datetime import datetime

now = datetime.now()
 
print("now =", now)

dt_string = now.strftime("%d%m%Y_%H%M%S")
print("date and time =", dt_string)	

# @title ****Saving The Model****
title = "/content/drive/MyDrive/Educational Workspace/DL_Workspace/TFS/Projects/pet_segmentor/models/" + dt_string + ".h5"
print(title)
model.save(title)