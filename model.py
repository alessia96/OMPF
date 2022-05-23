# model from scratch

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Dense
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import load_model


data_dir = 'MAIN-FOLDER-PATH/'

batch_size = 10
img_height = 120
img_width = 120


#extracts images for the training set
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#extracts images for the validation set
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
 
 
# get class labels and number of classes
class_names = train_ds.class_names
num_classes = len(class_names)


# define data augmentation parameters
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2)
  ]
)


model = tf.keras.models.Sequential([
     data_augmentation,
     # normalize input
     keras.layers.Normalization(),
     
     # 1st block
     tf.keras.layers.Conv2D(32,(3,3), padding="valid", activation = "relu" , input_shape = (img_width, img_height)) ,
     tf.keras.layers.Conv2D(32,(3,3), padding="same",  activation="relu", kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4)) ,
     tf.keras.layers.BatchNormalization(),
     
     # 2nd block
     tf.keras.layers.Conv2D(64,(3,3), padding="valid", strides=2, activation = "relu", kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4)) ,   
     tf.keras.layers.BatchNormalization(),  
     tf.keras.layers.Conv2D(64,(3,3), padding="same", activation="relu", kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4)) ,  
     tf.keras.layers.BatchNormalization(),
     
     # 3rd block
     tf.keras.layers.Conv2D(128,(3,3), padding="valid", strides=2, activation = "relu", kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4)) ,   
     tf.keras.layers.BatchNormalization(),  
     tf.keras.layers.Conv2D(128,(3,3), padding="same", activation="relu", kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4)) ,  
     tf.keras.layers.BatchNormalization(),     
     tf.keras.layers.Conv2D(128,(3,3), padding="same", activation="relu", kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4)) ,  
     tf.keras.layers.BatchNormalization(),     
     
     tf.keras.layers.GlobalAveragePooling2D(),
     
     tf.keras.layers.Flatten(),

     tf.keras.layers.Dense(num_classes ,activation = "softmax") ])


model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


epochs = 100
history = model.fit(
train_ds,
validation_data=val_ds,
epochs=epochs,
callbacks= [keras.callbacks.ModelCheckpoint('/PATH/MODEL-NAME.h5' ,save_freq=10)])




# plot accuracy and validation loss history for train and val

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
#
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()









