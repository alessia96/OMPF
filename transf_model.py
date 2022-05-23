# transfer learning model

import os
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator




datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)


data_dir = 'MAIN-FOLDER-PATH/'

batch_size = 10
img_height = 120
img_width = 120

# load train and validation images
train_ds = datagen.flow_from_directory(data_dir, 
					target_size=(img_height, img_width),
					class_mode="sparse",batch_size=batch_size,
					subset='training')
val_ds = datagen.flow_from_directory(data_dir, 
					target_size=(img_height, img_width),
					class_mode="sparse",batch_size=batch_size,
					subset='validation')

# get number of classes
num_classes = len(set([i for i in train_ds.classes]))

# load the resnet50 model
base_model = tf.keras.applications.resnet50.ResNet50(weights="imagenet", input_shape=(img_height, img_width, 3), include_top=False)
# set the resnet50 as not trainable
base_model.trainable = False

# define the input layer
inputs = keras.Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(500, activation='relu')(x)
x = keras.layers.Dense(300, activation='relu')(x)
outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

# define new model
model = keras.Model(inputs, outputs)

model.compile(optimizer=tf.optimizers.Adam(),
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

epochs = 50
history = model.fit(
train_ds,
validation_data=val_ds,
epochs=epochs,
callbacks= [keras.callbacks.ModelCheckpoint('/PATH/MODEL-NAME.h5',save_freq=5)])


# fine tuning

# set resnet50 layers as trainable
base_model.trainable = True

# compile with a lower learning rate
model.compile(optimizer=tf.optimizers.Adam(1e-7),
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

# train again
epochs = 50
history = model.fit(
train_ds,
validation_data=val_ds,
epochs=epochs,
callbacks= [keras.callbacks.ModelCheckpoint('/PATH/TUNED-MODEL-NAME.h5',save_freq=5)])


