"""Run this script to re-make the model.

Requires the MNIST dataset in the data/ folder."""

import os
import json
import tensorflow as tf
import gzip
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras import Sequential, Input, datasets
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, CenterCrop, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.preprocessing.image import ImageDataGenerator


ohe = OneHotEncoder(sparse=False)

# the training data is present locally...
if os.path.isfile("data/train-images-idx3-ubyte.gz"):
    # load directly
    # loading in training data
    with gzip.open("data/train-images-idx3-ubyte.gz","r") as f:
        image_size = 28

        f.read(16)
        buf = f.read(image_size * image_size * 60000)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(60000, image_size, image_size)
        
    with gzip.open('data/train-labels-idx1-ubyte.gz','r') as f:
        f.read(8)
        labels = np.frombuffer(f.read(60000), dtype=np.uint8)
    
    # loading in test data
    with gzip.open("data/t10k-images-idx3-ubyte.gz","r") as f:
        image_size = 28

        f.read(16)
        buf = f.read(image_size * image_size * 10000)
        test_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        test_data = test_data.reshape(10000, image_size, image_size)
        
    with gzip.open('data/t10k-labels-idx1-ubyte.gz','r') as f:
        f.read(8)
        test_labels = np.frombuffer(f.read(10000), dtype=np.uint8)
else:
    # load from the keras source
    (data,labels), (test_data,test_labels) = datasets.mnist.load_data(path="mnist.npz")


def kerasmodel(dim, classes):
    # data augmentations.
    data_aug_layer = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.2),
        RandomZoom(0.2)
    ], name="data_augmentation")

    model = Sequential([
        Input((*dim, 1)),
        Conv2D(16, kernel_size=(3, 3), activation="relu", name="conv1"),
        MaxPooling2D((2, 2), name="pool1"),
        Conv2D(32, kernel_size=(3, 3), activation="relu", name="conv2"),
        MaxPooling2D((2, 2), name="pool2"),
        Flatten(),
        Dense(64, activation="relu", name="dense1"),
        Dropout(0.2, name="drop1"),
        Dense(32, activation="relu", name="dense2"),
        Dense(classes, activation="softmax")
    ])
    return model

DIGIT_DIMENSIONS = 28
N_CLASSES = 10
#model = kerasmodel((DIGIT_DIMENSIONS, DIGIT_DIMENSIONS), N_CLASSES)
BATCH_SIZE = 256
SHUFFLE_BUFFER_SIZE = 100
TRAIN_STEPS_PER_EPOCH = 200
VAL_STEPS_PER_EPOCH = 50

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
checkpoint = tf.keras.callbacks.ModelCheckpoint("model_weights.h5", monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")

# BINARIZE DATA 
bin_data = np.where(data>0, 1, 0).reshape(data.shape + (1,))
bin_test_data = np.where(test_data>0, 1, 0).reshape(test_data.shape + (1,))
# make the labels 2D (keras likes this...)
y_ = ohe.fit_transform(labels.reshape(-1,1))
y_test = ohe.fit_transform(test_labels.reshape(-1,1))
# make a tensor slice dataset... looks cool at least
#train_dataset = tf.data.Dataset.from_tensor_slices((bin_data, y_))
#train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

datagen = ImageDataGenerator(horizontal_flip=True, height_shift_range=0.1, rotation_range=30, zoom_range=0.1)
datagen_test = ImageDataGenerator()

datagen.fit(bin_data)
it_train = datagen.flow(bin_data, y_, batch_size=BATCH_SIZE)
datagen_test.fit(bin_test_data)
it_test = datagen_test.flow(bin_test_data, y_test, batch_size=BATCH_SIZE)

#define a model
model = kerasmodel((DIGIT_DIMENSIONS,DIGIT_DIMENSIONS), N_CLASSES)
# compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# train
history = model.fit_generator(it_train, epochs=50, steps_per_epoch=60000/BATCH_SIZE,
                    validation_data=it_test, validation_steps=10000/BATCH_SIZE,
                    callbacks=[earlystop, checkpoint], verbose=1) 

# save the model
model.save("models/bin_classifier")
# as jSON also
model_json = model.to_json()
with open("models/bin_classifier_js.json","w") as json_file:
    json_file.write(model_json)
