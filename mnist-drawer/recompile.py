"""Run this script to re-make the model.

Requires the MNIST dataset in the data/ folder."""

import os
import numpy as np

from ._model import kerasmodel, LeNet
from ._load_data import load_data_tensorflow, loader_mnist_pytorch


def retrain_pyt():

    import torch
    import torch.optim as optim
    import torch.nn as nn
    from torch.autograd import Variable
    import torch.nn.functional as F

    ROOT = "./data"

    if not os.path.exists(ROOT):
        os.mkdir(ROOT)

    # check whether cuda is available
    use_cuda = torch.cuda.is_available()

    BATCH_SIZE = 100
    TOTAL_EPOCHS = 20
    train_loader, test_loader = loader_mnist_pytorch(BATCH_SIZE)

    model = LeNet()
    if use_cuda:
        model = model.cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=.9)

    try:
        from tqdm import tqdm
        itr = tqdm(range(TOTAL_EPOCHS))
    except ImportError:
        itr = range(TOTAL_EPOCHS)

    for epoch in itr:
        # training
        ave_loss = 0
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            
            x, target = Variable(x), Variable(target)
            out = model(x)
            loss = F.nll_loss(out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        # testing


def retrain_tf():

    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.preprocessing import OneHotEncoder

    ohe = OneHotEncoder(sparse=False)

    DIGIT_DIMENSIONS = 28
    N_CLASSES = 10
    BATCH_SIZE = 256

    (data,labels), (test_data,test_labels) = load_data_tensorflow()

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
    checkpoint = tf.keras.callbacks.ModelCheckpoint("model_weights.h5", monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")

    # BINARIZE DATA 
    bin_data = np.where(data>0, 1, 0).reshape(data.shape + (1,))
    bin_test_data = np.where(test_data>0, 1, 0).reshape(test_data.shape + (1,))
    # make the labels 2D (keras likes this...)
    y_ = ohe.fit_transform(labels.reshape(-1,1))
    y_test = ohe.fit_transform(test_labels.reshape(-1,1))

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
