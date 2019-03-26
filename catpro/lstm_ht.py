#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

    Code to train model, LSTM and two branch LSTM into feedforward.
    I tried to clean it up from the original script so that it is least confusing.
    
    Authors: 
    - Daniel M. Low, BCS, MIT & Harvard University, March 2019
    - Tuka Alhanai, CSAIL MIT April 4th 2018
    
    

"""



import os
import itertools
import time
import logging
# import sys

import numpy as np
import pandas as pd
# from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
# from sklearn import preprocessing
# import statsmodels.api as sm
# from sklearn.isotonic import IsotonicRegression as IR
# from sklearn.mixture import GaussianMixture
# import csv
# from scipy import signal
# from scipy.stats import kurtosis, skew, spearmanr
# import pickle
# from sklearn import preprocessing
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, model_from_json, Model
from keras.layers import LSTM, Dense, Flatten, Dropout, Bidirectional, concatenate, Input
from keras.utils import to_categorical
from keras import optimizers
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler, \
    EarlyStopping, ReduceLROnPlateau, Callback
import keras.backend as K
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
# from keras.layers import Lambda
# from keras.layers import Embedding

import data_helpers
import config
import data_handler
import feature_generator
import doc2vec
# from data_handler import * #TODO: put load_data functions into data_handler when your finished



np.random.seed(0) #todo, figure this out. add tensorflow, keras.


# ============================================================================================
# Train LSTM Model
# ============================================================================================
def trainLSTM(X_train=None, y_train=None, X_dev=None, y_dev=None, R_train=None, R_dev=None, hyperparams=None, path_to_dir=None):
    """
        Method to train LSTM model.

        X_{train,dev}: should be [Nexamples, Ntimesteps, Nfeatures]
        y_{train,dev}: is a vector of binary outcomes
        R_{train,dev}: is the subject ID, useful for later when calculating performance at the subject level
        hyperparams:   is a dict

    """
    # seed generator


    # grab hyperparamters
    # exp         = hyperparams['exp']
    timesteps   = hyperparams['timesteps']
    batch_size  = hyperparams['batchsize']
    epochs      = hyperparams['epochs']
    activation_function = hyperparams['activation_function']
    optimizer   = hyperparams['optimizer']
    # lr          = hyperparams['lr'] #TODO: make tunable
    hsize       = hyperparams['hsize']
    nlayers     = hyperparams['nlayers']
    loss        = hyperparams['loss']
    # momentum    = hyperparams['momentum']
    # decay       = hyperparams['decay']
    dropout     = hyperparams['dropout']
    dropout_rec = hyperparams['dropout_rec']
    layertype   = hyperparams['layertype']
    balClass    = hyperparams['balClass']
    act_output  = hyperparams['act_output']

    if layertype=='bi-lstm':

        merge_mode = hyperparams['merge_mode']

    # grab input dimension and number of timesteps (i.e. number of interview sequences)

    # X_train.shape = (8050-dev_set, 30 timesteps, 32 dimensions) #todo: I added this




    dim = X_train.shape[2]
    # timesteps = X_train.shape[1]

    # balance training classes
    if balClass:
        cweight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    else:
        cweight = np.array([1, 1])

    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()


    if layertype == 'lstm':
        if nlayers == 1:
            # model.add(Embedding(input_dim=6000, input_length = timesteps, output_dim=dim, batch_input_shape=(batch_size, timesteps, dim),
            #                       trainable=True, mask_zero=True))
            # model.add(LSTM(hsize, activation=activation_function, return_sequences=False, batch_input_shape = (batch_size, timesteps, dim), recurrent_dropout=dropout_rec, dropout=dropout))
            model.add(LSTM(hsize, activation=activation_function, return_sequences=False, input_shape=(timesteps, dim), recurrent_dropout=dropout_rec, dropout=dropout))
            # model.add(Dense(10, activation='relu'))
            # model.add(Lambda(lambda x: x, output_shape=lambda s: s, mask=None, arguments=None)) #TODO: added, mask?
            # model.add(Flatten())

            # model.add(Dense(dsize, activation=act_output))

        if nlayers == 2:
            model.add(LSTM(hsize, activation=activation_function, return_sequences=True,   input_shape=(timesteps, dim), recurrent_dropout=dropout_rec, dropout=dropout))
            model.add(LSTM(hsize, activation=activation_function, return_sequences=False, recurrent_dropout=dropout_rec))
            # model.add(Dense(dsize, activation=act_output))

        if nlayers == 3:
            model.add(LSTM(hsize, activation=activation_function, return_sequences=True,  input_shape=(timesteps, dim), recurrent_dropout=dropout_rec, dropout=dropout,))
            model.add(LSTM(hsize, activation=activation_function, return_sequences=True,  recurrent_dropout=dropout_rec))
            model.add(LSTM(hsize, activation=activation_function, return_sequences=False, recurrent_dropout=dropout_rec))
            # model.add(Dense(dsize, activation=act_output))

        if nlayers == 4:
            model.add(LSTM(hsize, activation=activation_function, return_sequences=True, input_shape=(timesteps, dim), recurrent_dropout=dropout_rec, dropout=dropout))
            model.add(LSTM(hsize, activation=activation_function, return_sequences=True, recurrent_dropout=dropout_rec,))
            model.add(LSTM(hsize, activation=activation_function, return_sequences=True, recurrent_dropout=dropout_rec))
            model.add(LSTM(hsize, activation=activation_function, return_sequences=False, recurrent_dropout=dropout_rec))
            # model.add(Dense(dsize, activation=act_output))

    elif layertype == 'bi-lstm':
        if nlayers == 1:
            model.add(Bidirectional(LSTM(hsize, activation=activation_function, return_sequences=False, recurrent_dropout=dropout_rec,
                           dropout=dropout), input_shape=(timesteps, dim), merge_mode=merge_mode))
            # model.add(Flatten())
            # model.add(Dense(dsize, activation=act_output))

        if nlayers == 2:
            model.add(Bidirectional(LSTM(hsize, activation=activation_function, return_sequences=True, recurrent_dropout=dropout_rec,
                           dropout=dropout),input_shape=(timesteps, dim), merge_mode=merge_mode))
            model.add(Bidirectional(LSTM(hsize, activation=activation_function, return_sequences=False, recurrent_dropout=dropout_rec), merge_mode=merge_mode))
            # model.add(Flatten())
            # model.add(Dense(dsize, activation=act_output))

        if nlayers == 3:
            model.add(Bidirectional(LSTM(hsize, activation=activation_function, return_sequences=True, recurrent_dropout=dropout_rec,
                           dropout=dropout),input_shape=(timesteps, dim),merge_mode=merge_mode))
            model.add(Bidirectional(LSTM(hsize, activation=activation_function, return_sequences=True, recurrent_dropout=dropout_rec),merge_mode=merge_mode))
            model.add(Bidirectional(LSTM(hsize, activation=activation_function, return_sequences=False,recurrent_dropout=dropout_rec),merge_mode=merge_mode))
            # model.add(Flatten())
            # model.add(Dense(dsize, activation=act_output))

        if nlayers == 4:
            model.add(Bidirectional(LSTM(hsize, activation=activation_function, return_sequences=True, recurrent_dropout=dropout_rec,
                           dropout=dropout),input_shape=(timesteps, dim), merge_mode=merge_mode))
            model.add(Bidirectional(LSTM(hsize, activation=activation_function, return_sequences=True,  recurrent_dropout=dropout_rec),merge_mode=merge_mode))
            model.add(Bidirectional(LSTM(hsize, activation=activation_function, return_sequences=True,  recurrent_dropout=dropout_rec),merge_mode=merge_mode))
            model.add(Bidirectional(LSTM(hsize, activation=activation_function, return_sequences=False, recurrent_dropout=dropout_rec),merge_mode=merge_mode))
            # model.add(Flatten())
            # model.add(Dense(dsize, activation=act_output))

    # add the final output node
    # this check is useful if training a multiclass model
    if act_output == 'sigmoid':
        dsize = 1
        model.add(Dense(dsize, activation=act_output))

    elif act_output == 'softmax':
        dsize = 27
        model.add(Dense(dsize, activation=act_output))
        y_train = to_categorical(R_train, num_classes=27)
        y_dev = to_categorical(R_dev, num_classes=27)

    elif act_output == 'relu':
        dsize = 1
        def myrelu(x):
            return (K.relu(x, alpha=0.0, max_value=27))
        model.add(Dense(dsize, activation=myrelu))
        y_train = R_train
        y_dev = R_dev


    # logger.info info on network
    logger.info(model.summary())
    # logger.info('--- network has layers:', nlayers, ' hsize:',hsize, ' bsize:', batch_size,
    #       ' lr:', lr, ' epochs:', epochs, ' loss:', loss, ' act_o:', act_output)

    # define optimizer
    # sgd = optimizers.SGD(lr=lr, momentum=momentum, decay=0, nesterov=True) #todo: determine in crossvalid

    # compile model
    if regression:
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=['acc','mae','mse']) #TODO which metric?
    else:
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=['acc'])  # TODO which metric?
                # metrics=[metrics.f1_score])
    # model.fit(X_train, y_train,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           validation_data=(X_dev, y_dev),
    #           class_weight=cweight)
    #
    # pred        = model.predict(X_dev,   batch_size=None, verbose=0) #TODO: erased parameter steps
    # pred_train  = model.predict(X_train, batch_size=None, verbose=0)
    #
    # return pred, pred_train

    # defining callbacks - creating directory to dump files
    # path_to_dir = path_to_dir + str(exp)

    call_back = False  # TODO fix: ValueError: Expected array-like (array or non-string sequence), got <tf.Tensor 'dense_40_target_1:0' shape=(?, ?) dtype=float32>

    if call_back:
        # serialize model to JSON
        model_json = model.to_json()
        with open(path_to_dir + "/model.json", "w") as json_file:
            json_file.write(model_json)

        # checkpoints
        # filepaths to checkpoints
        filepath_best       = path_to_dir + "/weights-best.hdf5"
        filepath_epochs     = path_to_dir + "/weights-{epoch:02d}-{loss:.2f}.hdf5"

        # log best model
        checkpoint_best     = ModelCheckpoint(filepath_best,   monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

        # log improved model
        checkpoint_epochs   = ModelCheckpoint(filepath_epochs, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

        # log results to csv file
        csv_logger          = CSVLogger(path_to_dir + '/training.log')

        # loss_history        = LossHistory()
        # lrate               = LearningRateScheduler()

        # update decay as a function of epoch and lr
        lr_decay            = lr_decay_callback(lr, decay)

        # early stopping criterion
        early_stop          = EarlyStopping(monitor='loss', min_delta=1e-04, patience=25, verbose=0, mode='auto')
        # reduce_lr         = ReduceLROnPlateau(monitor='acc', factor=0.2, patience=5, min_lr=0.0001)

        # log files to plot via tensorboard
        tensorboard         = TensorBoard(log_dir=path_to_dir + '/logs', histogram_freq=0, write_graph=True, write_images=False)

        #calculate custom performance metric
        perf                = Metrics()

        # these are the callbacks we care for
        callbacks_list      = [checkpoint_best, checkpoint_epochs, early_stop, lr_decay, tensorboard, csv_logger]

        # train model
        model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(X_dev, y_dev),
                  class_weight=cweight,
                  callbacks=callbacks_list)

        # load best model and evaluate

        model.load_weights(filepath=filepath_best)

        # gotta compile it
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=[metrics.f1_score])

    # return predictions of best model
    pred_dev = model.predict_classes(X_dev,   batch_size=None, verbose=0) #TODO: erased parameter steps
    # pred_train  = model.predict(X_train, batch_size=None, verbose=0)
    return pred_dev
    # return pred, pred_train


# defines step decay
# =================================================
def lr_decay_callback(lr_init, lr_decay):
    def step_decay(epoch):
        return lr_init * (lr_decay ** (epoch + 1))
    return LearningRateScheduler(step_decay)


# logger.infos additional metrics to log file
# =================================================
class Metrics(Callback):

    # log performance on every epoch
    def on_epoch_end(self, epoch, logs):

        # checking if more than one validation data is being used
        try:
            pred = np.asarray(self.model.predict([self.validation_data[0],self.validation_data[1]]))
            targ = self.validation_data[2]
        except:
            pred = np.asarray(self.model.predict(self.validation_data[0]))
            targ = self.validation_data[1]

        # calculate f1 score
        logs['val_f1'] = metrics.f1_score(targ, np.round(pred), pos_label=1)

        return

# ============================================================================================
# train DNN model that combines audio and doc LSTM branches.
# ============================================================================================
# def trainHierarchy(X_train_fuse, y_train, X_dev_fuse, y_dev, R_train, R_dev, hyperparams):
#
#         """
#         Method to train LSTM model.
#
#         X_{train,dev}_fuse: should be [Nexamples, Nfeatures]
#         y_{train,dev}: 		is a vector of binary outcomes
#         R_{train,dev}: 		is the subject ID, useful for later when calculating performance at the subject level
#         hyperparams:   		is a dict
#
#     """
#
#     # init random seed
#     np.random.seed(1337)
#
#     # number of features
#     dim = X_train_fuse.shape[1]
#
#     # hyperparameters
#     loss = hyperparams['loss']
#     lr = hyperparams['lr']
#     momentum = hyperparams['momentum']
#     batch_size = hyperparams['batchsize']
#     dsize = hyperparams['dsize']
#     epochs = hyperparams['epochs']
#     decay = hyperparams['decay']
#     act = hyperparams['act']
#     nlayers = hyperparams['nlayers']
#     dropout = hyperparams['dropout']
#     exppath = hyperparams['exppath']
#     act_output = hyperparams['act_output']
#
#     # define input
#     input = Input(shape=(dim,))
#
#     # define number of DNN layers
#     if nlayers == 1:
#         final = Dense(dsize, activation=act)(input)
#         final = Dropout(dropout)(final)
#
#     if nlayers == 2:
#         final = Dense(dsize, activation=act)(input)
#         final = Dropout(dropout)(final)
#         final = Dense(dsize, activation=act)(final)
#         final = Dropout(dropout)(final)
#
#
#     if nlayers == 3:
#         final = Dense(dsize, activation=act)(input)
#         final = Dropout(dropout)(final)
#         final = Dense(dsize, activation=act)(final)
#         final = Dropout(dropout)(final)
#         final = Dense(dsize, activation=act)(final)
#         final = Dropout(dropout)(final)
#
#     if nlayers == 4:
#         final = Dense(dsize, activation=act)(input)
#         final = Dropout(dropout)(final)
#         final = Dense(dsize, activation=act)(final)
#         final = Dropout(dropout)(final)
#         final = Dense(dsize, activation=act)(final)
#         final = Dropout(dropout)(final)
#
#     # add final output node
#     final = Dense(1, activation='sigmoid')(final)
#
#     # define model
#     model = Model(inputs=input, outputs=final)
#
#     # logger.info summary
#     logger.info(model.summary())
#     logger.info('--- network has layers:', nlayers, 'dsize:', dsize, 'bsize:', batch_size, 'lr:', lr, 'epochs:',
#           epochs)
#
#
#     # defining files to save
#     # path_to_dir = path_to_dir + str(exp)
#     os.system('mkdir ' + exppath)
#
#     # serialize model to JSON
#     model_json = model.to_json()
#     with open(exppath + "/model.json", "w") as json_file:
#         json_file.write(model_json)
#
#     # define optimizer
#     sgd = optimizers.SGD(lr=lr, momentum=momentum, decay=0, nesterov=True)
#
#     # compile model
#     model.compile(loss=loss,
#                   optimizer=sgd,
#                   metrics=['accuracy'])
#
#     # filepaths to checkpoints
#     filepath_best = exppath + "/weights-best.hdf5"
#     filepath_epochs = exppath + "/weights-{epoch:02d}-{loss:.2f}.hdf5"
#
#     # save best model
#     checkpoint_best = ModelCheckpoint(filepath_best, monitor='loss', verbose=0, save_best_only=True, mode='auto')
#
#     # save improved model
#     checkpoint_epochs = ModelCheckpoint(filepath_epochs, monitor='loss', verbose=0, save_best_only=True, mode='auto')
#
#     # log performance to csv file
#     csv_logger = CSVLogger(exppath + '/training.log')
#     # loss_history        = LossHistory()
#     # lrate               = LearningRateScheduler()
#
#     # update decay as function of epoch and lr
#     lr_decay = lr_decay_callback(lr, decay)
#
#     # define early stopping criterion
#     early_stop = EarlyStopping(monitor='loss', min_delta=1e-04, patience=25, verbose=0, mode='auto')
#     # reduce_lr         = ReduceLROnPlateau(monitor='acc', factor=0.2, patience=5, min_lr=0.0001)
#
#     # log data to view via tensorboard
#     tensorboard = TensorBoard(log_dir=exppath + '/logs', histogram_freq=0, write_graph=True, write_images=False)
#
#     # define metrics
#     perf = Metrics()
#
#     # callbacks we are interested in
#     callbacks_list = [checkpoint_best, checkpoint_epochs, early_stop, lr_decay, perf, tensorboard, csv_logger]
#
#     # train model
#     model.fit(X_train_fuse, y_train,
#               batch_size=batch_size,
#               epochs=epochs,
#               validation_data=(X_dev_fuse, y_dev),
#               callbacks=callbacks_list)
#     # load best model and evaluate
#     model.load_weights(filepath=filepath_best)
#     model.compile(loss=loss, optimizer=sgd,metrics=['accuracy'])
#     # return predictions of best model
#     pred_train = model.predict(X_train_fuse, batch_size=None, verbose=0, steps=None)
#     pred = model.predict(X_dev_fuse, batch_size=None, verbose=0, steps=None)
#     return pred, pred_train



# ============================================================================================
# Combinding Features
# ============================================================================================
def combineFeats():
    """
        Combining audio and doc features.
    """


    # PROCESSING AUDIO
    # ===============================
    hyperparams = {'exp': 20, 'timesteps': 30, 'stride': 1, 'lr': 9.9999999999999995e-07, 'nlayers': 3, 'hsize': 128, 'batchsize': 128, 'epochs': 300, 'momentum': 0.80000000000000004, 'decay': 0.98999999999999999, 'dropout': 0.20000000000000001, 'dropout_rec': 0.20000000000000001, 'loss': 'binary_crossentropy', 'dim': 100, 'min_count': 3, 'window': 3, 'wepochs': 25, 'layertype': 'bi-lstm', 'merge_mode': 'mul', 'path_to_dir': 'data/LSTM_10-audio/', 'exppath': 'data/LSTM_10-audio/20/', 'text': 'data/Step10/alltext.txt', 'balClass': False}
    exppath = hyperparams['exppath']

    # load model
    with open(exppath + "/model.json", "r") as json_file:
        model_json = json_file.read()
    try:
        model = model_from_json(model_json)
    except:
        model = model_from_json(model_json, custom_objects={'myrelu':myrelu})

    lr = hyperparams['lr']
    loss = hyperparams['loss']
    momentum = hyperparams['momentum']
    nlayers = hyperparams['nlayers']
    # text = 'data/Step10/alltext.txt'

    # load best model and evaluate
    filepath_best = exppath + "/weights-best.hdf5"
    model.load_weights(filepath=filepath_best)
    logger.info('--- load weights')

    # sgd = optimizers.SGD(lr=lr, momentum=momentum, decay=0, nesterov=True)

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    logger.info('--- compile model')

    # load data
    X_train, y_train, X_dev, y_dev, R_train, R_dev = loadAudio()
    logger.info('--- load data')

    # getting activations from final layer
    layer = model.layers[nlayers-1]
    inputs = [K.learning_phase()] + model.inputs
    _layer2 = K.function(inputs, [layer.output])
    acts_train = np.squeeze(_layer2([0] + [X_train]))
    acts_dev = np.squeeze(_layer2([0] + [X_dev]))
    logger.info('--- got activations')

    # PROCESSING DOCS
    # ===============================
    hyperparams = {'exp': 330, 'timesteps': 7, 'stride': 3, 'lr': 0.10000000000000001, 'nlayers': 2, 'hsize': 4, 'batchsize': 64, 'epochs': 300, 'momentum': 0.84999999999999998, 'decay': 1.0, 'dropout': 0.10000000000000001, 'dropout_rec': 0.80000000000000004, 'loss': 'binary_crossentropy', 'dim': 100, 'min_count': 3, 'window': 3, 'wepochs': 25, 'layertype': 'bi-lstm', 'merge_mode': 'concat', 'path_to_dir': 'data/LSTM_10/', 'exppath': 'data/LSTM_10/330/', 'text': 'data/Step10/alltext.txt', 'balClass': False}
    exppath = hyperparams['exppath']

    # load model
    with open(exppath + "/model.json", "r") as json_file:
        model_json = json_file.read()
    try:
        model = model_from_json(model_json)
    except:
        model = model_from_json(model_json, custom_objects={'myrelu':myrelu})

    lr = hyperparams['lr']
    loss = hyperparams['loss']
    momentum = hyperparams['momentum']
    nlayers = hyperparams['nlayers']

    # load best model and evaluate
    filepath_best = exppath + "/weights-best.hdf5"
    model.load_weights(filepath=filepath_best)
    logger.info('--- load weights')

    # sgd = optimizers.SGD(lr=lr, momentum=momentum, decay=0, nesterov=True)

    model.compile(loss=loss,
                  optimizer=sgd,
                  metrics=['accuracy'])
    logger.info('--- compile model')

    # load data
    X_train_doc, y_train, X_dev_doc, y_dev, R_train_doc, R_dev_doc = loadDoc()
    logger.info('--- load data')

    # getting activations from final layer
    layer = model.layers[nlayers - 1]
    inputs = [K.learning_phase()] + model.inputs
    _layer2 = K.function(inputs, [layer.output])
    acts_train_doc = np.squeeze(_layer2([0] + [X_train_doc]))
    acts_dev_doc   = np.squeeze(_layer2([0] + [X_dev_doc]))
    logger.info('--- got activations')

    # FUSE EMBEDDINGS
    # ============================
    acts_train_doc_pad = []
    for idx, subj in enumerate(np.unique(S_train)):
        index = np.where(S_train == subj)[0]
        j = 0
        indexpad = np.where(S_train_doc == subj)[0]
        for i,_ in enumerate(index):
            # logger.info(i)
            if i%4 == 0 and i > 0 and j < indexpad.shape[0]-1:
                j = j+1
            acts_train_doc_pad.append(acts_train_doc[indexpad[j],:])

    acts_dev_doc_pad = []
    for idx, subj in enumerate(np.unique(S_dev)):
        index = np.where(S_dev == subj)[0]
        j = 0
        indexpad = np.where(S_dev_doc == subj)[0]
        for i,_ in enumerate(index):
            # logger.info(i)
            if i%4 == 0 and i > 0 and j < indexpad.shape[0]-1:
                j = j+1
            acts_dev_doc_pad.append(acts_dev_doc[indexpad[j],:])

        # CMVN
        # scaler = preprocessing.StandardScaler().fit(np.asarray(acts_train_doc_pad))
        # acts_train_doc_pad = scaler.transform(np.asarray(acts_train_doc_pad))
        # acts_dev_doc_pad = scaler.transform(np.asarray(acts_dev_doc_pad))

        X_train_fuse = np.hstack((np.asarray(acts_train_doc_pad),acts_train))
        X_dev_fuse = np.hstack((np.asarray(acts_dev_doc_pad),acts_dev))

        # optional
        np.save('data/fuse/X_train.npy', X_train_fuse)
        np.save('data/fuse/features/X_dev.npy', X_dev_fuse)
        np.save('data/fuse/features/y_train.npy', y_train)
        np.save('data/fuse/features/y_dev.npy', y_dev)
        np.save('data/fuse/features/S_train.npy', S_train)
        np.save('data/fuse/features/S_dev.npy', S_dev)
        np.save('data/fuse/features/R_train.npy', R_train)
        np.save('data/fuse/features/R_dev.npy', R_dev)


# ============================================================================================
# Data Loading
# ============================================================================================
# you will need to point to your data directory
def loadAudio():

    X_train, y_train = np.load('./X_train_audio.npz')['a'], np.load('./X_train_audio.npz')['b']
    # X_train, y_train = np.load('data/audio/X_train.npy'), np.load('data/audio/y_train.npy')
    # X_dev, y_dev = np.load('data/audio/X_dev.npy'), np.load('data/audio/y_dev.npy')
    # R_train, R_dev = np.load('data/audio/R_dev.npy'), np.load('data/audio/R_dev.npy')
    # return X_train, y_train, X_dev, y_dev, R_train, R_dev
    return X_train, y_train


def loadDoc():

    X_train, y_train = np.load('data/doc/X_train.npy'), np.load('data/doc/y_train.npy')
    X_dev, y_dev = np.load('data/doc/X_dev.npy'), np.load('data/doc/y_dev.npy')
    R_train, R_dev = np.load('data/doc/R_dev.npy'), np.load('data/doc/R_dev.npy')

    return X_train, y_train, X_dev, y_dev, R_train, R_dev


def loadFuse():

    X_train, y_train = np.load('data/fuse/X_train.npy'), np.load('data/fuse/y_train.npy')
    X_dev, y_dev = np.load('data/fuse/X_dev.npy'), np.load('data/fuse/y_dev.npy')
    R_train, R_dev = np.load('data/fuse/R_dev.npy'), np.load('data/fuse/R_dev.npy')
    return X_train, y_train, X_dev, y_dev, R_train, R_dev



# ============================================================================================
# main script
# ============================================================================================

def load_train_data_lstm(dataset='train', group_by='response', run_audio=config.config['run_audio'], run_text=config.config['run_text'], use_doc2vec=False):
    '''
    
    :param type: 
    :param timesteps: 
    :param group_by: {'response', 'interview'} each sample, where each time step is segment or response, respectively
    :return: 
    '''
    # test = self.kwargs.get('test')
    # inputPath = self.kwargs.get('input')
    # audio_file = self.kwargs.get('audio_file')
    logger.info('loading data...')
    if group_by=='interview':
        inputPath = config.config['input'] #TODO: change to variable , inestead of dictionary
        audio_file = 'text_audio_df.csv'
        text_audio = pd.read_csv(inputPath + audio_file)
        #
        # One row had NaN value, detected:
        # for row_i in range(text_audio.shape[0]):
        #     row = text_audio.iloc[row_i,:]
        #     if row.isna().sum()>1:
        #         break
        # So we replace with previous question or answer
        # text_audio.iloc[12927, :]=text_audio.iloc[12927-2, :] #TODO: this isn't happening anymore? Weird.

        # Each sample is one participant. Take all responses and pad to max. of 100.

        # Train
        # TODO: set in
        text_audio_train = text_audio[text_audio.FILE_TYPE == dataset.upper()]  # TODO load text from here too?
        text_audio_train_response = text_audio_train[text_audio_train.PARTICIPANT == 'Participant']  # without Ellie's questions
        # y_train_audio =np.array(text_audio_train_response.LABEL)
        # X_train_audio = np.array(text_audio_train_response.iloc[:,8:])
        # audio_features = np.array(text_audio_train_response.columns[8:])

        # Group by participant
        X_train_text = []
        X_train_audio = []
        y_train = []
        for participant in list(set(text_audio_train_response.FILE_NAME)):
            text_audio_train_response_participant = text_audio_train_response[text_audio_train_response.FILE_NAME==participant] #all the responses of 1 participant
            X_train_text_participant = np.array(text_audio_train_response_participant.UTTERANCE)
            X_train_audio_participant = np.array(text_audio_train_response_participant.iloc[:, 8:])
            y_train_participant = set(text_audio_train_response_participant.LABEL)
            if 0 in y_train_participant: #turn list into int w/o using set
                y_train_participant=0
            else:
                y_train_participant=1
            # Append participants data to datasets
            X_train_text.append(X_train_text_participant)
            X_train_audio.append(X_train_audio_participant )
            y_train.append(y_train_participant)
        X_train_audio = np.array(X_train_audio )
        X_train_text = np.array(X_train_text)
        np.save(X_train_text, output_dir + 'X_train_text_groupedby_interview.npy')
        np.save(X_train_audio , output_dir + 'X_train_audio_groupedby_interview.npy')
        # TODO: fix
        # for participant_matrix in X_train_audio_padded:
        #     break
        #     participant_vector = np.array(pd.DataFrame(participant_matrix ).mean())
        # X_train_audio_padded= pad_sequences(X_train_audio, maxlen=timesteps, padding=padding, dtype='float32')
        # X_train_text_padded = pad_sequences(X_train_text, maxlen=timesteps, padding=padding, dtype='str')
        if type == 'train':
            return X_train_text, X_train_audio, y_train,
        # elif type == 'test' and test: #TODO
        # if test:
        #   text_audio_test =      # TODO
        # responses_per_participant_disorder= []
        # responses_per_participant_control = []
        # for participant in list(set(list(text_audio_train_response.id))):
        #     text_audio_train_response_participant = text_audio_train_response[text_audio_train_response['id']==participant]
        #     amount_of_responses = text_audio_train_response_participant .shape[0]
        #     amount_of_words_per_response = []
        #     for response in list(text_audio_train_response_participant.UTTERANCE):
        #         amount_of_words_per_response.append(len(response.split()))
        #     amount_of_words_per_interview = np.sum(amount_of_words_per_response)
        #     if 0 in set(text_audio_train_response_participant.LABEL):
        #         responses_per_participant_control.append([amount_of_responses, amount_of_words_per_interview])
        #     elif 1 in set(text_audio_train_response_participant.LABEL):
        #         responses_per_participant_disorder.append([amount_of_responses, amount_of_words_per_interview])
        # # Get statistics
        # depression_responses = np.sum([n[0] for n in responses_per_participant_disorder])
        # depression_words = np.sum([n[1] for n in responses_per_participant_disorder])
        # control_responses = np.sum([n[0] for n in responses_per_participant_control])
        # control_words = np.sum([n[1] for n in responses_per_participant_control])
        #
        # # See if you can classify according to fluency
        # y_train_disorder = [1]*len(responses_per_participant_disorder)
        # y_train_control = [0]*len(responses_per_participant_control)
        #
        # y_train_all = np.concatenate((y_train_disorder, y_train_control))
        # X_train_all = np.concatenate((responses_per_participant_disorder, responses_per_participant_control))
        #
        #
        # import matplotlib.pyplot as plt
        #
        # # evenly sampled time at 200ms intervals
        # x = [n[0] for n in responses_per_participant_disorder]
        # y = [n[1] for n in responses_per_participant_disorder]
        # x1 = [n[0] for n in responses_per_participant_control]
        # y1 = [n[1] for n in responses_per_participant_control]
        #
        # # red dashes, blue squares and green triangles
        # plt.clf()
        # # y=np.zeros(len(x))
        # # y1 = np.zeros(len(x1))
        # plt.plot(x,y, 'ro', x1, y1, 'bo')
        # plt.savefig(config.config['output_dir']+'2D_responses_per_participant.png', dpi=200)
        #
        # X_train, X_dev, y_train, y_dev, = train_test_split(X_train_all, y_train_all,
        #                                                                             test_size=0.20, random_state=0,
        #                                                                             shuffle=True)  # TODO: save and add test_size to parameters.
        #
        # # y_dev = [int(n) for n in y_dev]
        # # TODO: shuffle, I think text has some of the test set.
        # clf = LinearSVC(C=0.1, random_state=0)  # only cv
        # clf = GaussianMixture(n_components=2)
        #
        # clf.fit(X_train, y_train)
        #
        # y_pred = clf.predict(X_dev)
        # y_pred = [int(n) for n in y_pred]
        # # For interpretation purposes, we compute the decision confidence (i.e., normalized distance from boundary)
        # # TODO add interpretation.py functions here.
        # f1 = f1_score(y_dev, y_pred)
        # logger.info(f1)
        # acc = accuracy_score(y_dev, y_pred)
        # logger.info(acc)
        # roc_auc = roc_auc_score(y_dev, y_pred)
        # precision = precision_score(y_dev, y_pred)
        # recall = recall_score(y_dev, y_pred)
        #




        #     return X_test_audio, y_test_audio, audio_features
    elif group_by=='response':
        inputPath = config.config['input']  # TODO: make variable in config
        audio_file = 'text_audio_df_nonconcat.csv' #TODO, buld one using nonconcat.
        text_audio = pd.read_csv(inputPath + audio_file)
        text_audio = text_audio[text_audio.FILE_TYPE==dataset.upper()]
        # if run_text:
        #     # Normalize text and reduce to k dimensions
        #     # text_audio_participants = text_audio[text_audio.PARTICIPANT=='Participant']
        #     text = text_audio_participants.UTTERANCE #TODO. CHange name to responses

        #TODO: I could seperate loading audio and text, but then i have to load text_audio_df_nonconcat.csv twice

        # Normalize audio and reduce to k dimensions
        audio_features = text_audio.iloc[:, 8:].values
        # It's hard to normalize with such long sequences, so convert to replace inf for 0
        bad_indices = np.where(np.isnan(audio_features))
        audio_features[bad_indices] = 0
        # audio_features_wo_inf = []
        # for row in audio_features:
        #     bad_indices = np.where(np.isinf(row))
        #     row[bad_indices] = 0
        #     audio_features_wo_inf.append(row)
        #     logger.info(bad_indices)
        # audio_features_wo_inf = np.array(audio_features_wo_inf)
        # audio_features.dtype  # > dtype('float16')

        audio_normalized = data_helpers.normalize(array_train=audio_features)#TODO: do for text as well
        k = 32
        audio_features_names = text_audio.columns[8:]
        audio_normalized_best, kbest_features_names_audio = data_helpers.f_feature_selection(X=audio_normalized, y=np.array(text_audio.LABEL),k=k, audio_features=audio_features_names , print_= True) #TODO fix print_, I'd like to log this.
        audio_normalized_best = pd.DataFrame(audio_normalized_best , columns=kbest_features_names_audio)
        text_audio_wo_audio = text_audio.iloc[:, :8].reset_index() #reset index in order to concat
        text_audio_normalized_best = pd.concat((text_audio_wo_audio ,audio_normalized_best),axis=1) #TODO: add text
        # Each sample is one response. Take all segments and pad to max. of 100.
        # Group by response: each sample is a response, each time step is a segment.
        X_train_text = []
        X_train_audio = []
        y_train = []
        segments = 0
        for row in range(text_audio_normalized_best.shape[0]):
        # loop through rows of segments
            speaker = text_audio_normalized_best.PARTICIPANT.iloc[row]
            if speaker == 'Ellie':
                if segments == 0:
                    continue
                else:
                    X_train_text.append(one_response_text)
                    X_train_audio.append(one_response_audio)
                    y_train.append(int(text_audio_normalized_best.LABEL.iloc[row]))
                    segments = 0  # each time Ellie speaks a new response begins
            elif speaker == 'Participant' and segments==0:
                    one_response_text = []
                    one_response_audio = []
                    one_response_text.append(text_audio_normalized_best.UTTERANCE.iloc[row])
                    one_response_audio.append(np.array(text_audio_normalized_best.iloc[row, 8:]))

                    segments += 1
            elif speaker == 'Participant' and segments > 0:
                    one_response_text.append(text_audio_normalized_best.UTTERANCE.iloc[row])
                    one_response_audio.append(np.array(text_audio_normalized_best.iloc[row, 8:]))
                    segments += 1
        y_train = np.array(y_train)
        X_train_audio = np.array(X_train_audio)
        X_train_text = np.array(X_train_text)
        # TODO: this is temporary, replace with full features or average embeddings, elmo, doc2vec
        if use_doc2vec:
            # Following Alhaini (2018), train doc2vec on both questions and responses
            config_params = config.config
            if config_params['create_features']:
                logger.info('creating doc2vec vectors...')
                # questions
                text_audio_ellie = text_audio[text_audio.PARTICIPANT=='Ellie'].UTTERANCE.values #TODO: maybe do not include Ellie, lot of repeated sentences
                texts_ellie = []
                for i in text_audio_ellie:
                    try: texts_ellie.append(i.split())
                    except: logger.info('skipped segment during doc2vec training')
                texts_participants = []
                for j in X_train_text:
                    text_one_participant = []
                    for i in j:
                        try:
                            text_one_participant.append(i.split())
                        except:
                            logger.info(i)
                            logger.info('skipped segment during doc2vec training')
                    texts_participants.append(text_one_participant)
                # doc2vec neads one lists of lists, not list of list of list like text_participants is
                text_participants_lists = []
                for i in texts_participants:
                    #     concatenate
                    flattened_list = [item for sublist in i for item in sublist]
                    text_participants_lists.append(flattened_list)
                texts = texts_ellie + text_participants_lists

                doc2vec_model = doc2vec.doc2model(documents=texts, output_dir=config_params['output_dir'])
                X_train_text_doc2vec = []
                for i in texts_participants:
                    X_train_text_doc2vec_one_participant = doc2vec.model2vec(documents=i, input_dir = config_params['output_dir'], model=doc2vec_model)
                    X_train_text_doc2vec.append(X_train_text_doc2vec_one_participant)
                # np.save('./data/datasets/X_train_text_doc2vec3.npy', X_train_text_doc2vec)
                np.save('./data/datasets/X_train_text_doc2vec_latest.npy', X_train_text_doc2vec)
            else:
                X_train_text_doc2vec = np.load(config_params['input'] + 'X_train_text_doc2vec.npy')


            X_train_text = X_train_text_doc2vec[:]
            # X_train_text_padded = pad_sequences(X_train_text_doc2vec, maxlen=timesteps, padding=padding,
            #                                     dtype='str')
        else:
            config_params = config.config
            featureNames = config_params.get('features').split(',')
            featGenr = feature_generator.FeatureGenerator(featureNames, config_params)
            dataHandler = data_handler.DataHandler(config_params)
            vocabs = dataHandler.loadAllVocabs(inputPath)
            vectors = dataHandler.loadEmbedding(vocabs,vector_length=100)  # TODO: save loaded version, cause it takes a while
            allFeatureList = featGenr.initFeatures()
            unknown_vec = np.random.normal(0, 0.17, 100)
            X_train_text_embeddings = []
            for response in X_train_text:
                response_embeddings = []
                for segment in response:
                    try:
                        embeds = list(featGenr.generateEmbeddingFeatures(segment, vectors, unknown_vec,
                                                                100).values())  # TODO: set 100 in config
                        response_embeddings.append(embeds)
                    except: logger.info('skipped segment')
                response_embeddings= response_embeddings[:timesteps]

                X_train_text_embeddings.append(response_embeddings)
            X_train_text = X_train_text_embeddings[:]
            # X_train_text_padded = pad_sequences(X_train_text_embeddings, maxlen=timesteps, padding=padding,
            #                                     dtype='str')
        #     max=10 segments per response, mean =2.7
        # TODO: make sure y_train.shape[0] or X_train_audio.shape[0], i.e., the amount of samples matches text_audio_df_concat.csv=='Participant' length once the latter is re-preprocessed with 3 missing people.
        # TODO: could save below and try with different timesteps
        # np.savez_compressed('./data/input/X_train_nonconcatenated.npz', a=X_train_text, b=X_train_audio,
        #                     c=y_train)
        # timesteps should be near max, but not too far from mean or median of segments per response.
        # X_train_audio_padded = pad_sequences(X_train_audio, maxlen=timesteps, padding=padding, dtype='float32')


        # if test:
        #   text_audio_test =      # TODO
        if dataset == 'train':
            return X_train_text, X_train_audio, y_train
        # elif type == 'test' and test: #TODO
        #     return X_test_audio, y_test_audio, audio_features









if __name__ == "__main__":

    config_params = config.config
    path_to_dir = data_helpers.make_output_dir(os.path.join(config.config['output_dir'], 'neural_networks/'))

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(os.path.join(path_to_dir, 'training.log'))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    # Let the user know if they are not using a GPU
    if K.tensorflow_backend._get_available_gpus() == []:
        logger.info('YOU ARE NOT USING A GPU ON THIS DEVICE!')
    
    # 1. load the data for audio
    # X_train, y_train, X_dev, y_dev, R_train, R_dev = loadAudio() #TODO
    # X_train, y_train = loadAudio()  # TODO: add other sets
    # X_train_text,X_train_audio, y_train =  load_train_data_lstm(type='train')
    group_by = config_params['group_by']
    regression = config_params['regression']
    if group_by == 'interview':
        X_train_text, X_train_audio, y_train_all = load_train_data_lstm(dataset='train',
                                                                        group_by='interview')  # TODO, use above

    elif group_by=='response':
        X_train_text, X_train_audio, y_train_all = load_train_data_lstm(dataset='train',
                                                                        group_by='response', use_doc2vec=True)  # TODO, use put parameters in config
    test_size = config.config['train_test_split']
    config_params = config.config
    run_text = config_params['run_text']
    run_audio = config_params['run_audio']
    # TODO: cross-validation
    # if regression: #TODO
        # Your bins need to be appropriate for your output values
        # e.g. 0 to 50 with 25 bins
        # bins = np.linspace(0, 50, 25)
        # y_binned = np.digitize(y_full, bins)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y_binned)
    # 2. train lstm model
    if run_audio:
        sgd = optimizers.SGD(lr=0.000001, momentum=0.8, decay=0.99, nesterov=True)
        lr = 0.000001
        momentum = 0.8
        decay = 0.99
        timesteps = [3,20,33] #33 is max segments per response
        hsize=[64, 128, 256]
        dropout_rec = [0.2]
        nlayers = [1,3]
    elif run_text:
        sgd = optimizers.SGD(lr=0.1, momentum=0.8, decay=0, nesterov=True)
        lr = 0.1
        momentum = 0.8
        decay = 0
        timesteps = [3,7,33]
        hsize= [4, 32,128]
        dropout_rec = [0.2]
        nlayers = [1,2]
    epochs = [3] #TODO add to stop early

    padding = 'pre' #was post before, but lstm is biased towards ending. #TODO: compare
    # if regression: #TODO:
    #     loss =

    # hyperparams_final = {'exp': 1, 'timesteps': timesteps, 'stride': 3, 'activation_function': activation_function, 'lr': lr, 'nlayers': nlayers, 'hsize': hsize,
    #                'batchsize': 64, 'epochs': epochs, 'momentum': 0.8, 'decay': 0.99, 'dropout': 0.2,
    #                'dropout_rec': 0.2, 'loss': 'binary_crossentropy', 'dim': 100, 'min_count': 3, 'window': 3,
    #                'wepochs': 25, 'layertype': layer_type, 'merge_mode': 'mul', 'exppath': 'data/runs/',
    #                'text': 'data/Step10/alltext.txt', 'balClass': balClass, 'act_output': act_output, 'optimizer': optimizer}  # TODO: Tuka had 'balClass': False

    hyperparams_lstm = {'timesteps': timesteps, 'activation_function': ['relu', 'tanh'], 'act_output':['sigmoid'],  'nlayers': nlayers, 'hsize': hsize,
    'batchsize': [64, 128], 'epochs': epochs, 'dropout': [0.2, 0.4, 0.6],'loss': ['binary_crossentropy'], 'layertype': ['lstm'],
    'balClass': [True, False],'optimizer': ['Adam', sgd], 'dropout_rec':dropout_rec}
    hyperparams_bilstm = {'timesteps': timesteps, 'activation_function': ['relu', 'tanh'], 'act_output': ['sigmoid'],
                        'nlayers': nlayers, 'hsize': hsize,
                        'batchsize': [32, 64, 128], 'epochs': epochs, 'dropout': [0.2, 0.4, 0.6],
                        'loss': ['binary_crossentropy'], 'layertype': ['bi-lstm'],
                        'balClass': [True, False], 'optimizer': ['Adam', sgd], 'dropout_rec':dropout_rec, 'merge_mode': ['mul', 'concat']}
    hyperparams_final = {'timesteps': [33], 'activation_function': ['relu'], 'act_output': ['sigmoid'],'nlayers': [2], 'hsize': [128],
                        'batchsize': [128], 'epochs': [20], 'dropout': [0.4],
                        'loss': ['binary_crossentropy'], 'layertype': ['lstm'],
                        'balClass': [False], 'optimizer': [sgd], 'dropout_rec': [0.2]}
    # ==== Run gridsearch
    model = 'lstm' #TODO: add in loop. if they're not too long
    if model == 'lstm':
        hyperparams = hyperparams_lstm
    elif model == 'bi-lstm':
        hyperparams = hyperparams_bilstm
    if config_params['test']== True:
        hyperparams = hyperparams_final #TODO: add final option to config. these arent the final ones anyway.

    hyperparams_list = list(hyperparams.items())
    hyperparams_keys = [n[0] for n in hyperparams_list] # This is so i make sure keys and values or in the same order
    hyperparams_values = [n[1] for n in hyperparams_list]  # This is so i make sure keys and values or in the same order
    gridsearch = list(itertools.product(*hyperparams_values ))

    logger.info(len(gridsearch), 'runs...')
    scores_all = []
    gridsearch_hyperparams_ordered = []
    logger.info('The following %d parameter sets are running...' % len(gridsearch))
    logger.info(str(hyperparams))
    with open(path_to_dir+'/gridsearch.txt','w') as f:
        f.write(str(gridsearch))
    for i, gridsearch_i in enumerate(gridsearch):
        start = time.time()
        hyperparams_i = {}
        for j, value in enumerate(gridsearch_i):
            hyperparams_i[hyperparams_keys[j]] = value
        gridsearch_hyperparams_ordered.append(list(hyperparams_i.items()))
        # Pad and split data
        # =====
        timesteps_i = hyperparams_i['timesteps']
        X_train_audio_i = pad_sequences(X_train_audio, maxlen=timesteps_i, padding=padding, dtype='float32') # Not specifying dtype results in wrong data
        X_train_text_i = pad_sequences(X_train_text, maxlen=timesteps_i, padding=padding, dtype='float32')
        # except: logger.info('padding float did not work')
        # try:
        #     X_train_audio_padded = pad_sequences(X_train, maxlen=timesteps, padding=padding, dtype='float32')
        # except: logger.info('padding float did not work')
        if run_audio and run_text:

            X_train_all = np.concatenate((X_train_text_i, X_train_audio_i), axis=2)

            X_train, X_dev, y_train, y_dev = train_test_split(X_train_all, y_train_all, test_size=test_size,
                                                              random_state=0,
                                                              shuffle=True, stratify=y_train_all)  # TODO: save fixed.
        elif not run_audio and run_text:
            X_train, X_dev, y_train, y_dev = train_test_split(X_train_text_i, y_train_all, test_size=test_size,
                                                              random_state=0,
                                                              shuffle=True, stratify=y_train_all)  # TODO: save fixed.
        elif run_audio and not run_text:
            X_train, X_dev, y_train, y_dev = train_test_split(X_train_audio_i, y_train_all, test_size=test_size,
                                                              random_state=0, shuffle=True,
                                                              stratify=y_train_all)  # TODO: save fixed.
        # Check stratification
        logger.info((np.sum(y_train) / len(y_train)).round(2) == (np.sum(y_dev) / len(y_dev)).round(
            2))  # TODO: ask if this is reasonable given depression is not 30%, but 10%
        # Run =====
        y_pred = trainLSTM(X_train=X_train, y_train=y_train, X_dev=X_dev, y_dev=y_dev,
                           hyperparams=hyperparams_i, path_to_dir=path_to_dir)  # TODO, add sets

        # # 5. evaluate performance ====
        # cweight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
        # mean_confidence = np.nanmean([np.abs(n - 0.5) for n in pred_dev]).round(3)  # how far from 0.5 is it. 0.5 is max.
        # y_pred2 = np.array([int(n) for n in np.round(pred_dev)])
        f1 = metrics.f1_score(y_dev, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_dev, y_pred)
        f1_weighted = metrics.f1_score(y_dev, y_pred, average='weighted')
        acc = metrics.accuracy_score(y_dev, y_pred)
        precision = metrics.precision_score(y_dev, y_pred)
        recall = metrics.recall_score(y_dev, y_pred)
        end = time.time()
        # logger.info('confidence: ', np.round(mean_confidence, 3))
        # str(np.sum(y_pred)), 'out of', str(y_dev.shape[0]))
        #
        time_i = np.round(end - start, 2)
        # perform train-dev split
        # logger.info('=======executing train-dev split model')
        # logger.info('round(f1, 2), hyperparams_i, '\n')
        scores_names = ['F1', 'Prec.', 'Rec.', 'ROC AUC', 'F1 weighted', 'Acc.', 'Time', 'Depressed', 'Parameter id']
        scores = [round(f1, 2),
                  round(precision, 2),
                  round(recall, 2),
                  round(roc_auc, 3),
                  round(f1_weighted, 2),
                  round(acc, 2),
                  time_i,
                  str(np.sum(y_pred))+'/'+str(len(y_pred)),
                  list(hyperparams_i.items())]  # TODO test

        scores_all.append(scores)
        results_all = pd.DataFrame(scores_all, columns=scores_names)
        results_all.to_csv(os.path.join(path_to_dir+'/scores_temp.csv'))
    # Print latex file for all results

    scores_all_sorted = sorted(scores_all, key=lambda x: x[0])  # TODO change name. Sorted from small to large.
    results_all = pd.DataFrame(scores_all_sorted, columns=scores_names)
    logger.info(str(results_all.to_latex()))
    results_all.to_csv(os.path.join(path_to_dir + '/scores_final_sorted.csv'))
    # Print latex file for best result
    results_best = results_all.iloc[-1, :]
    best_parameters = results_all.iloc[-1, -1]
    perform_cross_validation = config_params['perform_cross_validation']
    if perform_cross_validation:
        index = 'LSTM CV'
    else:
        index = 'LSTM train-dev split'
    results_best = pd.DataFrame(results_best).T
    results_best.index = [index]
    logger.info('\n\n')
    logger.info('best results: ')
    logger.info('\n')
    logger.info(str(results_best.to_latex()))
    logger.info(str(best_parameters))
