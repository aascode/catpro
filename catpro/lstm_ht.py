#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

    Code to train model, LSTM and two branch LSTM into feedforward.
    I tried to clean it up from the original script so that it is least confusing.
    
    Author: Tuka Alhanai, CSAIL MIT April 4th 2018

"""

import os
# import sys
import pprint
import numpy as np
import pandas as pd
# from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.utils import class_weight
from sklearn import preprocessing
# import statsmodels.api as sm
# from sklearn.isotonic import IsotonicRegression as IR
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
from sklearn.model_selection import train_test_split
# from keras.layers import Lambda
# from keras.layers import Embedding
import data_helpers


# from data_handler import * #TODO: put load_data functions into data_handler when your finished

# Let the user know if they are not using a GPU
if K.tensorflow_backend._get_available_gpus() == []:
    print('YOU ARE NOT USING A GPU ON THIS DEVICE! EXITING!')
    # sys.exit()

# ============================================================================================
# Train LSTM Model
# ============================================================================================

# def lstm_gridsearch(X_train=None, y_train=None, X_dev=None, y_dev=None, R_train=None, R_dev=None, hyperparams_fixed={}, hyperparams_gridsearch = {}):
#     """
#         Method to train LSTM model.
#
#         X_{train,dev}: should be [Nexamples, Ntimesteps, Nfeatures]
#         y_{train,dev}: is a vector of binary outcomes
#         R_{train,dev}: is the subject ID, useful for later when calculating performance at the subject level
#         hyperparams:   is a dict, where each key has the values to be tested. {'layers': [1,2,3,4], 'hidden_neurons': [16,32,64], }
#
#     """
#     # seed generator
#     np.random.seed(0)
#     parameter_sets = []
#
#     hyperparams_fixed = {}
#
#
#     # TODO: do faster, more intelligent type of hyperparameter tuning instead of gridsearch
#     # recurrent dropout rates of {0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.8}, merge mode of {sum, mul, con- cat, ave}, and batch size of {32, 64, 128, ..., 4096}}
#     for hyperparameter in list(hyperparams_gridsearch.values()):
#
#     for kernel_ in kernels:
#         for C in Cs:
#             parameter_sets.append([kernel_, C])
#     logger.info(str(len(parameter_set)+' parameter sets will be run'))
#     scores_mean_all = []
#     for parameter_set in parameter_sets:
#         logger.info(str(parameter_set))
#         kernel = parameter_set[0]
#         C = parameter_set[1]
#         if kernel == 'linear':
#             clf = LinearSVC(C=C, class_weight=class_weight, random_state=0)
#         else:
#             clf = SVC(C=C, kernel=kernel, class_weight=class_weight, probability=probability, random_state=0)
#         print('training model...\n')
#
#         if perform_cross_validation:
#             logger.info('=======executing cross validation model')
#             scores = cross_validate(clf, X_train, y_train, scoring = scoring_classification, cv = cv, return_train_score = False)
#             # scores_mean = [] #TODO: do automatically depending on metrics chosen
#             # for metric in scoring_classification:
#             scores_mean = [round(scores['test_f1'].mean(),2),
#                            round(scores['test_f1'].std(),2),
#                            round(scores['test_roc_auc'].mean(),2),
#                            round(scores['test_precision'].mean(),2),
#                            round(scores['test_recall'].mean(),2),
#                            parameter_set[0],parameter_set[1] ]
#             scores_mean_all.append(scores_mean)
#             print(str(round(scores['test_f1'].mean(),4))+' '+str(parameter_set))
#             print(str(round(scores['test_f1'].std(),4)) + ' ' +'\n\n')
#            # #     X_test_csr = X_test.tocsr()
#            #      predictions_all = lr1.predict(X_test)
#            #      acc_score_all = accuracy_score(Y_test, predictions_all)
#            #      f1_score_all = f1_score(Y_test, predictions_all,average='macro')
#            #      if f1_score_all > best_f1:
#            #          best_f1 = f1_score_all
#            #          best_c = C
#            #          best_norm = norm
#            #          best_kernel = kernel
#            #      print ('cost and norm ' + str(C) + ' ' + str(norm))
#            #  logger.info("Classification report for classifier %s:\n%s\n" % (scores, metrics.classification_report( Y_test, predictions_all,digits=3)))
#         # print ('best params (macro-avg): ',best_f1, best_c, best_norm, best_kernel)
#         else:
#             # perform train-dev split
#             logger.info('=======executing train-dev split model')
#             X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.20, random_state=0,
#                                                               shuffle=True)  # TODO: save and add test_size to parameters.
#             clf.fit(X_train, y_train)
#             y_pred = clf.predict(X_dev)
#             f1 = f1_score(y_dev, y_pred)
#             roc_auc = roc_auc_score(y_dev, y_pred)
#             precision = precision_score(y_dev, y_pred)
#             recall = recall_score(y_dev, y_pred)
#             print('\n', f1, parameter_set, '\n')
#             scores = [round(f1,2),
#                       '-',
#                       round(roc_auc,2),
#                       round(precision,2),
#                       round(recall,2),
#                       parameter_set[0],
#                       parameter_set[1] ] # the '-' is for f1 std, not available with
#             scores_mean_all.append(scores) #TODO fix name (not "mean" for train-dev split)
#     # Print latex file for all results
#     scores_mean_all_sorted = sorted(scores_mean_all, key=lambda x: x[0])  # TODO change name. Sorted from small to large.
#     results_all = pd.DataFrame(scores_mean_all_sorted)
#
#     columns = scoring_classification + ['parameter1', 'parameter2'] #TODO: do dynamically depending on amount of hyparameters
#     columns.insert(1, 'f1 std')
#     results_all.columns = columns
#     logger.info(str(results_all.to_latex()))
#     # Print latex file for best result
#     results_best = results_all.iloc[-1,:]
#     if perform_cross_validation:
#         index = 'SVM 5-fold CV'
#     else:
#         index = 'SVM train-dev split'
#     results_best = pd.DataFrame(results_best).T
#     results_best.index = [index]
#     logger.info(str(results_best.to_latex()))
#     # best_params = scores_mean_all_sorted[-1]
#     # logger.info('best params (means of f1, roc_auc, recall, recall): ' + str(best_params))
#     return results_best
#




# ============================================================================================
# Train LSTM Model
# ============================================================================================
def trainLSTM(X_train=None, y_train=None, X_dev=None, y_dev=None, R_train=None, R_dev=None, hyperparams=None):
    """
        Method to train LSTM model.

        X_{train,dev}: should be [Nexamples, Ntimesteps, Nfeatures]
        y_{train,dev}: is a vector of binary outcomes
        R_{train,dev}: is the subject ID, useful for later when calculating performance at the subject level
        hyperparams:   is a dict

    """
    # seed generator
    np.random.seed(0)

    # grab hyperparamters
    exp         = hyperparams['exp']
    batch_size  = hyperparams['batchsize']
    epochs      = hyperparams['epochs']
    activation_function = hyperparams['activation_function']
    optimizer   = hyperparams['optimizer']
    lr          = hyperparams['lr']
    hsize       = hyperparams['hsize']
    nlayers     = hyperparams['nlayers']
    loss        = hyperparams['loss']
    dirpath     = hyperparams['dirpath']
    momentum    = hyperparams['momentum']
    decay       = hyperparams['decay']
    dropout     = hyperparams['dropout']
    dropout_rec = hyperparams['dropout_rec']
    merge_mode  = hyperparams['merge_mode']
    layertype   = hyperparams['layertype']
    balClass    = hyperparams['balClass']
    act_output  = hyperparams['act_output']


    # grab input dimension and number of timesteps (i.e. number of interview sequences)

    # X_train.shape = (8050-dev_set, 30 timesteps, 32 dimensions) #todo: I added this

    dim = X_train.shape[2]
    timesteps = X_train.shape[1]

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


    # print info on network
    print(model.summary())
    print('--- network has layers:', nlayers, ' hsize:',hsize, ' bsize:', batch_size,
          ' lr:', lr, ' epochs:', epochs, ' loss:', loss, ' act_o:', act_output)

    # define optimizer
    sgd = optimizers.SGD(lr=lr, momentum=momentum, decay=0, nesterov=True) #todo: determine in crossvalid

    # compile model
    model.compile(loss=loss,
                  optimizer=sgd,
                  metrics=['accuracy','mae','mse'])
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
    dirpath = dirpath + str(exp)
    os.system('mkdir ' + dirpath)

    # serialize model to JSON
    model_json = model.to_json()
    with open(dirpath + "/model.json", "w") as json_file:
        json_file.write(model_json)

    # checkpoints
    # filepaths to checkpoints
    filepath_best       = dirpath + "/weights-best.hdf5"
    filepath_epochs     = dirpath + "/weights-{epoch:02d}-{loss:.2f}.hdf5"

    # log best model
    checkpoint_best     = ModelCheckpoint(filepath_best,   monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    # log improved model
    checkpoint_epochs   = ModelCheckpoint(filepath_epochs, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    # log results to csv file
    csv_logger          = CSVLogger(dirpath + '/training.log')

    # loss_history        = LossHistory()
    # lrate               = LearningRateScheduler()

    # update decay as a function of epoch and lr
    lr_decay            = lr_decay_callback(lr, decay)

    # early stopping criterion
    early_stop          = EarlyStopping(monitor='loss', min_delta=1e-04, patience=25, verbose=0, mode='auto')
    # reduce_lr         = ReduceLROnPlateau(monitor='acc', factor=0.2, patience=5, min_lr=0.0001)

    # log files to plot via tensorboard
    tensorboard         = TensorBoard(log_dir=dirpath + '/logs', histogram_freq=0, write_graph=True, write_images=False)

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
    call_back = False #TODO fix
    if call_back:
        model.load_weights(filepath=filepath_best)

        # gotta compile it
        model.compile(loss=loss,
                      optimizer=sgd,
                      metrics=[metrics.f1_score])

    # return predictions of best model
    pred        = model.predict(X_dev,   batch_size=None, verbose=0) #TODO: erased parameter steps
    pred_train  = model.predict(X_train, batch_size=None, verbose=0)

    return pred, pred_train


# defines step decay
# =================================================
def lr_decay_callback(lr_init, lr_decay):
    def step_decay(epoch):
        return lr_init * (lr_decay ** (epoch + 1))
    return LearningRateScheduler(step_decay)


# prints additional metrics to log file
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
#     # print summary
#     print(model.summary())
#     print('--- network has layers:', nlayers, 'dsize:', dsize, 'bsize:', batch_size, 'lr:', lr, 'epochs:',
#           epochs)
#
#
#     # defining files to save
#     # dirpath = dirpath + str(exp)
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
    hyperparams = {'exp': 20, 'timesteps': 30, 'stride': 1, 'lr': 9.9999999999999995e-07, 'nlayers': 3, 'hsize': 128, 'batchsize': 128, 'epochs': 300, 'momentum': 0.80000000000000004, 'decay': 0.98999999999999999, 'dropout': 0.20000000000000001, 'dropout_rec': 0.20000000000000001, 'loss': 'binary_crossentropy', 'dim': 100, 'min_count': 3, 'window': 3, 'wepochs': 25, 'layertype': 'bi-lstm', 'merge_mode': 'mul', 'dirpath': 'data/LSTM_10-audio/', 'exppath': 'data/LSTM_10-audio/20/', 'text': 'data/Step10/alltext.txt', 'balClass': False}
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
    print('--- load weights')

    sgd = optimizers.SGD(lr=lr, momentum=momentum, decay=0, nesterov=True)

    model.compile(loss=loss,
                  optimizer=sgd,
                  metrics=['accuracy'])
    print('--- compile model')

    # load data
    X_train, y_train, X_dev, y_dev, R_train, R_dev = loadAudio()
    print('--- load data')

    # getting activations from final layer
    layer = model.layers[nlayers-1]
    inputs = [K.learning_phase()] + model.inputs
    _layer2 = K.function(inputs, [layer.output])
    acts_train = np.squeeze(_layer2([0] + [X_train]))
    acts_dev = np.squeeze(_layer2([0] + [X_dev]))
    print('--- got activations')

    # PROCESSING DOCS
    # ===============================
    hyperparams = {'exp': 330, 'timesteps': 7, 'stride': 3, 'lr': 0.10000000000000001, 'nlayers': 2, 'hsize': 4, 'batchsize': 64, 'epochs': 300, 'momentum': 0.84999999999999998, 'decay': 1.0, 'dropout': 0.10000000000000001, 'dropout_rec': 0.80000000000000004, 'loss': 'binary_crossentropy', 'dim': 100, 'min_count': 3, 'window': 3, 'wepochs': 25, 'layertype': 'bi-lstm', 'merge_mode': 'concat', 'dirpath': 'data/LSTM_10/', 'exppath': 'data/LSTM_10/330/', 'text': 'data/Step10/alltext.txt', 'balClass': False}
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
    print('--- load weights')

    sgd = optimizers.SGD(lr=lr, momentum=momentum, decay=0, nesterov=True)

    model.compile(loss=loss,
                  optimizer=sgd,
                  metrics=['accuracy'])
    print('--- compile model')

    # load data
    X_train_doc, y_train, X_dev_doc, y_dev, R_train_doc, R_dev_doc = loadDoc()
    print('--- load data')

    # getting activations from final layer
    layer = model.layers[nlayers - 1]
    inputs = [K.learning_phase()] + model.inputs
    _layer2 = K.function(inputs, [layer.output])
    acts_train_doc = np.squeeze(_layer2([0] + [X_train_doc]))
    acts_dev_doc   = np.squeeze(_layer2([0] + [X_dev_doc]))
    print('--- got activations')

    # FUSE EMBEDDINGS
    # ============================
    acts_train_doc_pad = []
    for idx, subj in enumerate(np.unique(S_train)):
        index = np.where(S_train == subj)[0]
        j = 0
        indexpad = np.where(S_train_doc == subj)[0]
        for i,_ in enumerate(index):
            # print(i)
            if i%4 == 0 and i > 0 and j < indexpad.shape[0]-1:
                j = j+1
            acts_train_doc_pad.append(acts_train_doc[indexpad[j],:])

    acts_dev_doc_pad = []
    for idx, subj in enumerate(np.unique(S_dev)):
        index = np.where(S_dev == subj)[0]
        j = 0
        indexpad = np.where(S_dev_doc == subj)[0]
        for i,_ in enumerate(index):
            # print(i)
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

def load_train_data_lstm(dataset='train', timesteps=33, group_by='response'):
    '''
    
    :param type: 
    :param timesteps: 
    :param group_by: {'response', 'interview'} each sample, where each time step is segment or response, respectively
    :return: 
    '''
    # test = self.kwargs.get('test')
    # inputPath = self.kwargs.get('input')
    # audio_file = self.kwargs.get('audio_file')

    if group_by=='interview':
        inputPath = './data/input/'  # TODO: load from config
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
        # TODO: fix
        X_train_audio_padded= pad_sequences(X_train_audio, maxlen=timesteps, padding='post', dtype='float32')
        X_train_text_padded = pad_sequences(X_train_text, maxlen=timesteps, padding='post', dtype='str')
        # if test:
        #   text_audio_test =      # TODO
        if type == 'train':
            return X_train_text_padded, X_train_audio_padded, y_train
        # elif type == 'test' and test: #TODO
        #     return X_test_audio, y_test_audio, audio_features
    elif group_by=='response':
        inputPath = './data/input/'  # TODO: load from config
        audio_file = 'text_audio_df_nonconcat.csv' #TODO, buld one using nonconcat.
        text_audio = pd.read_csv(inputPath + audio_file)
        text_audio = text_audio[text_audio.FILE_TYPE==dataset.upper()]

        # Obtain features for text. Normalize and reduce to k dimensions

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
        #     print(bad_indices)
        # audio_features_wo_inf = np.array(audio_features_wo_inf)
        # audio_features.dtype  # > dtype('float16')

        audio_normalized = data_helpers.normalize(array_train=audio_features)#TODO: do for text as well
        k = 32
        audio_features_names = text_audio.columns[8:]
        audio_normalized_best, kbest_features_names = data_helpers.f_feature_selection(X=audio_normalized, y=np.array(text_audio.LABEL),k=k, audio_features=audio_features_names , print_= True)
        audio_normalized_best = pd.DataFrame(audio_normalized_best , columns=kbest_features_names)
        text_audio_wo_audio = text_audio.iloc[:, :8].reset_index() #reset index in order to concat
        text_audio_normalized_best = pd.concat((text_audio_wo_audio ,audio_normalized_best),axis=1) #TODO: add text
        # Each sample is one response. Take all segments and pad to max. of 100.
        # Group by response: each sample is a response, each time step is a segment.
        X_train_text = []
        X_train_audio = []
        y_train = []
        segments = 0
        for row in range(text_audio_normalized_best.shape[0]):
        # for row in range(1000:):
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
                    one_response_text.append(np.array(text_audio_normalized_best.UTTERANCE.iloc[row]))
                    one_response_audio.append(np.array(text_audio_normalized_best.iloc[row, 8:]))

                    segments += 1
            elif speaker == 'Participant' and segments > 0:
                    one_response_text.append(np.array(text_audio_normalized_best.UTTERANCE.iloc[row]))
                    one_response_audio.append(np.array(text_audio_normalized_best.iloc[row, 8:]))
                    segments += 1
        X_train_audio = np.array(X_train_audio)

        X_train_text = np.array(X_train_text)
        y_train = np.array(y_train)
        # TODO: make sure y_train.shape[0] or X_train_audio.shape[0], i.e., the amount of samples matches text_audio_df_concat.csv=='Participant' length once the latter is re-preprocessed with 3 missing people.
        # TODO: could save below and try with different timesteps
        # np.savez_compressed('./data/input/X_train_nonconcatenated.npz', a=X_train_text, b=X_train_audio,
        #                     c=y_train)
        # timesteps should be near max, but not too far from mean or median of segments per response.
        X_train_audio_padded = pad_sequences(X_train_audio, maxlen=timesteps, padding='post', dtype='float32')
        X_train_text_padded = pad_sequences(X_train_text, maxlen=timesteps, padding='post', dtype='str')

        # if test:
        #   text_audio_test =      # TODO
        if dataset == 'train':
            return X_train_text_padded, X_train_audio_padded, y_train
        # elif type == 'test' and test: #TODO
        #     return X_test_audio, y_test_audio, audio_features






if __name__ == "__main__":
    # 1. load the data for audio
    # X_train, y_train, X_dev, y_dev, R_train, R_dev = loadAudio() #TODO
    # X_train, y_train = loadAudio()  # TODO: add other sets
    # X_train_text,X_train_audio, y_train =  load_train_data_lstm(type='train')
    X_train_text,X_train, y_train =  load_train_data_lstm(dataset='train', timesteps=5) #TODO, use above
    test_size = 0.2
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train , test_size = test_size , random_state = 0, shuffle=True) #TODO: save fixed.



    # 2. train lstm model

    # hyperparams_fixed = {}
    # hyperparams_gridsearch = {
    #     'layers': [1, 2, 3],
    #     'hidden_nodes': [4, 8, 16, 32, 64, 128, 256],
    #     'activation_function': ['relu', 'elu', 'tanh', 'hard_sigmoid'],
    #     'dropout_rate': [0, 0.1, 0.2, 0.4, 0.6, 0.8],
    #     'recurrent_dropout_rate': [0, 0.1, 0.2, 0.4, 0.6, 0.8],
    #     'merge_mode': ['sum', 'mul', 'concat', 'ave'],
    #     'batch': [32, 64, 128, 256, 512, 1024, 2056, 4096],
    #     'loss_binary': ['binary crossentropy'],
    #     'loss_regression': ['categorical crossentropy', 'mean_square_error', 'mean_absolute_error']
    # }
    # # pred_dev_audio, pred_train_audio = lstm_gridsearch(X_train=X_train, y_train=y_train, X_dev=X_dev, y_dev=y_dev,
    # #                                              hyperparams_fixed=hyperparams_fixed,
    # #                                              hyperparams_gridsearch=hyperparams_gridsearch)  # TODO, add sets

    epochs = 3
    timesteps=3
    nlayers= 1 #was 3
    hsize=32 #was 128
    balClass= True
    layer_type = 'lstm' #was 'bi-lstm'
    act_output = 'sigmoid' # was 'sigmoid'
    activation_function = 'relu'
    lr = 0.000001
    momentum = 0.8
    sgd = optimizers.SGD(lr=lr, momentum=momentum, decay=0, nesterov=True)
    optimizer = [sgd]


    hyperparams_final = {'exp': 1, 'timesteps': timesteps, 'stride': 1, 'activation_function': activation_function, 'lr': lr, 'nlayers': nlayers, 'hsize': hsize,
                   'batchsize': 64, 'epochs': epochs, 'momentum': 0.8, 'decay': 0.99, 'dropout': 0.2,
                   'dropout_rec': 0.2, 'loss': 'binary_crossentropy', 'dim': 100, 'min_count': 3, 'window': 3,
                   'wepochs': 25, 'layertype': layer_type, 'merge_mode': 'mul',
                   'dirpath': 'data/runs/', 'exppath': 'data/runs/',
                   'text': 'data/Step10/alltext.txt', 'balClass': balClass, 'act_output': act_output, 'optimizer': optimizer}  # TODO: Tuka had 'balClass': False

    pred_dev_audio, pred_train_audio = trainLSTM(X_train=X_train, y_train=y_train, X_dev=X_dev, y_dev=y_dev, hyperparams=hyperparams_final) #TODO, add sets
    # TODO: fix. X_dev[1097] is return pred_dev_audio=nan
    # # 1b. load the doc data
    # X_train, y_train, X_dev, y_dev, R_train, R_dev = loadDoc()
    #
    # # 2b. train lstm model for doc data
    # pred_audio, pred_train_audio = trainLSTM(X_train, y_train, X_dev, y_dev, R_train, R_dev, hyperparams)
    #
    # # 3. concatenate last layer features for each audio and doc branch.
    # combineFeats()
    # X_train, y_train, X_dev, y_dev, R_train, R_dev = loadFuse()
    #
    # # 4. train feedforward.
    # # hyperparams can be different (e.g. learning rate, decay, momentum, etc.)
    # pred, pred_train = trainHierarchy(X_train_fuse, y_train, X_dev_fuse, y_dev, hyperparams)
    #
    # # 5. evaluate performance
    # cweight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    mean_confidence = np.nanmean([np.abs(n-0.5) for n in pred_dev_audio]).round(3) #how far from 0.5 is it. 0.5 is max.
    print('mean confidence: ', np.round(mean_confidence, 3))
    pred_dev_audio_int = [int(n) for n in np.round(pred_dev_audio)]
    print('how many were predicted as depression? ', str(np.sum(pred_dev_audio_int)), 'out of', str(y_dev.shape[0]))
    f1 = metrics.f1_score(y_dev, pred_dev_audio_int, pos_label=1)
    print('f1: ',np.round(f1, 3))
    f1_weighted = metrics.f1_score(y_dev, pred_dev_audio, average='weighted')
    print('f1_weighted: ', np.round(f1_weighted, 3))
    acc = metrics.accuracy_score(y_dev, pred_dev_audio)
    print('acc: ',np.round(acc, 3))





