

import importlib
import os

import pandas as pd
import numpy as np
import config


# Paths
importlib.reload(config)
input_dir = config.input_dir
output_dir = config.output_dir
input_file = config.input_file




def load(day = [1,2,3,4], response_type = ['freeresp', 'sentences'], dataset = 'train'):
    '''

    :param day: recordings from several days. Can use first day to predict other days.
    :param response_type:  one or all of these ['freeresp', 'sentences', 'background']
    :return: X, y, ids for each sample
    :dataset = 'train', 'test', or 'all'
    '''
    for i in response_type:
        if i.startswith('free') and i != 'freeresp':
            print('incorrect values for response_type: must be one or all of these %s and is ' % ['freeresp', 'sentences', 'background'] + str(response_type))
            os._exit()
    print('loading_data...')
    if dataset == 'train':
        df = pd.read_csv(input_dir + 'uic_dataset_04112019_train.csv')
    elif dataset == 'test':
        df = pd.read_csv(input_dir + 'uic_dataset_04112019_test.csv')
    elif dataset == 'all':
        df = pd.read_csv(input_dir + 'uic_dataset_04112019.csv')
    # time-point
    if day == [1]:
        df = df[df['day'] == 1]
    elif day == [2]:
        df = df[df['day'] == 2]
    elif day == [3]:
        df = df[df['day'] == 3]
    elif day == [4]:
        df = df[df['day'] == 4]
    elif day == [1,2]:
        df = df [(df ['day'] == 1) | (df ['day'] == 2) ]
    elif day == [1,2,3]:
        df = df[df['day'] != 4]
    # Response type
    if response_type == ['freeresp']:
        df = df[df['response_type'] == 'freeresp']
    elif response_type == ['sentences']:
        df = df[df['response_type'] == 'sentences']
    elif response_type == ['freeresp', 'sentences']:
        df = df[(df['response_type'] == 'freeresp') | (df['response_type'] == 'sentences')]
    cg = df[df['group'] == 'hc']
    pg = df[df['group'] != 'hc']
    # Create X and y
    if 'background' not in response_type:
        cgX = cg.iloc[:, 6:].values.astype(np.float)
        cgY = np.zeros(cgX.shape[0])
        cggroups = cg['id'].values
        print('\ncgX, cgY and cgGroups shapes')
        print(cgX.shape, cgY.shape, cggroups.shape)

        pgX = pg.iloc[:, 6:].values.astype(np.float)
        pgY = np.ones(pgX.shape[0])
        pggroups = pg['id'].values
        print('\npgX, pgY and pgGroups shapes')
        print(pgX.shape, pgY.shape, pggroups.shape)

        X = np.vstack((cgX, pgX))
        y = np.concatenate((cgY, pgY))
        groups = np.concatenate((cggroups, pggroups))
        print('\nconcatenated X, Y and Groups shapes')
        print(X.shape, y.shape, groups.shape)
        return X, y, groups, cg, pg
    else:
        return cg,pg