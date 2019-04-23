#!/usr/bin/env python
# coding: utf-8

# authors:
# Daniel M. Low (Harvard, MIT)
# Satra Ghosh (MIT)



# get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import importlib
import random
from scipy.stats import percentileofscore
import itertools
import logging
import os

import pandas as pd
import numpy as np
from numpy import std, mean, sqrt
# import seaborn as sns
# sns.set_context('poster')
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, ShuffleSplit, permutation_test_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (explained_variance_score, auc, f1_score, classification_report,
                             confusion_matrix, roc_auc_score, make_scorer, matthews_corrcoef, precision_score, recall_score)
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVR
from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from scipy.stats.mstats import mode

from data.datasets import load_uic
import data_helpers
import config
import plot_outputs

np.random.seed(0)

def gss(day=[1],  grid_search = None, response_type=['freeresp','sentences'], dataset='all', grid_search_parameters = None, plot=False, model = 'extra-trees', N=100, scoring = 'f1', cv = 3, remove_constant_columns = False, segmented=True, make_verbose = False):
    X, y, groups, cg,pg = load_uic.load(day=day, response_type=response_type, dataset=dataset, segmented=segmented)


    # if remove_constant_columns:
    #     # Remove columns with constant or close to constant values because they will become constant in one of the folds:
    #     # X_std = (np.std(X, axis=0) == 0) # where the column has a std == 0 means it's constant
    #     # total = np.sum(X_std)
    #     if segmented:
    #         remove = 6
    #     else:
    #         remove = 4
    #     # normalizer = MinMaxScaler(feature_range=(0,1))  # feature_range=(-1, 1).
    #     normalizer = StandardScaler()  # feature_range=(-1, 1).
    #     X = normalizer.fit_transform(X)
    #     X = pd.DataFrame(X)
    #     X2 = pd.DataFrame(X)
    #     j = 0
    #     for i in range(X2.shape[1]):
    #         if np.sum(X2.iloc[:, i]) < remove:
    #             X = X.drop([i], axis=1)
    #             logger.info(j)
    #             j+=1
    #
    #     X = X.drop([2801], axis=1)
    #     X = np.array(X)
    logger.info('features: '+ str(X.shape[1]))
    # TODO: return most important audio features, (see baseline_ht.py)
    # audio_features = pd.read_csv(input_dir+input_file).columns[6:]

    # Split into N stratified sets and then take the median
    if model == 'extra-trees':
        clf = Pipeline([('std', StandardScaler()),
                        ('feature_selection',
                         SelectFromModel(ExtraTreesClassifier(n_estimators=N,
                                                              class_weight='balanced'))),
                        ('et', ExtraTreesClassifier(n_estimators=N,
                                                    class_weight='balanced'))])
    elif model == 'svm':
        pipeline = Pipeline([
            ('normalization', None),
            ('feature_selection', None),
            ('classifier', SVC(class_weight='balanced', probability=False, random_state=0, gamma='scale'))
        ])
        # MinMaxScaler, StandardScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer
        parameters = {
            'normalization': [MinMaxScaler((0, 1)), RobustScaler()],
            'feature_selection': [SelectKBest()],
            'feature_selection__k': (32, 64, 128, 256, 512),
            'classifier__C': (0.01, 0.1, 1, 10, 100),
            'classifier__kernel': ['linear', 'rbf'],
        }

    gss = GroupShuffleSplit(n_splits=N, test_size=0.20, random_state=0)
    scores = []
    permutation_scores = []
    best_parameters_all = []
    best_training_scores = []
    i = 0
    for train_index, test_index in gss.split(X, y, groups=groups): # I believe test sizes are different sizes due to stratifying participants
        # 0.2 is 17.6, and test sets go from around 14 to 21.
        logger.info(str(i)+'-------------------------------------------------------------------------')
        i+=1
        # todo: add more SVC parameters
        if model == 'extra-trees':
            clf.fit(X[train_index], y[train_index])
            outputs = clf.predict(X[test_index])  # i
            best_score = f1_score(y[test_index], outputs)
            best_training_scores.append(best_score)
        elif model == 'svm':
            if grid_search:
                clf = GridSearchCV(pipeline, parameters, cv=cv,
                                       n_jobs=-1, verbose=0, scoring=scoring, iid=False)
                clf.fit(X[train_index], y[train_index])
                best_parameters_all.append(clf.best_estimator_.get_params())
                best_score = clf.best_score_
                best_training_scores.append(best_score)
                outputs = clf.predict(X[test_index])  # it is independent training for each set != partial_fit, but the latter would overfit without a true test set.
            else:
                print('using', grid_search_parameters)
                clf= Pipeline([
                    ('normalization',MinMaxScaler((0, 1)) ),
                    ('feature_selection', SelectKBest(k=64)),
                    ('classifier', SVC(C=grid_search_parameters [1], kernel= grid_search_parameters [0], class_weight='balanced', probability=False, random_state=0, gamma='scale'))
                ])
                clf.fit(X[train_index], y[train_index])
                # best_parameters_all.append(clf.best_estimator_.get_params())
                outputs = clf.predict(X[
                                          test_index])  # it is independent training for each set != partial_fit, but the latter would overfit without a true test set.
                best_score = f1_score(y[test_index], outputs)
                best_training_scores.append(best_score)

            # clf = RandomizedSearchCV(pipeline, parameters, cv=cv,
            #                            n_jobs=-1, verbose=0, scoring=scoring, iid=False)

            # save best parameters and their score on training set

        try:
            scores.append([roc_auc_score(y[test_index], outputs),
                       f1_score(y[test_index], outputs),
                       precision_score(y[test_index], outputs),
                       recall_score(y[test_index], outputs)])
        except Exception:
            continue
        # Permutation test

        y_shuffled = y.copy()
        random.shuffle(y_shuffled)
        clf.fit(X[train_index], y_shuffled[train_index])
        permutation_outputs = clf.predict(X[test_index])  # it is independent training for each set != partial_fit, but the latter would overfit without a true test set.
        try:
            permutation_scores.append([roc_auc_score(y[test_index], permutation_outputs),
                                   f1_score(y[test_index], permutation_outputs),
                                   precision_score(y[test_index], permutation_outputs),
                                   recall_score(y[test_index], permutation_outputs)])
        except Exception:
            continue

        if make_verbose:
            if best_score:
                logger.info('Best score training set: %0.2f' % best_score)
            logger.info('median best scores: %0.2f' % np.median(best_training_scores))
            logger.info('median roc_auc: %0.2f' % np.median([n[0] for n in scores]))
            logger.info('median f1: %0.2f' % np.median([n[1] for n in scores]))
            if best_training_scores:
                logger.info('mean best scores: %0.2f' % np.mean(best_training_scores))
            logger.info('mean roc_auc: %0.2f' % np.mean([n[0] for n in scores]))
            logger.info('mean f1: %0.2f' % np.mean([n[1] for n in scores]))

    # Summary results
    median_score_roc_auc= np.median([n[0] for n in scores])
    median_score_f1 = np.median([n[1] for n in scores])
    median_score_precision = np.median([n[2] for n in scores])
    median_score_recall = np.median([n[3] for n in scores])
    median_scores = [median_score_roc_auc, median_score_f1, median_score_precision, median_score_recall]

    permutation_median_score_roc_auc = np.median([n[0] for n in permutation_scores])
    permutation_median_score_f1 = np.median([n[1] for n in permutation_scores])
    permutation_median_score_precision = np.median([n[2] for n in permutation_scores])
    permutation_median_score_recall = np.median([n[3] for n in permutation_scores])
    median_permutation_scores = [permutation_median_score_roc_auc, permutation_median_score_f1, permutation_median_score_precision, permutation_median_score_recall]

    pvalue_roc = (100-(percentileofscore([n[0] for n in permutation_scores], median_score_roc_auc, kind='rank')))/100
    pvalue_f1  = (100 - (percentileofscore([n[1] for n in permutation_scores], median_score_f1, kind='rank'))) / 100
    pvalue_precision = (100 - (percentileofscore([n[0] for n in permutation_scores], median_score_precision, kind='rank'))) / 100
    pvalue_recall = (100 - (percentileofscore([n[1] for n in permutation_scores], median_score_recall, kind='rank'))) / 100
    pvalues = [pvalue_roc,pvalue_f1,pvalue_precision,pvalue_recall]
    return best_training_scores, scores, median_scores, permutation_scores, median_permutation_scores, pvalues

def plot_distributions(best_training_scores, scores, median_scores, permutation_scores, median_permutation_scores, pvalues, toy = False, metric_name = 'roc_auc'):
    if metric_name == 'roc_auc':
        metric = 0
    elif metric_name == 'f1':
        metric = 1
    elif metric_name == 'precision':
        metric = 2
    elif metric_name == 'recall':
        metric = 3

    # example
    if toy:
        scores= [n / 100 for n in np.random.normal(79, 15, 100)]
        median_score= np.median(scores)
        permutation_scores= [n/100 for n in np.random.normal(50, 15, 100)]
        pvalue = (100 - (percentileofscore(permutation_scores, median_score, kind='rank'))) / 100
    else:
        median_score = np.round(median_scores[metric], 2)
        pvalue = np.round(pvalues[metric], 2)
        scores = [n[metric] for n in scores]
        permutation_scores = [n[metric] for n in permutation_scores]
    bins = 30
    alpha = 0.3
    plt.clf()
    # plt.figure(figsize=(8, 6))
    plt.hist(scores, bins = bins, color='blue', alpha=alpha, label = 'GSS scores')
    plt.axvline(median_score, color='green', linewidth=3, linestyle='dashed')
    plt.title('Median F1 score: {0:.2f}'.format(median_score))
    plt.hist(permutation_scores, bins = bins, label='Permutation scores', color='gray', alpha=alpha)
    ylim = plt.ylim(0,13)
    plt.xlim(0,1)
    plt.plot(2 * [median_score], ylim, '--g', linewidth=3,label='Classification Score: (p-value = %s)' % pvalue)
    plt.plot(2 * [1. / 2], ylim, '--k', linewidth=3, label='Luck')
    plt.legend()
    plt.xlabel('Score')
    plt.xticks(np.arange(0,1,0.1))
    plt.show()
    return












def predict_across_samples(output_dir=None,day=[1],  grid_search = None, response_type=['freeresp','sentences'], dataset='all', grid_search_parameters = None, plot=False, model = 'extra-trees', N=100, scoring = 'f1'):
    # now make pairwise-prediction matrix between best time-point models
    best_params = {1: ['linear', 1, 0.46],
                   2: ['rbf', 10, 0.53],
                   3: ['rbf', 10, 0.49],
                   4: ['linear', 100, 0.53],
                   'all': ['rbf', 0.1, 0.51],}
    # load all time-points as test sets
    day_data = {}
    for i in range(1,5):
        X, y, groups, cg, pg = load_uic.load(day=[i], response_type=response_type, dataset=dataset)
        ratio_pg = np.round(pg.shape[0] / (cg.shape[0] + pg.shape[0]), 2)
        day_data[i] = [X,y, ratio_pg]

    # run model without cv and test on test set?
    combinations_triu = list(itertools.combinations([1, 2, 3, 4], 2))
    combinations_tril = list(itertools.combinations([4, 3, 2, 1], 2))
    combinations = combinations_triu+combinations_tril+[(1,1,),(2,2),(3,3),(4,4)]
    predictions = []

    for pair in combinations:
        # load first day in pair to train model
        params = best_params.get(pair[0])
        X, y, groups, cg, pg = load_uic.load(day=[pair[0]], response_type=response_type, dataset=dataset)
        ratio_pg_train = np.round(pg.shape[0] / (cg.shape[0] + pg.shape[0]), 2)
        # clf = SVC(C=params[1], kernel=params[0], class_weight='balanced', probability=False, random_state=0)
        if model == 'extra-trees':
            clf = Pipeline([('std', StandardScaler()),
                            ('feature_selection',
                             SelectFromModel(ExtraTreesClassifier(n_estimators=N,class_weight='balanced'))),
                            ('et', ExtraTreesClassifier(n_estimators=N,class_weight='balanced'))])

        elif model == 'svm':

            clf = Pipeline([
                ('normalization', MinMaxScaler((0, 1))),
                ('feature_selection', SelectKBest(k=128)),
                ('classifier',
                 SVC(C=0.1, kernel='rbf', class_weight='balanced',
                     probability=False, random_state=0, gamma='scale'))
            ])
            GridSearchCV(clf, params)
        clf.fit(X, y)
        # bring second day in pair to test
        X_test = day_data.get(pair[1])[0]
        y_test = day_data.get(pair[1])[1]
        ratio_pg_test = day_data.get(pair[1])[2]
        y_pred = clf.predict(X_test)
        y_pred = [int(n) for n in y_pred]
        # For interpretation purposes, we compute the decision confidence (i.e., normalized distance from boundary)
        # TODO add interpretation.py functions here.
        f1 = f1_score(y_test, y_pred)
        # acc = accuracy_score(y_test, y_pred)
        # logger.info(f1)
        roc_auc = roc_auc_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        # logger.info('\n', f1, parameter_set, '\n')
        support = [ratio_pg_train, ratio_pg_test]
        scores = [np.round(f1, 2),
                  np.round(roc_auc, 2),
                  np.round(precision, 2),
                  np.round(recall, 2)]
        predictions.append([pair,scores, support])


    # build pairwise prediction matrix

    importlib.reload(plot_outputs)

    prediction_matrix = pd.DataFrame(np.zeros((4,4))) #y axis is trained model, x axis is test set
    metric = 'f1'
    for i in predictions:
        if metric == 'roc_auc':
            prediction_matrix.iloc[i[0][0]-1,i[0][1]-1] = i[1][1] # i[1][1] is roc-auc, i[1][0] is f1
        elif metric == 'f1':
            prediction_matrix.iloc[i[0][0]-1,i[0][1]-1] = i[1][0] # i[1][1] is roc-auc, i[1][0] is f1
    rand = np.random.randint(1000,9999)

    plot_outputs.plot_heatmap(output_dir,prediction_matrix,['Day 1','Day 2','Day 3','Day 4'],'prediction_across_days_'+metric+'_'+str(rand), value_range=[0,1])

    #
    ratios = [predictions[0][2][0]]
    for i in predictions[1:4]:
        ratios.append((i[2][1]))

    support_all_days = pd.DataFrame(ratios, index= ['Day 1','Day 2','Day 3','Day 4'], columns =  ['Patient ratio'])
    print('\n' + str(support_all_days))

    return



if __name__ == '__main__':
    importlib.reload(config)
    input_dir = config.input_dir
    input_file = config.input_file
    output_dir = config.output_dir

    # mkdir and log
    path_to_dir = data_helpers.make_output_dir(os.path.join(output_dir))
    handler = logging.FileHandler(os.path.join(path_to_dir, 'self_training.log'))
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    

    # Parameters
    all_results_extra_trees = []
    all_results_svm = []

    segmented = False
    days = [1,2,3,4] #[1, 2, 3, 4]
    response_type = ['freeresp'] #TODO: careful ['freeresp', 'sentences'] in that order. change in loud_uic to make flexible
    model = 'extra-trees'  # extra-trees, svm
    remove_constant_columns = True# CAREFUL when set to True, eliminates cols
    # svm
    grid_search = False
    cv = 3
    scoring = 'f1'

    # others
    plot = False
    N = 100 #iterations
    round = 2
    round_pvalue = 2 #TODO: change to 4 once they improve
    provide_parameters = False
    summerize = False
    make_verbose = True

    # For days 1 through 5 or a subset:
    results = pd.DataFrame(columns=['F1', 'ROC-AUC', 'Precision', 'Recall'], index=['Day 1', 'Day 2', 'Day 3', 'Day 4'])
    logger.info('\n\nModel ' + model + str(response_type)+'==========================\n\n')
    for i in days:

        logger.info('\n\nDay '+str(i)+'======================================================================================\n\n')

        if grid_search:
            params = None
        else:
            # grid_searchon one train_test split, TODO prob. won't use, erase
            best_params = {1: ['linear', 1, 0.46],
                           2: ['rbf', 10, 0.53],
                           3: ['rbf', 10, 0.49],
                           4: ['linear', 100, 0.53],
                           'all': ['rbf', 0.1, 0.51], }
            params = best_params.get(i)
        # main 
        best_training_scores, scores, median_scores, permutation_scores, median_permutation_scores, pvalues = gss(model=model, grid_search= grid_search, day=[i],response_type=response_type,
                                                                                                   dataset='all', grid_search_parameters =params, N=N, scoring = scoring, cv = cv, remove_constant_columns = remove_constant_columns, segmented=segmented, make_verbose=make_verbose)
        # summerize results for one day:
        # logger.info('\nFinal results: ============================================================================')
        # logger.info('median ROC AUC score (pvalue) and permutation median: ', np.round(median_scores[0],round), np.round(median_permutation_scores[0],round), np.round(pvalues[0],4))
        # logger.info('median F1 score (pvalue) and permutation median: ', np.round(median_scores[1],round), np.round(median_permutation_scores[1], round), np.round(pvalues[1],4))
        # logger.info('median precision (pvalue) and permutation median: ', np.round(median_scores[2], round),np.round(median_permutation_scores[2], round), np.round(pvalues[2], 4))
        # logger.info('median recall score (pvalue) and permutation median: ', np.round(median_scores[3], round),np.round(median_permutation_scores[3],round), np.round(pvalues[3], 4))

        # make table
        logger.info(model+': '+str(response_type))
        data = [str(np.round(median_scores[1], round))+' (%s)' % np.round(pvalues[1], round_pvalue),
                   str(np.round(median_scores[0], round)) + ' (%s)' % np.round(pvalues[0], round_pvalue),
                   str(np.round(median_scores[2], round)) + ' (%s)' % np.round(pvalues[2], round_pvalue),
                   str(np.round(median_scores[3], round)) + ' (%s)' % np.round(pvalues[3], round_pvalue)]

        results.iloc[i-1] = data
        logger.info(results)
        logger.info(results.to_latex())

        # save data
        if model == 'extra-trees':
            all_results_extra_trees.append([best_training_scores, scores, median_scores, permutation_scores, median_permutation_scores, pvalues])
        elif model == 'svm':
            all_results_svm.append([best_training_scores, scores, median_scores, permutation_scores, median_permutation_scores, pvalues])
        if plot:
            plot_distributions(best_training_scores, scores, median_scores, permutation_scores, median_permutation_scores, pvalues, metric_name = 'roc_auc')
    # predict_across_samples(output_dir=path_to_dir, day=[1], grid_search=None, response_type=['freeresp', 'sentences'], dataset='all',
    participants = pd.read_csv(input_dir+'participants_0.csv')
    plot_outputs.plot_heatmap(output_dir, participants,
                              'participants', value_range=[0, 4])
    #                        grid_search_parameters=None, plot=False, model='extra-trees', N=100, scoring='f1')



    # Summarize results for all days:
    # if summerize: #TODO: improve
    #     logger.info('Day '+model+'================================================================================================\n\n')
    #     if model == 'extra-trees':
    #         results = all_results_extra_trees.copy()
    #     elif model == 'svm':
    #         results = all_results_svm.copy()
    #     logger.info('roc_auc')
    #     for i in results:
    #         logger.info(i[0][0], '(' + str(np.round(i[0][2], 3)) + ')')  # roc_auc
    #     logger.info('f1')
    #     for i in results:
    #         logger.info(i[0][1], '(' + str(np.round(i[0][2], 3)) + ')')  # f1
    # #                     todo put in latex df
    #



# original code
# def run(day=[1], response_type=['freeresp', 'sentences'], dataset='all'):
#     X, y, groups, cg, pg = load_uic.load(day=day, response_type=response_type, dataset=dataset)
#     audio_features = pd.read_csv(input_dir + input_file).columns[6:]
#     # X_normalized = data_helpers.normalize(array_train=X)
#     # X_normalized_kbest, kbest_features_names = data_helpers.f_feature_selection(X=X , y=y, k=1024, audio_features=audio_features, logger.info_=True)  # 0.5991, 0.48
#     # Obtain ROC AUC score for extra trees classifier
#     N = 64
#     clf = Pipeline([('std', StandardScaler()),
#                     ('feature_selection',
#                      SelectFromModel(ExtraTreesClassifier(n_estimators=N,
#                                                           class_weight='balanced'))),
#                     ('et', ExtraTreesClassifier(n_estimators=N,
#                                                 class_weight='balanced'))])
#
#     gss = GroupShuffleSplit(n_splits=N, test_size=0.2, random_state=0)
#     predictions = [[], []]
#     scores = []
#     importances = []
#     splits = []
#     for train_index, test_index in gss.split(X, y, groups=groups):
#         clf.fit(X[train_index], y[train_index])
#         outputs = clf.predict(X[test_index])
#         scores.append(roc_auc_score(y[test_index], outputs))
#         importances.append(clf.steps[-2][1].inverse_transform(clf.steps[-1][1].feature_importances_[None, :]))
#         predictions[0].extend(y[test_index])
#         predictions[1].extend(outputs)
#     scores = np.array(scores)
#     importances = np.squeeze(np.array(importances))
#
#     logger.info('median ROC AUC score: ', np.median(scores))
#     plt.figure(figsize=(8, 6))
#     plt.clf()
#     plt.hist(scores, N, color='gray', alpha=0.5)
#     plt.axvline(np.median(scores), color='r')
#     plt.title('Median AUC: {0:.2f}'.format(np.median(scores)))
#     plt.xlabel('AUC')
#
#     w_importances = scores.dot(importances) / np.sum(scores)
#     plt.hist(w_importances, 32)
#     index = np.argsort(w_importances)[::-1]
#     index = index[np.where(w_importances[index] > 0.002)]
#     # index
#
#     plt.plot(w_importances[index], np.arange(len(index)), '.-')
#     plt.yticks(np.arange(len(index)), cg.iloc[:, 2:-2].keys()[index])  # , rotation='vertical')
#     plt.ylabel('Features')
#     plt.xlabel('Importance score')
#     plt.gca().invert_yaxis()
#     results = permutation_test_score(clf, X, y, groups=groups, cv=gss, scoring=make_scorer(matthews_corrcoef))
#     logger.info('results 2: ', results[2])
#     return
#
# # for i in range(1,5):
# #     logger.info(i,'================\n')
# #     run(day=[i],  response_type=['freeresp','sentences'], dataset='all')
#
#
#
#
#
# 1