#TODO: figure out which libraries might be used by other features.
# TODO: add contributors to each script
# TODO: replace config.config with config_params.

# from _elementtree import Element
# import codecs
# from collections import Counter
# import csv
import logging
# import datetime
# import math
import os
# import operator
# from random import shuffle
# import string

# import nltk
# from nltk.tbl import feature
# from numpy import float16
from scipy import sparse
import scipy
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import pandas as pd
# from scipy.sparse.coo import coo_matrix
from sklearn.model_selection import cross_validate
# from sklearn import preprocessing
# from sklearn import svm, metrics, naive_bayes
# from sklearn.cross_validation import StratifiedKFold
# from sklearn.datasets import load_svmlight_file
# from sklearn.feature_selection import RFECV
# from sklearn.model_selection import cross_val_score
# from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
# from sklearn.grid_search import GridSearchCV
# from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
# from sklearn.metrics import jaccard_similarity_score
# from sklearn.naive_bayes import BernoulliNB, MultinomialNB
# from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
# from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from sklearn.feature_selection import SelectFromModel, SelectFpr
# import tensorflow as tf
# import tensorflow_hub as hub

# from gensim.models.doc2vec import Doc2Vec, TaggedDocument



import feature_generator
# from process_properties import PreProcessor
import data_handler
import config
import data_helpers
import plot_outputs




#from template_functions import *
#from scipy import csr_matrix

#
#
# if config.local_or_cluster:
#     # directory_name = '18-06-10-20-12-59'
#     directory_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
#     file_name = 'cnn'
# else:
#     # directory_name = '18-06-10-20-12-59'
#     directory_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
#     file_name = os.path.basename(__file__)
#
# print('running '+directory_name+' '+file_name)


# def output_to_dir(output_dir): #TODO
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


config_params = config.config
input_dir = config_params['input']

path_to_dir = data_helpers.make_output_dir(os.path.join(config_params['output_dir'], 'baselines/'))
handler = logging.FileHandler(os.path.join(path_to_dir,'self_training.log'))
handler.setLevel(logging.INFO)
logger.addHandler(handler)

sentObject = SentimentIntensityAnalyzer()
dataHandler = data_handler.DataHandler(config_params)
# def doc2vec(documents):
#     model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
#     return model


def createRegularTrainingFeatureVector(featureList,config_params,featureNames,featGenr,\
                                       feature_cache,maxFeatPosn,vectors,\
                                       unknown_vec,featPosnNorm):
    indptr = [0]
    indices = []
    data = []

    labels =[]

    totalPos = 0
    totalNeg = 0
    neutral = 0


    training_data = dataHandler.loadInputData(type='train')

    categories = training_data.get('labels')

    for index,quote in enumerate(training_data.get('input')):

        label = categories[index]


        featureMap = {}

        ''' try each feature(s) set '''

        if 'TAGQ' in featureNames:
            tags = featGenr.generateTagqFeatures(quote,response,'TARGET')
            featureMap.update(tags)


        if 'INTERJ' in featureNames:
            interjs = featGenr.generateInterjFeatures(quote,response,'TARGET')
            featureMap.update(interjs)

        if 'EMBED' in featureNames:
            embeds = featGenr.generateEmbeddingFeatures(quote,vectors,unknown_vec,100) #TODO: set 100 in config
            featureMap.update(embeds)

        if 'VADER' in featureNames:
            vader_senti = featGenr.getSentiVaderFeatures(quote, 'TARGET')
            featureMap.update(vader_senti)

        if 'JACCARD' in featureNames:
            jaccards = featGenr.generateJaccardFeatures(quote,response)
            featureMap.update(jaccards)
            #      jaccard_bin = convertToBin(jaccard_arg,'jaccard_arg')
            #      if jaccard_bin is not None:
            #         featureMap[jaccard_bin] = 1.0

        if 'PUNCT' in featureNames:
            puncts = featGenr.generatePunctFeatures(response)
            featureMap.update(puncts)

        if 'MODAL' in featureNames:
            modals = featGenr.generateModalFeatures(quote,'TARGET')
            featureMap.update(modals)

        if 'DISCOURSE' in featureNames:
            discourses = featGenr.generateDiscourseFeatures(quote,'TARGET')
            featureMap.update(discourses)

        if 'AGREE_LEX' in featureNames:
            agreements = featGenr.generateAgreeDisagreeFeatures(quote,response,'TARGET')
            featureMap.update(agreements)

        if 'NGRAM' in featureNames:
            ngrams = featGenr.generateNGramFeatures(quote,'TARGET')
            featureMap.update(ngrams)

        if 'FIRSTLAST' in featureNames:
            ngrams = featGenr.generateLexFeatures(quote,'TARGET')
            featureMap.update(ngrams)

        if 'SENTI' in featureNames :
            sentis =  featGenr.generateMPQAFeatures(quote,'TARGET')
            featureMap.update(sentis)

        if 'HYPER' in featureNames:
            hypers = featGenr.generateHyperFeatures(quote,response,'TARGET')
            featureMap.update(hypers)

        if 'PUNCT' in featureNames:
            puncts = featGenr.generatePunctFeatures(response)
            featureMap.update(puncts)

        if 'EMOT' in featureNames:
            emots = featGenr.generateEmotFeatures(quote,response,'BOTH')
            featureMap.update(emots)

        if 'HEDGE' in featureNames:
            hedge = featGenr.generateHedgeFeatures(quote,'TARGET')
            featureMap.update(hedge)

        if 'PERSON' in featureNames:
            persons = featGenr.getPersonFeatures(quote,'TARGET')
            featureMap.update(persons)

        if 'SARC' in featureNames:
            sarc_map ={}
            sarc_map[sarcasm] = 1.0
            featureMap.update(sarc_map)

        if 'TOPVERBS' in featureNames:
            verbs = getVerbFeatures(config_params['topverbs'],argument)
            featureMap.update(verbs)

        if 'LIWC' in featureNames:
            allLiwcFeatures = featGenr.generateLIWCFeatures(quote,'TARGET')
            featureMap.update(allLiwcFeatures)
            #       liwcs = getLIWCFeatures(liwcFeats,argument)
            #       featureMap.update(liwcs)





 #       fv = createFeatureVector(featureMap,featureList,maxFeatPosn,featPosnNorm)
        fv = createFeatureVectorForSVM(featureMap,featureList,maxFeatPosn,featPosnNorm)


     #   label = convert(label)
        if label == 1.0:
            totalPos+=1
        if label == 0.0:
            totalNeg+=1

        for feature in fv:
            indices.append(feature[0])
            data.append(feature[1])

        indptr.append(len(indices))

        labels.append(label)

    msg = 'total (pos,neg) is ' + str(totalPos) + ' ,' + str(totalNeg)
    logger.info(msg)

    trainingData = data,indices,indptr,labels

    return trainingData,feature_cache

def createFeatureVector(featureMap,allFeatures,maxFeatPosn,featPosnNorm):

    valList = []
    for index,key in enumerate(allFeatures):
        value = featureMap.get(key)
        if value is not None and value > 0 :
            valList.append(featPosnNorm+index) # = value

    ''' there is an issue in # of features for train/test in scikit learn'''
    ''' we need to set explicitly the "max" of feature for train otherwise there is '''
    ''' len(Feature) issue'''
    ''' SOLVED (above): use transform function?'''

#    if featureType == 'pattern':
#        return valList

    valList.append(maxFeatPosn)

    '''
    key = allFeatures[-1]
    value = featureMap.get(key)
    if value is None  :
        #valMap[len(allFeatures)] = 0.0
        #valList.append(len(allFeatures)-1) # = value
        valList.append(maxFeatPosn)
    '''
    return valList


'''
def getDiscourseMarkers(discourses,argument):
    
    featureMap = {}
    words1 = nltk.word_tokenize(argument.lower(), language='english')
    count = 0.0
    for word in words1:
        if 'discourse|||'+word.strip() in discourses:
            featureMap[ 'discourse|||'+word.strip()] =1.0
    
# not for the single words!!!
    for discourse in discourses:
        ds  = discourse.split()
        if len(ds)>1:
            discourse = discourse.replace('discourse|||','')
            disc_txt = discourse.replace(', ',' ').strip()
            run_text_nopunct = argument.translate(string.maketrans("",""), string.punctuation)
            if disc_txt.strip() in run_text_nopunct:
                featureMap[ 'discourse|||'+disc_txt.strip()] =1.0
 
    
    
    return featureMap
'''
'''
def getVerbFeatures(verbs,argument):
    
    featureMap = {}
    words1 = nltk.word_tokenize(argument.lower(), language='english')
    count = 0.0
    for word in words1:
        if 'verb|||'+word.strip() in verbs:
            featureMap[ 'verb|||'+word.strip()] =1.0
            
    return featureMap

def getNgrams(ngrams,argument):
    
    featureMap = {}
    words1 = nltk.word_tokenize(argument.lower().decode('utf8'), language='english')
    count = 0.0
    for word in words1:
        if 'unigram|||'+word.strip() in ngrams:
            featureMap[ 'unigram|||'+word.strip()] =1.0
            
    return featureMap




def getHedgeFeatures(hedges,argument):
    
    featureMap = {}
    words1 = nltk.word_tokenize(argument.lower(), language='english')
    count = 0.0
    for word in words1:
        if 'hedge|||'+word.strip() in hedges:
            featureMap[ 'hedge|||'+word.strip()] =1.0
    
    # not for the single words!!!
    for hedge in hedges:
        ds  = hedge.split()
        if len(ds)>1:
            hedge = hedge.replace('hedge|||','')
            hedge = hedge.replace(', ',' ').strip()
            run_text_nopunct = argument.translate(string.maketrans("",""), string.punctuation)
            if hedge.strip() in run_text_nopunct:
                featureMap[ 'hedge|||'+hedge.strip()] =1.0
 
    
    
    return featureMap

'''


def createFeatureVector(featureMap,allFeatures,maxFeatPosn,featPosnNorm):

    valList = []
    for index,key in enumerate(allFeatures):
        value = featureMap.get(key)
        if value is not None and value > 0 :
           # if 'JACCARD' in allFeatures:
           #     valList.append(value) # = value

            valList.append(featPosnNorm+index) # = value

    ''' there is an issue in # of features for train/test in scikit learn'''
    ''' we need to set explicitly the "max" of feature for train otherwise there is '''
    ''' len(Feature) issue'''
    ''' SOLVED (above): use transform function?'''

#    if featureType == 'pattern':
#        return valList

    valList.append(maxFeatPosn)

    '''
    key = allFeatures[-1]
    value = featureMap.get(key)
    if value is None  :
        #valMap[len(allFeatures)] = 0.0
        #valList.append(len(allFeatures)-1) # = value
        valList.append(maxFeatPosn)
    '''
    return valList


def createFeatureVectorForSVM(featureMap,allFeatures,maxFeatPosn,featPosnNorm):

    valList = []
    for index,key in enumerate(allFeatures):
        value = featureMap.get(key)

        if value is not None and value > 0  :

          #  svmBuffer = svmBuffer + str(index+1) + ':' + str(value) + ' '
            tuple = featPosnNorm+index,value
            valList.append(tuple)
      #  svmBuffer = svmBuffer # + ' ' + fileIdStr

    ''' there is an issue in # of features for train/test in scikit learn'''
    ''' we need to set explicitly the "max" of feature for train otherwise there is '''
    ''' len(Feature) issue'''
    '''
    key = allFeatures[-1]
    value = featureMap.get(key)
    if value is None  :
        svmBuffer = svmBuffer + str(len(allFeatures)-1+1) + ':' + '0.0' + ' '
    
    svmBuffer = svmBuffer.strip()
    return svmBuffer
    '''
    tuple = maxFeatPosn,0
    valList.append(tuple)
    '''
    key = allFeatures[-1]
    value = featureMap.get(key)
    if value is None  :
        #valMap[len(allFeatures)] = 0.0
        #valList.append(len(allFeatures)-1) # = value
        valList.append(maxFeatPosn)
    '''
    return valList


def avg_feature_vector(words, model, num_features, index2word_set):
        #function to average all words vectors in a given paragraph
        featureVec = np.zeros((num_features,), dtype="float32")
        nwords = 0

        #list containing names of words in the vocabulary
        #index2word_set = set(model.index2word) this is moved as input param for performance reasons
        for word in words:
            if word in index2word_set:
                nwords = nwords+1
                featureVec = np.add(featureVec, model[word])

        if(nwords>0):
            featureVec = np.divide(featureVec, nwords)
        return featureVec

# def load_audio_features()

# TODO: change name to gridsearch_CV
def TrainAndValidate(X_train=None, y_train=None,X_test=None,y_test=None,class_weight = 'balanced', scoring_classification = ['f1','roc_auc', 'precision', 'recall'], probability = True, Cs=[0.01, 0.1, 1, 10, 100], norms=None, kernels = ['linear', 'rbf'], cv = 5, k_audio_features=32, perform_cross_validation=True, perform_train_dev_split=True):
    # TODO: change name to gridsearch
    # TODO: add different bl models like regression
    # TODO: add scoring_regression
    '''
    :param X_train: shape = (N, M) 
    :param y_train: 
    :param X_test: 
    :param y_test: 
    :param class_weight: 
    :param scoring_classification: grid search will return best parameters of first metric.  
    :param probability: enable probability estimates. Will slow down, but enables confidence estimation.
    :param Cs: 
    :param norms: 
    :param kernels: 
    :param cv: 
    :param k_audio_features: 
    :param perform_cross_validation: 
    :return: 
    '''
    parameter_sets = []
    for kernel_ in kernels:
        for C in Cs:
            parameter_sets.append([kernel_, C])
    scores_mean_all = []
    for parameter_set in parameter_sets:
        logger.info(str(parameter_set))
        kernel = parameter_set[0]
        C = parameter_set[1]
        # if kernel == 'linear':
        #     clf = LinearSVC(C=C, class_weight=class_weight, random_state=0) #only cv
        #     clf = CalibratedClassifierCV(LinearSVC(C=C, class_weight=class_weight, random_state=0))
        # else:
        clf = SVC(C=C, kernel=kernel, class_weight=class_weight, probability=probability, random_state=0)
        print('training model...\n')

        if perform_cross_validation:
            logger.info('=======executing cross validation model')
            scores = cross_validate(clf, X_train, y_train, scoring = scoring_classification, cv = cv, return_train_score = False)
            # scores_mean = [] #TODO: do automatically depending on metrics chosen
            # for metric in scoring_classification:
            scores_mean = [round(scores['test_f1'].mean(),2),
                           round(scores['test_f1'].std(),2),
                           round(scores['test_roc_auc'].mean(),2),
                           round(scores['test_precision'].mean(),2),
                           round(scores['test_recall'].mean(),2),
                           parameter_set[0],parameter_set[1] ]
            scores_mean_all.append(scores_mean)
            print(str(round(scores['test_f1'].mean(),4))+' '+str(parameter_set))
            print(str(round(scores['test_f1'].std(),4)) + ' ' +'\n\n')
           # #     X_test_csr = X_test.tocsr()
           #      predictions_all = lr1.predict(X_test)
           #      acc_score_all = accuracy_score(Y_test, predictions_all)
           #      f1_score_all = f1_score(Y_test, predictions_all,average='macro')
           #      if f1_score_all > best_f1:
           #          best_f1 = f1_score_all
           #          best_c = C
           #          best_norm = norm
           #          best_kernel = kernel
           #      print ('cost and norm ' + str(C) + ' ' + str(norm))
           #  logger.info("Classification report for classifier %s:\n%s\n" % (scores, metrics.classification_report( Y_test, predictions_all,digits=3)))
        # print ('best params (macro-avg): ',best_f1, best_c, best_norm, best_kernel)
        else:
            # perform train-dev split
            logger.info('=======executing train-dev split model')
            text = dataHandler.loadInputData(type='train').get('input')
            X_train, X_dev, y_train, y_dev, X_train_text, X_dev_text = train_test_split(X_train, y_train, text[:4974], test_size=0.20, random_state=0,
                                                              shuffle=False)  # TODO: save and add test_size to parameters.
            y_dev = [int(n) for n in y_dev]
            # TODO: shuffle, I think text has some of the test set.
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_dev)
            y_pred = [int(n) for n in y_pred]
            # For interpretation purposes, we compute the decision confidence (i.e., normalized distance from boundary)
            # TODO add interpretation.py functions here.


            f1 = f1_score(y_dev, y_pred)
            acc = accuracy_score(y_dev, y_pred)
            print(f1)
            roc_auc = roc_auc_score(y_dev, y_pred)
            precision = precision_score(y_dev, y_pred)
            recall = recall_score(y_dev, y_pred)
            print('\n', f1, parameter_set, '\n')
            scores = [round(f1,2),
                      '-',
                      round(roc_auc,2),
                      round(precision,2),
                      round(recall,2),
                      parameter_set[0],
                      parameter_set[1] ] # the '-' is for f1 std, not available with
            scores_mean_all.append(scores) #TODO fix name (not "mean" for train-dev split)
    # Print latex file for all results
    scores_mean_all_sorted = sorted(scores_mean_all, key=lambda x: x[0])  # TODO change name. Sorted from small to large.
    results_all = pd.DataFrame(scores_mean_all_sorted)

    columns = scoring_classification + ['parameter1', 'parameter2'] #TODO: do dynamically depending on amount of hyparameters
    columns.insert(1, 'f1 std')
    results_all.columns = columns
    logger.info(str(results_all.to_latex()))
    # Print latex file for best result
    results_best = results_all.iloc[-1,:]
    if perform_cross_validation:
        index = 'SVM 5-fold CV'
    else:
        index = 'SVM train-dev split'
    results_best = pd.DataFrame(results_best).T
    results_best.index = [index]
    logger.info(str(results_best.to_latex()))
    # best_params = scores_mean_all_sorted[-1]
    # logger.info('best params (means of f1, roc_auc, recall, recall): ' + str(best_params))
    return results_best







def generateFeatureAndClassification(config_params, perform_cross_validation=True):
    toy = config_params['toy']
    run_text = config_params['run_text']
    run_audio = config_params['run_audio']
    create_features = config_params['create_features']
    test = config_params['test']
    inputPath = config_params['inputPath']
    output_dir = config_params['output_dir']
    trainingFile = config_params.get('trainingFile')
    feature_numbers =  int(config_params.get('feature_numbers'))
    featureNames = config_params.get('features').split(',')
    dataHandler = data_handler.DataHandler(config_params)
    featGenr = feature_generator.FeatureGenerator(featureNames,config_params)
    logger.info('Feature names are initialized')



    if run_text:
        if 'EMBED' in featureNames:
            print('loading embeddings...\n')
            vocabs = dataHandler.loadAllVocabs(inputPath)
            vectors = dataHandler.loadEmbedding(vocabs,vector_length=100) # TODO: save loaded version, cause it takes a while

        allFeatureList = featGenr.initFeatures()
        logger.info('features are initialized...')
     #   maxFeatPosn = len(allFeatureList)
     #    Train run_text data
        training_data = dataHandler.loadInputData(type='train')
        logger.info('training data loaded...')

        if test:
            dev_data = dataHandler.loadInputData(type='dev')
        # logger.info(trainingFile)

    #    test_data = loadInputData(inputPath+trainingFile,type='test')
    #    logger.info('test data loaded...')
        if create_features:
            training_regular_cache = []
            dev_regular_cache = []
            test_regular_cache = []

            boundary_index = len(allFeatureList)#.index('BOUNDARY_FEATURE')
            vec_length = 100 #TODO: automatically shape from glove embeddings file name
            unknown_vec = np.random.normal(0,0.17,vec_length)
            print('loading features...')
            training_reg_features,training_regular_cache = createRegularTrainingFeatureVector \
            (allFeatureList,config_params,featureNames,featGenr,training_regular_cache,boundary_index,vectors,unknown_vec,0) #TODO: takes several minutes

            regTrainingData, indices, indptr, y_train_text = training_reg_features[0],training_reg_features[1],\
            training_reg_features[2],training_reg_features[3]

            X_train_text = scipy.sparse.csr_matrix((regTrainingData, indices, indptr))
            print('normalizing...')
            X_train_text_normalized = data_helpers.normalize(array_train=X_train_text.toarray())
            print("feature selection...================================================")
            # print(X_train_text_normalized.shape)
            X_train_text_normalized_best, kbest_features_names_text= data_helpers.f_feature_selection(X=X_train_text_normalized, y=y_train_text, k=1000,
                                                               audio_features=allFeatureList,
                                                               print_=True)  # 0.5991, 0.48
            # np.savez_compressed('./X_train_text', a = X_train_text_normalized_best)
            # np.save('./y_train_text.npy', y_train_text)



            if test: #TODO
                #  printFeatures(X_pat_train)
                dev_reg_features, dev_regular_cache = createRegularTrainingFeatureVector(allFeatureList,dev_data, featureNames,
                                                                                         featGenr, dev_regular_cache,
                                                                                         boundary_index, vectors,
                                                                                         unknown_vec, 0)

                regDevData, indices, indptr, Y_dev = dev_reg_features[0], dev_reg_features[1], \
                                                     dev_reg_features[2], dev_reg_features[3]
                X_dev = scipy.sparse.csr_matrix((regDevData, indices, indptr))
        else:

            X_train_text_normalized_best = np.load(os.path.join(input_dir , 'X_train_text.npz'))['a']
            y_train_text = np.load(os.path.join(input_dir , 'y_train_text.npy'))




        # Feature selection
        # Text
        logger.info('All vector is loaded...')

        allFeatureList.append('BLANK_FEATURE')

        # feat_len = min(feature_numbers,len(allFeatureList))


        # nonzero_per_example = []
        # for i in range(len(X_train_text_normalized_best)):
        #     nonzero_per_example.append(np.count_nonzero(X_train_text_normalized_best[i]))
        #
        # print(np.mean(nonzero_per_example))

       #  if test:
       #      X_best_dev = X_kbest_reg.transform(X_dev)
            # X_best_test = X_kbest_reg.transform(X_test)

    # Audio
    # ========================================
    if run_audio:
        ## Load
        if create_features: #TODO: change trainAndValidate to gridsearch, and add k as a hyperparameter
            X_train_audio, y_train_audio, audio_features = dataHandler.load_audio_data(type='train') #TODO fix paths within load_audio_data
            #
            X_train_audio_normalized = data_helpers.normalize(array_train=X_train_audio)
            ## Feature selection #TODO: turn into function
            print('====================================feature selection...====================================')
            print(X_train_audio_normalized.shape)
            # X_train_audio_normalized_best = l1_feature_selection(X=X_train_audio_normalized, y = y_train_audio) #61.42 and 49.71
            # TODO: select k with gridsearch
            X_train_audio_normalized_best, kbest_features_names_audio = data_helpers.f_feature_selection(X=X_train_audio_normalized, y=y_train_audio, k=32, audio_features=audio_features, print_=True) #0.5991, 0.48
            # X_train_audio_normalized_best = data_helpers.f_feature_selection(X=X_train_audio_normalized, y=y_train_audio, k=32, audio_features=audio_features, print_=True) #k=32: 0.5934, 0.4740
            # X_train_audio_normalized_best = feature_selection_fpr(X=X_train_audio_normalized, y=y_train_audio, alpha=0.01)
            print(X_train_audio_normalized_best.shape)
            # np.savez_compressed('./X_train_audio', a = X_train_audio_normalized_best, b=y_train_audio)
        else:
            X_train_audio_normalized_best, y_train_audio = np.load(os.path.join(input_dir, 'X_train_audio.npz'))['a'], np.load(os.path.join(input_dir, 'X_train_audio.npz'))['b']

    if run_text and run_audio:
        '''test: are the labels of audio and run_text the same? If so then the below sum should return 6218
        y_train = np.array([int(n) for n in y_train])
        print(np.sum(y_train_audio == y_train))
        '''
        # concatenate features
        X_train_all = np.concatenate((X_train_text_normalized_best, X_train_audio_normalized_best), axis=1)
        # print(X_train_all.shape)
        if toy:

            logger.info('Running multimodal SVM gridsearch============================================================')
            TrainAndValidate(X_train=X_train_all, y_train=y_train_text,
                             perform_cross_validation=perform_cross_validation, kernels = ['linear'], Cs = [1, 10])

            # logger.info(
            #     'Running run_text SVM gridsearch==================================================================')
            # TrainAndValidate(X_train=X_train_text_normalized_best, y_train=y_train_text,
            #                  perform_cross_validation=perform_cross_validation, kernels = ['linear'], Cs = [1, 10])
            #
            # logger.info('Running audio SVM gridsearch=================================================================')
            # TrainAndValidate(X_train=X_train_audio_normalized_best, y_train=y_train_audio,
            #                  perform_cross_validation=perform_cross_validation, kernels = ['linear'], Cs = [1, 10])
        else:
            logger.info('Running multimodal SVM gridsearch============================================================')
            TrainAndValidate(X_train=X_train_all, y_train=y_train_text, perform_cross_validation=perform_cross_validation)

            logger.info('Running run_text SVM gridsearch==================================================================')
            TrainAndValidate(X_train=X_train_text_normalized_best, y_train=y_train_text, perform_cross_validation=perform_cross_validation)

            logger.info('Running audio SVM gridsearch=================================================================')
            TrainAndValidate(X_train=X_train_audio_normalized_best, y_train=y_train_audio, perform_cross_validation=perform_cross_validation) # Cs=[10],kernels=['rbf']

    elif run_text and not run_audio:
        logger.info('Running run_text SVM gridsearch==================================================================')
        TrainAndValidate(X_train=X_train_text_normalized_best, y_train=y_train_text, perform_cross_validation=perform_cross_validation)

    elif run_audio and not run_text:

        logger.info('Running audio SVM gridsearch=================================================================')
        TrainAndValidate(X_train=X_train_audio_normalized_best , y_train=y_train_audio, perform_cross_validation=perform_cross_validation)


# def loadParameters(configFile):

#     processorObj = PreProcessor(configFile)
#     return processorObj.getArgs()

# def main(argv):

    # config_params = loadParameters(argv[1])
    # config_params = loadParameters(config)
  #  input = config_params['input']
  #  svm_op = config_params['output']
  #  utilpath = config_params['util']
 #   featureList = config_params['features']

  #  featureList = ['NGRAM','HEDGE','LIWC','VADER', 'SENTI','MODAL','EMBED','PERSON',
   #                'DISCOURSE']


#    featureList = arguFeatList

   # featureList = ['NGRAM', 'MODAL','SENTI','AGREE_LEX',  \
    #               'DISCOURSE', 'VADER','EMBED','TAGQ', 'HYPER', 'LIWC', 'PUNCT', 'EMOT' ]

  #  featureList = ['PUNCT', 'INTERJ','LIWC','EMBED','EMOT','HYPER','NGRAM']

 #   featureList = ['EMBED']


    vectors = None
  #  featureList = ['TAGQ']
  #   generateFeatureAndClassification(config_params)

if __name__ == '__main__':
    # Parameters
    # y_train_text = np.load(os.path.join(input_dir, 'y_train_text.npy')) #TODO: maybe do with y_test
    # y_train, y_dev = train_test_split(y_train_text,test_size=0.20, random_state=0,shuffle=True)
    # distributions, significance_scores = data_helpers.permutation_test(y_test=y_dev, alpha=0.01)
    for i in [True, False]:
        if i == True:
            logger.info('Running Cross')
        # main(config)
        generateFeatureAndClassification(config_params, perform_cross_validation=i)
        # main(sys.argv[1:])
