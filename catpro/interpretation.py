import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import data_helpers
import config


config_params = config.config

def svm_confidence(clf, X_dev):
    normalized_distance = data_helpers.normalize(pd.DataFrame(clf.decision_function(X_dev)), feature_range=(-1,1)) #normalized distance from SVM margin
    # confidence =  [str(np.round(np.abs(n[0])* 100, 1))+ '%' for n in normalized_distance ]
    confidence =  [np.round(np.abs(n[0]), 2) for n in normalized_distance ]
    return normalized_distance, confidence


def top_k_feature_values(X_dev=None, X_dev_text= None, y_dev = None, y_pred = None, clf= None, kbest_features_names_text = None, kbest_features_names_audio = None, total_text_features=1000, total_audio_features = 32, k=10):
    '''
    these values were already normalized between 0,1 as default during preprocessing
    :return: 
    '''
    # Extract indexes of feature vector from feature names
    ## Text
    feature_indexes_text = []
    feature_names = ['embed|||3',
                     'embed|||18',
                     'YOU_SECOND', 'WE_PLURAL',
                     '11.csv', '14.csv',
                     'TARGET_VADER|||compound',
                     'unigram|||.', 'unigram|||depression',
                     'pos|||good',
                     'hedge|||basically',
                     'TARGET|||DISCOURSE|||but',
                     ]
    for feature in feature_names:
        feature_indexes_text.append(kbest_features_names_text.index(feature))
    ## Audio
    feature_indexes_audio = []
    feature_names = ['covarep_0_mean',
                     'covarep_22_mean',
                     'covarep_0_min', 'covarep_0_max','covarep_61_max',
                     'covarep_0_median', 'covarep_22_median',
                     'covarep_48_std', 'covarep_70_std', 'covarep_65_skew']
    for feature in feature_names:
        feature_indexes_audio.append(kbest_features_names_audio.index(feature))
    ### since these are appended to N text features, we add N to these indexes:
    feature_indexes_audio = [n+total_text_features for n in feature_indexes_audio]

    # Main
    confidences = clf.predict_proba(X_dev)
    confidences = [np.round(np.max(n),2) for n in confidences]

    interpretations = []
    for i in range(len(X_dev)):
        text = X_dev_text[i]
        label = y_dev[i]
        prediction = y_pred[i]
        confidence = confidences[i]
        features_all = X_dev[i]
        features_text = np.array([features_all[i] for i in feature_indexes_text])
        features_audio = np.array([features_all [i] for i in feature_indexes_audio])
        interpretations.append([text, label, prediction, confidence, features_text,features_audio])
    interpretations = np.array(interpretations)
    # obtained from covarep matlab order, see below
    dimension_names = ['label', 'prediction', 'confidence', 'embed dim 3',
                     'embed dim 18',
                     'YOU_SECOND', 'WE_PLURAL',
                     'LIWC 11', 'LIWC 14',
                     'sentiment_score',
                     'unigram: .', 'unigram: depression',
                     'pos: good',
                     'hedge: basically',
                     'DISCOURSE: but',
                       'f0 mean',
                       'MCEP 11 mean',
                       'f0 min', 'f0 max', 'HMPDD 0 max',
                       'f0 median', 'MCEP 11 median',
                       'HMPDM 12 std', 'HMPDD 9 std', 'HMPDD 9 skew']
    return interpretations, dimension_names



def interpretations_heatmap(interpretations, dimension_names): #output_dir, df_corr, column_names, output_file_name='similarity_experiment',with_corr_values=True, ):
    for index, segment in enumerate(interpretations):
        # expand feature values and round
        values  = np.concatenate([segment[1:-2], segment[-2], segment[-1] ])
        rounded_values = []
        for i in values:
            try: i=np.round(i,2)
            except: pass
            rounded_values.append(i)
        df = pd.DataFrame(rounded_values, index = dimension_names)

        plt.clf()
        sns.heatmap(df, cmap="RdPu", vmin=0, vmax=1.0, cbar_kws={"ticks": [0.0, 0.5, 1.0]}, annot=True)
                    # , annot=with_corr_values)
        plt.yticks(rotation=0)
        title = segment[0].split()
        if len(title)/2 >= 4:
            title.insert(int(len(title) / 2), '\n')
        title = ' '.join(title)
        plt.title(title)
        plt.tick_params(axis=u'both', which=u'both', length=0)
        plt.xticks([])
        plt.tight_layout(2)
        plt.savefig(config.config['output_dir']+ 'interpretation/interpreation_heatmap_'+str(index)+'.png', dpi=200)
        return


# Covarep order:
# https://github.com/covarep/covarep/blob/master/feature_extraction/COVAREP_feature_extraction.m
#
names=['F0','VUV','NAQ','QOQ','H1H2','PSP','MDQ','peakSlope','Rd',
    'Rd_conf','creak','MCEP_0','MCEP_1','MCEP_2','MCEP_3','MCEP_4','MCEP_5',
    'MCEP_6','MCEP_7','MCEP_8','MCEP_9','MCEP_10','MCEP_11','MCEP_12',
    'MCEP_13','MCEP_14','MCEP_15','MCEP_16','MCEP_17','MCEP_18',
    'MCEP_19','MCEP_20','MCEP_21','MCEP_22','MCEP_23','MCEP_24',
    'HMPDM_0','HMPDM_1','HMPDM_2','HMPDM_3','HMPDM_4','HMPDM_5',
    'HMPDM_6','HMPDM_7','HMPDM_8','HMPDM_9','HMPDM_10','HMPDM_11','HMPDM_12',
    'HMPDM_13','HMPDM_14','HMPDM_15','HMPDM_16','HMPDM_17','HMPDM_18',
    'HMPDM_19','HMPDM_20','HMPDM_21','HMPDM_22','HMPDM_23','HMPDM_24',
    'HMPDD_0','HMPDD_1','HMPDD_2','HMPDD_3','HMPDD_4','HMPDD_5',
    'HMPDD_6','HMPDD_7','HMPDD_8','HMPDD_9','HMPDD_10','HMPDD_11','HMPDD_12']
