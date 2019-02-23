from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectFromModel, SelectFpr
from sklearn import preprocessing
import datetime
import os

def make_output_dir(output_dir):
    # TODO
    '''

    :param output_dir: 
    :return: 
    '''
    directory_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")  # random number so its not replaced
    path_to_dir = os.path.join(output_dir, directory_name)
    try:
        os.mkdir(os.path.join(output_dir, directory_name))
    except:
        print('file was not created or already exists')
    return path_to_dir



def normalize(array_train = None, array_test = None, test=False): #TODO put several options in if statements, change variable to scaler.
    min_max_scaler = preprocessing.StandardScaler()  #this method isn't working, returns inf on LSTM.
    # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))  # feature_range=(-1, 1). TODO: verify best method and if I should do on audio data
    X_minmax_train = min_max_scaler.fit_transform(array_train)
    if test:
        X_minmax_dev = min_max_scaler.fit_transform(array_test)
        return X_minmax_train, X_minmax_dev
    else:
        return X_minmax_train


# feature selection methods
# =============================================================================
def l1_feature_selection(X = None, y = None):
    lclf = LinearSVC(C = .01, penalty="l1", dual=False).fit(X, y)
    model = SelectFromModel(lclf, prefit=True)
    X_new = model.transform(X)
    return X_new


def f_feature_selection(X=None, y=None, k=32, audio_features=None, print_=True):
        kbest_features= SelectKBest(f_classif, k=k)
        X_new = kbest_features.fit_transform(X, y)
        kbest_features_names = []
        if print_==True:
            print('best features: ')
            for i in kbest_features.get_support(indices=True):
                print('index ', i, audio_features[i])
                kbest_features_names.append(audio_features[i])
        return X_new, kbest_features_names

def feature_selection_fpr(X=None, y=None, alpha=0.01):
    X_new = SelectFpr(f_classif, alpha=alpha).fit_transform(X, y)
    return X_new