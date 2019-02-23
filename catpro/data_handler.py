

# TODO: change paths to config
# TODO: merge with data_helpers.py

from collections import defaultdict
import codecs
from collections import Counter
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

import config

config_params = config.config

class DataHandler:
    def  __init__(self,kwargs):
        print('inside data handler')
        self.kwargs = kwargs

    def loadEmbedding(self,vocabs,vector_length):

        path = '/Users/danielmlow/Dropbox/data/glove.6B/'
        word_vec_file = 'glove.6B.100d.txt'

       # glove_path = '/home/z003ndhf/work_debanjan/data/glove_vectors/'
       # glove_file = glove_path + 'glove_' + source + '.txt'
        model = {}
        k = -1

        with open(path+word_vec_file,'r') as f:
            for line in f:
              #  print line
                features = line.split()
                token = features[0]
                if token not in vocabs:
                    continue
                vector = np.array(features[1:],dtype="float32")
                model[token] = vector
                k = len(vector)

        print ('word model is loaded...')
        print ('# of word dimensions is '  + str(k))
        return model


    def loadAllVocabs(self,inputPath):

        cutoff = 5
        file1 = 'depression_all_data.txt'

        f = open(inputPath+file1)
        allWords = []
        for line in f:

            elements = line.strip().split('\t')
            quote = elements[6].lower().strip()
            allText = quote
            words = allText.lower().split()
      #      words = nltk.word_tokenize(allText.lower(), language='english')
            allWords.extend(words)

        f.close()
        cntx = Counter( [ w for w in allWords ] )
        lst = [ x for x, y in cntx.items() if y > cutoff ]

        return lst

    def load_audio_data(self, type):
        test = self.kwargs.get('test')
        inputPath = self.kwargs.get('input')
        audio_file = self.kwargs.get('audio_file')
        inputPath = config_params['input']
        audio_file = 'text_audio_df.csv'
        text_audio = pd.read_csv(inputPath + audio_file)
        # One row had NaN value, detected:
        # for row_i in range(text_audio.shape[0]):
        #     row = text_audio.iloc[row_i,:]
        #     if row.isna().sum()>1:
        #         break
        # So we replace with previous question or answer
        text_audio.iloc[12927, :]=text_audio.iloc[12927-2, :]
        # Train
        text_audio_train = text_audio[text_audio.FILE_TYPE=='TRAIN']  # TODO load text from here too?
        text_audio_train_response = text_audio_train[text_audio_train.PARTICIPANT=='Participant'] #without Ellie's questions
        y_train_audio =np.array(text_audio_train_response.LABEL)
        X_train_audio = np.array(text_audio_train_response.iloc[:,8:])
        audio_features = np.array(text_audio_train_response.columns[8:])


        # if test:
        #   text_audio_test =      # TODO
        if type == 'train':
            return X_train_audio, y_train_audio, audio_features
        # elif type == 'test' and test: #TODO
        #     return X_test_audio, y_test_audio, audio_features




    def load_train_data_lstm(self, type):
        test = self.kwargs.get('test')
        inputPath = self.kwargs.get('input')
        audio_file = self.kwargs.get('audio_file')
        inputPath = './data/input/' #TODO:
        audio_file = 'text_audio_df.csv'
        text_audio = pd.read_csv(inputPath + audio_file)
        #
        # One row had NaN value, detected:
        # for row_i in range(text_audio.shape[0]):
        #     row = text_audio.iloc[row_i,:]
        #     if row.isna().sum()>1:
        #         break
        # So we replace with previous question or answer
        text_audio.iloc[12927, :]=text_audio.iloc[12927-2, :]

        # Each sample is one participant. Take all responses and pad to max. of 100.

        # Train
        text_audio_train = text_audio[text_audio.FILE_TYPE=='TRAIN']  # TODO load text from here too?
        text_audio_train_response = text_audio_train[text_audio_train.PARTICIPANT=='Participant'] #without Ellie's questions
        # y_train_audio =np.array(text_audio_train_response.LABEL)
        # X_train_audio = np.array(text_audio_train_response.iloc[:,8:])
        # audio_features = np.array(text_audio_train_response.columns[8:])
        # Group by participant
        X_train_text = []
        X_train_audio = []
        y_train = []

        for participant in list(text_audio_train_response.FILE_NAME):

            text_audio_train_response_participant = text_audio_train_response[text_audio_train_response.FILE_NAME==participant]
            X_train_text_participant = np.array(text_audio_train_response_participant.UTTERANCE)
            X_train_audio_participant = np.array(text_audio_train_response_participant.iloc[:, 8:])
            y_train_participant = set(text_audio_train_response_participant.LABEL)
            if 0 in y_train_text_participant:
                y_train_text_participant=0
            else:
                y_train_text_participant=1

            # Append participants data to datasets

            X_train_text.append(X_train_text_participant)
            X_train_audio.append(X_train_audio_participant )
            y_train.append(y_train_participant)

        X_train_audio = pad_sequences(X_train_audio , maxlen=100, padding='post')
        X_train_text = pad_sequences(X_train_text, maxlen=100, padding='post')
        # if test:
        #   text_audio_test =      # TODO
        if type == 'train':
            return X_train_text, X_train_audio, y_train
        # elif type == 'test' and test: #TODO
        #     return X_test_audio, y_test_audio, audio_features










    def loadInputData(self,type):

        inputPath = self.kwargs.get('input')
        trainingFile = self.kwargs.get('trainingFile')


        input_kwargs = {}
        quotes = []
        categories = []
        category_map = defaultdict(int)
        f = codecs.open(inputPath+trainingFile,'r','utf8')

        for line in f:
            elements = line.strip().split('\t')
            dataType = elements[0]
            if dataType.lower() != type:
                continue
            participant = elements[5]
            if participant != 'Participant':
                continue

            category = float(elements[2])
            categories.append(category)
            category_map[category]+=1
            quote = elements[6]
            quotes.append(quote)

        f.close()
        input_kwargs['input'] = quotes
        input_kwargs['labels'] = categories

        return input_kwargs

