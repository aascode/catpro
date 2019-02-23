import codecs
from collections import defaultdict
import os

import nltk


class LIWCFeatures:
    
    
    def __init__(self):
        print ('in liwc')
        self.liwc = {}
        self.cache = {}
        self.uniques = []
        
    def loadLIWCDictionaries(self,path):
        files = os.listdir(path)
        uniq = []
        for file in files:
            #print file
            f = codecs.open(path+file,'r','latin-1')
            content = ['liwc|||'+line.strip() for line in f.readlines()]
            self.liwc[file] = content
            self.uniques.append(file)
        
        ''' LIWC similarity between two vectors'''
        self.uniques.append('LIWC_SIMILARITY')
            
    def getLIWCFeatures(self,text):
    
        featureMap = defaultdict(int)
        words = nltk.word_tokenize(text)
        for index1,word in enumerate(words):
            word = word.lower()
            dicts = self.cache.get('liwc|||'+word)
            if dicts is not None:
                for key in dicts:
                    featureMap[key]+=1
                    
            else:
                for file in self.liwc.keys():
                    words = self.liwc.get(file)
                    if 'liwc|||'+word in words:
                        featureMap[file]+=1
                        presents = self.cache.get('liwc|||'+word)
                        if presents is None:
                            presents = []
                        presents.append(file)
                        self.cache['liwc|||'+word] = presents
    
        return featureMap

    def getLIWC(self):
        
        return self.uniques
