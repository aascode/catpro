import codecs
from collections import defaultdict
import re

import nltk


class InterjectFeatures:
    
    def __init__(self):
        print ('in interject q')

    def loadInterjections(self,path):
        file = 'interjections.txt'
        f = codecs.open(path+file,'r','utf8')
        self.interj = ['interj|||'+line.strip() for line in f.readlines()]
        f.close()
    
    def getInterjFeatures(self):
        
        return self.interj
        
    def generateInterjFeatures(self,argument):
        
        features  = defaultdict(int)
        words = nltk.word_tokenize(argument.lower())
        for word in words:
            if 'interj|||' + word in self.interj:
                features['interj|||' + word]+=1
        
        return features
            
