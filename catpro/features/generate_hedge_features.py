import string

import nltk


class hedgeFeatures:
    
    def __init__(self):
        print ('inside hedge functon')
        self.hedges  =[]
        
    def loadHedgeFile(self,path):
    
        file  = 'hedges.txt'
        self.hedges  = ['hedge'+'|||'+line.strip() for line in open(path+file)]
        return self.hedges

    def getHedges(self):

        return self.hedges

    def getHedgeFeatures(self,argument):
    
        featureMap = {}
        words1 = nltk.word_tokenize(argument.lower(), language='english')
        count = 0.0
        for word in words1:
            if 'hedge|||'+word.strip() in self.hedges:
                featureMap[ 'hedge|||'+word.strip()] =1.0
        
        ''' not for the single words!!!'''
    #    argument = argument.encode('utf8')
        translator = str.maketrans('', '', string.punctuation)
        for hedge in self.hedges:
            ds  = hedge.split()
            if len(ds)>1:
                hedge = hedge.replace('hedge|||','')
                hedge = hedge.replace(', ',' ').strip()
                text_nopunct = argument.translate(translator)
                if hedge.strip() in text_nopunct:
                    featureMap[ 'hedge|||'+hedge.strip()] =1.0
     
        
        
        return featureMap

