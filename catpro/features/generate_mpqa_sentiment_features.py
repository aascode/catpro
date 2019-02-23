import nltk


class mpqaFeatures:
    
    def __init__(self):
    
        self.positives = []
        self.negatives = []
        self.hypers = []
        
    
    def loadMPQA(self,path):
    
        file = open(path+'subjclueslen1-HLTEMNLP05.tff')
        for line in file:
            elements = line.strip().split()
            type = elements[0].strip().split('=')[1]
       #     if type.startswith('weak'):
        #        continue
            word = elements[2].strip().split('=')[1]
            
            if len(elements) == 6:
                sentiment = elements[5].strip().split('=')[1]
            elif len(elements) == 7:
                sentiment = elements[6].strip().split('=')[1]
                
            if sentiment.startswith('pos'):
                self.positives.append('pos|||'+word.lower())
            if sentiment.startswith('neg'):
                self.negatives.append('neg|||'+word.lower())
            
            if type =="strongsubj":
                self.hypers.append('hyper|||'+sentiment+'|||'+word.lower())
        
        file.close()
        
        self.positives = list(set(self.positives))
        self.negatives = list(set(self.negatives))
        self.hypers = list(set(self.hypers))

    def getSentiMPQAFeatures(self,text):
    
        featureMap = {}
        words = nltk.word_tokenize(text)
        for index1,word in enumerate(words):
            if 'pos|||'+word.lower().strip() in self.positives:
                    featureMap['pos|||'+word.lower().strip()] = 1.0
            elif 'neg|||'+word.lower().strip() in self.negatives:
                    featureMap['neg|||'+word.lower().strip()] = 1.0
    
        return featureMap
    
    def getHyperMPQAFeatures(self,text):
    
        featureMap = {}
        words = nltk.word_tokenize(text)
        for index1,word in enumerate(words):
            if 'pos|||'+word.lower().strip() in self.positives:
                    featureMap['pos'] = 1.0
            if 'neg|||'+word.lower().strip() in self.negatives:
                    featureMap['neg'] = 1.0
            
            if 'hyper|||positive|||'+word.lower().strip() in self.hypers:
                featureMap['hyper|||positive|||'+word.lower()] = 1.0
            if 'hyper|||negative|||'+word.lower().strip() in self.hypers:
                featureMap['hyper|||negative|||'+word.lower()] = 1.0

    
        return featureMap
    

    def getSentimentTerms(self):
        
        sentiments = []
        sentiments.extend(self.positives)
        sentiments.extend(self.negatives)
        
        return sentiments

    def getSentimentTypes(self):
        
        sentiments = []
        sentiments.append('pos')
        sentiments.append('neg')
        sentiments.append('senti_diff')
        
        
        sentiments.extend(self.hypers)
        return sentiments
