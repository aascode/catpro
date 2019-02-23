
import nltk


class agreementFeatures:
    
    def __init__(self,path):
        
        self.agrees = set()
        self.disagrees = set()
        self.negations = set()
        self.agreePath = path
        self.window = 5
	
    def loadAgreeFile(self):
        f = open(self.agreePath + 'agree_seeds.txt')
        self.agrees  = [ agree.strip() for agree in f.readlines()]
        f.close()
		
    def loadDisagreeFile(self):
        f = open(self.agreePath + 'disagree_seeds.txt')
        self.disagrees  = [ disagree.strip() for disagree in f.readlines()]
        f.close()

    def loadNegationFile(self):
        f = open(self.agreePath + 'negations.txt')
        self.negations  = [ negation.strip() for negation in f.readlines()]
        f.close()

    def getAlllAgreeDisagreeFeats(self):
        allFeats = []
        allFeats.extend(self.agrees)
        allFeats.extend(self.disagrees)
        allFeats.extend(self.negations)
        return allFeats
    
    def generateAgreeDisagreeFeatures(self,argument,type):
        
        featMap = {}
        words = nltk.word_tokenize(argument.lower())
        agree_posn = -1
        agree_posns = []
        for index, word in enumerate(words):
            if word in self.agrees:
                agree_posn = index
                agree_posns.append(index)
        
        if len(agree_posns) > 0: 
            for agree_posn in agree_posns:
                agree_words = words[max(0,agree_posn-self.window):agree_posn]
                disagree_posn = -1
                for index, word in enumerate(agree_words):
                    if word in self.negations:
                        featMap[type+'_DISAGREE'] = 1.0
                        disagree_posn = index
                if disagree_posn == -1 :
                    featMap[type+'_AGREE'] = 1.0
        
        common = list(set(words).intersection(self.disagrees))
        if len(common) >0:
            featMap[type+'_DISAGREE'] = 1.0
        
        common = list(set(words).intersection(self.negations))
        if len(common) >0:
            featMap[type+'_NEGATION'] = 1.0
		
        return featMap
