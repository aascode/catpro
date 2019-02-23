import nltk


class modalFeatures:
    
    def __init__(self):
        print ('inside modal functon')
        self.modals  =[]
        self.DF = 0
    def loadModalFile(self,path):
        file = 'modals_all_1231.txt'
        self.ModalList = self.read_from_file(path+file, filter=self.DF)
    
    def getModals(self):
        return self.ModalList

    def read_from_file(self,filename,filter):
        f = open(filename)
        type = 'MODAL'
        x_grams = {}
        for line in f:
            ngram,count = line.split('\t')
            ngram = ngram.strip()
            count = count.strip()
            if filter is not None:
                if int (count) >= int(filter):
                    x_grams[type + '|||' + ngram] = count
            else:
                x_grams[type + '|||' + ngram] = count
        f.close()
        return x_grams
 
    def generateModalFeatures(self,clause):
        
        modalFeatures = {}
        postags = nltk.pos_tag(nltk.word_tokenize(clause.lower()))
        
        for word,tag in postags:
            typeWord = 'MODAL' + '|||' + word
            if 'MD' in tag and typeWord in self.ModalList:
                old = modalFeatures.get(typeWord)
                if old is None:
                    old = 0
                modalFeatures[typeWord] = old+1
                
        return modalFeatures

