
# TODO: get file for 208 and uncomment 208, 175, 
# import math

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from features.generate_agree_features import *
from features.generate_hedge_features import *
from features.generate_indicator_features import *
from features.generate_interject_features import *
from features.generate_lexical_features import *
from features.generate_liwc_features import *
from features.generate_modal_features import *
from features.generate_mpqa_sentiment_features import *
from features.generate_tagq_features import *
import numpy as np
from utils import *


#from generate_emot_features import *
#from generate_punct_features import *
sentObject = SentimentIntensityAnalyzer()



#from decorator import init

class FeatureGenerator:
    def  __init__(self,featureTypes,kwargs):
        
#        self.mainPath = './data/'
        self.input_path = kwargs.get('util') #TODO: redundant with first lines of depression_bl_handler
        self.liwc_path = kwargs.get('liwc')
        self.unigram_file = 'depression_unigrams.txt'
        self.bigram_file = 'depression_bigrams.txt'
        self.trigram_file = 'depression_trigrams.txt'
        self.modal_file = 'modals_all_1231.txt'
        self.wordPair_file = 'argument_wp.txt'
        self.firstthree_file = 'argument_firstthree.txt'
        self.firstlast_file = 'argument_firstlast.txt'
        self.discourseFile = 'pdtb2_unique_lc_markers_notempo_0801.txt'

        self.feature_types = featureTypes
        # this runs a method that
        self.initObjects()

    def initObjects(self):
        
        if 'TAGQ' in self.feature_types:
            self.tagFeat = TagQFeatures()
        
        #if 'PUNCT' in self.feature_types:
         #   self.punctFeat = PunctFeatures()
         #   self.initPunctFeatures()
        
        if 'INTERJ' in self.feature_types:
            self.interjFeat = InterjectFeatures()
        
        if 'NGRAM' in self.feature_types or  'FIRSTLAST' in self.feature_types:
            self.lexFeat = LexicalFeatures()
        
        if 'NGRAM' in self.feature_types:
            self.initNGramFiles()
        if 'FIRSTLAST' in self.feature_types:
            self.initLexFiles()
        if 'MODAL' in self.feature_types:
            self.mdFeat = modalFeatures()
        if 'AGREE_LEX' in self.feature_types:
            self.agreeFeat = agreementFeatures( self.input_path)
        
        if 'DISCOURSE' in self.feature_types:
            self.discourseFeat = IndicatorFeatures()
        
        if 'HEDGE' in self.feature_types:
            self.hedgeFeat = hedgeFeatures()
            
        #if 'EMOT' in self.feature_types:
         #   self.emotFeat = EmotFeatures()
            
        if 'SENTI' in self.feature_types or 'HYPER' in self.feature_types:
            self.mpqaFeat = mpqaFeatures()
        
        if 'LIWC' in self.feature_types:
            self.liwcFeat = LIWCFeatures()
            
        
    def initFeatures(self):
        
        self.features_list = []
        
        if 'TAGQ' in self.feature_types:
            self.features_list.extend(self.tagFeat.getTagFeatures())
        
        if 'INTERJ' in self.feature_types:
            self.interjFeat.loadInterjections(self.input_path)
            self.features_list.extend(self.interjFeat.getInterjFeatures())
        
        if 'EMBED' in self.feature_types:
            
            vector_length = 100
            embedding_features = np.zeros(vector_length)
            embedding_features = ['embed|||'+str(posn) for posn in range(len(embedding_features)) ]
            self.features_list.extend(embedding_features)
        
        if 'EMOT' in self.feature_types:
            self.emotFeat.loadEmoticons(self.input_path)
            self.emotFeat.convertEmoticons(self.input_path)
            self.features_list.extend(self.emotFeat.getAllEmoticonFeatures())    
            self.features_list.append('emot_oppos')
        
        if 'PUNCT' in self.feature_types:
            self.features_list.append('LENGTH')
            self.features_list.append("!_MUL") 
            self.features_list.append("!_SINGLE") 
            self.features_list.append("?_SINGLE") 
            self.features_list.append("?_MUL") 
            self.features_list.append("!?_MIX") 
            
        if 'PERSON' in self.feature_types:
            self.first_persons = ['I_FIRST','ME_FIRST', 'MINE_FIRST','MYSELF_FIRST', 'MY_FIRST']
            self.second_person = ['YOU_SECOND', 'YOUR_SECOND', 'YOURS_SECOND','YOURSELF_SECOND']
            self.plural_person = ['WE_PLURAL', 'US_PLURAL']
            
            self.features_list.extend(self.first_persons)
            self.features_list.extend(self.second_person)
            self.features_list.extend(self.plural_person)
            
        
        if 'LIWC' in self.feature_types:
            self.initLIWC()
            self.features_list.extend(self.liwcFeat.getLIWC())
        
        if 'JACCARD' in self.feature_types:
            self.features_list.append('COMMON_SIM')
            self.features_list.append('JACCARD_SIM')

        
        if 'VADER' in self.feature_types:
            self.features_list.append('SOURCE_VADER|||pos')
            self.features_list.append('SOURCE_VADER|||neg')
            self.features_list.append('SOURCE_VADER|||neu')
            self.features_list.append('SOURCE_VADER|||compound')

            self.features_list.append('TARGET_VADER|||pos')
            self.features_list.append('TARGET_VADER|||neg')
            self.features_list.append('TARGET_VADER|||neu')
            self.features_list.append('TARGET_VADER|||compound')
        
        if 'MODAL' in self.feature_types:
            self.initModals()
            self.features_list.extend(self.mdFeat.getModals()) 

        if 'NGRAM' in self.feature_types:
            #self.features_list.extend(self.lexFeat.getWordPairFeats())
            self.features_list.extend(self.lexFeat.getUnigramFeats())
            self.features_list.extend(self.lexFeat.getBigramFeats())
            self.features_list.extend(self.lexFeat.getTrigramFeats())
            
        if 'FIRSTLAST' in self.feature_types:
            #self.features_list.extend(self.lexFeat.getWordPairFeats())
            self.features_list.extend(self.lexFeat.getFirstThirdFeats())
            self.features_list.extend(self.lexFeat.getFirstLastFeats())

        if 'SENTI' in self.feature_types:
            self.initMPQAFeatures()
            self.features_list.extend(self.mpqaFeat.getSentimentTerms())
            
        if 'HYPER' in self.feature_types:
            self.initMPQAFeatures()
            self.features_list.extend(self.mpqaFeat.getSentimentTypes())
        
        if 'HEDGE' in self.feature_types:
            self.initHedgeFeatures()
            self.features_list.extend(self.hedgeFeat.getHedges())

        if 'DISCOURSE' in self.feature_types:
            self.initDiscourseFeatures()
            self.features_list.extend(self.discourseFeat.getSourceDiscourseFeats())
            self.features_list.extend(self.discourseFeat.getTargetDiscourseFeats())

        if 'AGREE_LEX' in self.feature_types:
            self.initAgreeDisagreeLexfiles()
                    
            featureNames = ["SOURCE_DISAGREE","TARGET_DISAGREE","SOURCE_AGREE","TARGET_AGREE",
                        "SOURCE_NEGATION", "TARGET_NEGATION"]
            self.features_list.extend(featureNames)
   #         self.features_list.extend(self.agreeFeat.getAlllAgreeDisagreeFeats())
            
        if 'SARC' in self.feature_types: #sarc targets 
            self.features_list.append('sarc')
            self.features_list.append('notsarc')

        return self.features_list

    def initLIWC(self):
        
  #      liwcPath = '/Users/dg513/work/eclipse-workspace/bbn-deft-workspace-2/deft/columbia/relation/target/test-classes/edu/columbia/relation/sarcasm/en-US/dictionary/'
        self.liwcFeat.loadLIWCDictionaries(self.liwc_path)

    def initMPQAFeatures(self):
        self.mpqaFeat.loadMPQA(self.input_path)

    def initModals(self):
        self.mdFeat.loadModalFile(self.input_path)
    
    def initHedgeFeatures(self):
        self.hedgeFeat.loadHedgeFile(self.input_path)

    def initDiscourseFeatures(self):
        self.discourseFeat.setDiscourseMarkers(self.input_path+self.discourseFile)
 
    
    def initAgreeDisagreeLexfiles(self):
        
        self.agreeFeat.loadAgreeFile()
        self.agreeFeat.loadDisagreeFile()
        self.agreeFeat.loadNegationFile()
        
    def initNGramFiles(self):
        
        unigramFile =  self.input_path + self.unigram_file
        bigramFile = self.input_path + self.bigram_file
        trigramFile = self.input_path + self.trigram_file
     #   verbs = self.mainPath + self.input_path + self.verb_file
     #   adverbs = self.mainPath +self.input_path + self.adverb_file
    #    modals = self.mainPath + self.input_path + self.modal_file
        wordPairFile = self.input_path + self.wordPair_file
        firstThreeFile =   self.input_path  + self.firstthree_file
        firstLastFile = self.input_path  +   self.firstlast_file
        fileNames = {}
        fileNames['unigram'] = unigramFile
        fileNames['bigram'] = bigramFile
        fileNames['trigram'] = trigramFile
  #      fileNames['modals'] = modals
        
        self.lexFeat.setFileNames(fileNames)
    
    def initLexFiles(self):
        
    #    wordPairFile = self.mainPath + self.input_path + self.wordPair_file
        firstThreeFile =  self.input_path  + self.firstthree_file
        firstLastFile = self.input_path  +   self.firstlast_file
    #    self.lexFeat.setArgRelationWordPairFile(wordPairFile)
        self.lexFeat.setFirstThreeFile(firstThreeFile)
        self.lexFeat.setFirstLast(firstLastFile)
    
    def populate_features(self):
        
        self.features_list  =[]
        '''   
        self.features_list = [ "S_TOKENS", "T_TOKENS", "TOKEN_DIFFERENCE", "S_PUNCS","T_PUNCS" ,
                              "PUNC_DIFFERENCE", "S_POSITION", "T_POSITION","S_POSN_INTRO", "S_POSN_CONCL", 
                                "T_BEFORE_S", "SENT_DIST", "SAME_SENT","COMMON_PRODS" ] #Structural Features
        
        featureNames = ["SUB-CLAUSES","DEPTH","PRESENT_TENSE"]
        self.features_list.extend(featureNames)
     
        '''
        
        
        featureNames = ["COMMON_TOKENS"]
        self.features_list.extend(featureNames)
        
        #featureNames = ["S_ARGTYPE","T_ARGTYPE"]
        #self.features_list.extend(featureNames)

        
         #update with all feature names!!!
        self.getAllNgramFeatureNames()
        
        return self.features_list

  #  def getAllNgramFeatureNames(self):
        
   #     nonNGramFeatSize = len(self.features_list)
   
    def generatePunctFeatures(self,target_arg):
        
        features = {}
        punct_features = self.punctFeat.generatePunctFeatures(target_arg)
        features.update(punct_features)
        return features
    
    def generateEmotFeatures(self,source_arg,target_arg,post_type):
        

        features = {}
        if post_type == 'BOTH':
            emot_source = self.emotFeat.generateEmotFeatures(source_arg)
        
        emot_target = self.emotFeat.generateEmotFeatures(target_arg)
        features.update(emot_target)
        
        if 'emot|||senti|||'+'positive' in emot_source.keys() and \
             'emot|||senti|||'+'negative' in  emot_target.keys(): \
            features.update('emot_oppos')
        
        elif 'emot|||senti|||'+'negative' in emot_source.keys() and \
             'emot|||senti|||'+'positive' in  emot_target.keys(): \
            features.update('emot_oppos')
        
        return features

    def generateInterjFeatures(self,source_arg,target_arg,post_type):
        
        features ={}
        target_inter = self.interjFeat.generateInterjFeatures(target_arg)
        features.update(target_inter)
        return features
   
    def generateModalFeatures(self,target_arg,post_type):
        
        features = {}

        target_modals = self.mdFeat.generateModalFeatures(target_arg)
        features.update(target_modals)
        
        return features 

    def generateTagqFeatures(self,source_arg,target_arg,post_type):
        
        features ={}

        
        tagqs = self.tagFeat.generateTagFeatures(target_arg)
        features.update(tagqs)
        return features
    
    def generateAgreeDisagreeFeatures(self,source_arg,target_arg,post_type):
        
        features = {}

        if post_type == 'BOTH':
            source_arg_features = self.agreeFeat.generateAgreeDisagreeFeatures(source_arg,'SOURCE')
            self.features.update(source_arg_features)

        target_arg_features = self.agreeFeat.generateAgreeDisagreeFeatures(target_arg, 'TARGET')
        features.update(target_arg_features)
        
        return features
    
    def generateEmbeddingFeatures(self,argument,vectors,unknown_vec,vec_length):
        
        featureMap = {}
    
        word_list = argument.lower().split()
        filtered_words = [word for word in word_list if word not in nltk.corpus.stopwords.words('english') and word.isalpha()]
        embedding_sum = np.zeros(vec_length)
        num = 0
        for filtered_word in filtered_words:
            num+=1
            glove_embedding = vectors.get(filtered_word.lower())
            if glove_embedding is None:
                    #print 'embedding is missing?'
                glove_embedding  = unknown_vec
            embedding_sum = np.sum([embedding_sum, glove_embedding], axis=0)  #TODO: try averaging.
            # vector sum (similar to Mitchell / Lapata but with predicted vectors)
        
            embedding_sum = np.divide(embedding_sum,num)

        for index,val in enumerate(embedding_sum):
            featureMap['embed|||'+str(index)] =val
        
        
        #featureMap['embedding'] = np.array(embedding_sum)
       # featureMap['embedding'] = (embedding_sum)
    
        return featureMap


    def generateDiscourseFeatures(self,target_arg,post_type):
        
        features = {}
        discourseFeatures2 = self.discourseFeat.get_implicit_type_discourse_marker(target_arg,'TARGET')
        features.update(discourseFeatures2)
        
        return features
    '''
    def getPunctuations(self,target_arg):
        
        featureMap = defaultdict(int)
        
        words = nltk.word_tokenize(target_arg.lower(), language='english')
        for word in words:
            if word.strip() == '!' or word.strip() == '?' or word.strip():
                featureMap['PUNCT|||'+word]+=1
        
        featureMap['LENGTH'] = len(words)*1.

        return featureMap
    '''
    
    def getPersonFeatures(self,target_arg,post_type):
    
        featureMap = {}
    
        words1 = nltk.word_tokenize(target_arg.lower(), language='english')
        count = 0.0
        
        for word in words1:
            if (word+'_first').upper() in self.first_persons:
                featureMap[(word+'_first').upper()] = 1.0
            if (word+'_second').upper() in self.second_person:
                featureMap[(word+'_second').upper()] = 1.0
            if (word+'_plural').upper() in self.plural_person:
                featureMap[(word+'_plural').upper()] = 1.0
            
        return featureMap
    
    def getSentiVaderFeatures(self,target_arg,post_type):
    
        features = {}
        ss_af = sentObject.polarity_scores(target_arg)
        for ss in ss_af.keys():
                features['TARGET_VADER'+ '|||'+ss] =  ss_af.get(ss)
        
        return features
    
    def generateMPQAFeatures(self,target_arg,post_type):
        features = {}
               
        mpqa_target = self.mpqaFeat.getSentiMPQAFeatures(target_arg)
        features.update(mpqa_target)
        return features

    def generateHyperFeatures(self,source_arg,target_arg,post_type):
        features = {}
        mpqa_source = self.mpqaFeat.getHyperMPQAFeatures(source_arg)
            
        mpqa_target = self.mpqaFeat.getHyperMPQAFeatures(target_arg)
        features.update(mpqa_target)
        
        if 'pos' in mpqa_source.keys() and 'neg' in mpqa_target.keys():
            features['senti_diff'] =1.0
        if 'neg' in mpqa_source.keys() and 'pos' in mpqa_target.keys():
            features['senti_diff'] =1.0

        return features


    def generateHedgeFeatures(self,target_arg,post_type):
        features = {}
        
        hedge_target = self.hedgeFeat.getHedgeFeatures(target_arg)
        features.update(hedge_target)
        return features

    def generateSarcFeature(self,sarc):
        
        features = {}
        self.features[sarc] = 1.0
    
    def generateLexFeatures(self,source_arg,target_arg,post_type):
        
        features = {}

        if post_type == 'BOTH':
            firstThirdWords_source = self.lexFeat.createFirstThirdWords(source_arg,'SOURCE')
            features.update(firstThirdWords_source)
            
            firstLastWords_source = self.lexFeat.createImplicitFirstLastWords(source_arg,'SOURCE')
            features.update(firstLastWords_source)
        
   #     print 'firstthird features are ready'
   
        #features from the first sentence
        target_arg_sentences = nltk.sent_tokenize(target_arg)
        
        firstThirdWords_target  = self.lexFeat.createFirstThirdWords(target_arg_sentences[0],'TARGET')
        features.update(firstThirdWords_target)
            
        firstLastWords_target = self.lexFeat.createImplicitFirstLastWords(target_arg_sentences[0],'TARGET')
        features.update(firstLastWords_target)
        
        return features

    def generateNGramFeatures(self,target_arg,post_type):
        
        features = {}
        
        target_arg_sentences = target_arg

        unigram_target = self.lexFeat.createUnigrams(target_arg_sentences)
        features.update(unigram_target)
        bigram_target = self.lexFeat.createBigrams(target_arg_sentences)
        features.update(bigram_target)
        trigram_target = self.lexFeat.createTrigrams(target_arg_sentences)
        features.update(trigram_target)
        
       # print(features)
        return features

        
    #    wordPairs = self.lexFeat.createWordPairs(source_arg,target_arg)
    #    self.features.update(wordPairs)
        
    #    print 'firstlast features are ready'
    
    def generateJaccardFeatures(self,source_arg,target_arg):
        
        features = {}
        words1 = nltk.word_tokenize(source_arg.lower(), language='english')
        words1 = set([word for word in words1 if word not in nltk.corpus.stopwords.words('english') and word.isalpha()])

        words2 = nltk.word_tokenize(target_arg.lower(), language='english')
        words2 = set([word for word in words2 if word not in nltk.corpus.stopwords.words('english') and word.isalpha()])

        max_len = max(len(words1),len(words2))


        common_words = set(words1).intersection(words2)
    
        if len(common_words) == 0 :
            return features

        features['COMMON_SIM'] = len(common_words)*1.
        features['JACCARD_SIM'] = float(len(common_words))/float(max_len)
        
        return features

    
    def generateLIWCFeatures(self,target_arg,post_type):
    
        features = {}
        liwc_target = self.liwcFeat.getLIWCFeatures(target_arg)
        features.update(liwc_target)
        
        return features
    
    def getAllFeatures(self,source_arg,target_arg,post_type):
        
        return self.features
    
    '''
    def generateNGramFeatures(self,data,lexFeat):
        
         first generate unigram/bigram/trigrams
        quote,response = data[0],data[1]
        
        self.lexFeat.
        
        
        quote_ngrams = get_ngrams(quote)
        response_ngrams =  get_ngrams(response)
        
        quotes_unigram = nltk.word_tokenize(quote)
        quotes_unigram_features = getFeatureList(quotes_unigram,'unigram')
        quotes_pos = nltk.pos_tag(quote)
        quotes_bigram = list(nltk.bigram(quotes_unigram))
        quotes_trigram = list(nltk.trigrams(quotes_unigram))
        
        response_unigram = nltk.word_tokenize(response)
        response_pos = nltk.pos_tag(response)
        response_bigram = list(nltk.bigram(response_unigram))
        response_trigram = list(nltk.trigrams(response_unigram))
        
    '''


    
    
    