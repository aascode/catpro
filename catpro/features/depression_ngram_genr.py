from nltk import bigrams,trigrams
import string
import nltk
from collections import Counter
 
def listVerbs(postags, dict_x):
     
    counts = Counter(word if tag.startswith('V') else None for word,tag in postags )
    for word in counts:
        if word is not None:
            old = dict_x.get(word)
            if old is None:
                old = 0
            dict_x[word] = old +counts[word]
 
def listAdverbs(postags, dict_x):
     
    counts = Counter(word if 'RB' in tag else None for word,tag in postags )
    for word in counts:
        if word is not None:
            old = dict_x.get(word)
            if old is None:
                old = 0
            dict_x[word] = old +counts[word]
 
def listModals(postags, dict_x):
     
    counts = Counter(word if 'MD' in tag else None for word,tag in postags )
    for word in counts:
        if word is not None:
            old = dict_x.get(word)
            if old is None:
                old = 0
            dict_x[word] = old +counts[word]
 
 
def counts(list_x,dict_x):
    for word in list_x:
        old = dict_x.get(word)
        if old is None:
            old = 0
         
        dict_x[word] = old +1
    return dict_x
 
def write_to_file(file_name,dict_x):
    for word in dict_x:
        file_name.write(str(word)+"\t"+str(dict_x[word])+"\n")
 
 
 
unigram = open("./data/util/unigrams_depression.txt","w")
bigram = open("./data/util/bigrams_depression.txt","w")
trigram = open("./data/util/trigrams_depression.txt","w")
#verbFile = open("./data/util/verbs_all_1231.txt","w")
#adverbFile = open("./data/util/adverbs_all_1231.txt","w")
modalFile = open("./data/util/modals_depression.txt","w")
 
 
 
#file_type = "train"
 
input_tree = open('./data/input/depression_all_data.txt','r')
 
 
input_tree.readline()
clauses = []
for line in input_tree:
    temp = line.split("\t")
    clauses.append(temp[6])
 
uni = {}
bi = {}
tri = {}
verbs = {}
adverbs = {}
modals = {}
 
for clause in clauses:
  #  clause = clause.translate(string.maketrans("",""), string.punctuation)
    temp = nltk.word_tokenize(clause.lower())
    #temp = clause.lower().split()
    counts(temp,uni)
    counts(bigrams(temp),bi)
    counts(trigrams(temp),tri)
     
    posTags = nltk.pos_tag(temp)
    listModals(posTags,modals)
     
 
write_to_file(unigram,uni)
write_to_file(bigram,bi)
write_to_file(trigram,tri)
write_to_file(modalFile,modals)
 
 
unigram.close()
bigram.close()
trigram.close()
modalFile.close()