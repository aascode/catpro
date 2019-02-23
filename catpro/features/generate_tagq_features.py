import re

import nltk


class TagQFeatures:
    
    def __init__(self):
        print ('in tag q')
    
    def getTagFeatures(self):
        
        tags  = []
        tags.append('NEGATIVE_TAG_PRESENT')
        tags.append('POSITIVE_TAG_PRESENT')
        tags.append('TAG_PRESENT')
        return tags

    
    def generateTagFeatures(self,argument):
 #       argument = 'monitoring the decision process ? does he approve of the sins that are commited'
 
        features  = {}
        tag_regex ="((is(n'?t)?\s+(he|she|it))|" \
                + "(does(n'?t)?\s+(he|she|it))|" \
                + "(do(n'?t)?\s+(they|it|we|you|ya|i))|" \
                + "(are(n'?t)?\s+(it|they|we|you|ya|i))|" \
                + "(was(n'?t)?\s+(he|she|it|i))|" \
                + "(did(n'?t)?\s+(they|he|she|it|you|ya|i))|" \
                + "(were(n'?t)?\s+(they|we|you|ya|i))|" \
                + "(have(n'?t)\s+(they|we|you|ya|i))|" \
                + "(has(n'?t)?\s+(he|she|it))|" \
                + "(had(n'?t)?\s+(they|we|he|she|it|you|ya|i))|" \
                + "(won('?t)?\s+(they|we|he|she|it|you|ya|i))|" \
                + "(can('?t)?\s+(they|we|he|she|it|you|ya|i))|" \
                + "(must(n'?t)?\s+(they|we|he|she|it|you|ya|i))|" \
                + "(will\s+(they|we|i|he|she|it|you|ya|i))|" \
                + "(can\s+(they|it|i|he|she|it|you|ya)))" 
        
#        tag_regex ="(does(n'?t)?\s+(he|she|it))"
#        tag_regex = "(does he)"
        
#        p = re.compile(tag_regex,re.UNICODE)
#        m = p.match(argument)
#        m = p.match(argument.encode('utf-8'))
        
        match = re.search(tag_regex, argument.encode('utf-8'))
        if match:
            features['TAG_PRESENT'] = 1.0
            if ('n\'t' in match.group(0) ) or ( 'nt' in  match.group(0) ) :
                features['NEGATIVE_TAG_PRESENT'] = 1.0
            else:
            
                features['POSITIVE_TAG_PRESENT'] = 1.0
            
        return features