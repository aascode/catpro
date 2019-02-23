# TODO: merge with data_helpers.py

import math

def convertCategory(score):
    
    if score <=-2:
        return -1.0
    elif score >1:
        return 1.0
    else:
        return 0.0

def cosine(dic1,dic2):
    
    numerator = 0.0
    dena = 0.0
    for key1 in dic1.keys():
        val1 = dic1.get(key1)
        numerator += val1*dic2.get(key1,0.0)
        dena += val1*val1
    denb = 0.0
    for val2 in dic2.values():
        denb += val2*val2
    if dena == 0.0 or denb == 0.0:
        return 0.0
        
    return numerator/math.sqrt(dena*denb)
