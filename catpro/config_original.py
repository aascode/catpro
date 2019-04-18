
# TODO: change this to simple variable or use .ini

'''
'toy'=False makes the model run only linear and Cs=[1,10]. 
'''
config = {
    'regression':False,
    'perform_cross_validation': False,
    'toy':True,
    'cluster': False,
    'test': True,
    'run_text':False,
    'run_audio':True,
    'create_features':False,
    'group_by': 'response',
    'gpt2': '/Users/danielmlow/Dropbox/gpt-2/models/117M/',
    'liwc':'/Users/danielmlow/Dropbox/data/liwc_english_dictionary/',
    'input':'./data/datasets/',
    'util':'./data/util/',
    'output_dir':'./data/outputs/',
    'trainingFile':'depression_all_data.txt',
    'inputPath':'/Users/danielmlow/Dropbox/depression/data/input/',
    'audio_file':'text_audio_df.csv',
    'features':'NGRAM,HEDGE,LIWC,VADER,SENTI,MODAL,EMBED,PERSON,DISCOURSE',
    'feature_numbers':1000,
    'train_test_split': 0.2


}
