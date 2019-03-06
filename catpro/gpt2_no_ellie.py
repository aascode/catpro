#!/usr/bin/env python3

import os
import datetime
import time
# import sys
# sys.path.insert(0, './../../keras-gpt-2/')

import pandas as pd
import numpy as np
from keras_gpt_2 import load_trained_model_from_checkpoint, get_bpe_from_files, generate
# from sklearn import metrics

import config

amount_removed = .5
feature = '_no_ellie_'
length =30



config_params = config.config

inputPath = config.config['input']
output_dir = config.config['output_dir']
model_folder = config_params['gpt2']
config_path = os.path.join(model_folder, 'hparams.json')
checkpoint_path = os.path.join(model_folder, 'model.ckpt')
encoder_path = os.path.join(model_folder, 'encoder.json')
vocab_path = os.path.join(model_folder, 'vocab.bpe')




## Example
# print('Generate text...')
# output = generate(model, bpe, ['Am I depressed?'], length=10, top_k=1) #grows with length
# If you are using the 117M model and top_k equals to 1, then the result will be:
# "From the day forth, my arm was broken, and I was in a state of pain. I was in a state of pain,"
# print(output[0])






# import data_helpers #itertools created a problem with docker.




def make_output_dir(output_dir):
    # TODO
    '''

    :param output_dir: 
    :return: 
    '''
    directory_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")  # random number so its not replaced
    path_to_dir = os.path.join(output_dir, directory_name)
    try:
        os.mkdir(os.path.join(output_dir, directory_name))
    except:
        print('file was not created or already exists')
    return path_to_dir



path_to_dir = make_output_dir(os.path.join(config_params['output_dir']+'gpt2/'))



# TODO: add 'Q: ' and 'A: ' before questions and answers like in paper.
# TODO: I prob want to change 'i' to 'I'.

def load_data(dataset='train', timesteps=33, group_by='interview', participant_only=False):
    '''
    :param type: 
    :param timesteps: 
    :param group_by: {'response', 'interview'} each sample, where each time step is segment or response, respectively
    :return: 
    '''
    print('loading data...')
    if group_by == 'interview':
        audio_file = 'text_audio_df.csv'
        text_audio = pd.read_csv(inputPath + audio_file)
        # Each sample is one participant.
        # Train
        text_audio_train = text_audio[text_audio.FILE_TYPE == dataset.upper()]  # TODO load text from here too?
        if participant_only:
            text_audio_train = text_audio_train[text_audio_train.PARTICIPANT == 'Participant']  # without Ellie's questions
            X_train_text = []
            y_train = []
            for participant in list(set(text_audio_train.id)):
                text_train_participant = text_audio_train[text_audio_train.id == participant]
                text = list(text_train_participant.UTTERANCE)
                del text[-2:]
                X_train_text.append('. '.join(text))
                y_train.append(list(set(text_train_participant.LABEL))[0])
            return X_train_text, y_train
        X_train_text = []
        y_train = []
        for participant in list(set(text_audio_train.id)):
            text_audio_train_participant = text_audio_train[text_audio_train.id==participant]
            y_train.append(list(set(text_audio_train_participant.LABEL))[0])
            interview = list(text_audio_train_participant.UTTERANCE)
            person_speaking = list(text_audio_train_participant.PARTICIPANT)
            interview_cleaned = []
            for i, turn in enumerate(interview):
                if person_speaking[i]=='Ellie':
                    # it is a question
                    if "(" in turn:
                        questions = turn[turn.find("(")+1:turn.find(")")]
                    else:
                        questions = turn[:]
                    question_cleaned = []
                    for question in questions.split(' . '):
                        question_cleaned.append(question)
                    question_cleaned = '. '.join(question_cleaned)
                    question_cleaned+="?\n" #TODO: not all of Ellie's utterances are questions: yeah. that sounds like a great situation.
                    # TODO: So maybe make a dictionary out of her utterances and manually choose which ones don't go with interrogation sign

                    interview_cleaned.append(question_cleaned)
                else:
                    # response
                    response = turn+'\n\n'
                    interview_cleaned.append(response)
            interview_cleaned = interview_cleaned[:-4] #TODO: this isnt that precise, but approximate.
            if person_speaking [-1]=='Ellie':
                # You don't want to end with ellie's goodbye but add questions
                interview_cleaned[-1]='Are you depressed?'
            else:
                interview_cleaned.append('Are you depressed?')
            X_train_text.append(interview_cleaned)
        # np.savez_compressed(output_dir+'/datasets/')
        return np.array(X_train_text), np.array(y_train)


            # TODO temp, different way of doing it:
        # person_speaking = list(text_audio_train.PARTICIPANT)
        # interview = list(text_audio_train.UTTERANCE)
        # QAs = []
        # for i in range(0, len(interview), 2):
        #     if person_speaking[i] == 'Ellie':
        #         questions = interview[i]
        #         question_cleaned = []
        #         for question in questions.split(' . '):
        #             if "(" in question:
        #                 question = question[question.find("(")+1:question.find(")")]
        #             question_cleaned.append(question)
        #         question_cleaned = '. '.join(question_cleaned)
        #         question_cleaned+="?"
        #     elif person_speaking[i] == 'Participant' and person_speaking[i-1]=='Ellie':
        #             # q==len(interview)-1:
        #             # continue
        #         # else:
        #         answer = '\n'+interview[i+1]+'\n\n'
        #         question_cleaned += answer
        #     elif person_speaking[i] == 'Participant' and person_speaking[i-1]=='Participant':
        #         answer = interview[i] + '\n\n'
        #         question_cleaned = question_cleaned.replace('\n\n', '. ')
        #         question_cleaned+=answer
        #     elif person_speaking[i] == 'Ellie' and person_speaking[i-1]=='Ellie':
        #         questions = interview[i]
        #         question_cleaned = []
        #         for question in questions.split(' . '):
        #             if "(" in question:
        #                 question = question[question.find("(") + 1:question.find(")")]
        #             question_cleaned.append(question)
        #         question_cleaned += '. '.join(question_cleaned) #add to previous question.
        #         question_cleaned += "?"
        #     question_cleaned = question_cleaned.replace(' .', '.')
        #     QAs.append([question_cleaned])
        # QAs_flat = [n for i in QAs for n in i]



        # X_train_audio = np.array(X_train_audio)
        # X_train_text = np.array(X_train_text)
        # # np.save(output_dir + 'datasets/X_train_text.npy', X_train_text)
        # # np.save(output_dir + 'datasets/X_train_audio.npy', X_train_audio)
        # # TODO: fix
        # # for participant_matrix in X_train_audio_padded:
        # #     break
        # #     participant_vector = np.array(pd.DataFrame(participant_matrix).mean())
        # # X_train_audio_padded = pad_sequences(X_train_audio, maxlen=timesteps, padding='post', dtype='float32')
        # # X_train_text_padded = pad_sequences(X_train_text, maxlen=timesteps, padding='post', dtype='str')
        # return X_train_text, X_train_audio, y_train


phq8_bin = ['Do I have little interest or pleasure in doing things?',
            "Am I feeling down, depressed, or hopeless?",
            "Do I have trouble falling or staying asleep, or sleeping too much?",
            "Am I feeling tired or having little energy?",
            "Do I have a poor appetite or am I overeating?",
            "Am I feeling bad about myself or that I am a failure, or have let myself or my family down?",
            "Do I have trouble concentrating on things, such as reading the newspaper or watching television?",
            "Am I moving or speaking so slowly that other people could have noticed. Or the opposite, have I been being so fidgety or restless that I have been moving around a lot more than usual?"
            ]

phq8_simple = ['Do I have little interest in doing things?',
            "Am I feeling depressed?",
            "Do I have trouble falling asleep",
            "Am I feeling tired?",
            "Do I have a poor appetite?",
            "Am I feeling that I am a failure?",
            "Do I have trouble concentrating on things?",
            "Am I moving or speaking slowly?"
            ]

phq8_simpleQ = ['Q: Do I have little interest in doing things?', 'Q: Am I feeling depressed?', 'Q: Do I have trouble falling asleep', 'Q: Am I feeling tired?', 'Q: Do I have a poor appetite?', 'Q: Am I feeling that I am a failure?', 'Q: Do I have trouble concentrating on things?', 'Q: Am I moving or speaking slowly?']




# phq8_simple_multi = ['Do I have little interest or pleasure in doing things? Not at all? Nearly every day?'] #TODO


# y_train_df = pd.read_csv(inputPath + 'train_split_Depression_AVEC2017.csv')

# Use sample and ask question:
# ======================================================
# 0. Put narrative without questions and ask "Am I depressed?"
# 1. score without  participant sample, with question, this indicates randomness for each question, as it does not know anything about the participant.
# 2. score with     participant sample, with question

# try 1) with Q: A:, 2) with \n, 3)without all that, 4) without questions, then adding "Am I depressed?", 5. with PHQ8 questions.

# 3. score with     participant sample, with question, with PHQ8 questions before questions
# 4. score without  participant sample, with question + training on samples and testing on test set
# 5. score with     participant sample, with question + training on samples and testing on test set
# 3. score with     participant sample, with question, with PHQ8 questions before questions
# Compute confidence
# . other models.

# Maybe rank answer with sentiment analysis or if it's in negation dict: "I'm not" = 0, "not really"= 0.2, "yes"=1

# text_audio_train.UTTERANCE





if __name__ == '__main__':
    print('Load model from checkpoint...')
    model = load_trained_model_from_checkpoint(config_path, checkpoint_path)  # take ~ 20sec
    print('Load BPE from files...')
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    # Load data
    X_train, y_train = load_data(dataset='train', timesteps=33, group_by='interview', participant_only=True)

    # output1 = generate(model, bpe, ['Hello. are you depressed?'], length=10, top_k=1)  # grows with length
    '''
    for each string it returns an answer starting with the phrase i give it. length isn't really working like i expect.
    So I need to combine interview into one string.m
    '''
    # print(output1, '=======baseline')
    # Generate
    completions = []
    for participant in range(len(X_train)): #TODO uncomment
        print(participant)
        subset = int(len(X_train[participant])*amount_removed)
        # X_train_participant_subset = X_train[participant][subset:]
        X_train_participant_subset = X_train[participant][subset:]
        # output = generate(model, bpe, X_train[i], length=10, top_k=2) #grows with length
        X_train_participant_subset = X_train_participant_subset.replace(' .', '.')
        X_train_participant_subset+= '. Am I depressed? '
        '''
        limit of 1024. if i remove /n, then i can include more . NOT SURE.
        '''
        # X_train_participant_subset==X_train_participant_subset[-1024:]
        start = time.time()
        try:
            output1 = generate(model, bpe, [X_train_participant_subset], length=length, top_k=1)  # grows with length #TODO: make length longer.
        except:
            output1 = 'size issue'
            print(output1)
        end = time.time()
        time_elapsed = end - start
        print(time_elapsed)
        # print(output1)
        # output2
        # start = time.time()
        # output2 = generate(model, bpe, [X_train_participant_subset], length=length, top_k=2)  # grows with length
        # print(output2)
        # end = time.time()
        # time_elapsed = end - start
        # print(time_elapsed)
        completions.append([output1[0], time_elapsed])
        # completions.append([output1[0], output2[0], time_elapsed ])
    pd.DataFrame(completions).to_csv(output_dir+'gpt2/completions'+feature+str(length)+'_'+str(amount_removed)+'.csv')



# analyze=False
# if analyze:
#     # completions = pd.read_csv(output_dir+'gpt2/completions_.7.csv')
#     completions = pd.read_csv(output_dir+'gpt2/completions_.7_phq8.csv')
#     # completions8['1'].sum()/60
#     # >>> both were 112 minutes
#     completions = [n[-80:] for n in list(completions['0'])]
#     completions_responses = []
#     # manual annotation (biased)
#     with open(output_dir+'gpt2/completions.txt', 'w') as f:
#         f.write('\n==========\n'.join(completions))
#     for i in completions:
#         print('====================================')
#         print(i)
#         # resp = input('response: ')
#         # completions_responses.append(resp)
#     y_pred = []
#     y_train_reduced = list(y_train[:])
#     for i,pred in enumerate(completions_responses):
#         if pred=='0':
#             y_pred.append(0)
#         elif pred=='1':
#             y_pred.append(1)
#         else:
#             y_train_reduced[i] = '-'
#
#     y_train_reduced = [x for x in y_train_reduced if not isinstance(x, str)]
#
    # f1 = metrics.f1_score(y_train_reduced , y_pred)
    # acc = metrics.accuracy_score(y_train_reduced , y_pred)
    # precision = metrics.precision_score(y_train_reduced , y_pred)
    # recall = metrics.recall_score(y_train_reduced , y_pred)
    # print('f1: \t', np.round(f1, 2))
    # print('acc: \t', np.round(acc, 2))
    # print('prec: \t', np.round(precision, 2))
    # print('rec: \t', np.round(recall, 2))






    # # Add Q: and A: to turns
    # participant = 4
    # X_train_1_cleaned = []
    # for i in range(0,len(X_train[participant ]),2):
    #     question = 'Q: '+X_train[participant ][i].replace('\n', '')
    #     try: answer= 'A: ' + X_train[participant ][i+1].replace('\n', '')
    #     except: pass
    #     X_train_1_cleaned.append(question)
    #     X_train_1_cleaned.append(answer)
    #
    # a = []
    # a.append("Q: Are you depressed?")
    # ' '.join(a)

    # Find final phrases to remove
    # c = 0
    # for i in X_train:
    #     for j in i[-7:]:
    #         if 'goodbye?\n' in j or "okay i think i have asked everything i need to?\n" 'thanks for sharing your thoughts with me?\n' in j:
    #             c+=1
    ##Which quesitons to add
    # for i in range(0, len(X_train[0]), 2):
    #     print(X_train[0][i])




