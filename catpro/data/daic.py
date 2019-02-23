'''
Daniel M. Low, Harvard University-MIT

DAIC dataset contains zip files for each subject. 
This script extracts speech and text files and erases video files
due to size, and because we're not using them.

'''

# TODO: this script is depracted and replaced by preprocess_dataset.py

import os
import zipfile
import pandas as pd
import numpy as np
# import timeit
import time
# import operator
# import math
# import shutil

# Cluster
input_dir = '/om2/data/public/daiccorpus/input/' #TODO: to config
output_dir = '/om2/user/dlow/'


# Create y vectors
def create_labels(input_dir = None):
    # Save training ids
    df = pd.read_csv(os.path.join(input_dir, 'train_split_Depression_AVEC2017.csv'))
    np.save(os.path.join(input_dir, 'ids_train'), list(df['Participant_ID']))
    # Save dev ids
    df = pd.read_csv(os.path.join(input_dir, 'dev_split_Depression_AVEC2017.csv'))
    np.save(os.path.join(input_dir, 'ids_dev'), list(df['Participant_ID']))

    # Save binary, 24-way linear regression
    ## Save train
    y_train = pd.read_csv(os.path.join(input_dir, 'train_split_Depression_AVEC2017.csv'))
    np.save(os.path.join(input_dir, 'y_train_binary'), list(y_train['PHQ8_Binary']))
    np.save(os.path.join(input_dir, 'y_train_24'), list(y_train['PHQ8_Score']))
    ## Save dev (our test set)
    y_train = pd.read_csv(os.path.join(input_dir, 'dev_split_Depression_AVEC2017.csv'))
    np.save(os.path.join(input_dir, 'y_dev_binary'), list(y_train['PHQ8_Binary']))
    np.save(os.path.join(input_dir, 'y_dev_24'), list(y_train['PHQ8_Score']))
    return

## Create X matrices
# 1. Extract files from .zips and remove CLNF (visual) files, since they are heavy and we're not using them
def unzip_daic_audio(input_dir = None, output_dir = None):
    '''
    :param input_dir: directory with .zip files
    :param output_dir: where the unzipped files will be
    :return: None
    '''

    try:
        os.mkdir(output_dir+'unzipped')
    except:
        print('directory already exists')
    files = os.listdir(input_dir)
    for file in files:
        if file.endswith('_P.zip'):
            participant_id = file[:3]
            # Extract zip
            with zipfile.ZipFile(os.path.join(input_dir, file), "r") as zip_ref:
                os.mkdir(os.path.join(output_dir,'unzipped/',participant_id))
                zip_ref.extractall(os.path.join(output_dir, 'unzipped', participant_id))
            files_participant = os.listdir(os.path.join(output_dir, 'unzipped', participant_id))
            # delete CLNF video files
            clnf_files = [n for n in files_participant if '_CLNF_' in n]
            for clnf_file in clnf_files:
                os.remove(os.path.join(output_dir, 'unzipped', participant_id,clnf_file))
            # shutil.rmtree(os.path.join(output_dir, '__MACOSX')) #only necessary in MacOS machine
    return


# 2. Append all unzipped csv files to one csv
def unzipped_to_csv(input_dir=None, output_dir=None):
    '''

    :param input_dir: 
    :param output_dir: 
    :return: 
    '''
    df_train = pd.read_csv(os.path.join(output_dir, 'train_split_Depression_AVEC2017.csv'))
    df_dev = pd.read_csv(os.path.join(output_dir, 'dev_split_Depression_AVEC2017.csv'))
    # Create dataframe with the data and labels for all responses.
    columns = ['id', 'train_test', 'y_binary', 'y_24', 'gender', 'start_time', 'stop_time', 'speaker', 'train_text']
    text = pd.DataFrame(columns=columns)
    # Loop through subjects responses
    files = os.listdir(input_dir)
    for file in files:
        try:
            participant_id = int(file)  # because you may have extra file such as .DS_Store
            if type(participant_id) == int:
                text_participant = pd.DataFrame(columns=columns)
                df_train_participant = df_train.loc[df_train.Participant_ID == int(participant_id)]
                # Open text responses
                text = pd.read_csv(os.path.join(input_dir, file, file + '_TRANSCRIPT.csv'), sep='\t')
                numb_responses = text.shape[0]
                # Determine if subject is part of train or test set
                # TODO: erase "train" and add as parameter. Then you use the same code for train or dev. Difference in the id file you load.
                # TODO: add values for each of 8 questions in PHQ8.
                if int(participant_id) in list(df_train.Participant_ID):
                    text_participant['id'] = [int(participant_id)] * numb_responses
                    text_participant['train_test'] = ['train'] * numb_responses
                    text_participant['y_binary'] = [int(df_train_participant.PHQ8_Binary)] * numb_responses
                    text_participant['y_24'] = [int(df_train_participant.PHQ8_Score)] * numb_responses
                    text_participant['gender'] = [int(df_train_participant.Gender)] * numb_responses
                    text_participant['start_time'] = text.start_time
                    text_participant['stop_time'] = text.stop_time
                    text_participant['speaker'] = text.speaker
                    text_participant['train_text'] = text.value
                    text = text.append(text_participant)
                    # elif participant_id in df_dev.Participant_ID:
                    #     participant_train_test = 'test' #dataset does not include test set, so we treat dev set as test set
                    # else:
                    #     participant_train_test = 'TBD'
                    # train_test = [participant_train_test] * numb_responses
                    #
                    # columns = ['y_24', 'gender', 'start_time', 'stop_time', 'speaker', 'text']
                    #
                    #
                    # text['start_time'] = text['start_time']
                    # text['start_time'] = text['start_time']


                    # Open audio feature responses
                    # covarep = pd.read_csv(os.path.join(input_dir, participant_id+'_COVAREP.csv'))
                    # formant = pd.read_csv(os.path.join(input_dir, participant_id+'_formant.csv'))
        except:
            continue
    text.to_csv(output_dir + 'train.csv')
    return


# 3. Concat covarep and formant features
def save_audio_features(input_dir = None, output_dir = None):
    '''
    temporary input_dir: 
    :param input_dir: 
    :param output_dir: 
    :return: 
    '''
    files = os.listdir(input_dir)
    # Make table to append to
    covarep_columns = ['covarep_' + str(n) for n in range(74)]
    formant_columns = ['formant_' + str(n) for n in range(5)]
    # audio = pd.DataFrame(columns=covarep_columns+formant_columns)
    # Load text
    # Segmented data: 25500 rows (with questions)
    # text = pd.read_csv(output_dir + 'train_df.csv')
    # Array of segments seperated by participant:
    X_audio, participant = np.load(output_dir+'audio.npz')['a'],np.load(output_dir+'audio.npz')['b']
    segments_per_interview = []
    for sample in X_audio:
        segments_per_interview.append(sample.shape[0])
    # Concatenated segments into utterances: 21377 rows (with questions)
    text = pd.read_csv(output_dir + 'depression_all_data.txt', sep='\t') #segments concatenated into utterances
    # See how many utterance per each participants's interview
    utterances_per_interview = []
    participants = list(set(text.FILE_NAME))
    for participant in participants:
        df_participant = text[text.FILE_NAME==participant]
        df_participant = df_participant[df_participant.PARTICIPANT=='Participant'] #exclude Ellie's questions
        utterances_per_interview.append(df_participant.shape[0])
    # TODO: why do some participants have 1 answer??
    # Find file_names:
    utterances_per_interview_sorted = utterances_per_interview[:]
    utterances_per_interview_sorted.sort() #this tells us there are 3 with answers 1.
    for i, number_of_utterances in enumerate(utterances_per_interview):
        if number_of_utterances ==1:
            print(i)
    # index 25, 57, 105 appear to have one response. This was an error in preprocessing.
    print(participants[25],participants[57],participants[105])
    # 480_P.zip 451_P.zip 458_P.zip
    # verify:
    # text[text.FILE_NAME == '480_P.zip']
    # Jus to make sure another shorter one (34 utterances) is complete:
    print(participants[34])
    # 469_P.zip
    text[text.FILE_NAME == '469_P.zip'].UTTERANCE




        # See how many segments on average:
    words_per_segment =[]

    for segment, speaker in zip(np.array(text.UTTERANCE), np.array(text.PARTICIPANT)):
        if speaker == 'Participant':
            try:
                sent_len = len(segment.split(' '))
                # print(segment.split(' '))
                if sent_len >= 3:
                    words_per_segment.append(sent_len)
            except: continue
    # How many segments per utterance?
    text = pd.read_csv(output_dir + 'train_df.csv')
    segments_per_utterance = []
    participant=0
    for speaker in np.array(text.speaker):
        if speaker == 'Participant':
            participant += 1
        else:
            segments_per_utterance.append(participant)
            participant = 0
    segments_per_utterance = list(filter((0).__ne__, segments_per_utterance ))

    # Create text_audio and, for each participant (i.e., file), for each line, add mean audio
    ids = [int(n[:-6]) for n in list(text.FILE_NAME)]
    text['id'] = ids
    # audio_columns = covarep_columns + formant_columns
    # descriptive_statistics_columns = [n + '_mean' for n in audio_columns] + [n + '_min' for n in
    #                                                                                      audio_columns] + [
    #                                      n + '_max' for n in audio_columns] + [n + '_median' for n in
    #                                                                                        audio_columns] + [
    #                                      n + '_std' for n in audio_columns] + [n + '_skew' for n in
    #                                                                                        audio_columns] + [
    #                                      n + '_kurtosis' for n in audio_columns]
    # descriptive_statistics = pd.DataFrame(index=range(text.shape[0]), columns=descriptive_statistics_columns)
    # text_audio = pd.concat((text, descriptive_statistics), sort=False, axis=1)
    audio = []
    ids_audio = []
    for file in files:
        try:
            participant_id = int(file) #must loop through files such as 303, 313 and avoid extra files such as .DS_Store
        except:
            continue
        if type(participant_id) == int: # Then we include participants that in train set
            start = time.time()
            text_one_participant = text[text.id == participant_id] # leave only text responses of that participant
            text_one_participant.index =  range(text_one_participant.shape[0]) # reindex
            covarep = pd.read_csv(os.path.join(input_dir, str(participant_id), str(participant_id) + '_COVAREP.csv'), sep=',', header=None)
            formant = pd.read_csv(os.path.join(input_dir, str(participant_id), str(participant_id) + '_FORMANT.csv'), sep=',', header=None)
            covarep.columns = covarep_columns
            formant.columns = formant_columns
            '''
            most differ in the exact number by 1, for example:
            diff=-1, covarep: 68079, formant: 68080
            diff= 1, covarep: 108921, formant: 108920
            so I need to take the minimum size and use those rows from both (the larger one will remove a row at the end). 
            '''
            min_numb_rows = np.min([covarep.shape[0], formant.shape[0]])
            covarep = covarep.iloc[:min_numb_rows, :]
            formant = formant.iloc[:min_numb_rows, :]
            # Concat both audio modalities

            audio_participant = pd.concat([covarep, formant], axis=1, sort=False)


            # audio_participant['id2'] = [participant_id] * min_numb_rows #sanity check to match with text df
            # average audio segments to track text segments
            # loop through text segments, identify time range of each segment and take the mean of the audio_participant

            # end_audio_row = int((list(text_one_participant.END_TIME)[-1]*1000)/10)
            audio_one_participant = []
            ids_audio = []
            # text_audio.iloc[:,:8] = text_one_participant.iloc[:,:]
            temp = []
            for text_row in range(text_one_participant.shape[0]):
                # Here you can round
                start_audio_row = int(((text_one_participant.START_TIME[text_row] * 1000) / 10 ).round())
                msec_per_segment = (text_one_participant.END_TIME[text_row] - text_one_participant.START_TIME[text_row]) * 1000
                # Tuka says they use 20 msec windows and 10 msec stride, but doesnt add up.
                # audio_rows_to_be_averaged = int(int(msec_per_segment) -20+10/ 10)
                # 10 msec: see DAIC WOZ Depression Database
                audio_rows_to_be_averaged = int((msec_per_segment/10).round())
                # TODO: dont go over max rows, if not it returns vector of NaNs
                audio_for_one_utterance_mean = audio_participant.iloc[start_audio_row :(start_audio_row +audio_rows_to_be_averaged ), :].mean().tolist()
                audio_for_one_utterance_max = audio_participant.iloc[start_audio_row:(start_audio_row + audio_rows_to_be_averaged),:].max().tolist()
                audio_for_one_utterance_min = audio_participant.iloc[start_audio_row:(start_audio_row + audio_rows_to_be_averaged),:].min().tolist()
                audio_for_one_utterance_median = audio_participant.iloc[
                                                 start_audio_row:(start_audio_row + audio_rows_to_be_averaged),
                                                 :].median().tolist()
                audio_for_one_utterance_std = audio_participant.iloc[start_audio_row:(start_audio_row + audio_rows_to_be_averaged),:].std().tolist()
                audio_for_one_utterance_skew = audio_participant.iloc[
                                                 start_audio_row:(start_audio_row + audio_rows_to_be_averaged),
                                                 :].skew().tolist()
                audio_for_one_utterance_kurtosis = audio_participant.iloc[
                                                 start_audio_row:(start_audio_row + audio_rows_to_be_averaged),
                                                 :].kurtosis().tolist()
                # for feature in mean_audio_for_one_utterance:
                # Add to df
                # text_audio.iloc[text_row,8:] = audio_for_one_utterance_mean + audio_for_one_utterance_max + audio_for_one_utterance_min + audio_for_one_utterance_median + audio_for_one_utterance_std + audio_for_one_utterance_skew + audio_for_one_utterance_kurtosis # add mean to text_audio df
                # mean_audio_one_participant[text_row] = mean_audio_for_one_utterance
                audio_one_participant.append(audio_for_one_utterance_mean+audio_for_one_utterance_min+ audio_for_one_utterance_max+audio_for_one_utterance_median+audio_for_one_utterance_std+audio_for_one_utterance_skew+audio_for_one_utterance_kurtosis)
                # Test:
                # temp.append([msec_per_segment, start_audio_row, audio_rows_to_be_averaged])
                start_audio_row += audio_rows_to_be_averaged
            '''
            # test: subject 300 ends at row 62127, which corresponds to 62127*10=621270ms /1000/60 = 10.34 minutes
            last_row= temp[-1][1]+temp[-1][2]
            print(np.abs(last_row* 10 / 1000  - list(text_one_participant.END_TIME)[-1]))
            It falls between 10 msec accuracy. Very good.
            '''
            # text_audio.iloc[text_one_participant.shape[0], -80:] = audio_for_one_utterance_mean  # add mean to text_audio df
            end = time.time()
            print(end-start)
        # for each participant, append to overall data:
        audio.append(audio_one_participant)
        ids_audio.append(participant_id)
    # text_audio.to_csv(output_dir+'text_audio.csv')
    np.savez_compressed(output_dir+'audio.npz', a = audio, b= ids_audio)
    print(str(len(ids_audio))+' participants completed')
    return



def text_audio_df_nonconcat(input_dir = None, output_dir = None):
    '''
    Create a dataframe with every row being either a question or a segment of a response. In the previous these were 
    concatenated, but for sequential models, we will take each segment as a timestep. 
    temporary input_dir: 
    :param input_dir: 
    :param output_dir: 
    :return: 
    '''
    input_dir='./data/'
    output_dir = './data/'
    files = os.listdir(input_dir) #/unzipped/ where each subdirectory contains the data for 1 participant.
    covarep_columns = ['covarep_' + str(n) for n in range(74)]
    formant_columns = ['formant_' + str(n) for n in range(5)]

    text = pd.DataFrame()
    for participant in files:
        text = pd.concat(text, participant+'_TRANSCRIPT.csv')

    # Now obtain time for each segment, and average audio like you did for audio.npz






    text = pd.read_csv(output_dir + 'depression_all_data.txt', sep='\t') #segments concatenated into utterances
    # See how many utterance per each participants's interview
    utterances_per_interview = []
    participants = list(set(text.FILE_NAME))
    for participant in participants:
        df_participant = text[text.FILE_NAME==participant]
        df_participant = df_participant[df_participant.PARTICIPANT=='Participant'] #exclude Ellie's questions
        utterances_per_interview.append(df_participant.shape[0])
    # TODO: why do some participants have 1 answer??
    # Find file_names:
    utterances_per_interview_sorted = utterances_per_interview[:]
    utterances_per_interview_sorted.sort() #this tells us there are 3 with answers 1.
    for i, number_of_utterances in enumerate(utterances_per_interview):
        if number_of_utterances ==1:
            print(i)
    # index 25, 57, 105 appear to have one response. This was an error in preprocessing.
    print(participants[25],participants[57],participants[105])
    # 480_P.zip 451_P.zip 458_P.zip
    # verify:
    # text[text.FILE_NAME == '480_P.zip']
    # Jus to make sure another shorter one (34 utterances) is complete:
    print(participants[34])
    # 469_P.zip
    text[text.FILE_NAME == '469_P.zip'].UTTERANCE

        # See how many segments on average:
    words_per_segment =[]

    for segment, speaker in zip(np.array(text.UTTERANCE), np.array(text.PARTICIPANT)):
        if speaker == 'Participant':
            try:
                sent_len = len(segment.split(' '))
                # print(segment.split(' '))
                if sent_len >= 3:
                    words_per_segment.append(sent_len)
            except: continue
    # How many segments per utterance?
    text = pd.read_csv(output_dir + 'train_df.csv')
    segments_per_utterance = []
    participant=0
    for speaker in np.array(text.speaker):
        if speaker == 'Participant':
            participant += 1
        else:
            segments_per_utterance.append(participant)
            participant = 0
    segments_per_utterance = list(filter((0).__ne__, segments_per_utterance ))

    # Create text_audio and, for each participant (i.e., file), for each line, add mean audio
    ids = [int(n[:-6]) for n in list(text.FILE_NAME)]
    text['id'] = ids
    audio = []
    ids_audio = []
    for file in files:
        try:
            participant_id = int(file) #must loop through files such as 303, 313 and avoid extra files such as .DS_Store
        except:
            continue
        if type(participant_id) == int: # Then we include participants that in train set
            start = time.time()
            text_one_participant = text[text.id == participant_id] # leave only text responses of that participant
            text_one_participant.index =  range(text_one_participant.shape[0]) # reindex
            covarep = pd.read_csv(os.path.join(input_dir, str(participant_id), str(participant_id) + '_COVAREP.csv'), sep=',', header=None)
            formant = pd.read_csv(os.path.join(input_dir, str(participant_id), str(participant_id) + '_FORMANT.csv'), sep=',', header=None)
            covarep.columns = covarep_columns
            formant.columns = formant_columns
            '''
            most differ in the exact number by 1, for example:
            diff=-1, covarep: 68079, formant: 68080
            diff= 1, covarep: 108921, formant: 108920
            so I need to take the minimum size and use those rows from both (the larger one will remove a row at the end). 
            '''
            min_numb_rows = np.min([covarep.shape[0], formant.shape[0]])
            covarep = covarep.iloc[:min_numb_rows, :]
            formant = formant.iloc[:min_numb_rows, :]
            # Concat both audio modalities

            audio_participant = pd.concat([covarep, formant], axis=1, sort=False)


            # audio_participant['id2'] = [participant_id] * min_numb_rows #sanity check to match with text df
            # average audio segments to track text segments
            # loop through text segments, identify time range of each segment and take the mean of the audio_participant

            # end_audio_row = int((list(text_one_participant.END_TIME)[-1]*1000)/10)
            audio_one_participant = []
            ids_audio = []
            # text_audio.iloc[:,:8] = text_one_participant.iloc[:,:]
            temp = []
            for text_row in range(text_one_participant.shape[0]):
                # Here you can round
                start_audio_row = int(((text_one_participant.START_TIME[text_row] * 1000) / 10 ).round())
                msec_per_segment = (text_one_participant.END_TIME[text_row] - text_one_participant.START_TIME[text_row]) * 1000
                # Tuka says they use 20 msec windows and 10 msec stride, but doesnt add up.
                # audio_rows_to_be_averaged = int(int(msec_per_segment) -20+10/ 10)
                # 10 msec: see DAIC WOZ Depression Database
                audio_rows_to_be_averaged = int((msec_per_segment/10).round())
                # TODO: dont go over max rows, if not it returns vector of NaNs
                audio_for_one_utterance_mean = audio_participant.iloc[start_audio_row :(start_audio_row +audio_rows_to_be_averaged ), :].mean().tolist()
                audio_for_one_utterance_max = audio_participant.iloc[start_audio_row:(start_audio_row + audio_rows_to_be_averaged),:].max().tolist()
                audio_for_one_utterance_min = audio_participant.iloc[start_audio_row:(start_audio_row + audio_rows_to_be_averaged),:].min().tolist()
                audio_for_one_utterance_median = audio_participant.iloc[
                                                 start_audio_row:(start_audio_row + audio_rows_to_be_averaged),
                                                 :].median().tolist()
                audio_for_one_utterance_std = audio_participant.iloc[start_audio_row:(start_audio_row + audio_rows_to_be_averaged),:].std().tolist()
                audio_for_one_utterance_skew = audio_participant.iloc[
                                                 start_audio_row:(start_audio_row + audio_rows_to_be_averaged),
                                                 :].skew().tolist()
                audio_for_one_utterance_kurtosis = audio_participant.iloc[
                                                 start_audio_row:(start_audio_row + audio_rows_to_be_averaged),
                                                 :].kurtosis().tolist()
                # for feature in mean_audio_for_one_utterance:
                # Add to df
                # text_audio.iloc[text_row,8:] = audio_for_one_utterance_mean + audio_for_one_utterance_max + audio_for_one_utterance_min + audio_for_one_utterance_median + audio_for_one_utterance_std + audio_for_one_utterance_skew + audio_for_one_utterance_kurtosis # add mean to text_audio df
                # mean_audio_one_participant[text_row] = mean_audio_for_one_utterance
                audio_one_participant.append(audio_for_one_utterance_mean+audio_for_one_utterance_min+ audio_for_one_utterance_max+audio_for_one_utterance_median+audio_for_one_utterance_std+audio_for_one_utterance_skew+audio_for_one_utterance_kurtosis)
                # Test:
                # temp.append([msec_per_segment, start_audio_row, audio_rows_to_be_averaged])
                start_audio_row += audio_rows_to_be_averaged
            '''
            # test: subject 300 ends at row 62127, which corresponds to 62127*10=621270ms /1000/60 = 10.34 minutes
            last_row= temp[-1][1]+temp[-1][2]
            print(np.abs(last_row* 10 / 1000  - list(text_one_participant.END_TIME)[-1]))
            It falls between 10 msec accuracy. Very good.
            '''
            # text_audio.iloc[text_one_participant.shape[0], -80:] = audio_for_one_utterance_mean  # add mean to text_audio df
            end = time.time()
            print(end-start)
        # for each participant, append to overall data:
        audio.append(audio_one_participant)
        ids_audio.append(participant_id)
    # text_audio.to_csv(output_dir+'text_audio.csv')
    np.savez_compressed(output_dir+'audio.npz', a = audio, b= ids_audio)
    print(str(len(ids_audio))+' participants completed')
    return














# # Counting training samples
# responses = 0
# previous_i=None
# speaker = list(text.speaker)
# for i in range(len(speaker)):
#     try:
#         print(speaker[i])
#         if previous_i != 'Participant':
#             responses += 1
#             print('+1')
#         previous_i=speaker[i]
#     except: continue
#
#
# df1 = pd.read_csv('./data/depression_all_data.txt', sep='\t')
#
# df1 = pd.read_csv('./data/depression_all_data.txt', sep='\t')
#
# df1.iloc[:100,:].to_csv('./data/sample_dataset_text_concat.csv')
# text.iloc[:100,:].to_csv('./data/sample_dataset_text_nonconcat.csv')







# Main
# =====================================================================
# create_labels(input_dir = input_dir)

if __name__ == "__main__":
    # unzip_daic_audio(input_dir = input_dir, output_dir = output_dir)
    save_audio_features(input_dir= '/om2/user/dlow/unzipped/', output_dir= './data/')
    # save_audio_features(input_dir='./data/', output_dir='./data/')
    # unzipped_to_csv(input_dir='./data/', output_dir='./data/')
    # unzipped_to_csv(input_dir= '/om2/user/dlow/unzipped/', output_dir= './data/')