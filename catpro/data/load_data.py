
import logging
import pandas as pd
import numpy as np

import config

config_params = config.config
# path_to_dir = data_helpers.make_output_dir(os.path.join(config.config['output_dir'], 'neural_networks/'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# handler = logging.FileHandler(os.path.join(path_to_dir, 'training.log'))
# handler.setLevel(logging.INFO)
# logger.addHandler(handler)


'''
Interview, group by:
- Each sample is a list of lists, useful for timestep models or bayesian model
    -  Interview: each sample is an interview (list of responses). 
    -  Response: each sample is a response (list of speech segments)
- single_response: each sample is a single response, good for linear models
- with_question
'''





def load_data(dataset='train', group_by='single_response', run_audio=config.config['run_audio'],
                         run_text=config.config['run_text'], use_doc2vec=False):
    '''

    :param type: 
    :param timesteps: 
    :param group_by: {'response', 'interview'} each sample, where each time step is segment or response, respectively
    :return: 
    '''
    # test = self.kwargs.get('test')
    # inputPath = self.kwargs.get('input')
    # audio_file = self.kwargs.get('audio_file')
    logger.info('loading data...')

    inputPath = config.config['input']  # TODO: change to variable , inestead of dictionary
    audio_file = 'text_audio_df.csv'
    text_audio = pd.read_csv(inputPath + audio_file)

    if group_by == 'single_response' and group_by != 'with_question':
        text_audio = text_audio[text_audio.PARTICIPANT == 'Participant']
    if dataset == 'train':
        # TODO: Normal way is having a load_data for each dataset depending on the datatypes. And it turns it into Xs and ys. I could add a config section where you input the names of the columns.
        text_audio = text_audio[text_audio.FILE_TYPE == dataset.upper()]  # TODO load text from here too?
        X_train_text = np.array(text_audio.UTTERANCE)
        X_train_audio = np.array(text_audio.iloc[:, 8:])
        y_train = np.array(text_audio.LABEL)
        return X_train_text, X_train_audio, y_train #TODO: set in config whether there is both text and audio or not.












    if group_by == 'interview':
        # Each sample is one participant. Take all responses and pad to max. of 100.

        # Train
        # TODO: set in
        text_audio_train = text_audio[text_audio.FILE_TYPE == dataset.upper()]  # TODO load text from here too?
        text_audio_train_response = text_audio_train[
            text_audio_train.PARTICIPANT == 'Participant']  # without Ellie's questions
        # y_train_audio =np.array(text_audio_train_response.LABEL)
        # X_train_audio = np.array(text_audio_train_response.iloc[:,8:])
        # audio_features = np.array(text_audio_train_response.columns[8:])

        # Group by participant
        X_train_text = []
        X_train_audio = []
        y_train = []
        for participant in list(set(text_audio_train_response.FILE_NAME)):
            text_audio_train_response_participant = text_audio_train_response[
                text_audio_train_response.FILE_NAME == participant]  # all the responses of 1 participant
            X_train_text_participant = np.array(text_audio_train_response_participant.UTTERANCE)
            X_train_audio_participant = np.array(text_audio_train_response_participant.iloc[:, 8:])
            y_train_participant = set(text_audio_train_response_participant.LABEL)
            if 0 in y_train_participant:  # turn list into int w/o using set
                y_train_participant = 0
            else:
                y_train_participant = 1
            # Append participants data to datasets
            X_train_text.append(X_train_text_participant)
            X_train_audio.append(X_train_audio_participant)
            y_train.append(y_train_participant)
        X_train_audio = np.array(X_train_audio)
        X_train_text = np.array(X_train_text)
        np.save(X_train_text, output_dir + 'X_train_text_groupedby_interview.npy')
        np.save(X_train_audio, output_dir + 'X_train_audio_groupedby_interview.npy')
        # TODO: fix
        # for participant_matrix in X_train_audio_padded:
        #     break
        #     participant_vector = np.array(pd.DataFrame(participant_matrix ).mean())
        # X_train_audio_padded= pad_sequences(X_train_audio, maxlen=timesteps, padding=padding, dtype='float32')
        # X_train_text_padded = pad_sequences(X_train_text, maxlen=timesteps, padding=padding, dtype='str')
        if type == 'train':
            return X_train_text, X_train_audio, y_train,
            # elif type == 'test' and test: #TODO
            # if test:
            #   text_audio_test =      # TODO
            # responses_per_participant_disorder= []
            # responses_per_participant_control = []
            # for participant in list(set(list(text_audio_train_response.id))):
            #     text_audio_train_response_participant = text_audio_train_response[text_audio_train_response['id']==participant]
            #     amount_of_responses = text_audio_train_response_participant .shape[0]
            #     amount_of_words_per_response = []
            #     for response in list(text_audio_train_response_participant.UTTERANCE):
            #         amount_of_words_per_response.append(len(response.split()))
            #     amount_of_words_per_interview = np.sum(amount_of_words_per_response)
            #     if 0 in set(text_audio_train_response_participant.LABEL):
            #         responses_per_participant_control.append([amount_of_responses, amount_of_words_per_interview])
            #     elif 1 in set(text_audio_train_response_participant.LABEL):
            #         responses_per_participant_disorder.append([amount_of_responses, amount_of_words_per_interview])
            # # Get statistics
            # depression_responses = np.sum([n[0] for n in responses_per_participant_disorder])
            # depression_words = np.sum([n[1] for n in responses_per_participant_disorder])
            # control_responses = np.sum([n[0] for n in responses_per_participant_control])
            # control_words = np.sum([n[1] for n in responses_per_participant_control])
            #
            # # See if you can classify according to fluency
            # y_train_disorder = [1]*len(responses_per_participant_disorder)
            # y_train_control = [0]*len(responses_per_participant_control)
            #
            # y_train_all = np.concatenate((y_train_disorder, y_train_control))
            # X_train_all = np.concatenate((responses_per_participant_disorder, responses_per_participant_control))
            #
            #
            # import matplotlib.pyplot as plt
            #
            # # evenly sampled time at 200ms intervals
            # x = [n[0] for n in responses_per_participant_disorder]
            # y = [n[1] for n in responses_per_participant_disorder]
            # x1 = [n[0] for n in responses_per_participant_control]
            # y1 = [n[1] for n in responses_per_participant_control]
            #
            # # red dashes, blue squares and green triangles
            # plt.clf()
            # # y=np.zeros(len(x))
            # # y1 = np.zeros(len(x1))
            # plt.plot(x,y, 'ro', x1, y1, 'bo')
            # plt.savefig(config.config['output_dir']+'2D_responses_per_participant.png', dpi=200)
            #
            # X_train, X_dev, y_train, y_dev, = train_test_split(X_train_all, y_train_all,
            #                                                                             test_size=0.20, random_state=0,
            #                                                                             shuffle=True)  # TODO: save and add test_size to parameters.
            #
            # # y_dev = [int(n) for n in y_dev]
            # # TODO: shuffle, I think text has some of the test set.
            # clf = LinearSVC(C=0.1, random_state=0)  # only cv
            # clf = GaussianMixture(n_components=2)
            #
            # clf.fit(X_train, y_train)
            #
            # y_pred = clf.predict(X_dev)
            # y_pred = [int(n) for n in y_pred]
            # # For interpretation purposes, we compute the decision confidence (i.e., normalized distance from boundary)
            # # TODO add interpretation.py functions here.
            # f1 = f1_score(y_dev, y_pred)
            # logger.info(f1)
            # acc = accuracy_score(y_dev, y_pred)
            # logger.info(acc)
            # roc_auc = roc_auc_score(y_dev, y_pred)
            # precision = precision_score(y_dev, y_pred)
            # recall = recall_score(y_dev, y_pred)
            #




            #     return X_test_audio, y_test_audio, audio_features
    elif group_by == 'response':
        inputPath = config.config['input']  # TODO: make variable in config
        audio_file = 'text_audio_df_nonconcat.csv'  # TODO, buld one using nonconcat.
        text_audio = pd.read_csv(inputPath + audio_file)
        text_audio = text_audio[text_audio.FILE_TYPE == dataset.upper()]
        # if run_text:
        #     # Normalize text and reduce to k dimensions
        #     # text_audio_participants = text_audio[text_audio.PARTICIPANT=='Participant']
        #     text = text_audio_participants.UTTERANCE #TODO. CHange name to responses

        # TODO: I could seperate loading audio and text, but then i have to load text_audio_df_nonconcat.csv twice

        # Normalize audio and reduce to k dimensions
        audio_features = text_audio.iloc[:, 8:].values
        # It's hard to normalize with such long sequences, so convert to replace inf for 0
        bad_indices = np.where(np.isnan(audio_features))
        audio_features[bad_indices] = 0
        # audio_features_wo_inf = []
        # for row in audio_features:
        #     bad_indices = np.where(np.isinf(row))
        #     row[bad_indices] = 0
        #     audio_features_wo_inf.append(row)
        #     logger.info(bad_indices)
        # audio_features_wo_inf = np.array(audio_features_wo_inf)
        # audio_features.dtype  # > dtype('float16')

        audio_normalized = data_helpers.normalize(array_train=audio_features)  # TODO: do for text as well
        k = 32
        audio_features_names = text_audio.columns[8:]
        audio_normalized_best, kbest_features_names_audio = data_helpers.f_feature_selection(X=audio_normalized,
                                                                                             y=np.array(
                                                                                                 text_audio.LABEL), k=k,
                                                                                             audio_features=audio_features_names,
                                                                                             print_=True)  # TODO fix print_, I'd like to log this.
        audio_normalized_best = pd.DataFrame(audio_normalized_best, columns=kbest_features_names_audio)
        text_audio_wo_audio = text_audio.iloc[:, :8].reset_index()  # reset index in order to concat
        text_audio_normalized_best = pd.concat((text_audio_wo_audio, audio_normalized_best), axis=1)  # TODO: add text
        # Each sample is one response. Take all segments and pad to max. of 100.
        # Group by response: each sample is a response, each time step is a segment.
        X_train_text = []
        X_train_audio = []
        y_train = []
        segments = 0
        for row in range(text_audio_normalized_best.shape[0]):
            # loop through rows of segments
            speaker = text_audio_normalized_best.PARTICIPANT.iloc[row]
            if speaker == 'Ellie':
                if segments == 0:
                    continue
                else:
                    X_train_text.append(one_response_text)
                    X_train_audio.append(one_response_audio)
                    y_train.append(int(text_audio_normalized_best.LABEL.iloc[row]))
                    segments = 0  # each time Ellie speaks a new response begins
            elif speaker == 'Participant' and segments == 0:
                one_response_text = []
                one_response_audio = []
                one_response_text.append(text_audio_normalized_best.UTTERANCE.iloc[row])
                one_response_audio.append(np.array(text_audio_normalized_best.iloc[row, 8:]))

                segments += 1
            elif speaker == 'Participant' and segments > 0:
                one_response_text.append(text_audio_normalized_best.UTTERANCE.iloc[row])
                one_response_audio.append(np.array(text_audio_normalized_best.iloc[row, 8:]))
                segments += 1
        y_train = np.array(y_train)
        X_train_audio = np.array(X_train_audio)
        X_train_text = np.array(X_train_text)
        # TODO: this is temporary, replace with full features or average embeddings, elmo, doc2vec
        if use_doc2vec:
            # Following Alhaini (2018), train doc2vec on both questions and responses
            config_params = config.config
            if config_params['create_features']:
                logger.info('creating doc2vec vectors...')
                # questions
                text_audio_ellie = text_audio[
                    text_audio.PARTICIPANT == 'Ellie'].UTTERANCE.values  # TODO: maybe do not include Ellie, lot of repeated sentences
                texts_ellie = []
                for i in text_audio_ellie:
                    try:
                        texts_ellie.append(i.split())
                    except:
                        logger.info('skipped segment during doc2vec training')
                texts_participants = []
                for j in X_train_text:
                    text_one_participant = []
                    for i in j:
                        try:
                            text_one_participant.append(i.split())
                        except:
                            logger.info(i)
                            logger.info('skipped segment during doc2vec training')
                    texts_participants.append(text_one_participant)
                # doc2vec neads one lists of lists, not list of list of list like text_participants is
                text_participants_lists = []
                for i in texts_participants:
                    #     concatenate
                    flattened_list = [item for sublist in i for item in sublist]
                    text_participants_lists.append(flattened_list)
                texts = texts_ellie + text_participants_lists

                doc2vec_model = doc2vec.doc2model(documents=texts, output_dir=config_params['output_dir'])
                X_train_text_doc2vec = []
                for i in texts_participants:
                    X_train_text_doc2vec_one_participant = doc2vec.model2vec(documents=i,
                                                                             input_dir=config_params['output_dir'],
                                                                             model=doc2vec_model)
                    X_train_text_doc2vec.append(X_train_text_doc2vec_one_participant)
                # np.save('./data/datasets/X_train_text_doc2vec3.npy', X_train_text_doc2vec)
                np.save('./data/datasets/X_train_text_doc2vec_latest.npy', X_train_text_doc2vec)
            else:
                X_train_text_doc2vec = np.load(config_params['input'] + 'X_train_text_doc2vec.npy')

            X_train_text = X_train_text_doc2vec[:]
            # X_train_text_padded = pad_sequences(X_train_text_doc2vec, maxlen=timesteps, padding=padding,
            #                                     dtype='str')
        else:
            config_params = config.config
            featureNames = config_params.get('features').split(',')
            featGenr = feature_generator.FeatureGenerator(featureNames, config_params)
            dataHandler = data_handler.DataHandler(config_params)
            vocabs = dataHandler.loadAllVocabs(inputPath)
            vectors = dataHandler.loadEmbedding(vocabs,
                                                vector_length=100)  # TODO: save loaded version, cause it takes a while
            allFeatureList = featGenr.initFeatures()
            unknown_vec = np.random.normal(0, 0.17, 100)
            X_train_text_embeddings = []
            for response in X_train_text:
                response_embeddings = []
                for segment in response:
                    try:
                        embeds = list(featGenr.generateEmbeddingFeatures(segment, vectors, unknown_vec,
                                                                         100).values())  # TODO: set 100 in config
                        response_embeddings.append(embeds)
                    except:
                        logger.info('skipped segment')
                response_embeddings = response_embeddings[:timesteps]

                X_train_text_embeddings.append(response_embeddings)
            X_train_text = X_train_text_embeddings[:]
            # X_train_text_padded = pad_sequences(X_train_text_embeddings, maxlen=timesteps, padding=padding,
            #                                     dtype='str')
        #     max=10 segments per response, mean =2.7
        # TODO: make sure y_train.shape[0] or X_train_audio.shape[0], i.e., the amount of samples matches text_audio_df_concat.csv=='Participant' length once the latter is re-preprocessed with 3 missing people.
        # TODO: could save below and try with different timesteps
        # np.savez_compressed('./data/input/X_train_nonconcatenated.npz', a=X_train_text, b=X_train_audio,
        #                     c=y_train)
        # timesteps should be near max, but not too far from mean or median of segments per response.
        # X_train_audio_padded = pad_sequences(X_train_audio, maxlen=timesteps, padding=padding, dtype='float32')


        # if test:
        #   text_audio_test =      # TODO
        if dataset == 'train':
            return X_train_text, X_train_audio, y_train
            # elif type == 'test' and test: #TODO
            #     return X_test_audio, y_test_audio, audio_features
