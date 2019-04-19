import pandas as pd
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io as io
import numpy as np
import data.config_uic as config

import importlib
importlib.reload(config)

input_dir = config.input_dir
output_dir = config.output_dir
input_file = config.input_file


# Build dataset
participants = pd.read_csv(input_dir+'participants.csv')
samples = pd.DataFrame(columns = ['id', 'group', 'day', 'response_type'])
ids = [[n]*8 for n in participants.id]
ids = [n for i in ids for n in i]
group = [[n.lower()]*8 for n in participants.group]
group = [n for i in group for n in i]
day = [1,1,2,2,3,3,4,4] * participants.shape[0]

np.sum(participants.total_fr)


# opensmile
# SMILExtract -C config/MFCC12_E_D_A.conf -I ./../catpro/catpro/data/datasets/banda/556_b.wav -csvoutput output.csv
# 'bash ./data/datasets/uic/all_subjects/extract_features.sh' does IS13 ComParE.conf

for sample in participants
df = pd.read_csv(input_dir+'all_subjects/hc_08501_p1_freeresp.csv', sep=';')
df.iloc[:,1:].to_csv(input_dir+'all_subjects_temp/hc_08501_p1_freeresp2.csv')











(rate,sig) = wav.read(input_dir+input_file,)
mfcc_feat = mfcc(sig,rate)
fbank_feat = logfbank(sig,rate)


print(fbank_feat[1:3,:])


# test sampling performed
import soundfile as sf
f = sf.SoundFile(input_dir+input_file)
print('samples = {}'.format(len(f)))
print('sample rate = {}'.format(f.samplerate))
print('seconds = {}'.format(len(f) / f.samplerate))

#From opensmile pdf: 25ms audio frames (sampled at a rate of 10ms) (Hamming window).
