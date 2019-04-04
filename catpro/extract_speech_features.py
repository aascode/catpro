import pandas as pd
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io as io

import config_banda

import importlib
importlib.reload(config_banda)

input_dir = config_banda.input_dir
output_dir = config_banda.output_dir
input_file = config_banda.input_file

(rate,sig) = wav.read(input_dir+input_file,)
mfcc_feat = mfcc(sig,rate)
fbank_feat = logfbank(sig,rate)


print(fbank_feat[1:3,:])



# opensmile
# SMILExtract -C config/MFCC12_E_D_A.conf -I ./../catpro/catpro/data/datasets/banda/556_b.wav -csvoutput output.csv

df = pd.read_csv('./../../opensmile-2.3.0/output.csv', sep=';')

# test sampling performed
import soundfile as sf
f = sf.SoundFile(input_dir+input_file)
print('samples = {}'.format(len(f)))
print('sample rate = {}'.format(f.samplerate))
print('seconds = {}'.format(len(f) / f.samplerate))

#From opensmile pdf: 25ms audio frames (sampled at a rate of 10ms) (Hamming window).
