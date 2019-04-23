#!/usr/bin/env python
#coding: utf-8
'''

Author: Daniel M. Low (Harvard-MIT)

# Quality control:
ivector: the feature vectors should be more similar within speakers than across speakers
visualizations to understand dataset distribution and biases: 
spectrogram (silences), 
amount of speakers, 
heatmap on participant info.


'''
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import os
import config




def extract_triu(rsm):
    triu = rsm.where(np.triu(np.ones(rsm.shape)).astype(np.bool)).reset_index(drop=True)
    return triu

# Replace diagonal
def remove_diagonal(triu):
    for i in range(triu.shape[0]):
            triu.iloc[i,i] = np.nan
    return triu

def remove_nan(triu):
    triu = list(triu[np.logical_not(np.isnan(triu))])
    return triu





def ivector(input_dir=config.input_dir, input_file = 'uic_dataset_04112019.csv', normalize = False,normalizer=RobustScaler(), day = None, response_type = None):
    df = pd.read_csv(input_dir + input_file)

    if day:
        df = df[df['day'] == day]
    if response_type:
        df_subset = df[df['response_type'] == response_type]
    else:
        df_subset = df[df['response_type']!= 'background']
    if normalize:
        normalizer = normalizer  # feature_range=(-1, 1).
        df_subset  = pd.concat((
            df_subset.iloc[:,:4].reset_index().iloc[:,1:],
            pd.DataFrame(normalizer.fit_transform(df_subset.iloc[:,4:])).reset_index().iloc[:,1:]
                                ), axis=1, ignore_index=True,
                               )
        df_subset.columns = df.columns
    within_ids = []
    across_ids = []
    for id in np.unique(list(df_subset.id)):
        # take the vectors of the same id
        df_id = df_subset[df_subset['id'] == id].iloc[:,4:]
        # df_id = pd.DataFrame(normalizer.fit_transform(df_id))
        rsm = df_id.T.corr(method='pearson')
        triu = extract_triu(rsm)
        triu = remove_diagonal(triu)
        triu = triu.values.flatten()
        triu = remove_nan(triu)
        within_ids.append(np.mean(triu))
        # across
        df_not_id = df_subset[df_subset['id'] != id].iloc[:,4:]
        across_ids_one_id = []
        for sample in df_id.values:
            correlation = [pearsonr(sample, n)[0] for n in df_not_id.values]
            across_ids_one_id.append(np.mean(correlation))
        across_ids.append(np.mean(across_ids_one_id))
    return  np.round(np.nanmean(within_ids), 3), np.round(np.nanmean(across_ids),3)



def confusion_participant_by_timestep(input_dir = config.input_dir, participant_by_day = 'participants.csv'):
    '''
    Goal: see if confusions are within certain subjects, certain response types or random
    '''
    participants = pd.read_csv(input_dir+participant_by_day)











""" stft, logscale_spec, plot_stft is licensed under a Creative Commons Attribution 3.0 Unported License.
    Frank Zalkow, 2012-2013 """

""" short time fourier transform of audio signal """


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize / 2.0)), sig)
    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


""" scale frequency axis logarithmically """


def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins - 1) / max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            newspec[:, i] = np.sum(spec[:, scale[i]:], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, scale[i]:scale[i + 1]], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i + 1]])]

    return newspec, freqs


""" plot spectrogram"""


def plotstft(audiopath, binsize=2 ** 10, plotpath=None, colormap="jet"):
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins - 1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins - 1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs * len(samples) / timebins) + (0.5 * binsize)) / samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins - 1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()

    plt.clf()





corr_within_ids, corr_across_ids = ivector(input_dir=config.input_dir, input_file = 'uic_dataset_trimmed_segmented_04212019.csv', normalize=True, response_type='freeresp', normalizer=StandardScaler())
print('Nonsegmented, all days, only freeresp, normalized with Standard scalar: ',corr_within_ids, corr_across_ids )

corr_within_ids, corr_across_ids = ivector(input_dir=config.input_dir, input_file = 'uic_dataset_trimmed_segmented_04212019.csv', normalize=True, response_type='sentences', normalizer=StandardScaler())
print('Nonsegmented, all days, only sentences, normalized with Standard scalar: ',corr_within_ids, corr_across_ids )



# corr_within_ids, corr_across_ids = ivector(input_dir=config.input_dir, input_file = 'uic_dataset_04112019.csv', normalize=False)
# print('Nonsegmented, all days, freeresp+sentences, no normalization: ',corr_within_ids, corr_across_ids )
#
# corr_within_ids, corr_across_ids = ivector(input_dir=config.input_dir, input_file = 'uic_dataset_04112019.csv', normalize=True, normalizer=StandardScaler())
# print('Nonsegmented, all days, freeresp+sentences, normalized with Standard scalar: ',corr_within_ids, corr_across_ids )
#
# corr_within_ids, corr_across_ids = ivector(input_dir=config.input_dir, input_file = 'uic_dataset_04112019.csv', normalize=False, response_type='freeresp')
# print('Nonsegmented, all days, only freeresp, no normalization: ',corr_within_ids, corr_across_ids )
#
# corr_within_ids, corr_across_ids = ivector(input_dir=config.input_dir, input_file = 'uic_dataset_04112019.csv', normalize=True, response_type='freeresp', normalizer=StandardScaler())
# print('Nonsegmented, all days, only freeresp, normalized with Standard scalar: ',corr_within_ids, corr_across_ids )
#
# corr_within_ids, corr_across_ids = ivector(input_dir=config.input_dir, input_file = 'uic_dataset_04112019.csv', normalize=False, response_type='sentences')
# print('Nonsegmented, all days, only sentences, no normalization: ',corr_within_ids, corr_across_ids )
#
# corr_within_ids, corr_across_ids = ivector(input_dir=config.input_dir, input_file = 'uic_dataset_04112019.csv', normalize=True, response_type='sentences', normalizer=StandardScaler())
# print('Nonsegmented, all days, only sentences, normalized with Standard scalar: ',corr_within_ids, corr_across_ids )




# 'uic_dataset_trimmed_segmented_04212019.csv'
# print('segmented=======================================================')
#
# corr_within_ids, corr_across_ids = ivector(input_dir=config.input_dir, input_file = 'uic_dataset_trimmed_segmented_04212019.csv', normalize=False, response_type='freeresp', day='p1')
# print('segmented, all days, only freeresp, no normalization: ',corr_within_ids, corr_across_ids )
# corr_within_ids, corr_across_ids = ivector(input_dir=config.input_dir, input_file = 'uic_dataset_trimmed_segmented_04212019.csv', normalize=True, response_type='freeresp', day='p1')
# print('segmented, normalized with RobustScaler: ',corr_within_ids, corr_across_ids )



#
#
# # main
#
# input_dir = './data/datasets/uic/all_subjects/'
# output_dir = './data/datasets/uic/spectrograms/'
#
#
#
#
#
#
# files0 = os.listdir(input_dir)
# files = [n for n in files0 if n.endswith('freeresp.wav')]
# for i in files[:10]:
#     plotstft(input_dir+i, plotpath=output_dir+i[:-4] + '.png')
#
#
#
# files0 = os.listdir(input_dir)
# files = [n for n in files0 if n.endswith('freeresp.wav')]
# for i in files[:10]:
#     sample_rate, samples = wavfile.read(input_dir+i)
#     frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
#     plt.pcolormesh(times, frequencies, spectrogram)
#     plt.imshow(spectrogram)
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [sec]')
#     # plt.ylim(0,np.max(frequencies))
#
#     plt.savefig(output_dir+i[:-4]+'.png')
#
