
'''


repository name: Voicebook
repository version: 1.0
repository link: https://github.com/jim-schwoebel/voicebook
author: Jim Schwoebel
author contact: js@neurolex.co
description: 🗣️ a book and repo to get you started programming voice applications in Python - 10 chapters and 200+ scripts.
license category: opensource
license: Apache 2.0 license
organization name: NeuroLex Laboratories, Inc.
location: Seattle, WA
website: https://neurolex.ai
release date: 2018-09-28

This code (voicebook) is hereby released under a Apache 2.0 license license.

For more information, check out the license terms below.

'''
import soundfile as sf
import os
import importlib
# import pandas as pd
# import numpy as np

import wave
import contextlib


def trim_silence(filename):
	new_filename=filename[0:-4]+'_trimmed.wav'
	command='sox %s %s silence -l 1 0.1 1'%(filename, new_filename)+"% -1 2.0 1%"
	os.system(command)
	return new_filename

# trim the leading and trailing silence => (test_trimmed.wav)


def segment_audio(filename, start, end):
	clip_duration=end-start
	new_filename=filename[0:-4]+'_segmented_'+str(start)+'_'+str(end)+'.wav'
	command='sox %s %s trim %s %s'%(filename,new_filename,str(start),str(clip_duration))
	os.system(command)
	return new_filename

# trim from second 30 to 40 => (test_trimmed_30_40.wav)

def audio_length(file):
    with contextlib.closing(wave.open(file,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration



def remove_noise(filename):
	'''
	TODO doesnt work well
	:param filename:
	:return:
	'''
	# now use sox to denoise using the noise profile
	data, samplerate = sf.read(filename)
	duration = data / samplerate
	first_data = samplerate / 10
	filter_data = list()
	for i in range(int(first_data)):
		filter_data.append(data[i])
	noisefile = 'noiseprof.wav'
	sf.write(noisefile, filter_data, samplerate)
	os.system('sox %s -n noiseprof noise.prof' % (noisefile))
	filename2 = 'tempfile.wav'
	filename3 = 'tempfile2.wav'
	noisereduction = "sox %s %s noisered noise.prof 0.21 " % (filename, filename2)
	command = noisereduction
	# run command
	os.system(command)
	print(command)
	# reduce silence again
	# os.system(silenceremove)
	# print(silenceremove)
	# rename and remove files
	os.remove(filename)
	os.rename(filename2, filename)
	# os.remove(filename2)
	os.remove(noisefile)
	os.remove('noise.prof')

	return filename



    # Create CSV files.







