import librosa
import numpy as np
import soundfile
import os
from scipy.io import wavfile
import sys

max_length = 661501

my_map = {'bus':0, 'cafe/restaurant':1, 'car':2, 'city_center':3, 'forest_path':4, 'grocery_store':5,
'home':6, 'beach':7, 'library':8, 'metro_station':9, 'office':10, 'residential_area':11, 
'train':12, 'tram':13, 'park':14}


def get_wave(filename):
	data, samplerate = soundfile.read(filename)
	soundfile.write('new.wav', data, samplerate, subtype='PCM_16')
	n_fft = 2048 
	fs = 22050 
	y, sr = librosa.load('new.wav', sr=fs, mono=True) 
	y = y.astype(np.float32)
	if len(y) > max_length:
		y = y[0:max_length]
	return y


def find(name, path):
    # print path
    for file in os.listdir(path):
        # print file, name
        if name.endswith(file):
            return os.path.join(path, file)




file = open(sys.argv[1])
num_data = sum(1 for line in file)
file = open(sys.argv[1])
images = np.empty(shape = [num_data,max_length])
labels = np.empty(num_data)
counter = 0
for iter,line in enumerate(file.readlines()): # first line of traininData.csv is header (only for trainingData.csv)
    filename = line.split()[0]
    label = line.split()[1]
    wave = get_wave(filename)
    # plt.imshow(spectrogram)
    # plt.show()
    images[counter] = wave
    labels[counter] = my_map[label]
    counter += 1
    print (counter)

print(str(counter))  
if num_data != counter:
    print('num_data and counter dont match')                     
np.save('waves/2016_wave_' + sys.argv[1][:-4] + '_images.npy',images)
np.save('waves/2016_wave_' + sys.argv[1][:-4] + '_labels.npy', labels)
