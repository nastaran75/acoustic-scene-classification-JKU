import librosa
import numpy as np
import os
import soundfile
import scipy.misc
import sys

my_map = {'bus':0, 'cafe/restaurant':1, 'car':2, 'city_center':3, 'forest_path':4, 'grocery_store':5,
'home':6, 'beach':7, 'library':8, 'metro_station':9, 'office':10, 'residential_area':11, 
'train':12, 'tram':13, 'park':14}

def wave2spec(filename):
	data, samplerate = soundfile.read(filename)
	soundfile.write('new.wav', data, samplerate, subtype='PCM_16')
	n_fft = 2048 
	sr = 22050 
	y, sr = librosa.load('new.wav', sr=sr, mono=True)  
	stft = librosa.stft(y, n_fft=n_fft, hop_length=2048, win_length=None, window='hann', center=True, pad_mode='reflect') 
	stft = np.abs(stft) 
	freqs = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft) 
	stft = librosa.perceptual_weighting(stft**2, freqs, ref=1.0, amin=1e-10, top_db=80.0) 
	spectrogram = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=128)
	return spectrogram


def find(name, path):
    # print path
    for file in os.listdir(path):
        # print file, name
        if name.endswith(file):
            return os.path.join(path, file)

wave_directory = '.'

height = 128
width = 323


file = open(sys.argv[1])
num_data = sum(1 for line in file)
file = open(sys.argv[1])
images = np.empty(shape = [num_data,1,height,width])
labels = np.empty(num_data)
counter = 0
for iter,line in enumerate(file.readlines()): # first line of traininData.csv is header (only for trainingData.csv)
    filename = line.split()[0]
    label = line.split()[1]
    spectrogram = wave2spec(filename)
    # plt.imshow(spectrogram)
    # plt.show()
    images[counter] = spectrogram
    labels[counter] = my_map[label]
    counter += 1
    print (counter)

print(str(counter))  
if num_data != counter:
    print('num_data and counter dont match')                     
np.save(sys.argv[1][:-4]+ '_2016_images.npy',images)
np.save(sys.argv[1][:-4] + '_2016_labels.npy', labels)



