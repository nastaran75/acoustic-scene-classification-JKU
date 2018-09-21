import librosa
import numpy as np
import os
import soundfile
import scipy.misc
import matplotlib.pyplot as plt

my_map = {'bus':0, 'cafe/restaurant':1, 'car':2, 'city_center':3, 'forest_path':4, 'grocery_store':5,
'home':6, 'beach':7, 'library':8, 'metro_station':9, 'office':10, 'residential_area':11, 
'train':12, 'tram':13, 'park':14}

def wave2spec(filename):
	data, samplerate = soundfile.read(filename)
	soundfile.write('new.wav', data, samplerate, subtype='PCM_16')
	n_fft = 2048 
	sr = 22050 
	y, sr = librosa.load('new.wav', sr=sr, mono=True) 
	stft = librosa.stft(y, n_fft=n_fft, hop_length=512, win_length=None, window='hann', center=True, pad_mode='reflect') 
	stft = np.abs(stft) 
	freqs = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft) 
	stft = librosa.perceptual_weighting(stft**2, freqs, ref=1.0, amin=1e-10, top_db=80.0) 
	spectrogram = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=149)
	return spectrogram


def find(name, path):
    # print path
    for file in os.listdir(path):
        # print file, name
        if name.endswith(file):
            return os.path.join(path, file)

wave_directory = 'DCASE-Wav-Files/'
spec_directory = 'TUT-Spec-Files/'

height = 128
width = 431

spec = wave2spec('a001_0_30.wav')
print spec.shape

# for dir_filename in os.listdir(wave_directory):
#     if dir_filename.endswith(".txt"):
#         path = os.path.join(wave_directory, dir_filename)
#         file = open(path)
#         num_data = sum(1 for line in file)
#         file = open(path)
#         images = np.empty(shape = [num_data,1,height,width])
#         labels = np.empty(num_data)
#         counter = 0
#         for iter,line in enumerate(file.readlines()): # first line of traininData.csv is header (only for trainingData.csv)
#             filename = line.split()[0]
#             label = line.split()[1]
#             image_folder = dir_filename[:-4]
#             for wave_dir in os.listdir(wave_directory):
#                 if(wave_dir.startswith('TUT')):
#                     for inner_wave_dir in os.listdir(os.path.join(wave_directory,wave_dir)):
#                         if(inner_wave_dir.startswith('TUT')):
#                             wave_file = find(filename, os.path.join(wave_directory, wave_dir, inner_wave_dir,'audio'))
#                             if(wave_file):
#                                 spectrogram = wave2spec(wave_file)
#                                 images[counter] = spectrogram
#                                 plt.imshow(images[counter].reshape(height,width))
#                                 plt.show()
#                                 labels[counter] = my_map[label]
#                                 counter += 1
#                                 break
#         print(str(counter))                       
#         # np.save(image_folder + '_images.npy',images)
#         # np.save(image_folder + '_labels.npy', labels)    

