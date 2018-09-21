import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import PIL.Image as Image
import os
import soundfile

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = int(np.ceil((len(samples) - frameSize) / float(hopSize)) + 1)
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    
    return np.fft.rfft(frames)    
    
""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=22050, factor=24, alpha=1.0, f0=0.9, fmax=1):
    spec = spec[:, 0:256]
    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins) #** factor
    
    # http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310
    scale = np.array(map(lambda x: x * alpha if x <= f0 else (fmax-alpha*f0)/(fmax-f0)*(x-f0)+alpha*f0, scale))
    scale *= (freqbins-1)/max(scale)

    newspec = np.complex128(np.zeros([timebins, freqbins]))
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = [0.0 for i in range(freqbins)]
    totw = [0.0 for i in range(freqbins)]
    for i in range(0, freqbins):
        if (i < 1 or i + 1 >= freqbins):
            newspec[:, i] += spec[:, i]
            freqs[i] += allfreqs[i]
            totw[i] += 1.0
            continue
        else:
            # scale[15] = 17.2
            w_up = scale[i] - np.floor(scale[i])
            w_down = 1 - w_up
            j = int(np.floor(scale[i]))
           
            newspec[:, j] += w_down * spec[:, i]
            freqs[j] += w_down * allfreqs[i]
            totw[j] += w_down
            
            newspec[:, j + 1] += w_up * spec[:, i]
            freqs[j + 1] += w_up * allfreqs[i]
            totw[j + 1] += w_up
    
    for i in range(len(freqs)):
        if (totw[i] > 1e-6):
            freqs[i] /= totw[i]
    
    return newspec, freqs

""" plot spectrogram"""
def plotstft(audiopath, binsize=2**11, plotpath=None, colormap="gray", channel=0, name='tmp.png', alpha=1, offset=0):
    samplerate, samples = wav.read(audiopath)
    samples = samples[:, channel]
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1, sr=samplerate, alpha=alpha)
    sshow = sshow[2:, :]
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
    timebins, freqbins = np.shape(ims)
    
    ims = np.transpose(ims)
    ims = ims[0:256, offset:offset+768] # 0-11khz, ~9s interval
    #print "ims.shape", ims.shape

    image = Image.fromarray(ims) 
    image = image.convert('L')
    image.save(name)


def find(name, path):
    # print path
    for file in os.listdir(path):
        # print file, name
        if name.endswith(file):
            print 'hi'
            return os.path.join(path, file)

# wave_directory = 'DCASE-Wav-Files/'
# spec_directory = 'TUT-Spec-Files/'
# for dir_filename in os.listdir(wave_directory):
#     if dir_filename.endswith("fold4_train.txt"):
#         path = os.path.join(wave_directory, dir_filename)
#         file = open(path)
#         for iter,line in enumerate(file.readlines()): # first line of traininData.csv is header (only for trainingData.csv)
#             filename = line.split()[0];
#             for wave_dir in os.listdir(wave_directory):
#                 if(wave_dir.startswith('TUT')):
#                     for inner_wave_dir in os.listdir(os.path.join(wave_directory,wave_dir)):
#                         if(inner_wave_dir.startswith('TUT')):
#                             wave_file = find(filename, os.path.join(wave_directory, wave_dir, inner_wave_dir,'audio'))
#                             if(wave_file):
#                                 print wave_file
#                                 data, samplerate = soundfile.read(wave_file)
#                                 soundfile.write('new.wav', data, samplerate, subtype='PCM_16')
#                                 alpha = 1.0
#                                 offset = 45
#                                 plotstft('new.wav', channel=1, name=os.path.join(spec_directory,dir_filename[:-4],filename[6:-4]+'.png'),
#                                   alpha=alpha, offset=offset)
#                                 break
data, samplerate = soundfile.read('/home/nastaran/ASC/git/Acoustic_Scene_Analysis/DCASE/a001_10_20.wav')
soundfile.write('new.wav', data, samplerate, subtype='PCM_16')        
plotstft('new.wav', channel=1, name= 'tmp.png',
                                  )