# Illustration of the contents of the audio2cf dataset.
# Andreu Vall

from __future__ import print_function
import cPickle
import string
import os
import numpy as np

from asc.config.settings import DATA_ROOT_AUDIO2CF
from audio_processors import RandomAudioSliceProcessor
from data_pools import AugmentedAudioFileClassificationDataPool

np.set_printoptions(threshold=4)

SPEC_BINS = 149
SPEC_CONTEXT = 469


def load_audio2cf(n_seconds=15, n_workers=10):
    """ """

    # load list of valid songs
    song_info_file = os.path.join(DATA_ROOT_AUDIO2CF, 'noncorrupted.txt')
    with open(song_info_file, 'r') as f:
        song_ids = f.readlines()

    # load song information
    song_info_file = os.path.join(DATA_ROOT_AUDIO2CF, 'audio2cf.songs')
    with open(song_info_file, 'r') as f:
        song2track, song2artist, song2title = cPickle.load(f)
    
    # get the song's CF factors
    factor_file = os.path.join(DATA_ROOT_AUDIO2CF, 'audio2cf.factors')
    with open(factor_file, 'r') as f:
        factors = cPickle.load(f)

    # collect corresponding audio paths
    all_factors = []
    audio_paths = []
    for i_song, song in enumerate(song_ids):
        song = song.strip()
        
        # the song's track id gives the path to its audio file
        audio_path = os.path.join('audio',
                                  string.join(song2track[song][2:5], '/'),
                                  song2track[song] + '.mp3'
                                  )
        audio_path = os.path.join(DATA_ROOT_AUDIO2CF, audio_path)
        
        # if audio exists keep data
        if os.path.exists(audio_path):
            all_factors.append(factors[song])
            audio_paths.append(audio_path)

        # TODO: remove this
        if len(all_factors) >= 5000:
            break
    
    all_factors = np.asarray(all_factors, dtype=np.float32)
    audio_paths = np.asarray(audio_paths, dtype=np.object)
    print(len(all_factors), "songs collected.")

    # init split
    #s = int(0.9 * len(all_factors))
    n_test = 2000
    tag_count = all_factors.shape[1]

    # init audio processor
    audio_processor = RandomAudioSliceProcessor(n_seconds=n_seconds, num_target_frames=SPEC_CONTEXT)

    train_pool = AugmentedAudioFileClassificationDataPool(audio_paths[:-n_test], all_factors[:-n_test], audio_processor, n_workers,
                                                          shuffle=True, use_cache=True, n_classes=tag_count, target_type=np.float32)
    valid_pool = AugmentedAudioFileClassificationDataPool(audio_paths[-n_test:], all_factors[-n_test:], audio_processor, n_workers,
                                                          shuffle=True, use_cache=True, n_classes=tag_count, target_type=np.float32)
    test_pool = AugmentedAudioFileClassificationDataPool(audio_paths[-n_test:], all_factors[-n_test:], audio_processor, n_workers,
                                                         shuffle=True, use_cache=True, n_classes=tag_count, target_type=np.float32)

    print("train", train_pool.shape[0])
    print("valid", valid_pool.shape[0])
    print("test ", test_pool.shape[0])

    return {'train': train_pool, 'valid': valid_pool, 'test': test_pool}


if __name__ == '__main__':
    """ main """
    import time
    
    data = load_audio2cf(n_seconds=15, n_workers=2)
    
    def prepare(X, y):
        """ Prepare data for network processing """
        shape = X.shape
        X = X.reshape((shape[0], 1, shape[1], shape[2]))

        return X, y

    from lasagne_wrapper.batch_iterators import BatchIterator
    bi = BatchIterator(batch_size=10, prepare=prepare, k_samples=100, shuffle=True)

    for e in xrange(5):
        start = time.time()
        for i, (x, y) in enumerate(bi(data['train'])):
            stop = time.time()
            print(e, i, x.shape, y.shape, stop-start, "sec batch time")
            start = stop
