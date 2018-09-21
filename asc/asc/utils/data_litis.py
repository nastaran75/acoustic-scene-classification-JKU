
import os
import numpy as np

from asc.config.settings import DATA_ROOT_LITIS
from data_pools import AugmentedAudioFileClassificationDataPool
from audio_processors import processor


ID_CLASS_MAPPING = dict()
ID_CLASS_MAPPING[0] = "avion"
ID_CLASS_MAPPING[1] = "busystreet"
ID_CLASS_MAPPING[2] = "bus"
ID_CLASS_MAPPING[3] = "cafe"
ID_CLASS_MAPPING[4] = "car"
ID_CLASS_MAPPING[5] = "hallgare"
ID_CLASS_MAPPING[6] = "kidgame"
ID_CLASS_MAPPING[7] = "market"
ID_CLASS_MAPPING[8] = "metro-paris"
ID_CLASS_MAPPING[9] = "metro-rouen"
ID_CLASS_MAPPING[10] = "poolhall"
ID_CLASS_MAPPING[11] = "quietstreet"
ID_CLASS_MAPPING[12] = "hall"
ID_CLASS_MAPPING[13] = "restaurant"
ID_CLASS_MAPPING[14] = "ruepietonne"
ID_CLASS_MAPPING[15] = "shop"
ID_CLASS_MAPPING[16] = "train-ter"
ID_CLASS_MAPPING[17] = "train-tgv"
ID_CLASS_MAPPING[18] = "tubestation"

CLASS_ID_MAPPING = dict(zip(ID_CLASS_MAPPING.values(), ID_CLASS_MAPPING.keys()))

N_WORKERS = 10

SPEC_CONTEXT = 938
SPEC_BINS = 149


def load_id_to_file_mapping():
    """ load file name / id mapping """

    with open(os.path.join(DATA_ROOT_LITIS, 'relation_wav_examples.txt'), 'rb') as fp:
        lines = fp.readlines()

    id_to_file = dict()
    for line in lines:
        file_name, file_id = line.split()
        id_to_file[int(file_id)] = file_name

    return id_to_file


def parse_files(file, fold):
    """ parse files and labels for fold """

    id_to_file = load_id_to_file_mapping()

    with open(os.path.join(DATA_ROOT_LITIS, file), 'rb') as fp:
        lines = fp.readlines()

    file_paths = []
    targets = []
    for entry in lines[fold].split():
        file_id = int(float(entry))
        for class_name in CLASS_ID_MAPPING.keys():
            if class_name in id_to_file[file_id]:
                break

        file_path = os.path.join(DATA_ROOT_LITIS, "data_rouen", id_to_file[file_id])
        file_paths.append(file_path)
        targets.append(CLASS_ID_MAPPING[class_name])

    file_paths = np.asarray(file_paths, dtype=np.object)
    targets = np.asarray(targets, dtype=np.int32)

    return file_paths, targets


def load_data(fold=0, n_workers=N_WORKERS):
    """ load data """
    train_files, train_targets = parse_files('fold3026-matrices_Learn.txt', fold)
    test_files, test_targets = parse_files('fold3026-matrices_Test.txt', fold)

    train_pool = AugmentedAudioFileClassificationDataPool(train_files, train_targets, processor, n_workers=n_workers, shuffle=True)
    valid_pool = AugmentedAudioFileClassificationDataPool(test_files, test_targets, processor, n_workers=n_workers, shuffle=True)
    test_pool = AugmentedAudioFileClassificationDataPool(test_files, test_targets, processor, n_workers=n_workers, shuffle=True)

    return {'train': train_pool, 'valid': valid_pool, 'test': test_pool}


def prepare(X, y):
    """ Prepare data for network processing """
    shape = X.shape
    X = X.reshape((shape[0], 1, shape[1], shape[2]))

    return X, y


if __name__ == "__main__":
    """ main """
    import time
    import matplotlib.pyplot as plt
    from lasagne_wrapper.batch_iterators import BatchIterator, threaded_generator_from_iterator

    # load data
    data = load_data(n_workers=1)

    # init batch iterator
    bi = BatchIterator(batch_size=50, k_samples=None, shuffle=True, prepare=prepare)

    print "Train Files:", data['test'].shape

    # iterate train data
    for e in xrange(10):
        start_time = time.time()
        generator = threaded_generator_from_iterator(bi(data['test']))
        for i, (X, y) in enumerate(generator):
            print e, X.shape, y.shape

            # get timing
            end_time = time.time()
            print "%.2f seconds required for batch." % (end_time - start_time)
            start_time = end_time

            # # show data
            # plt.figure("sample data")
            # plt.imshow(X[0, 0].T, aspect='auto', origin='lower')
            # plt.show(block=True)
