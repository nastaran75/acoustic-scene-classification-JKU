
from __future__ import print_function

import os
import numpy as np
import argparse

from lasagne_wrapper.network import Network

from utils.data_litis import load_data as load_data_litis
from utils.data_litis import ID_CLASS_MAPPING as id_class_mapping_litis
from utils.data_tut17 import load_data as load_data_tut17
from utils.data_tut17 import load_data_test as load_data_tut17_test
from utils.data_tut17 import load_mfcc as load_mfcc_tut17
from utils.data_tut17 import ID_CLASS_MAPPING as id_class_mapping_tut17
from utils.data_jamendo import load_jamendo_tags
from utils.data_audio2cf import load_audio2cf
from config.settings import EXP_ROOT

# seed seed for consistency
np.random.seed(4711)


def select_model(model_path):
    """ select model and train function """

    model_str = os.path.basename(model_path)
    model_str = model_str.split('.py')[0]
    exec('from models import ' + model_str + ' as model')

    model.EXP_NAME = model_str
    return model


def load_data(data_set, fold, n_workers):
    if data_set == "litis":
        data = load_data_litis(fold=fold, n_workers=n_workers)
        id_class_mapping = id_class_mapping_litis
    elif data_set == "jamendo_tags":
        data = load_jamendo_tags(count_thresh=3000, n_seconds=20, n_workers=n_workers)
        tag_to_id = data['tag_to_id']
        id_class_mapping = dict(zip(tag_to_id.values(), tag_to_id.keys()))
    elif data_set == "jamendo156_tags":
        data = load_jamendo_tags(count_thresh=1000, n_seconds=20, n_workers=n_workers)
        tag_to_id = data['tag_to_id']
        id_class_mapping = dict(zip(tag_to_id.values(), tag_to_id.keys()))
    elif data_set == "audio2cf":
        data = load_audio2cf(n_seconds=15, n_workers=n_workers)
        id_class_mapping = None
    elif data_set == "tut17":
        data = load_data_tut17(fold=fold, n_workers=n_workers)
        id_class_mapping = id_class_mapping_tut17
    elif data_set == "tut17_test":
        data = load_data_tut17_test(n_workers=n_workers)
        id_class_mapping = id_class_mapping_tut17
    elif data_set == "tut17_mc":
        data = load_data_tut17(fold=fold, n_workers=n_workers, multi_channel=True)
        id_class_mapping = id_class_mapping_tut17
    elif data_set == "tut17_raw":
        data = load_data_tut17(fold=fold, n_workers=n_workers, raw_audio=True)
        id_class_mapping = id_class_mapping_tut17
    elif data_set == "tut17_mfcc":
        data = load_mfcc_tut17(fold=fold)
    elif data_set == "tut17_mfs":
        data = load_data_tut17(fold=fold, n_workers=n_workers, multi_framesize=True)
        id_class_mapping = id_class_mapping_tut17

    return data, id_class_mapping


def get_dump_file_paths(out_path, fold):
    par = 'params.pkl' if fold is None else 'params_%d.pkl' % fold
    log = 'results.pkl' if fold is None else 'results_%d.pkl' % fold
    dump_file = os.path.join(out_path, par)
    log_file = os.path.join(out_path, log)
    print("parameter dump file", dump_file)
    return dump_file, log_file


if __name__ == '__main__':
    """ main """
    # add argument parser
    parser = argparse.ArgumentParser(description='Train multi-modality model.')
    parser.add_argument('--model', help='select model to train.')
    parser.add_argument('--data', help='select model to train.')
    parser.add_argument('--fold', help='train split.', type=int, default=None)
    parser.add_argument('--n_workers', help='train split.', type=int, default=1)
    args = parser.parse_args()

    # set number of threads for spectrogram processing
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # select model
    model = select_model(args.model)

    # load data
    print("Loading data ...")
    data, _ = load_data(args.data, args.fold, args.n_workers)

    # set model dump file
    out_path = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME)
    dump_file, log_file = get_dump_file_paths(out_path, args.fold)

    # compile network
    net = model.build_model()

    # initialize neural network
    my_net = Network(net)

    # train network
    my_net.fit(data, model.train_strategy, log_file=log_file, dump_file=dump_file)
