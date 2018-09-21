#!/usr/bin/env python
import numpy as np

import lasagne
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import batch_norm_dnn as batch_norm
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
from lasagne.layers import FlattenLayer, GlobalPoolLayer, NonlinearityLayer, dropout_channels

from lasagne_wrapper.training_strategy import TrainingStrategy, \
    RefinementStrategy
from lasagne_wrapper.learn_rate_shedules import get_constant
from lasagne_wrapper.parameter_updates import get_update_adam
from lasagne_wrapper.batch_iterators import BatchIterator

from asc.utils.data_tut17 import SPEC_CONTEXT, prepare, prepare_random_slice

INI_LEARNING_RATE = np.float32(0.002)

BATCH_SIZE = 50
MAX_EPOCHS = 1000
PATIENCE = 20
L2 = None
SAMPLE_LENGTH = 100

INPUT_SHAPE = [3, SAMPLE_LENGTH, 206]

init_conv = lasagne.init.HeNormal


def get_batch_iterator():
    """
    Get batch iterator
    """

    def batch_iterator(batch_size, k_samples, shuffle):
        if shuffle:
            return BatchIterator(batch_size=batch_size, prepare=prepare_random_slice, k_samples=k_samples, shuffle=shuffle)
        else:
            return BatchIterator(batch_size=batch_size, prepare=prepare, k_samples=k_samples, shuffle=shuffle)

    return batch_iterator


def build_model(batch_size=BATCH_SIZE):
    """ Compile net architecture """
    nonlin = lasagne.nonlinearities.elu

    # --- input layers ---
    l_in = lasagne.layers.InputLayer(shape=(batch_size, INPUT_SHAPE[0], None, INPUT_SHAPE[2]), name='Input')

    # --- conv layers ---
    n_filt = 32
    net = Conv2DLayer(l_in, num_filters=n_filt, filter_size=5, stride=2, pad=2, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=n_filt, filter_size=3, stride=1, pad=1, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = MaxPool2DLayer(net, pool_size=2)
    net = dropout_channels(net, p=0.1)

    net = Conv2DLayer(net, num_filters=n_filt * 2, filter_size=3, stride=1, pad=1, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=n_filt * 2, filter_size=3, stride=1, pad=1, W=init_conv(), nonlinearity=nonlin)
    net = MaxPool2DLayer(net, pool_size=2)
    net = dropout_channels(net, p=0.1)

    net = Conv2DLayer(net, num_filters=n_filt * 4, filter_size=3, stride=1, pad=1, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=n_filt * 4, filter_size=3, stride=1, pad=1, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=n_filt * 4, filter_size=3, stride=1, pad=1, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=n_filt * 4, filter_size=3, stride=1, pad=1, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = MaxPool2DLayer(net, pool_size=2)
    net = dropout_channels(net, p=0.1)

    net = Conv2DLayer(net, num_filters=n_filt * 8, filter_size=3, pad=0, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = dropout_channels(net, p=0.3)
    net = Conv2DLayer(net, num_filters=n_filt * 8, filter_size=1, pad=0, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = dropout_channels(net, p=0.3)

    # --- feed forward part ---
    net = Conv2DLayer(net, num_filters=15, filter_size=1, W=init_conv(),
                      nonlinearity=nonlin)
    net = batch_norm(net)
    net = GlobalPoolLayer(net)
    net = FlattenLayer(net)
    net = NonlinearityLayer(net, nonlinearity=lasagne.nonlinearities.softmax)

    return net


def get_train_strategy():
    return TrainingStrategy(
        batch_size=BATCH_SIZE,
        ini_learning_rate=INI_LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        L2=L2,
        adapt_learn_rate=get_constant(),
        update_function=get_update_adam(),
        valid_batch_iter=get_batch_iterator(),
        train_batch_iter=get_batch_iterator(),
        best_model_by_accurary=True,
        refinement_strategy=RefinementStrategy(
            n_refinement_steps=9,
            refinement_patience=15,
            learn_rate_multiplier=0.5
        )
    )


train_strategy = get_train_strategy()
