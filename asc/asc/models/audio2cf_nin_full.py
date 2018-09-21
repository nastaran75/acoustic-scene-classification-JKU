#!/usr/bin/env python
import numpy as np

import theano.tensor as T

import lasagne
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import batch_norm_dnn as batch_norm
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
from lasagne.layers import DropoutLayer, GlobalPoolLayer

from lasagne_wrapper.training_strategy import TrainingStrategy, RefinementStrategy
from lasagne_wrapper.learn_rate_shedules import get_constant
from lasagne_wrapper.parameter_updates import get_update_adam
from lasagne_wrapper.batch_iterators import BatchIterator
from lasagne_wrapper.optimization_objectives import mean_squared_error

from asc.utils.data_audio2cf import SPEC_BINS, SPEC_CONTEXT

INI_LEARNING_RATE = np.float32(0.01)

BATCH_SIZE = 10
MAX_EPOCHS = 1000000
PATIENCE = 20
L2 = None
EPOCH_SIZE = 1000

INPUT_SHAPE = [1, SPEC_CONTEXT, SPEC_BINS]

init_conv = lasagne.init.HeNormal


def prepare(X, y):
    """ Prepare data for network processing """
    shape = X.shape
    X = X.reshape((shape[0], 1, shape[1], shape[2]))

    return X, y


def get_batch_iterator():
    """
    Get batch iterator
    """

    def batch_iterator(batch_size, k_samples, shuffle):
        return BatchIterator(batch_size=batch_size, prepare=prepare, k_samples=k_samples, shuffle=shuffle)

    return batch_iterator


def build_model(batch_size=BATCH_SIZE):
    """ Compile net architecture """
    nonlin = lasagne.nonlinearities.rectify

    # --- input layers ---
    l_in = lasagne.layers.InputLayer(shape=(batch_size, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]), name='Input')

    # --- conv layers ---
    net = Conv2DLayer(l_in, num_filters=32, filter_size=5, stride=2, pad=2, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=32, filter_size=3, stride=1, pad=1, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = MaxPool2DLayer(net, pool_size=2)
    net = DropoutLayer(net, p=0.0)

    net = Conv2DLayer(net, num_filters=64, filter_size=3, stride=1, pad=1, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=64, filter_size=3, stride=1, pad=1, W=init_conv(), nonlinearity=nonlin)
    net = MaxPool2DLayer(net, pool_size=2)
    net = DropoutLayer(net, p=0.0)

    net = Conv2DLayer(net, num_filters=256, filter_size=3, stride=1, pad=1, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=256, filter_size=3, stride=1, pad=1, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=256, filter_size=3, stride=1, pad=1, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=256, filter_size=3, stride=1, pad=1, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = MaxPool2DLayer(net, pool_size=2)
    net = DropoutLayer(net, p=0.3)

    net = Conv2DLayer(net, num_filters=512, filter_size=3, pad=0, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = DropoutLayer(net, p=0.0)
    net = Conv2DLayer(net, num_filters=512, filter_size=1, pad=0, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = DropoutLayer(net, p=0.0)

    # --- feed forward part ---
    net = Conv2DLayer(net, num_filters=200, filter_size=1, W=init_conv(), nonlinearity=lasagne.nonlinearities.identity)
    net = batch_norm(net, name='l1')
    net = GlobalPoolLayer(net, name='l0')

    return net


def cosine_distance_loss(pred, targ):
    """ cosine distance loss """

    # normalize to length 1.0
    pred = pred / pred.norm(2, axis=1).reshape((pred.shape[0], 1))
    targ = targ / targ.norm(2, axis=1).reshape((targ.shape[0], 1))

    cosine_distances = 1.0 - (pred * targ).sum(axis=1)

    return cosine_distances.mean()


train_strategy = TrainingStrategy(
    batch_size=BATCH_SIZE,
    ini_learning_rate=INI_LEARNING_RATE,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
    L2=L2,
    samples_per_epoch=EPOCH_SIZE,
    y_tensor_type=T.matrix,
    objective=cosine_distance_loss,
    adapt_learn_rate=get_constant(),
    refinement_strategy=RefinementStrategy(n_refinement_steps=5, refinement_patience=5, learn_rate_multiplier=0.5),
    update_function=get_update_adam(),
    valid_batch_iter=get_batch_iterator(),
    train_batch_iter=get_batch_iterator()
)
