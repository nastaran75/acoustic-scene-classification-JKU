#!/usr/bin/env python
import numpy as np

import theano.tensor as T

import lasagne
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import batch_norm_dnn as batch_norm
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
from lasagne.layers import DropoutLayer, FlattenLayer, GlobalPoolLayer, NonlinearityLayer

# from lasagne_wrapper.training_strategy import TrainingStrategy
# from lasagne_wrapper.learn_rate_shedules import get_constant
# from lasagne_wrapper.parameter_updates import get_update_adam
# from lasagne_wrapper.batch_iterators import BatchIterator

INI_LEARNING_RATE = np.float32(0.01)

BATCH_SIZE = 16
MAX_EPOCHS = 1000
PATIENCE = 20
L2 = 0.00001

INPUT_SHAPE = [220501]

init_conv = lasagne.init.HeNormal


def prepare(X, y):
    return X, y


def get_batch_iterator():
    """
    Get batch iterator
    """

    def batch_iterator(batch_size, k_samples, shuffle):
        return BatchIterator(batch_size=batch_size, prepare=None, k_samples=k_samples, shuffle=shuffle)

    return batch_iterator


def build_model(batch_size=BATCH_SIZE):
    """ Compile net architecture """
    nonlin = lasagne.nonlinearities.rectify

    # --- input layers ---
    l_in = lasagne.layers.InputLayer(shape=(batch_size, INPUT_SHAPE[0]), name='Input')

    # reshape to convolution format
    net = l_in
    net = lasagne.layers.ReshapeLayer(net, shape=(-1, 1, INPUT_SHAPE[0], 1))

    # --- convolution layers ---
    net = Conv2DLayer(net, num_filters=32, filter_size=(25, 1), stride=(5, 1), pad=0, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=64, filter_size=(5, 1), stride=(3, 1), pad=0, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = MaxPool2DLayer(net, pool_size=(2, 1))
    net = DropoutLayer(net, p=0.0)

    net = Conv2DLayer(net, num_filters=128, filter_size=(3, 1), stride=(3, 1), pad=0, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=128, filter_size=(3, 1), stride=(1, 1), pad=0, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = MaxPool2DLayer(net, pool_size=(2, 1))
    net = DropoutLayer(net, p=0.0)

    net = Conv2DLayer(net, num_filters=256, filter_size=(5, 1), stride=(3, 1), pad=0, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=256, filter_size=(5, 1), stride=(3, 1), pad=0, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = MaxPool2DLayer(net, pool_size=(2, 1))
    net = DropoutLayer(net, p=0.0)

    net = Conv2DLayer(net, num_filters=512, filter_size=(3, 1), stride=(1, 1), pad=0, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=512, filter_size=(3, 1), stride=(1, 1), pad=0, W=init_conv(), nonlinearity=nonlin)
    net = batch_norm(net)
    net = MaxPool2DLayer(net, pool_size=(2, 1))
    net = DropoutLayer(net, p=0.0)

    # --- feed forward part ---
    net = Conv2DLayer(net, num_filters=15, filter_size=1, W=init_conv(),
                      nonlinearity=nonlin)
    net = batch_norm(net)
    net = GlobalPoolLayer(net)
    net = FlattenLayer(net)
    net = NonlinearityLayer(net, nonlinearity=lasagne.nonlinearities.softmax)

    return net

# from lasagne_wrapper.training_strategy import RefinementStrategy

# def get_train_strategy():
#     return TrainingStrategy(
#         batch_size=BATCH_SIZE,
#         ini_learning_rate=INI_LEARNING_RATE,
#         max_epochs=MAX_EPOCHS,
#         patience=PATIENCE,
#         refinement_strategy=RefinementStrategy(n_refinement_steps=5, refinement_patience=20, learn_rate_multiplier=0.5),
#         L2=L2,
#         adapt_learn_rate=get_constant(),
#         update_function=get_update_adam(),
#         valid_batch_iter=get_batch_iterator(),
#         train_batch_iter=get_batch_iterator(),
#         best_model_by_accurary=True)

# train_strategy = get_train_strategy()
