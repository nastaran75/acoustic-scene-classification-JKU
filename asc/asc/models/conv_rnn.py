#!/usr/bin/env python
import numpy as np

import lasagne
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import batch_norm_dnn as batch_norm
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
from lasagne.layers import DropoutLayer, FlattenLayer, ReshapeLayer, LSTMLayer, DenseLayer, ElemwiseSumLayer
from lasagne.layers.recurrent import Gate
from lasagne import init

from lasagne_wrapper.training_strategy import TrainingStrategy, RefinementStrategy
from lasagne_wrapper.learn_rate_shedules import get_constant
from lasagne_wrapper.parameter_updates import get_update_adam
from lasagne_wrapper.batch_iterators import BatchIterator

from asc.utils.data_litis import SPEC_BINS

INI_LEARNING_RATE = np.float32(0.002)

BATCH_SIZE = 10
MAX_EPOCHS = 1000
PATIENCE = 25
L2 = 0.0002

N_STEPS = 10
INPUT_SHAPE = [N_STEPS, 1, 100, SPEC_BINS]

init_conv = init.HeNormal
gain = np.sqrt(2.0)


def prepare(X, y):

    shape = X.shape
    X = X.reshape((shape[0], 1, shape[1], shape[2]))

    max_shift = INPUT_SHAPE[3] // 4

    # sample steps from audio recording
    steps = np.linspace(0, X.shape[2] - INPUT_SHAPE[2], N_STEPS)
    steps = np.floor(steps).astype(np.int)

    # slightly shift samples (data augmentation)
    for i in xrange(N_STEPS):
        start_idx = steps[i] + np.random.randint(-max_shift, max_shift)
        steps[i] = np.clip(start_idx, 0, X.shape[2] - INPUT_SHAPE[2])

    # init new data matrix
    X_new = np.zeros([X.shape[0]] + INPUT_SHAPE, dtype=np.float32)

    # collect samples
    for i in xrange(X.shape[0]):

        sample = np.zeros(INPUT_SHAPE, dtype=np.float32)
        for j, start_idx in enumerate(steps):
            sample[j] = X[i, :, start_idx:start_idx + INPUT_SHAPE[2], :]

        X_new[i] = sample

    return X_new, y


def get_rnn_batch_iterator():
    """
    Image segmentation batch iterator which randomly flips images (and mask) left and right
    """

    def batch_iterator(batch_size, k_samples, shuffle):
        return BatchIterator(batch_size=batch_size, prepare=prepare, k_samples=k_samples, shuffle=shuffle)

    return batch_iterator


def build_model(batch_size=BATCH_SIZE):
    """ Compile net architecture """
    nonlin = lasagne.nonlinearities.rectify

    # --- input layers ---
    in_shape = (batch_size, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], INPUT_SHAPE[3])
    l_in = lasagne.layers.InputLayer(shape=in_shape, name='Input')
    net = l_in

    # --- reshape for convolutions ---
    net = ReshapeLayer(net, shape=(batch_size*N_STEPS, INPUT_SHAPE[1], INPUT_SHAPE[2], INPUT_SHAPE[3]))

    # --- conv layers ---
    net = Conv2DLayer(net, num_filters=32, filter_size=5, stride=2, pad=2, W=init_conv(gain=gain), nonlinearity=nonlin)
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=32, filter_size=3, stride=1, pad=1, W=init_conv(gain=gain), nonlinearity=nonlin)
    net = batch_norm(net)
    net = MaxPool2DLayer(net, pool_size=2)
    net = DropoutLayer(net, p=0.3)

    net = Conv2DLayer(net, num_filters=64, filter_size=3, stride=1, pad=1, W=init_conv(gain=gain), nonlinearity=nonlin)
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=64, filter_size=3, stride=1, pad=1, W=init_conv(gain=gain), nonlinearity=nonlin)
    net = MaxPool2DLayer(net, pool_size=2)
    net = DropoutLayer(net, p=0.3)

    net = Conv2DLayer(net, num_filters=128, filter_size=3, stride=1, pad=1, W=init_conv(gain=gain), nonlinearity=nonlin)
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=128, filter_size=3, stride=1, pad=1, W=init_conv(gain=gain), nonlinearity=nonlin)
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=128, filter_size=3, stride=1, pad=1, W=init_conv(gain=gain), nonlinearity=nonlin)
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=128, filter_size=3, stride=1, pad=1, W=init_conv(gain=gain), nonlinearity=nonlin)
    net = batch_norm(net)
    net = MaxPool2DLayer(net, pool_size=2)
    net = DropoutLayer(net, p=0.3)

    net = Conv2DLayer(net, num_filters=256, filter_size=3, pad=0, W=init_conv(gain=gain), nonlinearity=nonlin)
    net = batch_norm(net)
    net = DropoutLayer(net, p=0.5)
    net = Conv2DLayer(net, num_filters=256, filter_size=1, pad=0, W=init_conv(gain=gain), nonlinearity=nonlin)
    net = batch_norm(net)
    net = DropoutLayer(net, p=0.5)

    # --- feed forward part ---
    net = Conv2DLayer(net, num_filters=19, filter_size=1, W=init_conv(gain=gain), nonlinearity=nonlin)
    net = batch_norm(net)

    # --- reshape to rnn format ---
    net = FlattenLayer(net)
    net = ReshapeLayer(net, (batch_size, N_STEPS, -1))

    # --- rnn part ---

    # initialize
    in_gate = Gate(W_in=init.Normal(0.01), W_hid=init.Normal(0.01), W_cell=init.Normal(0.01), b=init.Constant(0.))
    forget_gate = Gate(W_in=init.Normal(0.01), W_hid=init.Normal(0.01), W_cell=init.Normal(0.01), b=init.Constant(0.))
    out_gate = Gate(W_in=init.Normal(0.01), W_hid=init.Normal(0.01), W_cell=init.Normal(0.01), b=init.Constant(0.))
    cell_init = init_conv(gain=0.0)
    hid_init = init.Constant(0.)

    net_fwd = LSTMLayer(net, num_units=256, only_return_final=True,
                        ingate=in_gate, forgetgate=forget_gate, outgate=out_gate,
                        cell_init=cell_init, hid_init=hid_init)
    net_bwd = LSTMLayer(net, num_units=256, only_return_final=True, backwards=True,
                        ingate=in_gate, forgetgate=forget_gate, outgate=out_gate,
                        cell_init=cell_init, hid_init=hid_init)
    net = ElemwiseSumLayer((net_fwd, net_bwd))

    net = DenseLayer(net, num_units=19, nonlinearity=lasagne.nonlinearities.softmax)

    return net


refinement_strategy = RefinementStrategy(n_refinement_steps=10, refinement_patience=PATIENCE, learn_rate_multiplier=0.5)


train_strategy = TrainingStrategy(
    batch_size=BATCH_SIZE,
    ini_learning_rate=INI_LEARNING_RATE,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
    L2=L2,
    refinement_strategy=refinement_strategy,
    adapt_learn_rate=get_constant(),
    update_function=get_update_adam(),
    valid_batch_iter=get_rnn_batch_iterator(),
    train_batch_iter=get_rnn_batch_iterator()
)
