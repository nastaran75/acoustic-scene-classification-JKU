#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Creates a DenseNet model in Lasagne, following the paper
"Densely Connected Convolutional Networks"
by Gao Huang, Zhuang Liu, Kilian Q. Weinberger, 2016.
https://arxiv.org/abs/1608.06993

This closely follows the authors' Torch implementation.
See densenet_fast.py for a faster formulation.

Author: Jan Schlüter
"""

import lasagne
from lasagne.layers import (InputLayer, Conv2DLayer, ConcatLayer, DenseLayer,
                            DropoutLayer, Pool2DLayer, GlobalPoolLayer,
                            NonlinearityLayer,RecurrentLayer)
from lasagne.nonlinearities import rectify, softmax,leaky_rectify,very_leaky_rectify,tanh
try:
    from lasagne.layers.dnn import BatchNormDNNLayer as BatchNormLayer
except ImportError:
    from lasagne.layers import BatchNormLayer


def build_densenet(input_shape=(None, 1, 1, 66150), input_var=None, classes=15,
                   depth=36, first_output=16, growth_rate=12, num_blocks=5,
                   dropout=0,nonL='rectify'):
    

    if (depth - 1) % num_blocks != 0:
        raise ValueError("depth must be num_blocks * n + 1 for some n")

    # input and initial convolution
    network = InputLayer(input_shape, input_var, name='input')
    network = Conv2DLayer(network, first_output, filter_size=(1,5), pad='same', stride=(1,2),
                          W=lasagne.init.HeNormal(gain='relu'),
                          b=None, nonlinearity=None, name='pre_conv')
    # note: The authors' implementation does *not* have a dropout after the
    #       initial convolution. This was missing in the paper, but important.
    # if dropout:
    #     network = DropoutLayer(network, dropout)
    # dense blocks with transitions in between
    n = (depth - 1) // num_blocks
    for b in range(num_blocks):
        print 'dense block number: ' + str(b)
        network = dense_block(network, 4, growth_rate, dropout,
                              name_prefix='block%d' % (b + 1),nonL=nonL)
        if b < num_blocks - 1:
          if b==0 or b==1:
            filter_size=(1,5)
            pad='same'
          else:
            filter_size=(1,2)
            pad='valid'

          if b==0:
            conv_stride = (1,1)
          else:
            conv_stride=(1,2)
          print 'transition layer number : ' + str(b)
          network = transition(network, dropout,filter_size=filter_size,
                                 name_prefix='block%d_trs' % (b + 1),conv_stride=conv_stride,pad=pad)
    # post processing until prediction
    # network = BatchNormLayer(network, name='post_bn')
    nonlinearity=rectify

    if nonL=='leaky_rectify':
      nonlinearity=leaky_rectify

    elif nonL=='very_leaky_rectify':
      nonlinearity=very_leaky_rectify

    # gate_parameters=lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(),
    #   W_hid=lasagne.init.Orthogonal())

    # cell_parameters=lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(),
    #   W_hid=lasagne.init.Orthogonal(),W_cell=None,
    #   nonlinearity=lasagne.nonlinearities.tanh)

    # N_HIDDEN = 256
    # lstm=lasagne.layers.recurrent.LSTMLayer(network,N_HIDDEN,ingate=gate_parameters,forgetgate=gate_parameters,
    #   cell=cell_parameters,outgate=gate_parameters,learn_init=True,grad_clipping=100)

    # lstm_back=lasagne.layers.recurrent.LSTMLayer(network,N_HIDDEN,ingate=gate_parameters,forgetgate=gate_parameters,
    #   cell=cell_parameters,outgate=gate_parameters,learn_init=True,grad_clipping=100,backwards=True)

    # network = lasagne.layers.ElemwiseSumLayer([lstm,lstm_back])

    # network = lasagne.layers.ReshapeLayer(network,(-1,N_HIDDEN))

    # network = DenseLayer(network, classes, nonlinearity=softmax,
                         # W=lasagne.init.HeNormal(gain=1), name='output')


    # l_forward = lasagne.layers.RecurrentLayer(
    #     network, 256, grad_clipping=100,
    #     W_in_to_hid=lasagne.init.HeUniform(),
    #     W_hid_to_hid=lasagne.init.HeUniform(),
    #     nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)

    # l_backward = lasagne.layers.RecurrentLayer(
    #     network, 256, grad_clipping=100,
    #     W_in_to_hid=lasagne.init.HeUniform(),
    #     W_hid_to_hid=lasagne.init.HeUniform(),
    #     nonlinearity=lasagne.nonlinearities.tanh,
    #     only_return_final=True, backwards=True)

    # network = lasagne.layers.ConcatLayer([l_forward, l_backward])

    # network = Conv2DLayer(network, 15, filter_size=(1,1), pad='same', stride=(1,1),
    #                       W=lasagne.init.HeNormal(gain='relu'),
    #                       b=None, nonlinearity='rectify', name='pre_conv')
    # # # network = NonlinearityLayer(network, nonlinearity=nonlinearity,
    # # #                             name='post_relu')
    network = GlobalPoolLayer(network, name='post_pool')
    # network = lasagne.layers.NonlinearityLayer(network,nonlinearity=lasagne.nonlinearities.softmax)
    network = DenseLayer(network, classes, nonlinearity=softmax,
                         W=lasagne.init.HeNormal(gain=1), name='output')
    return network


def dense_block(network, num_layers, growth_rate, dropout, name_prefix,nonL='rectify'):
    # concatenated 3x3 convolutions
    for n in range(num_layers):
        print 'dense layer : ' + str(n)
        conv = bn_relu_conv(network, channels=growth_rate,
                            filter_size=(1,3), dropout=dropout,
                            name_prefix=name_prefix + '_l%02d' % (n + 1),nonL=nonL)
        network = ConcatLayer([network, conv], axis=1,
                              name=name_prefix + '_l%02d_join' % (n + 1))
    return network


def transition(network, dropout,filter_size, name_prefix,conv_stride=(1,2),pool_size=(1,2),pad='valid',nonL='rectify'):
    # a transition 1x1 convolution followed by avg-pooling
    network = bn_relu_conv(network, channels=network.output_shape[1],
                           filter_size=filter_size, dropout=dropout,
                           name_prefix=name_prefix, stride=conv_stride,pad=pad,nonL=nonL)
    network = Pool2DLayer(network, pool_size=pool_size, stride=(1,2), mode='average_inc_pad',
                          name=name_prefix + '_pool')
    return network


def bn_relu_conv(network, channels, filter_size, dropout, name_prefix,stride=1,pad='same',nonL='rectify'):
    network = BatchNormLayer(network, name=name_prefix + '_bn')
    nonlinearity=rectify

    if nonL=='leaky_rectify':
      nonlinearity=leaky_rectify

    elif nonL=='very_leaky_rectify':
      nonlinearity=very_leaky_rectify

    network = NonlinearityLayer(network, nonlinearity=nonlinearity,
                                name=name_prefix + '_relu')
    # pad = 'valid'
    # if filter_size==(1,5):
    #   pad = 'same'
    # print pad
    network = Conv2DLayer(network, channels, filter_size, pad=pad,
                          W=lasagne.init.HeNormal(gain='relu'),
                          b=None, nonlinearity=None,
                          name=name_prefix + '_conv',stride=stride)
    if dropout:
        network = DropoutLayer(network, dropout)
    return network


# class DenseNetInit(lasagne.init.Initializer):
#     """
#     Reproduces the initialization scheme of the authors' Torch implementation.
#     At least for the 40-layer networks, lasagne.init.HeNormal works just as
#     fine, though. Kept here just in case. If you want to swap in this scheme,
#     replace all W= arguments in all the code above with W=DenseNetInit().
#     """
#     def sample(self, shape):
#         import numpy as np
#         rng = lasagne.random.get_rng()
#         if len(shape) >= 4:
#             # convolutions use Gaussians with stddev of sqrt(2/fan_out), see
#             # https://github.com/liuzhuang13/DenseNet/blob/cbb6bff/densenet.lua#L85-L86
#             # and https://github.com/facebook/fb.resnet.torch/issues/106
#             fan_out = shape[0] * np.prod(shape[2:])
#             W = rng.normal(0, np.sqrt(2. / fan_out),
#                            size=shape)
#         elif len(shape) == 2:
#             # the dense layer uses Uniform of range sqrt(1/fan_in), see
#             # https://github.com/torch/nn/blob/651103f/Linear.lua#L21-L43
#             fan_in = shape[0]
#             W = rng.uniform(-np.sqrt(1. / fan_in), np.sqrt(1. / fan_in),
#                             size=shape)
#         return lasagne.utils.floatX(W)
