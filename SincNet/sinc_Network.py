import theano
import lasagne
from lasagne.layers import InputLayer, DenseLayer, batch_norm
from SincLayer import SincConv
import theano.tensor as T
from lasagne.nonlinearities import LeakyRectify

def build_cnn(data_len,input_var = None, division=1):

  length = data_len
  channel = 1
  dropout_ratio = .5
  filter_size = 3
  pool_length = 3
  stride = 1
  dropout_internal = .5
  num_filters = 32

  my_leaky_rectify = LeakyRectify(0.2)

  # input
  my_input = lasagne.layers.InputLayer(shape=(None, channel, length), input_var=input_var)

  my_input = lasagne.layers.StandardizationLayer(my_input)

  ################################################################
  #0
  conv0 = SincConv(my_input,fs=16000,N_filt=80,Filt_dim=251)
  # conv0 = T.abs(conv0)

  pool0 = lasagne.layers.MaxPool1DLayer(conv0, pool_size=3)

  std0 = lasagne.layers.StandardizationLayer(pool0)

  act0 =  lasagne.layers.NonlinearityLayer(std0,nonlinearity=my_leaky_rectify)
  ###########################################################################
  #1
  conv1 = lasagne.layers.Conv1DLayer(act0, num_filters=60, filter_size=5,
  W=lasagne.init.GlorotNormal())

  pool1 = lasagne.layers.MaxPool1DLayer(conv1, pool_size=3)

  std1 = lasagne.layers.StandardizationLayer(pool1)

  act1 =  lasagne.layers.NonlinearityLayer(std1,nonlinearity=my_leaky_rectify)
  ##############################################################################
  #2
  conv2 = lasagne.layers.Conv1DLayer(act1, num_filters=60, filter_size=5,
  W=lasagne.init.GlorotNormal())

  pool2 = lasagne.layers.MaxPool1DLayer(conv2, pool_size=3)

  std2 = lasagne.layers.StandardizationLayer(pool2)

  act2 =  lasagne.layers.NonlinearityLayer(std2,nonlinearity=my_leaky_rectify)
  ###########################################################################


  conv3 = batch_norm(lasagne.layers.DenseLayer(act2, num_units=2048, W=lasagne.init.GlorotNormal(), nonlinearity=my_leaky_rectify))

  conv4 = batch_norm(lasagne.layers.DenseLayer(conv3, num_units=2048, W=lasagne.init.GlorotNormal(), nonlinearity=my_leaky_rectify))

  conv5 = batch_norm(lasagne.layers.DenseLayer(conv4, num_units=2048, W=lasagne.init.GlorotNormal(), nonlinearity=my_leaky_rectify))

  network = lasagne.layers.DenseLayer(conv5, num_units=15,
  nonlinearity=lasagne.nonlinearities.softmax)

  # network = lasagne.layers.NonlinearityLayer(conv6,nonlinearity=lasagne.nonlinearities.softmax)

  return network