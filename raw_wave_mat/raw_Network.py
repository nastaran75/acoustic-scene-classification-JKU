import theano
import lasagne
from lasagne.layers import InputLayer, DenseLayer, batch_norm

def build_cnn(data_len,input_var = None, division=1):

	length = data_len
	channel = 1
	dropout_ratio = .5
	filter_size = 3
	pool_length = 3
	stride = 1
	dropout_internal = .5
	num_filters = 32

	# input
	my_input = lasagne.layers.InputLayer(shape=(None, channel, length), input_var=input_var)

	################################################################
	#0
	conv0 = batch_norm(lasagne.layers.Conv1DLayer(my_input, num_filters=num_filters, filter_size=25,
	 nonlinearity=lasagne.nonlinearities.leaky_rectify,pad=0, stride=5, W=lasagne.init.HeNormal()))
	###########################################################################
	#1
	conv1 = batch_norm(lasagne.layers.Conv1DLayer(conv0, num_filters=num_filters*2, filter_size=5,
	nonlinearity=lasagne.nonlinearities.leaky_rectify, pad=0, stride=3, W=lasagne.init.HeNormal()))	

	# max pool
	pool1 = lasagne.layers.MaxPool1DLayer(conv1, pool_size=2)

	# pool1 = lasagne.layers.GaussianNoiseLayer(pool1, sigma=0.1)

	# pool1 = lasagne.layers.dropout(pool1, p=dropout_internal)

	################################################################

	#2
	conv2 = batch_norm(lasagne.layers.Conv1DLayer(pool1, num_filters=num_filters*4, filter_size=3,
	nonlinearity=lasagne.nonlinearities.leaky_rectify, pad=0, stride=3, W=lasagne.init.HeNormal()))

	#2
	conv2 = batch_norm(lasagne.layers.Conv1DLayer(conv2, num_filters=num_filters*4, filter_size=3,
	nonlinearity=lasagne.nonlinearities.leaky_rectify, pad=0, stride=1, W=lasagne.init.HeNormal()))


	# max pool
	pool2 = lasagne.layers.MaxPool1DLayer(conv2, pool_size=2)

	# pool2 = lasagne.layers.GaussianNoiseLayer(pool2, sigma=0.1)

	################################################################

	#3
	conv3 = batch_norm(lasagne.layers.Conv1DLayer(pool2, num_filters=num_filters*8, filter_size=5,
	nonlinearity=lasagne.nonlinearities.leaky_rectify, pad=0, stride=3, W=lasagne.init.HeNormal()))

	#3
	conv3 = batch_norm(lasagne.layers.Conv1DLayer(conv3, num_filters=num_filters*8, filter_size=5,
	nonlinearity=lasagne.nonlinearities.leaky_rectify, pad=0, stride=3, W=lasagne.init.HeNormal()))

	# max pool
	pool3 = lasagne.layers.MaxPool1DLayer(conv3, pool_size=2)

	# pool3 = lasagne.layers.GaussianNoiseLayer(pool3, sigma=0.1)

	# pool3 = lasagne.layers.dropout(pool3, p=dropout_internal)

	####################################################################

	#4
	conv4 = batch_norm(lasagne.layers.Conv1DLayer(pool3, num_filters=num_filters*16, filter_size=3,
	nonlinearity=lasagne.nonlinearities.leaky_rectify, pad=0, stride=1, W=lasagne.init.HeNormal()))

	#4
	conv4 = batch_norm(lasagne.layers.Conv1DLayer(conv4, num_filters=num_filters*16, filter_size=3,
	nonlinearity=lasagne.nonlinearities.leaky_rectify, pad=0, stride=1, W=lasagne.init.HeNormal()))

	# max pool
	pool4 = lasagne.layers.MaxPool1DLayer(conv4, pool_size=2)

	# pool4 = lasagne.layers.GaussianNoiseLayer(pool4, sigma=0.1)

	####################################################################

	#4
	conv5 = batch_norm(lasagne.layers.Conv1DLayer(pool4, num_filters=num_filters*32, filter_size=5,
	nonlinearity=lasagne.nonlinearities.leaky_rectify, pad=0, stride=1, W=lasagne.init.HeNormal()))

	#4
	conv5 = batch_norm(lasagne.layers.Conv1DLayer(conv5, num_filters=num_filters*32, filter_size=5,
	nonlinearity=lasagne.nonlinearities.leaky_rectify, pad=0, stride=1, W=lasagne.init.HeNormal()))

	# max pool
	pool5 = lasagne.layers.MaxPool1DLayer(conv5, pool_size=2)

	# #5
	# conv5 = batch_norm(lasagne.layers.Conv1DLayer(pool4, num_filters=num_filters*2, filter_size=filter_size,
	# nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=stride, W=lasagne.init.HeNormal()))

	# # max pool
	# pool5 = lasagne.layers.MaxPool1DLayer(conv5, pool_size=pool_length)

	# pool5 = lasagne.layers.GaussianNoiseLayer(pool5, sigma=0.1)

	# pool5 = lasagne.layers.dropout(pool5, p=dropout_internal)

	####################################################################

	#6
	# conv6 = batch_norm(lasagne.layers.Conv1DLayer(pool5, num_filters=num_filters*2, filter_size=filter_size,
	# nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=stride, W=lasagne.init.HeNormal()))

	# # max pool
	# pool6 = lasagne.layers.MaxPool1DLayer(conv6, pool_size=pool_length)

	# pool6 = lasagne.layers.GaussianNoiseLayer(pool6, sigma=0.1)

	####################################################################

	# #7
	# conv7 = batch_norm(lasagne.layers.Conv1DLayer(pool6, num_filters=num_filters*2, filter_size=filter_size,
	# nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=stride, W=lasagne.init.HeNormal()))

	# # max pool
	# pool7 = lasagne.layers.MaxPool1DLayer(conv7, pool_size=pool_length)

	# pool7 = lasagne.layers.GaussianNoiseLayer(pool7, sigma=0.1)

	# pool7 = lasagne.layers.dropout(pool7, p=dropout_internal)

	####################################################################

	# #8
	# conv8 = batch_norm(lasagne.layers.Conv1DLayer(pool7, num_filters=num_filters*2, filter_size=filter_size,
	# nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=stride, W=lasagne.init.HeNormal()))

	# # max pool
	# pool8 = lasagne.layers.MaxPool1DLayer(conv8, pool_size=pool_length)

	# pool8 = lasagne.layers.GaussianNoiseLayer(pool8, sigma=0.1)

	####################################################################

	# #9
	# conv9 = batch_norm(lasagne.layers.Conv1DLayer(pool8, num_filters=num_filters*4, filter_size=filter_size,
	# nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=stride, W=lasagne.init.HeNormal()))

	# # max pool
	# pool9 = lasagne.layers.MaxPool1DLayer(conv9, pool_size=pool_length)

	# pool9 = lasagne.layers.GaussianNoiseLayer(pool9, sigma=0.1)

	####################################################################

	# #10
	# conv10 = batch_norm(lasagne.layers.Conv1DLayer(pool9, num_filters=256, filter_size=filter_size,
	# nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=1, W=lasagne.init.HeNormal()))

	# # max pool
	# pool10 = lasagne.layers.MaxPool1DLayer(conv10, pool_size=pool_length)

	# ####################################################################
	# #11
	# conv11 = batch_norm(lasagne.layers.Conv1DLayer(pool10, num_filters=256, filter_size=filter_size,
	# nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=1, W=lasagne.init.HeNormal()))

	# # max pool
	# pool11 = lasagne.layers.MaxPool1DLayer(conv11, pool_size=pool_length)

	####################################################################
	# #13
	# conv13 = batch_norm(lasagne.layers.Conv1DLayer(pool9, num_filters=256, filter_size=filter_size,
	# nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=1, W=lasagne.init.HeNormal()))

	# # max pool
	# pool13 = lasagne.layers.MaxPool1DLayer(conv13, pool_size=pool_length)
	# #############################################################################

	# #14
	# conv14 = batch_norm(lasagne.layers.Conv1DLayer(pool9, num_filters=512, filter_size=filter_size,
	# nonlinearity=lasagne.nonlinearities.rectify, pad='same', stride=1, W=lasagne.init.HeNormal()))

	# # max pool
	# pool14 = lasagne.layers.MaxPool1DLayer(conv14, pool_size=pool_length)
	# ######################################################################

	#15
	conv15 = batch_norm(lasagne.layers.Conv1DLayer(pool5, num_filters=15, filter_size=1,
	nonlinearity=lasagne.nonlinearities.leaky_rectify, W=lasagne.init.HeNormal()))

	conv15 = lasagne.layers.GlobalPoolLayer(conv15)


	# dropout1 = lasagne.layers.dropout(conv15, p=dropout_ratio)

	####################################################################

	flattened = lasagne.layers.FlattenLayer(conv15)

	#####################################################################

	network = lasagne.layers.NonlinearityLayer(flattened,nonlinearity=lasagne.nonlinearities.softmax)

	return network