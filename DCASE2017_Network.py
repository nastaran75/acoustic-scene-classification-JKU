import theano
import lasagne

def build_cnn(height,width,channel,input_var = None):

	# height = 128
	# width = 323
	# channel = 1
	dropout_ratio = .3

	#the input layer
	network = lasagne.layers.InputLayer(shape=(None, channel, height, width), input_var=input_var)


	#the 1st convolution layer
	network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5,5),
	nonlinearity=lasagne.nonlinearities.rectify, pad=2, stride=2, W=lasagne.init.HeNormal())

	# the 2nd convolution layer
	network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3,3),
	nonlinearity=lasagne.nonlinearities.rectify, pad=1, stride=1, W=lasagne.init.HeNormal())


	#the first maxpooling layer
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2,2))

	#the first dropout layer
	network = lasagne.layers.dropout(network, p=dropout_ratio)




	#the 3rd convolution layer
	network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(3,3),
	nonlinearity=lasagne.nonlinearities.rectify, pad=1, stride=1, W=lasagne.init.HeNormal())

	#the 4th convolution layer
	network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(3,3),
	nonlinearity=lasagne.nonlinearities.rectify, pad=1, stride=1, W=lasagne.init.HeNormal())

	#the second maxpooling layer
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2,2))

	#the second dropout layer
	network = lasagne.layers.dropout(network, p=dropout_ratio)




	#the 5th convolution layer
	network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(3,3),
	nonlinearity=lasagne.nonlinearities.rectify, pad=1, stride=1, W=lasagne.init.HeNormal())

	#the 6th convolution layer
	network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(3,3),
	nonlinearity=lasagne.nonlinearities.rectify, pad=1, stride=1, W=lasagne.init.HeNormal())

	#the 7th convolution layer
	network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(3,3),
	nonlinearity=lasagne.nonlinearities.rectify, pad=1, stride=1, W=lasagne.init.HeNormal())

	#the 8th convolution layer
	network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(3,3),
	nonlinearity=lasagne.nonlinearities.rectify, pad=1, stride=1, W=lasagne.init.HeNormal())

	#the 3rd maxpooling layer
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2,2))

	#the 3rd dropout layer
	network = lasagne.layers.dropout(network, p=dropout_ratio)





	#the 9th convolution layer
	network = lasagne.layers.Conv2DLayer(network, num_filters=256, filter_size=(3,3),
	nonlinearity=lasagne.nonlinearities.rectify, pad=0, stride=1, W=lasagne.init.HeNormal())

	#the 4th dropout layer
	network = lasagne.layers.dropout(network, p=0.5)


	#the 9th convolution layer
	network = lasagne.layers.Conv2DLayer(network, num_filters=256, filter_size=(1,1),
	nonlinearity=lasagne.nonlinearities.rectify, pad=0, stride=1, W=lasagne.init.HeNormal())


	#the 5th dropout layer
	network = lasagne.layers.dropout(network, p=0.5)


	#the 9th convolution layer
	network = lasagne.layers.Conv2DLayer(network, num_filters=15, filter_size=(1,1),
	nonlinearity=lasagne.nonlinearities.rectify, pad=0, stride=1, W=lasagne.init.HeNormal())

	network = lasagne.layers.GlobalPoolLayer(network)



	#the final output layer
	network = lasagne.layers.NonlinearityLayer(network,nonlinearity=lasagne.nonlinearities.softmax)

	return network