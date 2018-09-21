import SCNN_PrepareDataset as PD
import theano
import lasagne
import theano.tensor as T
from theano import shared
import SCNN_Network as Network
from time import gmtime, strftime
import os
import numpy as np
import sys

log_dir = 'log/'
division = 3
fold = 1

my_map = ['bus', 'cafe/restaurant', 'car', 'city_center', 'forest_path', 'grocery_store',
'home', 'beach', 'library', 'metro_station', 'office', 'residential_area', 
'train', 'tram', 'park']

def test():
	batch_size = 6 * division
	num_classes = 15
	
	#tensor variables for inputs and targets
	input_var = T.tensor3('inputs')
	target_var = T.ivector('targets')

	network = Network.build_cnn(input_var,division)

	#loss for validation/testing, note that deterministic=true means we disable droput layers for test/eval
	test_prediction = lasagne.layers.get_output(network, deterministic = True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
	test_loss = test_loss.mean()

	#computing the accuracy
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1),target_var), dtype=theano.config.floatX)

	#test and eval (no updates)
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_prediction])

	print("Loading data...")
	X_test, y_test = PD.load_testset(division)


	#training is over, compute the test loss
	
	test_pred = np.zeros([X_test.shape[0]/division,num_classes])
	test_correct_pred = np.zeros(X_test.shape[0]/division)
	visit = np.zeros(X_test.shape[0]/division)

	for fold_number in range(1,fold+1):
		with np.load('model_'+ str(fold_number)+ '.npz') as f:
			# print f['arr_0'].shape
			param_values = [f['arr_%d' % i] for i in range(len(f.files))]
			# print len(param_values)

		lasagne.layers.set_all_param_values(network, param_values)
		test_err = 0
		test_acc = 0
		test_batches = 0

		for batch in PD.iterate_minibatches(X_test, y_test, batch_size, shuffle = False):
			print " test_batches:\t{}".format(test_batches)
			inputs, targets, indices = batch
			test_error, test_accuracy, prediction= val_fn(inputs, targets)
			test_err += test_error
			test_acc += test_accuracy
			test_batches += 1
			for i in range(len(indices)):
				visit[indices[i]/division] += 1
				test_pred[indices[i]/division] = np.maximum(prediction[i],test_pred[indices[i]/division])
				test_correct_pred[indices[i]/division] = targets[i]

		print "final results on fold " + str(fold_number) + ":"
		print " test loss:\t{:.6f}".format(test_err/test_batches)
		print " test accuracy:\t{:.2f} %".format(test_acc/test_batches *100)
		

	for v in visit:
		if v != fold*division:
			print v
	test_predicted_label = np.argmax(test_pred, axis=1)
	real_test_accuracy = np.mean(np.equal(np.argmax(test_pred, axis=1),test_correct_pred))

	print "final results:"
	print " real test accuracy:\t{:.2f} %".format(real_test_accuracy*100)

division = int(sys.argv[2])
fold = int(sys.argv[1])
test()