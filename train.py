import PrepareDataset as PD
import theano
import lasagne
import theano.tensor as T
from theano import shared
import DCASE2017_Network 
from time import gmtime, strftime
import os
import sys
import numpy as np

log_dir = 'log/'
patience = 'False'

def train():
	num_epoches = 300
	batch_size = 100
	date = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
	model_name = date + '_' + dataset_name + '_model_' + str(fold) + '.npz'
	print date
	print("Loading data...")
	X_train, y_train, X_val, y_val, X_test, y_test = PD.load_dataset(dataset_name, fold)
	print X_train.shape
	height = X_train.shape[2]
	width = X_train.shape[3]
	channel = X_train.shape[1]
	
	#tensor variables for inputs and targets
	input_var = T.tensor4('inputs')
	target_var = T.ivector('targets')
	learning_rate = theano.shared(lasagne.utils.floatX(0.002), name='learning_rate')
	best_acc = 0
	max_patience = 20
	current_patience = 0
	my_train_buffer = np.empty(max_patience+2, dtype=str)
	my_validation_buffer = np.empty(max_patience+2, dtype=str)

	network = DCASE2017_Network.build_cnn(height,width,channel, input_var)

	#the final prediction of the network
	prediction = lasagne.layers.get_output(network)

	#define the loss
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()
	l2_loss = 1e-4 * lasagne.regularization.regularize_network_params(
	network, lasagne.regularization.l2, {'trainable': True})
	
	#computing the accuracy
	accuracy = T.mean(T.eq(T.argmax(prediction, axis=1),target_var), dtype=theano.config.floatX)

	#update the parameters based on SGD with nesterov momentum
	params = lasagne.layers.get_all_params(network, trainable = True)
	updates = lasagne.updates.nesterov_momentum(
		loss + l2_loss, params, learning_rate=learning_rate, momentum=0.9)



	#loss for validation/testing, note that deterministic=true means we disable droput layers for test/eval
	test_prediction = lasagne.layers.get_output(network, deterministic = True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
	test_loss = test_loss.mean()

	#computing the accuracy
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1),target_var), dtype=theano.config.floatX)

	#perform the training
	train_fn = theano.function([input_var, target_var], [loss,accuracy], updates = updates)

	l2_fn = theano.function([], l2_loss)

	#test and eval (no updates)
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

	
	train_log = log_dir + date+ 'fold_' + str(fold) +'_training_log.txt'
	validation_log = log_dir + date +  'fold_' + str(fold) +'_validation_log.txt'
	test_log = log_dir + date +  'fold_' + str(fold) +'_test_log.txt'
	best_train_log = log_dir + date+ 'best_fold_' + str(fold) +'_training_log.txt'
	best_validation_log = log_dir + date +  'best_fold_' + str(fold) +'_validation_log.txt'


	#iterate over epoches
	
	for epoch in range(num_epoches):
		# print " epoch:\t{}".format(epoch)
		#iterate over the whole training set in each epoch
		# if epoch>0 and epoch%5==0:
		# 	learning_rate.set_value(learning_rate.get_value() * lasagne.utils.floatX(0.5))
		train_err = 0
		train_batches = 0
		train_acc = 0.0
		for batch in PD.iterate_minibatches(X_train, y_train, batch_size, shuffle = True):
			# print " train_batches:\t{}".format(train_batches)
			inputs, targets = batch
			train_error,train_accuracy = train_fn(inputs, targets)
			train_err += train_error
			train_acc += train_accuracy
			train_batches += 1

		l2_loss = l2_fn()
		#iterate over the validation set
		val_err = 0
		val_acc = 0.0
		val_batches = 0
		for batch in PD.iterate_minibatches(X_val, y_val, batch_size, shuffle = False):
			# print " val_batches:\t{}".format(val_batches)
			inputs, targets = batch
			val_error, val_accuracy = val_fn(inputs, targets)
			val_err += val_error
			val_acc += val_accuracy
			val_batches += 1

		#training is over, compute the test loss
		test_err = 0
		test_acc = 0
		test_batches = 0

		# X_test, y_test = PD.load_testset(dataset_name)
		for batch in PD.iterate_minibatches(X_test, y_test, batch_size, shuffle = False):
			# print " test_batches:\t{}".format(test_batches)
			inputs, targets = batch
			test_error, test_accuracy = val_fn(inputs, targets)
			test_err += test_error
			test_acc += test_accuracy
			test_batches += 1

	
		my_train_buffer[current_patience] = "Epoch {} of {} fold: {} training_loss: {:.6f} \n".format(epoch, num_epoches,fold,train_err/ train_batches)
		my_validation_buffer[current_patience] = "Epoch {} of {} fold: {} validation_loss: {:.6f} validation_accuracy: {:.2f} %\n".format(epoch, num_epoches,fold, val_err/ val_batches,val_acc/val_batches*100)
		# counter += 1

		if patience=='True':
			#updating the best model
			epoch_acc = val_acc/val_batches*100
			if(epoch_acc > best_acc):
				print 'best model updated...'

				best_acc = epoch_acc
				#update the best model
				np.savez(model_name, *lasagne.layers.get_all_param_values(network))
				# for i in range(current_patience+1):
				# 	with open(best_train_log, 'a') as text_file:
				# 		text_file.write(my_train_buffer[i])
				# 	with open(best_validation_log, 'a') as text_file:
				# 		text_file.write(my_validation_buffer[i])
				# my_train_buffer = np.empty(max_patience+2, dtype=str)
				# my_validation_buffer = np.empty(max_patience+2, dtype=str)
				current_patience = 0
					
			else:
				print 'current_patience increased to: ' + str(current_patience)
				current_patience += 1
				if(current_patience>max_patience):
					print 'resumed from best model...'
					# my_train_buffer = np.empty(max_patience+2, dtype=str)
					# my_validation_buffer = np.empty(max_patience+2, dtype=str)
					if(learning_rate.get_value()>0.000016):
						learning_rate.set_value(learning_rate.get_value()*lasagne.utils.floatX(0.5))
					current_patience = 0
					with np.load(model_name) as f:
						# print f['arr_0'].shape
						param_values = [f['arr_%d' % i] for i in range(len(f.files))]
						# print len(param_values)

					lasagne.layers.set_all_param_values(network, param_values)




		print " Epoch {} of {} \n".format(epoch + 1, num_epoches)
		print " training loss:\t{:.6f}".format(train_err/ train_batches)
		print "  L2 loss:      \t{:.6f}".format(l2_loss)
		print " training accuracy:\t{:.2f} %\n".format(train_acc/train_batches*100)
		print " validation loss:\t{:.6f}".format(val_err/ val_batches)
		print " validation accuracy:\t{:.2f} %\n".format(val_acc/val_batches*100)
		print " test loss:\t{:.6f}".format(test_err/test_batches)
		print " test accuracy:\t{:.2f} %".format(test_acc/test_batches *100)
		print "---------------------------------------------------------------------------------------"
		with open(train_log, 'a') as text_file:
			text_file.write("Epoch {} of {} fold: {} training_loss: {:.6f} \n".format(epoch, num_epoches,fold,train_err/ train_batches))
		with open(validation_log, 'a') as text_file:
			text_file.write("Epoch {} of {} fold: {} validation_loss: {:.6f} validation_accuracy: {:.2f} %\n".format(epoch, num_epoches,fold, val_err/ val_batches,val_acc/val_batches*100))

	if patience=='True':
		print 'loaded the best model for testing...'
		with np.load(model_name) as f:
		# print f['arr_0'].shape
			param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		# print len(param_values)
		lasagne.layers.set_all_param_values(network, param_values)
	#training is over, compute the test loss
	test_err = 0
	test_acc = 0
	test_batches = 0

	# X_test, y_test = PD.load_testset(dataset_name)
	for batch in PD.iterate_minibatches(X_test, y_test, batch_size, shuffle = False):
		# print " test_batches:\t{}".format(test_batches)
		inputs, targets = batch
		test_error, test_accuracy = val_fn(inputs, targets)
		test_err += test_error
		test_acc += test_accuracy
		test_batches += 1

	print "final results:"
	print " test loss:\t{:.6f}".format(test_err/test_batches)
	print " test accuracy:\t{:.2f} %".format(test_acc/test_batches *100)
	with open(test_log, 'a') as text_file:
		text_file.write(" Epoch {} of {} fold: {} test loss: {:.6f} test accuracy: {:.2f} \n%".format(epoch, num_epoches,fold,test_err/ test_batches,test_acc/test_batches*100))


dataset_name = sys.argv[1]
fold = sys.argv[2]
patience = sys.argv[3]

train()