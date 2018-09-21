import raw_PrepareDataset as PD
import theano
import lasagne
import theano.tensor as T
from theano import shared
import raw_Network as Network
from time import gmtime, strftime
import os
import numpy as np
import sys
# import progress
import background as bg
import parser 
from helpers import print_net_architecture,onehoter
from data_tut17 import load_data as load_data_tut17

log_dir = 'log/'
# division = 1
# fold = 1
# patience='False'
# representation = 'Mid'
# augment = True

my_map = ['bus', 'cafe/restaurant', 'car', 'city_center', 'forest_path', 'grocery_store',
'home', 'beach', 'library', 'metro_station', 'office', 'residential_area', 
'train', 'tram', 'park']

num_classes = 15



def train(dataset_name,fold,division,augment,patience,representation,batch_size,epochs,learning_rate):
	def iterate(name,data,label,num_replicates,batch_size,shuffle=False):
		assert(num_replicates==division)
		train_err = 0
		train_acc = 0.0
		train_batches = 0
		visit = np.zeros(data.shape[0])
		train_pred = np.zeros([data.shape[0],num_classes])
		train_voting_pred = np.zeros([data.shape[0],num_classes])
		train_correct_pred = np.zeros(data.shape[0])

		noise=False
		mixup=False
		val=True
		if name=='train':
			noise=True
			mixup=False
			val=True

		batches = PD.iterate_minibatches(data, label,num_samples=division,division=division,batch_size=batch_size,
		shuffle=shuffle,mixup=mixup,noise=noise,val=val)
		batches = bg.generate_in_background(batches)
		# print sum(1 for x in batches)
		# if augment and name=='train':
		# 	batches = PD.augment_minibatches(batches)
		# 	batches = bg.generate_in_background(batches)
			# print sum(1 for x in batches)

		for batch_cnt,batch in enumerate(batches):
			# print " train_batches:\t{}".format(train_batches)
			inputs, targets, indices = batch
			
			if name == 'train':
				train_error, train_accuracy, prediction = train_fn(inputs, targets)
			else :
				train_error, train_accuracy, prediction = val_fn(inputs, targets)
			train_err += train_error
			train_acc += train_accuracy
			train_batches += 1

			if name!='train':
				for i in range(len(targets)):
					train_pred[indices[i]/num_replicates] += prediction[i]
					train_voting_pred[indices[i]/num_replicates][np.argmax(prediction[i])] += 1
					train_correct_pred[indices[i]/num_replicates] = np.argmax(targets[i])
					visit[indices[i]/num_replicates] += 1

		train_predicted_label = np.argmax(train_pred, axis=1)
		if name!='train':
			real_train_accuracy = np.mean(np.equal(np.argmax(train_pred, axis=1),train_correct_pred))
			voting_train_accuracy = np.mean(np.equal(np.argmax(train_voting_pred, axis=1),train_correct_pred))
		# l2_loss = l2_fn()
		
		print name + " loss:\t{:.6f}".format(train_err/ train_batches)
		# print " l2 loss:\t{:.6f}".format(l2_loss)
		print name + " accuracy:\t{:.2f} %".format(train_acc/train_batches *100)
		if name=='train':
			print '\n'
		if name!='train':
			print "real " + name + " accuracy:\t{:.2f} %\n".format(real_train_accuracy *100)

		return float(train_err/ train_batches),float(train_acc/train_batches *100)
		

	date = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
	model_name = 'models/' + date + '_' + dataset_name + '_wave_model_' + str(fold) \
	 + '_' + str(division) + '_' + str(patience) + '_' + representation + '_' + str(augment) + '.npz'
	num_classes = 15
	print date
	print "Loading data..."
	# data = load_data_tut17(fold=fold, n_workers=1, raw_audio=True)
	# print data['train'][0][0].shape
	# for i in data['train']:
	# 	print i
	# X_train = np.zeros([data['train'].shape[0],1,661501],dtype = np.float32)
	# y_train = np.zeros(data['train'].shape[0],dtype = np.int32)
	# X_val = np.zeros([data['valid'].shape[0],1,661501],dtype = np.float32)
	# y_val = np.zeros(data['valid'].shape[0],dtype = np.int32)
	# X_test = np.zeros([data['test'].shape[0],1,661501],dtype = np.float32)
	# y_test = np.zeros(data['test'].shape[0],dtype = np.int32)
	# print data['train'].shape
	# # i = 0
	# for i in range(data['train'].shape[0]):
	# 	X_train[i] = data['train'][i][0]
	# 	y_train[i] = data['train'][i][1]
	# 	# print i
	# 	# i += 1
	# print 'Loaded train dataset...'
	# for i in range(data['valid'].shape[0]):
	# 	X_val[i] = data['valid'][i][0]
	# 	y_val[i] = data['valid'][i][1]
	# 	# print i
	# 	# i += 1
	# print 'Loaded Validation dataset...'
	# for i in range(data['test'].shape[0]):
	# 	X_test[i] = data['test'][i][0]
	# 	y_test[i] = data['test'][i][1]
		# print i
		# i += 1
	# X_train =  data['train'][:,0]
	# print X_train
	# print y_train
	# print 'Loaded test dataset...'

	# X_train = data['train']

	# y_train = data['train']['y']
	# X_val = data['valid'].X
	# y_val = data['valid'].targets
	# X_test = data['test'].files
	# y_test = data['test'].targets
	# print X_train
	# [X, y] = data['valid']
	# X_test = data['test']
	X_train, y_train, X_val, y_val, X_test,y_test, num_replicates = PD.load_dataset(dataset_name,fold,division,representation)
	# num_train_replicates = num_replicates
	# num_val_replicates = num_replicates
	# num_test_replicates = num_replicates
	assert (num_replicates==division)

	print X_train.shape,X_val.shape,X_test.shape
	# X_train = (X_train - np.mean(X_train)) / np.std(X_train)
	# X_val = (X_val - np.mean(X_val)) / np.std(X_val)
	# X_test = (X_test - np.mean(X_test)) / np.std(X_test)

	
	print "Instantiating network..."
	#tensor variables for inputs and targets
	input_var = T.tensor3('inputs')
	target_var = T.fmatrix('targets')
	# min_learning_rate = 0.000016
	best_acc = 0
	max_patience = 20
	# decrease_factor = 0.2
	current_patience = 0
	my_train_buffer = np.empty(max_patience+2, dtype=str)
	my_validation_buffer = np.empty(max_patience+2, dtype=str)

	network = Network.build_cnn(X_train.shape[1]/division,input_var = input_var)
	print str(X_train.shape[1]/division)
	print_net_architecture(network,detailed=True)
	print "%d layers with weights, %d parameters" % (sum(hasattr(l, 'W') for l in lasagne.layers.get_all_layers(network)),
		lasagne.layers.count_params(network, trainable=True))

	print "Compiling training function..."
	#the final prediction of the network
	prediction = lasagne.layers.get_output(network)

	#define the loss
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()
	# l2_loss = 1e-4 * lasagne.regularization.regularize_network_params(
	# network, lasagne.regularization.l2, {'trainable': True})

	learning_rate = theano.shared(lasagne.utils.floatX(learning_rate), name='learning_rate')
	
	#computing the accuracy
	train_acc = T.mean(T.eq(T.argmax(prediction, axis=1),T.argmax(target_var,axis=1)), dtype=theano.config.floatX)

	#update the parameters based on SGD with nesterov momentum
	params = lasagne.layers.get_all_params(network, trainable = True)
	updates = lasagne.updates.nesterov_momentum(
		loss, params, learning_rate=learning_rate, momentum=0.9)

	#perform the training
	train_fn = theano.function([input_var, target_var], [loss, train_acc, prediction], updates = updates)
	# l2_fn = theano.function([], l2_loss)

	print "Compiling testing function..."
	#loss for validation/testing, note that deterministic=true means we disable droput layers for test/eval
	test_prediction = lasagne.layers.get_output(network, deterministic = True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
	test_loss = test_loss.mean()

	#computing the accuracy
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1),T.argmax(target_var,axis=1)), dtype=theano.config.floatX)
	#test and eval (no updates)
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_prediction])


	
	train_log = log_dir + date+ 'fold_' + str(fold) +'_training_log.txt'
	validation_log = log_dir + date +  'fold_' + str(fold) +'_validation_log.txt'
	test_log = log_dir + date +  'fold_' + str(fold) +'_test_log.txt'
	best_train_log = log_dir + date+ 'best_fold_' + str(fold) +'_training_log.txt'
	best_validation_log = log_dir + date +  'best_fold_' + str(fold) +'_validation_log.txt'


	#iterate over epoches
	print 'training SampleCNN on the dataset ' + dataset_name +  ' divided into ' + str(division) + " segments and fold number: " + str(fold) + " learning_rate = "+ str(learning_rate.get_value()) + " patience = " + str(patience) + 'augmentation = ' + str(augment) + ' representation = ' + representation
	
	# best_model = lasagne.layers.get_all_param_values(network)
	
	for epoch in range(epochs):
		print " Epoch {} of {} \n".format(epoch + 1, epochs)
		# num_replicates = 1
		train_err,train_acc = iterate('train', X_train,y_train,num_replicates,batch_size,shuffle=True)
		val_err, val_acc = iterate('validation', X_val,y_val,num_replicates,batch_size,shuffle=False)
		test_err, test_acc = iterate('test', X_test,y_test,num_replicates,batch_size,shuffle=False)
		#iterate over the whole training set in each epoch
		# train_err = 0
		# train_acc = 0.0
		# train_batches = 0
		# visit = np.zeros(X_train.shape[0]/num_train_replicates)
		# train_pred = np.zeros([X_train.shape[0]/num_train_replicates,num_classes])
		# train_voting_pred = np.zeros([X_train.shape[0]/num_train_replicates,num_classes])
		# train_correct_pred = np.zeros(X_train.shape[0]/num_train_replicates)

		# batches = PD.iterate_minibatches(X_train, y_train, batch_size,
		# shuffle=True)
		# # print sum(1 for x in batches)
		# if augment:
		# 	batches = PD.augment_minibatches(batches)
		# 	batches = bg.generate_in_background(batches)
		# 	# print sum(1 for x in batches)


		# # batches = progress.progress(batches, desc='Epoch %d/%d, Batch ' % (epoch + 1, epochs),
		# # 	total=train_batches)

		# for batch_cnt,batch in enumerate(batches):
		# 	# print " train_batches:\t{}".format(train_batches)
		# 	print str(batch_cnt)
		# 	inputs, targets, indices = batch
			
		# 	train_error, train_accuracy, prediction = train_fn(inputs, targets)
		# 	train_err += train_error
		# 	train_acc += train_accuracy
		# 	train_batches += 1

		# 	for i in range(len(targets)):
		# 		train_pred[indices[i]/num_train_replicates] += prediction[i]
		# 		train_voting_pred[indices[i]/num_train_replicates][np.argmax(prediction[i])] += 1
		# 		train_correct_pred[indices[i]/num_train_replicates] = targets[i]
		# 		visit[indices[i]/num_train_replicates] += 1
		# 		# print visit[indices[i]/3]

		# 	# if batch_cnt%50==0:
		# 	# 	print 'batch_cnt = ' + str(batch_cnt)
		# 	# 	print " training loss so far :\t{:.6f}".format(train_err/ train_batches)
		# 	# 	print " l2 loss so far:\t{:.6f}".format(l2_loss)
		# 	# 	print " train accuracy so far:\t{:.2f} %".format(train_acc/train_batches *100)
		# 	# 	print " real train accuracy so far:\t{:.2f} %\n".format(real_train_accuracy *100)

		# # for v in visit:
		# # 	if v!=division:
		# # 		print '-----------Error occured in training segmentation....................'
		# # 		print v
		# train_predicted_label = np.argmax(train_pred, axis=1)
		# real_train_accuracy = np.mean(np.equal(np.argmax(train_pred, axis=1),train_correct_pred))
		# voting_train_accuracy = np.mean(np.equal(np.argmax(train_voting_pred, axis=1),train_correct_pred))
		# # l2_loss = l2_fn()
		
		# print " training loss:\t{:.6f}".format(train_err/ train_batches)
		# # print " l2 loss:\t{:.6f}".format(l2_loss)
		# print " train accuracy:\t{:.2f} %".format(train_acc/train_batches *100)
		# print " real train accuracy:\t{:.2f} %\n".format(real_train_accuracy *100)
		# # print " voting train accuracy:\t{:.2f} %\n".format(voting_train_accuracy *100)

		# with open('train_predicted_label.txt', 'w') as text_file:	
		# 	text_file.write('predicted\tground_truth\n')
		# 	total=np.zeros([num_classes])
		# 	correct = np.zeros([num_classes])
		# 	for i in range(len(train_pred)):
		# 		# print int(test_predicted_label[i]),int(test_correct_pred[i])
		# 		# print my_map[int(test_predicted_label[i])],my_map[int(test_correct_pred[i])]
		# 		if train_predicted_label[i] == train_correct_pred[i]:
		# 			correct[int(train_correct_pred[i])] += 1
		# 		total[int(train_correct_pred[i])] += 1
		# 		text_file.write(my_map[int(train_predicted_label[i])] + '\t' + my_map[int(train_correct_pred[i])] + '\n')
			# print 'Class-wise Results : '
			# for i in range(num_classes):
			# 	print my_map[i] + ' correct = ' + str(correct[i]) + ' total = '+ str(total[i]) + ' acc = ' + str(correct[i]/total[i]*100)


		#iterate over the validation set
		# val_err = 0
		# val_acc = 0.0
		# val_batches = 0
		# visit = np.zeros(X_val.shape[0]/num_val_replicates)
		# val_pred = np.zeros([X_val.shape[0]/num_val_replicates,num_classes])
		# val_voting_pred = np.zeros([X_val.shape[0]/num_val_replicates,num_classes])
		# val_correct_pred = np.zeros(X_val.shape[0]/num_val_replicates)

		# for batch in PD.iterate_minibatches(X_val, y_val, batch_size, shuffle = False):
		# 	# print " val_batches:\t{}".format(val_batches)
		# 	inputs, targets, indices= batch
		# 	val_error, val_accuracy, prediction = val_fn(inputs, targets)
		# 	val_err += val_error
		# 	val_acc += val_accuracy
		# 	val_batches += 1

		# 	for i in range(len(targets)):
		# 		visit[indices[i]/num_val_replicates] += 1
		# 		val_pred[indices[i]/num_val_replicates] += prediction[i]
		# 		val_voting_pred[indices[i]/num_val_replicates][np.argmax(prediction[i])] += 1
		# 		val_correct_pred[indices[i]/num_val_replicates] = targets[i]


		# # for v in visit:
		# # 	if v!=division:
		# # 		print '-----------Error occured in validation segmentation....................'
		# # 		print v
		# val_predicted_label = np.argmax(val_pred, axis=1)
		# real_val_accuracy = np.mean(np.equal(np.argmax(val_pred, axis=1),val_correct_pred))
		# voting_val_accuracy = np.mean(np.equal(np.argmax(val_voting_pred, axis=1),val_correct_pred))

		# print " validation loss:\t{:.6f}".format(val_err/ val_batches)
		# print " validation accuracy:\t{:.2f} %".format(val_acc/val_batches*100)
		# print " real validation accuracy:\t{:.2f} %\n".format(real_val_accuracy*100)
		# # print " voting validation accuracy:\t{:.2f} %\n".format(voting_val_accuracy*100)

		# with open('validation_predicted_label.txt', 'w') as text_file:	
		# 	text_file.write('predicted\tground_truth\n')
		# 	total=np.zeros([num_classes])
		# 	correct = np.zeros([num_classes])
		# 	for i in range(len(val_pred)):
		# 		# print int(test_predicted_label[i]),int(test_correct_pred[i])
		# 		# print my_map[int(test_predicted_label[i])],my_map[int(test_correct_pred[i])]
		# 		if val_predicted_label[i] == val_correct_pred[i]:
		# 			correct[int(val_correct_pred[i])] += 1
		# 		total[int(val_correct_pred[i])] += 1
		# 		text_file.write(my_map[int(val_predicted_label[i])] + '\t' + my_map[int(val_correct_pred[i])] + '\n')
		# 	# print 'Class-wise Results : '
		# 	# for i in range(num_classes):
		# 	# 	print my_map[i] + ' correct = ' + str(correct[i]) + ' total = '+ str(total[i]) + ' acc = ' + str(correct[i]/total[i]*100)


		# #training is over, compute the test loss
		# test_err = 0
		# test_acc = 0
		# test_batches = 0
		# test_pred = np.zeros([X_test.shape[0]/num_test_replicates,num_classes])
		# test_voting_pred = np.zeros([X_test.shape[0]/num_test_replicates,num_classes])
		# test_correct_pred = np.zeros(X_test.shape[0]/num_test_replicates)
		# visit = np.zeros(X_test.shape[0]/num_test_replicates)

		# for batch in PD.iterate_minibatches(X_test, y_test, batch_size, shuffle = False):
		# 	# print " test_batches:\t{}".format(test_batches)
		# 	inputs, targets, indices = batch
		# 	test_error, test_accuracy, prediction= val_fn(inputs, targets)
		# 	test_err += test_error
		# 	test_acc += test_accuracy
		# 	test_batches += 1
		# 	for i in range(len(indices)):
		# 		visit[indices[i]/num_test_replicates] += 1
		# 		test_pred[indices[i]/num_test_replicates] += prediction[i]
		# 		test_voting_pred[indices[i]/num_test_replicates][np.argmax(prediction[i])] += 1
		# 		test_correct_pred[indices[i]/num_test_replicates] = targets[i]

		# # for v in visit:
		# # 	if v!=division:
		# # 		print '-----------Error occured in testing segmentation....................'
		# # 		print v

		# test_predicted_label = np.argmax(test_pred, axis=1)
		# real_test_accuracy = np.mean(np.equal(np.argmax(test_pred, axis=1),test_correct_pred))
		# voting_test_accuracy = np.mean(np.equal(np.argmax(test_voting_pred, axis=1),test_correct_pred))

		# print " test loss:\t{:.6f}".format(test_err/test_batches)
		# print " test accuracy:\t{:.2f} %".format(test_acc/test_batches *100)
		# print " real test accuracy:\t{:.2f} %\n".format(real_test_accuracy*100)
		# # print " voting test accuracy:\t{:.2f} %\n".format(voting_test_accuracy*100)

		# with open('test_predicted_label.txt', 'w') as text_file:	
		# 	text_file.write('predicted\tground_truth\n')
		# 	total=np.zeros([num_classes])
		# 	correct = np.zeros([num_classes])
		# 	for i in range(len(test_pred)):
		# 		# print int(test_predicted_label[i]),int(test_correct_pred[i])
		# 		# print my_map[int(test_predicted_label[i])],my_map[int(test_correct_pred[i])]
		# 		if test_predicted_label[i] == test_correct_pred[i]:
		# 			correct[int(test_correct_pred[i])] += 1
		# 		total[int(test_correct_pred[i])] += 1
		# 		text_file.write(my_map[int(test_predicted_label[i])] + '\t' + my_map[int(test_correct_pred[i])] + '\n')
		# 	# print 'Class-wise Results : '
		# 	# for i in range(num_classes):
		# 	# 	print my_map[i] + ' correct = ' + str(correct[i]) + ' total = '+ str(total[i]) + ' acc = ' + str(correct[i]/total[i]*100)
		
		# my_train_buffer[current_patience] = "Epoch {} of {} fold: {} training_loss: {:.6f} real_train_accuracy: {:.6f}\n".format(epoch, epochs,fold,train_err/ train_batches, real_train_accuracy)
		# my_validation_buffer[current_patience] = "Epoch {} of {} fold: {} validation_loss: {:.6f} validation_accuracy: {:.2f} %\n".format(epoch, epochs,fold, val_err/ val_batches,val_acc/val_batches*100)
		

		if(patience==True):
			#updating the best model
			epoch_acc = val_acc
			if(epoch_acc > best_acc):
				print 'best model updated...'
				
				#update the best model
				best_acc = epoch_acc
				# best_model = lasagne.layers.get_all_param_values(network)
				np.savez(model_name, *lasagne.layers.get_all_param_values(network))
				for i in range(current_patience+1):
					with open(best_train_log, 'a') as text_file:
						text_file.write(my_train_buffer[i])
					with open(best_validation_log, 'a') as text_file:
						text_file.write(my_validation_buffer[i])

				my_train_buffer = np.empty(max_patience+2, dtype=str)
				my_validation_buffer = np.empty(max_patience+2, dtype=str)
				current_patience = 0
			else:
				print 'current_patience increased to: ' + str(current_patience)
				print 'learning_rate = ' + str(learning_rate.get_value())
				current_patience += 1
				if(current_patience>max_patience):
					print 'resumed from best model with accuracy : ' + str(best_acc)
					my_train_buffer = np.empty(max_patience+2, dtype=str)
					my_validation_buffer = np.empty(max_patience+2, dtype=str)
					current_patience = 0
					if learning_rate.get_value()>lasagne.utils.floatX(0.000016):
						learning_rate.set_value(learning_rate.get_value()* lasagne.utils.floatX(0.5))

					
					with np.load(model_name) as f:
						# print f['arr_0'].shape
						param_values = [f['arr_%d' % i] for i in range(len(f.files))]
						# print len(param_values)

					lasagne.layers.set_all_param_values(network, param_values)


		print "---------------------------------------------------------------------------------------"
		# with open(train_log, 'a') as text_file:
		# 	text_file.write("Epoch {} of {} fold: {} training_loss: {:.6f} training_accuracy: {:.2f} % ,real_train_accuracy: {:.2f} %\n\n".format(epoch, epochs,fold,train_err/ train_batches, train_acc/train_batches*100,real_train_accuracy*100))
		# with open(validation_log, 'a') as text_file:
		# 	text_file.write("Epoch {} of {} fold: {} validation_loss: {:.6f} validation_accuracy: {:.2f} % ,real_validation_accuracy: {:.2f} %\n".format(epoch, epochs,fold, val_err/ val_batches,val_acc/val_batches*100,real_val_accuracy*100))

	if patience == True:
		#save the model for each fold
		with np.load(model_name) as f:
			# print f['arr_0'].shape
			param_values = [f['arr_%d' % i] for i in range(len(f.files))]
			# print len(param_values)

		lasagne.layers.set_all_param_values(network, param_values)

		
	#training is over, compute the test loss
	test_err = 0
	test_acc = 0
	test_batches = 0

	test_pred = np.zeros([X_test.shape[0]/num_replicates,num_classes])
	test_voting_pred = np.zeros([X_test.shape[0]/num_replicates,num_classes])
	test_correct_pred = np.zeros(X_test.shape[0]/num_replicates)
	visit = np.zeros(X_test.shape[0]/num_replicates)

	for batch in PD.iterate_minibatches(X_test, y_test, batch_size, shuffle = False):
		# print " test_batches:\t{}".format(test_batches)
		inputs, targets, indices = batch
		test_error, test_accuracy, prediction= val_fn(inputs, targets)
		test_err += test_error
		test_acc += test_accuracy
		test_batches += 1
		for i in range(len(indices)):
			visit[indices[i]/num_replicates] += 1
			test_pred[indices[i]/num_replicates] += prediction[i]
			test_voting_pred[indices[i]/num_replicates][np.argmax(prediction[i])] += 1
			test_correct_pred[indices[i]/num_replicates] = targets[i]

	# for v in visit:
	# 	if v!=division:
	# 		print '-----------Error occured in testing segmentation....................'
	# 		print v
	test_predicted_label = np.argmax(test_pred, axis=1)
	real_test_accuracy = np.mean(np.equal(np.argmax(test_pred, axis=1),test_correct_pred))
	voting_test_accuracy = np.mean(np.equal(np.argmax(test_voting_pred, axis=1),test_correct_pred))

	

	print " final results:"
	print " test loss:\t{:.6f}".format(test_err/test_batches)
	print " test accuracy:\t{:.2f} %".format(test_acc/test_batches *100)
	print " real test accuracy:\t{:.2f} %\n".format(real_test_accuracy*100)
	# print " voting test accuracy:\t{:.2f} %\n".format(voting_test_accuracy*100)

	with open('test_predicted_label.txt', 'w') as text_file:	
		text_file.write('predicted\tground_truth\n')
		for i in range(len(test_pred)):
			# print int(test_predicted_label[i]),int(test_correct_pred[i])
			# print my_map[int(test_predicted_label[i])],my_map[int(test_correct_pred[i])]
			text_file.write(my_map[int(test_predicted_label[i])] + '\t' + my_map[int(test_correct_pred[i])] + '\n')



	with open(test_log, 'a') as text_file:
		text_file.write(" Epoch {} of {} fold: {} test loss: {:.6f} test accuracy: {:.2f} % ,real_test_accuracy: {:.2f} \n%".format(epoch, epochs,fold,test_err/ test_batches,test_acc/test_batches*100, real_test_accuracy*100))

	if patience==False:
		np.savez(model_name, *lasagne.layers.get_all_param_values(network))


# dataset_name = sys.argv[1]
# fold = int(sys.argv[2])
# division = int(sys.argv[3])
# patience = sys.argv[4]
# representation = sys.argv[5]
# augmentation = sys.argv[6]
# train()
def main():
    # parse command line
    my_parser = parser.opts_parser()
    args = my_parser.parse_args()

    # run
    train(**vars(args))


if __name__ == "__main__":
	main()

