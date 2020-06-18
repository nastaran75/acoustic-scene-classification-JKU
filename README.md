# acoustic-scene-classification


## Training:
The sinc_train.py inside the SincNet folder is the training script. You can mention what network you want and pass the parameter --nework as a command line argument.
For example:

For SincNet: python sinc_train.py --network = Sinc

For Sample CNN: python sinc_train.py --network = SampleCNN

For Sample CNN and SincLayer as the first layer: python sinc_train.py --network = SampleCNN+Sinc

For DSNP and SincLayer as the first layer: python sinc_train.py --network = DSNP+Sinc

The train script takes the right network and update strategy based on the network you selected.


### Passing Other Training Prameters:
You can also fix other parameters while training (look at parser.py). For example you can mention you database (currently it can be either '2channel2016_madmom' or '2channel2017_madmom' for DCASE 2016 and 2017 Respectively)
There are other parameters such as division (I used 10 for DCASE'16 and 3 for DCASE'17), batchsize(I used 28), fold (1..4) etc.
You can also adjust learning rate and patience parameters.

For exapmple: python sinc_train.py --network = SampleCNN --dataset_name = 2channel2016_madmom --fold = 2 --division=3


## Testing:
The trained models are saved in the "final_models" folder in SincNet. For testing, take your desired models to the folder "testModels" and run test.py with the same parameters as the training script (It iterates over the four folds so you do not need to mention the fold parameter)
It iterates over models and prints the final accuracy.

For example: python test.py --network = SampleCNN --division = 3
