Machine Learning Models. The definition of machine learning models for sound
recognition requires standard techniques of data science (like the separation of
data entries in training and testing sets, definition of neural network architec-
tures, etc.) that will not be discussed here. Basic knowledge of Keras is also
assumed. MUSICNTWRK module timbrePy contains many auxiliary functions to
deal with such tasks. Here we limit to report the API for the main machine
learning functions:

- `def trainNNmodel(mfcc,label,gpu=0,cpu=4,niter=100,nstep=10,neur=16,test=0.08,numclasses=2,epoch=30,verb=0,thr=0.85,w=False)` -
	 train a 2 layer neural network model on the ful MFCC spectrum of sounds.
	 Returns: model,training and testing sets,data for re-scaling and normaliza-
	 tion,data to asses the accuracy of the training session.
	- mfcc (float) list of all the MFCCs (or PSCCs) in the repository
	- gpu, cpu (int) number of GPUs or CPSs used for the run
	- niter (int) max number of model fit sessions
	- nstep (int) how often the training and testing sets are redefined
	- neur (int)number of neurons in first layer (it is doubled on the second
		layer
	- test (float) defines the relative size of training and testing sets
	- numclasses=2 (int) dimension of the last layer
	- epoch (int) number of epochs in the training of the neural network
	- verb (int) verbose - print information during the training run
	- thr (float) keep the model if accuracy is ¿ test
	- w (logical) write model on file if accuracy is abovethr
- `def trainCNNmodel(mfcc,label,gpu=0,cpu=4,niter=100,nstep=10,neur=16,test=0.08,numclasses=2,epoch=30,verb=0,thr=0.85,w=False)` -
	 train a convolutional neural network (CNN) model on the full MFCC/PSCC
	 spectrum of sounds. Returns: model,training and testing sets, data for rescaling and normalization, data to assess the accuracy of the training session.
	 Parameters are defined as above.

For a complete description and example see the notebook Examples-advanced_timbre.ipynb avaialble on GitHub.
