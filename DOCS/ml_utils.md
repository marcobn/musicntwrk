## Machine Learning Models (<span>`ml_utils`</span>)


<span>__def trainNNmodel__(mfcc, label, gpu=0, cpu=4, niter=100, nstep = 10,
neur = 16, test=0.08, num\_classes=2, epoch=30, verb=0, thr=0.85,
w=False)</span>  
train a 2 layer neural network model on the ful MFCC spectrum of sounds.
Returns: model,training and testing sets,data for re-scaling and
normalization,data to asses the accuracy of the training session.

<span><span>mfcc (float)</span> </span>
list of all the MFCCs (or PSCCs) in the repository

<span><span>gpu, cpu (int)</span> </span>
number of GPUs or CPUs used for the run

<span><span>niter (int)</span> </span>
max number of model fit sessions

<span><span>nstep (int)</span> </span>
how often the training and testing sets are redefined

<span><span>neur (int)</span> </span>
number of neurons in first layer (it is doubled on the second layer

<span><span>test (float)</span> </span>
defines the relative size of training and testing sets

<span><span>num\_classes=2 (int)</span> </span>
dimension of the last layer

<span><span>epoch (int)</span> </span>
number of epochs in the training of the neural network

<span><span>verb (int)</span> </span>
verbose - print information during the training run

<span><span>thr (float)</span> </span>
keep the model if accuracy is \> test

<span><span>w (logical)</span> </span>
write model on file if accuracy is above <span>thr</span>

<span>__def trainCNNmodel__(mfcc, label, gpu=0, cpu=4, niter=100, nstep=10,
neur=16, test=0.08, num\_classes=2, epoch=30, verb=0, thr=0.85, w =
False)</span>  
train a convolutional neural network (CNN) model on the full MFCC/PSCC
spectrum of sounds. Returns: model,training and testing sets,data for
re-scaling and normalization,data to asses the accuracy of the training
session.  
Parameters are defined as above.

For a complete description and example see the notebook on GitHub.
