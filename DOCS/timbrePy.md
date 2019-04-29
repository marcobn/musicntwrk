timbrePy comprises of two sections: the first deals with orchestration color and
it is the natural extension of the score analyzer in pscPy; the second deals with
analysis and characterization of timbre from a (psycho-)acoustical point of view.
In particular, it provides: the characterization of sound using, among others,
Mel Frequency or Power Spectrum Cepstrum Coefficients (MFCC or PSCC); the
construction of timbral networks using descriptors based on MF- or PS-CCs; and
machine learning models for timbre recognition through the TensorFlow Keras
framework.

Orchestration analysis. The orchestration analysis section of timbrePy com-
prises of the following modules:

- `def orchestralVector(inputfile,barplot=True)` -
    Builds the orchestral vector sequence from score inmusicxmlformat. Re-
    turns the score sliced by beat; orchestration vector.
    - barplot=True plot the orchestral vector sequence as a matrix
- `def orchestralNetwork(seq)` -
    Generates the directional network of orchestration vectors from any score in
    musicxml format. Use orchestralScore() to import the score data as sequence.
    Returns nodes and edges as Pandas DataFrames; average degree, modularity
    and partitioning of the network.
    - seq (int) list of orchestration vectors extracted from the score
- `def orchestralVectorColor(orch,dnodes,part,color=plt.cm.binary)` -
    Plots the sequence of the orchestration vectors color-coded according to the
    modularity class they belong. Requires the output of orchestralNetwork()
    - seq (int) list of orchestration vectors extracted from the score

Sound classification. The sound classification section of timbrePy comprises
of modules for specific sound analysis that are based on the librosa python
library for audio signal processing. We refer the interested reader to the librosa
documentation at https://librosa.github.io/librosa/index.html. Specific audio signal processing functions are:

- `def computeMFCC(inputpath,inputfile,barplot=True,zero=True)` -
    read audio files in repository and compute a normalized MEL Frequency
    Cepstrum Coefficients and single vector map of the full temporal evolution
    of the sound as the convolution of the timeresolved MFCCs convoluted with
    the normalized first MFCC component (power distribution). Returns the list
    of files in repository, MFCC0, MFCC coefficients.
   - inputpath (str) path to repository
   - inputfile (str) filenames (accepts '\*')
   - barplot (logical) plot the MFCC0 vectors for every sound in the
      repository
   - zero (logical) If False, disregard the power distribution component.
- `def computePSCC(inputpath,inputfile,barplot=True,zero=True)` -
    Reads audio files in repository and compute a normalized Power Spectrum
    Frequency Cepstrum Coefficients and single vector map of the full tempo-
    ral evolution of the sound as the convolution of the time-resolved PSCCs
    convoluted with the normalized first PSCC component (power distribution).
    Returns the list of files in repository, PSCC0, PSCC coefficients. Other vari-
    ables as above.
- `def computeStandardizedMFCC(inputpath,inputfile,nmel=16,
    nmfcc=13,lmax=None,nbins=None)` -
    read audio files in repository and compute the standardized (equal number
    of samples per file) and normalized MEL Frequency Cepstrum Coefficient.
    Returns the list of files in repository, MFCC coefficients, standardized sample
    length.
   - nmel (int) number of Mel bands to use in filtering
   - nmfcc (int) number of MFCCs to return
   - lmax (int) max number of samples per file
   - nbins (int) number of FFT bins
- `def computeStandardizedPSCC(inputpath,inputfile,nmel=16,
    psfcc=13,lmax=None,nbins=None)` -
    read audio files in repository and compute the standardized (equal number
    of samples per file) and normalized Power Spectrum Frequency Cepstrum
    Coefficients. Returns the list of files in repository, PSCC coefficients, stan-
    dardized sample length.
    Variables defined as for MFCCs.
- `def timbralNetwork(waves,vector,thup=10,thdw=0.1)` -
    generates the network of MFCC vectors from sound recordings. Returns the
    nodes and edges tables as pandas DataFrames
   - seq (float) list of MFCC0 vectors
   - waves (str) names of sound files

Machine Learning Models. The definition of machine learning models for sound
recognition requires standard techniques of data science (like the separation of
data entries in training and testing sets, definition of neural network architec-
tures, etc.) that will not be discussed here. Basic knowledge of Keras is also
assumed. MUSICNTWRK module timbrePy contains many auxiliary functions to
deal with such tasks. Here we limit to report the API for the main machine
learning functions:

- `def trainNNmodel(mfcc,label,gpu=0,cpu=4,niter=100,nstep=10,
    neur=16,test=0.08,numclasses=2,epoch=30,verb=0,thr=0.85,w=False)` -
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
- `def trainCNNmodel(mfcc,label,gpu=0,cpu=4,niter=100,nstep=10,
    neur=16,test=0.08,numclasses=2,epoch=30,verb=0,thr=0.85,w=False)` -
    train a convolutional neural network (CNN) model on the full MFCC/PSCC
    spectrum of sounds. Returns: model,training and testing sets, data for rescaling and normalization, data to assess the accuracy of the training session.
    Parameters are defined as above.
