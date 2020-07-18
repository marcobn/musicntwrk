## `timbre`

<span>`timbre`</span> contains all the modules that deal with analysis
and characterization of timbre from a (psycho-)acoustical point of view
and provides the characterization of sound using, among others, Mel
Frequency or Power Spectrum Cepstrum Coefficients (MFCC or PSCC) that
can be used in the construction of timbral networks using these
descriptors

#### Sound classification. 

<span>__def computeMFCC__(input\_path,input\_file,barplot=True,zero=True)</span>  
read audio files in repository and compute a normalized MEL Frequency
Cepstrum Coefficients and single vector map of the full temporal
evolution of the sound as the convolution of the time-resolved MFCCs
convoluted with the normalized first MFCC component (power
distribution). Returns the list of files in repository, MFCC0, MFCC
coefficients.

<span><span>input\_path (str)</span> </span>
path to repository

<span><span>input\_file (str)</span> </span>
filenames (accepts "\*")

<span><span>barplot (logical)</span> </span>
plot the MFCC0 vectors for every sound in the repository

<span><span>zero (logical)</span> </span>
If False, disregard the power distribution component.

<span>*Returns*</span>

<span><span>waves (list of strings)</span> </span>
filenames of the .wav files in the repository

<span><span>mfcc0 (list)</span> </span>
vector of MFCC0 (MFCC with DC component taken out)

<span><span>mfcc (list)</span> </span>
vector of MFCC (full MFCC)

<span>__def computePSCC__(input\_path,input\_file,barplot=True,zero=True)</span>  
Reads audio files in repository and compute a normalized Power Spectrum
Frequency Cepstrum Coefficients and single vector map of the full
temporal evolution of the sound as the convolution of the time-resolved
PSCCs convoluted with the normalized first PSCC component (power
distribution). Returns the list of files in repository, PSCC0, PSCC
coefficients. Other variables and output as above.

<span>__def computeStandardizedMFCC__(input\_path,input\_file,nmel=16,  
nmfcc=13,lmax=None,nbins=None)</span>  
read audio files in repository and compute the standardized (equal
number of samples per file) and normalized MEL Frequency Cepstrum
Coefficient. Returns the list of files in repository, MFCC coefficients,
standardized sample length.

<span><span>nmel (int)</span> </span>
number of Mel bands to use in filtering

<span><span>nmfcc (int)</span> </span>
number of MFCCs to return

<span><span>lmax (int)</span> </span>
max number of samples per file

<span><span>nbins (int)</span> </span>
number of FFT bins

<span>*Returns*</span>  
As above.

<span>__def computeStandardizedPSCC__(input\_path,input\_file,nmel=16,  
psfcc=13,lmax=None,nbins=None)</span>  
read audio files in repository and compute the standardized (equal
number of samples per file) and normalized Power Spectrum Frequency
Cepstrum Coefficients. Returns the list of files in repository, PSCC
coefficients, standardized sample length.  
Variables defined as for MFCCs. 

<span>*Returns*</span>  
As above.

