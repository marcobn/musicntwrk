`Sound classification.` 
Modules for specific sound analysis that are based on the librosa python
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
- `def computeStandardizedMFCC(inputpath,inputfile,nmel=16,nmfcc=13,lmax=None,nbins=None)` -
	read audio files in repository and compute the standardized (equal number
	of samples per file) and normalized MEL Frequency Cepstrum Coefficient.
	Returns the list of files in repository, MFCC coefficients, standardized sample
	length.
   - nmel (int) number of Mel bands to use in filtering
   - nmfcc (int) number of MFCCs to return
   - lmax (int) max number of samples per file
   - nbins (int) number of FFT bins
- `def computeStandardizedPSCC(inputpath,inputfile,nmel=16,psfcc=13,lmax=None,nbins=None)` -
	read audio files in repository and compute the standardized (equal number
	of samples per file) and normalized Power Spectrum Frequency Cepstrum
	Coefficients. Returns the list of files in repository, PSCC coefficients, stan-
	dardized sample length.
	Variables defined as for MFCCs.
- `def computeASCBW(input_path,input_file)` -
    sound descriptor as normalized sound decay (alpha), spectral centoid and spectral bandwidth
    as in Aramaki et al. 2009 (also available in standardized form)
