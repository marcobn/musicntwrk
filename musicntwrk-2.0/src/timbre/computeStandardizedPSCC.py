#
# MUSIC𝄞NTWRK
#
# A python library for pitch class set and rhythmic sequences classification and manipulation,
# the generation of networks in generalized music and sound spaces, and the sonification of arbitrary data
#
# Copyright (C) 2018 Marco Buongiorno Nardelli
# http://www.materialssoundmusic.com, mbn@unt.edu
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

import numpy as np
import glob, os
import librosa

def computeStandardizedPSCC(input_path,input_file,ncc,lmax,maxi,nbins):
    # read audio files in repository and compute the standardized (equal number of samples per file) 
    # and normalized PSCC
    waves = list(glob.glob(os.path.join(input_path,input_file)))
    wf = []
    for wav in np.sort(waves):
        y, sr = librosa.load(wav)
        wf.append(y)
    wf = np.asarray(wf)
    # standardization of the number of sample in every sound wav
    if lmax == None:
        lwf = []
        for n in range(wf.shape[0]):
            lwf.append(wf[n].shape[0])
        lwf = np.asarray(lwf)
        lmax = np.max(lwf)
    pscc = []
    for n in range(wf.shape[0]):
        if wf[n].shape[0] <= lmax:
            wtmp = np.pad(wf[n], (0, lmax-wf[n].shape[0]), 'constant')
        else:
            wtmp = wf[n][:lmax]
        if nbins == None:
            hopl = 512
        else:
            hopl = int((lmax/nbins)*2/2+1)
        # power (energy-squared) spectrogram
        D = np.abs(librosa.stft(wtmp,hop_length=hopl))**2
        log_D = librosa.power_to_db(D, ref=np.max)
        temp = librosa.feature.mfcc(S=log_D, n_mfcc=ncc)
        # normalize pscc[0] first
        temp[0] = (temp[0]-np.min(temp[0]))/(np.max(temp[0])-np.min(temp[0]))
        pscc.append(temp)
    pscc = np.asarray(pscc)
    return(np.sort(waves),pscc,lmax)
        
