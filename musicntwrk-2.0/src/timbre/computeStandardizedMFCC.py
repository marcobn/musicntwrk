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

def computeStandardizedMFCC(input_path,input_file,nmel,nmfcc,lmax,maxi,nbins):
    # read audio files in repository and compute the standardized (equal number of samples per file) 
    # and normalized MFCC
    waves = list(glob.glob(os.path.join(input_path,input_file)))
    wf = []
    for wav in np.sort(waves):
        y, sr = librosa.load(wav)
        wf.append(np.array(y))
#   wf = np.array(wf)
    # standardization of the number of sample in every sound wav
    if lmax == None:
        lwf = []
        for n in range(len(wf)):
            lwf.append(wf[n].shape[0])
        lwf = np.asarray(lwf)
        lmax = np.max(lwf)
    mfcc = []
    for n in range(len(wf)):
        if wf[n].shape[0] <= lmax:
            wtmp = np.pad(wf[n], (0, lmax-wf[n].shape[0]), 'constant')
        else:
            wtmp = wf[n][:lmax]
        if nbins == None:
            hopl = 512
        else:
            hopl = hopl = int((lmax/nbins)*2/2+1) #round(int(lmax/nbins)/2)*2
        S = librosa.feature.melspectrogram(y=wtmp, sr=sr, n_mels=nmel,hop_length=hopl)
        log_S = librosa.power_to_db(S, ref=np.max)
        temp = librosa.feature.mfcc(S=log_S, n_mfcc=nmfcc)
        # normalize mfcc[0] first
        try:
#            print(n,np.max(temp[0]),np.min(temp[0]))
            temp[0] = (temp[0]-np.min(temp[0]))/(np.max(temp[0])-np.min(temp[0]))
        except:
            print(n,np.max(temp[0]),np.min(temp[0]))
#        temp = np.abs(temp)
#        if maxi == None:
#            maxtemp = np.max(temp[1:])
#        else:
#            maxtemp = maxi
#        temp[1:] = temp[1:]/maxtemp
        mfcc.append(temp)
    mfcc = np.asarray(mfcc)
    return(np.sort(waves),mfcc,lmax)

