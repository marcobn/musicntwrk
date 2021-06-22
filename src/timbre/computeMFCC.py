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

def computeMFCC(input_path,input_file,nmel,ncc,zero):
    # read audio files in repository and compute the MFCC
    waves = list(glob.glob(os.path.join(input_path,input_file)))
    mfcc0 = []
    for wav in np.sort(waves):
        y, sr = librosa.load(wav)
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=nmel)
        log_S = librosa.power_to_db(S, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=ncc)
#        # Here we take the average over a single impulse (for lack of a better measure...)
#        mfcc0.append(np.sum(mfcc,axis=1)/mfcc.shape[1])
        # use mfcc[0] as weighting function for the average of the mfcc's over the full impulse
        mfnorm = (mfcc[0]-np.min(mfcc[0]))/np.max(mfcc[0]-np.min(mfcc[0]))
        mfcc0.append(mfcc.dot(mfnorm)/mfcc.shape[1])
    if zero:
        mfcc0 = np.asarray(mfcc0)
        mfcc[0] = mfnorm
    else:
        # take out the zero-th MFCC - DC value (power distribution)
        temp = np.asarray(mfcc0)
        mfcc0 = temp[:,1:]
    
    return(np.sort(waves),np.ascontiguousarray(mfcc0),mfcc)
    
