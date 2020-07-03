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

import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

from .mfccSoundDecayPiecewise import mfccSoundDecayPiecewise
from .minimizeBKPT import minimizeBKPT

def computeModifiedASCBW(input_path,input_file,scnd,method,nstep):
    # sound descriptor as normalized sound decay from the fit of the 0-th component of the MFCC, 
    # spectral centoid and spectral bandwidth
    eps=1.0e-10
    waves = list(glob.glob(os.path.join(input_path,input_file)))
    ascbw = []
    for wav in np.sort(waves):
        y, sr = librosa.load(wav)
        maxsp = int(np.argwhere(np.abs(y) < eps)[0])
        cent = librosa.feature.spectral_centroid(y=y, sr=sr,hop_length=maxsp)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr,hop_length=maxsp)
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=16)
        log_S = librosa.power_to_db(S, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
        try:
            a,_,_,_,_ = mfccSoundDecayPiecewise(mfcc[0],breakpoints=minimizeBKPT(mfcc[0],method=method,nstep=nstep)[1])
        except:
            print(wav)
            break
        alpha0 = a[0]
        if np.abs(a[1]) > np.abs(a[2]): a[1] = a[2]
        if a[1] > 0: a[1] = 0
        alpha1 = a[1]
        if scnd: 
            ascbw.append([np.abs(alpha0),np.abs(alpha1),cent[0,0],spec_bw[0,0]])
            na = 4
        else:
            ascbw.append([np.abs(alpha0),cent[0,0],spec_bw[0,0]])
            na = 3
    ascbw = np.asarray(ascbw)
    # normalization to np.max
    ascbwu = np.zeros(ascbw.shape,dtype=float)
    where_are_NaNs = np.isnan(ascbw)
    ascbw[where_are_NaNs] = 0
    for i in range(na):
        ascbwu[:,i] = ascbw[:,i]
        ascbw[:,i] /= np.max(ascbw[:,i])
    
    print('sound is ',waves)
    print('primary decay constant   = ',ascbwu[0,0].round(3))
    print('secondary decay constant = ',ascbwu[0,1].round(3))
    print('spectral centroid        = ',int(ascbwu[0,2]))
    print('bandwidth                = ',int(ascbwu[0,3]))
    
    return(np.sort(waves),ascbw,ascbwu)

