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

from .normSoundDecay import normSoundDecay

def computeASCBW(input_path,input_file):
    # sound descriptor as normalized sound decay (alpha), spectral centoid and spectral bandwidth
    # as in Aramaki et al. 2009
    eps=1.0e-10
    waves = list(glob.glob(os.path.join(input_path,input_file)))
    ascbw = []
    for wav in np.sort(waves):
        y, sr = librosa.load(wav)
        maxsp = int(np.argwhere(np.abs(y) < eps)[0])
        try:
            alpha0,_,_ = normSoundDecay(y,sr,plot=False)
        except:
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            y = y[(onset_frames[0]+1):]
            maxsp = int(np.argwhere(np.abs(y) < eps)[0])
            alpha0,_,_ = normSoundDecay(y,sr)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr,hop_length=maxsp)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr,hop_length=maxsp)
#        ascbw.append([alpha0,cent[0,0],spec_bw[0,0]])
        ascbw.append([alpha0/cent[0,0],cent[0,0],spec_bw[0,0]])
    ascbw = np.asarray(ascbw)
    # normalization to np.max
    for i in range(3):
        ascbw[:,i] /= np.max(ascbw[:,i])
    
    return(np.sort(waves),ascbw)

