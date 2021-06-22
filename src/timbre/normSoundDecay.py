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
from scipy.signal import hilbert

def normSoundDecay(signal,sr):
    # evaluate the normalized sound decay envelope
    zero=1.0e-10
    t = np.arange(len(signal))/sr
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    maxsp = int(np.argwhere(np.abs(signal) < zero)[0])
    alpha = np.poly1d(np.polyfit(t[:maxsp],np.log(amplitude_envelope[:maxsp]),1))
    return(np.abs(alpha[1]),alpha,t)

