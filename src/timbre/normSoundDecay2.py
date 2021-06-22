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

def normSoundDecay2(signal,sr,maxsp=None,zero=1.0e-10,plot=False):
    # evaluate the normalized sound decay envelope
    t = np.arange(len(signal)) #/sr
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    idmax = np.argwhere(amplitude_envelope == np.max(amplitude_envelope))[0,0]
    amplitude_envelope = amplitude_envelope[idmax:]
    if maxsp == None: maxsp = len(amplitude_envelope)-1 #int(np.argwhere(np.abs(signal) < zero)[0])
    alpha = np.poly1d(np.polyfit(t[:maxsp],np.log(amplitude_envelope[:maxsp]),1))
    if plot:
        plt.plot(t[:maxsp],np.log(amplitude_envelope[:maxsp]))
        tp = np.linspace(0,t[maxsp],200)
        plt.plot(tp,alpha(tp),'.')
        plt.show()
    return(np.abs(alpha[1]),alpha,t)

