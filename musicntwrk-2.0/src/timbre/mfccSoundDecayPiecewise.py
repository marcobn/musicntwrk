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

from ..utils.SegmentedLinearReg import SegmentedLinearReg

def mfccSoundDecayPiecewise(mfcc,breakpoints=[]):
    idmax = np.argwhere(mfcc == np.max(mfcc))[0,0]
    y = mfcc[idmax:100]
    y = y-np.min(y)
    y = np.log(y[y>1.e-10])
    x = np.arange(len(y))

    initialBreakpoints = breakpoints
    xfit,yfit = SegmentedLinearReg( x, y, initialBreakpoints )

    # Evaluate residual (np.abs(a*x+b-mfcc)/len(mfcc))
    line = np.zeros(x.shape,dtype=float)
    a = []
    b = []
    for n in range(1,xfit.shape[0]):
        a0 = (yfit[n]-yfit[n-1])/(xfit[n]-xfit[n-1])
        b0 = yfit[n-1]-a0*xfit[n-1]
        a.append(a0)
        b.append(b0)
        for i in range(x.shape[0]):
            if x[i] >= xfit[n-1] and x[i] <= xfit[n]:
                line[i] = a0*x[i]+b0
    res = np.sum(np.abs(y-line)/x.shape[0])

    return(a,b,xfit,yfit,res)

