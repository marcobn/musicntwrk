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
from .mfccSoundDecayPiecewise import mfccSoundDecayPiecewise

def minimizeBKPT(vec,method='MC',nstep=100,extrema=[(0,20),(15,40),(30,60)]):

    def func(x,*args):
        return(mfccSoundDecayPiecewise(vec,breakpoints=x,plot=False)[4])

    if method == 'MC':
        # MonteCarlo optimization of breakpoints for piecewise fitting
        # needs nstep
        xopt = [1,1,1]
        res = res0 = 10
        for n in range(nstep):
            x = np.sort([np.random.randint(1,20),np.random.randint(1,40),np.random.randint(1,80)])
            try:
                res = func(x,vec)
            except:
                pass
            if res < res0:
                xopt = np.sort(x)
                res0 = res
    elif method == 'uniform':
        # find minimum of residual on a uniform grid
        # needs extrema as list of tuples
        x = []
        for i in range(extrema[0][0],extrema[0][1]):
            for j in range(extrema[1][0],extrema[1][1]):
                for k in range(extrema[2][0],extrema[2][1]):
                    x.append([i,j,k])
        x = np.asarray(x)
        res = []
        xres = []
        for n in range(x.shape[0]):
            try:
                res.append(func(np.sort(x[n]),vec))
                xres.append(x[n])
            except:
                pass
        res = np.asarray(res)
        xres = np.asarray(xres)
        r = res[~np.isnan(res)]
        xopt = np.sort(xres[np.argwhere(res==np.min(r))][0,0])
        res0 = res[np.argwhere(res==np.min(r))][0,0]

    return(res0,xopt)
            
