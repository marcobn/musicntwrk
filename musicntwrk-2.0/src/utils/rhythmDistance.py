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
import fractions as fr
import sklearn.metrics as sklm

def rhythmDistance(a,b,distance):
    '''
    •	calculates the minimal duration distance between two rhythmic cells of same cardinality (bijective)
    •	a,b (str) – rhythmic cells
    '''
    a = np.asarray(a.normalOrder())
    b = np.asarray(b.normalOrder())
    n = a.shape[0]
    if a.shape[0] != b.shape[0]:
        print('dimension of arrays must be equal')
        sys.exit()
    iTET = np.vstack([np.identity(n,dtype=int),-np.identity(n,dtype=int)])
    iTET = np.vstack([iTET,np.zeros(n,dtype=int)])
    diff = np.zeros(2*n+1,dtype=float)
    v = []
    for i in range(2*n+1):
        r = np.sort(b - iTET[i])
        diff[i] = sklm.pairwise_distances(a.reshape(1, -1),r.reshape(1, -1),metric=distance)[0]
    imin = np.argmin(diff)
    return(fr.Fraction(diff.min()))
