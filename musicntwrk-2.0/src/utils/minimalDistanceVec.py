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

def minimalDistanceVec(a,b,TET,distance):
    '''
    •	calculates the minimal distance between two pcs of same cardinality (bijective)
    •	a,b (int) – pcs as numpy arrays or lists
    •   vector version
    '''
    a = np.asarray(a)
    b = np.asarray(b)
    N = a.shape[0]
    n = a.shape[1]
    if a.shape[0] != b.shape[0]:
        print('dimension of arrays must be equal')
        sys.exit()
    a = np.sort(a)
    iTET = np.vstack([np.identity(n,dtype=int)*TET,-np.identity(n,dtype=int)*TET])
    iTET = np.vstack([iTET,np.zeros(n,dtype=int)])
    diff = np.zeros((N,2*n+1),dtype=float)
    for i in range(2*n+1):
        r = np.sort(b - iTET[i])
        diff[:,i] = np.sqrt(np.sum((a-r)**2,axis=1))

    return(np.amin(diff,axis=1))

