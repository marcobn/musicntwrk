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

import itertools as iter
import numpy as np
from .minimalDistance import minimalDistance

def minimalNoBijDistance(a,b,TET,distance):
    '''
    •	calculates the minimal distance between two pcs of different cardinality (non bijective) – uses minimalDistance()
    •	a,b (int) – pcs as lists or numpy arrays
    '''
    a = np.asarray(a)
    b = np.asarray(b)
    ndif = np.sort(np.array([a.shape[0],b.shape[0]]))[1] - np.sort(np.array([a.shape[0],b.shape[0]]))[0]
    c = np.asarray(list(iter.combinations_with_replacement(b,ndif)))
    r = np.zeros((c.shape[0],a.shape[0]))
    for l in range(c.shape[0]):
        r[l,:b.shape[0]] = b
        r[l,b.shape[0]:] = c[l]
    dist = np.zeros(r.shape[0])
    for l in range(r.shape[0]):
        dist[l],_=minimalDistance(a,r[l],TET,distance)
    imin = np.argmin(dist)
        
    return(min(dist),r[imin].astype(int))
