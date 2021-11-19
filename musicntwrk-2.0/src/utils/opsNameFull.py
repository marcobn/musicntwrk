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
from ..musicntwrk import PCSet

def opsNameFull(a,b,TET):
    # given two vectors returns the name of the normal ordered distance operator (R) that connects them
    a = PCSet(a,UNI=False).normalOrder()
    b = PCSet(b,UNI=False).normalOrder()   
    d = np.zeros((b.shape[0]),dtype=int) 
    for n in range(b.shape[0]):
        c = np.roll(b,n)
        diff = a-c
        for i in range(diff.shape[0]):
            if diff[i] >= int(TET/2):
                diff[i] -= TET
            if diff[i] < -int(TET/2):
                diff[i] += TET
        diff = np.abs(diff)
        d[n] = diff.dot(diff)
    nmin = np.argmin(d)
    b = np.roll(b,nmin)
    diff = b-a
    for i in range(diff.shape[0]):
        if diff[i] >= int(TET/2):
            diff[i] -= TET
        if diff[i] < -int(TET/2):
            diff[i] += TET

    return('R('+np.array2string(diff,separator=',').replace(" ","").replace("[","").replace("]","")+')')

def opsNameAbs(a,b,TET):
    # given two vectors returns the name of the normal ordered distance operator (R) that connects them
    a = PCSet(a,UNI=False).normalOrder()
    b = PCSet(b,UNI=False).normalOrder()   
    d = np.zeros((b.shape[0]),dtype=int) 
    for n in range(b.shape[0]):
        c = np.roll(b,n)
        diff = a-c
        for i in range(diff.shape[0]):
            if diff[i] >= int(TET/2):
                diff[i] -= TET
            if diff[i] < -int(TET/2):
                diff[i] += TET
        diff = np.abs(diff)
        d[n] = diff.dot(diff)
    nmin = np.argmin(d)
    b = np.roll(b,nmin)
    diff = b-a
    for i in range(diff.shape[0]):
        if diff[i] >= int(TET/2):
            diff[i] -= TET
        if diff[i] < -int(TET/2):
            diff[i] += TET
    diff = np.trim_zeros(np.sort(np.abs(diff)))

    return('O('+np.array2string(diff,separator=',').replace(" ","").replace("[","").replace("]","")+')')
