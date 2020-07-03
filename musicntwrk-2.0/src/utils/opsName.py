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

def opsName(a,b,TET):
    # given two vectors returns the name of the minimal distance operator that connects them
    a = np.sort(a)
    b = np.sort(b)   
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
    diff = a-b
    for i in range(diff.shape[0]):
        if diff[i] >= int(TET/2):
            diff[i] -= TET
        if diff[i] < -int(TET/2):
            diff[i] += TET
    diff = np.sort(np.abs(diff))

    return('O('+np.array2string(np.trim_zeros(diff),separator=',').replace(" ","").replace("[","").replace("]","")+')')
