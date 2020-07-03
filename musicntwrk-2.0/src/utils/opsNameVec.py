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

def opsNameVec(a,b,TET=12):
    # given two arrays of vectors returns the array of the names of the operators that connects them
    # vector version
    a = np.sort(a,axis=1)
    b = np.sort(b,axis=1)
    d = np.zeros((b.shape[0],b.shape[1]),dtype=int) 
    diff = np.zeros((b.shape[0],b.shape[1]),dtype=int)
    nmin = np.zeros((b.shape[0],b.shape[1]),dtype=int)

    for n in range(b.shape[1]):
        c = np.roll(b,n,axis=1)
        diff = a-c
        aux = np.where(diff >= int(TET/2),diff-TET,diff)
        diff = np.abs(np.where(aux < -int(TET/2),aux+TET,aux)) 

        d[:,n] = np.sum(diff*diff,axis=1)
    nmin = np.argmin(d,axis=1)
    for i in range(b.shape[0]):
        b[i] = np.roll(b[i],nmin[i])
        diff[i] = a[i]-b[i]
    aux = np.where(diff >= int(TET/2),diff-TET,diff)
    diff = np.sort(np.abs(np.where(aux < -int(TET/2),aux+TET,aux)))
    name = []
    for i in range(b.shape[0]):
        name.append('O('+np.array2string(np.trim_zeros(diff[i]),separator=',')\
                    .replace(" ","").replace("[","").replace("]","")+')')

    return(np.asarray(name))
