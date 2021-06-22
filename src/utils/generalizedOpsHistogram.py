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

from .opsDistance import opsDistance

def generalizedOpsHistogram(values,counts):
    ops = []
    for i in range(-3,3):
        ops.append('O('+str(i)+')')
    for i in range(-3,3):
        for j in range(i,3):
            ops.append('O('+str(i)+','+str(j)+')')
    for i in range(-3,3):
        for j in range(i,3):
            for k in range(j,3):
                ops.append('O('+str(i)+','+str(j)+','+str(k)+')')
    for i in range(-3,3):
        for j in range(i,3):
            for k in range(j,3):
                for l in range(k,3):
                    ops.append('O('+str(i)+','+str(j)+','+str(k)+','+str(l)+')')
    ops = np.array(ops)
    dist = np.zeros(ops.shape[0])
    for i in range(ops.shape[0]):
        dist[i] = opsDistance(ops[i])[1]
    idx = np.argsort(dist)
    ops = ops[idx]

    ops_dict = {}
    for i in range(len(ops)):
        ops_dict.update({ops[i]:0})

    for i in range(len(values)):
        ops_dict.update({values[i]:counts[i]})

    newvalues = np.asarray(list(ops_dict.keys()))
    newcounts = np.asarray(list(ops_dict.values()))
    return(newvalues,newcounts,ops_dict,dist[idx])

