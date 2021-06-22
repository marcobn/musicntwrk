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

import re, sys, os, time
import numpy as np
import sklearn.metrics as sklm
import itertools as iter
import pandas as pd
import networkx as nx

from operator import mul
from functools import reduce

from ..musicntwrk import PCmidiR

def spiralChordSpace(chord,scale,octaves=3,TET=12,distance='euclidean',thdw=0.01,thup=12):

#     chord and scale are string where notes are separated by commas
#     or midi numbers
    
#     Translate note names in midi
    if isinstance(chord[0],str):
        chord = PCmidiR(chord).midi
        scale = PCmidiR(scale).midi
    elif chord.all() >= 20:
        pass
    else:
        print('enter chord and scale as note+octave or midi numbers (C4 = 60)')
        print('enter accidentals as -`~# or increments of 0.5 (quarter tones)')
        sys.exit()
    
#     Identify the interval sequence of the chord in the scale
    idx = []
    for c in chord:
        idx.append(np.argwhere(scale == c)[0][0])
    idx = np.array(idx)
    
#     Generate all permutation of the chord in 3 octaves (transposition along the chord) 
#     and for all chords in the scale (transposition along the scale)

#     permutation and transposition along the chord
    if octaves == 1:
        octave = (-1,0)
        comb = np.zeros(chord.shape[0])
    elif octaves == 2:
        octave = (-1,1)
    elif octaves == 3:
        octave = (-1,2)
    elif octaves == 4:
        octave = (-2,2)
    elif octaves == 5:
        octave = (-2,3)
    else:
        print('too many octaves - set to 3 by default')
        octave=(-1,2)

    if octaves != 1:
        ranges = ((octave, ) * chord.shape[0])

        comb = [[octave[0]]*chord.shape[0]] # need to add the first transposition vector
        operations=reduce(mul,(p[1]-p[0] for p in ranges))-1
        result=[i[0] for i in ranges]
        pos=len(ranges)-1
        increments=0
        while increments < operations:
            if result[pos]==ranges[pos][1]-1:
                result[pos]=ranges[pos][0]
                pos-=1
            else:
                result[pos]+=1
                increments+=1
                pos=len(ranges)-1 #increment the innermost loop
                comb.append(result.copy())
                
#     transposition along the scale
    totalP = []
    comb = np.array(comb)*TET
    for i in range(len(scale)):
        chP = np.unique(list(iter.permutations(scale[(idx+i)%len(scale)])),axis=0)
        for ch in chP:
            for n in comb:
                totalP.append(ch+n)
    totalP = np.unique(np.sort(totalP),axis=0)
    
#     create dictionary of chords
    reference = []
    for t in totalP:
        entry = [PCmidiR(t).pitches,PCmidiR(t).midi]
        reference.append(entry)

    dictionary = pd.DataFrame(reference,columns=['pitches','midi'])
    
    
#     build the network
    df = np.asarray(dictionary)
#     nodes
    dnodes = pd.DataFrame(None,columns=['Label'])
    for n in range(len(df)):
        s = ','
        p = s.join(df[n,0])
        dnodes = dnodes.append(pd.DataFrame([[str(p)]],columns=['Label']))

#     edges according to a metric
    thup = np.sqrt(thup)

    N = df[:,1].shape[0]
    dedges = pd.DataFrame(None,columns=['Source','Target','Weight','Label'])
    for i in range(N):
        a  = np.asarray(df[i,1])
        for j in range(i,N):
            b  = np.asarray(df[j,1])
            pair = sklm.pairwise_distances(a.reshape(1, -1),b.reshape(1, -1),metric=distance)[0]
            if pair <= thup and pair >= thdw:
                tmp = pd.DataFrame([[str(i),str(j),str(1/pair[0]),str(int(np.round(pair[0]**2,0)))]],
                                   columns=['Source','Target','Weight','Label'])
                dedges = dedges.append(tmp)
    
    Gxu = nx.from_pandas_edgelist(dedges,'Source','Target',['Weight','Label'])
    
    return(dictionary,dnodes,dedges,Gxu)