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

import sys
import itertools as itr
import numpy as np
import pandas as pd

from ..musicntwrk import RHYTHMSeq

from ..utils.communications import *
from ..utils.load_balancing import *
from ..utils.Sublists import *
from ..utils.Remove import *
from ..utils.str2frac import *
from ..utils.str2float import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def rhythmPDictionary(N,Nc,REF):

    '''
    •	Generate the dictionary of all possible rhythmic sequences from all possible groupings of N 
        REF durations
    •	N (int)– number of REF units
    •	Nc cardinality of the grouping
    •	returns the dictionary as pandas DataFrame and indicates all non retrogradable cells
    '''
    name = []
    prime = []
    
    r = [REF] * N
    r = RHYTHMSeq(r)
    idx = np.linspace(0,N-1,N,dtype=int)
    
    subidx = list(Sublists(idx))
    seq=[]
    for l in range(len(subidx)):
        subseq = []
        for i in range(len(subidx[l])):
            aux = 0
            for k in range(len(subidx[l][i])):
                aux += r.rseq[subidx[l][i][k]]
            subseq.append(aux)
        seq.append(subseq)
    seq = Remove(seq)
    
    # select groupings with requested cardinality
    seqx = []
    for n in range(len(seq)):
        if len(seq[n]) == Nc:
            seqx.append(seq[n])
    seq = seqx
        
    # put all cells in prime/normal order form
    s = []
    v = []
    for i in range(len(seq)):
        p = RHYTHMSeq(seq[i][:],REF)
        s.append(p.rseq)
        v.append(p.rIntervalVector()[0])
    s = np.asarray(s)
    vector = np.asarray(v)

    for i in range(len(seq)):
        name.append(str(Nc)+'-'+str(i+1))
        prime.append(str(s[i,:]).replace('Fraction','').replace(', ','/')\
                            .replace('(','').replace(')','').replace('\n','').replace('[','').replace(']',''))
        
    dictionary = None
    
    # Create dictionary of rhythmic cells
    reference = []
    for n in range(len(name)):
        entry = [name[n],prime[n],
                np.array2string(vector[n,:],separator=',').replace(" ","")]
        reference.append(entry)

    dictionary = pd.DataFrame(reference,columns=['cell','r-seq','r-vec'])
    dictionary = dictionary.drop_duplicates(subset=['r-seq', 'r-vec'])
    
    # clean dictionary
    for n in range(len(dictionary)):
        dictionary.loc[n][1] = str(RHYTHMSeq(str2frac(dictionary.loc[n][1])).normalOrder())\
        .replace('Fraction','').replace(', ','/').replace('(','').replace(')','')\
        .replace('\n','').replace('[','').replace(']','')
    dictionary = dictionary.drop_duplicates(subset=['r-seq', 'r-vec']).reset_index(drop=True)
    
    # rename entries in ascending order and check for non-retrogradability
    for n in range(len(dictionary)):
        dictionary.loc[n][0] = str(Nc)+'-'+str(n+1)
        if RHYTHMSeq(str2frac(dictionary.loc[n][1])).isNonRetro(): 
            dictionary.loc[n][0] += 'N'
    
    # find those that are Z-related (have same interval onset vector)
    
    vector = []
    for n in range(len(dictionary)):
        vector.append(str2float(dictionary['r-vec'][n]))
    u, indeces = np.unique(vector, return_inverse=True,axis=0)
    ZrelT = []
    for n in range(u.shape[0]):
        if np.array(np.where(indeces == n)).shape[1] != 1:
            indx = np.array(np.where(indeces == n))[0]
            Zrel = []
            for m in range(indx.shape[0]):
                dictionary.loc[indx[m]][0] = dictionary.loc[indx[m]][0]+'Z'
                Zrel.append(dictionary.loc[indx[m]][0])
            ZrelT.append(Zrel)    
    return(dictionary,ZrelT)

