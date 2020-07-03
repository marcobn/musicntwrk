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

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def rhythmDictionary(Nc,a,REF):

    '''
    •	Generate the dictionary of all possible rhythmic sequences of Nc length in a generalized meter 
        space of N durations
    •	Nc (int)– cell length
    •	a is the list of durations in the rhythm sequence (str)
    •	returns the dictionary as pandas DataFrame and indicates all non retrogradable cells
    '''
    name = []
    prime = []
    if a == None:
        sys.exit()
    a = RHYTHMSeq(a).normalOrder()
    a = RHYTHMSeq(a).floatize()
    a = np.unique(np.asarray(list(itr.combinations(a,Nc))),axis=0)
        
    # put all cells in prime/normal order form

    s = []
    v = []
    for i in range(a.shape[0]):
        p = RHYTHMSeq(a[i,:].tolist(),REF)
        s.append(p.normalOrder())
        v.append(p.durationVector()[0])
    s = np.asarray(s)
    vector = np.asarray(v)

    for i in range(a.shape[0]):
        name.append(str(Nc)+'-'+str(i+1))
        prime.append(str(s[i,:]).replace('Fraction','').replace(', ','/')\
                            .replace('(','').replace(')','').replace('\n','').replace('[','').replace(']',''))
        
        
    dictionary = None
    
    # find those that can be made non retrogradable
    
    for n in range(a.shape[0]):
        perm = np.asarray(list(itr.permutations(a[n,:],a.shape[1])))
        perm = np.unique(perm,axis=0)
        for i in range(perm.shape[0]):
            if RHYTHMSeq(perm[i].tolist(),REF).isNonRetro():
                name[n] = name[n]+'N'
    
    # find those that are Z-related (have same duration vector)
    
    ZrelT = None
    if rank == 0:
        # find pc sets in Z relation
        u, indeces = np.unique(vector, return_inverse=True,axis=0)
        ZrelT = []
        for n in range(u.shape[0]):
            if np.array(np.where(indeces == n)).shape[1] != 1:
                indx = np.array(np.where(indeces == n))[0]
                Zrel = []
                for m in range(indx.shape[0]):
                    name[indx[m]] = name[indx[m]]+'Z'
                    Zrel.append(name[indx[m]])
                ZrelT.append(Zrel)
                    
    # Create dictionary of rhythmic cells
    reference = []
    for n in range(len(name)):
        entry = [name[n],prime[n],
                np.array2string(vector[n,:],separator=',').replace(" ","")]
        reference.append(entry)

    dictionary = pd.DataFrame(reference,columns=['cell','r-seq','d-vec'])
    dictionary.drop_duplicates(subset=['r-seq', 'd-vec'])
    
    return(dictionary,ZrelT)
