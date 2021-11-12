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
import pandas as pd

from ..utils.communications import *
from ..utils.load_balancing import *

try:
    from mpi4py import MPI
    # initialize parallel execution
    comm=MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    para = True
except:
    rank = 0
    size = 1
    para = False

from ..musicntwrk import PCSet

def pcsDictionary(Nc,row,a,order,prob,TET):

    '''
    •	Generate the dictionary of all possible pcs of a given cardinality in a generalized musical space of TET pitches
    •	Nc (int)– cardinality
    •	order (logical)– if 0 returns pcs in prime form, if 1 returns pcs in normal order, if 2, returns pcs in normal 0 order
    •	row (logical)– if True build dictionary from tone row, if False, build dictionary from all combinatorial pcs of Nc cardinality given the totality of TET.
    •	if row = True, a is the list of pitches in the tone row (int)
    •	returns the dictionary as pandas DataFrame and the list of all Z-related pcs
    '''
    name = prime = commonName = None
    if rank == 0:
        name = []
        prime = []
        commonName = []
        
    # generate all possible combinations of n integers or of the row in argument
    if row:
        a = np.asarray(list(iter.combinations(a,Nc)))
        if order == 3:
            tmp = a.tolist()
            for n in a:
                p = list(iter.permutations(n))
                for c in p:
                    tmp.append(c)
            a = np.asarray(tmp)
    else:
        a = np.asarray(list(iter.combinations(range(TET),Nc)))

    # put all pcs in prime/normal order form if needed
    s = np.zeros((a.shape[0],Nc),dtype=int)
    ini,end = load_balancing(size, rank, a.shape[0])
    nsize = end-ini

    aux = scatter_array(a)
    saux = np.zeros((nsize,Nc),dtype=int)
    if para: comm.Barrier()
    for i in range(nsize):
        p = PCSet(aux[i,:],TET=TET)
        if order == 0:
            saux[i,:] = p.primeForm()[:]
        elif order == 1:
            saux[i,:] = p.normalOrder()[:]
        elif order == 2:
            saux[i,:] = p.normal0Order()[:]
        elif order == 3:
            p = PCSet(aux[i,:],TET=TET,ORD=False,UNI=False)
            saux[i,:] = p.pcs
        else:
            if rank == 0: print('no ordering specified')
    if para:
        comm.Barrier()
        gather_array(s,saux,sroot=0)
    else:
        s = saux

    if rank == 0:
        # eliminate duplicates in s
        s = np.unique(s,axis=0)

        # calculate interval vectors and assign names
        v = []
        for i in range(s.shape[0]):
            p = PCSet(s[i,:],TET)
            v.append(p.intervalVector())
            name.append(str(Nc)+'-'+str(i+1))
            prime.append(np.array2string(s[i,:],separator=',').replace(" ",""))

        vector = np.asarray(v)

    dictionary = ZrelT = None
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

        name = np.asarray(name)
        
        # Create dictionary of pitch class sets
        reference = []
        for n in range(len(name)):
            entry = [name[n],prime[n],
                    np.array2string(vector[n,:],separator=',').replace(" ","")]
            reference.append(entry)

        dictionary = pd.DataFrame(reference,columns=['class','pcs','interval'])
        
        if prob != None:
            df = np.asarray(dictionary)
            pruned = pd.DataFrame(None,columns=['class','pcs','interval'])
            for i in range(df.shape[0]):
                r = np.random.rand()
                if r <= prob:
                    tmp = pd.DataFrame([[str(df[i,0]),str(df[i,1]),str(df[i,2])]],columns=['class','pcs','interval'])
                    pruned = pruned.append(tmp)
            dictionary = pruned
        
    return(dictionary,ZrelT)
