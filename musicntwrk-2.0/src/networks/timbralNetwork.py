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

import sys,os
import numpy as np
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

def timbralNetwork(waves,vector,thup,thdw):
    
    ''' 
    •    generates the network of MFCC vectors from sound recordings
    •    seq – list of MFCC vectors
    •    waves - names of wave files
    '''
    # build the network

    dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
    dnodes = pd.DataFrame(None,columns=['Label'])
    for n in range(len(waves)):
        nameseq = pd.DataFrame([waves[n].split('/')[-1].split('.')[0]],columns=['Label'])
        dnodes = dnodes.append(nameseq)
    df = np.array(dnodes)
    dnodes = pd.DataFrame(None,columns=['Label'])
    dff,idx = np.unique(df,return_inverse=True)
    for n in range(dff.shape[0]):
        nameseq = pd.DataFrame([[str(dff[n])]],columns=['Label'])
        dnodes = dnodes.append(nameseq)
    
    N = vector.shape[0]
    index = np.linspace(0,vector.shape[0]-1,vector.shape[0],dtype=int)
    # parallelize over interval vector to optimize the vectorization in sklm.pairwise_distances
    ini,end = load_balancing(size, rank, N)
    nsize = end-ini
    vaux = scatter_array(vector)
    #pair = sklm.pairwise_distances(vaux,vector,metric=distance)
    dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
    for i in range(nsize):
        tmp = pd.DataFrame(None,columns=['Source','Target','Weight'])
        tmp['Source'] = (i+ini)*np.ones(vector.shape[0],dtype=int)[:]
        tmp['Target'] = index[:]
        tmp['Weight'] = np.sqrt(np.sum((vaux[i,:]-vector[:,:])**2,axis=1))
        tmp = tmp.query('Weight<='+str(thup)).query('Weight>='+str(thdw))
        dedges = dedges.append(tmp)
            
    dedges = dedges.query('Weight<='+str(thup)).query('Weight>='+str(thdw))
    dedges['Weight'] = dedges['Weight'].apply(lambda x: 1/x)
    # do some cleaning
    cond = dedges.Source > dedges.Target
    dedges.loc[cond, ['Source', 'Target']] = dedges.loc[cond, ['Target', 'Source']].values
    dedges = dedges.drop_duplicates(subset=['Source', 'Target'])

    # write csv for partial edges
    dedges.to_csv('edges'+str(rank)+'.csv',index=False)
    if para: comm.Barrier()
    
    if size != 1 and rank == 0:
        dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
        for i in range(size):
            tmp = pd.read_csv('edges'+str(i)+'.csv')
            dedges = dedges.append(tmp)
            os.remove('edges'+str(i)+'.csv')
        # do some cleaning
        cond = dedges.Source > dedges.Target
        dedges.loc[cond, ['Source', 'Target']] = dedges.loc[cond, ['Target', 'Source']].values
        dedges = dedges.drop_duplicates(subset=['Source', 'Target'])
        # write csv for edges
        dedges.to_csv('edges.csv',index=False)
    elif size == 1:
        os.rename('edges'+str(rank)+'.csv','edges.csv')

    return(dnodes,dedges)
    
