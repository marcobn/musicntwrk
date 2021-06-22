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

import sys,re,os
import pandas as pd
import numpy as np
import sklearn.metrics as sklm

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

def pcsEgoNetwork(label,dictionary,thup_e,thdw_e,thup,thdw,distance,write,TET):
    
    '''
    •	network generated from a focal node (ego) and the nodes to whom ego is directly connected to (alters)
    •	label (str)– label of the ego node
    •	thup_e, thdw_e (float) - upper and lower thresholds for edge creation from ego node
    •	thup, thdw (float)– upper and lower thresholds for edge creation among alters
    •	in output it writes the nodes_ego.csv, edges_ego.csv and edges_alters.csv as separate files in csv format
    '''
    
    if thdw_e < 1e-9:
        print('ego should not link to itself')
        sys.exit()
    
    # Create the ego network of pcs from a given node using the pcsDictionary
    
    df = dictionary

    # define nodes as distance 1 from ego
    # ego
    dict_class = df.set_index("class", drop = True)
    ego = np.asarray(list(map(int,re.findall('\d+',dict_class.loc[label][1]))))
    # alters
    dfv = np.asarray(df)
    vector = np.zeros((dfv[:,2].shape[0],int(TET/2)),dtype=int)
    for i in range(dfv[:,2].shape[0]):
        vector[i]  = np.asarray(list(map(int,re.findall('\d+',dfv[i,2]))))
    name = []
    pair = sklm.pairwise_distances(ego.reshape(1, -1), vector, metric=distance)
    for i in range(dfv[:,2].shape[0]):
        if pair[0,i] <= thup_e and pair[0,i] >= thdw_e:
            name.append(dfv[i,0])
    # add ego node
    name.append(label)
                      
    # write csv for nodes
    dnodes = pd.DataFrame(np.asarray(name),columns=['Label'])
    if write: dnodes.to_csv('nodes_ego.csv',index=False)
    nodes_ego = dnodes
    
    # find edges according to a metric
    # ego edges with proportional weights
    N = len(name)
    vector = np.zeros((N,int(TET/2)),dtype=int)
    dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
    for j in range(N):
        vector[j] = np.asarray(list(map(int,re.findall('\d+',dict_class.loc[name[j]][1]))))
    pair = sklm.pairwise_distances(ego.reshape(1, -1), vector, metric=distance)
    for j in range(N):
        if pair[0,j] <= thup_e and pair[0,j] >= thdw_e:
            tmp = pd.DataFrame([[str(i),str(j),str(1/pair[0,j])]],columns=['Source','Target','Weight'])
            dedges = dedges.append(tmp)
    # write csv for ego's edges
    if write: dedges.to_csv('edges_ego.csv',index=False)   
    edges_ego = dedges     
    
    # alters edges
    # parallelize over interval vector to optimize the vectorization in sklm.pairwise_distances
    if size != 1:
        ini,end = load_balancing(size, rank, N)
        nsize = end-ini
        vaux = scatter_array(vector)
        pair = sklm.pairwise_distances(vaux, vector, metric=distance)
        index = np.linspace(0,N,N,dtype=int)
        dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
        for i in range(nsize):
            tmp = pd.DataFrame(None,columns=['Source','Target','Weight'])
            tmp['Source'] = (i+ini)*np.ones(N,dtype=int)[:]
            tmp['Target'] = index[:]
            tmp['Weight'] = pair[i,:]
            dedges = dedges.append(tmp)
        dedges = dedges.query('Weight<='+str(thup)).query('Weight>='+str(thdw))
        dedges['Weight'] = dedges['Weight'].apply(lambda x: 1/x)
        # do some cleaning
        cond = dedges.Source > dedges.Target
        dedges.loc[cond, ['Source', 'Target']] = dedges.loc[cond, ['Target', 'Source']].values
        dedges = dedges.drop_duplicates(subset=['Source', 'Target'])

        # write csv for partial edges
        dedges.to_csv('edges'+str(rank)+'.csv',index=False)
        
        if rank == 0:
            dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
            for i in range(size):
                tmp = pd.read_csv('edges'+str(i)+'.csv')
                dedges = dedges.append(tmp)
                os.remove('edges'+str(i)+'.csv')
            # write csv for edges
            if write: dedges.to_csv('edges_alters.csv',index=False)
            edges_alters = dedges
    else:
        N = len(name)-1
        dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
        for i in range(N):
            vector_i = np.asarray(list(map(int,re.findall('\d+',dict_class.loc[name[i]][1]))))
            for j in range(i,N):
                vector_j = np.asarray(list(map(int,re.findall('\d+',dict_class.loc[name[j]][1]))))
                pair = sklm.pairwise.paired_euclidean_distances(vector_i.reshape(1, -1),vector_j.reshape(1, -1))
                if pair <= thup and pair >= thdw:
                    tmp = pd.DataFrame([[str(i),str(j),str(1/pair[0])]],columns=['Source','Target','Weight'])
                    dedges = dedges.append(tmp)

        # write csv for alters' edges
        if write: dedges.to_csv('edges_alters.csv',index=False)
        edges_alters = dedges
    
    return(nodes_ego, edges_ego, edges_alters)
