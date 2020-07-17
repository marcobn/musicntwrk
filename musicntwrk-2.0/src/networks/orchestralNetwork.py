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
import pandas as pd
import networkx as nx
import community as cm

from ..utils.minimalDistance import minimalDistance

def orchestralNetwork(seq,distance,TET):
    
    ''' 
    •    generates the directional network of orchestration vectors from any score in musicxml format
    •    seq (int) – list of orchestration vectors extracted from the score
    •    use orchestralScore() to import the score data as sequence
    '''
    # build the directional network of the full orchestration progression

    dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
    dnodes = pd.DataFrame(None,columns=['Label'])
    for n in range(len(seq)):
        nameseq = pd.DataFrame([np.array2string(seq[n]).replace(" ","").replace("[","").replace("]","")],\
                               columns=['Label'])
        dnodes = dnodes.append(nameseq)
    df = np.asarray(dnodes)
    dnodes = pd.DataFrame(None,columns=['Label'])
    dff,idx = np.unique(df,return_inverse=True)
    for n in range(dff.shape[0]):
        nameseq = pd.DataFrame([[str(dff[n])]],columns=['Label'])
        dnodes = dnodes.append(nameseq)
    for n in range(1,len(seq)):
        a = np.asarray(seq[n-1])
        b = np.asarray(seq[n])
        pair,r = minimalDistance(a,b,TET,distance)
        tmp = pd.DataFrame([[str(idx[n-1]),str(idx[n]),str(pair+0.1)]],
                           columns=['Source','Target','Weight'])
        dedges = dedges.append(tmp)
    
    # evaluate average degree and modularity
    gbch = nx.from_pandas_edgelist(dedges,'Source','Target','Weight',create_using=nx.DiGraph())
    gbch_u = nx.from_pandas_edgelist(dedges,'Source','Target','Weight')
    # modularity 
    part = cm.best_partition(gbch_u)
    modul = cm.modularity(part,gbch_u)
    # average degree
    nnodes=gbch.number_of_nodes()
     # average degree
    nnodes=gbch.number_of_nodes()
    avg = 0
    for node in gbch.in_degree():
        avg += node[1]
    avgdeg = avg/float(nnodes)
        
    return(dnodes,dedges,avgdeg,modul,part,gbch,gbch_u)

