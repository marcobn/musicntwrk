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

import time,re
import numpy as np
import itertools as iter
import pandas as pd
import music21 as m21

from ..musicntwrk import PCSet
from ..utils.minimalDistanceVec import minimalDistanceVec
from ..utils.opsCheckByNameVec import opsCheckByNameVec

def vLeadNetworkByNameVec(dictionary,name,distance,prob,write,pcslabel,TET):
    
    # Create network of minimal voice leadings from the pcsDictionary
    # vector version - only bijective and no probability distribution

    df = np.asarray(dictionary)

    # write csv for nodes
    if pcslabel:
        dnodes = pd.DataFrame(None,columns=['Label'])
        for n in range(len(df)):
            p = PCSet(np.asarray(list(map(int,re.findall('\d+',df[n,1])))))
            if p.pcs.shape[0] == 1:
                nn = ''.join(m21.chord.Chord(p.pcs.tolist()).pitchNames)
            else:
                nn = ''.join(m21.chord.Chord(p.normalOrder().tolist()).pitchNames)
            nameseq = pd.DataFrame([[str(nn)]],columns=['Label'])
            dnodes = dnodes.append(nameseq)
    else:
        dnodes = pd.DataFrame(df[:,0],columns=['Label'])
    if write: dnodes.to_csv('nodes.csv',index=False)

    # find edges according to a metric
    N = df[:,1].shape[0]
    dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
    vector_i = np.zeros((N,len(list(map(int,re.findall('\d+',df[0,1]))))),dtype=int)
    disx = np.zeros((N,N),dtype=float)
    pairx = np.zeros((N,N),dtype=bool)
    dis = np.zeros((N,N),dtype=float)
    pair = np.zeros((N,N),dtype=bool)
    # vector of pcs
    for i in range(N):
        vector_i[i] = np.asarray(list(map(int,re.findall('\d+',df[i,1]))))
    # matrix of distances
    for i in range(N):
        disx[i,:] = minimalDistanceVec(vector_i,np.roll(vector_i,-i,axis=0),TET,distance)
        pairx[i,:] = opsCheckByNameVec(vector_i,np.roll(vector_i,-i,axis=0),name,TET)

    for i in range(N):
        dis += np.diag(disx[i,:(N-i)],k=i)
        pair += np.diag(pairx[i,:(N-i)],k=i)
        
    ix,iy = np.nonzero(dis)
    for n in range(ix.shape[0]):
        if pair[ix[n],iy[n]]:
            tmp = pd.DataFrame([[str(ix[n]),str(iy[n]),str(1/dis[ix[n],iy[n]])]],columns=['Source','Target','Weight'])
            dedges = dedges.append(tmp)

    # write csv for edges
    if write: dedges.to_csv('edges.csv',index=False)
    
    return(dnodes,dedges)
