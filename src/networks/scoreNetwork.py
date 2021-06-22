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

import pandas as pd
import numpy as np
import music21 as m21
import networkx as nx
import community as cm
import matplotlib.pyplot as plt

from ..musicntwrk import PCSet
from ..utils.minimalDistance import minimalDistance
from ..utils.minimalNoBijDistance import minimalNoBijDistance
from ..utils.generalizedOpsName import generalizedOpsName

def scoreNetwork(seq,ntx,general,distance,TET):
    
    ''' 
    •	generates the directional network of chord progressions from any score in musicxml format
    •	seq (int) – list of pcs for each chords extracted from the score
    •	use readScore() to import the score data as sequence
    '''
    # build the directional network of the full progression in the chorale

    dedges = pd.DataFrame(None,columns=['Source','Target','Weight','Label'])
    dnodes = pd.DataFrame(None,columns=['Label'])
    for n in range(len(seq)):
        p = PCSet(np.asarray(seq[n]),TET)
        if TET == 12:
            if p.pcs.shape[0] == 1:
                nn = ''.join(m21.chord.Chord(p.pcs.tolist()).pitchNames)
            else:
                nn = ''.join(m21.chord.Chord(p.normalOrder().tolist()).pitchNames)
            nameseq = pd.DataFrame([[str(nn)]],columns=['Label'])
        elif TET == 24:
            dict24 = {'C':0,'C~':1,'C#':2,'D-':2,'D`':3,'D':4,'D~':5,'D#':6,'E-':6,'E`':7,'E':8,
                                'E~':9,'F`':9,'F':10,'F~':11,'F#':12,'G-':12,'G`':13,'G':14,'G~':15,'G#':16,
                                'A-':16,'A`':17,'A':18,'A~':19,'A#':20,'B-':20,'B`':21,'B':22,'B~':23,'C`':23}
            tmp = []
            for i in p.pcs:
                tmp.append(list(dict24.keys())[list(dict24.values()).index(i)]) 
            nameseq = pd.DataFrame([[''.join(tmp)]],columns=['Label'])
        dnodes = dnodes.append(nameseq)
    df = np.asarray(dnodes)
    dnodes = pd.DataFrame(None,columns=['Label'])
    dcounts = pd.DataFrame(None,columns=['Label','Counts'])
    dff,idx,cnt = np.unique(df,return_inverse=True,return_counts=True)
    for n in range(dff.shape[0]):
        nameseq = pd.DataFrame([[str(dff[n])]],columns=['Label'])
        dnodes = dnodes.append(nameseq)
        namecnt = pd.DataFrame([[str(dff[n]),cnt[n]]],columns=['Label','Counts'])
        dcounts = dcounts.append(namecnt)

    for n in range(1,len(seq)):
        if len(seq[n-1]) == len(seq[n]):
            a = np.asarray(seq[n-1])
            b = np.asarray(seq[n])
            pair,r = minimalDistance(a,b,TET,distance)
        else:
            if len(seq[n-1]) > len(seq[n]):
                a = np.asarray(seq[n-1])
                b = np.asarray(seq[n])
                pair,r = minimalNoBijDistance(a,b,TET,distance)
            else: 
                b = np.asarray(seq[n-1])
                a = np.asarray(seq[n])
                pair,r = minimalNoBijDistance(a,b,TET,distance)
        if pair != 0:
            if general == False:
                tmp = pd.DataFrame([[str(idx[n-1]),str(idx[n]),str(1/pair),opsName(a,r,TET)]],
                                    columns=['Source','Target','Weight','Label'])
            else:
                tmp = pd.DataFrame([[str(idx[n-1]),str(idx[n]),str(1/pair),generalizedOpsName(a,r,TET,distance)[1]]],
                                    columns=['Source','Target','Weight','Label'])
            dedges = dedges.append(tmp)
            
# write dataframe with pcs rather than indeces
#           if pair != 0:
#                if general == False:
#                    tmp = pd.DataFrame([[str(seq[n-1]),str(seq[n]),str(1/pair),opsName(a,r,TET)]],
#                                        columns=['Source','Target','Weight','Label'])
#                else:
#                    tmp = pd.DataFrame([[str(seq[n-1]),str(seq[n]),str(1/pair),generalizedOpsName(a,r,TET)[1]]],
#                                        columns=['Source','Target','Weight','Label'])
#                dedges = dedges.append(tmp)

    
    if ntx:
        # evaluate average degree and modularity
        gbch = nx.from_pandas_edgelist(dedges,'Source','Target',['Weight','Label'],create_using=nx.DiGraph())
        gbch_u = nx.from_pandas_edgelist(dedges,'Source','Target',['Weight','Label'])
        # modularity 
        part = cm.best_partition(gbch_u)
        modul = cm.modularity(part,gbch_u)
        # average degree
        nnodes=gbch.number_of_nodes()
        avg = 0
        for node in gbch.degree():
            avg += node[1]
        avgdeg = avg/float(nnodes)
        return(dnodes,dedges,dcounts,avgdeg,modul,gbch,gbch_u)
    else:
        return(dnodes,dedges,dcounts)
