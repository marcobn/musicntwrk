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
import networkx as nx
import fractions as fr
import music21 as m21

from ..musicntwrk import RHYTHMSeq
from ..harmony.chinese_postman import chinese_postman
from ..utils.floatize import floatize


def rhythmicDesign(dictionary,nnodes,nedges,refnodes,refedges,nstart=None,seed=None,reverse=None,
                   random=None):
    # network generator (see documentation on networkx)
    scfree = nx.barabasi_albert_graph(nnodes,nedges,seed)
    # node degree distribution
    node = np.zeros((nnodes),dtype=int)
    weight = np.zeros((nnodes),dtype=int)
    for n in range(nnodes):
        node[n] = np.array(scfree.degree())[n][0]
        weight[n] = np.array(scfree.degree())[n][1]
    idx = np.argsort(weight)[::-1]
    if nstart == None:
        nstart = idx[0]
    euler_circuit = chinese_postman(scfree,nstart)
    print('Length of Eulerian circuit: {}'.format(len(euler_circuit)))
    # reference node degree distribution
    try:
        bnet = nx.from_pandas_edgelist(refedges,'Source','Target',['Weight','Label'])
    except:
        bnet = nx.from_pandas_edgelist(refedges,'Source','Target',['Weight'])
    bnode = np.zeros((nnodes),dtype=int)
    bweight = np.zeros((nnodes),dtype=int)
    for n in range(nnodes):
        bnode[n] = np.array(bnet.degree())[n][0]
        bweight[n] = np.array(bnet.degree())[n][1]
    bidx = np.argsort(bweight)[::-1]
    # associate reference nodes to network
    a = node[idx[:]]
    b = bnode[bidx[:]]
    eudict = dict(zip(a,b))
    # write score
    dict_cell = dictionary.set_index("cell", drop = True)
    eulerseq = []
    for i in range(len(euler_circuit)):
        label = np.asarray(refnodes)[eudict[int(euler_circuit[i][0])]].tolist()[0]
        if random:
            dur = RHYTHMSeq(np.random.permutation(list(map(floatize,list(map(
                    lambda x: fr.Fraction(x),dict_cell.loc[label][0].split())))))).rseq
        else:
            dur = RHYTHMSeq(np.array(list(map(floatize,list(map(
                    lambda x: fr.Fraction(x),dict_cell.loc[label][0].split())))))).rseq
        for j in range(len(dur)):
            eulerseq.append(m21.duration.Duration(4*dur[j]))

    if reverse: eulerseq = eulerseq[::-1]

    return(eulerseq)
