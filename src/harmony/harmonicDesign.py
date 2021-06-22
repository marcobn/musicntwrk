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
import networkx as nx
import numpy as np
import music21 as m21
import community as cm

from ..harmony.chinese_postman import chinese_postman
from ..data.WRITEscoreOps import WRITEscoreOps
from ..plotting.drawNetwork import drawNetwork


def harmonicDesign(mk,nnodes,refnodes,refedges,nedges=2,dualedge=3,nstart=None,scfree='barabasialbert',seed=None,prob=None,reverse=None,
                   display=None,write=None,verbose=False):
    # network generator (see documentation on networkx)
    if scfree == 'barabasialbert':
        if verbose: print('Barabasi-Albert')
        scfree = nx.barabasi_albert_graph(nnodes,nedges,seed)
    elif scfree == 'dual':
        if verbose: print('dual Barabasi-Albert')
        scfree = nx.dual_barabasi_albert_graph(nnodes,nedges,dualedge,prob,seed)
    elif scfree == 'erdosrenyi':
        if verbose: print('Erdos-Renyi')
        scfree = nx.erdos_renyi_graph(nnodes,prob)
    else:
        scfree = scfree
    # node degree distribution
    node = np.zeros((nnodes),dtype=int)
    weight = np.zeros((nnodes),dtype=int)
    for n in range(nnodes):
        node[n] = np.array(scfree.degree())[n][0]
        weight[n] = np.array(scfree.degree())[n][1]
    idx = np.argsort(weight)[::-1]
    if nstart == None:
        nstart = idx[0]
    euler_circuit = chinese_postman(scfree,starting_node=nstart)
    if verbose: print('Length of Eulerian circuit: {}'.format(len(euler_circuit)))
    # modularity 
    if not scfree.is_directed():
        part = cm.best_partition(scfree)
        modul = cm.modularity(part,scfree)
    # average degree
    ntot=scfree.number_of_nodes()
    avg = 0
    for n in scfree.degree():
        avg += n[1]
    avgdeg = avg/float(ntot)
    if verbose: print('Average degree: ', avgdeg, ' modularity = ',modul)
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
    eulerseq = []
    for i in range(len(euler_circuit)):
        ch = []
        for c in np.asarray(refnodes)[eudict[int(euler_circuit[i][0])]].tolist()[0]:
            if c == '#' or c == '-':
                pc = str(ch[-1])+c
                ch.pop()
                ch.append(pc)
            else:
                ch.append(c)
        eulerseq.append(m21.chord.Chord(ch).normalOrder)
    ch = []
    for c in np.asarray(refnodes)[eudict[int(euler_circuit[i][1])]].tolist()[0]:
        if c == '#' or c == '-':
            pc = str(ch[-1])+c
            ch.pop()
            ch.append(pc)
        else:
            ch.append(c)
    eulerseq.append(m21.chord.Chord(ch).normalOrder)
    if reverse: eulerseq = eulerseq[::-1]
    if display:
        eunodes,euedges,_,_,_,_,_ = mk.network(space='score',seq=eulerseq,ntx=True,general=True,
                                               distance='euclidean',grphtype='directed')
        drawNetwork(nodes=eunodes,edges=euedges,grphtype='directed')
    if write:
        WRITEscoreOps(eulerseq,w=write)
    
    if not scfree.is_directed():
        return(eulerseq,avgdeg,modul)
    else:
        return(eulerseq,avgdeg,None)