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
import community as cm
import matplotlib.pyplot as plt

def drawNetwork(Gx=None,Gxu=None,nodes=None,edges=None,forceiter=100,grphtype='undirected',dx=10,dy=10,colormap='jet',scale=1.0,
    layout='force',drawlabels=True,giant=False,equi=False,res=0.5,k=None,edge_labels=False,font=12):

    if grphtype == 'directed':
        if Gx == None and Gxu == None:
            Gx = nx.from_pandas_edgelist(edges,'Source','Target',['Weight'],create_using=nx.DiGraph())
            Gxu = nx.from_pandas_edgelist(edges,'Source','Target',['Weight'])
        if giant: 
            print('not implemented')
    else:
        if Gx == None:
            Gx = nx.from_pandas_edgelist(edges,'Source','Target',['Weight'])
        if giant and not nx.is_connected(Gx):
            S = [Gx.subgraph(c).copy() for c in nx.connected_components(Gx)]
            size = []
            for s in S:
                size.append(len(s))
            idsz = np.argsort(size)
            print('found ',np.array(size)[idsz],' connected components')
            index = int(input('enter index '))
            Gx = S[idsz[index]]
    if layout == 'force' or layout==None:
        pos = nx.spring_layout(Gx,k=1,iterations=forceiter)
    elif layout == 'spiral':
        pos = nx.spiral_layout(Gx,equidistant=equi,resolution=res)
    df = np.array(nodes)
    if len(df.shape) == 1:
        df = np.reshape(df,(len(df),1))
    nodelabel = dict(zip(np.linspace(0,len(df[:,0])-1,len(df[:,0]),dtype=int),df[:,0]))
    labels = {}
    for idx, node in enumerate(Gx.nodes()):
        labels[node] = nodelabel[int(node)]
    if grphtype == 'directed':
        part = cm.best_partition(Gxu)
        values = [part.get(node) for node in Gxu.nodes()]
    else:
        part = cm.best_partition(Gx)
        values = [part.get(node) for node in Gx.nodes()]
    d = nx.degree(Gx)
    dsize = [(d[v]+1)*100*scale for v in Gx.nodes()]
    plt.figure(figsize=(dx, dy))
    if edge_labels:
        edge_labels = nx.get_edge_attributes(Gx, 'Label')
        nx.draw_networkx_edge_labels(Gx, pos, edge_labels, font_size=font)
    nx.draw_networkx(Gx,pos=pos,labels=labels,with_labels=drawlabels,cmap=plt.get_cmap(colormap),node_color=values,
                    node_size=dsize)
    plt.show()

    