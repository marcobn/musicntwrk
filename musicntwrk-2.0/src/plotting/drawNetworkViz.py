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
from graphviz import Digraph

def drawNetworkViz(nodes,edges,view=True,seed=None,engine='circo',format='svg',strict=True,size='20,20',ratio='0.9',colorlist=None):

    if colorlist == None:
        colorlist=['black','red','blue','green','cyan','magenta']
    
    if seed == None:
        np.random.seed(seed)

    # Generate graph using GraphViz

    G = Digraph(engine=engine,format=format,strict=strict)

    G.attr(rankdir='LR',model='subset',ratio=ratio,size=size,pad='0.2',splines='ortho',
           overlap='false',concentrate='false',mode='KK')
    G.attr('graph',label='',fontname='Helvetica Neue',fontsize='32',labelloc='t',nodesep='0.01')
    G.attr('node', shape='rectangle',fontname='Helvetica Neue',fontsize='16')
    G.attr('edge',arrowType='normal')

    nodelist = []
    if not isinstance(edges,list):
        edges = [edges]
    for i,e in enumerate(edges):
        df = e
        for idx, row in df.iterrows():
            try:
                node1, node2, weight, label = [str(i) for i in row]
            except:
                node1, node2, weight = [str(i) for i in row]

            if node1 not in nodelist:
                G.node(node1)
                nodelist.append(node1)
            if node2 not in nodelist:
                G.node(node2)
                nodelist.append(node2)
            if node1 != node2:
                if i == 0:
                    G.edge(node1,node2,color=colorlist[i])
                else:
                    G.edge(node1,node2,color=colorlist[i],arrowhead='none')
            else:
                pass

    for i,g in enumerate(nodelist):
        G.node(str(g),str(np.array(nodes)[int(g)][0])+'\n')

    G.render('./SVGnetworks',view=bool(view))