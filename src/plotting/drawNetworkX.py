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

def drawNetworkX(Gx,forceiter=100,grphtype='undirected',dx=10,dy=10,colormap='jet',
                 scale=1.0,drawlabels=True,giant=False):
    if giant:
        Gx = list(Gx.subgraph(c) for c in nx.connected_components(Gx))[0]
    pos = nx.spring_layout(Gx,iterations=forceiter)
    part = cm.best_partition(Gx)
    values = [part.get(node) for node in Gx.nodes()]
    d = nx.degree(Gx)
    dsize = [(d[v]+1)*100*scale for v in Gx.nodes()]
    plt.figure(figsize=(dx, dy))
    nx.draw_networkx(Gx,pos=pos,cmap=plt.get_cmap(colormap),node_color=values,
                    node_size=dsize)
    plt.show()
    return(part)
