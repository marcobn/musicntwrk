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

import ruptures as rpt
import networkx as nx
import community as cm
import numpy as np
from musicntwrk.utils.Remove import Remove
import pandas as pd
import matplotlib.pyplot as plt

# def changePoint(value,model='rbf',penalty=1.0,brakepts=None,plot=False):
#     # change point detection
#     # available models: "rbf", "l1", "l2", rbf", "linear", "normal", "ar", "mahalanobis"
#     signal = np.array(value)
#     algo = rpt.Binseg(model=model).fit(signal)
#     my_bkps = algo.predict(pen=penalty,n_bkps=brakepts)
#     if plot:
#         # show results
#         rpt.show.display(signal, my_bkps, figsize=(10, 3))
#         plt.show()
#     # define regions from breaking points
#     sections = my_bkps
#     sections.insert(0,0)
#     # check last point
#     sections[-1] -= 1
#     if plot: print('model = ',model,' - sections = ',sections)
#     return(sections)

def changePoint(value,model='rbf',penalty=1.0,brakepts=None,plot=False,visibility=True):
    # change point detection
    # available models: "rbf", "l1", "l2", rbf", "linear", "normal", "ar", "mahalanobis"
    signal = np.array(value)
    algo = rpt.Binseg(model=model).fit(signal)
    my_bkps = algo.predict(pen=penalty,n_bkps=brakepts)
    
    # define regions from breaking points
    sections = my_bkps
    sections.insert(0,0)
    # check last point
    sections[-1] -= 1

    if visibility:
        # build the visibility network from the time series analysis
        series_list = [range(len(value)),value]
        g = []
        for s in series_list:
            g.append(nx.visibility_graph(s))
        modul = [] 
        res = np.linspace(0.2,3,40)
        for r in res:
            part = cm.best_partition(g[1],resolution=r,randomize=False)
            modul.append(cm.modularity(part,g[1]))
        mmax = np.argmax(modul)
        partm = cm.best_partition(g[1],resolution=res[mmax],randomize=False)
        # modulm = cm.modularity(partm,g[1])
        keys = list(partm.keys())
        values = list(partm.values())
        sorted_value_index = np.argsort(keys)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}
        sec = np.array([n for n in sorted_dict.values()])
        vis_sections = [0]
        for i in range(len(sec)-1):
            if sec[i] != sec[i+1]:
                vis_sections.append(i)
        vis_sections.append(i+1)

    if plot:
        # show results
        rpt.show.display(signal, my_bkps, figsize=(10, 3))
        # plt.scatter(np.array(vis['a']),np.array(vis['c']))
        plt.scatter(range(len(sec)),sec)
        plt.show()
        print('model = ',model,' - sections = \t',sections)
        print('from visibility - sections = \t',vis_sections)
    if visibility:
        return(sections,vis_sections)
    else:
        return(sections)

def visnet_to_csv(value,name,plot=True):
    # save time series data to nodelist and edgelist of visibility network
    series_list = [range(len(value)),value]
    g = []
    for s in series_list:
        g.append(nx.visibility_graph(s))
    labels = Remove(name)
    Gx = nx.DiGraph(g[1].edges)
    pc = np.array(['C','C#','D','E-','E','F','F#','G','A-','A','B-','B'])
    label = []
    for c in name:
        s = ''
        for m in c:
            s += pc[m]
        label.append(str(s))
    mapping = dict(zip(Gx, label))
    node_data = []
    for node_id, attributes in Gx.nodes(data=True):
        row = {"Label": mapping[node_id]}  # Add the node ID as a column
        row.update(attributes)   # Add all node attributes
        node_data.append(row)

    # Create a Pandas DataFrames from the extracted data
    df_nodes = pd.DataFrame(node_data)
    df_nodes.to_csv('vis_nodes.csv',index=False)
    df_edges = nx.to_pandas_edgelist(Gx)
    df_edges.to_csv('vis_edges.csv',index=False)

    if plot:
        plt.figure(figsize=(15, 15))
        positions = nx.spring_layout(Gx,k=0.06)
        _ = nx.draw_networkx_edges(Gx,pos=positions)
        _ = nx.draw_networkx_nodes(Gx,pos=positions,node_size=100,node_color='white',alpha=0.4)
        _ = nx.draw_networkx_labels(Gx,pos=positions,labels=mapping,font_size=4)
        plt.show()

    return(df_nodes,df_edges,Gx)