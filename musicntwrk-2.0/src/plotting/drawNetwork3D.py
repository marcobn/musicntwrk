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
import community as cm
import plotly.graph_objects as go

import numpy as np

def drawNetwork3D(nodes,edges,scale=5,seed=None,colorscale='haline',title='',width=800,height=800):
    # draw network in 3D using plotly
    net = nx.from_pandas_edgelist(edges,'Source','Target','Weight')
    spring_3D = nx.spring_layout(net,dim=3,seed=seed)
    Num_nodes = len(nodes)

    df = np.array(nodes)
    idx = []
    for i,n in enumerate(net.nodes()):
        idx.append(int(n))
    df = df[idx]
    
    if len(df.shape) == 1:
        df = np.reshape(df,(len(df),1))
    nodelabel = dict(zip(np.linspace(0,len(df[:,0])-1,len(df[:,0]),dtype=int),df[:,0]))
    labels = list(nodelabel.values())
    
    part = cm.best_partition(net)
    values = [part.get(node) for node in net.nodes()]
    d = nx.degree(net)
    scale = scale
    dsize = np.array([(d[v]+1)*scale for v in net.nodes()])

    #we need to seperate the X,Y,Z coordinates for Plotly
    
    x_nodes = np.array([spring_3D[str(idx[i])][0] for i,n in enumerate(df)])
    y_nodes = np.array([spring_3D[str(idx[i])][1] for i,n in enumerate(df)])
    z_nodes = np.array([spring_3D[str(idx[i])][2] for i,n in enumerate(df)])

    #We also need a list of edges to include in the plot
    edge_list = net.edges()

    #we  need to create lists that contain the starting and ending coordinates of each edge.
    x_edges=[]
    y_edges=[]
    z_edges=[]

    #need to fill these with all of the coordiates
    def getlabel(source,target):
        tmp0 = edges[edges['Source']==source]
        tmp1 = tmp0[tmp0['Target']==target]
        try:
            if np.array(tmp1['Label']).size == 0:
                tmp0 = edges[edges['Source']==target]
                tmp1 = tmp0[tmp0['Target']==source]
            return(np.array(tmp1['Label'])[0])
        except:
            return(None)
    
    edge_label = []
    for edge in edge_list:
        #format: [beginning,ending,None]
        x_coords = [spring_3D[edge[0]][0],spring_3D[edge[1]][0],None]
        x_edges += x_coords

        y_coords = [spring_3D[edge[0]][1],spring_3D[edge[1]][1],None]
        y_edges += y_coords

        z_coords = [spring_3D[edge[0]][2],spring_3D[edge[1]][2],None]
        z_edges += z_coords
        
        edge_label += [getlabel(edge[0],edge[1]),getlabel(edge[1],edge[0]),None]
        
    #create a trace for the edges
    trace_edges = go.Scatter3d(x=x_edges,
                            y=y_edges,
                            z=z_edges,
                            mode='lines',
                            line=dict(color='black', width=3),
                            text=edge_label,
                            hoverinfo='text')

    #create a trace for the nodes
    trace_nodes = go.Scatter3d(x=x_nodes,
                               y=y_nodes,
                               z=z_nodes,
                               mode='markers',
                               marker=dict(symbol='circle',
                                      size=dsize,
                                      color=values,
                                      colorscale=colorscale),
                               text=labels,
                               hoverinfo='text')

    #we need to set the axis for the plot 
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')

    #also need to create the layout for our plot
    layout = go.Layout(title=title,
                    width=width,
                    height=height,
                    showlegend=False,
                    scene=dict(xaxis=dict(axis),
                            yaxis=dict(axis),
                            zaxis=dict(axis),
                            ),
                    margin=dict(t=100),
                    hovermode='closest')

    #Include the traces we want to plot and create a figure
    data = [trace_edges, trace_nodes]
    fig = go.Figure(data=data, layout=layout)

    fig.show()