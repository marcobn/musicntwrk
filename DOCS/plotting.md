## Plotting and general utility functions

There are many utility functions (in <span>utils</span>) that are used
by other modules and that should be transparent to the average user. We
recall here only the ones that are used for computing distances in pitch
space.

<span>__def minimalNoBijDistance__(a, b, TET, distance)</span>  
calculates the minimal distance between two pcs of different cardinality
(non bijective)

<span><span>a,b (int)</span> </span>
pcs as lists or numpy arrays

<span><span>distance (str)</span> </span>
choice of norm in the musical space

<span>*Returns*</span>

<span><span>dist (float)</span> </span>
minimal distance

<span><span>r (array of int)</span> </span>
multiset corresponding to minimal distance

<span>__def generalizedOpsName__(a,b,TET,distance)</span>  
finds the voice leading operator that connects two pcs (also for non
bijective transformations)

<span><span>a,b (int)</span> </span>
pcs as lists or numpy arrays

<span><span>distance (str)</span> </span>
choice of norm in the musical space

<span>*Returns*</span>

<span><span>r (array of int)</span> </span>
multiset corresponding to minimal distance

<span><span>Op (string)</span> </span>
VL operator that connects the two pcs

Finally, it is worth mentioning the network plotting utility
<span>drawNetwork</span> in <span>plotting</span>:

<span>__def drawNetwork__(nodes, edges, forceiter=100, grphtype=’undirected’, dx
= 10, dy=10, colormap=’jet’, scale=1.0, drawlabels=True,
giant=False)</span>  
draws the network using <span>networkx</span> and
<span>matplotlib</span>

<span><span>nodes, edges (pandas dataframe)</span> </span>
nodes and edges of the network

<span><span>forceiter (floats)</span> </span>
iterations in the <span>networks</span> force layout

<span><span>grphtype (string)</span> </span>
’directed’ or ’undirected’

<span><span>dx, dy (floats)</span> </span>
dimensions of the canvas

<span><span>colormap (string)</span> </span>
colormap for <span>plt.get\_cmap(colormap)</span>

<span><span>scale (float)</span> </span>
scale factor for node radius

<span><span>drawlabels (logical)</span> </span>
draw labels on nodes

<span><span>giant (logical)</span> </span>
if <span>True</span> draws only the giant component of the network
