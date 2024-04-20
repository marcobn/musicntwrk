#
# msctools: my collection of composing and performing tools in python
#
# © 2023 Marco Buongiorno Nardelli
#

import itertools
import numpy as np
import networkx as nx

import re, sys, os, time

import music21 as m21

	
import json
import pandas as pd
from ..harmony.scoreFilter import scoreFilter
from ..harmony.changePoint import changePoint


from .. import musicntwrk
from ..musicntwrk import PCSet
from ..plotting.drawNetwork import drawNetwork
mk = musicntwrk.musicntwrk(TET=12)

from ..harmony.harmonicDesign import harmonicDesign
from ..harmony.networkHarmonyGen import networkHarmonyGen
from ..harmony.rhythmicDesign import rhythmicDesign
from ..harmony.scoreDesign import scoreDesign

from .utils import importSoundfiles

def chinese_postman(graph,starting_node=None,verbose=False):
	
	def get_shortest_distance(graph, pairs, edge_weight_name):
		return {pair : nx.dijkstra_path_length(graph, pair[0], pair[1], edge_weight_name) for pair in pairs}
	
	def create_graph(node_pairs_with_weights, flip_weight = True):
		graph = nx.Graph()
		for k,v in node_pairs_with_weights.items():
			wt = -v if flip_weight else v
			graph.add_edge(k[0], k[1], **{'distance': v, 'weight': wt})
		return graph
	
	def create_new_graph(graph, edges, starting_node=None):
		g = nx.MultiGraph()
		for edge in edges:
			aug_path  = nx.shortest_path(graph, edge[0], edge[1], weight="distance")
			aug_path_pairs  = list(zip(aug_path[:-1],aug_path[1:]))
			
			for aug_edge in aug_path_pairs:
				aug_edge_attr = graph[aug_edge[0]][aug_edge[1]]
				g.add_edge(aug_edge[0], aug_edge[1], attr_dict=aug_edge_attr)
		for edge in graph.edges(data=True):
			g.add_edge(edge[0],edge[1],attr_dict=edge[2:])
		return g
	
	def create_eulerian_circuit(graph, starting_node=starting_node):
		return list(nx.eulerian_circuit(graph,source=starting_node))
	
	odd_degree_nodes = [node for node, degree in dict(nx.degree(graph)).items() if degree%2 == 1]
	odd_degree_pairs = itertools.combinations(odd_degree_nodes, 2)
	odd_nodes_pairs_shortest_path = get_shortest_distance(graph, odd_degree_pairs, "distance")
	graph_complete_odd = create_graph(odd_nodes_pairs_shortest_path, flip_weight=True)
	if verbose:
		print('Number of nodes (odd): {}'.format(len(graph_complete_odd.nodes())))
		print('Number of edges (odd): {}'.format(len(graph_complete_odd.edges())))
	odd_matching_edges = nx.algorithms.max_weight_matching(graph_complete_odd, True)
	if verbose: print('Number of edges in matching: {}'.format(len(odd_matching_edges)))
	multi_graph = create_new_graph(graph, odd_matching_edges)
	
	return(create_eulerian_circuit(multi_graph, starting_node))

def BachBAChorale(chorale,random=False,nseed=None):
	
	# Chorale reconstruction from J.S. Bach with BA networks of pitch and rhythm
	# Uses the music21 corpus
	
	
	
	bachChorale = m21.corpus.parse(chorale).corpusFilepath
	seq,chords,_ = mk.dictionary(space='score',scorefil=bachChorale,music21=True,show=False)
	bnodes,bedges,_ = mk.network(space='score',seq=seq,ntx=False,general=True,distance='euclidean',
										grphtype='directed')
	
	euseq,_,_ = harmonicDesign(mk,len(bnodes),bnodes,bedges,nedges=2,seed=nseed,reverse=True,display=False,write=False)
	
	# make all chords of cardinality 4
	ch = np.zeros((len(euseq),4))
	for i,c in enumerate(euseq):
		if len(c) == 4:
			ch[i,:] = np.array(c)
		elif len(c) == 3:
			ch[i,:3] = np.array(c)
			ch[i,3] = c[0]
		elif len(c) == 2:
			ch[i,:2] = np.array(c)
			ch[i,2] = c[0]
			ch[i,3] = c[1]
		else:
			ch[i,:] = c
	if random:
		c = []
		for i in range(len(euseq)):
			c.append(np.random.permutation(np.asarray(ch[i])))
		c = np.array(c)
	else:
		c = ch
	
	return(c)

def hungarianArtNetwork(jfile):
	
	with open(jfile,) as f:
		templates = json.load(f)
	
	nodesdict = dict(zip(templates['info']['nodes']['labels'],np.linspace(0,2258,2259,dtype=int)))
	linklabels = templates['info']['links']['labels']
	
	source = []
	target = []
	label = []
	for n in range(len(linklabels)):
		source.append(nodesdict[linklabels[n].split('; ')[0]])
		target.append(nodesdict[linklabels[n].split('; ')[1]])
		label.append(linklabels[n])
	links = []
	for n in range(len(linklabels)):
		links.append([source[n],target[n],1.0,label[n]])
		
	
	hunedges = pd.DataFrame(links,columns=['Source','Target','Weight','Label'])
	Gx = nx.from_pandas_edgelist(hunedges,'Source','Target',['Weight','Label'])
	
	source = []
	target = []
	label = []
	nlinks = 0
	for n in range(len(linklabels)):
		if Gx.degree[nodesdict[linklabels[n].split('; ')[0]]] < 6 or \
			Gx.degree[nodesdict[linklabels[n].split('; ')[1]]] < 6:
			pass
		else:
			source.append(nodesdict[linklabels[n].split('; ')[0]])
			target.append(nodesdict[linklabels[n].split('; ')[1]])
			label.append(linklabels[n])
			nlinks += 1
	filteredlinks = []
	for n in range(nlinks):
		filteredlinks.append([source[n],target[n],1.0,label[n]])
		
	hunedgesfilt = pd.DataFrame(filteredlinks,columns=['Source','Target','Weight','Label'])
	Gxf = nx.from_pandas_edgelist(hunedgesfilt,'Source','Target',['Weight','Label'])
	Gcc = Gxf.subgraph(sorted(nx.connected_components(Gxf), key=len, reverse=True)[0])
	
	path = chinese_postman(Gcc,starting_node=np.sort(Gcc.degree,axis=0)[-1][0],verbose=False)
	
	seq = []
	for p in path:
		seq.append(p[0])
	seq.append(path[-1][1])
	value,valuef,filtered = scoreFilter(seq,None,thr=3,plot=False)
	
	sections = changePoint(valuef,penalty=2.0,plot=False)
	regions = [None]*(len(sections)-1)
	for i in range(len(sections)-1):
		regions[i] = valuef[sections[i]:sections[i+1]]
	
	return(regions)

def chordDistr(mode='scale',random=True,seed=1010,step=7,extended=False):
	
	if mode == 'chord':
	# build an abstract distribution of chords
		Maj = np.array([0,4,7,0])
		Min = np.array([0,3,7,0])
		Maj7 = np.array([0,4,7,11])
		Min7 = np.array([0,3,7,10])
		Dom7 = np.array([0,4,7,10])
		Dim7 = np.array([0,3,6,9])
		Aug7 = np.array([0,4,8,11])
		
		tetrachords = []
		for p in range(12):
			tetrachords.append((Maj+p)%12)
			tetrachords.append((Maj+p)%12)
			tetrachords.append((Maj7+p)%12)
			tetrachords.append((Min+p)%12)
			tetrachords.append((Min+p)%12)
			tetrachords.append((Min7+p)%12)
			tetrachords.append((Dom7+p)%12)
			tetrachords.append((Dim7+p)%12)
			tetrachords.append((Aug7+p)%12)
		
		tetrachords = np.array(tetrachords)
	
		if random:
			rng = np.random.default_rng(seed)
			tetra = np.copy(tetrachords)
			rng.shuffle(tetra)
			c = []
			for i in range(len(tetra)):
				c.append(rng.permutation(np.asarray(tetra[i])))
			tetra = np.array(c)
			
		return(tetra)
	
	if mode == 'scale':
		# build a distribution of chords based on scales
		if extended: 
			chords = np.zeros((12,14,4),dtype=int)
		else:
			chords = np.zeros((12,7,4),dtype=int)
		chords[0,0,:] = np.array([0,4,7,0])
		chords[0,1,:] = np.array([2,5,9,2])
		chords[0,2,:] = np.array([4,7,11,4])
		chords[0,3,:] = np.array([5,9,0,5])
		chords[0,4,:] = np.array([7,11,2,5])
		chords[0,5,:] = np.array([9,0,4,9])
		chords[0,6,:] = np.array([11,2,5,9])
		if extended:
			chords[0,7,:] = np.array([0,4,7,11])
			chords[0,8,:] = np.array([2,5,9,0])
			chords[0,9,:] = np.array([4,7,11,2])
			chords[0,10,:] = np.array([5,9,0,4])
			chords[0,11,:] = np.array([7,11,2,5])
			chords[0,12,:] = np.array([9,0,4,7])
			chords[0,13,:] = np.array([11,2,5,9])
		
		if random:
			rng = np.random.default_rng(seed)
			rng.shuffle(chords[0])
			
			per  = int(12/np.unique([(l*step)%12 for l in range(12)]).shape[0])
			
			n = 0
			for p in range(per):
				for i in [(l*step)%12 for l in range(int(12/per))]:
					chords[n,:,:] = (chords[0,:,:] + i + p)%12
					n += 1
		else:
			per  = int(12/np.unique([(l*step)%12 for l in range(12)]).shape[0])
			
			n = 0
			for p in range(per):
				for i in [(l*step)%12 for l in range(int(12/per))]:
					chords[n,:,:] = (chords[0,:,:] + i + p)%12
					n += 1
		
		chords = np.reshape(chords,(12*7,4))
		
		return(chords)

def fourPartScore(c,dirpaths,N,Nc,nseed=[None,None,None,None],ref=['e','e','e','e']):
	
	assert len(dirpaths) == 4
	
	dictrtm,_ = mk.dictionary(space='rhythmP',N=N,Nc=Nc,REF=ref[0])
	nodes,edges = mk.network(space='rLead',dictionary=dictrtm,thup=30,thdw=0.1,
							distance='euclidean',prob=1,write=False)
	durations = rhythmicDesign(dictrtm,len(nodes),2,nodes,edges,random=True,seed=nseed[0],reverse=True)

	part1 = m21.stream.Stream()
	part1.insert(0, m21.meter.TimeSignature('4/4'))
	for i in range(c.shape[0]):
		nota = m21.note.Note(c[i,0])
		nota.duration = durations[i%len(durations)]
		nota.octave = np.random.choice([3,4,5])
		part1.append(nota)
		
	dictrtm,_ = mk.dictionary(space='rhythmP',N=N,Nc=Nc,REF=ref[1])
	nodes,edges = mk.network(space='rLead',dictionary=dictrtm,thup=30,thdw=0.1,
							distance='euclidean',prob=1,write=False)
	durations = rhythmicDesign(dictrtm,len(nodes),2,nodes,edges,random=True,seed=nseed[1],reverse=False)
	
	part2 = m21.stream.Stream()
	part2.insert(0, m21.meter.TimeSignature('4/4'))
	for i in range(c.shape[0]):
		nota = m21.note.Note(c[i,1])
		nota.duration = durations[i%len(durations)]
		nota.octave = np.random.choice([3,4,5])
		part2.append(nota)
	
	dictrtm,_ = mk.dictionary(space='rhythmP',N=N,Nc=Nc,REF=ref[2])
	nodes,edges = mk.network(space='rLead',dictionary=dictrtm,thup=30,thdw=0.1,
							distance='euclidean',prob=1,write=False)
	durations = rhythmicDesign(dictrtm,len(nodes),2,nodes,edges,random=True,seed=nseed[2],reverse=True)
	
	part3 = m21.stream.Stream()
	part3.insert(0, m21.meter.TimeSignature('4/4'))
	for i in range(c.shape[0]):
		nota = m21.note.Note(c[i,2])
		nota.duration = durations[i%len(durations)]
		nota.octave = np.random.choice([3,4,5])
		part3.append(nota)
	
	dictrtm,_ = mk.dictionary(space='rhythmP',N=N,Nc=Nc,REF=ref[3])
	nodes,edges = mk.network(space='rLead',dictionary=dictrtm,thup=30,thdw=0.1,
							distance='euclidean',prob=1,write=False)
	durations = rhythmicDesign(dictrtm,len(nodes),2,nodes,edges,random=True,seed=nseed[3],reverse=False)
	
	part4 = m21.stream.Stream()
	part4.insert(0, m21.meter.TimeSignature('4/4'))
	for i in range(c.shape[0]):
		nota = m21.note.Note(c[i,2])
		nota.duration = durations[i%len(durations)]
		nota.octave = np.random.choice([3,4,5])
		part4.append(nota)

	S = []
	for s in part1.recurse().notes:
		try:
			S.append([str(s.pitch),float(str(s.duration).split()[-1][:-1])])
		except:
			S.append([str(s.pitch),float(str(s.duration).split()[-1][:-1].split('/')[0])/\
								   float(str(s.duration).split()[-1][:-1].split('/')[1])])
	A = []
	for s in part2.recurse().notes:
		try:
			A.append([str(s.pitch),float(str(s.duration).split()[-1][:-1])])
		except:
			A.append([str(s.pitch),float(str(s.duration).split()[-1][:-1].split('/')[0])/\
								   float(str(s.duration).split()[-1][:-1].split('/')[1])])
				
	T = []
	for s in part3.recurse().notes:
		try:
			T.append([str(s.pitch),float(str(s.duration).split()[-1][:-1])])
		except:
			T.append([str(s.pitch),float(str(s.duration).split()[-1][:-1].split('/')[0])/\
								   float(str(s.duration).split()[-1][:-1].split('/')[1])])
				
	B = []
	for s in part4.recurse().notes:
		try:
			B.append([str(s.pitch),float(str(s.duration).split()[-1][:-1])])
		except:
			B.append([str(s.pitch),float(str(s.duration).split()[-1][:-1].split('/')[0])/\
								   float(str(s.duration).split()[-1][:-1].split('/')[1])])
				
	files = sorted(importSoundfiles(dirpath=dirpaths[0],filepath='*.wav'))

	idx = []
	fil = []
	for i,f in enumerate(files):
		idx.append(i)
		fil.append(f.split('/')[-1].split('.')[0][2:])
	
	Sdict = dict(zip(fil,idx))
	
	files = sorted(importSoundfiles(dirpath=dirpaths[1],filepath='*.wav'))
	
	idx = []
	fil = []
	for i,f in enumerate(files):
		idx.append(i)
		fil.append(f.split('/')[-1].split('.')[0][2:])
	
	Adict = dict(zip(fil,idx))
	
	files = sorted(importSoundfiles(dirpath=dirpaths[2],filepath='*.wav'))
	
	idx = []
	fil = []
	for i,f in enumerate(files):
		idx.append(i)
		fil.append(f.split('/')[-1].split('.')[0][2:])
	
	Tdict = dict(zip(fil,idx))
	
	files = sorted(importSoundfiles(dirpath=dirpaths[3],filepath='*.wav'))
	
	idx = []
	fil = []
	for i,f in enumerate(files):
		idx.append(i)
		fil.append(f.split('/')[-1].split('.')[0][2:])
	
	Bdict = dict(zip(fil,idx))
	
	Sseq = []
	Sdur = []
	Aseq = []
	Adur = []
	Tseq = []
	Tdur = []
	Bseq = []
	Bdur = []
	for n in S:
		try:
			Sseq.append(Sdict[n[0]])
			Sdur.append(n[1])
		except:
			if n[0][-1] == '5':
				new = n[0][:-1]+str(int(n[0][-1])-1)
			elif n[0][-1] == '3':
				new = n[0][:-1]+str(int(n[0][-1])+1)
			else:
				new = n[0]
			Sseq.append(Sdict[new])
			Sdur.append(n[1])
	for n in A:
		try:
			Aseq.append(Adict[n[0]])
			Adur.append(n[1])
		except:
			if n[0][-1] == '5':
				new = n[0][:-1]+str(int(n[0][-1])-1)
			elif n[0][-1] == '3':
				new = n[0][:-1]+str(int(n[0][-1])+1)
			else:
				new = n[0]
			Aseq.append(Adict[new])
			Adur.append(n[1])
	for n in T:
		try:
			Tseq.append(Tdict[n[0]])
			Tdur.append(n[1])
		except:
			if n[0][-1] == '5':
				new = n[0][:-1]+str(int(n[0][-1])-1)
			elif n[0][-1] == '3':
				new = n[0][:-1]+str(int(n[0][-1])+1)
			else:
				new = n[0]
			Tseq.append(Sdict[new])
			Tdur.append(n[1])
	for n in B:
		try:
			Bseq.append(Bdict[n[0]])
			Bdur.append(n[1])
		except:
			if n[0][-1] == '5':
				new = n[0][:-1]+str(int(n[0][-1])-1)
			elif n[0][-1] == '3':
				new = n[0][:-1]+str(int(n[0][-1])+1)
			else:
				new = n[0]
			Bseq.append(Sdict[new])
			Bdur.append(n[1])
		
	soprano = [Sseq,Sdur]
	alto = [Aseq,Adur]
	tenor = [Tseq,Tdur]
	bass = [Bseq,Bdur]
	
	return([soprano,alto,tenor,bass])

def fourPartScoreP(c,dirpaths,N,Nc,nseed=[None,None,None,None],ref=['e','e','e','e']):
	
	assert len(dirpaths) == 4
	
	dictrtm,_ = mk.dictionary(space='rhythmP',N=N,Nc=Nc,REF=ref[0])
	nodes,edges = mk.network(space='rLead',dictionary=dictrtm,thup=30,thdw=0.1,
							distance='euclidean',prob=1,write=False)
	durations = rhythmicDesign(dictrtm,len(nodes),2,nodes,edges,random=True,seed=nseed[0],reverse=True)

	part1 = m21.stream.Stream()
	part1.insert(0, m21.meter.TimeSignature('4/4'))
	for i in range(c.shape[0]):
		nota = m21.note.Note(c[i,0])
		nota.duration = durations[i%len(durations)]
		nota.octave = np.random.choice([4,5,6])
		part1.append(nota)
		
	dictrtm,_ = mk.dictionary(space='rhythmP',N=N,Nc=Nc,REF=ref[1])
	nodes,edges = mk.network(space='rLead',dictionary=dictrtm,thup=30,thdw=0.1,
							distance='euclidean',prob=1,write=False)
	durations = rhythmicDesign(dictrtm,len(nodes),2,nodes,edges,random=True,seed=nseed[1],reverse=False)
	
	part2 = m21.stream.Stream()
	part2.insert(0, m21.meter.TimeSignature('4/4'))
	for i in range(c.shape[0]):
		nota = m21.note.Note(c[i,1])
		nota.duration = durations[i%len(durations)]
		nota.octave = np.random.choice([4,5,6])
		part2.append(nota)
	
	dictrtm,_ = mk.dictionary(space='rhythmP',N=N,Nc=Nc,REF=ref[2])
	nodes,edges = mk.network(space='rLead',dictionary=dictrtm,thup=30,thdw=0.1,
							distance='euclidean',prob=1,write=False)
	durations = rhythmicDesign(dictrtm,len(nodes),2,nodes,edges,random=True,seed=nseed[2],reverse=True)
	
	part3 = m21.stream.Stream()
	part3.insert(0, m21.meter.TimeSignature('4/4'))
	for i in range(c.shape[0]):
		nota = m21.note.Note(c[i,2])
		nota.duration = durations[i%len(durations)]
		nota.octave = np.random.choice([4,5,6])
		part3.append(nota)
	
	dictrtm,_ = mk.dictionary(space='rhythmP',N=N,Nc=Nc,REF=ref[3])
	nodes,edges = mk.network(space='rLead',dictionary=dictrtm,thup=30,thdw=0.1,
							distance='euclidean',prob=1,write=False)
	durations = rhythmicDesign(dictrtm,len(nodes),2,nodes,edges,random=True,seed=nseed[3],reverse=False)
	
	part4 = m21.stream.Stream()
	part4.insert(0, m21.meter.TimeSignature('4/4'))
	for i in range(c.shape[0]):
		nota = m21.note.Note(c[i,3])
		nota.duration = durations[i%len(durations)]
		nota.octave = np.random.choice([4,5,6])
		part4.append(nota)

	S = []
	for s in part1.recurse().notes:
		try:
			S.append([str(s.pitch),float(str(s.duration).split()[-1][:-1])])
		except:
			S.append([str(s.pitch),float(str(s.duration).split()[-1][:-1].split('/')[0])/\
								   float(str(s.duration).split()[-1][:-1].split('/')[1])])
	A = []
	for s in part2.recurse().notes:
		try:
			A.append([str(s.pitch),float(str(s.duration).split()[-1][:-1])])
		except:
			A.append([str(s.pitch),float(str(s.duration).split()[-1][:-1].split('/')[0])/\
								   float(str(s.duration).split()[-1][:-1].split('/')[1])])
				
	T = []
	for s in part3.recurse().notes:
		try:
			T.append([str(s.pitch),float(str(s.duration).split()[-1][:-1])])
		except:
			T.append([str(s.pitch),float(str(s.duration).split()[-1][:-1].split('/')[0])/\
								   float(str(s.duration).split()[-1][:-1].split('/')[1])])
				
	B = []
	for s in part4.recurse().notes:
		try:
			B.append([str(s.pitch),float(str(s.duration).split()[-1][:-1])])
		except:
			B.append([str(s.pitch),float(str(s.duration).split()[-1][:-1].split('/')[0])/\
								   float(str(s.duration).split()[-1][:-1].split('/')[1])])
				
	files = sorted(importSoundfiles(dirpath=dirpaths[0],filepath='*.wav'))

	idx = []
	fil = []
	for i,f in enumerate(files):
		idx.append(i)
		fil.append(f.split('/')[-1].split('.')[0][2:])
	
	Sdict = dict(zip(fil,idx))
	
	files = sorted(importSoundfiles(dirpath=dirpaths[1],filepath='*.wav'))
	
	idx = []
	fil = []
	for i,f in enumerate(files):
		idx.append(i)
		fil.append(f.split('/')[-1].split('.')[0][2:])
	
	Adict = dict(zip(fil,idx))
	
	files = sorted(importSoundfiles(dirpath=dirpaths[2],filepath='*.wav'))
	
	idx = []
	fil = []
	for i,f in enumerate(files):
		idx.append(i)
		fil.append(f.split('/')[-1].split('.')[0][2:])
	
	Tdict = dict(zip(fil,idx))
	
	files = sorted(importSoundfiles(dirpath=dirpaths[3],filepath='*.wav'))
	
	idx = []
	fil = []
	for i,f in enumerate(files):
		idx.append(i)
		fil.append(f.split('/')[-1].split('.')[0][2:])
	
	Bdict = dict(zip(fil,idx))
	
	Sseq = []
	Sdur = []
	Aseq = []
	Adur = []
	Tseq = []
	Tdur = []
	Bseq = []
	Bdur = []
	for n in S:
		try:
			Sseq.append(Sdict[n[0]])
			Sdur.append(n[1])
		except:
			if n[0][-1] == '5':
				new = n[0][:-1]+str(int(n[0][-1])-1)
			elif n[0][-1] == '3':
				new = n[0][:-1]+str(int(n[0][-1])+1)
			else:
				new = n[0]
			Sseq.append(Sdict[new])
			Sdur.append(n[1])
	for n in A:
		try:
			Aseq.append(Adict[n[0]])
			Adur.append(n[1])
		except:
			if n[0][-1] == '5':
				new = n[0][:-1]+str(int(n[0][-1])-1)
			elif n[0][-1] == '3':
				new = n[0][:-1]+str(int(n[0][-1])+1)
			else:
				new = n[0]
			Aseq.append(Adict[new])
			Adur.append(n[1])
	for n in T:
		try:
			Tseq.append(Tdict[n[0]])
			Tdur.append(n[1])
		except:
			if n[0][-1] == '5':
				new = n[0][:-1]+str(int(n[0][-1])-1)
			elif n[0][-1] == '3':
				new = n[0][:-1]+str(int(n[0][-1])+1)
			else:
				new = n[0]
			Tseq.append(Sdict[new])
			Tdur.append(n[1])
	for n in B:
		try:
			Bseq.append(Bdict[n[0]])
			Bdur.append(n[1])
		except:
			if n[0][-1] == '5':
				new = n[0][:-1]+str(int(n[0][-1])-1)
			elif n[0][-1] == '3':
				new = n[0][:-1]+str(int(n[0][-1])+1)
			else:
				new = n[0]
			Bseq.append(Sdict[new])
			Bdur.append(n[1])
		
	soprano = [Sseq,Sdur]
	alto = [Aseq,Adur]
	tenor = [Tseq,Tdur]
	bass = [Bseq,Bdur]
	
	return([soprano,alto,tenor,bass])

	
def sequence(nxmodel='barabasi_albert_graph',*args):
    ''' 
    : sequence defined by the eulerian path on a network
    : network models can be found here: 
    : https://networkx.org/documentation/stable/reference/generators.html
                        : arguments are passed through *args
    '''

    mynetx = getattr(nx,nxmodel)
    Gx = mynetx(*args)
    chino = chinese_postman(Gx,None,verbose=False)
    seq = [chino[0][0]]
    for s in range(1,len(chino)):
        seq.append(chino[s][1])

    return(seq)