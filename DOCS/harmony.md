Utilities for harmonic analysis, design and autonomous scoring
	
See example in the notebooks on GitHub for possible uses of the functions contained here.

Harmonic analysis functions:

- `def keySections(sections,GxsecDi,dnodes):` - 
	region key identification

- `def changePoint(value,model='rbf',penalty=1.0,brakepts=None,plot=False):` -
	run a change point detection algorithm to isolate sections

- `def scoreAnalysis(seq,moduldict,keydict,first=None,keychange=None,altrn=None,table='',verbose=False):`

- `def scoreFilter(seq,chords,thr=0,plot=False):` - 
	filter out low recurring chords to facilitate change point detection and score partitioning in regions

- `def showAnalysis(nt,chords,seq,rn,ops,keydict,moduldict,wops=False,last=False,display=False):` - 
	display score with roman numeral analysis

- `def tonalAnalysis(chords,sections,key,enharm=[['C','C']],write=None):` -
	roman numeral analysis of regions

Harmonic design functions and autonomous scoring

- `def chinese_postman(graph,starting_node):` -
	solve the optimal routing problem of the chinese postman on an assigned network
		
- `def harmonicDesign(mk,nnodes,nedges,refnodes,refedges,nstart=None,seed=None,reverse=None,display=None,write=None):` - 
	build a scale-free network according to the Barabasi-Albert model of preferential attachment and assign chords to nodes

- `def networkHarmonyGen(mk,descriptor=None,dictionary=None,thup=None,thdw=None,names=None,distance=None,probs=None,write=None,pcslabel=None)` - 
	probabilistic chord distribution based on geometric distances 

- `def rhythmicDesign(dictionary,nnodes,nedges,refnodes,refedges,nstart=None,seed=None,reverse=None,random=None)` - 
	build a scale-free network according to the Barabasi-Albert model of preferential attachment and assign rhythmic figures to nodes

- `def scoreDesign(pitches,durations,fac=1,TET=12,write=False):`
	using the sequences of pitches and rhythms from the above, generate a score in musicxml

Helper functions

- `def tonalHarmonyCalculator():` - 
	multi-function harmony calculator: pitches, roman numerals, voice leading operations and more...

- `def tonalHarmonyModel(mode='minimal'):` - 
	build a minimal "harmony model" based on voice leading operators (to be used in the calculator)

- `def tonnentz(x,y):` - 
	build the tonnentz for the specified x,y relation