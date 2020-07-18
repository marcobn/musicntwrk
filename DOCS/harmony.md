## `harmony`

Utilities for harmonic analysis, design and autonomous scoring

See example in the notebooks on GitHub for possible uses of
the modules contained here.

_Harmonic analysis functions_

<span>__def scoreFilter__(seq,chords,thr=0,plot=False)</span>  
filter out low recurring chords to facilitate change point detection and
score partitioning in regions. Needs the output of
<span>readScore</span>

<span><span>seq (int)</span> </span>
list of pcs for each chords extracted from the score

<span><span>chords (music21 <span>object</span>)</span> </span>
chords as extracted by <span>music21</span>

<span>*Returns*</span>

<span><span>value, valuef (int)</span> </span>
integer identifier for the full and filtered chord sequence

<span><span>filtered (int)</span> </span>
sequence of filtered chords (by identifier as defined above)

<span><span>fmeasure (int)</span> </span>
measure number to which the filtered chord belongs

<span>__def changePoint__(value, vmodel=’rbf’, penalty=1.0, vbrakepts=None,
plot = False)</span>  
run a change point detection algorithm to isolate sections on the
filtered score. Uses the implementation in <span>ruptures</span>.

<span><span>value (int)</span> </span>
list of pcs for each chords as extracted from <span>scoreFilter</span>

<span><span>other variables</span> </span>
documentation for the other variables at
https://pypi.org/project/ruptures/

<span>*Returns*</span>

<span><span>sections (list of int)</span> </span>
list of breakpoints in the score sectioning (by position in the filtered
chord sequence)

<span>__def keySections__(sections, GxsecDi, dnodes)</span>  

region key identification

<span><span>sections (int)</span> </span>
list of region identified by the <span>  
changePoint</span> algorithm

<span><span>GxsecDi (networkx graph <span>object</span>)</span> </span>
score sub-network as computed by <span>scoreSubNetwork</span>

<span><span>dnodes (<span>pandas</span> dataframe)</span> </span>
dataframe of node label of the full score network

<span>*Returns*</span>

<span><span>key (string)</span> </span>
key associated with given chord in the sequence as identified by a
"prevalent chord" algorithm

<span><span>keySections (pandas dataframe)</span> </span>
dataframe that summarizes the key analysis results by ’section’,’chord
range’,’prevalent\_chord’,’region’

<span>__def tonalAnalysis__(chords, sections, key, enharm=\[\[’C’,’C’\]\],
write=None)</span>  
roman numeral analysis of regions in the full score

<span><span>sections (int)</span> </span>
see above

<span><span>chords (music21 <span>object</span>)</span> </span>
see above

<span><span>key (string)</span> </span>
see above

<span><span>enharm (list of strings)</span> </span>
optional enharmonic table for pitch respelling

<span>*Returns*</span>

<span><span>analysis (pandas dataframe)</span> </span>
dataframe that summarizes the full analysis of the score, including
roman numerals - the results can be visualized in musicxml format if
<span>write = True</span>.

_Harmonic design functions and autonomous scoring_

<span>__def chinese\_postman__(graph, starting\_node)</span>  
solve the optimal routing problem of the Chinese postman on an assigned
network

<span><span>graph (networkx graph <span>object</span>)</span> </span>
graph on which to find the optimal path

<span><span>starting\_node (int)</span> </span>
node from where to start the path

<span>*Returns*</span>

<span><span>graph (networkx graph <span>object</span>)</span> </span>
directed graph with optimal path (used in <span>harmonicDesign</span>

<span>__def networkHarmonyGen__(mk, descriptor=None, dictionary=None, thup=
None, thdw=None, names=None, distance=None, probs=None, write= None,
pcslabel=None)</span>  
probabilistic chord distribution based on geometric distances. It is a
wrapper around the modules in <span>networks</span>. See the discussion
in _networks_ for the description of the variables. Only
additions are:

<span><span>names (list of strings)</span> </span>
list of operators to slice the network of pitches using
<span>vLeadNetworkByName</span>

<span><span>probs (list of floats)</span> </span>
list of probabilities to slice the network of pitches using
<span>vLeadNetwork</span> by distance

<span>*Returns*</span>

<span><span>nodes, edges (pandas dataframe)</span> </span>
nodes and edges of the generated network

<span>__def harmonicDesign__(mk, nnodes, nedges, refnodes, refedges, nstart
= None, seed=None, reverse=None, display=None, write= None)</span>  
generate a scale-free network according to the Barabasi-Albert model of
preferential attachment and assign chords to nodes using the output of
<span>networkHarmonyGen</span>

<span><span>mk (musicntwrk class)</span> </span>

<span><span>nnodes, nedges (int)</span> </span>
number of nodes and edges per node used to generate the scale free
network using <span>networkx.barabasi\_albert\_graph</span>. See
documentation on <span>networkx</span> for details.

<span><span>nodes, edges (pandas dataframe)</span> </span>
nodes and edges of the network generated by
<span>networkHarmonyGen</span>

<span>*Returns*</span>

<span><span>pitches (list of lists of int)</span> </span>
chord sequence of the generated network. If <span>display = True</span>
it draws the graph, if <span>write = True</span> it writes the sequence
as musicxml.

<span>__def rhythmicDesign__(dictionary, nnodes, nedges, refnodes, refedges,
nstart = None, seed=None, reverse=None, random=None)</span>  
builds a scale-free network according to the Barabasi-Albert model of
preferential attachment and assign rhythmic figures to nodes. Details as
above using the result of any of the rhythm network generation function
of _networks_.

<span><span>durations (list of lists of int)</span> </span>
duration sequence of the generated network.

<span>d__ef scoreDesign__(pitches, durations, fac=1, TET=12,
write=False)</span>  
using the sequences of pitches and rhythms from the above, generate a
score in musicxml

Helper functions

  - <span>__def tonalHarmonyCalculator__()</span>  
    multi-function harmony calculator: pitches, roman numerals, voice
    leading operations and more. See the in app HELP for a description
    of usage.

  - <span>__def tonalHarmonyModel__(mode=’minimal’)</span>  
    build a minimal <span>*harmony model*</span> based on voice leading
    operators (to be used in the calculator)

  - <span>__def tonnentz__(x,y)</span>  
    build the <span>*tonnentz*</span> for the specified x,y relation