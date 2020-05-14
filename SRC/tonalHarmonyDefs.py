# Tonal Harmony Calculator
#
# MUSIC𝄞NTWRK
#
# A python library for pitch class set and rhythmic sequences classification and manipulation, 
# the generation of networks in generalized music and sound spaces, and the sonification of arbitrary data
#
# Copyright (C) 2020 Marco Buongiorno Nardelli
# http://www.materialssoundmusic.com, mbn@unt.edu
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

from pcsPy import *
import pickle, copy
import itertools

def shortHands():
        # This is taken from roman.py in music21 and modified here
#        figureShorthands = {
#            '42':'7',
#            '43':'7',
#            '53': '',
#            '54': '52',
#            '3': '',
#            '63': '',
#            '6' : '',
#            '64' : '',
#            '65': '7',
#            '753': '7',
#            '#753':'7',
#            '75': '7[no3]',  # controversial perhaps
#            '73': '7[no5]',  # controversial perhaps
#            '752': '9[no3]',
#            '9753': '9',
#            '975': '9',  # controversial perhaps
#            '953': '9[no7]',  # controversial perhaps
#            '97': '9[no7][no5]',  # controversial perhaps
#            '32': '9[no5][no3]',
#            '95': '9[no7][no3]',  # controversial perhaps
#            '93': '9[no7][no5]',  # controversial perhaps
#        #  '653': '65',
#            '653': '7',
#            '6b53': '6b5',
#            '643': '7',
#        #  '642': '42',
#        #  '642': '7[no5]',
#            'o64' : 'o',
#            'o6b5':'o7',
#            'o5b3':'o',
#            'bb7b5b3': 'o7',
#            'bb7b53': 'o7',
#            # '6b5bb3': 'o65',
#            'b7b5b3': '/o7',
#        }
        figureShorthands = {
            '53': '',
            '3': '',
            '63': '6',
            '753': '7',
            '75': '7',  # controversial perhaps
            '73': '7',  # controversial perhaps
            '9753': '9',
            '975': '9',  # controversial perhaps
            '953': '9',  # controversial perhaps
            '97': '9',  # controversial perhaps
            '95': '9',  # controversial perhaps
            '93': '9',  # controversial perhaps
            '653': '65',
            '6b53': '6b5',
            '643': '43',
            '642': '42',
            'bb7b5b3': 'o7',
            'bb7b53': 'o7',
            # '6b5bb3': 'o65',
            'b7b5b3': 'ø7',
        }
        return(figureShorthands)


def enharmonicDictionary():
    keys = ['C','C#','D-','D','D#','E-','E','F','F#','G-','G','G#','A-','A','A#','B-','B']
    enharmonicDict = {}
    for k in keys:
        ini = m21.note.Note(40).nameWithOctave
        end = m21.note.Note(96).nameWithOctave
        major = [p for p in m21.scale.MajorScale(k).getPitches(ini,end)]
        minor = [p for p in m21.scale.MinorScale(k).getPitches(ini,end)]
        melod = [p for p in m21.scale.MelodicMinorScale(k).getPitches(ini,end)]
        harmo = [p for p in m21.scale.HarmonicMinorScale(k).getPitches(ini,end)]

        allscales = major+minor+melod+harmo

        clean = Remove(allscales)

        Cscale = sorted([str(f) for f in clean])
        Cscale = np.array(Cscale)

        Cmidi = []
        for n in Cscale:
            Cmidi.append(m21.pitch.Pitch(n).midi)
        Cmidi = np.array(Cmidi)
        idx = np.argsort(Cmidi)

        Cdict = dict(zip(Cmidi[idx[:]],Cscale[idx[:]]))

        for i in range(60,84):
            try:
                tmp = Cdict[i]
            except:
                Cdict.update({i:m21.note.Note(i).nameWithOctave})

        enharmonicDict.update({k:Cdict})
    return(enharmonicDict)

def getRN(a,key,TET=12):
    
    ch = np.copy(PCSet(a,UNI=False,ORD=False,TET=TET).pcs.tolist())

    for n in range(1,len(ch)):
        if ch[n] < ch[n-1]: ch[n] += TET
    ch += 60
    
    enharmonicDict = enharmonicDictionary()
    if key.islower():
        keyup = key.upper()
    else:
        keyup = key
    p = []
    for c in ch:
        p.append(enharmonicDict[keyup][c])
    n = m21.chord.Chord(p)

    rn = m21.roman.romanNumeralFromChord(n,m21.key.Key(key)).figure
#    fig =m21.roman.postFigureFromChordAndKey(n, m21.key.Key(key))
#    try:
#        fig = figureShorthands[fig]
#    except:
#        pass
    fig = ''
    return(n,rn+fig)
    
def lookupOps(ops,table,header,Pnumber='',ch1='',ch2=''):
        operator = ops
        tab = np.array(table)
        if Pnumber != '':
                try:
                        print('Pnumber of operator '+operator+' =',Pnumber[operator],'\n')
                except:
                        print('operator not found in Pnumber')
                        return
        idx,idy = np.where(tab == operator)
        for n in range(len(idy)):
                if ch1 == '' and ch2 == '':
                        print(str(header[idx[n]]).ljust(12),str(' ->\t'+header[idy[n]]).rjust(0))
                elif ch1 != '' and ch2 == '':
                        if ch1[0] == '?':
                                if (ch1[1:] in str(header[idx[n]])):
                                        print(str(header[idx[n]]).ljust(12),str(' ->\t'+header[idy[n]]).rjust(0))
                        else:
                                if (ch1 == str(header[idx[n]])):
                                    print(str(header[idx[n]]).ljust(12),str(' ->\t'+header[idy[n]]).rjust(0))
                elif ch2 != '' and ch1 == '':
                        if ch2[0] == '?':
                                if (ch2[1:] in str(header[idy[n]])):
                                        print(str(header[idx[n]]).ljust(12),str(' ->\t'+header[idy[n]]).rjust(0))
                        else:
                                if (ch2 == str(header[idy[n]])):
                                    print(str(header[idx[n]]).ljust(12),str(' ->\t'+header[idy[n]]).rjust(0))
                elif ch2 != '' and ch1 != '':
                        if ch1[0] == '?' and ch2[0] != '?':
                                if (ch1[1:] in str(header[idx[n]])) and (ch2 == str(header[idy[n]])):
                                        print(str(header[idx[n]]).ljust(12),str(' ->\t'+header[idy[n]]).rjust(0))
                        elif ch1[0] != '?' and ch2[0] == '?':
                                if (ch1 == str(header[idx[n]])) and (ch2[1:] in str(header[idy[n]])):
                                    print(str(header[idx[n]]).ljust(12),str(' ->\t'+header[idy[n]]).rjust(0))
                        elif ch1[0] != '?' and ch2[0] != '?':
                                if (ch1 == str(header[idx[n]])) and (ch2 == str(header[idy[n]])):
                                    print(str(header[idx[n]]).ljust(12),str(' ->\t'+header[idy[n]]).rjust(0))
                        else:
                                continue

def lookupProgr(ch1,ch2,table,header):

        tab = np.array(table)
        head = np.array(header)
        idx = np.where(head == ch1)
        idy = np.where(head == ch2)
        try:
                print(str(ch1).ljust(8),'->',tab[idx[0],idy[0]][0],'->',str(ch2).rjust(8))
        except:
                print('no operator found')


def lookupWrapper(table,head,ops='',cstart='',cend=''):
        # if only cstart != None and ops != None prints matching progressions
        # if cstart, cend != None (pcs or roman numerals) and ops == None: prints operator connecting the chords
        # if cstart = roman numeral and cend = pitch: returns pcs of the chord
        if ops != '':
                lookupOps(ops,table,head,ch1=cstart,ch2=cend)
                print('===============================')
        elif cstart != '' and cend != '' and ops == '':
#        print('Major tonal space')
                lookupProgr(cstart,cend,table,head)
                print('===============================')

def applyOps(name,chord,prt=True):
    # operate on the pcs with a relational distance operator
    op = []
    for num in re.findall("[-\d]+", name):
        op.append(int(num))
    op = np.asarray(op)
    pcs = []
    for num in re.findall("[-\d]+", chord):
        pcs.append(int(num))
    pcs = np.asarray(pcs)
    if len(op) == len(pcs):
        selfto = (PCSet(pcs).normalOrder()+op)%12
        if prt: print(PCSet(selfto).normalOrder().tolist())
        return([str(PCSet(selfto).normalOrder().tolist())])
    elif len(op) - len(pcs) == 1:
        # duplicate pitches
        c = np.zeros(len(op),dtype=int)
        pitch = PCSet(pcs,UNI=False,ORD=False).normalOrder()
        c[:len(op)-1] = pitch
        add = []
        for i in range(len(op)-1):
            c[len(op)-1] = pitch[i]
            selfto = (PCSet(c,UNI=False,ORD=False).normalOrder()+op)%12
            add.append(str(PCSet(selfto).normalOrder().tolist()))
        if prt: print(Remove(add))
        return(Remove(add))
    elif len(pcs) - len(op) == 1:
        # add a unison operator (0)
        add = []
        for i in range(pcs.shape[0]): 
            c = np.insert(op,i,0)
            selfto = (PCSet(pcs).normalOrder()+c)%12
            add.append(str(PCSet(selfto).normalOrder().tolist()))
        if prt: print(Remove(add))
        return(Remove(add))
    else:
        print('operation not defined')

def tonalPartition(seq,chords,nodes,Gx,Gxu,resolution=1.0,randomize=None,display=False):
    part = cm.best_partition(Gxu,resolution=resolution,randomize=randomize)
    dn = np.array(nodes)
    labe = []
    modu = []
    modul = []
    for m in range(len(dn)):
        labe.append(str(dn[m][0]))
        modu.append(part[str(m)])
        modul.append([str(dn[m][0]),Gx.degree()[str(m)],part[str(m)]])
    modul = pd.DataFrame(modul,columns=['Label','Degree','Modularity'])
    moduldict = dict(zip(labe,modu))

    if display:
    # display the score with modularity classes
        mc = []
        seqmc = []
        for n in range(len(seq)):
            p = PCSet(np.asarray(seq[n]))
            if p.pcs.shape[0] == 1:
                nn = ''.join(m21.chord.Chord(p.pcs.tolist()).pitchNames)
            else:
                nn = ''.join(m21.chord.Chord(p.normalOrder().tolist()).pitchNames)
            mc.append(nn)
            seqmc.append(moduldict[nn])

        l = 0
        analyzed = copy.deepcopy(chords)
        for c in analyzed.recurse().getElementsByClass('Chord'):
            c.closedPosition(forceOctave=4,inPlace=True)
            c.addLyric(str(l))
            c.addLyric(seqmc[l])
            l += 1
        analyzed.show('musicxml')
    return(moduldict,modul)

def scoreAnalysis(seq,moduldict,keydict,first=None,keychange=None,altrn=None,table='',verbose=False):
# Score analysis

# Read tonal harmony model
    f = open(table,'rb')
    head = pickle.load(f)
    table = pickle.load(f)
    f.close()
    tab = np.array(table)
    
# Dictionary of enharmonics for notes in music21 Chords
    enharmonicDict = enharmonicDictionary()
# Dictionary of shorthand symbols for rn extensions
    figureShorthands = shortHands()

# Determination of operators
    try:
        ntot = seq.shape[0]
    except:
        ntot = len(seq)
    ops = []
    for i in range(ntot):
        if i < ntot-1: 
            ops.append(generalizedOpsName(seq[i],seq[i+1])[1])

# First chord
    rn = []
    if first == None:
        ch = np.copy(seq[0])
        for n in range(1,len(ch)):
            if ch[n] < ch[n-1]: ch[n] += 12
        ch += 60
        n = m21.chord.Chord(ch.tolist())
        chord = ''.join(n.pitchNames)
        key = keydict[moduldict[chord]] 
        rn.append(m21.roman.romanNumeralFromChord(n, m21.key.Key(key)).figure)
    else:
        rn.append(first)
    # Full score
    nxt = ntot-1
    i = 0
    check = 0
    while i < nxt:   
        try:
    #         Manual control of rn (when needed)
            if altrn != None and i in altrn.keys():
                rn.append(altrn[i])
            else:
                idx,idy = np.where(tab == ops[i])
                tmp = []
                for n in range(len(idy)):
                    if (rn[i] == str(head[idx[n]])):
                        tmp.append(head[idy[n]])
                if len(tmp) == 1:
                    rn.append(tmp[0])
                else:
                    chord = ''.join(m21.chord.Chord(PCSet(seq[i]).normalOrder().tolist()).pitchNames)
                    key = keydict[moduldict[chord]]
                    for n in range(len(tmp)):
                        ch = m21.roman.RomanNumeral(tmp[n],m21.key.Key(key)).pitchClasses
                        if PCSet(ch).normalOrder().tolist() == seq[i+1]:
                            rn.append(str(tmp[n]))
                            break
            i += 1
        except Exception as e:
            if verbose: print(i,'try',type(e),e,chord)
            try:
                if check == i:
                    print('check error')
                    break
                else:
                    print('modulation at or before chord no. ',i)
                    check = i
                    rn.pop()
                    ch = np.copy(PCSet(seq[i-1]).normalOrder().tolist())
#                    probably not needed
#                    for n in range(1,len(ch)):
#                        if ch[n] < ch[n-1]: ch[n] += 12
                    ch += 60
                    m = m21.chord.Chord(ch.tolist())
                    key = keydict[moduldict[''.join(m.pitchNames)]]
#                     Manual control of modulations (when needed)
                    if keychange != None and i in keychange.keys():
                        key = keychange[i]
                    p = []
                    for c in ch:
                        p.append(enharmonicDict[key][c])
                    n = m21.chord.Chord(p)
                    chord = ''.join(n.pitchNames)                
                    try:
                        rnum = m21.roman.romanNumeralFromChord(n,m21.key.Key(key)).romanNumeralAlone
                        fig = m21.roman.postFigureFromChordAndKey(n,m21.key.Key(key))
                        try:
                            fig = figureShorthands[fig]
                        except:
                            pass
                        rn.append(rnum+fig)
                        if verbose: print(i-1,'except',n,rn[i-1],key,'\n')
                        i -= 1
                    except Exception as e:
                        print(type(e),e,chord)
                        break
            except Exception as e:
                print(type(e),e) 
                break
    nxt = i
    return(nxt,rn,ops)

def showAnalysis(nt,chords,seq,rn,ops,keydict,moduldict,wops=False,last=False,display=False):
# Create dictionary of score analysis
    reference = []
    for n in range(nt):
        try:
            chord = ''.join(m21.chord.Chord(PCSet(seq[n]).normalOrder().tolist()).pitchNames)
            entry = [PCSet(seq[n]).normalOrder(),chord,rn[n],ops[n],keydict[moduldict[chord]],moduldict[chord]]
            reference.append(entry)
        except:
            pass
    if last:
    # Add last chord
        ops.append(' ')
        chord = ''.join(m21.chord.Chord(PCSet(seq[nt]).normalOrder().tolist()).pitchNames)
        entry = [PCSet(seq[nt]).normalOrder(),chord,rn[nt],ops[nt],keydict[moduldict[chord]]]
        reference.append(entry)

    # Set dictionary as pandas dataframe
    analysis = pd.DataFrame(reference,columns=['pcs','chord','rn','ops','key','modul'])
    
    if display:
        # display the analyzed score
        l = 0
        analyzed = copy.deepcopy(chords)
        for c in analyzed.recurse().getElementsByClass('Chord'):
            c.closedPosition(forceOctave=4,inPlace=True)
            c.addLyric('')
            c.addLyric('')
            if wops: c.addLyric(str(ops[l]))
            c.addLyric(str(rn[l]))
            if l < nt-1: 
                l += 1
            else: 
                break
        analyzed.show('musicxml')
    return(analysis)

def tonnentz(x,y):
    tnz = pd.DataFrame(None,columns=['','','','','','','','',''])
    tmp = pd.DataFrame([[str((4*y)%12),'',str((x+3*y)%12),'',str((2*x+2*y)%12),'',str((3*x+y)%12),'',str((4*x)%12)]],
                       columns=['','','','','','','','',''])
    tnz = tnz.append(tmp)
    tmp = pd.DataFrame([['',str((3*y)%12),'',str((x+2*y)%12),'',str((2*x+y)%12),'',str((3*x)%12),'']],
                       columns=['','','','','','','','',''])
    tnz = tnz.append(tmp)
    tmp = pd.DataFrame([[str((-x+3*y)%12),'',str((2*y)%12),'',str((x+y)%12),'',str((2*x)%12),'',str((3*x-y)%12)]],
                       columns=['','','','','','','','',''])
    tnz = tnz.append(tmp)
    tmp = pd.DataFrame([['',str((-x+2*y)%12),'',str(y%12),'',str(x%12),'',str((2*x-y)%12),'']],
                       columns=['','','','','','','','',''])
    tnz = tnz.append(tmp)
    tmp = pd.DataFrame([[str((-2*x+2*y)%12),'',str((-x+y)%12),'',str(0),'',str((x-y)%12),'',str((2*x-2*y)%12)]],
                       columns=['','','','','','','','',''])
    tnz = tnz.append(tmp)
    tmp = pd.DataFrame([['',str((-2*x+y)%12),'',str(-x%12),'',str(-y%12),'',str((x-2*y)%12),'']],
                       columns=['','','','','','','','',''])
    tnz = tnz.append(tmp)
    tmp = pd.DataFrame([[str((-3*x+y)%12),'',str(-2*x%12),'',str((-x-y)%12),'',str(-2*y%12),'',str((x-3*y)%12)]],
                       columns=['','','','','','','','',''])
    tnz = tnz.append(tmp)
    tmp = pd.DataFrame([['',str(-3*x%12),'',str((-2*x-y)%12),'',str((-x-2*y)%12),'',str(-3*y%12),'']],
                       columns=['','','','','','','','',''])
    tnz = tnz.append(tmp)
    tmp = pd.DataFrame([[str(-4*x%12),'',str((-3*x-y)%12),'',str((-2*x-2*y)%12),'',str((-x-3*y)%12),'',str(-4*y%12)]],
                       columns=['','','','','','','','',''])
    tnz = tnz.append(tmp)

    return(tnz)

def chinese_postman(graph,starting_node):
        
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

    def create_eulerian_circuit(graph, starting_node=None):
        return list(nx.eulerian_circuit(graph,source=starting_node))
    
    odd_degree_nodes = [node for node, degree in dict(nx.degree(graph)).items() if degree%2 == 1]
    odd_degree_pairs = itertools.combinations(odd_degree_nodes, 2)
    odd_nodes_pairs_shortest_path = get_shortest_distance(graph, odd_degree_pairs, "distance")
    graph_complete_odd = create_graph(odd_nodes_pairs_shortest_path, flip_weight=True)
    odd_matching_edges = nx.algorithms.max_weight_matching(graph_complete_odd, True)
    multi_graph = create_new_graph(graph, odd_matching_edges)

    return(create_eulerian_circuit(multi_graph, starting_node))

def clearOutput():
        return

def showPitches():
        return
    
def showRN():
        return
