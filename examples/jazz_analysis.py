#%%

import re, sys, os, time
import music21 as m21

#%% from musicntwrk import musicntwrk

class musicntwrk:

    def __init__(self,TET=12):
        self.TET = TET

    def dictionary(self,space=None,N=None,Nc=None,order=None,row=None,a=None,prob=None,REF=None,scorefil=None,music21=None,
        midi=None,show=None):
        '''
        define dictionary in the musical space specified in 'space': pcs, rhythm, rhythmP, score, orch
        '''
        if space == 'pcs':
            from .networks.pcsDictionary import pcsDictionary
            dictionary,ZrelT = pcsDictionary(Nc,row,a,order,prob,TET=self.TET)
            return(dictionary,ZrelT)

        if space == 'rhythm':
            from .networks.rhythmDictionary import rhythmDictionary
            dictionary,ZrelT = rhythmDictionary(Nc,a,REF)
            return(dictionary,ZrelT)

        if space == 'rhythmP':
            from .networks.rhythmPDictionary import rhythmPDictionary
            dictionary,ZrelT = rhythmPDictionary(N,Nc,REF)
            return(dictionary,ZrelT)

        if space == 'score':
            #from .networks.scoreDictionary import scoreDictionary
            #from .networks.readScore import readScore
            seq,chords = readScore(scorefil,music21,show,midi,TET=self.TET)
            if midi == None:
                dictionary = scoreDictionary(seq,TET=self.TET)
            else:
                from .networks.scoreMIDIDictionary import scoreMIDIDictionary
                dictionary = scoreMIDIDictionary(seq)
            return(seq,chords,dictionary)

        if space == 'orch':
            from .networks.orchestralVector import orchestralVector
            score,orch,num = orchestralVector(scorefil)
            return(score,orch,num)


    def network(self,space=None,label=None,dictionary=None,thup=None,thdw=None,thup_e=None,thdw_e=None,distance=None,prob=None,write=None,\
                pcslabel=None,vector=None,ops=None,name=None,ntx=None,general=None,seq=None,sub=None,start=None,end=None,grphtype=None,\
                wavefil=None,cepstrum=None,color=None):
        '''
        define networks in the musical space specified in 'space': pcs (reg and ego), vLead (reg, vec, name and nameVec),
        rhythm, rLead, score (reg, name and sub), timbre, orch
        '''
        if space == 'pcs':
            from .networks.pcsNetwork import pcsNetwork
            nodes, edges = pcsNetwork(dictionary,thup,thdw,distance,prob,write,pcslabel,TET=self.TET)
            return(nodes,edges)

        if space == 'pcsEgo':
            from .networks.pcsEgoNetwork import pcsEgoNetwork
            nodes_e, edges_e, edges_a = pcsEgoNetwork(label,dictionary,thup_e,thdw_e,thup,thdw,distance,write,TET=self.TET)
            return(nodes_e,edges_e,edges_a)

        if space == 'vLead' and vector != True and ops != True:
            from .networks.vLeadNetwork import vLeadNetwork
            nodes, edges = vLeadNetwork(dictionary,thup,thdw,distance,prob,write,pcslabel,TET=self.TET)
            return(nodes, edges)

        if space == 'vLead' and vector and ops != True:
            from .networks.vLeadNetworkVec import vLeadNetworkVec
            nodes, edges = vLeadNetworkVec(dictionary,thup,thdw,distance,prob,write,pcslabel,TET=self.TET)
            return(nodes, edges)

        if space == 'vLead' and vector != True and ops:
            from .networks.vLeadNetworkByName import vLeadNetworkByName
            nodes, edges = vLeadNetworkByName(dictionary,name,distance,prob,write,pcslabel,TET=self.TET)
            return(nodes, edges)

        if space == 'vLead' and vector and ops:
            from .networks.vLeadNetworkByNameVec import vLeadNetworkByNameVec
            nodes, edges = vLeadNetworkByNameVec(dictionary,name,distance,prob,write,pcslabel,TET=self.TET)
            return(nodes, edges)

        if space == 'rhythm':
            from .networks.rhythmNetwork import rhythmNetwork
            nodes, edges = rhythmNetwork(dictionary,thup,thdw,distance,prob,write)
            return(nodes, edges)

        if space == 'rLead':
            from .networks.rLeadNetwork import rLeadNetwork
            nodes, edges = rLeadNetwork(dictionary,thup,thdw,distance,prob,write)
            return(nodes, edges)

        if space == 'score' and sub != True:
            #from .networks.scoreNetwork import scoreNetwork
            if ntx:
                nodes,edges,counts,deg,modul,Gxfull,Gxufull = scoreNetwork(seq,ntx,general,distance,TET=self.TET)
                return(nodes,edges,counts,deg,modul,Gxfull,Gxufull)
            else:
                nodes, edges, counts = scoreNetwork(seq,ntx,general,distance,TET=self.TET)
                return(nodes, edges, counts)

        if space == 'score' and sub:
            #from .networks.scoreSubNetwork import scoreSubNetwork
            if ntx:
                nodes,edges,counts,deg,modul,Gxfull,Gxufull = scoreSubNetwork(seq,start,end,ntx,general,distance,grphtype,TET=self.TET)
                return(nodes,edges,counts,deg,modul,Gxfull,Gxufull)
            else:
                nodes, edges, counts = scoreSubNetwork(seq,start,end,ntx,general,distance,grphtype,TET=self.TET)
                return(nodes, edges, counts)

        if space == 'timbre':
            from .networks.timbralNetwork import timbralNetwork
            nodes, edges = timbralNetwork(wavefil,cepstrum,thup,thdw)
            return(nodes, edges)

        if space == 'orch':
            from .networks.orchestralNetwork import orchestralNetwork
            nodes,edges,deg,modul,part,_,_ = orchestralNetwork(seq,distance,TET=self.TET)
            return(nodes,edges,deg,modul,part)


    def timbre(self,descriptor=None,path=None,wavefil=None,standard=None,nmel=None,ncc=None,zero=None,lmax=None,maxi=None,nbins=None,\
                method=None,scnd=None,nstep=None):
        '''
        Define sound descriptor for timbral analysis
        '''
        if descriptor == 'PSCC' and standard != True:
            from .timbre.computePSCC import computePSCC
            waves, cepstrum0, cepstrum = computePSCC(path,wavefil,ncc,zero)
            return(waves, cepstrum0, cepstrum)

        if descriptor == 'PSCC' and standard:
            from .timbre.computeStandardizedPSCC import computeStandardizedPSCC
            waves, cepstrum0, lmax = computeStandardizedPSCC(path,wavefil,ncc,lmax,maxi,nbins)
            return(waves, cepstrum0, lmax)

        if descriptor == 'MFCC' and standard != True:
            from .timbre.computeMFCC import computeMFCC
            waves, cepstrum0, cepstrum = computeMFCC(path,wavefil,nmel,ncc,zero)
            return(waves, cepstrum0, cepstrum)

        if descriptor == 'MFCC' and standard:
            from .timbre.computeStandardizedMFCC import computeStandardizedMFCC
            waves, cepstrum0, lmax = computeStandardizedMFCC(path,wavefil,nmel,ncc,lmax,maxi,nbins)
            return(waves, cepstrum0, lmax)

        if descriptor == 'MFPS' and standard:
            from .timbre.computeStandardizedMFPS import computeStandardizedMFPS
            waves, cepstrum0, lmax = computeStandardizedMFPS(path,wavefil,nmel,lmax,maxi,nbins)
            return(waves, cepstrum0, lmax)

        if descriptor == 'ASCBW' and standard != True:
            from .timbre.computeASCBW import computeASCBW
            waves, ascbw = computeASCBW(path,wavefil)
            return(waves, ascbw)

        if descriptor == 'ASCBW' and standard:
            from .timbre.computeModifiedASCBW import computeModifiedASCBW
            waves, ascbw, ascbwu = computeModifiedASCBW(path,wavefil,scnd,method,nstep)
            return(waves, ascbw, ascbwu)

    def harmony(self,descriptor=None,mode=None,x=None,y=None):
        '''
        handler for calculating tonal harmony models, tonnentz and to launch the tonal harmony calculator
        '''
        if descriptor == 'calc':
            from .harmony.tonalHarmonyCalculator import tonalHarmonyCalculator
            tonalHarmonyCalculator()

        if descriptor == 'tonnentz':
            from .harmony.tonnentz import tonnentz
            tnz = tonnentz(x,y)
            return(tnz)

        if descriptor == 'model':
            from .harmony.tonalHarmonyModels import tonalHarmonyModels
            tonalHarmonyModels(mode)

    def sonify(self,descriptor=None,engine='pyo',data=None,length=None,midi=None,scalemap=None,ini=None,fin=None,
               fac=None,dur=None,transp=None,col=None,write=None,vnorm=None,plot=None,crm=None,tms=None,xml=None,
               sigpath=None,sigfil=None,firpath=None,firsig=None):
        '''
        sonification strategies - simple sound (spectral) or score (melodic progression)
        '''
        from .data.r_1Ddata import r_1Ddata
        from .data.i_spectral_pyo import i_spectral_pyo
        from .data.i_spectral import i_spectral
        from .data.i_spectral2 import i_spectral2
        from .data.i_spectral_pure import i_spectral_pure

        if descriptor == 'spectrum':
            if engine == engine == 'pyo':
                x, y = r_1Ddata(data)
                s,a = i_spectral_pyo(x,y[0])
                s.start()
                time.sleep(length)
                s.stop()
                s.shutdown()
            elif engine == 'csound':
                # Full csound
                x, y = r_1Ddata(data)
                i_spectral(x,y[0],itime=length)
            elif engine == 'csound+scipy':
                # csound + scipy for FIR
                x, y = r_1Ddata(data)
                i_spectral2(x,y[0],itime=length)
            elif engine == 'scipy':
                # Full scipy
                x, y = r_1Ddata(data)
                s = i_spectral_pure(sigpath,sigfil,firpath,firsig)
                return(s)
            else:
                print('no engine specified for sound')
                sts.exit()

        if descriptor == 'melody':
            from .data.r_1Ddata import r_1Ddata
            from .data.scaleMapping import scaleMapping
            from .data.MIDImap import MIDImap
            from .data.MIDIscore import MIDIscore
            from .data.MIDImidi import MIDImidi
            x, y = r_1Ddata(data)
            scale, nnote = scaleMapping(scalemap,ini=ini,fin=fin,fac=fac)
            MIDIscore(MIDImap(y[col],scale,nnote)-transp,dur=dur,w=write)
            if midi: MIDImidi(MIDImap(y[col],scale,nnote)-transp,dur=2*dur,vnorm=vnorm)

        if descriptor == 'sound':
            from .data.analyzeSound import analyzeSound
            from .data.WRITEscore import WRITEscore
            nseq,beat,prob = analyzeSound(data,outlist=['nseq','beat','prob'],plot=plot,crm=crm,tms=tms,xml=xml)
            WRITEscore(data,nseq.pcs,beat.rseq,w=write)

#%% from musicntwrk.musicntwrk import PCSet

import re, sys, time
import numpy as np
import music21 as m21
from functools import reduce
import fractions as fr
from math import gcd

class PCSet:

    def __init__(self,pcs,TET=12,UNI=True,ORD=True):
        '''
        •	pcs (int)– pitch class set as list or numpy array
        •	TET (int)- number of allowed pitches in the totality of the musical space (temperament). Default = 12 tones equal temperament
        •	UNI (logical) – if True, eliminate duplicate pitches (default)
        •   ORD (logical) - if True, sorts the pcs in ascending order
        '''
        self.pcs = np.asarray(pcs)%TET
        if UNI == True:
            self.pcs = np.unique(self.pcs)%TET
        if ORD == True:
            self.pcs = np.sort(self.pcs)%TET

        self.TET = TET

    def normalOrder(self):
        '''
        •	Order the pcs according to the most compact ascending scale in pitch-class space that spans less than an octave by cycling permutations.
        '''
        self.pcs = np.sort(self.pcs)

        # trivial sets
        if len(self.pcs) == 1:
            return(self.pcs-self.pcs[0])
#        if len(self.pcs) == 2:
#            return(self.pcs)

        # 1. cycle to find the most compact ascending order
        nroll = np.linspace(0,len(self.pcs)-1,len(self.pcs),dtype=int)
        dist = np.zeros((len(self.pcs)),dtype=int)
        for i in range(len(self.pcs)):
            dist[i] = (np.roll(self.pcs,i)[len(self.pcs)-1] - np.roll(self.pcs,i)[0])%self.TET

        # 2. check for multiple compact orders
        for l in range(1,len(self.pcs)):
            if np.array(np.where(dist == dist.min())).shape[1] != 1:
                indx = np.array(np.where(dist == dist.min()))[0]
                nroll = nroll[indx]
                dist = np.zeros((len(nroll)),dtype=int)
                i = 0
                for n in nroll:
                    dist[i] = (np.roll(self.pcs,n)[len(self.pcs)-(1+l)] - np.roll(self.pcs,n)[0])%self.TET
                    i += 1
            else:
                indx = np.array(np.where(dist == dist.min()))[0]
                nroll = nroll[int(indx[0])]
                pcs_norm = np.roll(self.pcs,nroll)
                break
        if np.array(np.where(dist == dist.min())).shape[1] != 1: pcs_norm = self.pcs
        return(pcs_norm)

    def normal0Order(self):
        '''
        •	As normal order, transposed so that the first pitch is 0
        '''
        return((self.normalOrder()-self.normalOrder()[0])%self.TET)

    def T(self,t=0):
        '''
        •	Transposition by t (int) units (modulo TET)
        '''
        return((self.pcs+t)%self.TET)

    def M(self,t=1):
        '''
        •	Multiplication by t (int) units (modulo TET)
        '''
        return(np.unique((self.pcs*t)%self.TET//1).astype(int))

    def multiplyBoulez(self,b):
        # Boulez pitch class multiplication of a x b
        ivec = self.LISVector()
        m = []
        for i in range(ivec.shape[0]-1):
            mm = (b+ivec[i])%self.TET
            m.append(mm.tolist())
        return(PCSet(Remove(flatten(m+b)),TET).normalOrder())

    def zeroOrder(self):
        '''
        •	transposed so that the first pitch is 0
        '''
        return((self.pcs-self.pcs[0])%self.TET)

    def I(self):
        '''
        •	inverse operation: (-pcs modulo TET)
        '''
        return(-self.pcs%self.TET)

    def primeForm(self):
        '''
        •	most compact normal 0 order between pcs and its inverse
        '''
        s_orig = self.pcs
        sn = np.sum((self.normalOrder()-self.normalOrder()[0])%self.TET)
        self.pcs = self.I()
        si = np.sum((self.normalOrder()-self.normalOrder()[0])%self.TET)
        if sn <= si:
            self.pcs = s_orig
            return((self.normalOrder()-self.normalOrder()[0])%self.TET)
        else:
            tmp = (self.normalOrder()-self.normalOrder()[0])%self.TET
            self.pcs = s_orig
            return(tmp)

    def intervalVector(self):
        '''
        •	 total interval content of the pcs
        '''
        npc = int((len(self.pcs)**2-len(self.pcs))/2)
        itv = np.zeros(npc,dtype=int)
        n= 0
        for i in range(len(self.pcs)):
            for j in range(i+1,len(self.pcs)):
                if np.abs(self.pcs[i]-self.pcs[j]) > self.TET/2:
                    itv[n] = self.TET-np.abs(self.pcs[i]-self.pcs[j])
                else:
                    itv[n] = np.abs(self.pcs[i]-self.pcs[j])
                n += 1
        bins = np.linspace(1,int(self.TET/2)+1,int(self.TET/2)+1,dtype=int)
        return(np.histogram(itv,bins)[0])

    def Op(self,name):
        # operate on the pcs with a generic distance operator
        from .utils.Remove import Remove

        def plusAndMinusPermutations(items):
            for p in iter.permutations(items):
                for signs in iter.product([-1,1], repeat=len(items)):
                    yield [a*sign for a,sign in zip(p,signs)]

        op = ' '.join(i for i in name if i.isdigit()).split()
        op = np.asarray([list(map(int, x)) for x in op])
        op = np.reshape(op,op.shape[0]*op.shape[1])
        if self.pcs.shape[0] == op.shape[0]:
            pop = np.asarray(list(plusAndMinusPermutations(op)))
            selfto = np.unique((self.pcs+pop)%self.TET,axis=0)
            outset = []
            for n in range(selfto.shape[0]):
                if minimalNoBijDistance(self.normalOrder(),PCSet(selfto[n]).normalOrder())[0] == opsDistance(name)[1]:
                    outset.append(PCSet(selfto[n]).normalOrder().tolist())
        if self.pcs.shape[0] > op.shape[0]:
            op = np.pad(op,(0,self.pcs.shape[0]-op.shape[0]),'constant')
            pop = np.asarray(list(plusAndMinusPermutations(op)))
            selfto = np.unique((self.pcs+pop)%self.TET,axis=0)
            outset = []
            for n in range(selfto.shape[0]):
                if minimalNoBijDistance(self.normalOrder(),PCSet(selfto[n]).normalOrder())[0] == opsDistance(name)[1]:
                    outset.append(PCSet(selfto[n]).normalOrder().tolist())
        if self.pcs.shape[0] < op.shape[0]:
            print("increase cardinality by duplicating pc's - program will stop")
            outset = None
        return(Remove(outset))

    def VLOp(self,name):
        # operate on the pcs with a normal-ordered voice-leading operator

        op = []
        for num in re.findall("[-\d]+", name):
            op.append(int(num))
        op = np.asarray(op)
        selfto = np.unique((self.pcs+op)%self.TET,axis=0)
        return(PCSet(selfto).normalOrder())

    def LISVector(self):
        '''
        •	Linear Interval Sequence Vector: sequence of intervals in an normal ordered pcs
        '''
        return((np.roll(self.normalOrder(),-1)-self.normalOrder())%self.TET)

    def intervals(self):
        '''
        •	Linear Interval Sequence Vector: sequence of intervals in an ordered pcs
        '''
        return((np.roll(self,-1)-self)%self.TET)

    def forteClass(self):
        '''
        •	Name of pcs according to the Forte classification scheme (only for TET=12)
        '''
        if self.TET != 12:
            print('Forte class defined only for 12-TET')
            return()
        Fname = m21.chord.Chord(self.primeForm().pcs.tolist()).forteClass
        return(Fname)

    def commonName(self):
        '''
        •	Display common name of pcs (music21 function - only for TET=12)
        '''
        return(m21.chord.Chord(np.ndarray.tolist(self.normalOrder()[:])).commonName)

    def commonNamePrime(self):
        '''
        •	As above, for prime forms
        '''
        return(m21.chord.Chord(np.ndarray.tolist(self.primeForm()[:])).commonName)

    def commonNamePitched(self):
        '''
        •	As above, for prime forms
        '''
        return(m21.chord.Chord(np.ndarray.tolist(self.normalOrder()[:])).pitchedCommonName)

    def displayNotes(self,show=True,xml=False,prime=False,chord=False):
        '''
        •	Display pcs in score in musicxml format. If prime is True, display the prime form. If chord is True
            display the note cluster
        '''
        fac = self.TET/12
        if  not chord:
            s = m21.stream.Stream()
            for i in range(self.pcs.shape[0]):
                if prime:
                    s.append(m21.note.Note(self.primeForm()[i]/fac+60))
                else:
                    s.append(m21.note.Note(self.pcs[i]/fac+60))
            if show: s.show()
            if xml: s.show('musicxml')
            return(s)
        else:
            ch = []
            if prime:
                for i in range(self.pcs.shape[0]):
                    ch.append(m21.note.Note(self.primeForm()[i]/fac+60))
            else:
                for i in range(self.pcs.shape[0]):
                    ch.append(m21.note.Note(self.pcs[i]/fac+60))
            c = m21.chord.Chord(ch)
            if show: c.show()
            if xml: c.show('musicxml')
            return(c)

#%% from musicntwrk.plotting.drawNetwork import drawNetwork

import numpy as np
import networkx as nx
import community as cm
import matplotlib.pyplot as plt

def drawNetwork(Gx=None,Gxu=None,nodes=None,edges=None,forceiter=100,grphtype='undirected',dx=10,dy=10,colormap='jet',scale=1.0,
    layout='force',drawlabels=True,giant=False,equi=False,res=0.5,k=None,edge_labels=False,font=12,node_label=None):

    colors = ['#ffaacc', '#ccffcc', '#ffffcc', '#ccaacc', '#ccffff', '#cccccc', '#ccaaff', '#ffffff', '#ccddee', '#e7ffac', '#ffc9de', '#fbe4ff', '#aff8d8']
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
        pos = nx.spring_layout(Gx,k=k,iterations=forceiter)
    elif layout == 'spiral':
        pos = nx.spiral_layout(Gx,equidistant=equi,resolution=res)
    df = np.array(nodes)
    if len(df.shape) == 1:
        df = np.reshape(df,(len(df),1))
    if node_label == None:
        nodelabel = dict(zip(np.linspace(0,len(df[:,0])-1,len(df[:,0]),dtype=int),df[:,0]))
    else:
        nodelabel = node_label
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
    nx.draw_networkx(Gx,pos=pos,labels=labels,with_labels=drawlabels,cmap=plt.get_cmap(colormap),node_color=np.array(colors)[values],
                    node_size=dsize)
    plt.show()
    return Gx, Gxu, part, nodelabel, colors

#%% #from .networks.readScore import readScore

import re,sys
import numpy as np
import music21 as m21

def readScore(input_xml,music21,show,midi,TET):
    '''
    •	read a score in musicxml format
    •	returns the sequence of chords
    '''
    if TET == 12:
        if music21:
            score = m21.corpus.parse(input_xml)
            try:
                score = score.mergeScores()
            except:
                pass
        else:
            score_real = m21.converter.parse(input_xml)
            score = m21.harmony.realizeChordSymbolDurations(score_real)
        chords = score.chordify()
        if show: chords.show()
        if midi:
            seq = []
            for c in chords.recurse().getElementsByClass('Chord'):
                m = []
                for p in c.pitches:
                    m.append(p.midi)
                seq.append(m)
            return(seq,chords)
        else:
            seq = []
            for c in chords.recurse().getElementsByClass('Chord'):
                seq.append(c.normalOrder)
            return(seq,chords)
    elif TET == 24:
        dict24 = {'C':0,'C~':1,'C#':2,'D-':2,'D`':3,'D':4,'D~':5,'D#':6,'E-':6,'E`':7,'E':8,
                            'E~':9,'F`':9,'F':10,'F~':11,'F#':12,'G-':12,'G`':13,'G':14,'G~':15,'G#':16,
                            'A-':16,'A`':17,'A':18,'A~':19,'A#':20,'B-':20,'B`':21,'B':22,'B~':23,'C`':23}

        score = m21.converter.parse(input_xml)
        chords = score.chordify()
        if show: chords.show()
        seq = []
        for c in chords.recurse().getElementsByClass('Chord'):
            seq.append(str(c))

        clean = []
        for n in range(len(seq)):
            line = ''.join(i for i in seq[n] if not i.isdigit()).split()[1:]
            c = []
            for l in line:
                c.append(dict24[re.sub('>', '', l)])
            clean.append(PCSet(c,TET=24).normalOrder().tolist())
        return(clean,chords)
    else:
        print('temperament needs to be added')
        sys.exit()
    return

#%% from .networks.scoreDictionary import scoreDictionary

import pandas as pd
import numpy as np
import music21 as m21

def scoreDictionary(seq,TET=12):
    '''
    •	build the dictionary of pcs in any score in musicxml format
    •	use readScore() to import the score data as sequence
    '''
    s = Remove(seq)
    v = []
    name = []
    prime = []
    if TET == 12:
        for i in range(len(s)):
            p = PCSet(np.asarray(s[i][:]),TET=TET)
            v.append(p.intervalVector())
            name.append(''.join(m21.chord.Chord(p.normalOrder().tolist()).pitchNames))
            prime.append(np.array2string(p.normalOrder(),separator=',').replace(" ",""))
        vector = np.asarray(v)
        name = np.asarray(name)
    elif TET == 24:
        dict24 = {'C':0,'C~':1,'C#':2,'D-':2,'D`':3,'D':4,'D~':5,'D#':6,'E-':6,'E`':7,'E':8,
                            'E~':9,'F`':9,'F':10,'F~':11,'F#':12,'G-':12,'G`':13,'G':14,'G~':15,'G#':16,
                            'A-':16,'A`':17,'A':18,'A~':19,'A#':20,'B-':20,'B`':21,'B':22,'B~':23,'C`':23}
        for i in range(len(s)):
            p = PCSet(np.asarray(s[i][:]),TET=TET)
            v.append(p.intervalVector())
            tmp = []
            for i in p.pcs:
                tmp.append(list(dict24.keys())[list(dict24.values()).index(i)])
            name.append(''.join(tmp))
            prime.append(np.array2string(p.normalOrder(),separator=',').replace(" ",""))
        vector = np.asarray(v)
        name = np.asarray(name)
    else:
        print('temperament needs to be added')
        sys.exit()

    # Create dictionary of pitch class sets
    reference = []
    for n in range(len(name)):
        entry = [name[n],prime[n],
                np.array2string(vector[n,:],separator=',').replace(" ","")]
        reference.append(entry)

    dictionary = pd.DataFrame(reference,columns=['class','pcs','interval'])

    return(dictionary)

#%% from ..utils.Remove import *

def Remove(duplicate):
    # function to remove duplicates from list
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    return final_list

#%% from .networks.scoreNetwork import scoreNetwork

import pandas as pd
import numpy as np
import music21 as m21
import networkx as nx
import community as cm
import matplotlib.pyplot as plt

#from ..musicntwrk import PCSet
#from ..utils.minimalDistance import minimalDistance
#from ..utils.minimalNoBijDistance import minimalNoBijDistance
#from ..utils.generalizedOpsName import generalizedOpsName

def scoreNetwork(seq,ntx,general,distance,TET):

    '''
    •	generates the directional network of chord progressions from any score in musicxml format
    •	seq (int) – list of pcs for each chords extracted from the score
    •	use readScore() to import the score data as sequence
    '''
    # build the directional network of the full progression in the chorale

    dedges = pd.DataFrame(None,columns=['Source','Target','Weight','Label'])
    dnodes = pd.DataFrame(None,columns=['Label'])
    for n in range(len(seq)):
        p = PCSet(np.asarray(seq[n]),TET)
        if TET == 12:
            if p.pcs.shape[0] == 1:
                nn = ''.join(m21.chord.Chord(p.pcs.tolist()).pitchNames)
            else:
                nn = ''.join(m21.chord.Chord(p.normalOrder().tolist()).pitchNames)
            nameseq = pd.DataFrame([[str(nn)]],columns=['Label'])
        elif TET == 24:
            dict24 = {'C':0,'C~':1,'C#':2,'D-':2,'D`':3,'D':4,'D~':5,'D#':6,'E-':6,'E`':7,'E':8,
                                'E~':9,'F`':9,'F':10,'F~':11,'F#':12,'G-':12,'G`':13,'G':14,'G~':15,'G#':16,
                                'A-':16,'A`':17,'A':18,'A~':19,'A#':20,'B-':20,'B`':21,'B':22,'B~':23,'C`':23}
            tmp = []
            for i in p.pcs:
                tmp.append(list(dict24.keys())[list(dict24.values()).index(i)])
            nameseq = pd.DataFrame([[''.join(tmp)]],columns=['Label'])
        dnodes = dnodes.append(nameseq)
    df = np.asarray(dnodes)
    dnodes = pd.DataFrame(None,columns=['Label'])
    dcounts = pd.DataFrame(None,columns=['Label','Counts'])
    dff,idx,cnt = np.unique(df,return_inverse=True,return_counts=True)
    for n in range(dff.shape[0]):
        nameseq = pd.DataFrame([[str(dff[n])]],columns=['Label'])
        dnodes = dnodes.append(nameseq)
        namecnt = pd.DataFrame([[str(dff[n]),cnt[n]]],columns=['Label','Counts'])
        dcounts = dcounts.append(namecnt)

    for n in range(1,len(seq)):
        if len(seq[n-1]) == len(seq[n]):
            a = np.asarray(seq[n-1])
            b = np.asarray(seq[n])
            pair,r = minimalDistance(a,b,TET,distance)
        else:
            if len(seq[n-1]) > len(seq[n]):
                a = np.asarray(seq[n-1])
                b = np.asarray(seq[n])
                pair,r = minimalNoBijDistance(a,b,TET,distance)
            else:
                b = np.asarray(seq[n-1])
                a = np.asarray(seq[n])
                pair,r = minimalNoBijDistance(a,b,TET,distance)
        if pair != 0:
            if general == False:
                tmp = pd.DataFrame([[str(idx[n-1]),str(idx[n]),str(1/pair),opsName(a,r,TET)]],
                                    columns=['Source','Target','Weight','Label'])
            else:
                tmp = pd.DataFrame([[str(idx[n-1]),str(idx[n]),str(1/pair),generalizedOpsName(a,r,TET,distance)[1]]],
                                    columns=['Source','Target','Weight','Label'])
            dedges = dedges.append(tmp)

# write dataframe with pcs rather than indeces
#           if pair != 0:
#                if general == False:
#                    tmp = pd.DataFrame([[str(seq[n-1]),str(seq[n]),str(1/pair),opsName(a,r,TET)]],
#                                        columns=['Source','Target','Weight','Label'])
#                else:
#                    tmp = pd.DataFrame([[str(seq[n-1]),str(seq[n]),str(1/pair),generalizedOpsName(a,r,TET)[1]]],
#                                        columns=['Source','Target','Weight','Label'])
#                dedges = dedges.append(tmp)


    if ntx:
        # evaluate average degree and modularity
        gbch = nx.from_pandas_edgelist(dedges,'Source','Target',['Weight','Label'],create_using=nx.DiGraph())
        gbch_u = nx.from_pandas_edgelist(dedges,'Source','Target',['Weight','Label'])
        # modularity
        part = cm.best_partition(gbch_u)
        modul = cm.modularity(part,gbch_u)
        # average degree
        nnodes=gbch.number_of_nodes()
        avg = 0
        for node in gbch.degree():
            avg += node[1]
        avgdeg = avg/float(nnodes)
        return(dnodes,dedges,dcounts,avgdeg,modul,gbch,gbch_u)
    else:
        return(dnodes,dedges,dcounts)

#%% from ..utils.minimalDistance import minimalDistance

import numpy as np
import sklearn.metrics as sklm

def minimalDistance(a,b,TET,distance):
    '''
    •	calculates the minimal distance between two pcs of same cardinality (bijective)
    •	a,b (int) – pcs as numpy arrays or lists
    '''
    a = np.asarray(a)
    b = np.asarray(b)
    n = a.shape[0]
    if a.shape[0] != b.shape[0]:
        print('dimension of arrays must be equal')
        sys.exit()
    a = np.sort(a)
    iTET = np.vstack([np.identity(n,dtype=int)*TET,-np.identity(n,dtype=int)*TET])
    iTET = np.vstack([iTET,np.zeros(n,dtype=int)])
    diff = np.zeros(2*n+1,dtype=float)
    v = []
    for i in range(2*n+1):
        r = np.sort(b - iTET[i])
        diff[i] = sklm.pairwise_distances(a.reshape(1, -1),r.reshape(1, -1),metric=distance)[0]
        v.append(r)
    imin = np.argmin(diff)
    return(diff.min(),np.asarray(v[imin]).astype(int))

#%% #from ..utils.minimalNoBijDistance import minimalNoBijDistance

import itertools as iter
import numpy as np

def minimalNoBijDistance(a,b,TET,distance):
    '''
    •	calculates the minimal distance between two pcs of different cardinality (non bijective) – uses minimalDistance()
    •	a,b (int) – pcs as lists or numpy arrays
    '''
    a = np.asarray(a)
    b = np.asarray(b)
    ndif = np.sort(np.array([a.shape[0],b.shape[0]]))[1] - np.sort(np.array([a.shape[0],b.shape[0]]))[0]
    c = np.asarray(list(iter.combinations_with_replacement(b,ndif)))
    r = np.zeros((c.shape[0],a.shape[0]))
    for l in range(c.shape[0]):
        r[l,:b.shape[0]] = b
        r[l,b.shape[0]:] = c[l]
    dist = np.zeros(r.shape[0])
    for l in range(r.shape[0]):
        dist[l],_=minimalDistance(a,r[l],TET,distance)
    imin = np.argmin(dist)

    return(min(dist),r[imin].astype(int))

#%% #from ..utils.generalizedOpsName import generalizedOpsName

def generalizedOpsName(a,b,TET,distance):
# generalizes the operator name function for no-bijective chord progression
    if len(a) == len(b):
        return(a,opsNameFull(a,b,TET))
    else:
        if len(a) > len(b):
            pair,r = minimalNoBijDistance(a,b,TET,distance)
            return(r,opsNameFull(a,r,TET))
        else:
            pair,r = minimalNoBijDistance(b,a,TET,distance)
            return(r,opsNameFull(r,b,TET))

#%% from .opsNameFull import opsNameFull

import numpy as np

def opsNameFull(a,b,TET):
    # given two vectors returns the name of the normal ordered distance operator (R) that connects them
    a = PCSet(a,UNI=False).normalOrder()
    b = PCSet(b,UNI=False).normalOrder()
    d = np.zeros((b.shape[0]),dtype=int)
    for n in range(b.shape[0]):
        c = np.roll(b,n)
        diff = a-c
        for i in range(diff.shape[0]):
            if diff[i] >= int(TET/2):
                diff[i] -= TET
            if diff[i] < -int(TET/2):
                diff[i] += TET
        diff = np.abs(diff)
        d[n] = diff.dot(diff)
    nmin = np.argmin(d)
    b = np.roll(b,nmin)
    diff = b-a
    for i in range(diff.shape[0]):
        if diff[i] >= int(TET/2):
            diff[i] -= TET
        if diff[i] < -int(TET/2):
            diff[i] += TET

    return('O('+np.array2string(diff,separator=',').replace(" ","").replace("[","").replace("]","")+')')

#%%

def staticNetAnalysis(modul, G):
    print("Modularity:", modul)
    print("Modularity above 0.6 means that the tonal regions are distinct.")

    print("--------------------------------------")

    degrees = [val for (node, val) in G.degree()]
    import matplotlib.pyplot as plt

    log_x = np.log10(np.arange(1,len(degrees)+1))
    log_y = np.log10(sorted(degrees, reverse=True))
    m, b = np.polyfit(log_x, log_y, 1)

    plt.subplot(1,2,1)
    plt.scatter(log_x, log_y, color='b')
    plt.plot(log_x, m*log_x + b)

    plt.subplot(1,2,2)
    plt.scatter(np.arange(1,len(degrees)+1), sorted(degrees, reverse=True), color='r')

    plt.show()

    print("If loglog is a straight line, it means the network is scale-free (with a couple of hubs).")
    print("--------------------------------------")

    import powerlaw
    data = np.array(sorted(degrees)) # data can be list or numpy array
    results = powerlaw.Fit(data,verbose=False)
    print("alpha:", results.power_law.alpha)
    print("xmin:", results.power_law.xmin)
    R, p = results.distribution_compare('power_law', 'lognormal')

    print("This gives best fit for loglog plot.")
    print("-------------------------------------")

    from collections import Counter
    freq_counter = Counter([tuple(sorted(map(int,i))) for i in list(G.edges())])
    plt.hist(freq_counter)
    plt.show()
    print(freq_counter)

    print("This is frequency of the progressions. Varied frequencies points to directedness.")
    print("-------------------------------------")

    sorted_nodes_by_part = sorted(part, key=lambda k:part[k])
    sorted_parts = list(part.values())
    num_parts = len(set(sorted_parts))

    communities = []
    for i in range(num_parts):
        communities.append(list())

    for node in sorted_nodes_by_part:
        communities[part.get(node)].append(node)

    for num, group in zip(range(len(communities)),communities):
        sorted_group = sorted(G.degree(group), key=lambda x: x[1], reverse=True)
        sorted_labeled_group = []
        for i in range(len(sorted_group)):
            list_tuple = list(sorted_group[i])
            label = full_nodelabel.get(int(list_tuple[0]))
            list_tuple[0] = label
            sorted_labeled_group.append(tuple(list_tuple))
        print("Group", str(num) + " - (color", colors[num] + "):", sorted_labeled_group)

    print("These are the most 'important' (most visited) chords within each tonal region")
    print("-------------------------------------")

#%% from musicntwrk.harmony.scoreFilter import scoreFilter

import music21 as m21
import numpy as np
import matplotlib.pyplot as plt

def scoreFilter(seq,chords,thr=0,plot=False):
    # score sequence and unique pcs count
    if chords != None:
        mea = []
        for c in chords.recurse().getElementsByClass('Chord'):
            mea.append(str(c.measureNumber))
    su = [str(f) for f in Remove(seq)]
    ind = []
    for n in range(len(su)):
        ind.append(str(n))
    su = np.asarray(su)
    ind = np.asarray(ind)
    labeltot = dict(zip(su,ind))
    indextot = dict(zip(ind,su))
    value = []
    for i in range(len(seq)):
        value.append(int(labeltot[str(seq[i])]))
    if plot:
        plt.plot(value,'o')
        plt.show()
        print('total number of chords = ',len(seq))
    hh,bb = np.histogram(np.array(value),bins=len(labeltot))
    if plot:
        plt.plot(hh,drawstyle='steps-mid')
        plt.show()
    # filtering pcs with occurrences lower than threshold = thr
    filtered = []
    if chords != None: fmeasure = []
    for i in range(len(seq)):
        if hh[int(labeltot[str(seq[i])])] > thr:
            filtered.append(seq[i])
            if chords != None: fmeasure.append(mea[i])
    su = [str(f) for f in Remove(filtered)]
    ind = []
    for n in range(len(su)):
        ind.append(str(n))
    su = np.asarray(su)
    ind = np.asarray(ind)
    labelfilt = dict(zip(su,ind))
    indexfilt = dict(zip(ind,su))
    valuef = []
    for i in range(len(filtered)):
        valuef.append(int(labelfilt[str(filtered[i])]))
    if plot:
        plt.plot(valuef,'or')
        plt.show()
        print('total number of filtered chords = ',len(filtered))
    if chords != None:
        return(value,valuef,filtered,fmeasure)
    else:
        return(value,valuef,filtered)

#%%

def timeSeriesAnalysis(seq,chords):
    value,valuef,filtered,fmeasure = scoreFilter(seq,chords,thr=0,plot=True)
    print("Check for: repetition of pcsets (centricity), clustering (referentiality), new pcsets added over time (heirarchy).")
    return valuef

#%% from musicntwrk.harmony.changePoint import changePoint

import ruptures as rpt
import numpy as np
import matplotlib.pyplot as plt

def changePoint(value,model='rbf',penalty=1.0,brakepts=None,plot=False):
    # change point detection
    # available models: "rbf", "l1", "l2", rbf", "linear", "normal", "ar", "mahalanobis"
    signal = np.array(value)
    algo = rpt.Binseg(model=model).fit(signal)
    my_bkps = algo.predict(pen=penalty,n_bkps=brakepts)
    if plot:
        # show results
        rpt.show.display(signal, my_bkps, figsize=(10, 3))
        plt.show()
    # define regions from breaking points
    sections = my_bkps
    sections.insert(0,0)
    # check last point
    sections[-1] -= 1
    if plot: print('model = ',model,' - sections = ',sections)
    return(sections)

#%% from .networks.scoreSubNetwork import scoreSubNetwork

import pandas as pd
import numpy as np
import music21 as m21
import networkx as nx
import community as cm

def scoreSubNetwork(seq,start,end,ntx,general,distance,grphtype,TET):

    '''
    •	generates the directional network of chord progressions from any score in musicxml format
    •	seq (int) – list of pcs for each chords extracted from the score
    •	use readScore() to import the score data as sequence
    '''
    # build the directional network of the full progression in the chorale

    dedges = pd.DataFrame(None,columns=['Source','Target','Weight','Label'])
    dnodes = pd.DataFrame(None,columns=['Label'])
    for n in range(len(seq)):
        p = PCSet(np.asarray(seq[n]),TET)
        if TET == 12:
            if p.pcs.shape[0] == 1:
                nn = ''.join(m21.chord.Chord(p.pcs.tolist()).pitchNames)
            else:
                nn = ''.join(m21.chord.Chord(p.normalOrder().tolist()).pitchNames)
            nameseq = pd.DataFrame([[str(nn)]],columns=['Label'])
        elif TET == 24:
            dict24 = {'C':0,'C~':1,'C#':2,'D-':2,'D`':3,'D':4,'D~':5,'D#':6,'E-':6,'E`':7,'E':8,
                                'E~':9,'F`':9,'F':10,'F~':11,'F#':12,'G-':12,'G`':13,'G':14,'G~':15,'G#':16,
                                'A-':16,'A`':17,'A':18,'A~':19,'A#':20,'B-':20,'B`':21,'B':22,'B~':23,'C`':23}
            tmp = []
            for i in p.pcs:
                tmp.append(list(dict24.keys())[list(dict24.values()).index(i)])
            nameseq = pd.DataFrame([[''.join(tmp)]],columns=['Label'])
        dnodes = dnodes.append(nameseq)
    df = np.asarray(dnodes)
    dnodes = pd.DataFrame(None,columns=['Label'])
    dcounts = pd.DataFrame(None,columns=['Label','Counts'])
    dff,idx,cnt = np.unique(df,return_inverse=True,return_counts=True)
    for n in range(dff.shape[0]):
        nameseq = pd.DataFrame([[str(dff[n])]],columns=['Label'])
        dnodes = dnodes.append(nameseq)
        namecnt = pd.DataFrame([[str(dff[n]),cnt[n]]],columns=['Label','Counts'])
        dcounts = dcounts.append(namecnt)

    for n in range(1,len(seq)):
        if len(seq[n-1]) == len(seq[n]):
            a = np.asarray(seq[n-1])
            b = np.asarray(seq[n])
            pair,r = minimalDistance(a,b,TET,distance)
        else:
            if len(seq[n-1]) > len(seq[n]):
                a = np.asarray(seq[n-1])
                b = np.asarray(seq[n])
                pair,r = minimalNoBijDistance(a,b,TET,distance)
            else:
                b = np.asarray(seq[n-1])
                a = np.asarray(seq[n])
                pair,r = minimalNoBijDistance(a,b,TET,distance)
        if pair != 0:
            if general == False:
                if n >= start and n < end:
                    tmp = pd.DataFrame([[str(idx[n-1]),str(idx[n]),str(1/pair),opsName(a,r,TET)]],
                                        columns=['Source','Target','Weight','Label'])
#                 else:
#                     tmp = pd.DataFrame([[str(idx[n-1]),str(idx[n]),str(0),opsName(a,r,TET)]],
#                                         columns=['Source','Target','Weight','Label'])
                    dedges = dedges.append(tmp)
            else:
                if n >= start and n < end:
                    tmp = pd.DataFrame([[str(idx[n-1]),str(idx[n]),str(1/pair),generalizedOpsName(a,r,TET,distance)[1]]],
                                        columns=['Source','Target','Weight','Label'])
#                 else:
#                     tmp = pd.DataFrame([[str(idx[n-1]),str(idx[n]),str(0),generalizedOpsName(a,r,TET)[1]]],
#                                         columns=['Source','Target','Weight','Label'])
                    dedges = dedges.append(tmp)

    if ntx:
        # evaluate average degree and modularity
        if grphtype == 'directed':
            gbch = nx.from_pandas_edgelist(dedges,'Source','Target',['Weight','Label'],create_using=nx.DiGraph())
        elif grphtype == 'multi':
            gbch = nx.from_pandas_edgelist(dedges,'Source','Target',['Weight','Label'],create_using=nx.MultiDiGraph())
        else:
            print('no graph type specified')
            sys.exit()
        gbch_u = nx.from_pandas_edgelist(dedges,'Source','Target',['Weight','Label'])
        # modularity
        part = cm.best_partition(gbch_u)
        if not(gbch_u.size() == 0):
            modul = cm.modularity(part,gbch_u)
        # average degree
        nnodes=gbch.number_of_nodes()
        avg = 0
        for node in gbch.in_degree():
            avg += node[1]
        avgdeg = avg/float(nnodes)
        return(dnodes,dedges,dcounts,avgdeg,modul,gbch,gbch_u)
    else:
        return(dnodes,dedges,dcounts)

#%% from musicntwrk.plotting.drawMultiLayerNetwork import *

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class LayeredNetworkGraph(object):

    def __init__(self, graphs, node_labels=None, layout=nx.spring_layout, ax=None):
        """Given an ordered list of graphs [g1, g2, ..., gn] that represent
        different layers in a multi-layer network, plot the network in
        3D with the different layers separated along the z-axis.
        Within a layer, the corresponding graph defines the connectivity.
        Between layers, nodes in subsequent layers are connected if
        they have the same node ID.
        Arguments:
        ----------
        graphs : list of networkx.Graph objects
                List of graphs, one for each layer.
        node_labels : dict node ID : str label or None (default None)
                Dictionary mapping nodes to labels.
                If None is provided, nodes are not labelled.
        layout_func : function handle (default networkx.spring_layout)
                Function used to compute the layout.
        ax : mpl_toolkits.mplot3d.Axes3d instance or None (default None)
                The axis to plot to. If None is given, a new figure and a new axis are created.
        Original code from Paul Brodersen:
        https://stackoverflow.com/questions/60392940/multi-layer-graph-in-networkx
        see also https://github.com/paulbrodersen/netgraph
        """

        # book-keeping
        self.graphs = graphs
        self.total_layers = len(graphs)

        self.node_labels = node_labels
        self.layout = layout

        if ax:
            self.ax = ax
        else:
            fig = plt.figure()
            self.ax = fig.add_subplot(111, projection='3d')

        # create internal representation of nodes and edges
        self.get_nodes()
        self.get_edges_within_layers()
        self.get_edges_between_layers()

        # compute layout and plot
        self.get_node_positions()
        self.draw()


    def get_nodes(self):
        """Construct an internal representation of nodes with the format (node ID, layer)."""
        self.nodes = []
        for z, g in enumerate(self.graphs):
            self.nodes.extend([(node, z) for node in g.nodes()])


    def get_edges_within_layers(self):
        """Remap edges in the individual layers to the internal representations of the node IDs."""
        self.edges_within_layers = []
        for z, g in enumerate(self.graphs):
            self.edges_within_layers.extend([((source, z), (target, z)) for source, target in g.edges()])


    def get_edges_between_layers(self):
        """Determine edges between layers. Nodes in subsequent layers are
        thought to be connected if they have the same ID."""
        self.edges_between_layers = []
        for z1, g in enumerate(self.graphs[:-1]):
            z2 = z1 + 1
            h = self.graphs[z2]
            shared_nodes = set(g.nodes()) & set(h.nodes())
            self.edges_between_layers.extend([((node, z1), (node, z2)) for node in shared_nodes])


    def get_node_positions(self, *args, **kwargs):
        """Get the node positions in the layered layout."""
        # What we would like to do, is apply the layout function to a combined, layered network.
        # However, networkx layout functions are not implemented for the multi-dimensional case.
        # Futhermore, even if there was such a layout function, there probably would be no straightforward way to
        # specify the planarity requirement for nodes within a layer.
        # Therefor, we compute the layout for the full network in 2D, and then apply the
        # positions to the nodes in all planes.
        # For a force-directed layout, this will approximately do the right thing.
        # TODO: implement FR in 3D with layer constraints.

        composition = self.graphs[0]
        for h in self.graphs[1:]:
            composition = nx.compose(composition, h)

        pos = self.layout(composition, *args, **kwargs)

        self.node_positions = dict()
        for z, g in enumerate(self.graphs):
            self.node_positions.update({(node, z) : (*pos[node], z) for node in g.nodes()})


    def draw_nodes(self, nodes, *args, **kwargs):
        x, y, z = zip(*[self.node_positions[node] for node in nodes])
        self.ax.scatter(x, y, z, *args, **kwargs)


    def draw_edges(self, edges, *args, **kwargs):
        segments = [(self.node_positions[source], self.node_positions[target]) for source, target in edges]
        line_collection = Line3DCollection(segments, *args, **kwargs)
        self.ax.add_collection3d(line_collection)


    def get_extent(self, pad=0.1):
        xyz = np.array(list(self.node_positions.values()))
        xmin, ymin, _ = np.min(xyz, axis=0)
        xmax, ymax, _ = np.max(xyz, axis=0)
        dx = xmax - xmin
        dy = ymax - ymin
        return (xmin - pad * dx, xmax + pad * dx), \
            (ymin - pad * dy, ymax + pad * dy)


    def draw_plane(self, z, *args, **kwargs):
        (xmin, xmax), (ymin, ymax) = self.get_extent(pad=0.1)
        u = np.linspace(xmin, xmax, 10)
        v = np.linspace(ymin, ymax, 10)
        U, V = np.meshgrid(u ,v)
        W = z * np.ones_like(U)
        self.ax.plot_surface(U, V, W, *args, **kwargs)


    def draw_node_labels(self, node_labels, *args, **kwargs):
        for node, z in self.nodes:
            if node in node_labels:
                self.ax.text(*self.node_positions[(node, z)], node_labels[node], *args, **kwargs)


    def draw(self):

        self.draw_edges(self.edges_within_layers,  color='k', alpha=0.3, linestyle='-', zorder=2)
        self.draw_edges(self.edges_between_layers, color='k', alpha=0.3, linestyle='--', zorder=2)

        for z in range(self.total_layers):
            self.draw_plane(z, alpha=0.2, zorder=1)
            d = nx.degree(self.graphs[z])
            dsize = [(d[v]+1)*100 for v in self.graphs[z].nodes()]
            self.draw_nodes([node for node in self.nodes if node[1]==z], s=dsize, zorder=3)

        if self.node_labels:
            self.draw_node_labels(self.node_labels,horizontalalignment='center',verticalalignment='center',zorder=100)


def drawMultiLayerNetwork(masternodes,graphs,sx=30,sy=40,azim=-40,elev=5,layout=nx.spring_layout):
    # create node label dictionary
    node_labels = {str(i):n[0] for i,n in enumerate(np.asarray(masternodes))}
    # initialise figure and plot
    fig = plt.figure(figsize=(sx,sy))
    ax = fig.add_subplot(111, projection='3d',azim=azim,elev=elev)
    LayeredNetworkGraph(graphs, node_labels=node_labels, ax=ax, layout=layout)
    ax.set_axis_off()
    plt.show()

#%%

import numpy as np
import pandas as pd
import music21 as m21

def keySections(sections,GxsecDi,dnodes):
    # key identification in the different regions
    # this is based on the ansatz that the tonic triad is the most connected (highest degree)
    # node of the region's network
    prevalent_key = []
    for n in range(len(sections)-1):
        prevalent_chord = str(dnodes.set_index("Label", drop = True).\
                                iloc[int(sorted((value, key) for (key,value) in GxsecDi[n].degree)[-1][1])]).split()[2].replace(",","")
        ccc = []
        for char in prevalent_chord:
            ccc.append(char)
        ch = []
        i = 0
        for n in range(len(ccc)):
            if ccc[n] != '-' and ccc[n] != '#':
                ch.append(ccc[n])
                try:
                    if ccc[n+1] == '-' or ccc[n+1] == '#':
                        ch.pop()
                        ch.append(ccc[n]+ccc[n+1])
                except:
                    pass
        prevalent_key.append(m21.chord.Chord(ch))
    key = []
    for n in prevalent_key:
        if n.isMajorTriad():
            key.append(str(n.root()))
        elif n.isMinorTriad():
            key.append(str(n.root()).lower())
        elif n.isDominantSeventh():
            key.append(str(n.getChordStep(3).transpose(1).name))
        else:
            key.append(str(n.root()))

    keySections = pd.DataFrame(None,columns=['Section','chord range','prevalent_chord','region'])
    for i in range(len(key)):
        tmp = pd.DataFrame([[str(i),str(sections[i])+'-'+str(sections[i+1])
                            ,str(prevalent_key[i].pitchNames),key[i]]],
                            columns=['Section','chord range','prevalent_chord','region'])
        keySections = keySections.append(tmp)

    return(key,keySections)

#%%

def getChangePointSections(sections):

    GxsecDi = []
    Gxsec = []
    for n in range(len(sections)-1):
        nodes,edges,_,_,_,Gx,Gxu = mk.network(space='score',sub=True,seq=seq,start=sections[n],end=sections[n+1],
                                        ntx=True,general=True,distance='euclidean',grphtype='directed')
        GxsecDi.append(Gx)
        Gxsec.append(Gxu)

    return GxsecDi, Gxsec, nodes

#%%

def changePointAnalysis(valuef,penalty,bnodes):
    sections = changePoint(valuef,penalty=penalty,plot=True)

    print("This shows the change points in the music that the key section will be built from.")
    print("-------------------------------------")

    GxsecDi, Gxsec, nodes = getChangePointSections(sections)

    print("This creates subgraphs of the sections identified above.")
    print("-------------------------------------")

    import gmatch4py as gm

    ged=gm.MCS()
    result=ged.compare(GxsecDi,None)

    print(result)

    plt.imshow(result, cmap='hot', interpolation='nearest')
    plt.show()

    print("This compares the different sections - similarity is defined by the largest matching subgraph.")
    print("-------------------------------------")

    sorted_gxsec_indeces = np.argsort([graph.number_of_nodes() for graph in Gxsec])[::-1]
    sorted_gxsec = list(np.array(Gxsec)[sorted_gxsec_indeces])
    sorted_gxsecdi = list(np.array(GxsecDi)[np.argsort([graph.number_of_nodes() for graph in GxsecDi])[::-1]])

    print("SECTION:", sections[sorted_gxsec_indeces[0]], "-", sections[sorted_gxsec_indeces[0]+1])
    part = cm.best_partition(sorted_gxsec[0])
    modul = cm.modularity(part,sorted_gxsec[0])

    Gx, Gxu, part, nodelabel, colors = drawNetwork(Gx=sorted_gxsecdi[0],Gxu=sorted_gxsec[0],nodes=sorted_gxsecdi[0].nodes,grphtype='directed',k=1.0, colormap='Pastel1',node_label=full_nodelabel)

    staticNetAnalysis(modul=modul, G=Gxu)

    print("This looks at whether subnetworks have same characteristics of tonality as the full network.")
    print("-------------------------------------")

    drawMultiLayerNetwork(bnodes,Gxsec)

    print("This shows the multilayer network. Notice the pivot chords.")

    key,keySecs = keySections(sections,GxsecDi,nodes)

    print(keySecs)

    print("This shows the separated sections along with their prevalent chord.")

    return sections

#%%

def compareAnnotations(section, jazzSheet):
    score_1 = m21.converter.parse(jazzSheet)
    score_chords_realized = m21.harmony.realizeChordSymbolDurations(score_1)
    score_chords = score_chords_realized.chordify()

    chord_offsets = []
    for c in score_chords.recurse().getElementsByClass('Chord'):
        chord_offsets.append(c.offset)

    chord_symbol_offsets = []
    for c in score_chords_realized.recurse().getElementsByClass('ChordSymbol'):
        chord_symbol_offsets.append(c.offset)

    def getMeasureOfChord(offset):
        try:
            return list(score_1.getElementById('Alto-Sax-1').getElementsByOffset(offset).getElementsByClass('Measure'))[0].measureNumber
        except IndexError:
            return getMeasureOfChord(offset - (1-((1-offset)%1)))

    chord_measures = []
    for i in range(len(chord_offsets)):
        chord_measures.append(getMeasureOfChord(chord_offsets[i]))

    chord_symbol_measures = []
    for i in range(len(chord_symbol_offsets)):
        chord_symbol_measures.append(getMeasureOfChord(chord_symbol_offsets[i]))

    change_point_measures = list(np.array(chord_measures)[sections])

    intersection_between_real_predicted = np.intersect1d(change_point_measures, chord_symbol_measures)

    print("Percent correctly annotated over true:", len(intersection_between_real_predicted) / len(chord_symbol_measures))
    print("Percent correctly annotated over false:", len(intersection_between_real_predicted) / len(chord_measures))

#%%

import networkx as nx
import numpy as np
import music21 as m21
import community as cm

#from ..harmony.chinese_postman import chinese_postman
#from ..data.WRITEscoreOps import WRITEscoreOps
#from ..plotting.drawNetwork import drawNetwork


def harmonicDesign(mk,nnodes,refnodes,refedges,nedges=2,dualedge=3,nstart=None,scfree='barabasialbert',seed=None,prob=None,reverse=None,
                   display=None,write=None,verbose=True):
    # network generator (see documentation on networkx)
    if scfree == 'barabasialbert':
        if verbose: print('Barabasi-Albert')
        scfree = nx.barabasi_albert_graph(nnodes,nedges,seed)
    elif scfree == 'dual':
        if verbose: print('dual Barabasi-Albert')
        scfree = nx.dual_barabasi_albert_graph(nnodes,nedges,dualedge,prob,seed)
    elif scfree == 'erdosrenyi':
        if verbose: print('Erdos-Renyi')
        scfree = nx.erdos_renyi_graph(nnodes,prob)
    else:
        scfree = scfree
    # node degree distribution
    node = np.zeros((nnodes),dtype=int)
    weight = np.zeros((nnodes),dtype=int)
    for n in range(nnodes):
        node[n] = np.array(scfree.degree())[n][0]
        weight[n] = np.array(scfree.degree())[n][1]
    idx = np.argsort(weight)[::-1]
    if nstart == None:
        nstart = idx[0]
    euler_circuit = chinese_postman(scfree,starting_node=nstart)
    if verbose: print('Length of Eulerian circuit: {}'.format(len(euler_circuit)))
    # modularity
    if not scfree.is_directed():
        part = cm.best_partition(scfree)
        modul = cm.modularity(part,scfree)
    # average degree
    ntot=scfree.number_of_nodes()
    avg = 0
    for n in scfree.degree():
        avg += n[1]
    avgdeg = avg/float(ntot)
    if verbose: print('Average degree: ', avgdeg, ' modularity = ',modul)
    # reference node degree distribution
    try:
        bnet = nx.from_pandas_edgelist(refedges,'Source','Target',['Weight','Label'])
    except:
        bnet = nx.from_pandas_edgelist(refedges,'Source','Target',['Weight'])
    bnode = np.zeros((nnodes),dtype=int)
    bweight = np.zeros((nnodes),dtype=int)
    for n in range(nnodes):
        bnode[n] = np.array(bnet.degree())[n][0]
        bweight[n] = np.array(bnet.degree())[n][1]
    bidx = np.argsort(bweight)[::-1]
    # associate reference nodes to network
    a = node[idx[:]]
    b = bnode[bidx[:]]
    eudict = dict(zip(a,b))
    # write score
    eulerseq = []
    for i in range(len(euler_circuit)):
        ch = []
        for c in np.asarray(refnodes)[eudict[int(euler_circuit[i][0])]].tolist()[0]:
            if c == '#' or c == '-':
                pc = str(ch[-1])+c
                ch.pop()
                ch.append(pc)
            else:
                ch.append(c)
        eulerseq.append(m21.chord.Chord(ch).normalOrder)
    ch = []
    for c in np.asarray(refnodes)[eudict[int(euler_circuit[i][1])]].tolist()[0]:
        if c == '#' or c == '-':
            pc = str(ch[-1])+c
            ch.pop()
            ch.append(pc)
        else:
            ch.append(c)
    eulerseq.append(m21.chord.Chord(ch).normalOrder)
    if reverse: eulerseq = eulerseq[::-1]
    if display:
        eunodes,euedges,_,_,_,_,_ = mk.network(space='score',seq=eulerseq,ntx=True,general=True,
                                               distance='euclidean',grphtype='directed')
        Gx,Gxu,_,_,_ = drawNetwork(nodes=eunodes,edges=euedges,grphtype='directed')
    if write:
        WRITEscoreOps(eulerseq,w=write)

    if not scfree.is_directed():
        return(eulerseq,euedges,avgdeg,modul,nstart,Gx,Gxu,len(euler_circuit))
    else:
        return(eulerseq,avgdeg,None)

#%%

import networkx as nx
import itertools

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
            print(edge[0],edge[1])
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

#%%

import music21 as m21
import numpy as np

#from ..utils.generalizedOpsName import generalizedOpsName
#from ..utils.opsName import opsName

def WRITEscoreOps(nseq,w=None,outxml='./music',outmidi='./music',keysig=None,abs=False,TET=12,distance='euclidean'):
    try:
        ntot = nseq.shape[0]
    except:
        ntot = len(nseq)
    m = m21.stream.Stream()
    m.append(m21.meter.TimeSignature('4/4'))
    for i in range(ntot):
        ch = np.copy(nseq[i])
        for n in range(1,len(ch)):
            if ch[n] < ch[n-1]: ch[n] += 12
        ch += 60
        n = m21.chord.Chord(ch.tolist())
        if i < ntot-1:
            n.addLyric(str(i)+' '+generalizedOpsName(nseq[i],nseq[i+1],TET,distance)[1])
            if abs:
                if len(nseq[i]) == len(nseq[i+1]):
                    n.addLyric(str(i)+' '+opsName(nseq[i],nseq[i+1]))
                else:
                    r = generalizedOpsName(nseq[i],nseq[i+1],TET,distance)[0]
                    if len(nseq[i]) > len(nseq[i+1]):
                        n.addLyric(str(i)+' '+opsName(nseq[i],r))
                    else:
                        n.addLyric(str(i)+' '+opsName(r,nseq[i+1]))
        if keysig != None:
            rn = m21.roman.romanNumeralFromChord(n, m21.key.Key(keysig))
            n.addLyric(str(rn.figure))
        m.append(n)
    if w == True:
        m.show('musicxml')
    elif w == 'MIDI':
        m.write('midi',outmidi+'.mid')
    else:
        m.show()

#%%

import numpy as np

def opsName(a,b,TET):
    # given two vectors returns the name of the minimal distance operator that connects them
    a = np.sort(a)
    b = np.sort(b)
    d = np.zeros((b.shape[0]),dtype=int)
    for n in range(b.shape[0]):
        c = np.roll(b,n)
        diff = a-c
        for i in range(diff.shape[0]):
            if diff[i] >= int(TET/2):
                diff[i] -= TET
            if diff[i] < -int(TET/2):
                diff[i] += TET
        diff = np.abs(diff)
        d[n] = diff.dot(diff)
    nmin = np.argmin(d)
    b = np.roll(b,nmin)
    diff = a-b
    for i in range(diff.shape[0]):
        if diff[i] >= int(TET/2):
            diff[i] -= TET
        if diff[i] < -int(TET/2):
            diff[i] += TET
    diff = np.sort(np.abs(diff))

    return('O('+np.array2string(np.trim_zeros(diff),separator=',').replace(" ","").replace("[","").replace("]","")+')')

#%%

#from ..utils.opsHistogram import opsHistogram
#from ..utils.generalizedOpsHistogram import generalizedOpsHistogram

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

#def plotOpsHistogram(newvalues,newcounts,fx=15,fy=4):
def plotOpsHistogram(edges,fx=15,fy=4,show=False):

    values = edges['Label'].value_counts().keys().tolist()
    counts = edges['Label'].value_counts().tolist()
    counts /= np.sum(counts)*0.01

    newvalues, newcounts,pal_dict,dist = generalizedOpsHistogram(values,counts)
    idx = np.argwhere(newcounts)
    if show:
        plt.rcParams['font.family'] = 'arial'
        plt.rcParams['axes.edgecolor']='#333F4B'
        plt.rcParams['axes.linewidth']=1.5
        plt.rcParams['xtick.color']='#333F4B'
        plt.rcParams['ytick.color']='#333F4B'

        plt.figure(figsize=(fx,fy))

        plt.ylabel('Percentage',fontsize=24, fontweight='black', color = '#333F4B')
        plt.yticks(fontsize=18,fontweight='black', color = '#333F4B')
        plt.setp(plt.gca().get_xticklabels(), rotation=-90, horizontalalignment='center',fontsize=10,
                fontweight='black', color = '#000000')
    #     plt.xticks([])
        plt.bar(newvalues[idx][:,0],newcounts[idx][:,0],width=0.85,color='grey')
    return(newvalues[idx],newcounts[idx])

#%%

import numpy as np

#from .opsDistance import opsDistance

def opsHistogram(values,counts):
    ops = []
    for i in range(1,6):
        ops.append('O('+str(i)+')')
    for i in range(1,6):
        for j in range(i,6):
            ops.append('O('+str(i)+','+str(j)+')')
    for i in range(1,6):
        for j in range(i,6):
            for k in range(j,6):
                ops.append('O('+str(i)+','+str(j)+','+str(k)+')')
    for i in range(1,6):
        for j in range(i,6):
            for k in range(j,6):
                for l in range(k,6):
                    ops.append('O('+str(i)+','+str(j)+','+str(k)+','+str(l)+')')
    ops = np.array(ops)
    dist = np.zeros(ops.shape[0])
    for i in range(ops.shape[0]):
        dist[i] = opsDistance(ops[i])[1]
    idx = np.argsort(dist)
    ops = ops[idx]

    ops_dict = {}
    for i in range(len(ops)):
        ops_dict.update({ops[i]:0})

    for i in range(len(values)):
        ops_dict.update({values[i]:counts[i]})

    newvalues = np.asarray(list(ops_dict.keys()))
    newcounts = np.asarray(list(ops_dict.values()))
    return(newvalues,newcounts,ops_dict,dist[idx])

#%%

import numpy as np

def opsDistance(name):
    # returns distance for given operator
    opname = np.asarray(' '.join(i for i in name if i.isdigit()).split())
    opdist = np.sqrt(np.sum(np.asarray([list(map(int, x)) for x in opname]).reshape(1,-1)[0]*
        np.asarray([list(map(int, x)) for x in opname]).reshape(1,-1)[0]))
    return(name,opdist)

#%%

import numpy as np

#from .opsDistance import opsDistance

def generalizedOpsHistogram(values,counts):
    ops = []
    for i in range(-3,3):
        ops.append('O('+str(i)+')')
    for i in range(-3,3):
        for j in range(i,3):
            ops.append('O('+str(i)+','+str(j)+')')
    for i in range(-3,3):
        for j in range(i,3):
            for k in range(j,3):
                ops.append('O('+str(i)+','+str(j)+','+str(k)+')')
    for i in range(-3,3):
        for j in range(i,3):
            for k in range(j,3):
                for l in range(k,3):
                    ops.append('O('+str(i)+','+str(j)+','+str(k)+','+str(l)+')')
    ops = np.array(ops)
    dist = np.zeros(ops.shape[0])
    for i in range(ops.shape[0]):
        dist[i] = opsDistance(ops[i])[1]
    idx = np.argsort(dist)
    ops = ops[idx]

    ops_dict = {}
    for i in range(len(ops)):
        ops_dict.update({ops[i]:0})

    for i in range(len(values)):
        ops_dict.update({values[i]:counts[i]})

    newvalues = np.asarray(list(ops_dict.keys()))
    newcounts = np.asarray(list(ops_dict.values()))
    return(newvalues,newcounts,ops_dict,dist[idx])

#%% Designed circuit

def compareGeneratedVsReal(bnodes,bedges,mk,Gxu,Gx):

    euseq,euedges,avgdeg,modul,nstart,graph,graph_u,length = harmonicDesign(mk,len(bnodes),bnodes,bedges,nedges=2,seed=10,reverse=True,display=True,write=False)

    staticNetAnalysis(modul,graph_u)

    my_euler_circuit = chinese_postman(nx.relabel_nodes(Gxu, lambda x: int(x)), starting_node=0)
    print('Length of Eulerian circuit: {}'.format(len(my_euler_circuit)))
    _, _, _, _, _ = drawNetwork(Gx=Gx,Gxu=Gxu,nodes=Gxu.nodes,grphtype='directed',k=1.0, colormap='Pastel1',node_label=full_nodelabel)

    staticNetAnalysis(modul,Gxu)

    vals,counts = plotOpsHistogram(bedges)
    plt.show()
    print("These are the voice-leading operator frequencies of the real graph.")

    euvals,eucounts = plotOpsHistogram(euedges)
    plt.show()
    print("These are the voice-leading operator frequencies of the generated graph.")

    both,vals_ind,euvals_ind=np.intersect1d(vals,euvals, return_indices=True)

    vals_common = vals[vals_ind]
    counts_common = counts[vals_ind]
    eucounts_common = eucounts[euvals_ind]

    plt.hist([np.arange(len(vals_common)), np.arange(len(vals_common))], np.arange(len(vals_common)+1), weights=[np.squeeze(counts_common),np.squeeze(eucounts_common)])
    plt.xticks(np.arange(len(vals_common)),vals_common, rotation=45, horizontalalignment='right')
    plt.show()
    print("These are the voice leading operators they have in common and their frequencies.")

    return vals,counts,euvals,eucounts,len(my_euler_circuit),length

#%%
# bach
tonalClassicalSheets = ['bwv1.6',
'bwv10.7',
'bwv101.7',
'bwv102.7',
'bwv103.6',
'bwv104.6',
'bwv108.6',
'bwv11.6',
'bwv110.7']

tonalJazzSheets = ['/Users/abbypribis/Downloads/It_Dont_Mean_a_Thing_Edited.mxl',
'/Users/abbypribis/Downloads/Sophisticated_Lady_Edited.mxl',
'/Users/abbypribis/Downloads/Moonlight_Serenade_Edited.mxl',
'/Users/abbypribis/Downloads/In_The_Mood_Edited.mxl',
'/Users/abbypribis/Downloads/All_The_Things_You_Are_Edited.mxl',
'/Users/abbypribis/Downloads/Cherokee_Edited.mxl',
'/Users/abbypribis/Downloads/One_OClock_Jump_Edited.mxl',
'/Users/abbypribis/Downloads/Stompin_At_The_Savoy_Edited.mxl',
'/Users/abbypribis/Downloads/Moon_Indigo_Edited.mxl']

modalClassicalSheets = ['Fava_Dicant_nunc_iudei',
'PMFC_01-Lugentium siccentur',
'PMFC_01-Rex quem metrorum',
'PMFC_01-Virtutibus laudabilis',
'PMFC_01-Vos Qui Admiramini Gratissima virginis species',
'PMFC_04-A lle s_andra lo spirt',
'PMFC_04-Cara mi donna']

modalJazzSheets = ['/Users/abbypribis/Downloads/Footprints_Edited.mxl',
'/Users/abbypribis/Downloads/Impressions_Edited.mxl',
'/Users/abbypribis/Downloads/Mr_PC_Edited.mxl',
'/Users/abbypribis/Downloads/Stella_by_Starlight_Edited.mxl',
'/Users/abbypribis/Downloads/Softly_as_in_a_Morning_Sunrise_Edited.mxl',
'/Users/abbypribis/Downloads/Cantaloupe_Island_Edited.mxl',
'/Users/abbypribis/Downloads/Equinox_Edited.mxl']


generated_voice_operator_values_t = []
generated_voice_operator_counts_t = []
real_voice_operator_values_t = []
real_voice_operator_counts_t = []

generated_lengths_t = []
real_lengths_t = []

generated_voice_operator_values_m = []
generated_voice_operator_counts_m = []
real_voice_operator_values_m = []
real_voice_operator_counts_m = []

generated_lengths_m = []
real_lengths_m = []
#%%

for sheet in [modalJazzSheets[5]]:
    try:
        mk = musicntwrk(TET=12)

        seq,chords,dictionary = mk.dictionary(space='score',scorefil=sheet,music21=False,show=True)

        bnodes,bedges,counts,avgdeg,modul,Gx,Gxu = mk.network(space='score',seq=seq,ntx=True,general=True,distance='euclidean',
                                   grphtype='directed')

        Gx, Gxu, part, full_nodelabel, colors = drawNetwork(nodes=bnodes,edges=bedges,grphtype='directed',k=1.0, colormap='Pastel1')

        staticNetAnalysis(modul, Gxu)

        valuef = timeSeriesAnalysis(seq,chords)

        sections = changePointAnalysis(valuef, 3, bnodes)

        compareAnnotations(sections, sheet)

        real_val,real_count,gen_val,gen_count,real_euler,gen_euler = compareGeneratedVsReal(bnodes,bedges,mk,Gxu,Gx)

        #generated_voice_operator_values_m.append(gen_val)
        #generated_voice_operator_counts_m.append(gen_count)
        #real_voice_operator_values_m.append(real_val)
        #real_voice_operator_counts_m.append(real_count)
        #generated_lengths_m.append(gen_euler)
        #real_lengths_m.append(real_euler)

    except:
        print(sheet)
        pass

#%%
for i in [0.25]:
    for sheet in [tonalJazzSheets[1]]:
        try:
            mk = musicntwrk(TET=12)

            seq,chords,dictionary = mk.dictionary(space='score',scorefil=sheet,music21=False,show=True)

            bnodes,bedges,counts,avgdeg,modul,Gx,Gxu = mk.network(space='score',seq=seq,ntx=True,general=True,distance='euclidean',
                                       grphtype='directed')

            Gx, Gxu, part, full_nodelabel, colors = drawNetwork(nodes=bnodes,edges=bedges,grphtype='directed',k=1.0, colormap='Pastel1')

            #staticNetAnalysis(modul, Gxu)

            #valuef = timeSeriesAnalysis(seq,chords)

            #sections = changePointAnalysis(valuef, i, bnodes)

            #compareAnnotations(sections, sheet)

            real_val,real_count,gen_val,gen_count,real_euler,gen_euler = compareGeneratedVsReal(bnodes,bedges,mk,Gxu,Gx)

            generated_voice_operator_values_t.append(gen_val)
            generated_voice_operator_counts_t.append(gen_count)
            real_voice_operator_values_t.append(real_val)
            real_voice_operator_counts_t.append(real_count)
            generated_lengths_t.append(gen_euler)
            real_lengths_t.append(real_euler)

        except:
            print(sheet)
            pass

#%%

for sheet in [tonalJazzSheets[1]]:
    try:
        mk = musicntwrk(TET=12)

        seq,chords,dictionary = mk.dictionary(space='score',scorefil=sheet,music21=False,show=True)

        bnodes,bedges,counts,avgdeg,modul,Gx,Gxu = mk.network(space='score',seq=seq,ntx=True,general=True,distance='euclidean',
                                   grphtype='directed')

        Gx, Gxu, part, full_nodelabel, colors = drawNetwork(nodes=bnodes,edges=bedges,grphtype='directed',k=1.0, colormap='Pastel1')

        staticNetAnalysis(modul, Gxu)

        valuef = timeSeriesAnalysis(seq,chords)

        sections = changePointAnalysis(valuef, 0.5, bnodes)

        compareAnnotations(sections, sheet)

        real_val,real_count,gen_val,gen_count,real_euler,gen_euler = compareGeneratedVsReal(bnodes,bedges,mk,Gxu,Gx)

        generated_voice_operator_values_t.append(gen_val)
        generated_voice_operator_counts_t.append(gen_count)
        real_voice_operator_values_t.append(real_val)
        real_voice_operator_counts_t.append(real_count)
        generated_lengths_t.append(gen_euler)
        real_lengths_t.append(real_euler)

    except:
        print(sheet)
        pass

#%%

for sheet in [tonalClassicalSheets[0]]:
    mk = musicntwrk(TET=12)

    seq,chords,dictionary = mk.dictionary(space='score',scorefil=sheet,music21=True,show=True)

    bnodes,bedges,counts,avgdeg,modul,Gx,Gxu = mk.network(space='score',seq=seq,ntx=True,general=True,distance='euclidean',
                                   grphtype='directed')

    Gx, Gxu, part, full_nodelabel, colors = drawNetwork(nodes=bnodes,edges=bedges,grphtype='directed',k=1.0, colormap='Pastel1')

    staticNetAnalysis(modul, Gxu)

    valuef = timeSeriesAnalysis(seq,chords)

    sections = changePointAnalysis(valuef, 1.25, bnodes)

    #compareAnnotations(sections, sheet)

    real_val,real_count,gen_val,gen_count,real_euler,gen_euler = compareGeneratedVsReal(bnodes,bedges,mk,Gxu,Gx)

    #%%
for sheet in [modalClassicalSheets[0]]:
    try:
        mk = musicntwrk(TET=12)

        seq,chords,dictionary = mk.dictionary(space='score',scorefil=sheet,music21=True,show=True)

        bnodes,bedges,counts,avgdeg,modul,Gx,Gxu = mk.network(space='score',seq=seq,ntx=True,general=True,distance='euclidean',
                                   grphtype='directed')

        Gx, Gxu, part, full_nodelabel, colors = drawNetwork(nodes=bnodes,edges=bedges,grphtype='directed',k=1.0, colormap='Pastel1')

        staticNetAnalysis(modul, Gxu)

        valuef = timeSeriesAnalysis(seq,chords)

        sections = changePointAnalysis(valuef, 1.0, bnodes)

        #compareAnnotations(sections, sheet)

        real_val,real_count,gen_val,gen_count,real_euler,gen_euler = compareGeneratedVsReal(bnodes,bedges,mk,Gxu,Gx)

        generated_voice_operator_values_m.append(gen_val)
        generated_voice_operator_counts_m.append(gen_count)
        real_voice_operator_values_m.append(real_val)
        real_voice_operator_counts_m.append(real_count)
        generated_lengths_m.append(gen_euler)
        real_lengths_m.append(real_euler)
    except:
        print(sheet)
        pass

#%%

np.save('/Users/abbypribis/Downloads/gen_vals_t.npy',np.array(generated_voice_operator_values_t))
np.save('/Users/abbypribis/Downloads/gen_counts_t.npy',np.array(generated_voice_operator_counts_t))
np.save('/Users/abbypribis/Downloads/real_vals_t.npy',np.array(real_voice_operator_values_t))
np.save('/Users/abbypribis/Downloads/real_counts_t.npy',np.array(real_voice_operator_counts_t))
np.save('/Users/abbypribis/Downloads/gen_lengths_t.npy',np.array(generated_lengths_t))
np.save('/Users/abbypribis/Downloads/real_lengths_t.npy',np.array(real_lengths_t))

np.save('/Users/abbypribis/Downloads/gen_vals_m.npy',np.array(generated_voice_operator_values_m))
np.save('/Users/abbypribis/Downloads/gen_counts_m.npy',np.array(generated_voice_operator_counts_m))
np.save('/Users/abbypribis/Downloads/real_vals_m.npy',np.array(real_voice_operator_values_m))
np.save('/Users/abbypribis/Downloads/real_counts_m.npy',np.array(real_voice_operator_counts_m))
np.save('/Users/abbypribis/Downloads/gen_lengths_m.npy',np.array(generated_lengths_m))
np.save('/Users/abbypribis/Downloads/real_lengths_m.npy',np.array(real_lengths_m))


#%%
score_real_yes = m21.converter.parse('/Users/abbypribis/Downloads/One_OClock_Jump_Edited_Symbol.mxl')
score_yes = m21.harmony.realizeChordSymbolDurations(score_real_yes)
score_real_no = m21.converter.parse('/Users/abbypribis/Downloads/Cherokee_Edited_No_Quarters.mxl')
score_no = m21.harmony.realizeChordSymbolDurations(score_real_no)
#%%
score_yes.chordify().show('text')
#%%
score_no.chordify().show('text')
