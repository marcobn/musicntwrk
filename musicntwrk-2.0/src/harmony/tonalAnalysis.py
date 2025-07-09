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

import copy
import music21 as m21
import numpy as np
import pandas as pd

from ..utils.opsDistance import opsDistance
from ..utils.generalizedOpsName import generalizedOpsName

def tonalAnalysis(chords,sections,key,enharm=[['C','C']],write=None):
    chs = []
    for c in chords:
        chs.append(m21.chord.Chord(c))
    regionKey = []
    for n in range(len(sections)-1):
        ini = sections[n]
        end = sections[n+1]
        for i in range(ini,end,1):
            for enh in enharm:
                if key[n] == enh[0]:
                    key[n] = enh[1]
            regionKey.append(key[n])
    regionKey.append(key[n])
    regionKey.append(key[n])
    reference = []
    rnum = []
    oper = []
    for n in range(len(chs)):
        rn = m21.roman.romanNumeralFromChord(chs[n],m21.key.Key(regionKey[n])).romanNumeralAlone
        fig = m21.roman.postFigureFromChordAndKey(chs[n], m21.key.Key(regionKey[n]))
        rnum.append(rn+fig)
        try:
            _,ops = generalizedOpsName(chs[n].pitchClasses,chs[n+1].pitchClasses,distance='euclidean',TET=12)
        except:
            ops = ''
        oper.append(ops)
        entry = [chs[n].pitchClasses,chs[n].pitchNames,rn+fig,ops,regionKey[n]]
        reference.append(entry)
        
    # Set dictionary as pandas dataframe
    analysis = pd.DataFrame(reference,columns=['pcs','chord','rn','ops','region'])
    
    if write:
        # write the analyzed score in musicxml
        l = 0
        analyzed = copy.deepcopy(chs)
        rn_score = m21.stream.Stream()
        for c in analyzed: #.recurse().getElementsByClass('Chord'):
            c.closedPosition(forceOctave=4,inPlace=True)
            c.addLyric('')
            c.addLyric('')
            if l == 0: c.addLyric(regionKey[0]+':'+str(rnum[0]))
            if l >= 1:
                if opsDistance(str(oper[l-1]))[1] == 0.0:
                    if regionKey[l] == regionKey[l-1]:
                        c.addLyric('')
                    else:
                        c.addLyric(regionKey[l]+':')
                else:
                    if regionKey[l] == regionKey[l-1]:
                        c.addLyric(str(rnum[l]))
                    else:
                        c.addLyric(regionKey[l]+':'+str(rnum[l]))
            if l < len(chs)-1: 
                l += 1
                rn_score.append(c)
            else: 
                break

        rn_score.show('musicxml')
        
    return(analysis)

def string2Chord(ch,key):
    s = []
    i=0
    while i < len(ch):
        if i+1 < len(ch):
            if ch[i+1] == '-':
                e = ch[i]+'-'
                i += 2
            elif ch[i+1] == '#':
                e = ch[i]+'#'
                i += 2
            else:
                e = ch[i]
                i += 1
        else:
            e = ch[i]
            i += 1    
        s.append(e)
    # get RN
    rn = m21.roman.romanNumeralFromChord(m21.chord.Chord(s),m21.key.Key(key)).figure
    return(s,rn)

def transMatrix(Gnodes,Gedges,idx,key='C',N=10):
    # calculate weights of edges of directional network (transition probability matrix)
    idsec = idx
    edge,weight = np.unique(np.array(Gedges[idsec][["Source","Target"]],dtype=int),return_counts=True,axis=0)
    weight = weight/len(Gedges[idsec])*100
    tmp = []
    for i in range(len(edge)):
        _,rn0 = string2Chord(Gnodes[idsec].iloc[edge[i][0]][0],key)
        _,rn1 = string2Chord(Gnodes[idsec].iloc[edge[i][1]][0],key)
        tmp.append([Gnodes[idsec].iloc[edge[i][0]][0],Gnodes[idsec].iloc[edge[i][1]][0],
                    rn0,rn1,weight[i]])
    transmatrix = pd.DataFrame(tmp,columns=['Source','Target','RN source','RN target','Weight'])
    print(transmatrix.sort_values('Weight',ascending=False).head(n=N).to_string(index=False))
    return(transmatrix.sort_values('Weight',ascending=False))