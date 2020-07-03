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

import pickle
import numpy as np
import music21 as m21

from ..musicntwrk import PCSet
from .enharmonicDictionary import enharmonicDictionary
from .shortHands import shortHands
from ...utils.generalizedOpsName import generalizedOpsName


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
