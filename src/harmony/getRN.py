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

import numpy as np
import music21 as m21
from ..musicntwrk import PCSet
from .enharmonicDictionary import enharmonicDictionary

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
