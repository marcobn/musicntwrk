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
    for c in chords.recurse().getElementsByClass('Chord'):
        chs.append(c)
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
        analyzed = copy.deepcopy(chords)
        for c in analyzed.recurse().getElementsByClass('Chord'):
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
            else: 
                break
        analyzed.show('musicxml')
        
    return(analysis)