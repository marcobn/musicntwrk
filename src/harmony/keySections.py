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