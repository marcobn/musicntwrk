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

import music21 as m21
import numpy as np
import pandas as pd

from ..musicntwrk import PCSet

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
