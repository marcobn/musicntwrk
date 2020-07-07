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

from ..utils.generalizedOpsName import generalizedOpsName
from ..utils.opsName import opsName

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
