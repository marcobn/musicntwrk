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
from ..musicntwrk import PCmidiR

def WRITEscoreOpsMIDI(nseq,midi=False,w=None,outxml='./music',outmidi='./music',tempo=80,keysig=None,opers=True,
                      normal=False,abs=False,scale=False,idx=None,TET=12,distance='euclidean'):
    try:
        ntot = nseq.shape[0]
    except:
        ntot = len(nseq)
    if midi:
        for n in range(len(nseq)):
            for i in range(len(nseq[n])):
                if nseq[n][i]%1 == 0.0: 
                    nseq[n][i] = int(nseq[n][i])
    m = m21.stream.Stream()
    m.append(m21.tempo.MetronomeMark(tempo))
    m.append(m21.meter.TimeSignature('4/4'))
    for i in range(ntot):
        ch = np.copy(nseq[i])
#        for n in range(1,len(ch)):
#            if ch[n] < ch[n-1]: ch[n] += 12
        if not midi: 
            ch += 60
        n = m21.chord.Chord(ch.tolist())
        if opers:
            if i < ntot-1: 
                if normal:
                    n.addLyric(str(i)+' '+generalizedOpsName(nseq[i],nseq[i+1],TET,distance)[1])
                elif abs:
                    if len(nseq[i]) == len(nseq[i+1]):
    #                    n.addLyric(str(i)+' '+opsName(nseq[i],nseq[i+1],TET))
                        n.addLyric(str(i)+' '+PCmidiR(nseq[i],TET).opsNameVL(PCmidiR(nseq[i+1])))
                    else:
                        r = generalizedOpsName(nseq[i],nseq[i+1],TET,distance)[0]
                        if len(nseq[i]) > len(nseq[i+1]):
                            n.addLyric(str(i)+' '+opsName(nseq[i],r,TET))
                        else:
                            n.addLyric(str(i)+' '+opsName(r,nseq[i+1],TET))
                elif scale:
                    if idx == None:
                        print('need scale degree index for L operators')
                        return
                    try:
                        diff = np.sort(idx[i+1])-np.sort(idx[i])
                        n.addLyric(str(i)+' '+'L('+np.array2string(diff,separator=',').replace(" ","").replace("[","").replace("]","")+')')
                        if i == 0: print('magnitude^2 of sequence operator = ',diff.dot(diff))
                    except:
                        pass
                else:
                    print('no operator mode specified')
        if keysig != None:
            rn = m21.roman.romanNumeralFromChord(n, m21.key.Key(keysig))
            n.addLyric(str(rn.figure))
        m.append(n)
    if w == True:
        m.show()
    elif w == 'musicxml':
        m.show('musicxml')
    elif w == 'MIDI':
        m.show()
        m.show('midi')
        m.write('midi',outmidi+'.mid')
    else:
        pass
