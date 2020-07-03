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

def WRITEscoreNoTime(nseq,rseq,w=None,outxml='./music',outmidi='./music'):
    
    m = m21.stream.Measure()
    for i in range(nseq.shape[0]):
        n = m21.note.Note(nseq[i])
        n.duration = m21.duration.Duration(4*rseq[i])
        m.append(n)
    m.append(m21.meter.SenzaMisuraTimeSignature('0'))
    
    if w == True:
        m.show('musicxml')
    elif w == 'MIDI':
        m.write('midi',outmidi+'.mid')
    else:
        m.show()
