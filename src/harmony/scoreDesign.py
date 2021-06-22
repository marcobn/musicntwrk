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

def scoreDesign(pitches,durations,fac=1,TET=12,write=False):

    notelist = []
    for i in range(len(pitches)):
        c = PCSet(np.random.permutation(np.asarray(pitches[i])),UNI=False,ORD=False,TET=TET)
        for i in range(c.pcs.shape[0]):
            notelist.append(c.pcs[i]/fac+60)
    notelist = np.asarray(notelist)

    stream1 = m21.stream.Stream()
    stream1.insert(0, m21.meter.TimeSignature('4/4'))
    for i in range(notelist.shape[0]):
        nota = m21.note.Note(notelist[i])
        nota.duration = durations[i%len(durations)]
        stream1.append(nota)
    if write:
        stream1.show('musicxml')
    else:
        stream1.show()