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

import pandas as pd
import numpy as np
import music21 as m21

from ..musicntwrk import PCmidiR
from ..utils.Remove import *

def scoreMIDIDictionary(seq):
    '''
    •	build the dictionary of pcs in any score in musicxml format
    •	use readScore() to import the score data as sequence - MIDI version
    '''
    s = Remove(seq)
    v = []
    name = []
    prime = []

    for i in range(len(s)):
        p = PCmidiR(np.asarray(s[i][:]))
        v.append(p.intervals())
        name.append(str([pcs.nameWithOctave for pcs in m21.chord.Chord(p.pitches).pitches]))
        prime.append(np.array2string(np.unique(p.normalOrder().pcs),separator=',').replace(" ",""))
    vector = np.asarray(v)
    name = np.asarray(name)

    # Create dictionary of pitch class sets
    reference = []
    for n in range(len(name)):
        entry = [name[n],prime[n],
                np.array2string(vector[n],separator=',').replace(" ","")]
        reference.append(entry)

    dictionary = pd.DataFrame(reference,columns=['chord','pcs','interval'])
    
    return(dictionary)
