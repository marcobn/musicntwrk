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

from ..musicntwrk import PCSet
from ..utils.Remove import *

def scoreDictionary(seq,TET=12):
    '''
    •	build the dictionary of pcs in any score in musicxml format
    •	use readScore() to import the score data as sequence
    '''
    s = Remove(seq)
    v = []
    name = []
    prime = []
    if TET == 12:
        for i in range(len(s)):
            p = PCSet(np.asarray(s[i][:]),TET=TET)
            v.append(p.intervalVector())
            name.append(''.join(m21.chord.Chord(p.normalOrder().tolist()).pitchNames))
            prime.append(np.array2string(p.normalOrder(),separator=',').replace(" ",""))
        vector = np.asarray(v)
        name = np.asarray(name)
    elif TET == 24:
        dict24 = {'C':0,'C~':1,'C#':2,'D-':2,'D`':3,'D':4,'D~':5,'D#':6,'E-':6,'E`':7,'E':8,
                            'E~':9,'F`':9,'F':10,'F~':11,'F#':12,'G-':12,'G`':13,'G':14,'G~':15,'G#':16,
                            'A-':16,'A`':17,'A':18,'A~':19,'A#':20,'B-':20,'B`':21,'B':22,'B~':23,'C`':23}
        for i in range(len(s)):
            p = PCSet(np.asarray(s[i][:]),TET=TET)
            v.append(p.intervalVector())
            tmp = []
            for i in p.pcs:
                tmp.append(list(dict24.keys())[list(dict24.values()).index(i)]) 
            name.append(''.join(tmp))
            prime.append(np.array2string(p.normalOrder(),separator=',').replace(" ",""))
        vector = np.asarray(v)
        name = np.asarray(name)
    else:
        print('temperament needs to be added')
        sys.exit()

    # Create dictionary of pitch class sets
    reference = []
    for n in range(len(name)):
        entry = [name[n],prime[n],
                np.array2string(vector[n,:],separator=',').replace(" ","")]
        reference.append(entry)

    dictionary = pd.DataFrame(reference,columns=['class','pcs','interval'])
    
    return(dictionary)
