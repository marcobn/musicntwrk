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

from ..utils.init_list_of_objects import init_list_of_objects

def orchestralVector(inputfile):
    '''
    build orchestral vector sequence from score
    '''
    score = m21.converter.parse(inputfile)
    score = score.sliceByBeat()
    Nparts = len(score.getElementsByClass(m21.stream.Part))
    orch = init_list_of_objects(Nparts)
    for p in range(Nparts):
        Nmeasures = len(score.getElementsByClass(m21.stream.Part)[p].\
                        getElementsByClass(m21.stream.Measure))
        for m in range(0,Nmeasures):
            mea = score.getElementsByClass(m21.stream.Part)[p].\
                    getElementsByClass(m21.stream.Measure)[m]
            try:
                for n in mea.notesAndRests:
                    if n.beat%1 == 0.0: 
                        if n.isRest:
                            orch[p].append(0)
                        else:
                            orch[p].append(1)
            except:
                print('exception: most likely an error in the voicing of the musicxml score',\
                      'part ',p,'measure ',m)
    orch = np.asarray(orch).T
    if len(orch.shape) == 1:
        print('WARNING: the number of beats per part is not constant')
        print('         check the musicxml file for internal consistency')
        a = []
        for i in range(orch.shape[0]):
            a.append(len(orch[i]))
        clean = np.zeros((min(a),orch.shape[0]),dtype=int)
        for j in range(orch.shape[0]):
            for i in range(min(a)):
                clean[i,j] = orch[j][i]
        orch = clean
    try:
        num = np.zeros(orch.shape[0],dtype=int)
        for n in range(orch.shape[0]):
            num[n] = int(''.join(str(x) for x in orch[n,:]), base=2)
    except:
        num = None
         
    return(score,orch,num)

