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

import re,sys
import numpy as np
import music21 as m21

def readScore(input_xml,music21,show,TET):
    '''
    •	read a score in musicxml format
    •	returns the sequence of chords
    '''
    if TET == 12:
        if music21: 
            score = m21.corpus.parse(input_xml)
            try:
                score = score.mergeScores()
            except:
                pass
        else:
            score = m21.converter.parse(input_xml)
        chords = score.chordify()
        if show: chords.show()
        seq = []
        for c in chords.recurse().getElementsByClass('Chord'):
            seq.append(c.normalOrder)
        return(seq,chords)
    elif TET == 24:
        dict24 = {'C':0,'C~':1,'C#':2,'D-':2,'D`':3,'D':4,'D~':5,'D#':6,'E-':6,'E`':7,'E':8,
                            'E~':9,'F`':9,'F':10,'F~':11,'F#':12,'G-':12,'G`':13,'G':14,'G~':15,'G#':16,
                            'A-':16,'A`':17,'A':18,'A~':19,'A#':20,'B-':20,'B`':21,'B':22,'B~':23,'C`':23} 
                            
        score = m21.converter.parse(input_xml)
        chords = score.chordify()
        if show: chords.show()
        seq = []
        for c in chords.recurse().getElementsByClass('Chord'):
            seq.append(str(c))

        clean = []
        for n in range(len(seq)):
            line = ''.join(i for i in seq[n] if not i.isdigit()).split()[1:]
            c = []
            for l in line:
                c.append(dict24[re.sub('>', '', l)])
            clean.append(PCSet(c,TET=24).normalOrder().tolist())
        return(clean,chords)
    else:
        print('temperament needs to be added')
        sys.exit()
    return
