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
import communities as cm

def tonalPartition(seq,chords,nodes,Gx,Gxu,resolution=1.0,randomize=None,display=False):
    part = cm.best_partition(Gxu,resolution=resolution,randomize=randomize)
    dn = np.array(nodes)
    labe = []
    modu = []
    modul = []
    for m in range(len(dn)):
        labe.append(str(dn[m][0]))
        modu.append(part[str(m)])
        modul.append([str(dn[m][0]),Gx.degree()[str(m)],part[str(m)]])
    modul = pd.DataFrame(modul,columns=['Label','Degree','Modularity'])
    moduldict = dict(zip(labe,modu))

    if display:
    # display the score with modularity classes
        mc = []
        seqmc = []
        for n in range(len(seq)):
            p = PCSet(np.asarray(seq[n]))
            if p.pcs.shape[0] == 1:
                nn = ''.join(m21.chord.Chord(p.pcs.tolist()).pitchNames)
            else:
                nn = ''.join(m21.chord.Chord(p.normalOrder().tolist()).pitchNames)
            mc.append(nn)
            seqmc.append(moduldict[nn])

        l = 0
        analyzed = copy.deepcopy(chords)
        for c in analyzed.recurse().getElementsByClass('Chord'):
            c.closedPosition(forceOctave=4,inPlace=True)
            c.addLyric(str(l))
            c.addLyric(seqmc[l])
            l += 1
        analyzed.show('musicxml')
    return(moduldict,modul)
