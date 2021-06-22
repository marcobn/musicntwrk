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
def enharmonicDictionary():
    keys = ['C','C#','D-','D','D#','E-','E','F','F#','G-','G','G#','A-','A','A#','B-','B']
    enharmonicDict = {}
    for k in keys:
        ini = m21.note.Note(40).nameWithOctave
        end = m21.note.Note(96).nameWithOctave
        major = [p for p in m21.scale.MajorScale(k).getPitches(ini,end)]
        minor = [p for p in m21.scale.MinorScale(k).getPitches(ini,end)]
        melod = [p for p in m21.scale.MelodicMinorScale(k).getPitches(ini,end)]
        harmo = [p for p in m21.scale.HarmonicMinorScale(k).getPitches(ini,end)]

        allscales = major+minor+melod+harmo

        clean = Remove(allscales)

        Cscale = sorted([str(f) for f in clean])
        Cscale = np.array(Cscale)

        Cmidi = []
        for n in Cscale:
            Cmidi.append(m21.pitch.Pitch(n).midi)
        Cmidi = np.array(Cmidi)
        idx = np.argsort(Cmidi)

        Cdict = dict(zip(Cmidi[idx[:]],Cscale[idx[:]]))

        for i in range(60,84):
            try:
                tmp = Cdict[i]
            except:
                Cdict.update({i:m21.note.Note(i).nameWithOctave})

        enharmonicDict.update({k:Cdict})
    return(enharmonicDict)
