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
from ..comptools.genmidi import Midi

# function to write the sequence as a multiple track MIDI file

def poly_midi(filename, part_list):
    midi = Midi(number_tracks = len(part_list),tempo=120)
    for i,p in enumerate(part_list):
        midi.seq_notes(p,track=i)
    midi.write("./" + filename)
    score = m21.converter.parse("./" + filename)
    score.show()