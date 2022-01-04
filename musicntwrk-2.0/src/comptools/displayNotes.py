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

# function to write the sequence as a multiple track MIDI file

def displayNotes(part_list,chord=False,show=None):

    if  not chord:
        score = m21.stream.Stream()
        for i,seq in enumerate(part_list):
            p = m21.stream.Part()
            for s in seq:
                try:
                    n = m21.note.Note(s.midi_number)
                    n.duration = m21.duration.Duration(4*s.dur)
                    p.append(n)
                except:
                    p.append(m21.note.Rest(s.dur))
            score.insert(0,p)
        if show == None: 
            score.show()
        elif show == 'xml': 
            score.show('musicxml')
        elif show == 'midi':
            score.show()
            score.show('midi')
    else:
        score = m21.stream.Stream()       
        for i,seq in enumerate(part_list):
            ch = []
            for s in seq:
                ch.append(m21.note.Note(s.midi_number))
            c = m21.chord.Chord(ch)
            score.append(c)
        if show == None: 
            score.show()
        elif show == 'xml': 
            score.show('musicxml')
        elif show == 'midi':
            score.show()
            score.show('midi')
