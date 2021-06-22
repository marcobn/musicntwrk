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

def MIDIscore(yvf,dur=2,w=None,outxml='./music',outmidi='./music'):
	s1 = m21.stream.Stream()
	for i in range(yvf.shape[0]):
		n = m21.note.Note(yvf[i])
		n.duration = m21.duration.Duration((abs(yvf[i]-yvf[i-1])+1)/dur)
		s1.append(n)
	if w == 'musicxml':
		s1.write('musicxml',outxml+'.xml')
	elif w == 'MIDI':
		s1.write('midi',outmidi+'.mid')
	else:
		s1.show()

