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

def MIDImidi(yvf,vnorm=80,dur=4,outmidi='./music'):
	mt = m21.midi.MidiTrack(1)
	# map MIDI velocity to data
	yminf = min(yvf)
	ymaxf = max(yvf)
	yvel = (yvf-yminf)/(ymaxf-yminf)*127 
	yvel=yvel.astype(int)
	# NOTE: full amplitute might be too high volume in some cases
	vmin = min(yvel)
	vmax = max(yvel)
	vel = (yvel-vmin)/(vmax-vmin)*vnorm
	vel = vel.astype(int)

	# duration, pitch, velocity
	data = []
	pb = []
	for i in range(yvf.shape[0]):
		p = [int(1024/dur*abs(yvf[i]-yvf[i-1])+1),
			 int(yvf[i]-12//1),
			 vel[i]]
		data.append(p)
		pb.append(int(np.around((yvf[i]-12)%1,3)*100))

	t = 0
	tLast = 0
	for d, p, v in data:
		dt = m21.midi.DeltaTime(mt)
		dt.time = t - tLast
		# add to track events
		mt.events.append(dt)
		
		me = m21.midi.MidiEvent(mt, type="PITCH_BEND", channel=1)
		#environLocal.printDebug(['creating event:', me, 'pbValues[i]', pbValues[i]])
		me.time = None #d
		me.setPitchBend(pb[i]) # set values in cents
		mt.events.append(me)

		dt = m21.midi.DeltaTime(mt)
		dt.time = t - tLast
		# add to track events
		mt.events.append(dt)
		
		me = m21.midi.MidiEvent(mt)
		me.type = "NOTE_ON"
		me.channel = 1
		me.time = None #d
		me.pitch = p
		me.velocity = v
		mt.events.append(me)

		# add note off / velocity zero message
		dt = m21.midi.DeltaTime(mt)
		dt.time = d
		# add to track events
		mt.events.append(dt)

		me = m21.midi.MidiEvent(mt)
		me.type = "NOTE_ON"
		me.channel = 1
		me.time = None #d
		me.pitch = p
		me.velocity = 0
		mt.events.append(me)

		tLast = t + d # have delta to note off
		t += d # next time

	# add end of track
	dt = m21.midi.DeltaTime(mt)
	dt.time = 0
	mt.events.append(dt)

	me = m21.midi.MidiEvent(mt)
	me.type = "END_OF_TRACK"
	me.channel = 1
	me.data = '' # must set data to empty string
	mt.events.append(me)

	#        for e in mt.events:
	#            print e

	mf = m21.midi.MidiFile()
	mf.ticksPerQuarterNote = 1024 # cannot use: 10080
	mf.tracks.append(mt)

	mf.open(outmidi+'.mid', 'wb')

	mf.write()
	mf.close()
