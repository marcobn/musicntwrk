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

def MIDImap(pdt,scale,nnote):
	
	# Data to MIDI conversion on given scale
	pmin = min(pdt)
	pmax = max(pdt)

	yvs = (pdt-pmin)/(pmax-pmin)*(nnote-1)
	yvs=yvs.astype(int)
	yvf = np.zeros(pdt.shape[0],dtype=float)
	for i in range(pdt.shape[0]):
		yvf[i]=scale[yvs[i]]
	
	return(yvf)

