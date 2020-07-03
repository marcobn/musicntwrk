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

import sys,os,re
import numpy as np
from scipy.signal import convolve
import scipy.fftpack as FFT
import librosa

def i_spectral_pure(sigpath,sigfil,firpath,firsig):
	
	# As i_spectral2 but does not need audio engine - requires a signal file (wav) to filter
	
	# read signal
	sf, sr = librosa.load(sigpath+sigfil)
	# read function for convolution
	xv, y = r_1Ddata(firpath,firsig)
	yv = y[0]
	nlines = xv.shape[0]

	nbins = int(np.sqrt(nlines)-np.sqrt(nlines)%1)**2
	while nbins > nlines or not(nbins != 0 and ((nbins & (nbins - 1)) == 0)):
		nbins = int((np.sqrt(nbins)-1)**2)
	yfft = np.zeros((nbins),dtype=int)
	for n in range(nbins):
		yfft[n] = n+1

	xminf = xv[0]
	xmaxf = xv[-1]
	xvf=np.asarray(xv)
	xvs = (xv-xminf)/(xmaxf-xminf)*nbins
	for line in range(nlines):
		if xvs[line] >= nbins: xvs[line] = -1 
		xvf[line] = yfft[int(xvs[line])]

	# Normalization of the data shape into MIDI velocity

	yminf = min(yv)
	ymaxf = max(yv)
	yvf=np.asarray(yv)
	yvf = (yv-yminf)/(ymaxf-yminf)*127

	vel=np.zeros((nbins),dtype=float)
	nvel=0
	for note in range(nbins):
		for line in range(nlines):
			if xvf[line] == yfft[note]:
				vel[nvel] = yvf[line]
				nvel=nvel+1
				break

	velmax = max(vel)
	vel /= velmax
	# FFT for FIR filter
	ftvel = FFT.irfft(vel)
	ftvel = FFT.fftshift(ftvel)
	# convolution:
	fir = convolve(sf,ftvel,mode='same')
	
	return(fir)

