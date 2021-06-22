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
import scipy.fftpack as FFT
import pyo as po

def i_spectral_pyo(xv,yv):
	# As i_spectral2 but uses pyo as audio engine
	'''
	# How to use the function 
	# NOTE: there is an instability between pyo and matplotlib when using the 
	# wx-based GUI - graphics MUST be set to False when running in a jupyter notebook
	
	import numpy as np
	import sys,os,re,time

	sys.path.append('/Users/marco/Dropbox (Personal)/Musica/Applications/musicntwrk')
	from sonifiPy import *

	path = './'
	infile = 'DOSCAR.dat'
	xv, y = r_1Ddata(path,infile)
	s,a = i_specral_pyo(xv,y[0],graphics=True)
	s.start()
	time.sleep(5)
	s.stop()
	'''
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

	# start the pyo server 
	s = po.Server().boot()

	# signal to filter 
	sf = po.PinkNoise(.5)

	# FIR filter
	# Create a table of length `buffer size`
	bs = ftvel.shape[0]
	# Create a table of length `buffer size`
	t = po.DataTable(size=bs)
	osc = po.TableRead(t)
	# Share the table's memory with a numpy array.
	arr = np.asarray(t.getBuffer())
	# assign ftvel to the table memory buffer
	arr[:] = ftvel

	# do the convolution 
	a = po.Convolve(sf, table=t, size=t.getSize(), mul=.5).out() #mix(2).out()

	return(s,a)

