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

# import CSound wrapper ctcsound (from csound installation directory)

import ctcsound


def i_spectral2(xv,yv,itime,path='./',instr='noise'):
	
	# Normalization of the energy into FFT bins
	# must be power of 2 for oscil opcode - FIR filter done using scipy.fftpack

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

	########## DSI file for CSound - Finite Impulse Response filter  ###########
	velmax = max(vel)
	vel = vel/velmax
	ftvel = FFT.irfft(vel)
	ftvel = FFT.fftshift(ftvel)
	f=open(path+'DSI_CSound.dat','w')
	for line in range(int(nvel)):
		f.write(str(ftvel[line])+'\n')
	f.close()
	

	########## Initialize and play CSound instruments ############

	if instr == 'noise':
		csd_header = '''
	<CsoundSynthesizer>

	<CsOptions>
	-odac
	;-o '''+path+'''DSI.wav -W
	</CsOptions>

	<CsInstruments>

	sr		=	44100
	ksmps		=	64
	nchnls		=	2
	0dbfs		=	1	;MAXIMUM AMPLITUDE
	massign	0,0
	
	ifn0 ftgen 0,0,'''+str(nbins)+''',-23,"'''+path+'''DSI_CSound.dat"
	gifn0 = ifn0
	'''

	# DSI player
		csd_instr = '''
	instr   99
		;kporttime   linseg  0,0.001,0.05
		;kamp  = 0.5
		;kamp    portk   kamp,kporttime
		;kbeta=  0
		;asigL   noise   kamp, kbeta
		;asigR   noise   kamp, kbeta
		asigL pinker
		asigR pinker
		asigL dconv asigL, ftlen(gifn0), gifn0
		asigR dconv asigR, ftlen(gifn0), gifn0
		kenv linseg 1, '''+str(float(itime)-0.1*float(itime))+''',1,'''+str(0.1*float(itime))+''',0
		asigL	butterlp	asigL, 3000 
		asigR	butterlp	asigR, 3000
		asigL = asigL*kenv
		asigR = asigR*kenv
		asigL clip asigL, 2, 0.7
		asigR clip asigR, 2, 0.7
		outs asigL, asigR
	endin
	'''

		csd_file = csd_header+csd_instr

		csd_tail = '''
	</CsInstruments>

	<CsScore>
	i 99 0 '''+str(itime)+'''
	e
	</CsScore>

	</CsoundSynthesizer>
	'''
		csd_file += csd_tail

	############  Play ############

	cs = ctcsound.Csound()
	cs.compileCsdText(csd_file)
	cs.start()
	cs.perform()
	cs.cleanup()
	cs.reset

	# Clean up files
	os.remove(path+'DSI_CSound.dat')
	
