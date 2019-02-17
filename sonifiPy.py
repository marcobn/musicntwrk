#
# sonifiPy
#
# A python/Csound/music21 library for data sonification
#
# Copyright (C) 2018 Marco Buongiorno Nardelli
# http://www.sonifipy.com, http://www.materialssoundmusic.com, mbn@unt.edu
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

import numpy as np
import numpy.random as rn
import sys,os,re
import scipy.fftpack as FFT
import random
import matplotlib.pyplot as plt
import music21 as m21

# import CSound wrapper ctcsound (from csound installation directory)
# path must be changed by user

sys.path.append('/Users/marco/Dropbox (Personal)/Musica/Applications/CSound/csound/interfaces/')
import ctcsound

def r_1Ddata(path,fileread):

	# Read data as columns (x,y1,y2,y3...)

	# Read the whole file into a single variable, which is a list of every row of the file.

	f=open(path+fileread,'r')
	lines=f.readlines()
	f.close()
	nlines = len(lines)
	ncol = len(lines[0].split())

	if ncol > 1:
		x = np.zeros(nlines,dtype=float)
		y = np.zeros((ncol-1,nlines),dtype=float)

		f=open(path+fileread,'r')
		for l,line in enumerate(f):
			p = line.split()
			x[l] = float(p[0])
			for n in range(0,ncol-1):
				y[n,l] = float(p[n+1])
		f.close()
	elif ncol ==1:
		x = np.linspace(0,nlines,nlines)
		y = np.zeros((1,nlines),dtype=float)
		f=open(path+fileread,'r')
		for l,line in enumerate(f):
			y[0,l] = float(line.split()[0])
		f.close()

	return(x,y)

def i_spectral(xv,yv,itime,path='./',instr='noise'):
	
	# Normalization of the energy into FFT bins
	# must be power of 2 for oscil opcode

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

	########## DSI file for CSound processing on fft grid ###########
	velmax = max(vel)
	f=open(path+'DSI_CSound.dat','w')
	for line in range(int(nvel)):
		f.write(str(vel[line]/float(velmax))+'\n')
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

	/****************************************************
	instruments
	*****************************************************/
	ifn0 ftgen 0,0,'''+str(nbins)+''',23,"'''+path+'''DSI_CSound.dat"
	gifn0 = ifn0

	/*****************************************************
	irn IR ifn
	irn - impulse response output function table number
	ifn - amplitude response function table number
	*****************************************************/
	opcode IR,i,i
		ifn xin
		iflen2 = ftlen(ifn)
		iflen = 2*iflen2
		iSpec[] init iflen2
		icnt init 0
		copyf2array iSpec,ifn
		iIR[] rifft r2c(iSpec)
		irn ftgen 0,0,iflen,7,0,iflen,0
		while icnt < iflen2 do
			itmp = iIR[icnt]
			iIR[icnt] = iIR[icnt + iflen2]
			iIR[icnt + iflen2] = itmp
			icnt +=1
		od
	copya2ftab iIR,irn
	xout irn
	endop

	/*****************************************************
	asig FIR ain,ifn
	ain - input audio
	ifn - amplitude response function table number
	*****************************************************/
	opcode FIR,a,ai
		asig,ifn xin
		irn IR ifn
		xout dconv(asig,ftlen(irn),irn)
	endop
	'''

	# DSI player
		csd_instr = '''
	instr   99
		kporttime   linseg  0,0.001,0.05
		kamp  = 0.5
		kamp    portk   kamp,kporttime
		kbeta=  0
		asigL   noise   kamp, kbeta
		asigR   noise   kamp, kbeta
		asigL FIR asigL, gifn0
		asigR FIR asigR, gifn0
		kenv linseg 1, '''+str(float(itime)-0.1*float(itime))+''',1,'''+str(0.1*float(itime))+''',0
		asigL	butterlp	asigL, 6000 
		asigR	butterlp	asigR, 6000 
		asigL = asigL*kenv
		asigR = asigR*kenv
		outs asigL, asigR
	endin
	'''

		csd_file = csd_header+csd_instr

		csd_tail = '''
	</CsInstruments>

	<CsScore>
	i 99 0 '''+itime+'''
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
	
def i_spectral2(xv,yv,itime,path='./',instr='noise'):
	
	# Normalization of the energy into FFT bins
	# must be power of 2 for oscil opcode

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
	print(ftvel[0])
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
	i 99 0 '''+itime+'''
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

def i_time_series(xv,yv,path='./',instr='csb701'):
	
	# Normalization of the bands into frequencies allowing microtones
	
	nbnd, nkpoint = yv.shape
	xminf = np.min(yv)
	xmaxf = np.max(yv)
	yvf = np.asarray(yv)
	yvf = (yv-xminf)/(xmaxf-xminf)*4158.51+27.5

	if instr == 'csb701':
		f=open(path+'DSIscore.inc','w')
		for band in range(nbnd):
			xvel = np.ones((nkpoint),dtype=float)/2.
			xdur = np.ones((nkpoint),dtype=float)*3.

			f.write ('i'+' '+str(band+1)+' '+'0'+' '+str(xdur[0])+' '+str(xvel[0])+' '+str(yvf[band,0])+'\n')
			for line in range(1,nkpoint):
				f.write ('i'+' '+str(band+1)+' '+str(line)+' '+str(xdur[line])+' '+str(xvel[line])+' '+str(yvf[band,line])+'\n')
			f.write (' '+'\n')
		f.close()

########## Initialize and play CSound instruments ############
		csd_file = '''
<CsoundSynthesizer>

<CsOptions>
-odac
;-o '''+path+'''DSI.wav -W
</CsOptions>

<CsInstruments>

sr		=	44100
ksmps		=	64
nchnls		=	2
0dbfs		=	2	;MAXIMUM AMPLITUDE
massign	0,0


/****************************************************
instruments
*****************************************************/

'''

# csound player

# dynamical allocation of instruments
		csd_instr = ''''''
		angle = np.linspace(-50,50,nbnd)
		kL = np.sqrt(2)/2.0 * (np.cos(angle) - np.sin(angle))
		kR = np.sqrt(2)/2.0 * (np.cos(angle) + np.sin(angle))
		for i in range(nbnd):
			csd_instr += '''
instr   '''+str(i+1)+'''
	idur	=	abs(p3)	; need positive dur for envelope
	ir	tival	 	; find out if this is a tied note
	i1	=	-1	; set oscil phase for a tied note
 	tigoto	slur	; skip reinit of env on tied notes
		i1	=	0	; first note, so reset oscil phase
		;katt	linseg     0, 0.01, 1, 0.1,  0.1, p3-0.21, 0.1, 0.1, 0	; overall envelope
		katt    cosseg 0, p3/4, 0.9, p3/2, 0.9, p3/4, 0
		;katt    transeg 0, p3/4, -4, 0.9, p3/2, 0, 0.9, p3/4, -4, 0
	slur:
		if ir == 0 kgoto tone	; no swell if first note
		kslur	linseg	0, idur/2, p4, idur/2, 0	; simple swell shape
		katt	=	katt+kslur	; add swell to primary envelope
	tone:
		asig	oscili	katt, p5, 1, i1	; reinit phase if first note
 	outs asig*'''+str(kL[i])+''',asig*'''+str(kR[i])+'''
endin
'''

			csd_file += csd_instr
			csd_instr = ''

		csd_tail = '''
</CsInstruments>

<CsScore>
f 1 0 1024  10    1     0.95  0.1   0.05  0.01  0.001
t 0 3600
#include "'''+path+'''DSIscore.inc"
e
</CsScore>

</CsoundSynthesizer>
'''

		csd_file += csd_tail

	if instr == 'csb702':
		f=open(path+'DSIscore.inc','w')
		for band in range(nbnd):
			xvel = np.ones((nkpoint),dtype=float)/2.
			xdur = np.ones((nkpoint),dtype=float)*(-3.)

			f.write ('i'+' '+str(band+1)+' '+'0'+' '+str(xdur[0])+' '+str(xvel[0])+' '+str(yvf[band,0])+' '+str(yvf[band,0])+' np4 '+'\n')
			for line in range(1,nkpoint-1):
				f.write ('i'+' '+str(band+1)+' '+'+'+' '+str(xdur[line])+' '+str(xvel[line])+' '+str(yvf[band,line])+' pp5 np4 '+'\n')
			f.write ('i'+' '+str(band+1)+' '+'+'+' '+str(xdur[nkpoint-1])+' '+str(xvel[nkpoint-1])+' '+str(yvf[band,nkpoint-1])+' pp5 np4 '+'\n')
			f.write (' '+'\n')
		f.close()

########## Initialize and play CSound instruments ############
		csd_file = '''
<CsoundSynthesizer>

<CsOptions>
-odac
;-o '''+path+'''DSI.wav -W
</CsOptions>

<CsInstruments>

sr		=	44100
ksmps		=	64
nchnls		=	2
0dbfs		=	2	;MAXIMUM AMPLITUDE
massign	0,0


/****************************************************
instruments
*****************************************************/

'''

# csound player

# dynamical allocation of instruments
		csd_instr = ''''''
		angle = np.linspace(-50,50,nbnd)
		kL = np.sqrt(2)/2.0 * (np.cos(angle) - np.sin(angle))
		kR = np.sqrt(2)/2.0 * (np.cos(angle) + np.sin(angle))
		for i in range(nbnd):
			csd_instr += '''
instr   '''+str(i+1)+'''
	idur      =  abs(p3)                            ; MAIN INIT BLOCK
	ipch1     =  cpspch(p6)
	ipch2     =  cpspch(p5)
	kpch      =  ipch2
	iport     =  0.01                                ; 100msec PORTAMENTO
	iatt      =  0.05                               ; DEFAULT DURS FOR AMPLITUDE RAMPS
	idec      =  0.05                               ; ASSUME THIS IS A TIED NOTE:
	iamp      =  p4                                 ; SO START AT p4 LEVEL...
	i1        =  -1                                 ; ... AND KEEP PHASE CONTINUITY
	ir        tival                                 ;  CONDITIONAL INIT BLOCK:TIED NOTE?
						tigoto    start
	i1        =  0                                  ; FIRST NOTE: RESET PHASE
	iamp      =  0                                  ; AND ZERO iamp
start:
	iadjust   =  iatt+idec
if              idur >= iadjust igoto doamp       ; ADJUST RAMP DURATIONS FOR SHORT...
	iatt      =  idur/2-0.005                       ; ... NOTES, 10msecs LIMIT
	idec      =  iatt                               ; ENSURE NO ZERO-DUR SEGMENTS
	iadjust   =  idur-0.01
	iport     =  0.005                              ; MAKE AMPLITUDE RAMP...
doamp:
																																; ... (arate FOR CLEANNESS) AND...
	ilen      =  idur-iadjust                       ; ... SKIP PITCH RAMP GENERATION...
	amp       linseg    iamp, iatt, p4, ilen, p4, idec, p7
if              ir == 0 || p6 == p5 kgoto slur    ;...IF FIRST NOTE OR TIE.
; MAKE PITCH RAMP, PORTAMENTO AT START OF NOTE
	kpramp    linseg    ipch1, iport, ipch2, idur-iport, ipch2
	kpch      =  kpramp
slur:
; MAKE THE NOTE
	aamp      =  amp
	asig      oscili    aamp, kpch, 1, i1

 	outs asig*'''+str(kL[i])+''',asig*'''+str(kR[i])+'''
endin
'''

			csd_file += csd_instr
			csd_instr = ''

		csd_tail = '''
</CsInstruments>

<CsScore>
f 1 0 1024  10    1     0.95  0.1   0.05  0.01  0.001
t 0 2640
#include "'''+path+'''DSIscore.inc"
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
	os.remove(path+'DSIscore.inc')

def scaleMapping(scale):

	# Definitions for mapping
	scale=str(scale)

	# 1024 bins for FFT

	if scale == 'fft1024':
	   scale =\
	   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024]
	   ys = np.array(scale)
	   nnote=len(scale)

	# 256 bins for FFT

	if scale == 'fft256':
	   scale =\
	   [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256]
	   ys = np.array(scale)
	   nnote=len(scale)

	# full chromatic scale
	if scale == 'chrom':
	   scale =  [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,\
	   60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,\
	   102,103,104,105,106,107,108]
	   ys = np.array(scale)
	   nnote=len(scale)

	# partial chromatic scale
	if scale == 'chrompar':
	   scale =  [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]
	   #,46,47,48,49,50,51,52,53,54,55,56,57,58,59,\
	   #60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108]
	   ys = np.array(scale)
	   nnote=len(scale)

	# pentatonic (all black keys)
	if scale == 'penta':
	   scale = [22,25,27,30,32,34,37,39,42,44,46,49,51,54,56,58,61,63,66,68,70,73,75,78,80,82,85,87,90,92,94,97,99,102,104,106]
	   ys = np.array(scale)
	   nnote=len(scale)

	# diatonic (all white keys - major, natural minor, modes)
	if scale == 'dia':
	   scale =  [22,23,24,26,28,29,31,33,35,36,38,40,41,43,45,47,48,50,52,53,55,57,59,60,\
	   62,64,65,67,69,71,72,74,76,77,79,81,83,84,86,88,89,91,93,95,96,98,100,101,103,105,107,108]
	   ys = np.array(scale)
	   nnote=len(scale)

	# harmonic minor
	if scale == 'harm':
	   scale =  [21,23,24,26,28,29,32,33,35,36,38,40,41,44,45,47,48,50,52,53,56,57,59,\
	   60,62,64,65,68,69,71,72,74,76,77,80,81,83,84,86,88,89,92,93,95,96,98,100,101,104,105,107,108]
	   ys = np.array(scale)
	   nnote=len(scale)

	# diminished
	if scale == 'dim':
	   scale =  [21,23,24,26,27,29,30,32,33,35,36,38,39,41,42,44,45,47,48,50,51,53,54,56,57,59,\
	   60,62,63,65,66,68,69,71,72,74,75,77,78,80,81,83,84,86,87,89,90,92,93,95,96,98,99,101,102,104,105,107,108]
	   ys = np.array(scale)
	   nnote=len(scale)

	# whole tone
	if scale == 'whole':
	   scale =  [21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,\
	   61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107]
	   ys = np.array(scale)
	   nnote=len(scale)

	# twelve tone row (B-A-C-H - from A. Webern, Sting Quartet op. 28)
	if scale == 'ttr1':
	   scale = [22,21,24,23,27,28,25,26,34,33,36,35,39,40,37,38,30,29,32,31,46,45,48,47,51,52,49,50,42,41,44,43,58,57,60,\
	   59,63,64,61,62,54,53,56,55,70,69,72,71,75,76,73,74,66,65,68,67,82,81,84,83,87,88,85,86,78,77,80,79,94,93,96,95,99,\
	   100,97,98,90,89,92,91,106,105,108,107,102,101,104,103]
	   ys = np.array(scale)
	   nnote=len(scale)

	# twelve tone row (random #1)
	if scale == 'ttr2':
	   scale = [23,30,21,28,25,29,32,27,31,24,26,22,35,42,33,40,37,41,44,39,43,36,38,34,47,54,45,52,49,53,56,51,55,48,50,46,59,66,57,64,61,65,68,63,67,60,62,58,71,78,69,76,73,77,80,75,79,72,74,70,83,90,81,88,85,89,92,87,91,84,86,82,95,102,93,100,97,101,104,99,103,96,98,94,107,105,108,106]
	   ys = np.array(scale)
	   nnote=len(scale)

	# O. Messiaen mode of limited transposition 3^2 <1 3 4 5 7 8 9 11 0>
	if scale == 'mlt3.2':
	   scale = [40,41,43,44,45,47,48,49,51,52,53,55,56,57,59,60,61,63,64,65,67,68,69,\
	   71,72,73,75,76,77,79,80,81,83,84,85,87,88,89,91,92,93,95,96,97,99,100,101,103,104,105,107,108]
	   ys = np.array(scale)
	   nnote=len(ys)

	# microtonal chromatic scale in given interval (from "Meditation" for Vla, voice and Cb)
	if scale == 'micro':
		# scale definition (including quarter tones)
		# Viola: 72-107
		# Double bass: 74-91
		i=72  # initial pitch (MIDI)
		f=107 # final pitch
		fac = 2 # microtone factor (fac = 2 allows quarter tones)
		ys = np.round(np.linspace(fac*i,fac*f))/2
		nnote=len(ys)
	
	# random distribution of pitches in given interval
	if scale == 'random':
	   ys = rn.randint(21,108,size=88)
	   nnote = 88

	return(ys,nnote)

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