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
	os.remove(path+'DSIscore.inc')

