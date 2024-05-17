#
# msctools: my collection of composing and performing tools in python
#
# © 2023 Marco Buongiorno Nardelli
#

import time
import numpy as np

from .converters import *
from .devices import Spat
import musicntwrk.msctools.cfg as cfg
from .decorators import threading_decorator

# Dynamics

@threading_decorator
def multiEnvLive(tracklist,T,omega=None):

	# general function that builds the envelope series for each individual channel
	# with a constant amplitude algorithm and arbiitrarily chosen time of flight for each channel
	# Om = list of frequencies for individual channels - determines the time spent on each speaker
	# len(OM) in input = number of channels to distribute sound to
	# T = length of the sample
	
	assert type(tracklist) == list, 'must be a list of tracks'
	if omega == None:
		omega = np.ones(len(tracklist)).tolist()
	Om = omega.copy()
	Om.append(1)
	nch = len(Om)-1
	if nch == 1:
		# trivial single channel case
		env = [np.ones(T)]
	else:
		# 2 or more channels
		sections = [0]
		for n in range(0,nch):
			sections.append(np.pi/2/Om[n])
		L = sum(sections)-sections[-1]
		x = np.linspace(0,L,int(T//cfg.CLOCK))
		env = [None]*(nch)
		ienv = [None]*(nch)
		zeroup = 0
		zerodown = 0
		for n in range(nch):
			zeroup += sections[n]
			zerodown = zeroup - sections[n]
			zeroflat = zeroup + sections[n+1]
			T = Om[n]
			Tp = Om[n-1]
			env[n] = np.cos(T*(x-zeroup))**2
			ienv[n] = np.sin((Tp)*(x-zerodown))**2
			env[n][x < zeroup] = ienv[n][x < zeroup]
			env[n][x < zerodown] = 0
			env[n][x > zeroflat] = 0
	env = scale(np.array(env),[0.0,1.0],[0.0,0.85])
	for i in range(len(x)):
		for n,tr in enumerate(tracklist):
			tr.volume(env[n][i],mode='set')
		time.sleep(cfg.CLOCK)

@threading_decorator
def crescendo(tracks,tracklist,Vini,Vend,T):
	assert type(tracklist) == list, 'must be a list of tracks'
	# input volumes in dB, time in seconds
	# set initial volume (decimal)
	Vini = db2value(Vini)
	Vend = db2value(Vend)
	assert Vini <= Vend
	for tr in tracklist:
		tracks[tr].volume(Vini,mode='set')
	nt = int(T/cfg.CLOCK)
	dV = (Vend - Vini)/nt
	V = Vini
	for t in range(nt):
		time.sleep(cfg.CLOCK)
		V += dV
		for tr in tracklist:
			tracks[tr].volume(V,mode='set')
	for tr in tracklist:
		tracks[tr].volume(Vend,mode='set')

@threading_decorator		
def decrescendo(tracks,tracklist,Vini,Vend,T):
	assert type(tracklist) == list, 'must be a list of tracks'
	# input volumes in dB, time in seconds
	# set initial volume (decimal)
	Vini = db2value(Vini)
	Vend = db2value(Vend)
	assert Vini >= Vend
	for tr in tracklist:
		tracks[tr].volume(Vini,mode='set')
	nt = int(T/cfg.CLOCK)
	dV = (Vini - Vend)/nt
	V = Vini
	for t in range(nt):
		time.sleep(cfg.CLOCK)
		V -= dV
		for tr in tracklist:
			tracks[tr].volume(V,mode='set')
	for tr in tracklist:
		tracks[tr].volume(Vend,mode='set')
		
def setVol(tracks,tracklist,V):
	assert type(tracklist) == list, 'must be a list of tracks'
	# input volumes in dB, time in seconds
	# set volume (decimal)
	V = db2value(V)
	for tr in tracklist:
		tracks[tr].volume(V,mode='set')
		
# Panning Live

@threading_decorator			
def lineCycleLive(tracks,tracklist,pan0,T,cycle=1,dir='r',*args):
	# Spans the whole range [-1.0,1.0] starting from an arbitrary position in time T
	# dir='r' starts movement in r direction ('l' for left)
	assert type(tracklist) == list, 'must be a list of tracks'
	if dir == 'r':
		c0 = 0
	else:
		c0 = 1
	nt = int(T/cfg.CLOCK)
	nt0 = int((pan0+1)/2*nt)
	
	X = np.linspace(-pan0,pan0,nt)

	if dir == 'r':
		for i in range(nt0,nt):
			time.sleep(cfg.CLOCK)
			for tr in tracklist:
				tracks[tr].panning(X[i*(-1)**c0])
			if cfg.stop_source[tracklist]: break
	else:
		for i in range(nt0)[::-1]:
			time.sleep(cfg.CLOCK)
			for tr in tracklist:
				tracks[tr].panning(X[i])
			if cfg.stop_source[tracklist]: break
	if c0 == 1: cycle += 1
	for c in range((c0+1),cycle):
		for i in range(nt):
			time.sleep(cfg.CLOCK)
			for tr in tracklist:
				tracks[tr].panning(X[i*(-1)**c0])
			if cfg.stop_source[tracklist]: break
		if cfg.stop_source[tracklist]: break

@threading_decorator			
def linePingPongLive(tracks,tracklist,pan0,T,cycle=1,dir='r',*args):
    # Spans the whole range [-1.0,1.0] starting from an arbitrary position in time T
    # dir='r' starts movement in r direction ('l' for left)
    assert type(tracklist) == list, 'must be a list of tracks'
    if dir == 'r':
        c0 = 0
    else:
        c0 = 1
    nt = int(T/cfg.CLOCK)
    nt0 = int((pan0+1)/2*nt)
    
    X = np.linspace(-pan0,pan0,nt)
    
    if dir == 'r':
        for c in range(cycle):
            for i in range(nt):
                time.sleep(cfg.CLOCK)
                for tr in tracklist:
                    tracks[tr].panning(X[i*(-1)**c0])
                if cfg.stop_source[tracklist]: break
            c0 = (c0+1)%2
            for i in range(nt):
                time.sleep(cfg.CLOCK) 
                for tr in tracklist:
                    tracks[tr].panning(X[i*(-1)**c0])
                if cfg.stop_source[tracklist]: break
            c0 = (c0+1)%2
            if cfg.stop_source[tracklist]: break
    elif dir == 'l':
        for c in range(cycle):
            for i in range(nt):
                time.sleep(cfg.CLOCK)
                for tr in tracklist:
                    tracks[tr].panning(X[i*(-1)**c0])
                if cfg.stop_source[tracklist]: break
            c0 = (c0+1)%2
            for i in range(nt):
                time.sleep(cfg.CLOCK) 
                for tr in tracklist:
                    tracks[tr].panning(X[i*(-1)**c0])
                if cfg.stop_source[tracklist]: break
            c0 = (c0+1)%2
            if cfg.stop_source[tracklist]: break

# Position (generic device)

@threading_decorator		
def lines(source,device,posA,posB,T,cycle=1,*args):
	# Draws a line between posA and posB in time T
	# to be used in source placement
	assert type(posA) == list, 'posA is a point in 3D space'
	assert type(posB) == list, 'posA is a point in 3D space'
	nt = int(T/cfg.CLOCK)
	# Formula to correct for the delay in the Spat OSC messaging server (empirical!!!)
	wait = (T-0.0028*nt)/nt
	X = np.linspace(posA[0],posB[0],nt)
	Y = np.linspace(posA[1],posB[1],nt)
	Z = np.linspace(posA[2],posB[2],nt)
	for c in range(cycle):
		for i in range(nt):
			try:
				device.position([X[i*(-1)**c],Y[i*(-1)**c],Z[i*(-1)**c]],mode='set')
				time.sleep(cfg.CLOCK)
			except:
				# for SpatGris
				device.car(X[i*(-1)**c],Y[i*(-1)**c],Z[i*(-1)**c],*args)
				time.sleep(wait)
			if cfg.stop_source[source]: break

@threading_decorator			
def lineCycle(source,device,X0,Y0,Z0,T,cycle=1,dir='r',*args):
	# Spans the whole range [-1.0,1.0] starting from an arbitrary position in time T
	# to be used in source placement - dir='r' starts movement in r direction ('l' for left)
	# version for MEET - change only X coordinate
	if dir == 'r':
		c0 = 0
	else:
		c0 = 1
	nt = int(T/cfg.CLOCK)
	nt0 = int((X0+1)/2*nt)
	# Formula to correct for the delay in the Spat OSC messaging server (empirical!!!)
	wait = (T-0.0028*nt)/nt
	X = np.linspace(-X0,X0,nt)
	Y = np.linspace(Y0,Y0,nt)
	Z = np.linspace(Z0,Z0,nt)
	if dir == 'r':
		for i in range(nt0,nt):
			device.car(X[i*(-1)**c0],Y[i*(-1)**c0],Z[i*(-1)**c0],*args)
			time.sleep(wait)
			if cfg.stop_source[source]: break
	else:
		for i in range(nt0)[::-1]:
			device.car(X[i],Y[i],Z[i],*args)
			time.sleep(wait)
			if cfg.stop_source[source]: break
	if c0 == 1: cycle += 1
	for c in range((c0+1),cycle):
		for i in range(nt):
			device.car(X[i*(-1)**c],Y[i*(-1)**c],Z[i*(-1)**c],*args)
			time.sleep(wait)
			if cfg.stop_source[source]: break
		if cfg.stop_source[source]: break

@threading_decorator
def circles(device,aziA,aziB,radius,T):
	narc = np.abs(aziB-aziA)
	sign = np.sign(aziB-aziA)
	nt = int(T/cfg.CLOCK)
	ddeg = narc/nt
	d = aziA
	for i in range(nt+1):
		x = radius*np.cos(-i*np.pi/180+np.pi/2)
		y = radius*np.sin(-i*np.pi/180+np.pi/2)/2
		z = 0.0
		device.position([x,y,z],mode='set')
		d += sign*ddeg
		time.sleep (cfg.CLOCK)
		device.position([x,y,z],mode='set')
		
# SpatGris control source position/envelopes

@threading_decorator
def circlesCar(source,aziA,aziB,radius,T,*args):
	narc = np.abs(aziB-aziA)
	sign = np.sign(aziB-aziA)
	nt = int(T/cfg.CLOCK)
	ddeg = narc/nt
	d = aziA
	for i in range(nt+1):
		if cfg.stop_source[source]:
			break
		x = radius*np.cos(-d*np.pi/180+np.pi/2)
		y = radius*np.sin(-d*np.pi/180+np.pi/2)
		z = 0.0
		Spat(source).car(x,y,z,*args)
		d += sign*ddeg
		time.sleep (cfg.CLOCK)

@threading_decorator
def circlesDeg(source,aziA,aziB,T,*args):
	narc = np.abs(aziB-aziA)
	sign = np.sign(aziB-aziA)
	nt = int(T/cfg.CLOCK)
	ddeg = narc/nt
	d = aziA
	for i in range(nt+1):
		if cfg.stop_source[source]:
			break
		Spat(source).deg(d,0.0,1.0,*args)
		d += sign*ddeg
		time.sleep(cfg.CLOCK)
		