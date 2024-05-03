#
# msctools: my collection of composing and performing tools in python
#
# © 2023 Marco Buongiorno Nardelli
#

import time
import numpy as np
import networkx as nx
import pyo

from .networks import *
from .devices import Spat
from .osctools import client

import musicntwrk.msctools.cfg as cfg

def playerA(clips,clipsdur,track,delay=0.0,source=None,random=False,Y0=1.0,Z0=0.0,azi=0.0,ele=0.0,mode='network',external=None,nxmodel='barabasi_albert_graph',*args):
	''' 
	Play clips in sequence waiting for next clip in following mode
	mode = "network"    : sequence defined by the eulerian path on a network
						: network models can be found here: 
						: https://networkx.org/documentation/stable/reference/generators.html
						: arguments are passed through *args
	mode = "sequential" : plays the clips in descending order
	mode = "random"     : plays clips in random order
	mode = "external"   : plays clip with a user supplied sequence
	'''
	# delay the start of playback
	time.sleep(delay)
	while True:
		if cfg.stop[track]:
			break
		if mode == 'network':
			mynetx = getattr(nx,nxmodel)
			Gx = mynetx(*args)
			chino = chinese_postman(Gx,None,verbose=False)
			seq = [chino[0][0]]
			for s in range(1,len(chino)):
				seq.append(chino[s][1])
				if cfg.stop[track]:
					break
		elif mode == 'sequential':
			seq = np.linspace(0,len(clips[track])-1,len(clips[track]),dtype=int).tolist()
		elif mode == 'random':
			seq = np.linspace(0,len(clips[track])-1,len(clips[track]),dtype=int).tolist()
			np.random.shuffle(seq)
		elif mode == 'external':
			seq = external
		else:
			print('mode not implemented')
		for n in range(len(seq)):
			# set position of Spat source if needed
			if random:
				X = 2.0*np.random.rand() - 1.0
				Spat(source).car(X,Y0,Z0,azi,ele)
			client("/live/clip/fire",[track,seq[n]],cfg.HOST,cfg.PORT).send()
			time.sleep(np.abs(clipsdur[track][seq[n]]+np.random.rand()*cfg.sleep[track]))
			if cfg.stop[track]:
				break

def playerB(clips,track,delay=0.0,source=False,azi=0.0,ele=0.0,mode='network',external=None,nxmodel='barabasi_albert_graph',*args):
	''' 
	Play clips in sequence at set time intervals quantizaed according to tempo in bpm: 1/n.
	mode = "network"    : sequence defined by the eulerian path on a network
						: network models can be found here: 
						: https://networkx.org/documentation/stable/reference/generators.html
						: arguments are passed through *args
	mode = "sequential" : plays the clips in descending order
	mode = "random"     : plays clips in random order
	mode = "external"   : plays clip with a user supplied sequence
	'''
	# delay the start of playback
	time.sleep(delay)
	while True:
		if cfg.stop[track]:
			break
		if mode == 'network':
			mynetx = getattr(nx,nxmodel)
			Gx = mynetx(*args)
			chino = chinese_postman(Gx,None,verbose=False)
			seq = [chino[0][0]]
			for s in range(1,len(chino)):
				seq.append(chino[s][1])
				if cfg.stop[track]:
					break
		elif mode == 'sequential':
			seq = np.linspace(0,len(clips[track])-1,len(clips[track]),dtype=int).tolist()
		elif mode == 'random':
			seq = np.linspace(0,len(clips[track])-1,len(clips[track]),dtype=int).tolist()
			np.random.shuffle(seq)
		elif mode == 'external':
			seq = external
		else:
			print('mode not implemented')
		for n in range(len(seq)):
			# set position of Spat source if needed
			if source:
				X = 2.0*np.random.rand() - 1.0
				Spat(track+1).car(X,1.0,0.0,azi,ele)
			client("/live/clip/fire",[track,seq[n]],cfg.HOST,cfg.PORT).send()
			tsleep = np.max([np.abs(60.0/cfg.tempo*cfg.beat[track]+np.random.rand()*cfg.sleep[track]),
							 np.abs(clips[track][seq[n]].dur()+np.random.rand()*cfg.sleep[track])])
			time.sleep(tsleep)
			if cfg.stop[track]:
				break
			
def scorePlayer(clips,track,score,delay=0,hold=False):
	''' 
	Play clips in sequence according to a score read as musicxml (pitch + duration)
	score[0] = pitches
	score[1] = durations in units of quantization (usually 1/4 if not specified otherwise)
	'''
	# delay the start of playback - if for any reason is desired
	time.sleep(delay)
	while True:
		if cfg.stop[track]:
			break
		seq = score[0]
		dur = score[1]
		for n in range(len(seq)):
			client("/live/clip/fire",[track,seq[n]],port=11000).send()
			if hold:
				tsleep = np.max([np.abs(60.0/cfg.tempo*cfg.beat[track]*dur[n]+(2*np.random.rand()-1)*cfg.sleep[track]),
								np.abs(clips[track][seq[n]].dur()+np.random.rand()*cfg.sleep[track])])
				time.sleep(tsleep)
			else:
				tsleep = np.abs(60.0/cfg.tempo*cfg.beat[track]*dur[n]+(2*np.random.rand()-1)*cfg.sleep[track])
				time.sleep(tsleep)
			if cfg.stop[track]:
				break


def playScene(scene,session,delay=0):
	# play a scene from the session view
	time.sleep(delay)
	for n in range(session.num_tracks()):
		client("/live/clip_slot/fire",[n,scene],cfg.HOST,cfg.PORT).send()