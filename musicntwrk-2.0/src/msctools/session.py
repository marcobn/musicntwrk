#
# msctools: my collection of composing and performing tools in python
#
# © 2023 Marco Buongiorno Nardelli
#

# container functions

import time

from .base import *
import musicntwrk.msctools.cfg as cfg

def trackList(session):
	num_tracks = session.num_tracks()
	tracks = []
	for t in range(num_tracks):
		tracks.append(Track(t))
	time.sleep(cfg.TICK)
	return(tracks)

def clipList(session,tracks):
	num_tracks = session.num_tracks()
	num_clips = []
	for n in range(num_tracks):
		nclips = 0
		for i in range(tracks[0].nclips()):
		    try:
		        if Clip(n,i).name(mode='get') != None:
		            nclips += 1
		    except:
		        pass
		num_clips.append(nclips)
	clips = []
	for n in range(num_tracks):
		clp = []
		for i in range(num_clips[n]):
			clp.append(Clip(n,i))
		clips.append(clp)
	time.sleep(cfg.TICK)
	return(clips)

def deviceList(session,tracks):
	num_tracks = session.num_tracks()
	num_devices = []
	for n in range(num_tracks):
		num_devices.append(tracks[n].ndevices())
	devices = []
	for n in range(num_tracks):
		dvc = []
		for i in range(num_devices[n]):
			dvc.append(Device(n,i))
		devices.append(dvc)
	time.sleep(cfg.TICK)
	return(devices)

def setSession():
	session = Song()
	tracks = trackList(session)
	clips = clipList(session,tracks)
	devices = deviceList(session,tracks)
	time.sleep(cfg.TICK)
	return(session,tracks,devices,clips)