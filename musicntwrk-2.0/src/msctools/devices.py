#
# msctools: my collection of composing and performing tools in python
#
# © 2023 Marco Buongiorno Nardelli
#

import time
import numpy as np
from scipy.io import wavfile

from .osctools import client
from .converters import *
import musicntwrk.msctools.cfg as cfg

class Dolby:
	
	def __init__(self,track,device,controls=None,host="127.0.0.1",port=11000):
		self.n = track
		self.d = device
		self.cntr = controls
		self.port = port
		self.host = host
		client("/live/device/get/name",[self.n,self.d],self.host,self.port).send()
		time.sleep(cfg.TICK)
		assert cfg.data[0] == 'Dolby Atmos Music Panner', 'wrong device!'
	
	'''
	Range is [0-1] for all parameters
	
	0 Device On
	1 Pan X - X coordinate of source
	2 Pan Y - Y coordinate of source
	3 Pan Z - Z coordinate of source
	4 Object Size
	5 Elevation Enable
	6 Elevation Mode
	7 Sequencer Enable
	8 Step Duration
	9 Step 1 Enable
	10 Step 2 Enable
	11 Step 3 Enable
	12 Step 4 Enable
	13 Step 5 Enable
	14 Step 6 Enable
	15 Step 7 Enable
	16 Step 8 Enable
	17 Step 9 Enable
	18 Step 10 Enable
	19 Step 11 Enable
	20 Step 12 Enable
	21 Step 13 Enable
	22 Step 14 Enable
	23 Step 15 Enable
	24 Step 16 Enable
	25 Channel Link Mode - Set to 0 to copy L+R on single source
	26 Pan X R
	27 Pan Y R
	28 Pan Z R
	29 Object Size R
	30 Channel Link Enabled
	31 Slow Step Duration Enable
	'''
	
	def source(self,mode='Copy'):
		if mode == 'Copy':
			client("/live/device/set/parameter/value",[self.n,self.d,25,0],self.host,self.port).send()
		if mode == 'MirrorX':
			client("/live/device/set/parameter/value",[self.n,self.d,25,1/3],self.host,self.port).send()
		if mode == 'MirrorY':
			client("/live/device/set/parameter/value",[self.n,self.d,25,2/3],self.host,self.port).send()
		if mode == 'MirrorXY':
			client("/live/device/set/parameter/value",[self.n,self.d,25,1],self.host,self.port).send()
			
	def size(self,val=None,mode='get'):
		if mode == 'get':
			client("/live/device/get/parameter/value",[self.n,self.d,4],self.host,self.port).send()
			time.sleep(cfg.TICK)
			return(cfg.data[0])
		if mode == 'set':
			client("/live/device/set/parameter/value",[self.n,self.d,4,val],self.host,self.port).send()
			time.sleep(cfg.TICK)
	
	def position(self,pos=[None,None,None],mode='get'):
		if mode == 'get':
			client("/live/device/get/parameter/value",[self.n,self.d,1],self.host,self.port).send()
			time.sleep(cfg.TICK)
			X = cfg.data[0]
			client("/live/device/get/parameter/value",[self.n,self.d,2],self.host,self.port).send()
			time.sleep(cfg.TICK)
			Y = cfg.data[0]
			client("/live/device/get/parameter/value",[self.n,self.d,3],self.host,self.port).send()
			time.sleep(cfg.TICK)
			Z = cfg.data[0]
			return(X,Y,Z)
		if mode == 'set':
			X = pos[0]
			Y = pos[1]
			Z = pos[2]
			client("/live/device/set/parameter/value",[self.n,self.d,1,X],self.host,self.port).send()
			client("/live/device/set/parameter/value",[self.n,self.d,2,Y],self.host,self.port).send()
			client("/live/device/set/parameter/value",[self.n,self.d,3,Z],self.host,self.port).send()
			
	def param(self,n,val=None,mode='name'):
		if mode == 'name':
			client("/live/device/get/parameter/name",[self.n,self.d,n],self.host,self.port).send()
			time.sleep(cfg.TICK)
			return(cfg.data[0])
		if mode == 'get':
			client("/live/device/get/parameter/value",[self.n,self.d,n],self.host,self.port).send()
			time.sleep(cfg.TICK)
			return(cfg.data[0])
		if mode == 'set':
			client("/live/device/set/parameter/value",[self.n,self.d,n,val],self.host,self.port).send()
			time.sleep(cfg.TICK)
			
			
class SpatControl:
	
	def __init__(self,track,device,controls=None,host="127.0.0.1",port=11000):
		self.n = track
		self.d = device
		self.cntr = controls
		self.port = port
		self.host = host
		client("/live/device/get/name",[self.n,self.d],self.host,self.port).send()
		time.sleep(cfg.TICK)
		assert cfg.data[0] == 'ControlGris', 'wrong device!'
		
	'''
	Range is [0-1] for all parameters
	
	0 Device On
	1 Recording Trajectory X - X coordinate of source
	2 Recording Trajectory Y - Y coordinate of source
	3 Recording Trajectory Z - Z coordinate of source
	4 Source Link
	5 Source Link Alt
	6 Position Preset
	7 Azimuth Span - horizontal span (size)
	8 Elevation Span - vertical span (size)
	9 Elevation Mode
	'''
		
	def azispan(self,val=None,mode='get'):
		if mode == 'get':
			client("/live/device/get/parameter/value",[self.n,self.d,7],self.host,self.port).send()
			time.sleep(cfg.TICK)
			return(cfg.data[0])
		if mode == 'set':
			client("/live/device/set/parameter/value",[self.n,self.d,7,val],self.host,self.port).send()
			time.sleep(cfg.TICK)

	def elespan(self,val=None,mode='get'):
		if mode == 'get':
			client("/live/device/get/parameter/value",[self.n,self.d,8],self.host,self.port).send()
			time.sleep(cfg.TICK)
			return(cfg.data[0])
		if mode == 'set':
			client("/live/device/set/parameter/value",[self.n,self.d,8,val],self.host,self.port).send()
			time.sleep(cfg.TICK)
			
	def position(self,pos=[None,None,None],mode='get'):
		if mode == 'get':
			client("/live/device/get/parameter/value",[self.n,self.d,1],self.host,self.port).send()
			time.sleep(cfg.TICK)
			X = cfg.data[0]
			client("/live/device/get/parameter/value",[self.n,self.d,2],self.host,self.port).send()
			time.sleep(cfg.TICK)
			Y = cfg.data[0]
			client("/live/device/get/parameter/value",[self.n,self.d,3],self.host,self.port).send()
			time.sleep(cfg.TICK)
			Z = cfg.data[0]
			return(X,Y,Z)
		if mode == 'set':
			X = pos[0]
			Y = pos[1]
			Z = pos[2]
			client("/live/device/set/parameter/value",[self.n,self.d,1,X],self.host,self.port).send()
			client("/live/device/set/parameter/value",[self.n,self.d,2,Y],self.host,self.port).send()
			client("/live/device/set/parameter/value",[self.n,self.d,3,Z],self.host,self.port).send()
			
	def param(self,n,val=None,mode='name'):
		if mode == 'name':
			client("/live/device/get/parameter/name",[self.n,self.d,n],self.host,self.port).send()
			time.sleep(cfg.TICK)
			return(cfg.data[0])
		if mode == 'get':
			client("/live/device/get/parameter/value",[self.n,self.d,n],self.host,self.port).send()
			time.sleep(cfg.TICK)
			return(cfg.data[0])
		if mode == 'set':
			client("/live/device/set/parameter/value",[self.n,self.d,n,val],self.host,self.port).send()
			time.sleep(cfg.TICK)
			
class Spat:

	def __init__(self,source,host="127.0.0.1",port=18032):
		self.s = source
		self.port = port
		self.host = host
	'''
	Direct messaging to SpatGris
	'''
	
	def deg(self,azi,ele,radius,azispan,elespan):
		# set position using polar coordinates in degrees
		client("/spat/serv",['deg',self.s,azi,ele,radius,azispan,elespan],self.host,self.port).send()
	
	def pol(self,azi,ele,radius,azispan,elespan):
		# set position using polar coordinates in radians
		client("/spat/serv",['pol',self.s,azi,ele,radius,azispan,elespan],self.host,self.port).send()
	
	def car(self,x,y,z,azispan,elespan):
		# set position using cartesian coordinates
		client("/spat/serv",['car',self.s,x,y,z,azispan,elespan],self.host,self.port).send()
		
	def clr(self):
		client("/spat/serv",['clr',self.s],self.host,self.port).send()