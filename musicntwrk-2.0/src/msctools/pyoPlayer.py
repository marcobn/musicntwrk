#
# msctools: my collection of composing and performing tools in python
#
# © 2023 Marco Buongiorno Nardelli
#

import time
import numpy as np
import pyo

from .decorators import threading_decorator

import musicntwrk.msctools.cfg as cfg

class pyoClip(pyoObject):
	
	# helper class to play a clip

	def __init__(self, input, speed=1, loop=False, offset=0, interp=2, mul=1, add=0):
		
		PyoObject.__init__(self)

		# Keep references of all raw arguments
		self.input = input
		self.speed = speed
		self.loop = loop
		self.offset = offset
		self.interp = interp
		self.mul = mul
		self.add = add

		# Using InputFader to manage input sound allows cross-fade when changing sources
	    self.in_fader = pyo.InputFader(input)

	    # Convert all arguments to lists for "multi-channel expansion"
	    in_fader, speed, loop, offset, interp, mul, add = convertArgsToLists(self._in_fader, speed, loop, offset, interp, mul, add)

	# Set methods and attributes
	def setInput(self, x, fadetime=0.05):
	    """
	    Replace the `input` attribute.

	    :Args:

	        x : PyoObject
	            New signal to process.
	        fadetime : float, optional
	            Crossfade time between old and new input. Defaults to 0.05.

	    """
	    self.input = x
	    self.in_fader.setInput(x, fadetime)

	def setSpeed(self, x):
		# set speed
		self.speed = x

	def setLoop(self,x):
		# set loop (True/False)
		self.loop = x

	def setOffset(self, x):
		# set offset
		self.offset = x

	def setInterp(self, x):
		# set offset
		self.interp = x

	def setMul(self, x):
		# set offset
		self.mul = x

	def setAdd(self, x):
		# set offset
		self.add = x


	










