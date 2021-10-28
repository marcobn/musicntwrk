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
# The following code has been extracted from the original program written by Dmitri Tymoczko, www.madmusicalscience.com

"""
Draw an abstract picture of chord space, emphsizing its circular dimension and the line containing n-note chords, which winds around this 
dimension n times.

Plots points showing how the chords in a scale are distributed on this abstract picture; clicking on this picture will play the chords

Usage:

	c = chordspace.ChordSpace(startNotes = [60, 64, 67], startScale = [0, 2, 4, 5, 7, 9, 11])

	startNotes
		if list: starting midi notes for playback
		if int: the size of the chord, the number of times the transpositions wind around chord space
	startScale
		the scale to use.

	There are many defaults built into the program; so you can just call chordspace.go(3, 12) (etc.).  It will select appropriately.

Many graphic parameters you can fiddle with, such as:

	gapAngle = a graphical parameter determining what proportion of the circle is taken up by the twist (.2 = 20%).  Adjust to taste.
	circleSeparation = how far apart the circles are drawn

Can draw arrows representing my chord lattices

Also:
	chordspace.line() draws a very simple (noninteractive) picture of the situation in pitch space, with no octave equivalence ... useful just as a point of contrast

"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
from threading import Timer
from music21 import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

for s in ['keymap.all_axes', 'keymap.back', 'keymap.forward', 'keymap.fullscreen', 'keymap.grid', 'keymap.home', 
			'keymap.pan', 'keymap.quit', 'keymap.save', 'keymap.xscale', 'keymap.yscale', 'keymap.zoom']:
	plt.rcParams[s] = ''

def linear_map(value, firstRange, secondRange):
	pct = 1.0 * (value - firstRange[0])/(firstRange[1] - firstRange[0])
	output = secondRange[0] + pct*(secondRange[1] - secondRange[0])
	return output

def euclidean_distance(l1, l2):
	return pow(sum([(l2[i] - l1[i])**2 for i in range(len(l1))]), .5)

def line(spacing = 1.7, offset = -.2):	
	"""Just draw a silly line with points on it, to contrast with the circular case"""
	plt.figure(figsize=(8,8))
	plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
	fig, ax = plt.subplots()
	plt.ioff()
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(False)
	ax.grid(False)
	ax.set_xlim([-8, 8])
	ax.set_ylim([-8, 8])
	yCoords = np.arange(-10 + offset, 10 + offset, spacing)
	xCoords = [0]*len(yCoords)
	ax.plot([-10, 10], [0, 0], color = 'black')
	ax.scatter(yCoords, xCoords, s = 50, c = 'black')
	ax.set_yticks([])  # no ticks
	ax.set_xticks([])
	plt.show()

class ChordSpace():
	
	def __init__(self, startNotes = 2, startScale = 7, **kwargs):
		
		"""PRESET: (6, 11): {3:[.1, -.15], 5: [-.1, 0], 7: [-.1, 0]},"""
	
		"""Arrow Presets 1 is the first two voice leadings on the generalized circle of fifths; preset 2 is the first square"""
		
		self.attributeDict = {	"labels": True, 
								"arrows": [], 
								"bigAngleDict": {},
								"gapAngle": .2, 
								"circleSeparation": 1, 
								"pointRotation": 0, 
								"labelAdjustments": False,
								"_CIRCLESIZE": 50,
								"_CLICKRADIUS": .5,
								"masterLetters": ['Fn', 'Cn', 'Gn', 'Dn', 'An', 'En', 'Bn', 'Fs', 'Cs', 'Gs', 'Ef', 'Bf'],
								"standardRList": [[0, 5, 50], [5, 7, 50]],
								"medWrapAround": [[0, 2, 100], [2, 10, 100]],
								"bigWrapAround": [[0, 1, 150], [1, 3, 100], [3, 6, 100]],
								"arrowPresets": {1: [[0, 1, .9], [1, 2, .9]], 2: [[0, 1, .9, None, 0, 'r'], [1, 2, .9], [1, 2, .9, 0], ['lastDest', 2, .9, None, 0, 'r']]},
								"globalWinding": 0,
								"shouldQuit": False,
								"keysPressed": [],
								"theCoords": False,
								"startNotes": None,
								"standardAdjustments": {
														(2, 12): {2:[-.1, 0]},
														(3, 12): {9: [-.1, 0], 11: [.1, 0]},
														(3, 7): {5: [-.1, 0]},
														(4, 10): {0: [0, -.1], 1: [-.03, .03], 3: [-.15, 0], 5: [0, -.1], 9: [.05, .1]},
														(4, 12): {0: [0, -.1], 3: [0, -.1], 6: [0, -.1], 9: [0, -.1], 1: [.1, .3], 4: [.1, .3], 7: [.1, .3], 10: [.1, .3],
																2: [0, .3], 5: [-.1, .3], 8: [-.1, .3], 11: [-.1, .3]},
														(5, 12): {0:[0, -.15], 1: [0, .3], 2: [.3, 0], 3: [-.1, 0], 4: [.15, -.08], 5: [-.05, -.1], 6: [0, .1], 7: [.05, -.1], 8: [-.1, 0], 9:[.1, 0], 10:[-.1, 0], 11:[0, .1]},
														(6, 11): {0:[0, 0], 10: [0, 0], 9: [0, 0], 8: [0, 0], 7: [-.15, 0], 6: [0, 0], 5: [-.1, 0], 4:[0, 0], 3:[-.1, 0], 2: [0, 0], 1:[0, 0]},
														(6, 12): {0:[0, -.5], 10: [0, -.5], 9: [0, .5], 8: [0, -.5], 7: [0, .5], 6: [0, -.5], 5: [0, .5], 4:[0, -.5], 3:[0, .5], 2: [0, -.5], 1:[0, .5], 11: [0, .5]},
														(7, 12): {0:[0, -.15], 1: [.1, .1], 2: [.05, -.05], 3: [-.1, 0], 4:[.2, .3], 5: [.2, .1], 6: [0, 0], 7: [-.1, 0.], 8: [-.1, 0], 9: [0, -.2], 10: [-.1, 0], 11: [.2, .1]},
																
														},
								"defaultPitches": {(1, 12): [[60], []], (2, 7): [[72, 76], []], (2, 11): [[72, 76], []], (2, 12): [[72, 76], []], (3, 7): [[60, 64, 67], []], (3, 12): [[60, 64, 67], []], 
													(4, 7): [[60, 64, 67, 71], []], (4, 12): [[60, 64, 67, 70], []], (4, 10): [[60, 63, 66, 68], [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]], (6, 11): [[60, 62, 64, 65, 67, 69],[]]
													},
								"internalPresets": {(3, 7): {"clickRadius": .9}, (7, 12): {"arpeggiate": .015, "clickRadius": .9, "rAdjustment": .6, "gapAngle": .23}, (6, 11): {"arpeggiate":.2, "allNotesOff":False, "rAdjustment": .6}},
								"preset": 0,
								"presetDict": {	1: {"startNotes": [43, 50, 57, 64, 66, 72, 77], "theScale": range(12), "rAdjustment": .6, "gapAngle": .23}, 2: {"startNotes": [67, 68, 69, 70, 71], "theScale": range(12), "voiceVelocities": [80, 54, 80, 54, 80]},
												3: {"startNotes": [60, 74], "theScale": [0, 2, 4, 5, 7, 9, 11], "baseVL": [2, 0]}, 4: {'startNotes': [50, 67, 69, 72], "theScale": [0, 2, 4, 5, 7, 9, 11], "voiceVelocities": [90, 72, 72, 72]},
												5: {'startNotes': [48, 55, 62, 65, 69, 76], "theScale": range(11)},
												10: {"startNotes": [60, 64, 67], "theScale": range(12), "alphaLabels": {x:.3 for x in [1, 11, 4, 8, 6, 3]}},
												11: {"startNotes": [60, 64, 67], "theScale": range(12), "alphaLabels": {x:.3 for x in [1, 11, 4, 8, 6, 3]}}},
								"graphs": [[[48, 85, 62, 75, 64], range(12)], [[60, 67, 76], 7], [60, 65, 67, 71], 7],
								"mainChordGraph": False,
								"midiin": False,
								"midiout": False,
								"bigPoint": False,
								"lastPoint": False,
								"lastIndex": 0,
								"lastChordNumber": 0,
								"chordIndex": 0,
								"arpeggiate": False, 
								"defaultScales": {5: [0, 2, 4, 7, 9], 7: [0, 2, 4, 5, 7, 9, 11], 10: [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]},
								"lastClick": 0,
								"diatonicObjectKeywords": {},
								"allNotesOff": True,
								"theScale": False,
								"activeNotes": [], 
								"notesPlayed": 0,
								"slices": 15,
								"sliceDelay": .001,
								"rAdjustment": .4,
								"recordTimer": None,
								"outputStream": stream.Stream(),
								"minimumRecordingDuration": .2,
								"deltaList": [],
								"_GLOBALREFLECTION": False,
								"lowercase": False,
								"suppressLabels": False,
								"quarterTone": False,
								"radius": 4,
								"alphaLabels": {},
								"innerRadius": 0,
								"outerRadius": 0,
								"isTsymmetrical": False,
								"drawPoints": True,
								"drawPie": False,
								"saveName": False, 
								"drawTarget": False,
								"baseAlpha": 1.,
								"drawMatrix": False,
								"matrixChords": False,
								"originalScaleSize": False,
								"graphCommas": [],
								"manualLabels": []
		 				   		}										
																	
		for key, value in self.attributeDict.items():
			if key in kwargs:
				setattr(self, key, kwargs[key])
			else:
				setattr(self, key, value)
		
		"""Lots of boring code to deal with various ways you can initialize the scale and chord: using midi notes, integers, etc."""
		
		if self.preset in self.presetDict:
			for key, value in self.presetDict[self.preset].items():
				setattr(self, key, value)
			self.chordSize = len(self.startNotes)
			# print(self.chordSize)
			self.scaleSize = len(self.theScale)
		else:		
			if type(startNotes) is int and type(startScale) is int:
				if (startNotes, startScale) in self.defaultPitches:
					self.startNotes, self.tempScale = self.defaultPitches[(startNotes, startScale)]
					if self.tempScale:
						self.theScale = self.tempScale[:]
						self.scaleSize = len(self.theScale)
						startScale = None
					self.chordSize = startNotes
					
				if (startNotes, startScale) in self.internalPresets:
					presetDict = self.internalPresets[(startNotes, startScale)]
					for key, value in presetDict.items():
						if key not in kwargs:
							setattr(self, key, value)
				if self.startNotes:
					startNotes = None
			if type(startScale) is int:
				self.scaleSize = startScale
				if startScale in self.defaultScales:
					self.theScale = self.defaultScales[startScale]
				elif startScale:
					self.theScale = range(startScale)
				startScale = None
			elif type(startScale) is list:
				self.theScale = startScale
				self.scaleSize = len(self.theScale)
			if type(startNotes) is int:
				self.chordSize = startNotes
				if self.theScale:
					self.startNotes = [60 + self.theScale[x] for x in self.get_maximally_even()]
			elif type(startNotes) is list:
				self.chordSize = len(startNotes)
				if max(startNotes) < 12:
					self.startNotes = [60 + self.theScale[x] for x in startNotes]
				else:
					self.startNotes = startNotes
		
		if (self.scaleSize) and (11 < max(self.theScale) < 24 or not (all([x == int(x) for x in self.theScale])) or not (all([x == int(x) for x in self.startNotes]))):
			self.quarterTone = True
		
		if self.preset == 11:
			self.masterLetters = ['VII', 'I', 'fII', 'II', 'fIII', 'III', 'IV', 'fV', 'V', 'fVI', 'VI', 'fVII']
		
		"""manage diatonic object keywords, which get passed to underlying diatonic object"""
		for s in ['baseVL']:
			if s in kwargs:
				self.diatonicObjectKeywords[s] = kwargs[s]
			elif hasattr(self, s):
				self.diatonicObjectKeywords[s] = getattr(self, s)
		
		if 'voiceVelocities' in kwargs:
			self.voiceVelocities = kwargs['voiceVelocities']
		elif not hasattr(self, 'voiceVelocities'):
			self.voiceVelocities = [72] * self.chordSize
		
		"""Deal with transpositional symmetry here"""
			
		self.setClass = [self.theScale.index(x % 12) if (x % 12) in self.theScale else -1 for x in self.startNotes]
		self.originalScaleSize = self.scaleSize
		
		"""GAPANGLE is the portion of the circle where the lines wind into each other"""
		
		if self.chordSize == 1:
			self._GAPANGLE = 2 * np.pi
		else:
			self._GAPANGLE = (1. - self.gapAngle) * 2 * np.pi

		if type(self.arrows) is int or type(self.arrows) is str:
			self.arrows = self.arrowPresets[self.arrows]
		
		"""Pyplot initialization code"""	
		if not self.drawTarget:
			sizeFactor = 10./8
			if self.drawMatrix:
				self.fig = plt.figure(figsize=(sizeFactor * 8, 8))
			else:
				self.fig = plt.figure(figsize=(8,8))
			plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
			self.ax = plt.subplot()
		
			self.ax.axes.get_xaxis().set_visible(False)
			self.ax.axes.get_yaxis().set_visible(False)
			self.ax.grid(False)
		
			self.bound = self.radius + (self.chordSize * self.circleSeparation) + 1
			
			if self.drawMatrix:
				newSize = 2 * self.bound * sizeFactor
				self.ax.set_xlim([-self.bound, newSize - self.bound])
				self.matrixLocation = newSize - self.bound
				self.make_matrix()
			else:
				self.ax.set_xlim([-self.bound, self.bound])
			self.ax.set_ylim([-self.bound, self.bound])
		else:
			self.fig = self.drawTarget.fig			# for drawing multiple graphs on one canvas
			self.ax = self.drawTarget.ax
		
		"""draw the circles without the connectors, leaving out an arc for connections"""
		for i in range(int(self.chordSize)):
			theCircle = ChordSpace.draw_circle(r = self.radius + (self.circleSeparation*i), thetaMax = self._GAPANGLE)
			self.ax.plot(*theCircle, color = 'black', alpha = self.baseAlpha)
		
		"""draw the internal connectors"""
		for i in range(int(self.chordSize) - 1):
			self.newAngles, self.newRs = ChordSpace.interpolate_between_circles(self._GAPANGLE, self.radius + (self.circleSeparation*i),
					 2*np.pi, self.radius + (self.circleSeparation*(i + 1)), self.standardRList)

			self.newAngles = [(2*np.pi * (self.chordSize - 1 - i)) + x for x in self.newAngles]
			self.bigAngleDict.update({self.newAngles[i]:self.newRs[i] for i in range(len(self.newAngles))})
			connection = ChordSpace.poltocar(self.newAngles, self.newRs)
			self.ax.plot(*connection, color = 'black', alpha = self.baseAlpha)
		
		"""draw the wraparound connector"""
		if self.chordSize > 1:
			if self.chordSize >= 5:
				wrapInts = self.bigWrapAround
			else:
				wrapInts = self.medWrapAround
			self.newAngles, self.newRs = ChordSpace.interpolate_between_circles(self._GAPANGLE, self.radius, 2*np.pi, self.radius + (self.circleSeparation*(self.chordSize-1)), wrapInts)
			self.newRs = self.newRs[::-1]	
			connection = ChordSpace.poltocar(self.newAngles, self.newRs)
			self.bigAngleDict.update({self.newAngles[i]:self.newRs[i] for i in range(len(self.newAngles))})
			self.ax.plot(*connection, color = 'black', alpha = self.baseAlpha)
		
		"""boundaries to represent the annulus"""
		if self.innerRadius:
			self.innerCircle = plt.Circle((0, 0), self.innerRadius, color='black', fill=False, linestyle = '--', alpha = .5)
			self.ax.add_artist(self.innerCircle)
		
		if self.outerRadius:
			self.outerCircle = plt.Circle((0, 0), self.outerRadius, color='black', fill=False, linestyle = '--', alpha = .5)
			self.ax.add_artist(self.outerCircle)
			
		if self.scaleSize and self.drawPie:
			r = self.radius + (self.circleSeparation*self.chordSize)
			a = (2*np.pi)/self.scaleSize
			for i in range(self.scaleSize):
				newSlice = ChordSpace.poltocar([0, i*a], [0, r])
				self.ax.plot(*newSlice, color = 'black', alpha = .5)
		
		"""you don't have to specify a scale, but if you do, the program will draw chords
		  	 
			 bigAngleDict[theta] = radius	
					shows the radius at each theta position; needed to draw the circles on the wraparound spot; theta can be greater than 2*pi (when winding multiple times)
			"""
		if self.scaleSize and self.drawPoints:
			
			self.theKeys = sorted(self.bigAngleDict.keys())
			self.totalDist = 2 * np.pi * self.chordSize
			self.theAngles = np.linspace(0 + self.pointRotation, self.totalDist + self.pointRotation, self.scaleSize + 1)
			self.basicDist = self.theAngles[1]
			self.theAngles = self.theAngles[:-1]
			self.theRs = []
			
			if self.graphCommas:					# for non-ET graphs
				self.theAngles = [self.theAngles[x] + self.graphCommas[x] for x in range(len(self.theAngles))]	 
				
			for a in self.theAngles:
				self.theRs.append(self.getRvalue(a))
			
			self.theCoords = ChordSpace.poltocar(self.theAngles, self.theRs)
			self.ax.scatter(*self.theCoords, s = self._CIRCLESIZE, c = 'black', alpha = self.baseAlpha)
			
			"""calculate the clickable radius of each circle as half the minimum distance between adjacent circles"""
			
			minDist = 10000
			
			for i in range(len(self.theCoords[0]) - 1):
				for j in range(i+1, len(self.theCoords[0])):
					d = euclidean_distance([self.theCoords[0][i], self.theCoords[1][i]], [self.theCoords[0][j], self.theCoords[1][j]])
					if d < minDist:
						minDist = d
			
			self._CLICKRADIUS = minDist/2.
			
			if self.drawTarget:
				self.drawTarget._CLICKRADIUS = 10
			
			"""thePoints
			
				a list whose elements are [angle % 2pi, angleWithWrapAround, radius]
				ordered chromatically (C, C#, D etc.)
			
			"""
			
			self.thePoints = [[round(x[0] % (2 * np.pi), 4)] + list(x) for x in zip(self.theAngles, self.theRs)]
			self.orderedPoints = sorted(self.thePoints)
			
			"""todo: need to account for pointRotation here in the sorting!"""
			
			self.simpleAngles = sorted(list(set([x[0] for x in self.thePoints])))
			self.fundamentalAngle = self.simpleAngles[1]
			
			self.zeroPoints = [x[1:] for x in self.thePoints if x[0] == self.thePoints[0][0]]
			self.zeroPointNumber = len(self.zeroPoints)
			self.crossSections = int(self.scaleSize/self.zeroPointNumber)
			
			"""
		
			The two cases of lattice:
			
				0 = size of chord relatively prime to size of scale (single generator)
				1 = size of chord divides or shares a common factor with the size of scale (two generators)
			
			For moving around the lattice, we require extra code to manage the second generator
			
			"""
			
			self.latticeStructure = int(not(self.zeroPointNumber == 1))
			
			"""simple code to attach letters to circles;
			
				standardAdjustments manually arranges the letters so as to avoid collisions.
			
			"""
			if self.labels and not self.suppressLabels:
				if self.scaleSize > 12:
					self.letterNames = [str(x + 1) for x in range(self.scaleSize)]
				else:
					if self.manualLabels:
						self.letterNames = self.manualLabels[:]
					elif self.isTsymmetrical:
						self.letterNames = sorted(self.masterLetters)
						self.letterNames = self.letterNames[self.letterNames.index('Cn'):] + self.letterNames[:self.letterNames.index('Cn')]
						self.letterNames = [x.replace('n', '') for x in self.letterNames[:self.scaleSize]]
					elif self.scaleSize == 5:
						self.letterNames = [x.replace('n', '') for x in sorted(self.masterLetters[1:6])]						
					elif self.preset != 11:
						self.letterNames = [x.replace('n', '') for x in sorted(self.masterLetters[:self.scaleSize])]
						self.letterNames = self.letterNames[self.letterNames.index('C'):] + self.letterNames[:self.letterNames.index('C')]
						if self.scaleSize == 11:
							self.letterNames[-1] = 'Bf'
					else:
						self.letterNames = self.masterLetters[self.masterLetters.index('I'):] + self.masterLetters[:self.masterLetters.index('I')]	
					
				if self.preset == 11:
					adjustmentTable = {0: [.04, 0], 1: [-.25, 0], 2: [-.1, 0], 3: [-.025, 0], 4: [-.1, -.05], 5: [-.05, 0], 6: [0, 0], 7: [-.05, 0], 8: [-.07, -.05], 9: [-.07, 0], 11: [.15, 0]}
				elif (self.chordSize, self.scaleSize) in self.standardAdjustments:
					adjustmentTable = self.standardAdjustments[(self.chordSize, self.scaleSize)]
#					print('here')
				else:
					adjustmentTable = {}
				
				self.letterNames = [self.letterNames[0]] + self.letterNames[len(self.letterNames):0:-1]
				
				if self.lowercase:
					self.letterNames = [x.lower() for x in self.letterNames]
					
				for i in range(self.scaleSize):
					myTheta, myR = self.thePoints[i][1:]
					myR = myR - self.rAdjustment
					myCoord = ChordSpace.poltocar([myTheta], [myR])	# x, y?
					myAdjustment = adjustmentTable.get(i, [0, 0])
					theText = self.letterNames[i]
					if theText.count('I') > 0 or theText.count('V') > 0:
						plainText = filter(lambda x: x in 'IViv', theText)
						theText = filter(lambda x: x not in 'IViv', theText)
						self.ax.text(myCoord[0][0] - .1 + myAdjustment[0], myCoord[1][0] + myAdjustment[1], plainText, alpha = self.alphaLabels.get(i, 1.), verticalalignment = 'center', horizontalalignment = 'left', fontname = 'Helvetica')
						self.ax.text(myCoord[0][0] - .3 + myAdjustment[0], myCoord[1][0] + .04 + myAdjustment[1], theText, alpha = self.alphaLabels.get(i, 1.), verticalalignment = 'center', horizontalalignment = 'left', fontname = 'EngraverTextT')
					else:
						self.ax.text(myCoord[0][0] + myAdjustment[0], myCoord[1][0] + myAdjustment[1], theText, alpha = self.alphaLabels.get(i, 1.), verticalalignment = 'center', horizontalalignment = 'center', fontname = 'EngraverTextT')
			
			"""Print generators"""
			if len(self.zeroPoints) > 1:
				self.theTransps = [[int(.001 + x[0] / (2. * np.pi)), int(.001 + x[0] / self.basicDist)] for x in self.zeroPoints[1:]]
				print("Zero sum generator", -self.theTransps[0][0], self.theTransps[0][1])
		
			if len(self.simpleAngles) > 1 and self.chordSize > len(self.zeroPoints):
				myPoints = []
				for p in self.thePoints:
					if abs(p[0] - self.fundamentalAngle) < .001:
						myPoints.append(p[1:])
				self.theTransps = [[int(.001 + x[0] / (2. * np.pi)), int(.001 + x[0] / self.basicDist)] for x in myPoints]
				print("Sum-1 generator", -self.theTransps[0][0], self.theTransps[0][1])
			
			"""add arrows"""
			if self.arrows:
				
				self.cartesianPoints = [ChordSpace.poltocar([x[1]], [x[2]]) for x in self.orderedPoints]
				self.cartesianPoints = [[x[0][0], x[1][0]] for x in self.cartesianPoints]
				
				for arrowItem in self.arrows:			# startpoint, endpoint, %of distance, newStartPoint, rotation, color
					defaultArrow = [arrowItem[0], None, .9, None, 0, 'black']
					newArrowItem = [arrowItem[i] if i < len(arrowItem) else defaultArrow[i] for i in range(len(defaultArrow))]
					#newArrowItem = arrowItem + [None] * (6 - len(arrowItem))
					i, j, pct, newStartPoint, rot, col = newArrowItem
					if i == 'lastDest':
						startPoint = lastDest[:]
					else:
						startPoint = self.cartesianPoints[i]
						if not j:
							j = (i + 1) % self.scaleSize
					deltaX, deltaY = [self.cartesianPoints[j][x] - startPoint[x] for x in [0, 1]]
					if newStartPoint != None:
						startPoint = self.cartesianPoints[newStartPoint]
					if rot != None:
						deltaX, deltaY = ChordSpace.rotate([deltaX, deltaY], rot*self.fundamentalAngle)
					lastDest = [startPoint[0] + deltaX, startPoint[1] + deltaY]
					self.ax.arrow(startPoint[0], startPoint[1], deltaX * pct, deltaY * pct, head_width=0.2, head_length=0.2, fc=col, ec=col)
		
		self.ax.set_yticks([]) 
		self.ax.set_xticks([])
		if self.saveName:
			plt.savefig(self.saveName, bbox_inches='tight')
		else:
			self.show()
		
	def show(self):			# would be nice to be able to reopen this, but that seems not to be possible
		plt.ion()
		plt.show()
		plt.pause(.0001)
	
	"""NOTE OFF CODE CAUSED THE PROGRAM TO BE SLUGGISH WHEN QUITTING, waiting for the Timers to quit; also produced sonic artifacts"""

	def getRvalue(self, a):
		
		"""utility code to get an R value for an angle a"""
		winding = int(a / (2 * np.pi))
		generalTheta = a % (2 * np.pi)
		if winding == 0:
			rCoord = self.radius
		else:
			rCoord = self.radius + ((self.chordSize - winding) * self.circleSeparation)
		if generalTheta < self._GAPANGLE:
			return rCoord
		else:
			for i, k in enumerate(self.theKeys):
				if k > a:
					rCoord = self.bigAngleDict[self.theKeys[i-1]]
					break
			return rCoord
	
	def get_maximally_even(self):
		return [int(x) for x in [f*(1.0*self.scaleSize)/self.chordSize for f in range(self.chordSize)]]
		
	@staticmethod
	def poltocar(theta, R):
		xCoords = []
		yCoords = []
		for i in range(len(theta)):
			t = theta[i]
			r = R[i]
			xCoords.append(r*np.sin(t))
			yCoords.append(r*np.cos(t))
		return [xCoords, yCoords]
		
	@staticmethod
	def draw_circle(thetaMin = 0, thetaMax = 0, r = 4, rMax = -1):
		theta = np.linspace(thetaMin, thetaMax, int((thetaMax - thetaMin) / 0.01))
		if rMax == -1:
			newR = [r] * len(theta)
		else:
			newR = np.linspace(r, rMax, len(theta))
		myCoords = ChordSpace.poltocar(theta, newR)
		return myCoords

	@staticmethod	
	def rotate(p, theta):
		return [p[0]*np.cos(theta) - p[1]*np.sin(theta), p[0]*np.sin(theta) + p[1]*np.cos(theta)]

	@staticmethod
	def make_RList(diffRanges):
		
		"""
		utility routine for drawing smooth curves, interpolating between one circle and noather
		you supply a list [firstdelta, seconddelta, numberofpoints], these are deltas between successive radii points
		
		example: 
		
			self.standardRList [[0, 5, 50], [5, 7, 50]] goes from 0 to 5 in 50 steps, then 5 to 7 in 50 steps (first steep then flattening)
		
		this program makes a symmetrical curve, so it adds the retrograde of the input pattern (this might not be the way to do things)
		
		"""
		diffSeq = []
		rangeList = [0]
		for r in diffRanges:
			diffSeq += list(np.linspace(*r))
		diffSeq += diffSeq[::-1]
		for d in diffSeq:
			rangeList.append(rangeList[-1] + d)
		return [linear_map(x, [rangeList[0], rangeList[-1]], [1, 0]) for x in rangeList]

	@staticmethod
	def interpolate_between_circles(firstAngle, firstR, secondAngle, secondR, rList):
		rList = ChordSpace.make_RList(rList)
		tempRList = [linear_map(x, [0, 1], [firstR, secondR]) for x in rList]
		newAngles = np.linspace(firstAngle, secondAngle, len(tempRList))
		return newAngles, tempRList
