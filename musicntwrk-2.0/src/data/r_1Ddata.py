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

import numpy as np

def r_1Ddata(fileread):

	# Read data as columns (x,y1,y2,y3...)

	# Read the whole file into a single variable, which is a list of every row of the file.

	f=open(fileread,'r')
	lines=f.readlines()
	f.close()
	nlines = len(lines)
	ncol = len(lines[0].split())

	if ncol > 1:
		x = np.zeros(nlines,dtype=float)
		y = np.zeros((ncol-1,nlines),dtype=float)

		f=open(fileread,'r')
		for l,line in enumerate(f):
			p = line.split()
			x[l] = float(p[0])
			for n in range(0,ncol-1):
				y[n,l] = float(p[n+1])
		f.close()
	elif ncol ==1:
		x = np.linspace(0,nlines,nlines)
		y = np.zeros((1,nlines),dtype=float)
		f=open(fileread,'r')
		for l,line in enumerate(f):
			y[0,l] = float(line.split()[0])
		f.close()

	return(x,y)

