# Tonal Harmony Model Generator
#
# MUSIC𝄞NTWRK
#
# A python library for pitch class set and rhythmic sequences classification and manipulation, 
# the generation of networks in generalized music and sound spaces, and the sonification of arbitrary data
#
# Copyright (C) 2020 Marco Buongiorno Nardelli
# http://www.materialssoundmusic.com, mbn@unt.edu
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

# Definition of chords and roman numeral symbols following the convention in
# The Annotated Beethoven Corpus (ABC): A Dataset of Harmonic Analyses of All Beethoven String Quartets Markus Neuwirth*, Daniel Harasim, Fabian C. Moss and Martin Rohrmeier
# Frontier in Digital Humanities, doi: 10.3389/fdigh.2018.00016
# adapted to music21 roman module
# chords can be added or removed at will - two sets, full and minimal, are provided - selection in the command line

from ..utils.generalizedOpsName import generalizedOpsName
import music21 as m21
import pickle, sys

def tonalHarmonyModel(mode='minimal'):
	
	if sys.argv[1] == 'full':
		# roman numeral classes - full
		rnlist_m = ['bi', 'i', '#i', 'bii', 'ii', '#ii', 'biii', 'iii', '#iii', 'biv', 'iv', '#iv', 'bv', 'v', '#v', 'bvi', 'vi', '#vi', 'bvii', 'vii', '#vii'] 
		rnlist_M = ['bI', 'I', '#I', 'bII', 'II', '#II', 'bIII', 'III', '#III', 'bIV', 'IV', '#IV','bV', 'V', '#V', 'bVI', 'VI', '#VI','bVII', 'VII', '#VII']
		dimlist = ['o','o7','/o7']
		auglist = ['+']
		suslist = ['52']
		extlist = ['7','b7','#7','7#43']
		extno5 = ['7[no5]','7[no5][no1]']
		filout = 'tonal.harmony.full'
	elif sys.argv[1] == 'minimal':
		# roman numeral classes - minimal
		rnlist_m = ['i', 'ii', 'biii', 'iii', 'iv', 'v', 'vi', 'bvii', 'vii'] 
		rnlist_M = ['I', 'bII','II', 'bIII', 'III', 'IV', 'V', 'VI', 'bVI' ,'bVII', 'VII']
		dimlist = ['o','o7','/o7']
		auglist = ['+']
		suslist = ['52']
		extlist = ['532','7','b7','#7','743','7#43'] #,'75#3']
		extno5 = ['[no5]','7[no5]','7[no5][no1]','9[no5]','9[no3]']
		extM7no5 = ['#7[no#3]']
		extm7no5 = ['b7[no3]']
		filout = 'tonal.harmony.minimal'
	else:
		print('model not available')
		sys.exit()

	# build the composite numerals and the corresponding pcs

	seq = []
	rn = []

	# minor triads
	for c in rnlist_m:
		chd = m21.roman.RomanNumeral(c)
		seq.append(chd.pitchClasses)
		rn.append(c)
	# major triads
	for c in rnlist_M:
		chd = m21.roman.RomanNumeral(c)
		seq.append(chd.pitchClasses)
		rn.append(c)
	# diminished
	for c in rnlist_m:
		for d in dimlist:
			f = c+d
			chd = m21.roman.RomanNumeral(f)
			seq.append(chd.pitchClasses)
			rn.append(f)
	# minor quadrichords
	for c in rnlist_m:
		for d in extlist:
			f = c+d
			chd = m21.roman.RomanNumeral(f)
			seq.append(chd.pitchClasses)
			rn.append(f)
	# major quadrichords
	for c in rnlist_M:
		for d in extlist:
			f = c+d
			chd = m21.roman.RomanNumeral(f)
			seq.append(chd.pitchClasses)
			rn.append(f)
	# sus triads
	for c in rnlist_m:
		f = c+suslist[0]
		chd = m21.roman.RomanNumeral(f)
		seq.append(chd.pitchClasses)
		rn.append(f)
	# 7[no5] triads (min+Maj)
	for c in rnlist_m:
		if c != 'bvii':
			for d in extno5:
				f = c+d
				chd = m21.roman.RomanNumeral(f)
				seq.append(chd.pitchClasses)
				rn.append(f)
	for c in rnlist_M:
		if c == 'III' or c == 'V':
			for d in extno5:
				f = c+d
				chd = m21.roman.RomanNumeral(f)
				seq.append(chd.pitchClasses)
				rn.append(f)
	# #7[no#3] triads
	for c in rnlist_M:
		f = c+extM7no5[0]
		chd = m21.roman.RomanNumeral(f)
		seq.append(chd.pitchClasses)
		rn.append(f)
	# b7[no3] triads
	for c in rnlist_m:
		if c == 'iv':
			f = c+extm7no5[0]
			chd = m21.roman.RomanNumeral(f)
			seq.append(chd.pitchClasses)
			rn.append(f)
	# augmented triads
	for c in rnlist_M:
		for d in auglist:
			f = c+d
			chd = m21.roman.RomanNumeral(f)
			seq.append(chd.pitchClasses)
			rn.append(f)
	# augmented quadrichords
	for c in rnlist_M:
		for d in extlist:
			f = c+auglist[0]+d
			chd = m21.roman.RomanNumeral(f)
			seq.append(chd.pitchClasses)
			rn.append(f)
			
	# ad-hoc additions
	adhoc = ['viio532','I[no3]','I543','IV543','V9[no5][no3]','I732','VII/o9[no3]','#iv/o','#viio','Ger']
	for f in adhoc:
		chd = m21.roman.RomanNumeral(f,m21.key.Key('C'))
		seq.append(chd.pitchClasses)
		rn.append(f)


	# make table of operators
	opstable = []
	for n in range(len(seq)):
		mat = []
		for m in range(len(seq)):
			_,m = generalizedOpsName(seq[n],seq[m])
			mat.append(m)
		opstable.append(mat)
		
	f = open(filout,'wb')
	pickle.dump(rn,f)
	pickle.dump(opstable,f)
	f.close()
	
	return
