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
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

def barplot(spectrum,colormap=plt.cm.coolwarm,flip=True):
	# print the mfcc0 matrix for all sounds
	axprops = dict(xticks=[], yticks=[])
	barprops = dict(aspect='auto', cmap=colormap, interpolation='nearest')
	fig = plt.figure()
	ax1 = fig.add_axes([0.1, 0.1, 3.1, 0.7], **axprops)
	if flip: 
		cax = ax1.matshow(np.flip(spectrum.T), **barprops)
	else:
		cax = ax1.matshow(spectrum.T, **barprops)
	fig.colorbar(cax)
	plt.show()

