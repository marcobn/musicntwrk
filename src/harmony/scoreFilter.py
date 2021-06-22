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
import music21 as m21
import numpy as np
import matplotlib.pyplot as plt
from ..utils.Remove import Remove

def scoreFilter(seq,chords,thr=0,plot=False):
    # score sequence and unique pcs count
    if chords != None:
        mea = []
        for c in chords.recurse().getElementsByClass('Chord'):
            mea.append(str(c.measureNumber))
    su = [str(f) for f in Remove(seq)]
    ind = []
    for n in range(len(su)):
        ind.append(str(n))
    su = np.asarray(su)
    ind = np.asarray(ind)
    labeltot = dict(zip(su,ind))
    indextot = dict(zip(ind,su))
    value = []
    for i in range(len(seq)):
        value.append(int(labeltot[str(seq[i])]))
    if plot:
        plt.plot(value,'o')
        plt.show()
        print('total number of chords = ',len(seq))
    hh,bb = np.histogram(np.array(value),bins=len(labeltot))
    if plot:
        plt.plot(hh,drawstyle='steps-mid')
        plt.show()
    # filtering pcs with occurrences lower than threshold = thr
    filtered = []
    if chords != None: fmeasure = []
    for i in range(len(seq)):
        if hh[int(labeltot[str(seq[i])])] > thr:
            filtered.append(seq[i])
            if chords != None: fmeasure.append(mea[i])
    su = [str(f) for f in Remove(filtered)]
    ind = []
    for n in range(len(su)):
        ind.append(str(n))
    su = np.asarray(su)
    ind = np.asarray(ind)
    labelfilt = dict(zip(su,ind))
    indexfilt = dict(zip(ind,su))
    valuef = []
    for i in range(len(filtered)):
        valuef.append(int(labelfilt[str(filtered[i])]))
    if plot:
        plt.plot(valuef,'or')
        plt.show()
        print('total number of filtered chords = ',len(filtered))
    if chords != None:
        return(value,valuef,filtered,fmeasure)
    else:
        return(value,valuef,filtered)