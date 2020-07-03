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
def mfccSoundDecayOptimal(mfcc,plot=False):
    tmp = []
    for i in range(5,300):
        a,_,_,res = mfccSoundDecay(mfcc,maxidx=i,plot=False)
        tmp.append([i,a,res/i])
    tmp = np.asarray(tmp)
    idx = argrelextrema(tmp[:,2], np.less)[0][-1]
    a,_,_,_ = mfccSoundDecay(mfcc,maxidx=idx,plot=plt)
    return(a)

