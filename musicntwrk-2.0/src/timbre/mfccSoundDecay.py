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
def mfccSoundDecay(mfcc,maxidx=25,plot=False):
    # evaluate the normalized sound decay envelope from the MFCC
    idmax = np.argwhere(mfcc == np.max(mfcc))[0,0]
    mfcc = mfcc[idmax:]
    t = np.arange(len(mfcc))
    mfcc = mfcc-np.min(mfcc)
    mfcc = np.log(mfcc[mfcc>1.e-10])
    if plot:
        plt.plot(mfcc)
    if maxidx > mfcc.shape[0]: maxidx = mfcc.shape[0]
    alpha,res,_,_,_ = np.polyfit(t[:maxidx],mfcc[:maxidx],1,full=True)
    alpha = np.poly1d(alpha)
    if plot:
        plt.plot(t[:maxidx],mfcc[:maxidx])
        tp = np.linspace(0,t[maxidx],50)
        plt.plot(tp,alpha(tp),'.')
        plt.show()
    return(np.abs(alpha[1]),alpha,t,res)
    
