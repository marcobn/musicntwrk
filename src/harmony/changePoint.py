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

import ruptures as rpt
import numpy as np
import matplotlib.pyplot as plt

def changePoint(value,model='rbf',penalty=1.0,brakepts=None,plot=False):
    # change point detection
    # available models: "rbf", "l1", "l2", rbf", "linear", "normal", "ar", "mahalanobis"
    signal = np.array(value)
    algo = rpt.Binseg(model=model).fit(signal)
    my_bkps = algo.predict(pen=penalty,n_bkps=brakepts)
    if plot:
        # show results
        rpt.show.display(signal, my_bkps, figsize=(10, 3))
        plt.show()
    # define regions from breaking points
    sections = my_bkps
    sections.insert(0,0)
    # check last point
    sections[-1] -= 1
    if plot: print('model = ',model,' - sections = ',sections)
    return(sections)