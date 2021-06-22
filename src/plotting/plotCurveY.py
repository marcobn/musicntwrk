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

import vpython as vp
import numpy as np

def plotCurveY(y):
    stage=vp.canvas()
    f1 = vp.gcurve(color=vp.color.green)
    for n in range(y.shape[0]):
        f1.plot(pos=(n,y[n]))
