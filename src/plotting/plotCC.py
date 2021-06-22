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

import librosa
import librosa.display

def plotCC(cc,title='None'):
    plt.figure(figsize=(10,8))
    plt.subplot(3, 1, 2)
    librosa.display.specshow(cc)
    plt.colorbar()
    plt.title(title)
    plt.axis('tight')
    plt.tight_layout()
    plt.show()

