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

def scaleDataSet(mfcc,scaler,normal):

    # Data standardization
    temp = np.reshape(mfcc,(mfcc.shape[0],mfcc.shape[1]*mfcc.shape[2]),order='C')
    temp_s = scaler.transform(temp)
    # Data normalization
    temp = normal.transform(temp_s)

    return(temp)
    
