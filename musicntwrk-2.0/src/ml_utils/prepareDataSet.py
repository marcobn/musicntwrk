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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer

def prepareDataSet(mfcc,label,size=0.2):
    
    # N sounds for training - ~size*N sounds for testing
    xtrain,xtest,y_train,y_test = train_test_split(mfcc, label, test_size=size)
    # Data standardization
    x_train = np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1]*xtrain.shape[2]),order='C')
    x_test = np.reshape(xtest,(xtest.shape[0],xtest.shape[1]*xtest.shape[2]),order='C')
    scaler = StandardScaler(with_std=True)
    scaler.fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)
    # Data normalization
    normal = Normalizer(norm='max').fit(x_train_s)
    x_train = normal.transform(x_train_s)
    x_test = normal.transform(x_test_s)
    
    return(x_train,y_train,x_test,y_test,scaler,normal)

