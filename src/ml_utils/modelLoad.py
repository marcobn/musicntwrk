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

import pickle
import joblib as jlib
import tensorflow as tf
import numpy as np

def modelLoad(filename,npy=False):
    model = tf.keras.models.load_model(filename+'.h5')
    scaler = jlib.load(filename+'.scaler') 
    normal = jlib.load(filename+'.normal')
    try:
        with open(filename+'.train.dict','rb') as file_pi:
            trdict=pickle.load(file_pi)
    except:
        try:
            with open(filename+'train.dict','rb') as file_pi:
                trdict=pickle.load(file_pi)
        except:
            pass
    if npy:
        x_test = np.load(filename+'.test.npy')
        y_test = np.load(filename+'.name_test.npy')
        x_train = np.load(filename+'.train.npy')
        y_train = np.load(filename+'.name_train.npy')
        return(model,x_train,y_train,x_test,y_test,scaler,normal,trdict)
    else:
        try:
            return(model,scaler,normal,trdict)
        except:
            return(model,scaler,normal)
    
