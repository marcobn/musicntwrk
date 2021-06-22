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

import os, pickle, time
import numpy as np
import joblib as jlib

def modelDump(model,x_train,y_train,x_test,y_test,scaler,normal,res,train):
    filename = str(hex(int(time.time())))+'_'+str(round(res,3))
    model.save(filename+'.h5')
    np.save(filename+'.test',x_test)
    np.save(filename+'.name_test',y_test)
    np.save(filename+'.train',x_train)
    np.save(filename+'.name_train',y_train)
    jlib.dump(scaler, filename+'.scaler') 
    jlib.dump(normal, filename+'.normal')
    with open(filename+'.train.dict','wb') as file_pi:
        pickle.dump(train.history, file_pi)
    os.system('tar cvf '+filename+'.tar '+filename+'*')
    os.system('rm '+filename+'.h5')
    os.system('rm '+filename+'*.npy')
    os.system('rm '+filename+'.scaler')
    os.system('rm '+filename+'.normal')
    os.system('rm '+filename+'*.dict')
    print('model saved in ',filename)
    
