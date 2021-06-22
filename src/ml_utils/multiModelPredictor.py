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

from .scaleDataSet import scaleDataSet

def multiModelPredictor(xnew,models,scalers,normals):
   
    try: 
        ynew = []
        for m in range(len(models)):
            temp = scaleDataSet(xnew,scalers[str(m)],normals[str(m)])
            try:
                ynew.append(models[str(m)].predict(temp)[0])
            except:
                temp = np.reshape(temp,(temp.shape[0],xnew.shape[1],xnew.shape[2],1),order='C')
                ynew.append(models[str(m)].predict(temp)[0])
        idx = np.argmax(np.sum(np.array(ynew),axis=0))
        prob = np.sum(np.array(ynew),axis=0)/len(models)
    except:
        temp = scaleDataSet(xnew,scalers,normals)
        try:
            prob = models.predict(temp)[0]
        except:
            temp = np.reshape(temp,(1,xnew.shape[1],xnew.shape[2],1),order='C')
            prob = models.predict(temp)[0]
        idx = np.argmax(np.array(prob))
    return(idx,prob)
                                                
