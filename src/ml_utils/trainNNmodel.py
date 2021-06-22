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
import tensorflow as tf

from .modelDump import modelDump
from .prepareDataSet import prepareDataSet

def trainNNmodel(mfcc,label,gpu=0,cpu=4,niter=100,nstep=10,neur=16,test=0.08,num_classes=2,epoch=30,verb=0,thr=0.85,w=False):
    # train a 2 layers NN

#    config = tf.ConfigProto(device_count={'GPU':gpu, 'CPU':cpu})
#    sess = tf.Session(config=config)

    # Train the model
    for trial in range(niter):

        if trial%nstep == 0: x_train,y_train,x_test,y_test,scaler,normal = prepareDataSet(mfcc,label,size=test)
        shapedata = (x_train.shape[1],)

        # train the model
        nnn = neur
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=shapedata),
            tf.keras.layers.Dense(nnn, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(2*nnn, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)])

        model.compile(optimizer='adam',
                                    loss='sparse_categorical_crossentropy',
                                    metrics=['accuracy'])

        train = model.fit(x_train, y_train, epochs=epoch, verbose=verb,validation_data=(x_test,y_test))

        res = model.evaluate(x_test, y_test, verbose=0)
        print('loss ',res[0],'accuracy ',res[1])
        if res[1] > thr and w == True:
            print('found good match ',round(res[1],3))
            modelDump(model,x_train,y_train,x_test,y_test,scaler,normal,res[1],train)
#    sess.close()
    return(model,x_train,y_train,x_test,y_test,scaler,normal,res[1],train)    

