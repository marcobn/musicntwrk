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

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU

from .modelDump import modelDump
from .prepareDataSet import prepareDataSet

def trainCNNmodel(mfcc,label,gpu=0,cpu=4,niter=100,nstep=10,neur=16,test=0.08,num_classes=2,
                                    epoch=30,verb=0,thr=0.85,w=False):
    # Convolutional NN

#    config = tf.ConfigProto(device_count={'GPU':gpu, 'CPU':cpu})
#    sess = tf.Session(config=config)

    # Train the model
    for trial in range(niter):

        if trial%nstep == 0: x_train,y_train,x_test,y_test,scaler,normal = prepareDataSet(mfcc,label,size=test)
        shapedata = (x_train.shape[1],)
        x_train = np.reshape(x_train,(x_train.shape[0],mfcc.shape[1],mfcc.shape[2],1),order='C')
        x_test = np.reshape(x_test,(x_test.shape[0],mfcc.shape[1],mfcc.shape[2],1),order='C')    

        # train the model
        batch_size = None
        nnn = neur

        model = Sequential()
        model.add(Conv2D(nnn, kernel_size=(3, 3),activation='linear',
                                         input_shape=(mfcc.shape[1],mfcc.shape[2],1),padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2),padding='same'))
        model.add(Dropout(0.25))
        model.add(Conv2D(2*nnn, (3, 3), activation='linear',padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        model.add(Conv2D(4*nnn, (3, 3), activation='linear',padding='same'))
        model.add(LeakyReLU(alpha=0.1))                  
        model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(4*nnn, activation='linear'))
        model.add(LeakyReLU(alpha=0.1))  
        model.add(Dropout(0.3))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer='adam',
                                    loss='sparse_categorical_crossentropy',
                                    metrics=['accuracy'])

        train = model.fit(x_train, y_train, epochs=epoch, verbose=verb,validation_data=(x_test,y_test))

        res = model.evaluate(x_test, y_test, verbose=0)
        print('loss ',res[0],'accuracy ',res[1])
        if res[1] >= thr and w == True:
            print('found good match ',round(res[1],3))
            modelDump(model,x_train,y_train,x_test,y_test,scaler,normal,res[1],train)
#    sess.close()
    return(model,x_train,y_train,x_test,y_test,scaler,normal,res[1],train)

