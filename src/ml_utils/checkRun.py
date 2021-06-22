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

import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

def checkRun(train,modelfiles):
    # plot accuracy and loss for training and validation sets over epochs
    try:
        try:
            accuracy = train.history['accuracy']
            val_accuracy = train.history['val_accuracy']
        except:
            accuracy = train.history['acc']
            val_accuracy = train.history['val_acc']
        loss = train.history['loss']
        val_loss = train.history['val_loss']
        epochs = range(len(accuracy))
        print('Model: ',modelfiles)
        plt.figure(figsize=(14,8))
        plt.subplot(3, 2, 3)
        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(3, 2, 4)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
    except:
        for n in range(len(train)):
            try:
                accuracy = train[str(n)]['accuracy']
                val_accuracy = train[str(n)]['val_accuracy']
            except:
                accuracy = train[str(n)]['acc']
                val_accuracy = train[str(n)]['val_acc']
            loss = train[str(n)]['loss']
            val_loss = train[str(n)]['val_loss']
            epochs = range(len(accuracy))
            print('Model: ',modelfiles[n])
            plt.figure(figsize=(14,8))
            plt.subplot(3, 2, 3)
            plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
            plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
            plt.title('Training and validation accuracy')
            plt.legend()
            plt.subplot(3, 2, 4)
            plt.plot(epochs, loss, 'bo', label='Training loss')
            plt.plot(epochs, val_loss, 'b', label='Validation loss')
            plt.title('Training and validation loss')
            plt.legend()
            plt.show()

