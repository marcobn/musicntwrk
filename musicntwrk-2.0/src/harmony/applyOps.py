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

import re
import numpy as np

from ..musicntwrk import PCSet
from ..utils.Remove import Remove

def applyOps(name,chord,prt=True):
    # operate on the pcs with a relational distance operator
    op = []
    for num in re.findall("[-\d]+", name):
        op.append(int(num))
    op = np.asarray(op)
    pcs = []
    for num in re.findall("[-\d]+", chord):
        pcs.append(int(num))
    pcs = np.asarray(pcs)
    if len(op) == len(pcs):
        selfto = (PCSet(pcs).normalOrder()+op)%12
        if prt: print(PCSet(selfto).normalOrder().tolist())
        return([str(PCSet(selfto).normalOrder().tolist())])
    elif len(op) - len(pcs) == 1:
        # duplicate pitches
        c = np.zeros(len(op),dtype=int)
        pitch = PCSet(pcs,UNI=False,ORD=False).normalOrder()
        c[:len(op)-1] = pitch
        add = []
        for i in range(len(op)-1):
            c[len(op)-1] = pitch[i]
            selfto = (PCSet(c,UNI=False,ORD=False).normalOrder()+op)%12
            add.append(str(PCSet(selfto).normalOrder().tolist()))
        if prt: print(Remove(add))
        return(Remove(add))
    elif len(pcs) - len(op) == 1:
        # add a unison operator (0)
        add = []
        for i in range(pcs.shape[0]): 
            c = np.insert(op,i,0)
            selfto = (PCSet(pcs).normalOrder()+c)%12
            add.append(str(PCSet(selfto).normalOrder().tolist()))
        if prt: print(Remove(add))
        return(Remove(add))
    else:
        print('operation not defined')
