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

def opsDistance(name):
    # returns distance for given operator
    opname = np.asarray(' '.join(i for i in name if i.isdigit()).split())
    opdist = np.sqrt(np.sum(np.asarray([list(map(int, x)) for x in opname]).reshape(1,-1)[0]*
        np.asarray([list(map(int, x)) for x in opname]).reshape(1,-1)[0]))
    return(name,opdist)
