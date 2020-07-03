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
#
# pcsPy
#
# A python library for pitch class set classification and manipulation, and network construction and analysis
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

def load_balancing(size,rank,n):
    # Load balancing
    splitsize = 1.0/size*n
    start = int(round(rank*splitsize))
    stop = int(round((rank+1)*splitsize))
    return(start,stop)

# For each processor calculate 3 values:
# 0 - Total number of items to be scattered/gathered on this processor
# 1 - Index in complete array where the subarray begins
# 2 - Dimension of the subarray on this processor
def load_sizes(size,n,dim):
    sizes = np.empty((size,3),dtype=int)
    splitsize = 1.0/size*n
    for i in range(size):
        start = int(round(i*splitsize))
        stop = int(round((i+1)*splitsize))
        sizes[i][0] = dim*(stop-start)
        sizes[i][1] = dim*start
        sizes[i][2] = stop-start
    return sizes
