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
def vectorDistance(a,b,distance='euclidean'):
    '''
    •	calculates the distance between two vectors (duration or inteval)
    •	a,b (str) – vectors
    '''
    a = np.asarray(a)
    b = np.asarray(b)
    n = a.shape[0]
    if a.shape[0] != b.shape[0]:
        print('dimension of arrays must be equal')
        sys.exit()
    diff = sklm.pairwise_distances(a.reshape(1, -1),b.reshape(1, -1),metric=distance)[0]
    return(diff)
