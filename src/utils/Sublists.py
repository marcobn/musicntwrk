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

import itertools as itr

def Sublists(lst):
    
    for doslice in itr.product([True, False], repeat=len(lst) - 1):
        slices = []
        start = 0
        for i, slicehere in enumerate(doslice, 1):
            if slicehere:
                slices.append(lst[start:i])
                start = i
        slices.append(lst[start:])
        yield slices
