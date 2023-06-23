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

def Remove(duplicate): 
    # function to remove duplicates from list
    final_list = [] 
    for num in duplicate: 
        if num not in final_list: 
            final_list.append(num) 
    return final_list 

def pruneList(original):
    # function to prune list from successive equal elements
    # conserving the information on multiplicity
    pruned = [original[0]]
    mult = [1]
    for n in original:
        if n == pruned[-1]:
            mult[-1] += 1
            pass
        else:
            pruned.append(n)
            mult.append(1)
    return np.array(pruned), np.array(mult)