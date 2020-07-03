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

def lookupProgr(ch1,ch2,table,header):

        tab = np.array(table)
        head = np.array(header)
        idx = np.where(head == ch1)
        idy = np.where(head == ch2)
        try:
                print(str(ch1).ljust(8),'->',tab[idx[0],idy[0]][0],'->',str(ch2).rjust(8))
        except:
                print('no operator found')

