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

from .lookupOps import lookupOps
from .lookupProgr import lookupProgr

def lookupWrapper(table,head,ops='',cstart='',cend=''):
        # if only cstart != None and ops != None prints matching progressions
        # if cstart, cend != None (pcs or roman numerals) and ops == None: prints operator connecting the chords
        # if cstart = roman numeral and cend = pitch: returns pcs of the chord
        if ops != '':
                lookupOps(ops,table,head,ch1=cstart,ch2=cend)
                print('===============================')
        elif cstart != '' and cend != '' and ops == '':
#        print('Major tonal space')
                lookupProgr(cstart,cend,table,head)
                print('===============================')
