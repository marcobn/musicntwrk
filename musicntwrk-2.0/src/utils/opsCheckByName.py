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

from .opsName import opsName
from .opsNameFull import opsNameAbs

def opsCheckByName(a,b,name,TET):
    # given two vectors returns check if the connecting operator is the one sought for
    opname = opsName(a,b,TET)
    if opname != name:
        return(False)
    else:
        return(True)

def opsCheckByNameAbs(a,b,name,TET):
    # given two vectors returns check if the connecting operator is the one sought for
    opname = opsNameAbs(a,b,TET)
    if opname != name:
        return(False)
    else:
        return(True)
