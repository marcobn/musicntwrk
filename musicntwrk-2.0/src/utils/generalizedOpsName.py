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

from .opsNameFull import opsNameFull, opsNameAbs
from .opsName import opsName
from .minimalNoBijDistance import minimalNoBijDistance

def generalizedOpsName(a,b,TET,distance):
# generalizes the operator name function for no-bijective chord progression
    if len(a) == len(b):
        return(a,opsNameFull(a,b,TET))
    else:
        if len(a) > len(b):
            pair,r = minimalNoBijDistance(a,b,TET,distance)
            return(r,opsNameFull(a,r,TET))
        else:
            pair,r = minimalNoBijDistance(b,a,TET,distance)
            return(r,opsNameFull(r,b,TET))

def generalizedOpsNameAbs(a,b,TET,distance):
# generalizes the operator name function for no-bijective chord progression
    if len(a) == len(b):
        return(a,opsNameAbs(a,b,TET))
    else:
        if len(a) > len(b):
            pair,r = minimalNoBijDistance(a,b,TET,distance)
            return(r,opsNameAbs(a,r,TET))
        else:
            pair,r = minimalNoBijDistance(b,a,TET,distance)
            return(r,opsNameAbs(r,b,TET))
