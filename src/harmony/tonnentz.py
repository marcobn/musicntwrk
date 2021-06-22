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

import pandas as pd

def tonnentz(x,y):
    tnz = pd.DataFrame(None,columns=['','','','','','','','',''])
    tmp = pd.DataFrame([[str((4*y)%12),'',str((x+3*y)%12),'',str((2*x+2*y)%12),'',str((3*x+y)%12),'',str((4*x)%12)]],
                       columns=['','','','','','','','',''])
    tnz = tnz.append(tmp)
    tmp = pd.DataFrame([['',str((3*y)%12),'',str((x+2*y)%12),'',str((2*x+y)%12),'',str((3*x)%12),'']],
                       columns=['','','','','','','','',''])
    tnz = tnz.append(tmp)
    tmp = pd.DataFrame([[str((-x+3*y)%12),'',str((2*y)%12),'',str((x+y)%12),'',str((2*x)%12),'',str((3*x-y)%12)]],
                       columns=['','','','','','','','',''])
    tnz = tnz.append(tmp)
    tmp = pd.DataFrame([['',str((-x+2*y)%12),'',str(y%12),'',str(x%12),'',str((2*x-y)%12),'']],
                       columns=['','','','','','','','',''])
    tnz = tnz.append(tmp)
    tmp = pd.DataFrame([[str((-2*x+2*y)%12),'',str((-x+y)%12),'',str(0),'',str((x-y)%12),'',str((2*x-2*y)%12)]],
                       columns=['','','','','','','','',''])
    tnz = tnz.append(tmp)
    tmp = pd.DataFrame([['',str((-2*x+y)%12),'',str(-x%12),'',str(-y%12),'',str((x-2*y)%12),'']],
                       columns=['','','','','','','','',''])
    tnz = tnz.append(tmp)
    tmp = pd.DataFrame([[str((-3*x+y)%12),'',str(-2*x%12),'',str((-x-y)%12),'',str(-2*y%12),'',str((x-3*y)%12)]],
                       columns=['','','','','','','','',''])
    tnz = tnz.append(tmp)
    tmp = pd.DataFrame([['',str(-3*x%12),'',str((-2*x-y)%12),'',str((-x-2*y)%12),'',str(-3*y%12),'']],
                       columns=['','','','','','','','',''])
    tnz = tnz.append(tmp)
    tmp = pd.DataFrame([[str(-4*x%12),'',str((-3*x-y)%12),'',str((-2*x-2*y)%12),'',str((-x-3*y)%12),'',str(-4*y%12)]],
                       columns=['','','','','','','','',''])
    tnz = tnz.append(tmp)

    return(tnz)
