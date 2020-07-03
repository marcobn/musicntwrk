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

def lookupOps(ops,table,header,Pnumber='',ch1='',ch2=''):
        operator = ops
        tab = np.array(table)
        if Pnumber != '':
                try:
                        print('Pnumber of operator '+operator+' =',Pnumber[operator],'\n')
                except:
                        print('operator not found in Pnumber')
                        return
        idx,idy = np.where(tab == operator)
        for n in range(len(idy)):
                if ch1 == '' and ch2 == '':
                        print(str(header[idx[n]]).ljust(12),str(' ->\t'+header[idy[n]]).rjust(0))
                elif ch1 != '' and ch2 == '':
                        if ch1[0] == '?':
                                if (ch1[1:] in str(header[idx[n]])):
                                        print(str(header[idx[n]]).ljust(12),str(' ->\t'+header[idy[n]]).rjust(0))
                        else:
                                if (ch1 == str(header[idx[n]])):
                                    print(str(header[idx[n]]).ljust(12),str(' ->\t'+header[idy[n]]).rjust(0))
                elif ch2 != '' and ch1 == '':
                        if ch2[0] == '?':
                                if (ch2[1:] in str(header[idy[n]])):
                                        print(str(header[idx[n]]).ljust(12),str(' ->\t'+header[idy[n]]).rjust(0))
                        else:
                                if (ch2 == str(header[idy[n]])):
                                    print(str(header[idx[n]]).ljust(12),str(' ->\t'+header[idy[n]]).rjust(0))
                elif ch2 != '' and ch1 != '':
                        if ch1[0] == '?' and ch2[0] != '?':
                                if (ch1[1:] in str(header[idx[n]])) and (ch2 == str(header[idy[n]])):
                                        print(str(header[idx[n]]).ljust(12),str(' ->\t'+header[idy[n]]).rjust(0))
                        elif ch1[0] != '?' and ch2[0] == '?':
                                if (ch1 == str(header[idx[n]])) and (ch2[1:] in str(header[idy[n]])):
                                    print(str(header[idx[n]]).ljust(12),str(' ->\t'+header[idy[n]]).rjust(0))
                        elif ch1[0] != '?' and ch2[0] != '?':
                                if (ch1 == str(header[idx[n]])) and (ch2 == str(header[idy[n]])):
                                    print(str(header[idx[n]]).ljust(12),str(' ->\t'+header[idy[n]]).rjust(0))
                        else:
                                continue
