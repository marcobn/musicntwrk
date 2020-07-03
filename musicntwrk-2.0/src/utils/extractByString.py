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
def extractByString(name,label,string):
    '''
    •	extract rows of any dictionary (from csv file or pandas DataFrame) according to a particular string in column 'label'
    •	name (str or pandas DataFrame) – name of the dictionary
    •	string (str) – string to find in column
    •	label (str) – name of column of string
    '''
    if type(name) is str: 
        df = pd.read_csv(name)
    else:
        df = name
    return(df[df[label].str.contains(string)])
