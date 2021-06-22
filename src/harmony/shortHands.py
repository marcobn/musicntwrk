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

def shortHands():
        # This is taken from roman.py in music21 and modified here
#        figureShorthands = {
#            '42':'7',
#            '43':'7',
#            '53': '',
#            '54': '52',
#            '3': '',
#            '63': '',
#            '6' : '',
#            '64' : '',
#            '65': '7',
#            '753': '7',
#            '#753':'7',
#            '75': '7[no3]',  # controversial perhaps
#            '73': '7[no5]',  # controversial perhaps
#            '752': '9[no3]',
#            '9753': '9',
#            '975': '9',  # controversial perhaps
#            '953': '9[no7]',  # controversial perhaps
#            '97': '9[no7][no5]',  # controversial perhaps
#            '32': '9[no5][no3]',
#            '95': '9[no7][no3]',  # controversial perhaps
#            '93': '9[no7][no5]',  # controversial perhaps
#        #  '653': '65',
#            '653': '7',
#            '6b53': '6b5',
#            '643': '7',
#        #  '642': '42',
#        #  '642': '7[no5]',
#            'o64' : 'o',
#            'o6b5':'o7',
#            'o5b3':'o',
#            'bb7b5b3': 'o7',
#            'bb7b53': 'o7',
#            # '6b5bb3': 'o65',
#            'b7b5b3': '/o7',
#        }
        figureShorthands = {
            '53': '',
            '3': '',
            '63': '6',
            '753': '7',
            '75': '7',  # controversial perhaps
            '73': '7',  # controversial perhaps
            '9753': '9',
            '975': '9',  # controversial perhaps
            '953': '9',  # controversial perhaps
            '97': '9',  # controversial perhaps
            '95': '9',  # controversial perhaps
            '93': '9',  # controversial perhaps
            '653': '65',
            '6b53': '6b5',
            '643': '43',
            '642': '42',
            'bb7b5b3': 'o7',
            'bb7b53': 'o7',
            # '6b5bb3': 'o65',
            'b7b5b3': 'ø7',
        }
        return(figureShorthands)

