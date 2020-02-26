# Tonal Harmony Calculator
#
# MUSIC𝄞NTWRK
#
# A python library for pitch class set and rhythmic sequences classification and manipulation, 
# the generation of networks in generalized music and sound spaces, and the sonification of arbitrary data
#
# Copyright (C) 2020 Marco Buongiorno Nardelli
# http://www.materialssoundmusic.com, mbn@unt.edu
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

from pcsPy import *
import pickle
import PySimpleGUI as sg

def shortHands():
        # This is taken from roman.py in music21 and modified here
        figureShorthands = {
            '42':'7',
            '43':'7',
            '53': '',
            '3': '',
            '63': '6',
            '65': '7',
            '753': '7',
            '75': '7[no3]',  # controversial perhaps
            '73': '7[no5]',  # controversial perhaps
            '9753': '9',
            '975': '9',  # controversial perhaps
            '953': '9[no7]',  # controversial perhaps
            '97': '9[no7][no5]',  # controversial perhaps
            '95': '9[no7][no3]',  # controversial perhaps
            '93': '9[no7][no5]',  # controversial perhaps
        #  '653': '65',
            '653': '7',
            '6b53': '6b5',
            '643': '7',
        #  '642': '42',
        #  '642': '7[no5]',
            'o6b5':'o7',
            'o5b3':'o',
            'bb7b5b3': 'o7',
            'bb7b53': 'o7',
            # '6b5bb3': 'o65',
            'b7b5b3': 'ø7',
        }
        return(figureShorthands)

def enharmonicDictionary():
    keys = ['C','C#','D-','D','D#','E-','E','F','F#','G-','G','G#','A-','A','A#','B-','B']
    enharmonicDict = {}
    for k in keys:
        ini = m21.note.Note(60).nameWithOctave
        end = m21.note.Note(84).nameWithOctave
        major = [p for p in m21.scale.MajorScale(k).getPitches(ini,end)]
        minor = [p for p in m21.scale.MinorScale(k).getPitches(ini,end)]
        melod = [p for p in m21.scale.MelodicMinorScale(k).getPitches(ini,end)]
        harmo = [p for p in m21.scale.HarmonicMinorScale(k).getPitches(ini,end)]

        allscales = major+minor+melod+harmo

        clean = Remove(allscales)

        Cscale = sorted([str(f) for f in clean])
        Cscale = np.array(Cscale)

        Cmidi = []
        for n in Cscale:
            Cmidi.append(m21.pitch.Pitch(n).midi)
        Cmidi = np.array(Cmidi)
        idx = np.argsort(Cmidi)

        Cdict = dict(zip(Cmidi[idx[:]],Cscale[idx[:]]))

        for i in range(60,84):
            try:
                tmp = Cdict[i]
            except:
                Cdict.update({i:m21.note.Note(i).nameWithOctave})

        enharmonicDict.update({k:Cdict})
    return(enharmonicDict)

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

def lookupProgr(ch1,ch2,table,header):

        tab = np.array(table)
        head = np.array(header)
        idx = np.where(head == ch1)
        idy = np.where(head == ch2)
        try:
                print(str(ch1).ljust(8),'->',tab[idx[0],idy[0]][0],'->',str(ch2).rjust(8))
        except:
                print('no operator found')


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

def applyOps(name,chord):
        # operate on the pcs with a relational distance operator

        op = []
        for num in re.findall("[-\d]+", name):
                op.append(int(num))
        op = np.asarray(op)
        pcs = []
        for num in re.findall("[-\d]+", chord):
                pcs.append(int(num))
        pcs = np.asarray(pcs)
        if len(op) == len(pcs):
                selfto = (PCSet(pcs).normalOrder()+op)%12
                print(PCSet(selfto).normalOrder().tolist())
        elif len(op) > len(pcs):
                if len(op) - len(pcs) == 1:
                        # duplicate pitches
                        c = np.zeros(len(op),dtype=int)
                        pitch = PCSet(pcs,UNI=False,ORD=False).normalOrder()
                        c[:len(op)-1] = pitch
                        for i in range(len(op)-1):
                                c[len(op)-1] = pitch[i]
                                selfto = (PCSet(c,UNI=False,ORD=False).normalOrder()+op)%12
                                print(PCSet(selfto).normalOrder().tolist())
                else:
                        print('operation not defined')
                        
def clearOutput():
        return

def showPitches():
        return
    
def showRN():
        return
