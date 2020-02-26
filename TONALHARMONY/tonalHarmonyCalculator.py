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

enharmonicDict = enharmonicDictionary()

# Lookup dictionary that maps button to function to call
func_dict = {'lookup':lookupWrapper,'applyOp':applyOps,'findOp':generalizedOpsName,'findRN':showRN,'pitches':showPitches,'CLEAR':clearOutput}
# Layout the design of the GUI
layout = [[sg.Text('Tonal Harmony Calculator', auto_size_text=True, font='Helvetica 24', background_color='#F68E01')],
          [sg.Text('Progression lookup / Apply operator / Operators / Roman numerals / Pitches', auto_size_text=True, font='Helvetica 18', background_color='#F68E01')],
          [sg.Text('tonal model', size=(12, 1),auto_size_text=True, font='Helvetica 18', background_color='#F68E01'), sg.Input(size=(50, 1)), sg.FileBrowse(), sg.Submit()],
          [sg.Text('operator', auto_size_text=True, font='Helvetica 18', background_color='#F68E01'),sg.InputText(key='ops',size=(16, 1)),
           sg.Text('initial chord', auto_size_text=True, font='Helvetica 18', background_color='#F68E01'),sg.InputText(key='ch1',size=(16, 1)),
           sg.Text('final chord', auto_size_text=True, font='Helvetica 18', background_color='#F68E01'),sg.InputText(key='ch2',size=(16, 1))],
          [sg.Button('lookup'),sg.Button('applyOp'),sg.Button('findOp'),sg.Button('findRN'),sg.Button('pitches'),sg.Button('CLEAR'),sg.Quit()],
          [sg.Output(size=(76, 20),key='_output_')],
          [sg.Text('Marco Buongiorno Nardelli - www.musicntwrk.com (2020)', auto_size_text=True, font='Helvetica 12', background_color='#3364FF')]]
  
# Show the Window to the user
sg.SetOptions(background_color='#3364FF', element_background_color='#F68E01', text_element_background_color='#F68E01', input_elements_background_color='#F68E01', font='Helvetica 18')
window = sg.Window('Unified Theory of Tonal Harmony', layout, keep_on_top = False)
  
# Event loop. Read buttons, make callbacks
while True:
    # Read the Window
    event, value = window.Read()
    if event in('Submit'):
        try:
            f = open(value[0],'rb')
            head = pickle.load(f)
            table = pickle.load(f)
            f.close()
        except:
            print('file not found')
    if event in ('Quit', None):
        break
    # Lookup event in function dictionary
    try:
        func_to_call = func_dict[event]   # look for a match in the function dictionary
        if func_to_call.__name__ == 'lookupWrapper':
            func_to_call(table,head,ops=value['ops'],cstart=value['ch1'],cend=value['ch2'])
        elif func_to_call.__name__ == 'applyOps':
            func_to_call(value['ops'],value['ch1'])
        elif func_to_call.__name__ == 'generalizedOpsName':
            a = []
            for num in re.findall("[-\d]+", value['ch1']):
                a.append(int(num))
            a = np.asarray(a)
            b = []
            for num in re.findall("[-\d]+", value['ch2']):
                b.append(int(num))
            b = np.asarray(b)
            _,op = generalizedOpsName(a,b)
            print(op)
        elif func_to_call.__name__ == 'showPitches':
            chd = m21.roman.RomanNumeral(value['ch1'],value['ch2'])
            pcs = chd.pitchClasses
            print(pcs)
        elif func_to_call.__name__ == 'showRN':
            a = []
            for num in re.findall("[-\d]+", value['ch1']):
                a.append(int(num))
            ch = np.copy(PCSet(a).normalOrder().tolist())
            for n in range(1,len(ch)):
                if ch[n] < ch[n-1]: ch[n] += 12
            ch += 60
            p = []
            for c in ch:
                p.append(enharmonicDict[value['ch2']][c])
            n = m21.chord.Chord(p)
            rn = m21.roman.romanNumeralFromChord(n,m21.key.Key(value['ch2'])).romanNumeralAlone
            fig =m21.roman.postFigureFromChordAndKey(n, m21.key.Key(value['ch2']))
            try:
                fig = figureShorthands[fig]
            except:
                pass
            print(rn+fig)
        elif func_to_call.__name__ == 'clearOutput':
            window.FindElement('_output_').Update('')
    except Exception as e:
        print(e)
        pass
  
window.Close()
