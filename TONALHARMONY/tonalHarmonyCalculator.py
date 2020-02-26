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
from tonalHarmonyDefs import *

enharmonicDict = enharmonicDictionary()
figureShorthands = shortHands()

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
#            probably not needed
#            for n in range(1,len(ch)):
#                if ch[n] < ch[n-1]: ch[n] += 12
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
