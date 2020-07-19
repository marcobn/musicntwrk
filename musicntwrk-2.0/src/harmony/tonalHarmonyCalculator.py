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

import pickle,re
import PySimpleGUI as sg
import numpy as np
import music21 as m21
from music21.figuredBass import realizerScale

from .applyOps import applyOps
from .lookupWrapper import lookupWrapper
from ..utils.generalizedOpsName import generalizedOpsName
from ..utils.Remove import Remove

def tonalHarmonyCalculator():
  
    def clearOutput():
        return

    def showPitches():
        return

    def figBass():
      return
      
    def showRN():
        return


    # Lookup dictionary that maps button to function to call
    func_dict = {'lookup':lookupWrapper,'applyOp':applyOps,'findOp':generalizedOpsName,'findRN':showRN,'pitches':showPitches,
    'figBass':figBass,'CLEAR':clearOutput}
    # Layout the design of the GUI
    layout = [[sg.Text('Tonal Harmony Calculator', auto_size_text=True, font='Helvetica 24', background_color='#F68E01')],
              [sg.Text('Progression lookup / Apply operator / Operators / Roman numerals / Pitches', auto_size_text=True, font='Helvetica 18', background_color='#F68E01'),sg.Button('HELP')],
              [sg.Text('tonal model', size=(12, 1),auto_size_text=True, font='Helvetica 18', background_color='#F68E01'), sg.Input(size=(50, 1)), sg.FileBrowse(), sg.Submit()],
              [sg.Text('operator', auto_size_text=True, font='Helvetica 18', background_color='#F68E01'),sg.InputText(key='ops',size=(16, 1)),
                sg.Text('initial chord', auto_size_text=True, font='Helvetica 18', background_color='#F68E01'),sg.InputText(key='ch1',size=(16, 1)),
                sg.Text('final chord', auto_size_text=True, font='Helvetica 18', background_color='#F68E01'),sg.InputText(key='ch2',size=(16, 1))],
              [sg.Button('lookup'),sg.Button('applyOp'),sg.Button('findOp'),sg.Button('findRN'),sg.Button('pitches'),sg.Button('figBass'),
              sg.Button('CLEAR'),sg.Quit()],
              [sg.Output(size=(76, 20),key='_output_')],
              [sg.Text('Marco Buongiorno Nardelli - www.musicntwrk.com (2020)', auto_size_text=True, font='Helvetica 12', background_color='#3364FF')]]
      
    # Show the Window to the user
    sg.SetOptions(background_color='#3364FF', element_background_color='#F68E01', text_element_background_color='#F68E01', input_elements_background_color='#F68E01', font='Helvetica 18')
    window = sg.Window('Unified Theory of Tonal Harmony', layout, keep_on_top = False)
      
    # Event loop. Read buttons, make callbacks
    while True:
        # Read the Window
        event, value = window.Read()
        if event in('HELP'):
            print('Usage of the "harmony calculator" app')
            print('')
            print('tonal model - is the table that stores the operators that connect every chord of a chosen tonal model')
            print('(in roman numeral form).')
            print('The chords that are included in the model can be chosen in tonalHarmonyModels that produces the final')
            print('table. it is read by clicking on "Submit".')
            print('A minimal model can be downloaded from')
            print('https://github.com/marcobn/musicntwrk/tree/master/musicntwrk-2.0/examples')
            print('')
            print('The next set of enties depend on the action that is requested by the user. They can be filled with:')
            print('')
            print('operator - voice leading operator as O(i,j,k,...) ')
            print('')
            print('initial chord - chord in roman numeral or pcs as list of integers')
            print('')
            print('final chord - chord in roman numeral or pcs as list of integers or key as letter (A,a,C#, Bb etc.)')
            print('')
            print('Actions:')
            print('')
            print('lookup - given an operator produces the list of all the chords that are connected by it')
            print('')
            print('applyOp - given an operator and a pcs as list of integers, produces the resulting pcs')
            print('')
            print('findOp - given an initial pcs and a final pcs, finds the operator that connects them')
            print('')
            print('findRN - given a pcs in "initial chord" and a key in "final chord" uses music21 to produce the corresponding roman numeral')
            print('')
            print('pitches - given a roman numeral in "initial chord" and a key in "final chord" uses music21 to produce the corresponding pcs  ')
            print('')
            print('FigBass - given a pitch and a qualifier (major, minor) in "operator", the bass note (letter) in "initial chord" and the figure in "final chord" produces the pithes of the figured bass')
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
                _ = func_to_call(value['ops'],value['ch1'])
            elif func_to_call.__name__ == 'generalizedOpsName':
                a = []
                for num in re.findall("[-\d]+", value['ch1']):
                    a.append(int(num))
                a = np.asarray(a)
                b = []
                for num in re.findall("[-\d]+", value['ch2']):
                    b.append(int(num))
                b = np.asarray(b)
                _,op = generalizedOpsName(a,b,TET=12,distance='euclidean')
                print(op)
            elif func_to_call.__name__ == 'showPitches':
                chd = m21.roman.RomanNumeral(value['ch1'],value['ch2'])
                pcs = chd.pitchClasses
                print(pcs)
            elif func_to_call.__name__ == 'figBass':
                a = value['ops']
                fbScale = realizerScale.FiguredBassScale(value['ops'].split()[0],value['ops'].split()[1])
                pc = []
                for p in fbScale.getPitches(value['ch1'],value['ch2']):
                    pc.append(m21.pitch.Pitch(p).pitchClass)
                print(value['ch1'],value['ch2'],Remove(pc))
            elif func_to_call.__name__ == 'showRN':
                a = []
                for num in value['ch1'].replace('[','').replace(']','').split(','):
                    if num.isdecimal():
                        a.append(int(num))
                    else:
                        a.append(num)
                n = m21.chord.Chord(a)
                rn = m21.roman.romanNumeralFromChord(n,m21.key.Key(str(value['ch2']))).figure 
                print(rn)
            elif func_to_call.__name__ == 'clearOutput':
                window.FindElement('_output_').Update('')
        except Exception as e:
            print(e)
            pass
      
    window.Close()
    
    return
