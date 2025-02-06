#
# msctools: my collection of composing and performing tools in python
#
# © 2023-2025 Marco Buongiorno Nardelli
#

import PySimpleGUI as sg
from pynput import mouse
import time
import musicntwrk.msctools.cfg as cfg

def on_click(x, y, button, pressed):
    if pressed:
        cfg.CLICK = 1

listener = mouse.Listener(on_click=on_click)
listener.start()

def bigCounterGUI(title = 'BIG COUNTER'):
    
    # Use the theme APIs to set the buttons to blend with background
    sg.theme_button_color((sg.theme_background_color(), sg.theme_background_color()))
    sg.theme_border_width(0)        # make all element flat

    menu_def = [['TIMER', ['Continuous', 'Restart', '---', 'Exit']]]

    # define layout of the rows
    layout= [[sg.Text(title,size=(15,1), font=("Helvetica", 25)),
              sg.Text(' ' * 10),
              sg.Menu(menu_def)],
             [sg.Text(' '*30)],
             [sg.Text(' ' * 10),
              sg.Text('TIMER', size = (8,1),font=("Helvetica", 84), justification='center', key='-TIMER-')],
             [sg.Text(' '*30)],
             [sg.HorizontalSeparator(color='white')],
             [sg.Text(' '*30)],
             [sg.Text(' ' * 10),
              sg.Text('#', size = (4,1),font=("Helvetica", 200), justification='center', key='-COUNTER-')],
             [sg.Text(' '*30)],
             ]

    # Open a form, note that context manager can't be used generally speaking for async forms
    window = sg.Window('COUNTER GUI', layout, default_element_size=(10, 1), font=("Helvetica", 25))
    # Our event loop

    cfg.COUNTER = 0
    restart = True
    first = True
    while True:
        event, values = window.read(timeout=100)  
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == 'Continuous':
            restart = False
            cfg.CLICK  = 0
        elif event == 'Restart':
            restart = True
            cfg.CLICK  = 0
        if event == sg.TIMEOUT_KEY and cfg.CLICK  == 1:
            cfg.COUNTER = cfg.COUNTER + 1
            window['-COUNTER-'].update(str(cfg.COUNTER))
            if restart: time0 = time.time()
            cfg.CLICK  = 0
            cfg.PLAY = True
        if event == sg.TIMEOUT_KEY and cfg.COUNTER == 1 and first == True:
            time0 = time.time()
            first = False
        if event == sg.TIMEOUT_KEY and cfg.COUNTER > 0:
            elapsed_time = time.time() - time0
            minutes, seconds = divmod(int(elapsed_time), 60)
            hours, minutes = divmod(minutes, 60)
            window['-TIMER-'].update(f'{hours:02d}:{minutes:02d}:{seconds:02d}')
