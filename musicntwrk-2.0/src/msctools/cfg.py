#
# msctools: my collection of composing and performing tools in python
#
# © 2025 Marco Buongiorno Nardelli
#

import numpy as np

addr = None 
data = None
source_addr = None 
source_data = None

write = False
tempo = 60.0

NTRACK = 128
NSOURCE = 128
stop = np.array([False]*NTRACK)
stop_source = np.array([False]*NSOURCE)
sleep = np.array([0.0]*NTRACK)
beat = np.array([1.0]*NTRACK)
pan = [0.5]*NTRACK
gain = [1.0]*NTRACK

TICK = 0.15
CLOCK = TICK/10
PORT = 11000
COMM_A = 18080
COMM_B = 18081
COMM_C = 18082
COMM_D = 18083
HOST = "127.0.0.1"

CLICK = 0
COUNTER = 0
PLAY = False

CLEANALL = False
MASTER_STOP = False
SCENE = None
