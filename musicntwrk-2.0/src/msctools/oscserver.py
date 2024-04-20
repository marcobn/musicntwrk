#
# msctools: my collection of composing and performing tools in python
#
# © 2023 Marco Buongiorno Nardelli
#

# Hand detection for live processing using mediapipe (version 0.9.1)

# OSC test server

import threading, time
	
from osc4py3.as_eventloop import *
from osc4py3 import oscmethod as osm

def handlerfunction(*args):
	# Will receive message data unpacked in *args
	global data
	data = args
	print(data)

	# Start the system.
osc_startup()

# Make server channels to receive packets.
osc_udp_server("127.0.0.1", 11000, "11000")

# Associate Python functions with message address patterns, using default
# argument scheme OSCARG_DATAUNPACK.
osc_method("/hand/*", handlerfunction)

# define listener function
def listener():
	while True:
		osc_process()
		time.sleep(1/30)

		# Start listening on port 11000
threading.Thread(target=listener,args=()).start()