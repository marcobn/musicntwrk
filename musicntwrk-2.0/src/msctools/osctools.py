#
# msctools: my collection of composing and performing tools in python
#
# © 2023 Marco Buongiorno Nardelli
#

from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

import musicntwrk.msctools.cfg as cfg
from .decorators import threading_decorator

class client:
	def __init__(self,address,values,host="127.0.0.1",port=11000):
		self.host = host
		self.port = port
		self.address = address
		self.values = values
		
	def send(self):
		return SimpleUDPClient(self.host,self.port).send_message(self.address,self.values)
	
def server(ip,port):
	def handler(address, *args):
		if address != '/live/song/beat': 
			cfg.data = args
			cfg.addr = address
			if cfg.write:
				print(f"{address}: {args}")
		if address == '/live/song/beat':
			cfg.livebeat = args
			
	dispatcher = Dispatcher()
	dispatcher.map("/live/*", handler)
	server = ThreadingOSCUDPServer((ip, port), dispatcher)
	server.serve_forever()  # Blocks forever
	
def serverSpat(ip,port):
	def handler(address, *args):
		cfg.source_data = args
		cfg.source_addr = address
		if cfg.write:
			print(f"{address}: {args}")
			
	dispatcher = Dispatcher()
	dispatcher.map("/source/*", handler)
	server = ThreadingOSCUDPServer((ip, port), dispatcher)
	server.serve_forever()  # Blocks forever

@threading_decorator
def server_thread(ip,port):
	def handler(address, *args):
		if address != '/live/song/beat': 
			cfg.data = args
			cfg.addr = address
			if cfg.write:
				print(f"{address}: {args}")
		if address == '/live/song/beat':
			cfg.livebeat = args
			
	dispatcher = Dispatcher()
	dispatcher.map("/live/*", handler)
	server = ThreadingOSCUDPServer((ip, port), dispatcher)
	server.serve_forever()  # Blocks forever

@threading_decorator
def serverSpat_thread(ip,port):
	def handler(address, *args):
		cfg.source_data = args
		cfg.source_addr = address
		if cfg.write:
			print(f"{address}: {args}")
			
	dispatcher = Dispatcher()
	dispatcher.map("/source/*", handler)
	server = ThreadingOSCUDPServer((ip, port), dispatcher)
	server.serve_forever()  # Blocks forever