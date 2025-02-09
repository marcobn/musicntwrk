#
# msctools: my collection of composing and performing tools in python
#
# © 2023 Marco Buongiorno Nardelli
#

# utility functions

import glob
import musicntwrk.msctools.cfg as cfg
from .decorators import threading_decorator
from tqdm.notebook import tqdm
import time

def importSoundfiles(dirpath='./',filepath='./'):
	# reading wavefiles
	try:
		fil = [None]*len(glob.glob(dirpath+filepath))
		n=0
		for file in sorted(glob.glob(dirpath+filepath)):
			fil[n] = file
			n += 1
	except:
		print('error in file reading',dirpath+filepath)
		pass
	return(fil)

@threading_decorator
def timer(sec):
    nstep = sec/cfg.TICK
    for i in tqdm(range(int(nstep))):
        time.sleep(cfg.TICK)