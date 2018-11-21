import sys
import numpy as np
import itertools as iter
import pandas as pd
import sklearn.metrics as sklm
import music21 as m21

sys.path.append('/Users/marco/Dropbox (Personal)/Musica/Applications/Set Theory/')
from pcsPy import *

from mpi4py import MPI

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

full =  None
z = None
temp = []
for i in range(7,8):
	full,z = pcsDictionary(i,TET=12)
	if rank == 0: temp.append(full)

if rank == 0: 
	full = pd.concat(temp,axis=0)
	full.to_csv('full.csv',index=False)
#print(z)

if rank == 0: pcsNetwork('full.csv',thup=1.5,thdw=0.1)
if rank == 0: pcsEgoNetwork('7-34','full.csv',thup_e=3.0,thdw=0.1)
if rank == 0: 
	subset = extractByString('full.csv','class','-14')
	subset.to_csv('subset.csv',index=False)
