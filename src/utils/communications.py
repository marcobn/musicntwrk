#
# MUSIC𝄞NTWRK
#
# A python library for pitch class set and rhythmic sequences classification and manipulation,
# the generation of networks in generalized music and sound spaces, and the sonification of arbitrary data
#
# Copyright (C) 2018 Marco Buongiorno Nardelli
# http://www.materialssoundmusic.com, mbn@unt.edu
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

import numpy as np
import time

try:
    from .load_balancing import *

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Scatters first dimension of an array of arbitrary length
    def scatter_array ( arr, sroot=0 ):
        # Compute data type and shape of the scattered array on this process
        pydtype = None
        auxlen = None

        # An array to store the size and dimensions of scattered arrays
        lsizes = np.empty((size,3), dtype=int)
        if rank == sroot:
            pydtype = arr.dtype
            auxshape = np.array(list(arr.shape))
            auxlen = len(auxshape)
            lsizes = load_sizes(size, arr.shape[0], np.prod(arr.shape[1:]))

        # Broadcast the data type and dimension of the scattered array
        pydtype = comm.bcast(pydtype, root=sroot)
        auxlen = comm.bcast(auxlen, root=sroot)

        # An array to store the shape of array's dimensions
        if rank != sroot:
            auxshape = np.zeros((auxlen,), dtype=int)

        # Broadcast the shape of each dimension
        for i in np.arange(auxlen):
            auxshape[i] = comm.bcast(auxshape[i], root=sroot)

        comm.Bcast([auxshape, MPI.INT], root=sroot)
        comm.Bcast([lsizes, MPI.INT], root=sroot)

        # Change the first dimension of auxshape to the correct size for scatter
        auxshape[0] = lsizes[rank][2]

        # Initialize aux array
        arraux = np.empty(auxshape, dtype=pydtype)

        # Get the datatype for the MPI transfer
        mpidtype = MPI._typedict[np.dtype(pydtype).char]

        # Scatter the data according to load_sizes
        comm.Scatterv([arr, lsizes[:,0], lsizes[:,1], mpidtype], [arraux, mpidtype], root=sroot)

        return arraux

    # Gathers first dimension of an array of arbitrary length
    def gather_array ( arr, arraux, sroot=0 ):

        # An array to store the size and dimensions of gathered arrays
        lsizes = np.empty((size,3), dtype=int)
        if rank == sroot:
            lsizes = load_sizes(size, arr.shape[0], np.prod(arr.shape[1:]))

        # Broadcast the data offsets
        comm.Bcast([lsizes, MPI.INT], root=sroot)

        # Get the datatype for the MPI transfer
        mpidtype = MPI._typedict[np.dtype(arraux.dtype).char]

        # Gather the data according to load_sizes
        comm.Gatherv([arraux, mpidtype], [arr, lsizes[:,0], lsizes[:,1], mpidtype], root=sroot)

except:
    
    def scatter_array (arr,sroot=0):
        return(arr)

    def gather_array ( arr, arraux, sroot=0 ):
        return