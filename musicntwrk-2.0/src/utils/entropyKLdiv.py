
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
import scipy as sci
import networkx as nx

def entropyKLdiv(Gx,eta):
    
    np.seterr(divide='ignore', invalid='ignore')
    
    # Calculate the entropy of the network and the KL divergency from the human expectation model
    # from Lynn et al. Nature Physics, 16, 965–973 (2020)
    
    if nx.is_directed(Gx):
        degree = np.asarray(Gx.out_degree(),dtype=float)[:,1]
    else:
        degree = np.asarray(Gx.degree(),dtype=float)[:,1]
    G = nx.to_numpy_array(Gx)
    
    # define probability distribution
    P = np.zeros((G.shape[0],G.shape[1]))
    for i in range(degree.shape[0]):
        for j in range(degree.shape[0]):
            P[i,j] = G[i,j]/degree[i]
    eig, vec, _ = sci.linalg.eig(P,left=True)
    n0 = np.argwhere(np.abs(eig-1) < 1.e-10)[0][0]
    pi = np.real(vec[:,n0] / np.sum(vec[:,n0]))
    logP = np.log2(P)
    logP[np.isinf(logP)] = 0
    
    # Entropy
    S = -pi.dot(np.sum(P*logP,axis=1))
    
    # KL divergence
    eta = 0.8
    Phat = (1-eta)*P.dot(np.linalg.inv(np.identity(P.shape[0])-eta*P))
    Phat[np.isnan(Phat)] = 0
    logPhat = np.log2(Phat)
    logPhat[np.isinf(logPhat)] = 0
    KLD = -pi.dot(np.sum((P*(logPhat - logP)),axis=1))
    
    return(S,KLD)