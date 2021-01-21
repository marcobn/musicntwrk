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

from ..utils.opsHistogram import opsHistogram
from ..utils.generalizedOpsHistogram import generalizedOpsHistogram

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

#def plotOpsHistogram(newvalues,newcounts,fx=15,fy=4):
def plotOpsHistogram(edges,fx=15,fy=4,show=True):
    
    values = edges['Label'].value_counts().keys().tolist()
    counts = edges['Label'].value_counts().tolist()
    counts /= np.sum(counts)*0.01

    newvalues, newcounts,pal_dict,dist = generalizedOpsHistogram(values,counts)
    idx = np.argwhere(newcounts)
    if show:
        plt.rcParams['font.family'] = 'arial'
        plt.rcParams['axes.edgecolor']='#333F4B'
        plt.rcParams['axes.linewidth']=1.5
        plt.rcParams['xtick.color']='#333F4B'
        plt.rcParams['ytick.color']='#333F4B'
    
        plt.figure(figsize=(fx,fy))
    
        plt.ylabel('Percentage',fontsize=24, fontweight='black', color = '#333F4B')
        plt.yticks(fontsize=18,fontweight='black', color = '#333F4B')
        plt.setp(plt.gca().get_xticklabels(), rotation=-90, horizontalalignment='center',fontsize=10, 
                fontweight='black', color = '#000000')
    #     plt.xticks([])
        plt.bar(newvalues[idx][:,0],newcounts[idx][:,0],width=0.85,color='grey')
    return(newvalues[idx],newcounts[idx])