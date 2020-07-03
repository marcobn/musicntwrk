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

import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

def plotOpsHistogram(newvalues,newcounts,fx=15,fy=4):
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
    plt.bar(newvalues,newcounts,width=0.85,color='grey')
