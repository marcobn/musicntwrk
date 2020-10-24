
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
import sys
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import itertools
import matplotlib.pyplot as plt
#plt.style.use('ggplot') 

from .communications import *
from .load_balancing import *

from ..harmony.changePoint import changePoint

try:
    from mpi4py import MPI
    # initialize parallel execution
    comm=MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    para = True
except:
    rank = 0
    size = 1
    para = False

'''
## Complexity measure based on Difusion Entropy Analysis
Original code by Garland Culbreth
based on a previous version by Garland Culbreth, Jacob Baxley, David Lambert and Paolo Grigolini
Center for Nonlinear Science, University of North Texas

This version by MBN, September 2020

References:
    Nicola Scafetta and Paolo Grigolini, Scaling detection in time series: Diffusion entropy analysis, PHYSICAL REVIEW E 66, 036130 (2002)
    Garland Culbreth, Bruce J.West and Paolo Grigolini, Entropic Approach to the Detection of Crucial Events, Entropy 2019, 21, 178 (2019)
'''

def apply_stripes(data, stripes):
    stripe_size = abs(max(data) - min(data)) / stripes
    rounded_data = data.copy()
    rounded_data = np.where(rounded_data >= 0, np.floor(rounded_data/stripe_size), rounded_data)
    rounded_data = np.where(rounded_data <= 0, np.ceil(rounded_data/stripe_size), rounded_data)
    return rounded_data

def find_events(series,evtype):
    if evtype == 0:
        events = (series[1:]!=series[:-1]).astype(int)
    elif evtype == 1:
        events = [(len(list(y))) for x, y in itertools.groupby(series)]
    else:
        print('type not implemented')
    return events
    
def make_trajectory(events):
    trajectory = np.cumsum(events)
    return trajectory

def entropy(trajectory):
    S = []
    window_lengths = np.arange(1, int(0.25*len(trajectory)), 1)
    for L in window_lengths:
        window_starts = np.arange(0, len(trajectory)-L, 1)
        window_ends = np.arange(L, len(trajectory), 1)
        displacements = trajectory[window_ends] - trajectory[window_starts]
        bin_counts = np.bincount(displacements)
        bin_counts = bin_counts[bin_counts != 0]
        P = bin_counts / np.sum(bin_counts)
        S.append(-np.sum(P * np.log(P)))
    return np.array(S), window_lengths
    
def get_mu(delta):
    mu = 1 + (1 / delta)
    if mu > 3:
        mu = 1 + delta
    return (mu)
    
def plotDEA(s,L,L_slice,fit,stripes,ks,xmin,xmax,save=False,figname=None):
    
    if save and figname == None:
        print('no name for fig file')
        return
    
    mu = get_mu(fit[0])
    fig = plt.figure(figsize = (6, 5))
    plt.plot(L, s, 'bo')
    plt.plot(L_slice, fit[0] * np.log(L_slice) + fit[1], color='red',
             label='$\delta = $'+str(np.round(fit[0], 3)))
    plt.plot([], [], linestyle='',label='$\mu = $'+str(np.round(mu, 3)))
    plt.plot([], [], linestyle='',label='KS test = '+str(np.round(ks, 3)))
    plt.plot([], [], ' ', label="# data points = "+str(len(s)))
    plt.plot([], [], ' ', label="# stripes = "+str(stripes))
    plt.plot([], [], ' ', label="xmin,xmax = "+str(round(xmin,1))+', '+str(round(xmax,1)))
    plt.xscale('log')
    plt.xlabel('$ln(L)$')
    plt.ylabel('$S(L)$')
    plt.grid()
    plt.legend(loc=0)
    if save:
        plt.savefig(figname)
        plt.close()
    else:
        plt.show()
        
def get_scaling(S, L, start, stop):
    S_slice = S[np.logical_and(L>=start,L<=stop)]
    L_slice = L[np.logical_and(L>=start,L<=stop)]
    linlaw = lambda x,a,b: x*a+b
    fit,_ = curve_fit(linlaw, np.log(L_slice), S_slice)
    kstest = stats.ks_2samp(S_slice,fit[0] * np.log(L_slice) + fit[1])
    return (L_slice, fit, kstest[0])
    
def DEA(data, stripes, start=0, stop=None, maxstop=0,evtype=0):
    rounded_data = apply_stripes(data, stripes)
    event_array = find_events(rounded_data,evtype)
    diffusion_trajectory = make_trajectory(event_array)
    S, L = entropy(diffusion_trajectory)
    
    # Find optimal xmin xmax for powerlaw fit (Kolmogorov-Smirnov test)
    if stop == None:
        xstop = []
        xstart = []
        kst = []
        if L[-1] < 30:
            nstep = 1
        else:
            nstep = 30
        if maxstop == 0:
            maxstop = L[-1]
        for x0 in np.linspace(start,maxstop,nstep,dtype=int):
            try:
                xmx = []
                kmn = []
                for x1 in np.arange(x0+3,maxstop,nstep,dtype=int):
                    _,_,km = get_scaling(S, L, x0, x1)
                    xmx.append(x1)
                    kmn.append(km)
                xstop.append(xmx[np.argmin(kmn)])
                xstart.append(x0)
                kst.append(np.min(kmn))
            except:
                pass
        xmin = xstart[np.argmin(kst)]
        xmax = xstop[np.argmin(kst)]
    else:
        xmin = start
        xmax = stop
    
    L_slice,fit,kstest = get_scaling(S, L, xmin, xmax)
    
    return(S,L,L_slice,fit,kstest,xmin,xmax)