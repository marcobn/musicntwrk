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

import collections
import numpy as np
from scipy.optimize import curve_fit


def scaleFreeFit(Gx,indeg=True,imin=0,undir=False,lfit='powerlaw',plot=True):
    # Fits the degree distribution to a power law - check for scale free network
    def curve_fit_log(xdata, ydata) :
        xdata = np.array(xdata)
        ydata = np.array(ydata)
        ind = xdata > 0
        ydata = ydata[ind]
        xdata = xdata[ind]
    #   Fit data to a power law in loglog scale (linear)
        xdata_log = np.log10(xdata)
        ydata_log = np.log10(ydata)
        if lfit == 'powerlaw':
            linlaw = lambda x,a,b: a+x*b
        elif lfit == 'truncatedpowerlaw':
            linlaw = lambda x,a,b,L: a+x*b+L*np.exp(x)
        else:
            print('lfit not defined')
        popt_log, pcov_log = curve_fit(linlaw, xdata_log, ydata_log)
        ydatafit_log = np.power(10, linlaw(xdata_log, *popt_log))
        return (xdata,ydata,popt_log, pcov_log, ydatafit_log)
    try:
        if indeg == True:
            data = np.array(sorted([d for n, d in Gx.in_degree()],reverse=True))
        elif indeg == False and undir == False:
            data = np.array(sorted([d for n, d in Gx.out_degree()],reverse=True))
        elif indeg == False and undir == True:
            data = np.array(sorted([d for n, d in Gx.degree()],reverse=True))
    except:
        data = Gx
    data = data[imin:]
    degreeCount = collections.Counter(data)
    deg, cnt = zip(*degreeCount.items())

    deg,cnt,popt,_,fit = curve_fit_log(deg,cnt)
    
    if plot:
        plt.loglog(deg,cnt, 'bo')
#         fit = 10**popt[0]*np.power(deg,popt[1])
        plt.loglog(deg,fit, 'r-')
        plt.ylabel("Count")
        plt.xlabel("Degree")
        plt.show()
    if lfit == 'powerlaw':
        print('power low distribution - count = ',10**popt[0],'*degree^(',popt[1])
    elif lfit == 'truncatedpowerlaw':
        print('power low distribution - count = ',10**popt[0],'*degree^(',popt[1],'* e^',popt[2])
    else:
        print('lfit not defined')
    return(deg,cnt,fit)
