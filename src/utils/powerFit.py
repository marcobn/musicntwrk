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

import pylab
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import powerlaw


def powerFit(Gx,mode='power_law',xmin=None,xmax=None,linear=False,indeg=True,undir=False,units=None):
    # set-up
    pylab.rcParams['xtick.major.pad']='24'
    pylab.rcParams['ytick.major.pad']='24'
    #pylab.rcParams['font.sans-serif']='Arial'

    from matplotlib import rc
    rc('font', family='sans-serif')
    rc('font', size=14.0)
    rc('text', usetex=False)

    panel_label_font = FontProperties().copy()
    panel_label_font.set_weight("bold")
    panel_label_font.set_size(12.0)
    panel_label_font.set_family("sans-serif")
    
    # fit power law distribution using the powerlaw package
    try:
        if indeg == True:
            data = np.array(sorted([d for n, d in Gx.in_degree()],reverse=True))
        elif indeg == False and undir == False:
            data = np.array(sorted([d for n, d in Gx.out_degree()],reverse=True))
        elif indeg == False and undir == True:
            data = np.array(sorted([d for n, d in Gx.degree()],reverse=True))
    except:
        data = Gx
    ####
    annotate_coord = (-.4, .95)
    fig = plt.figure(figsize=(16,8))
    linf = fig.add_subplot(1,2,1)
    x, y = powerlaw.pdf(data[data>0], linear_bins=True)
    ind = y>0
    y = y[ind]
    x = x[:-1]
    x = x[ind]
    linf.scatter(x, y, color='r', s=5.5)
    powerlaw.plot_pdf(data[data>0], color='b', linewidth=2, linear_bins=linear, ax=linf)
    linf.annotate(" ", annotate_coord, xycoords="axes fraction", fontproperties=panel_label_font)
    linf.set_ylabel(u"p(X)")# (10^n)")
    linf.set_xlabel(units)

    if xmin != None and xmax == None:
        fit = powerlaw.Fit(data,discrete=True,xmin=xmin)
    elif xmax != None and xmin == None:
        fit = powerlaw.Fit(data,discrete=True,xmax=xmax)
    elif xmax != None and xmin != None:
        fit = powerlaw.Fit(data,discrete=True,xmin=xmin,xmax=xmax)
    else:
        fit = powerlaw.Fit(data,discrete=True)

    fitf = fig.add_subplot(1,2,2, sharey=linf)
    fitf.set_xlabel(units)
    powerlaw.plot_pdf(data,color='b', linewidth=2, ax=fitf)
    if mode == 'truncated_power_law':
        fit.truncated_power_law.plot_pdf(color='r', linestyle='--', ax=fitf)
        print('alpha = ',fit.truncated_power_law.alpha)
        print('Lambda = ',fit.truncated_power_law.Lambda)
        print('xmin,xmax = ',fit.xmin, fit.xmax)
        print('Kolmogorov-Smirnov distance = ',fit.truncated_power_law.D)
    elif mode == 'power_law':
        fit.power_law.plot_pdf(color='r', linestyle='--', ax=fitf)
        print('alpha = ',fit.power_law.alpha)
        print('sigma = ',fit.power_law.sigma)
        print('xmin,xmax = ',fit.xmin, fit.xmax)
        print('Kolmogorov-Smirnov distance = ',fit.power_law.D)
    elif mode == 'lognormal':
        fit.lognormal.plot_pdf(color='r', linestyle='--', ax=fitf)
        print('mu = ',fit.lognormal.mu)
        print('sigma = ',fit.lognormal.sigma)
        print('xmin,xmax = ',fit.xmin, fit.xmax)
        print('Kolmogorov-Smirnov distance = ',fit.lognormal.D)
    elif mode == 'lognormal_positive':
        fit.lognormal_positive.plot_pdf(color='r', linestyle='--', ax=fitf)
        print('mu = ',fit.lognormal_positive.mu)
        print('sigma = ',fit.lognormal_positive.sigma)
        print('xmin,xmax = ',fit.xmin, fit.xmax)
        print('Kolmogorov-Smirnov distance = ',fit.lognormal_positive.D)
    elif mode == 'exponential':
        fit.exponential.plot_pdf(color='r', linestyle='--', ax=fitf)
        print('Lambda = ',fit.exponential.Lambda)
        print('xmin,xmax = ',fit.xmin, fit.xmax)
        print('Kolmogorov-Smirnov distance = ',fit.exponential.D)
    elif mode == 'stretched_exponential':
        fit.stretched_exponential.plot_pdf(color='r', linestyle='--', ax=fitf)
        print('Lambda = ',fit.stretched_exponential.Lambda)
        print('beta = ',fit.stretched_exponential.beta)
        print('xmin,xmax = ',fit.xmin, fit.xmax)
        print('Kolmogorov-Smirnov distance = ',fit.stretched_exponential.D)
    else:
        fit = None
        print('mode not allowed')
    return(data,fit)
