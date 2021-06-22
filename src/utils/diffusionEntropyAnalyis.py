
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
from scipy import stats
from scipy.optimize import curve_fit
import itertools
import matplotlib.pyplot as plt
plt.style.use('ggplot') 

from .communications import *
from .load_balancing import *

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
Original code by Garland Culbreth, Jacob Baxley, David Lambert and Paolo Grigolini
Center for Nonlinear Science, University of North Texas

This version by MBN, August 2020

References:
    Nicola Scafetta and Paolo Grigolini, Scaling detection in time series: Diffusion entropy analysis, PHYSICAL REVIEW E 66, 036130 (2002)
    Garland Culbreth, Bruce J.West and Paolo Grigolini, Entropic Approach to the Detection of Crucial Events, Entropy 2019, 21, 178 (2019)

## Interpreting Results:
 - $\eta$ is the scaling of the time-series process.
 
 - $\mu$ is a complexity index, defined as the power for the inter-event time distribution $1/\tau^{\mu}$ ($\tau$ being inter-event time).

 - For a totally random process, DEA yields $\eta = 0.5.$

 - The closer $\eta$ is to 1, and the closer $\mu$ is to 2, the more complex the data-series is. Those are the critical values of $\eta$ and $\mu$.

 - Two ways of calculating $\mu$ are employed: 
    - For $1 < \mu < 2$: $\mu = 1 + \eta$.
    - For $2 < \mu < 3$: $\mu = 1 + 1/\eta$.
    
 - To choose which calculation is correct, if $\mu = 1 + 1/\eta$ gives a $\mu > 3$ then use $\mu = 1 + \eta$. The program automatically checks this and indicates the recommended $\mu$ with an arrow in the figure legend. If you already have an expectation for what range $\mu$ should be in, e.g. from theoretical arguments, use that. 
'''

def DEAfunction(dell,Tau):
    """
    DEA function
    Called by DEAwithStripes; executes a standard DEA algorithm on the data passed.
    """
    diff = np.zeros((len(Tau),len(Tau)))
    fact = Tau+dell
    fact[fact>np.max(Tau)] = 0
    fact = (fact > 0).astype(int)
    for n in range(1,len(Tau)):
        diff[n-1,:] = np.roll(Tau,-n) - Tau
        diff[n-1,:][diff[n-1,:] <= 0] = dell+1
        diff[n-1,:] = (diff[n-1,:] <= dell).astype(int)
        if np.sum(diff[n-1,:]) == 0:
            break
    XF = np.sum(diff,axis=0)*fact
    XF = XF[XF > 0]
    # This part does the actual DEA computations
    nbins = len(np.arange(-max(XF),max(XF)+1,1))

    counts = np.histogram(XF,nbins)[0]
    counts = np.array(counts[counts != 0])

    P = counts/sum(counts)
    DE = -np.sum(P*np.log(P))  # This is the integral for Shannon Entropy S

    return DE
        
def DEAwithStripes(Data,NumberofStripes,base):
    """
    Applies stripes
    Applies the stripes to the passed Data, then calls DEAfunction to perform DEA on that. 
    """

    Data = np.array(Data)

    DataWidth = abs(max(Data) - min(Data))

    if NumberofStripes > 0:
        StripeSize = DataWidth / NumberofStripes
    else:
        StripeSize = 1

    # Rounding. 
    RoundedData = Data.copy()
    RoundedData = np.where(RoundedData > 0, np.floor(RoundedData/StripeSize), RoundedData)
    RoundedData = np.where(RoundedData < 0, np.ceil(RoundedData/StripeSize), RoundedData)
    
    # Finds the events
    if NumberofStripes > 0:
        Tau = [(len(list(y))) for x, y in itertools.groupby(RoundedData)]
    else:
        Tau = [(len(list(y))) for x, y in itertools.groupby(np.sign(RoundedData))]

    # Sums the Taus (distance between consecutive events) and defines the number of windows
    Tau = np.cumsum(Tau)
    l = np.floor(np.log(Tau[-2])/np.log(base))

    # Makes the window lengths and windows
    Delh = base**np.linspace(1,int(l),int(l))
    
    DE = np.zeros(len(Delh))

    if para: comm.Barrier()

    ini,end = load_balancing(size, rank, len(Delh))
    nsize = end-ini
    
    delhx = scatter_array(Delh)
    dex = scatter_array(DE)
    
    # Call the DEA function to execute on the data
    for q in range(nsize):
        try:
            dex[q] = DEAfunction(delhx[q], Tau)
        except:
            dex[q] = dex[q-1]


    gather_array(DE,dex)
    if para: comm.bcast(DE)
    
    # This is where the x-axis gets the log scale from. Don't do a semilog plot or you're logging twice.
    de = np.log(Delh)

    return (de, DE)

def curve_fit_log(xdata, ydata,lfit='powerlaw') :
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    ind = xdata > 0
    ydata = ydata[ind]
    xdata = xdata[ind]
    # Fit data to a power law in loglog scale (linear)
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
    kstest = stats.ks_2samp(ydata,ydatafit_log)
    return (xdata,ydata,popt_log, pcov_log, ydatafit_log,kstest)

def best_xmax(de,DE,xmax,lfit='powerlaw',st=0):
    x = np.exp(de[np.logical_and(de>=st, de<=xmax)])
    y = np.exp(DE[np.logical_and(de>=st, de<=xmax)])
    _,_,_,_,_,kstest = curve_fit_log(x,y,lfit='powerlaw')
    return(xmax,kstest[0])

def DEA(Data,NumberofStripes=0,base=1.025,plots=False,save=True,lfit='powerlaw',st=0,stp=None):

    """Computes the scaling.
    Calls the DEAwithStripes to perform DEA, and outputs plots. 
    Calculates the slope of the linear region of the DEA plot and overlays 
    a line. 
    args:
        Data -- 1D array. Your data. 
        NumberofStripes -- int. How many stripes to use.
        base -- float: 1 < base < 2. Determines number of points in final DEA plot. Smaller gives more points.
    returns:
        DE, de = Diffusion Entropy, window size
        eta (scaling parameter
        if plots == True - the plot showing the DEA and fit line.
    """
                
    de, DE = DEAwithStripes(Data,NumberofStripes,base)

    # The numbers st and stp set the interval of the DEA over which to perform the fitting to get the scaling

    if rank == 0:
#        defitInterval = de[np.where(np.logical_and(de>=st, de<=stp))]
#        DEfitInterval = DE[np.where(np.logical_and(de>=st, de<=stp))]
#        
#
#        slope, intercept, r_value, p_value, std_err = stats.linregress(defitInterval, DEfitInterval)
        
        # Find optimal xmin xmax for powerlaw fit (Kolmogorov-Smirnov test)

        if stp == None:
            xstop = []
            xstart = []
            kst = []
            for s in np.linspace(st,de[-1],30):
                try:
                    xmx = []
                    kmn = []
                    for n in np.linspace(s+1,de[-1],100):
                        xm,km = best_xmax(de,DE,n,lfit,st=s)
                        xmx.append(xm)
                        kmn.append(km)
                    xstop.append(xmx[np.argmin(kmn)])
                    xstart.append(s)
                    kst.append(np.min(kmn))
                except:
                    pass
            xmin = xstart[np.argmin(kst)]
            xmax = xstop[np.argmin(kst)]
        else:
            xmin = st
            xmax = stp
                
        x = np.exp(de[np.logical_and(de>=xmin, de<=xmax)])
        y = np.exp(DE[np.logical_and(de>=xmin, de<=xmax)])
        xfit,_,popt,_,fit,kstest = curve_fit_log(x,y,lfit=lfit)
        
        return(de,DE,xfit,fit,popt,xmin,xmax,kstest,NumberofStripes)

    else:
        
        de=DE=xfit=fit=popt=xmin=xmax=ktest=NumberofStripes = 1
        return(de,DE,xfit,fit,popt,xmin,xmax,kstest,NumberofStripes)
        
def plotDEA(de,DE,xfit,fit,popt,xmin,xmax,kstest,NumberofStripes,save=False,figname=None):
    
    if rank == 0:

        if save and figname == None:
            print('no name for fig file')
            return
        
        eta = popt[1]  # This parameter is the scaling
        mu_1 = 1 + 1/eta  # Applies when 2 < mu < 3
        mu_1_label = '$\\mu=1+1/\\delta=$'+ str(round(mu_1,3))
        mu_2 = 1 + eta  # Applies when 1 < mu < 2
        mu_2_label = '$\\mu=1+\\delta=$'+ str(round(mu_2,3))

        if mu_1 <= 3:
            mu_1_label =  '\u2192 '+ mu_1_label
        else:
            mu_2_label = '\u2192 ' + mu_2_label

        # Plot the results
        plt.figure(figsize = (8, 6))
        if NumberofStripes > 0:
            plt.title('DEA with stripes')
        else:
            plt.title('Ordinary DEA')
        plt.plot(de,DE,'.', label = 'DEA')
        plt.plot(np.log(xfit),np.log(fit), label='$\delta=$'+str(round(eta,3)))
        plt.xlabel('ln(l)')
        plt.ylabel('S(l)')
        if NumberofStripes > 0:
            plt.plot([], [], ' ', label= mu_1_label)
            plt.plot([], [], ' ', label= mu_2_label)
        plt.plot([], [], ' ', label="# stripes = "+str(NumberofStripes))
        plt.plot([], [], ' ', label="xmin,xmax = "+str(round(xmin,3))+','+str(round(xmax,3)))
        plt.plot([], [], ' ', label="KS test = "+str(round(kstest[0],3)))
        plt.legend()
        if save:
            plt.savefig(figname)
        else:
            plt.show()