
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

def DEA(Data,NumberofStripes=0,base=1.025,plots=False,st=0,stp=1):

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

        N = len(Data)

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
        Delh = np.zeros(int(l))
        for i in range(int(l)):
            Delh[i] = base**(i+1)
        
        DE = np.zeros(len(Delh))

        if para: comm.Barrier()

        ini,end = load_balancing(size, rank, len(Delh))
        nsize = end-ini
        
        delhx = scatter_array(Delh)
        dex = scatter_array(DE)
        
        # Call the DEA function to execute on the data
        for q in range(nsize):
            dex[q] = DEAfunction(delhx[q], Tau)

        gather_array(DE,dex)
        if para: comm.bcast(DE)
        
        # This is where the x-axis gets the log scale from. Don't do a semilog plot or you're logging twice.
        de = np.log(Delh)

        return (de, DE)
            
    # Main body
    
    de, DE = DEAwithStripes(Data,NumberofStripes,base)

    # The numbers st and stp set the interval of the DEA over which to perform the fitting to get the scaling

    if rank == 0:
        defitInterval = de[np.where(np.logical_and(de>=st, de<=stp))]
        DEfitInterval = DE[np.where(np.logical_and(de>=st, de<=stp))]
        

        slope, intercept, r_value, p_value, std_err = stats.linregress(defitInterval, DEfitInterval)

        eta = slope  # This parameter is the scaling
        mu_1 = 1 + 1/eta  # Applies when 2 < mu < 3
        mu_1_label = '$\\mu=1+1/\\eta=$'+ str(round(mu_1,3))
        mu_2 = 1 + eta  # Applies when 1 < mu < 2
        mu_2_label = '$\\mu=1+\\eta=$'+ str(round(mu_2,3))

        if mu_1 <= 3:
            mu_1_label =  '\u2192 '+ mu_1_label
        else:
            mu_2_label = '\u2192 ' + mu_2_label

        # Plot the results
        if plots:
            plt.figure(figsize = (8, 6))
            if NumberofStripes > 0:
                plt.title('DEA with stripes')
            else:
                plt.title('Ordinary DEA')
            plt.plot(de[:-1],DE[:-1],'.', label = 'DEA')
            plt.plot(defitInterval, intercept + slope*defitInterval, label='$\eta=$'+str(round(slope,3)))
            plt.xlabel('ln(l)')
            plt.ylabel('S(l)')
            if NumberofStripes > 0:
                plt.plot([], [], ' ', label= mu_1_label)
                plt.plot([], [], ' ', label= mu_2_label)
            plt.plot([], [], ' ', label="std err ="+str(round(std_err,3)))
            plt.legend()
            plt.show()
        
        return(de, DE, eta)
        
    else:
        de = 1
        DE = 1
        eta = 1
        return(de, DE, eta)