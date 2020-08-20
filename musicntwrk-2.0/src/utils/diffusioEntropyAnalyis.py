
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
import matplotlib.pyplot as plt
plt.style.use('ggplot') 

def DEAfunction(dell,Tau):
    """
    DEA function
    Called by DEAwithStripes; executes a standard DEA algorithm on the data passed.
    """
    p = len(Tau)
    XF = np.zeros(p)

    TauMax = Tau[-1]
    agedx = -1
    for i in range(p):
        if Tau[i] + dell < TauMax:
            m = i
            while (Tau[m + 1] < Tau[i] + dell) and (m + 1 <= p):
                XF[i] = XF[i] + 1
                m = m + 1

    XF = np.array(XF)
    XF = XF[XF != 0]
    # This part does the actual DEA computations
    nbins = len(np.arange(-max(XF),max(XF)+1,1))

    counts = np.histogram(XF,nbins)[0]
    counts = np.array(counts[counts != 0])

    P = counts/sum(counts)
    DE = -sum(P*np.log(P))  # This is the integral for Shannon Entropy S

    return DE, list(counts)
    
def DEAwithStripes(Data,NumberofStripes=int(max(Data)),base=1.025,plots=False,numbers=True):
    """Applies stripes

    Applies the stripes to the passed Data, then calls DEAfunction to perform DEA on that. 
    These plots do not show the scaling. See next cell for that.

    args:
        Data -- 1D array. Your data. 
        NumberofStripes -- int. How many stripes to use.
        base -- float: 1 < base < 2. Determines number of points in final DEA plot. Smaller gives more points.
        plots -- bool. Print plots or not. 
        numbers -- bool. Print numbers or not.

    returns:
        Data -- array. The data.
        NumberofStripes -- int. The number of stripes
        stripes -- line_collection. The y-axis locations of the stripes
        de -- float. ln(l), of the window lengths l.
        DE -- float. S(l), shannon entropy. 
    """
    # Data = xi
    Data = 1*np.array(Data)

    MaxData = max(Data)
    MinData = min(Data)
    DataWidth = abs(MaxData - MinData)

    # NumberofStripes = 10  # 4.4.20 (Garland): Commented this out so it doesn't overwrite the passed value.
    if NumberofStripes > 0:
        StripeSize = DataWidth / NumberofStripes
    else:
        StripeSize = 1

    N = len(Data)
    Delh = StripeSize
    # RoundedData = np.round(Data/Delh) # allows non-integer steps

    # An alternate way of doing the rounding. 
    RoundedData = np.zeros(len(Data))
    for i in range(len(Data)):
        if Data[i] > 0:
            RoundedData[i] = np.floor(Data[i]/Delh)
        elif Data[i] < 0:
            RoundedData[i] = np.ceil(Data[i]/Delh)
    
    # 4.4.20 (Garland): I think had a version of this loop that got it down to 1 line
    # of code, but it's only on my office computer and not in a cloud backup..
    # Finds the events
    Tau = np.zeros(N)
    k = 0
    Tau[0]=1
    if NumberofStripes > 0:
        for i in range(1,N):
            if (RoundedData[i] == RoundedData[i-1]):
                Tau[k] = Tau[k] + 1
            else:
                k = k + 1
                Tau[k] = 1
    else:
        for i in range(1,N):
            if (np.sign(RoundedData[i]) == np.sign(RoundedData[i-1])):
                Tau[k] = Tau[k] + 1
            else:
                k = k + 1
                Tau[k] = 1

    # Sums the Taus (distance between consecutive events)
    Tau = Tau[:k+1]
    WaitingTimes = Tau
    Tau = np.cumsum(Tau)
    l = np.floor(np.log(Tau[k-1])/np.log(base))

    # Makes the window lengths and windows
    Delh = np.zeros(int(l))
    for i in range(int(l)):
        Delh[i] = base**(i+1)
    DE = np.zeros(len(Delh))
                
    # Call the DEA function to execute on the data
    for q in range(len(Delh)):
        DEF, counts = DEAfunction(Delh[q], Tau)
        DE[q] = DEF
            
    # This is where the x-axis gets the log scale from. Don't do a semilog plot or you're logging twice.
    de = np.zeros(len(Delh))
    for t in range(len(Delh)):
        de[t] = np.log(Delh[t])        

    if NumberofStripes > 0:
        stripes = np.linspace(min(Data),max(Data),num = NumberofStripes)
    else:
        stripes = 0
    
    if plots == True:
        plt.figure(figsize = (18, 6))

        plt.subplot(121)
        plt.title('Data')
        plt.plot(Data)
        if NumberofStripes > 0:
            plt.hlines(y=stripes, xmin=0, xmax=len(Data))
        plt.xlabel('t')
        plt.ylabel('$\\xi$(t)')
        plt.subplot(122)
        if NumberofStripes > 0:
            plt.title('DEA with stripes')
        else:
            plt.title('Ordinary DEA')
        plt.plot(de[:-1],DE[:-1],'.')
        plt.xlabel('ln(l)')
        plt.ylabel('S(l)')
        plt.show()
    
    if numbers == True:
        return Data, NumberofStripes, stripes, de, DE, WaitingTimes
        
def DEA(DEAwithStripes,st=0,stp=1):
    """Computes the scaling.

    Calls the DEAwithStripes to perform DEA, and outputs plots. 
    Calculates the slope of the linear region of the DEA plot and overlays 
    a line. 

    args:
        DEAwithStripes -- function. Calls DEAwithStripes to perform the
        analysis. 
        st -- float, 0 < st < 1. Fraction of DEA from 0 to begin 
        linear fitting for scaling calculation. Must be < stp.
        stp -- float, 0 < stp < 1. Fraction of DEA from 0 to finish 
        linear fitting for scaling calculation. Must be > st.

    returns:
        Two plots (subplotted) one showing data and overlaid stripes, the other 
        showing the DEA and fit line.
    """
    Data = DEAwithStripes[0]
    NumberofStripes = DEAwithStripes[1]
    stripes1 = DEAwithStripes[2]
    de = DEAwithStripes[3]
    DE = DEAwithStripes[4]
    # The numbers set the interval of the DEA over which to perform the fitting to get the scaling

    # Change 1: to a larger number for larger values, straight line
    DEfitInterval = DE[int(np.round(len(DE)*st))-1:int(np.round(len(DE)*stp))]
    defitInterval = de[int(np.round(len(de)*st))-1:int(np.round(len(de)*stp))]

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
        
    return(de, DE)