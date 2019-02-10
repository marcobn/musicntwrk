#
# rhythmPy
#
# A python library for rhythmic sequence classification and manipulation, and network construction and analysis
#
# Copyright (C) 2018 Marco Buongiorno Nardelli
# http://www.materialssoundmusic.com, mbn@unt.edu
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

import sys,re,time,os
import numpy as np
from functools import reduce
import fractions as fr
from math import gcd
import pandas as pd
import sklearn.metrics as sklm
import networkx as nx
import community as cm
import music21 as m21
import itertools as itr
from mpi4py import MPI

from communications import *
from load_balancing import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class RHYTHMSeq:

    def __init__(self,rseq,REF='e',ORD=False):
        '''
        •	rseq (int)– rhythm sequence as list of strings/fractions/floats
        •	REF = reference duration for prime form (str)
        '''
        dur_dict = {'w':fr.Fraction(1,1),'h':fr.Fraction(1,2),'q':fr.Fraction(1,4),\
                    'e':fr.Fraction(1,8),'s':fr.Fraction(1/16),'t':fr.Fraction(1,32),\
                    'wd':fr.Fraction(3,2),'hd':fr.Fraction(3,4),'qd':fr.Fraction(3,8),\
                    'ed':fr.Fraction(3,16),'sd':fr.Fraction(3,32),\
                    'qt':fr.Fraction(1,6),'et':fr.Fraction(1,12),'st':fr.Fraction(1,24),\
                    'qq':fr.Fraction(1,5),'eq':fr.Fraction(1,10),'sq':fr.Fraction(1,20)}
        if np.array(rseq).dtype == '<U1' or np.array(rseq).dtype == '<U2':
            for n in range(len(rseq)):
                rseq[n] = dur_dict[rseq[n]]
        elif np.array(rseq).dtype == 'O':
            pass
        else:
            for n in range(len(rseq)):
                rseq[n] = fr.Fraction(rseq[n])
        if ORD:
            self.rseq = np.sort(np.asarray(rseq))
        else:
            self.rseq = np.asarray(rseq)
        self.dict = dur_dict
        self.REF = dur_dict[REF]

    def normalOrder(self):
        '''
        •	Sort the rhythm sequence in ascending order of duration
        '''
        return(np.sort(self.rseq))


    def augment(self,t='e'):
        '''
        •	Augmentation by t (string) units
        '''
        return((self.rseq+self.dict[t]))
    
    def diminish(self,t='e'):
        '''
        •	Diminution by t (string) units (might reduce the length of the sequence)
        '''
        diminish = []
        for n in range(len(self.rseq)):
            if self.rseq[n] > self.dict[t]:
                diminish.append(self.rseq[n] - self.dict[t])
        return(np.asarray(diminish))
    
    def retrograde(self):
        '''
        •	retrograde operation
        '''
        return(np.flip(self.rseq))

    def isNonRetro(self):
        '''
        •	check if the sequence is not retrogradable
        '''
        if np.sum((self.rseq - np.flip(self.rseq))**2) == 0:
            return(True)
        else:
            return(False)
        
    def floatize(self):
        '''
        •	transform the sequence of fractions in floats
        '''
        a = []
        for n in range(len(self.rseq)):
            a.append(self.rseq[n].numerator/self.rseq[n].denominator)
        return(np.asarray(a))
    
    def reduce2GCD(self):
        '''
        •	reduce the series of fractions to Greatest Common Divisor
        '''
        def lcm(a, b):
            return int(a * b / gcd(a, b))
        def lcms(numbers): 
            return(reduce(lcm, numbers))
        
        reduceGCD = []
        den = []
        for n in range(len(self.rseq)):
            den.append(self.rseq[n].denominator)
        commonden = reduce(lcm, den)
        for n in range(len(self.rseq)):
            reduceGCD.append(fr.Fraction(int(self.rseq[n].numerator*commonden/self.rseq[n].denominator),commonden))
        reduceGCD = np.sort(reduceGCD)    
        return(reduceGCD)
    
    def primeForm(self):
        '''
        •	reduce the series of fractions to prime form
        '''
        scale = self.REF*self.normalOrder()[0].denominator/self.normalOrder()[0].numerator
        return(self.normalOrder()*scale)
    
    def durationVector(self,lseq=None):
        '''
        •	 total relative duration ratios content of the sequence
        '''
        nseq = int((len(self.rseq)**2-len(self.rseq))/2)
        durv = np.zeros(nseq,dtype=float)
        n = 0
        for i in range(len(self.rseq)):
            for j in range(i+1,len(self.rseq)):
                durv[n] = np.abs(self.rseq[i]-self.rseq[j])
                n += 1
        bins = []
        if lseq == None:
            lseq = [fr.Fraction(1/8),fr.Fraction(2/8),fr.Fraction(3/8),fr.Fraction(4/8),\
                    fr.Fraction(5/8),fr.Fraction(6/8),fr.Fraction(7/8),fr.Fraction(8/8),fr.Fraction(9/8)]
        bins = np.sort(np.asarray(lseq))

        return(np.histogram(durv,bins)[0],str(bins[:(bins.shape[0]-1)]).replace('Fraction','').replace(', ','/')\
                            .replace('(','').replace(')','').replace('\n','').replace('[','').replace(']',''))
                            
    def rIntervalVector(self,lseq=None):
        '''
        •	 inter-onset duration interval content of the sequence
        '''
        durv = []
        durv.append(np.abs(self.rseq)) 
        durv.append(np.abs(self.rseq+np.roll(self.rseq,-1)))
        durv = np.asarray(durv)
        durv = np.reshape(durv,durv.shape[0]*durv.shape[1])

        bins = []
        if lseq == None:
            lseq = [fr.Fraction(1/8),fr.Fraction(2/8),fr.Fraction(3/8),fr.Fraction(4/8),\
                    fr.Fraction(5/8),fr.Fraction(6/8),fr.Fraction(7/8),fr.Fraction(8/8),fr.Fraction(9/8)]
        bins = np.sort(np.asarray(lseq))

        return(np.histogram(durv,bins)[0],str(bins[:(bins.shape[0]-1)]).replace('Fraction','').replace(', ','/')\
                            .replace('(','').replace(')','').replace('\n','').replace('[','').replace(']',''))
    
    def displayRhythm(self,xml=False,prime=False):
        '''
        •	Display rhythm sequence in score in musicxml format. If prime is True, display the prime form.
        '''
        m = m21.stream.Measure()
        if prime: 
            for l in range(self.rseq.shape[0]):
                n = m21.note.Note(60)
                n.duration = m21.duration.Duration(4*self.primeForm()[l])
                n.beams.fill('32nd', type='start')
                m.append(n)   
        else:
             for l in range(self.rseq.shape[0]):
                n = m21.note.Note(60)
                n.duration = m21.duration.Duration(4*self.rseq[l])
                n.beams.fill('32nd', type='start')
                m.append(n)  
        m.append(m21.meter.SenzaMisuraTimeSignature('0'))
        m.show()
        if xml: m.show('musicxml')
        return
    
########### Network functions ###########

def rhythmDictionary(Nc,a=None,REF='e'):

    '''
    •	Generate the dictionary of all possible rhythmic sequences of Nc length in a generalized meter 
        space of N durations
    •	Nc (int)– cell length
    •	a is the list of durations in the rhythm sequence (str)
    •	returns the dictionary as pandas DataFrame and indicates all non retrogradable cells
    '''
    name = []
    prime = []
        
    a = RHYTHMSeq(a).normalOrder()
    a = RHYTHMSeq(a).floatize()
    a = np.unique(np.asarray(list(itr.combinations(a,Nc))),axis=0)
        
    # put all cells in prime/normal order form

    s = []
    v = []
    for i in range(a.shape[0]):
        p = RHYTHMSeq(a[i,:].tolist(),REF)
        s.append(p.normalOrder())
        v.append(p.durationVector()[0])
    s = np.asarray(s)
    vector = np.asarray(v)

    for i in range(a.shape[0]):
        name.append(str(Nc)+'-'+str(i+1))
        prime.append(str(s[i,:]).replace('Fraction','').replace(', ','/')\
                            .replace('(','').replace(')','').replace('\n','').replace('[','').replace(']',''))
        
        
    dictionary = None
    
    # find those that can be made non retrogradable
    
    for n in range(a.shape[0]):
        perm = np.asarray(list(itr.permutations(a[n,:],a.shape[1])))
        perm = np.unique(perm,axis=0)
        for i in range(perm.shape[0]):
            if RHYTHMSeq(perm[i].tolist(),REF).isNonRetro():
                name[n] = name[n]+'N'
    
    # find those that are Z-related (have same duration vector)
    
    ZrelT = None
    if rank == 0:
        # find pc sets in Z relation
        u, indeces = np.unique(vector, return_inverse=True,axis=0)
        ZrelT = []
        for n in range(u.shape[0]):
            if np.array(np.where(indeces == n)).shape[1] != 1:
                indx = np.array(np.where(indeces == n))[0]
                Zrel = []
                for m in range(indx.shape[0]):
                    name[indx[m]] = name[indx[m]]+'Z'
                    Zrel.append(name[indx[m]])
                ZrelT.append(Zrel)
                    
    # Create dictionary of rhythmic cells
    reference = []
    for n in range(len(name)):
        entry = [name[n],prime[n],
                np.array2string(vector[n,:],separator=',').replace(" ","")]
        reference.append(entry)

    dictionary = pd.DataFrame(reference,columns=['cell','r-seq','d-vec'])
    dictionary.drop_duplicates(subset=['r-seq', 'd-vec'])
    
    return(dictionary,ZrelT)

def rhythmPDictionary(N,Nc,REF='e'):

    '''
    •	Generate the dictionary of all possible rhythmic sequences from all possible groupings of N 
        REF durations
    •	N (int)– number of REF units
    •	Nc cardinality of the grouping
    •	returns the dictionary as pandas DataFrame and indicates all non retrogradable cells
    '''
    name = []
    prime = []
    
    r = [REF] * N
    r = RHYTHMSeq(r)
    idx = np.linspace(0,N-1,N,dtype=int)
    
    subidx = list(Sublists(idx))
    seq=[]
    for l in range(len(subidx)):
        subseq = []
        for i in range(len(subidx[l])):
            aux = 0
            for k in range(len(subidx[l][i])):
                aux += r.rseq[subidx[l][i][k]]
            subseq.append(aux)
        seq.append(subseq)
    seq = Remove(seq)
    
    # select groupings with requested cardinality
    seqx = []
    for n in range(len(seq)):
        if len(seq[n]) == Nc:
            seqx.append(seq[n])
    seq = seqx
        
    # put all cells in prime/normal order form
    s = []
    v = []
    for i in range(len(seq)):
        p = RHYTHMSeq(seq[i][:],REF)
        s.append(p.normalOrder())
        v.append(p.rIntervalVector()[0])
    s = np.asarray(s)
    vector = np.asarray(v)

    for i in range(len(seq)):
        name.append(str(Nc)+'-'+str(i+1))
        prime.append(str(s[i,:]).replace('Fraction','').replace(', ','/')\
                            .replace('(','').replace(')','').replace('\n','').replace('[','').replace(']',''))
        
    dictionary = None
    
    # find those that can be made non retrogradable
    
    seq = np.asarray(seq)
    for n in range(seq.shape[0]):
        perm = np.asarray(list(itr.permutations(seq[n,:],Nc)))
        perm = np.unique(perm)
        for i in range(perm.shape[0]):
            if RHYTHMSeq(perm[i],REF).isNonRetro():
                name[n] = name[n]+'N'
    
    # find those that are Z-related (have same duration vector)
    
    ZrelT = None
    if rank == 0:
        # find pc sets in Z relation
        u, indeces = np.unique(vector, return_inverse=True,axis=0)
        ZrelT = []
        for n in range(u.shape[0]):
            if np.array(np.where(indeces == n)).shape[1] != 1:
                indx = np.array(np.where(indeces == n))[0]
                Zrel = []
                for m in range(indx.shape[0]):
                    name[indx[m]] = name[indx[m]]+'Z'
                    Zrel.append(name[indx[m]])
                ZrelT.append(Zrel)
                    
    # Create dictionary of rhythmic cells
    reference = []
    for n in range(len(name)):
        entry = [name[n],prime[n],
                np.array2string(vector[n,:],separator=',').replace(" ","")]
        reference.append(entry)

    dictionary = pd.DataFrame(reference,columns=['cell','r-seq','r-vec'])
    dictionary.drop_duplicates(subset=['r-seq', 'r-vec'])
    
    return(dictionary,ZrelT)

    
def rhythmNetwork(input_csv,thup=1.5,thdw=0.0,distance='euclidean',prob=1,w=False):
    
    '''
    •	generate the network of rhythmic cells based on distances between duration vectors
    •	input_csv (str)– file containing the dictionary generated by rhythmNetwork
    •	thup, thdw (float)– upper and lower thresholds for edge creation
    •	distance (str)– choice of norm in the musical space, default is 'euclidean'
    •	prob (float)– if ≠ 1, defines the probability of acceptance of any given edge
    •	in output it writes the nodes.csv and edges.csv as separate files in csv format
    '''

    # Create network of rhythmic cells from the rhythmDictionary 
    
    df = pd.read_csv(input_csv)
    df = np.asarray(df)
    
    dim = np.asarray(list(map(int,re.findall('\d+',df[0,2])))).shape[0]

    # write csv for nodes
    dnodes = pd.DataFrame(df[:,0],columns=['Label'])
    if w: dnodes.to_csv('nodes.csv',index=False)
    comm.Barrier()
    
    # find edges according to a metric
    
    vector = np.zeros((df[:,2].shape[0],dim))
    for i in range(df[:,2].shape[0]):
        vector[i]  = np.asarray(list(map(int,re.findall('\d+',df[i,2]))))
    N = vector.shape[0]
    index = np.linspace(0,vector.shape[0]-1,vector.shape[0],dtype=int)
    # parallelize over interval vector to optimize the vectorization in sklm.pairwise_distances
    ini,end = load_balancing(size, rank, N)
    nsize = end-ini
    vaux = scatter_array(vector)
    #pair = sklm.pairwise_distances(vaux,vector,metric=distance)
    dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
    for i in range(nsize):
        tmp = pd.DataFrame(None,columns=['Source','Target','Weight'])
        tmp['Source'] = (i+ini)*np.ones(vector.shape[0],dtype=int)[:]
        tmp['Target'] = index[:]
        tmp['Weight'] = np.sqrt(np.sum((vaux[i,:]-vector[:,:])**2,axis=1))
        tmp = tmp.query('Weight<='+str(thup)).query('Weight>='+str(thdw))
        if prob == 1:
            dedges = dedges.append(tmp)
        else:
            np.random.seed(int(time.time()))
            if np.random.rand() >= prob:
                dedges = dedges.append(tmp)
            else:
                pass
            
    dedges = dedges.query('Weight<='+str(thup)).query('Weight>='+str(thdw))
    dedges['Weight'] = dedges['Weight'].apply(lambda x: 1/x)
    # do some cleaning
    cond = dedges.Source > dedges.Target
    dedges.loc[cond, ['Source', 'Target']] = dedges.loc[cond, ['Target', 'Source']].values
    dedges = dedges.drop_duplicates(subset=['Source', 'Target'])

    # write csv for partial edges
    dedges.to_csv('edges'+str(rank)+'.csv',index=False)
    comm.Barrier()
    
    if size != 1 and rank == 0:
        dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
        for i in range(size):
            tmp = pd.read_csv('edges'+str(i)+'.csv')
            dedges = dedges.append(tmp)
            os.remove('edges'+str(i)+'.csv')
        # do some cleaning
        cond = dedges.Source > dedges.Target
        dedges.loc[cond, ['Source', 'Target']] = dedges.loc[cond, ['Target', 'Source']].values
        dedges = dedges.drop_duplicates(subset=['Source', 'Target'])
        # write csv for edges
        if w: dedges.to_csv('edges.csv',index=False)
    elif size == 1:
        os.rename('edges'+str(rank)+'.csv','edges.csv')

    return(dnodes,dedges)
    
def rhythmDistance(a,b,distance='euclidean'):
    '''
    •	calculates the minimal duration distance between two rhythmic cells of same cardinality (bijective)
    •	a,b (str) – rhythmic cells
    '''
    a = np.asarray(a.normalOrder())
    b = np.asarray(b.normalOrder())
    n = a.shape[0]
    if a.shape[0] != b.shape[0]:
        print('dimension of arrays must be equal')
        sys.exit()
    iTET = np.vstack([np.identity(n,dtype=int),-np.identity(n,dtype=int)])
    iTET = np.vstack([iTET,np.zeros(n,dtype=int)])
    diff = np.zeros(2*n+1,dtype=float)
    v = []
    for i in range(2*n+1):
        r = np.sort(b - iTET[i])
        diff[i] = sklm.pairwise_distances(a.reshape(1, -1),r.reshape(1, -1),metric=distance)[0]
    imin = np.argmin(diff)
    return(fr.Fraction(diff.min()))

def rLeadNetwork(input_csv,thup=1.5,thdw=0.1,w=True,distance='euclidean',prob=1):
    
    '''
    •	generation of the network of all minimal rhythm leadings in a generalized musical space of Nc-dim rhythmic cells – based on the rhythm distance operator
    •	input_csv (str)– file containing the dictionary generated by rhythmNetwork
    •	thup, thdw (float)– upper and lower thresholds for edge creation
    •	w (logical) – if True it writes the nodes.csv and edges.csv files in csv format
    •	returns nodes and edges tables as pandas DataFrames
    '''

    start=time.time()    
    # Create network of minimal rhythm leadings from the rhythmDictionary
    
    df = pd.read_csv(input_csv)
    df = np.asarray(df)

    # write csv for nodes
    dnodes = pd.DataFrame(df[:,0],columns=['Label'])
    if w: dnodes.to_csv('nodes.csv',index=False)
    #dnodes.to_json('nodes.json')
    
    # find edges according to a metric
    N = df[:,1].shape[0]
    dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
    np.random.seed(int(time.process_time()*10000))
    for i in range(N):
        vector_i = []
        for l in range(len(df[:,1][0].split())):
            vector_i.append(fr.Fraction(df[:,1][i].split()[l]))
        vector_i  = RHYTHMSeq(vector_i)
        for j in range(i,N):
            vector_j = []
            for l in range(len(df[:,1][0].split())):
                vector_j.append(fr.Fraction(df[:,1][j].split()[l]))
            vector_j  = RHYTHMSeq(vector_j)
            pair = floatize(rhythmDistance(vector_i,vector_j,distance))
            if pair < thup and pair > thdw:
                if prob == 1:
                    tmp = pd.DataFrame([[str(i),str(j),str(pair)]],columns=['Source','Target','Weight'])
                    dedges = dedges.append(tmp)
                else:
                    r = np.random.rand()
                    if r <= prob:
                        tmp = pd.DataFrame([[str(i),str(j),str(pair)]],columns=['Source','Target','Weight'])
                        dedges = dedges.append(tmp)
                    else:
                        pass

    # write csv for edges
    if w: dedges.to_csv('edges.csv',index=False)

    return(dnodes,dedges)
        
def floatize(frac):
    return(frac.numerator/frac.denominator)  
    
def Sublists(lst):
    for doslice in itr.product([True, False], repeat=len(lst) - 1):
        slices = []
        start = 0
        for i, slicehere in enumerate(doslice, 1):
            if slicehere:
                slices.append(lst[start:i])
                start = i
        slices.append(lst[start:])
        yield slices