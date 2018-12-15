#
# pcsPy
#
# A python library for pitch class set classification and manipulation, and network construction and analysis
#
# Copyright (C) 2018 Marco Buongiorno Nardelli
# http://www.sonifipy.com, http://www.materialssoundmusic.com, mbn@unt.edu
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

import sys,re,time,os
import numpy as np
import itertools as iter
import pandas as pd
import sklearn.metrics as sklm
import networkx as nx
import community as cm
import music21 as m21
from mpi4py import MPI

from communications import *
from load_balancing import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class PCSet:

    def __init__(self,pcs,TET=12,UNI=True):
        # chose if to eliminate duplicates - ascending order
        if UNI == True:
            self.pcs = np.unique(pcs)
        else:
            self.pcs = np.sort(pcs)
        self.TET = TET

    def normalOrder(self):

        self.pcs = np.sort(self.pcs)
        
        # trivial sets
        if len(self.pcs) == 1:
            return(self.pcs-self.pcs[0])
        if len(self.pcs) == 2:
            return(self.pcs)

        # 1. cycle to find the most compact ascending order
        nroll = np.linspace(0,len(self.pcs)-1,len(self.pcs),dtype=int)
        dist = np.zeros((len(self.pcs)),dtype=int)
        for i in range(len(self.pcs)):
            dist[i] = (np.roll(self.pcs,i)[len(self.pcs)-1] - np.roll(self.pcs,i)[0])%self.TET

        # 2. check for multiple compact orders
        for l in range(1,len(self.pcs)):
            if np.array(np.where(dist == dist.min())).shape[1] != 1:
                indx = np.array(np.where(dist == dist.min()))[0]
                nroll = nroll[indx]
                dist = np.zeros((len(nroll)),dtype=int)
                i = 0
                for n in nroll:
                    dist[i] = (np.roll(self.pcs,n)[len(self.pcs)-(1+l)] - np.roll(self.pcs,n)[0])%self.TET
                    i += 1
            else:
                indx = np.array(np.where(dist == dist.min()))[0]
                nroll = nroll[int(indx[0])]
                pcs_norm = np.roll(self.pcs,nroll)
                break
        if np.array(np.where(dist == dist.min())).shape[1] != 1: pcs_norm = self.pcs
        return(pcs_norm)

    def normal0Order(self):
        return((self.normalOrder()-self.normalOrder()[0])%self.TET)

    def transpose(self,t=0):
        return((self.pcs+t)%self.TET)
    
    def zeroOrder(self):
        return((self.pcs-self.pcs[0])%self.TET)

    def inverse(self):
        return(-self.pcs%self.TET)

    def primeForm(self):
        s_orig = self.pcs
        sn = np.sum((self.normalOrder()-self.normalOrder()[0])%self.TET)
        self.pcs = self.inverse()
        si = np.sum((self.normalOrder()-self.normalOrder()[0])%self.TET)
        if sn <= si:
            self.pcs = s_orig
            return((self.normalOrder()-self.normalOrder()[0])%self.TET)
        else:
            tmp = (self.normalOrder()-self.normalOrder()[0])%self.TET
            self.pcs = s_orig
            return(tmp)
        

    def intervalVector(self):
        npc = int((len(self.pcs)**2-len(self.pcs))/2)
        itv = np.zeros(npc,dtype=int)
        n= 0
        for i in range(len(self.pcs)):
            for j in range(i+1,len(self.pcs)):
                if np.abs(self.pcs[i]-self.pcs[j]) > self.TET/2:
                    itv[n] = self.TET-np.abs(self.pcs[i]-self.pcs[j])
                else:
                    itv[n] = np.abs(self.pcs[i]-self.pcs[j])
                n += 1
        bins = np.linspace(1,self.TET/2+1,self.TET/2+1,dtype=int)
        return(np.histogram(itv,bins)[0])

    def LISVector(self):
        return((np.roll(self.pcs,-1)-self.pcs)%self.TET)

    def forteClass(self):
        if self.TET != 12:
            print('Forte class defined only for 12-TET')
            return()
        forteDict = {'[012]':'[3-1]','[013]':'[3-2]','[014]':'[3-3]','[015]':'[3-4]','[016]':'[3-5]',
        '[024]':'[3-6]','[025]':'[3-7]','[026]':'[3-8]','[027]':'[3-9]','[036]':'[3-10]',
        '[037]':'[3-11]','[048]':'[3-12]','[0123]':'[4-1]','[0124]':'[4-2]','[0125]':'[4-4]',
        '[0126]':'[4-5]','[0127]':'[4-6]','[0134]':'[4-3]','[0135]':'[4-11]','[0136]':'[4-13]',
        '[0137]':'[4-Z29]','[0145]':'[4-7]','[0146]':'[4-Z15]','[0147]':'[4-18]','[0148]':'[4-19]',
        '[0156]':'[4-8]','[0157]':'[4-16]','[0158]':'[4-20]','[0167]':'[4-9]','[0235]':'[4-10]',
        '[0236]':'[4-12]','[0237]':'[4-14]','[0246]':'[4-21]','[0247]':'[4-22]','[0248]':'[4-24]',
        '[0257]':'[4-23]','[0258]':'[4-27]','[0268]':'[4-25]','[0347]':'[4-17]','[0358]':'[4-26]',
        '[0369]':'[4-28]','[01234]':'[5-1]','[01235]':'[5-2]','[01236]':'[5-4]','[01237]':'[5-5]',
        '[01245]':'[5-3]','[01246]':'[5-9]','[01247]':'[5-Z36]','[01248]':'[5-13]','[01256]':'[5-6]',
        '[01257]':'[5-14]','[01258]':'[5-Z38]','[01267]':'[5-7]','[01268]':'[5-15]','[01346]':'[5-10]',
        '[01347]':'[5-16]','[01348]':'[5-Z17]','[01356]':'[5-Z12]','[01357]':'[5-24]',
        '[01358]':'[5-27]','[01367]':'[5-19]','[01368]':'[5-29]','[01369]':'[5-31]','[01457]':'[5-Z18]',
        '[01458]':'[5-21]','[01468]':'[5-30]',
        '[01469]':'[5-32]','[01478]':'[5-22]','[01568]':'[5-20]','[02346]':'[5-8]','[02347]':'[5-11]',
        '[02357]':'[5-23]','[02358]':'[5-25]','[02368]':'[5-28]','[02458]':'[5-26]','[02468]':'[5-33]',
        '[02469]':'[5-34]','[02479]':'[5-35]','[03458]':'[5-Z37]','[012345]':'[6-1]','[012346]':'[6-2]',
        '[012347]':'[6-Z36]','[012348]':'[6-Z37]','[012356]':'[6-Z3]','[012357]':'[6-9]','[012358]':'[6-Z40]',
        '[012367]':'[6-5]','[012368]':'[6-Z41]','[012369]':'[6-Z42]','[012378]':'[6-Z38]','[012456]':'[6-Z4]',
        '[012457]':'[6-Z11]','[012458]':'[6-15]','[012467]':'[6-Z12]','[012468]':'[6-22]','[012469]':'[6-Z46]',
        '[012478]':'[6-Z17]','[012479]':'[6-Z47]','[012567]':'[6-Z6]','[012568]':'[6-Z43]','[012569]':'[6-Z44]',
        '[012578]':'[6-18]','[012579]':'[6-Z48]','[012678]':'[6-7]','[013457]':'[6-Z10]'
        ,'[013458]':'[6-14]','[013467]':'[6-Z13]','[013468]':'[6-Z24]','[013469]':'[6-27]','[013478]':'[6-Z19]',
        '[013479]':'[6-Z49]','[013568]':'[6-Z25]','[013569]':'[6-Z28]','[013578]':'[6-Z26]','[013579]':'[6-34]',
        '[013679]':'[6-30]','[014568]':'[6-16]','[014579]':'[6-31]','[014589]':'[6-20]','[014679]':'[6-Z50]',
        '[023457]':'[6-8]','[023458]':'[6-Z39]','[023468]':'[6-21]','[023469]':'[6-Z45]','[023568]':'[6-Z23]',
        '[023579]':'[6-33]','[023679]':'[6-Z29]','[024579]':'[6-32]','[0123456]':'[7-1]','[0123457]':'[7-2]',
        '[0123458]':'[7-3]','[0123467]':'[7-4]','[0123468]':'[7-9]','[0123469]':'[7-10]','[0123478]':'[7-6]',
        '[0123479]':'[7-Z12]','[0123567]':'[7-5]','[0123568]':'[7-Z36]','[0123569]':'[7-16]',
        '[0123578]':'[7-14]','[0123579]':'[7-24]','[0145679]':'[7-Z18]','[0123678]':'[7-7]','[0123679]':'[7-19]',
        '[0124568]':'[7-13]','[0124569]':'[7-Z17]','[0124578]':'[7-Z38]','[0124579]':'[7-27]','[0124589]':'[7-21]',
        '[0124678]':'[7-15]','[0124679]':'[7-29]','[0124689]':'[7-30]','[01246810]':'[7-33]','[0125679]':'[7-20]',
        '[0125689]':'[7-22]','[0134568]':'[7-11]','[0134578]':'[7-Z37]','[0134579]':'[7-26]','[0134679]':'[7-31]',
        '[0134689]':'[7-32]','[01346810]':'[7-34]','[0135679]':'[7-28]','[01356810]':'[7-35]','[0234568]':'[7-8]',
        '[0234579]':'[7-23]','[0234679]':'[7-25]','[01234567]':'[8-1]','[01234568]':'[8-2]','[01234569]':'[8-3]',
        '[01234578]':'[8-4]','[01234579]':'[8-11]','[01234589]':'[8-7]',
        '[01234678]':'[8-5]','[01234679]':'[8-13]','[01234689]':'[8-Z15]','[012346810]':'[8-21]','[01234789]':'[8-8]',
        '[01235678]':'[8-6]','[01235679]':'[8-Z29]','[01235689]':'[8-18]','[012356810]':'[8-22]',
        '[01235789]':'[8-16]','[012357810]':'[8-23]','[01236789]':'[8-9]','[01245679]':'[8-14]',
        '[01245689]':'[8-19]','[012456810]':'[8-24]','[01245789]':'[8-20]','[012457810]':'[8-27]',
        '[012467810]':'[8-25]','[01345679]':'[8-12]','[01345689]':'[8-17]','[013457810]':'[8-26]',
        '[013467910]':'[8-28]','[02345679]':'[8-10]','[012345678]':'[9-1]','[012345679]':'[9-2]',
        '[012345689]':'[9-3]','[0123456810]':'[9-6]','[012345789]':'[9-4]','[0123457810]':'[9-7]',
        '[012346789]':'[9-5]','[0123467810]':'[9-8]','[0123467910]':'[9-10]','[0123567810]':'[9-9]',
        '[0123567910]':'[9-11]','[0124568910]':'[9-12]','[0123456789]':'[10-1]','[01234567810]':'[10-2]',
        '[01234567910]':'[10-3]','[01234568910]':'[10-4]','[01234578910]':'[10-5]','[01234678910]':'[10-6]',
        '[012345678910]':'[11-1]','[01234567891011]':'[12-1]'}
        try:
            Fname = forteDict[np.array2string(self.primeForm(),separator='').replace(" ","")]
        except:
            print('set not found')
            Fname=None
        return(Fname)
        
    def jazzChord(self):
        if self.TET != 12:
            print('Jazz chords defined only for 12-TET')
            return()
        jazzDict = {'[047]':'Maj','[037]':'min','[036]':'dim','[048]':'+','[046]':'b5','[027]':'sus','[057]':'add4',
                    '[0237]':'m(add2)','[0357]':'m(add4)','[0247]':'(add2)','[0457]':'(add4)','[0369]':'dim','[03610]':'m7b5',
                    '[0379]':'m6','[03710]':'m7','[0479]':'6','[04710]':'7','[04711]':'Maj7','[03711]':'-Maj7','[04810]':'7+',
                    '[04811]':'Maj7+','[02379]':'m6/9','[023710]':'m9','[014710]':'m7b9','[024710]':'9','[023710]':'9b5',
                    '[034710]':'7#9','[024711]':'Maj9','[047910]':'6(add7)','[02479]':'6/9','[014810]':'7b9+','[027810]':'9+',
                    '[034810]':'7#9+','[025710]':'9sus4','[0235710]':'m11','[0134710]':'7b9#9','[0146710]':'7b9#11',
                    '[0246710]':'9#11','[0246711]':'Maj9#11','[0134810]':'7b9#9+','[0146810]':'7b9#11+'}
        try:
            Fname = jazzDict[np.array2string(self.normal0Order(),separator='').replace(" ","")]
        except:
            print('set not found')
            Fname=None
        return(Fname)
    
    def commonName(self):
        return(m21.chord.Chord(np.ndarray.tolist(self.normalOrder()[:])).commonName)
    
    def commonNamePrime(self):
        return(m21.chord.Chord(np.ndarray.tolist(self.primeForm()[:])).commonName)
    
    def nameWithPitchOrd(self):
        return(m21.note.Note(self.normalOrder()[0]).nameWithOctave+' '+self.commonName())

    def nameWithPitch(self):
        return(m21.note.Note(self.pcs[0]).nameWithOctave+' '+self.commonName())

    
    def displayNotes(self,xml=False,prime=False):
        s = m21.stream.Stream()
        fac = self.TET/12
        for i in range(self.pcs.shape[0]):
            if prime: 
                s.append(m21.note.Note(self.primeForm()[i]/fac+60))
            else:
                s.append(m21.note.Note(self.pcs[i]/fac+60))
        s.show()
        if xml: s.show('musicxml')
        return

########### Network functions ###########

def pcsDictionary(Nc,order=0,TET=12,row=False,a=np.array(None)):

    # Create dictionary of pcs from a given cardinality Nc
    name = prime = commonName = None
    if rank == 0:
        name = []
        prime = []
        commonName = []
        
    # generate all possible combinations of n integers or of the row in argument
    if row:
        a = np.asarray(list(iter.combinations(a,Nc)))
    else:
        a = np.asarray(list(iter.combinations(range(TET),Nc)))

    # put all pcs in prime/normal order form
    s = np.zeros((a.shape[0],Nc),dtype=int)
    ini,end = load_balancing(size, rank, a.shape[0])
    nsize = end-ini

    aux = scatter_array(a)
    saux = np.zeros((nsize,Nc),dtype=int)
    comm.Barrier()
    for i in range(nsize):
        p = PCSet(aux[i,:],TET)
        if order == 0:
            saux[i,:] = p.primeForm()[:]
        elif order == 1:
            saux[i,:] = p.normalOrder()[:]
        elif order == 2:
            saux[i,:] = p.normal0Order()[:]
        else:
            if rank == 0: print('no ordering specified')
    comm.Barrier()
    gather_array(s,saux,sroot=0)

    if rank == 0:
        # eliminate duplicates in s
        s = np.unique(s,axis=0)

        # calculate interval vectors and assign names
        v = []
        for i in range(s.shape[0]):
            p = PCSet(s[i,:],TET)
            v.append(p.intervalVector())
            name.append(str(Nc)+'-'+str(i+1))
            prime.append(np.array2string(s[i,:],separator=',').replace(" ",""))

        vector = np.asarray(v)

    dictionary = ZrelT = None
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

        name = np.asarray(name)
        
        # Create dictionary of pitch class sets
        reference = []
        for n in range(len(name)):
            entry = [name[n],prime[n],
                    np.array2string(vector[n,:],separator=',').replace(" ","")]
            reference.append(entry)

        dictionary = pd.DataFrame(reference,columns=['class','pcs','interval'])
        
    return(dictionary,ZrelT)

def pcsNetwork(input_csv,thup=1.5,thdw=0.0,TET=12,distance='euclidean',col=2,prob=1):

    # Create network of pcs from the pcsDictionary - parallel version
    # col = 2 - interval vector
    # col = 1 - voice leading - works only for a fixed cardinality - NOT ACCURATE for minimal voice leading
    
    df = pd.read_csv(input_csv)
    df = np.asarray(df)
    
    if col == 2: dim = np.asarray(list(map(int,re.findall('\d+',df[0,col])))).shape[0]
    if col == 1: 
        if rank == 0: print('NOT ACCURATE for minimal voice leading - use vLeadNetwork instead!')
        dim = int(TET/2)

    # write csv for nodes
    dnodes = pd.DataFrame(df[:,0],columns=['Label'])
    dnodes.to_csv('nodes.csv',index=False)
    # find edges according to a metric
    
    vector = np.zeros((df[:,col].shape[0],dim))
    for i in range(df[:,col].shape[0]):
        vector[i]  = np.asarray(list(map(int,re.findall('\d+',df[i,col]))))
    N = vector.shape[0]
    index = np.linspace(0,vector.shape[0]-1,vector.shape[0],dtype=int)
    # parallelize over interval vector to optimize the vectorization in sklm.pairwise_distances
    ini,end = load_balancing(size, rank, N)
    nsize = end-ini
    vaux = scatter_array(vector)
    pair = sklm.pairwise_distances(vaux, vector, metric=distance)
    dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
    for i in range(nsize):
        tmp = pd.DataFrame(None,columns=['Source','Target','Weight'])
        tmp['Source'] = (i+ini)*np.ones(vector.shape[0],dtype=int)[:]
        tmp['Target'] = index[:]
        tmp['Weight'] = pair[i,:]
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
    
    if size != 1 and rank == 0:
        dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
        for i in range(size):
            tmp = pd.read_csv('edges'+str(i)+'.csv')
            dedges = dedges.append(tmp)
            os.remove('edges'+str(i)+'.csv')
        # write csv for edges
        dedges.to_csv('edges.csv',index=False)
    elif size == 1:
        os.rename('edges'+str(rank)+'.csv','edges.csv')

    return()

def pcsEgoNetwork(label,input_csv,thup_e=5.0,thdw_e=0.1,thup=1.5,thdw=0.1,TET=12,distance='euclidean'):
    
    if thdw_e < 1e-9:
        print('ego should not link to itself')
        sys.exit()
    
    # Create the ego network of pcs from a given node using the pcsDictionary
    
    df = pd.read_csv(input_csv)

    # define nodes as distance 1 from ego
    # ego
    dict_class = df.set_index("class", drop = True)
    ego = np.asarray(list(map(int,re.findall('\d+',dict_class.loc[label][1]))))
    # alters
    dfv = np.asarray(df)
    vector = np.zeros((dfv[:,2].shape[0],int(TET/2)),dtype=int)
    for i in range(dfv[:,2].shape[0]):
        vector[i]  = np.asarray(list(map(int,re.findall('\d+',dfv[i,2]))))
    name = []
    pair = sklm.pairwise_distances(ego.reshape(1, -1), vector, metric=distance)
    for i in range(dfv[:,2].shape[0]):
        if pair[0,i] <= thup_e and pair[0,i] >= thdw_e:
            name.append(dfv[i,0])
    # add ego node
    name.append(label)
                      
    # write csv for nodes
    dnodes = pd.DataFrame(np.asarray(name),columns=['Label'])
    dnodes.to_csv('nodes_ego.csv',index=False)
    
    # find edges according to a metric
    # ego edges with proportional weights
    N = len(name)
    vector = np.zeros((N,int(TET/2)),dtype=int)
    dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
    for j in range(N):
        vector[j] = np.asarray(list(map(int,re.findall('\d+',dict_class.loc[name[j]][1]))))
    pair = sklm.pairwise_distances(ego.reshape(1, -1), vector, metric=distance)
    for j in range(N):
        if pair[0,j] <= thup_e and pair[0,j] >= thdw_e:
            tmp = pd.DataFrame([[str(i),str(j),str(1/pair[0,j])]],columns=['Source','Target','Weight'])
            dedges = dedges.append(tmp)
    # write csv for ego's edges
    dedges.to_csv('edges_ego.csv',index=False)        
    
    # alters edges
    # parallelize over interval vector to optimize the vectorization in sklm.pairwise_distances
    if size != 1:
        ini,end = load_balancing(size, rank, N)
        nsize = end-ini
        vaux = scatter_array(vector)
        pair = sklm.pairwise_distances(vaux, vector, metric=distance)
        index = np.linspace(0,N,N,dtype=int)
        dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
        for i in range(nsize):
            tmp = pd.DataFrame(None,columns=['Source','Target','Weight'])
            tmp['Source'] = (i+ini)*np.ones(N,dtype=int)[:]
            tmp['Target'] = index[:]
            tmp['Weight'] = pair[i,:]
            dedges = dedges.append(tmp)
        dedges = dedges.query('Weight<='+str(thup)).query('Weight>='+str(thdw))
        dedges['Weight'] = dedges['Weight'].apply(lambda x: 1/x)
        # do some cleaning
        cond = dedges.Source > dedges.Target
        dedges.loc[cond, ['Source', 'Target']] = dedges.loc[cond, ['Target', 'Source']].values
        dedges = dedges.drop_duplicates(subset=['Source', 'Target'])

        # write csv for partial edges
        dedges.to_csv('edges'+str(rank)+'.csv',index=False)
        
        if rank == 0:
            dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
            for i in range(size):
                tmp = pd.read_csv('edges'+str(i)+'.csv')
                dedges = dedges.append(tmp)
                os.remove('edges'+str(i)+'.csv')
            # write csv for edges
            dedges.to_csv('edges_alters.csv',index=False)
    else:
        N = len(name)-1
        dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
        for i in range(N):
            vector_i = np.asarray(list(map(int,re.findall('\d+',dict_class.loc[name[i]][1]))))
            for j in range(i,N):
                vector_j = np.asarray(list(map(int,re.findall('\d+',dict_class.loc[name[j]][1]))))
                pair = sklm.pairwise.paired_euclidean_distances(vector_i.reshape(1, -1),vector_j.reshape(1, -1))
                if pair <= thup and pair >= thdw:
                    tmp = pd.DataFrame([[str(i),str(j),str(1/pair[0])]],columns=['Source','Target','Weight'])
                    dedges = dedges.append(tmp)

        # write csv for alters' edges
        dedges.to_csv('edges_alters.csv',index=False)
    
    return()
    
def vLeadNetwork(input_csv,thup=1.5,thdw=0.1,TET=12,w=True,distance='euclidean',prob=1):

    start=time.time()    
    # Create network of minimal voice leadings from the pcsDictionary
    
    df = pd.read_csv(input_csv)
    df = np.asarray(df)

#    Nc = np.asarray(list(map(int,re.findall('\d+',df[0,1])))).shape[0]
#    last = np.asarray(list(map(int,re.findall('\d+',df[df[:,1].shape[0]-1,1])))).shape[0]
#    if Nc != last:
#        if rank == 0: print('voice leading network only for single cardinality!')
#        sys.exit()

    # write csv for nodes
    dnodes = pd.DataFrame(df[:,0],columns=['Label'])
    if w: dnodes.to_csv('nodes.csv',index=False)
    #dnodes.to_json('nodes.json')
    
    # find edges according to a metric - allows for non-bijective voice leading
    
#    vector = np.zeros((df[:,1].shape[0],Nc))
#    for i in range(df[:,1].shape[0]):
#        vector[i]  = np.asarray(list(map(int,re.findall('\d+',df[i,1]))))
#    reset=time.time()
    N = df[:,1].shape[0]
    dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
    np.random.seed(int(time.process_time()*10000))
    for i in range(N):
        vector_i  = np.asarray(list(map(int,re.findall('\d+',df[i,1]))))
        for j in range(i,N):
            vector_j  = np.asarray(list(map(int,re.findall('\d+',df[j,1]))))
            if vector_i.shape[0] == vector_j.shape[0]:
                pair = minimalDistance(vector_i,vector_j,TET,distance)
            else:
                if vector_i.shape[0] > vector_j.shape[0]:
                    a = vector_i 
                    b = vector_j
                else:
                    b = vector_i 
                    a = vector_j
                ndif = np.sort(np.array([a.shape[0],b.shape[0]]))[1] - np.sort(np.array([a.shape[0],b.shape[0]]))[0]
                c = np.asarray(list(iter.combinations_with_replacement(b,ndif)))
                r = np.zeros((c.shape[0],a.shape[0]))
                for l in range(c.shape[0]):
                    r[l,:b.shape[0]] = b
                    r[l,b.shape[0]:] = c[l]
                dist = np.zeros(r.shape[0])
                for l in range(r.shape[0]):
                    dist[l]=minimalDistance(a,r[l])
                pair = min(dist)
            if pair <= thup and pair >= thdw:
                if prob == 1:
                    tmp = pd.DataFrame([[str(i),str(j),str(1/pair)]],columns=['Source','Target','Weight'])
                    dedges = dedges.append(tmp)
                else:
                    r = np.random.rand()
                    if r >= prob:
                        tmp = pd.DataFrame([[str(i),str(j),str(1/pair)]],columns=['Source','Target','Weight'])
                        dedges = dedges.append(tmp)
                    else:
                        pass

    # write csv for edges
    if w: dedges.to_csv('edges.csv',index=False)

    return(dnodes,dedges)

def scoreNetwork(seq,TET=12):
    # build the directional network of chord progressions from any score chord sequence in musxml format
    ''' 
    example from the corpus of bach chorales:
        # read score
        bachChorale = m21.corpus.parse('bwv66.6')
        # extract chords:
        chords = bachChorale.chordify()
        seq = []
        for c in chords.recurse().getElementsByClass('Chord'):
            seq.append(c.normalOrder)
        # seq is the chord sequence
    '''
    # build the directional network of the full progression in the chorale

    dedges = pd.DataFrame(None,columns=['Source','Target','Weight','Label'])
    dnodes = pd.DataFrame(None,columns=['Label'])
    for n in range(len(seq)):
        p = PCSet(np.asarray(seq[n]),TET)
        nn = ''.join(m21.chord.Chord(p.normalOrder().tolist()).pitchNames)
        nameseq = pd.DataFrame([[str(nn)]],columns=['Label'])
        dnodes = dnodes.append(nameseq)
    df = np.asarray(dnodes)
    dnodes = pd.DataFrame(None,columns=['Label'])
    dff,idx = np.unique(df,return_inverse=True)
    for n in range(dff.shape[0]):
        nameseq = pd.DataFrame([[str(dff[n])]],columns=['Label'])
        dnodes = dnodes.append(nameseq)

    for n in range(1,len(seq)):
        if len(seq[n-1]) == len(seq[n]):
            a = np.asarray(seq[n-1])
            b = np.asarray(seq[n])
            pair = minimalDistance(a,b)
        else:
            if len(seq[n-1]) > len(seq[n]):
                a = np.asarray(seq[n-1])
                b = np.asarray(seq[n])
                pair = minimalNoBijDistance(a,b)
            else: 
                b = np.asarray(seq[n-1])
                a = np.asarray(seq[n])
                pair = minimalNoBijDistance(a,b)
        if pair != 0:
            tmp = pd.DataFrame([[str(idx[n-1]),str(idx[n]),str(1/pair),opsDictionary(pair)]],
                               columns=['Source','Target','Weight','Label'])
            dedges = dedges.append(tmp)
    
    # evaluate average degree and modularity
    gbch = nx.from_pandas_dataframe(dedges,'Source','Target',['Weight','Label'],create_using=nx.DiGraph())
    gbch_u = nx.from_pandas_dataframe(dedges,'Source','Target',['Weight','Label'])
    # modularity 
    part = cm.best_partition(gbch_u)
    modul = cm.modularity(part,gbch_u)
    # average degree
    nnodes=gbch.number_of_nodes()
    avgdeg = sum(gbch.in_degree().values())/float(nnodes)
        
    return(dnodes,dedges,avgdeg,modul)

def scoreDictionary(seq,TET=12):
    # build the dictionary of pcs in the score
    s = Remove(seq)
    v = []
    name = []
    prime = []
    for i in range(len(s)):
        p = PCSet(PCSet(np.asarray(s[i][:]),TET).transpose(0))
        v.append(p.intervalVector())
        name.append(''.join(m21.chord.Chord(p.normalOrder().tolist()).pitchNames))
        prime.append(np.array2string(p.normalOrder(),separator=',').replace(" ",""))

    vector = np.asarray(v)
    name = np.asarray(name)

    # Create dictionary of pitch class sets
    reference = []
    for n in range(len(name)):
        entry = [name[n],prime[n],
                np.array2string(vector[n,:],separator=',').replace(" ","")]
        reference.append(entry)

    dictionary = pd.DataFrame(reference,columns=['class','pcs','interval'])
    
    return(dictionary)

def extractByString(name,label,string):
    # extract rows of dictionary according to a particular string in column 'label'
    if type(name) is str: 
        df = pd.read_csv(name)
    else:
        df = name
    return(df[df[label].str.contains(string)])
    
def minimalDistance(a,b,TET=12,distance='euclidean'):
    # calculate minimal distance between two pcs of same cardinality
    n = a.shape[0]
    if a.shape[0] != b.shape[0]:
        print('dimension of arrays must be equal')
        sys.exit()
    a = np.sort(a)
    iTET = np.vstack([np.identity(n,dtype=int)*TET,-np.identity(n,dtype=int)*TET])
    iTET = np.vstack([iTET,np.zeros(n,dtype=int)])
    diff = np.zeros(2*n+1,dtype=float)
    for i in range(2*n+1):
        r = np.sort(b - iTET[i])
        diff[i] = sklm.pairwise_distances(a.reshape(1, -1),r.reshape(1, -1),metric=distance)[0]
    
    return(diff.min())
    
def minimalNoBijDistance(a,b):
    # calculate minimal distance between two pcs of different cardinality
    ndif = np.sort(np.array([a.shape[0],b.shape[0]]))[1] - np.sort(np.array([a.shape[0],b.shape[0]]))[0]
    c = np.asarray(list(iter.combinations_with_replacement(b,ndif)))
    r = np.zeros((c.shape[0],a.shape[0]))
    for l in range(c.shape[0]):
        r[l,:b.shape[0]] = b
        r[l,b.shape[0]:] = c[l]
    dist = np.zeros(r.shape[0])
    for l in range(r.shape[0]):
        dist[l]=minimalDistance(a,r[l])
        
    return(min(dist))

def opsDictionary(distance):
    # define the dictionary of O({n_i}) distance operators
    opsDict={1.0:'O(1)',1.4142:'O(1,1)',1.7321:'O(1,1,1)',2.0:'O(2)/O(1,1,1,1)',2.2361:'O(1,2)',2.4495:'O(1,1,2)',
        2.8284:'O(2,2)',3.0:'O(1,2,2)/O(3)',3.1623:'O(1,3)/O(1,1,2,2)',3.3166:'O(1,1,3)',3.4641:'O(2,2,2)',
        3.6056:'O(2,3)/O(1,2,2,2)',3.7417:'O(1,2,3)',2.6458:'O(1,1,1,2)',4.0:'O(2,2,2,2)',4.2426:'O(1,2,2,3)',
        3.873:'O(1,1,2,3)',4.5826:'O(2,2,2,3)',5.099:'O(2,2,3,3)',4.7958:'O(1,2,3,3)',4.4721:'O(1,1,3,3)',
        5.2915:'O(1,3,3,3)',5.5678:'O(2,3,3,3)',6.0:'O(3,3,3,3)'}
    try:
        Oname = opsDict[np.round(distance,4)]
    except:
        Oname=None
    return(Oname)
    
def Remove(duplicate): 
    # remove duplicates from list
    final_list = [] 
    for num in duplicate: 
        if num not in final_list: 
            final_list.append(num) 
    return final_list 
