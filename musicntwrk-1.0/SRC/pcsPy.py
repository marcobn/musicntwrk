#
# pcsPy
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

import sys,re,time,os
import numpy as np
import itertools as iter
import pandas as pd
import sklearn.metrics as sklm
import networkx as nx
import community as cm
import music21 as m21
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import vpython as vp

from scipy.optimize import curve_fit
import collections
import powerlaw

from mpi4py import MPI

from communications import *
from load_balancing import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class PCSet:

    def __init__(self,pcs,TET=12,UNI=True,ORD=True):
        '''
        •	pcs (int)– pitch class set as list or numpy array
        •	TET (int)- number of allowed pitches in the totality of the musical space (temperament). Default = 12 tones equal temperament
        •	UNI (logical) – if True, eliminate duplicate pitches (default)
        •   ORD (logical) - if True, sorts the pcs in ascending order
        '''
        if UNI == True:
            self.pcs = np.unique(pcs)%TET
        if ORD == True:
            self.pcs = np.sort(pcs)%TET
        else:
            self.pcs = np.asarray(pcs)%TET
        self.TET = TET

    def normalOrder(self):
        '''
        •	Order the pcs according to the most compact ascending scale in pitch-class space that spans less than an octave by cycling permutations.
        '''
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
        '''
        •	As normal order, transposed so that the first pitch is 0
        '''
        return((self.normalOrder()-self.normalOrder()[0])%self.TET)

    def transpose(self,t=0):
        '''
        •	Transposition by t (int) units (modulo TET)
        '''
        return((self.pcs+t)%self.TET)
        
    def multiply(self,t=1):
        '''
        •	Transposition by t (int) units (modulo TET)
        '''
        return(np.unique((self.pcs*t)%self.TET//1).astype(int))
        
    def multiplyBoulez(self,b,TET=12):
        # Boulez pitch class multiplication of a x b
        ivec = self.LISVector()
        m = []
        for i in range(ivec.shape[0]-1):
            mm = (b+ivec[i])%TET
            m.append(mm.tolist())
        return(PCSet(Remove(flatten(m+b)),TET).normalOrder())
    
    def zeroOrder(self):
        '''
        •	transposed so that the first pitch is 0
        '''
        return((self.pcs-self.pcs[0])%self.TET)

    def inverse(self):
        '''
        •	inverse operation: (-pcs modulo TET)
        '''
        return(-self.pcs%self.TET)

    def primeForm(self):
        '''
        •	most compact normal 0 order between pcs and its inverse
        '''
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
        '''
        •	 total interval content of the pcs
        '''
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
        bins = np.linspace(1,int(self.TET/2)+1,int(self.TET/2)+1,dtype=int)
        return(np.histogram(itv,bins)[0])

    def operator(self,name):
        # operate on the pcs with a distance operator
        
        def plusAndMinusPermutations(items):
            for p in iter.permutations(items):
                for signs in iter.product([-1,1], repeat=len(items)):
                    yield [a*sign for a,sign in zip(p,signs)]

        op = ' '.join(i for i in name if i.isdigit()).split()
        op = np.asarray([list(map(int, x)) for x in op])
        op = np.reshape(op,op.shape[0]*op.shape[1])
        if self.pcs.shape[0] == op.shape[0]:
            pop = np.asarray(list(plusAndMinusPermutations(op)))
            selfto = np.unique((self.pcs+pop)%self.TET,axis=0)
            outset = []
            for n in range(selfto.shape[0]):
                if minimalNoBijDistance(self.normalOrder(),PCSet(selfto[n]).normalOrder())[0] == opsDistance(name)[1]:
                    outset.append(PCSet(selfto[n]).normalOrder().tolist())
        if self.pcs.shape[0] > op.shape[0]:
            op = np.pad(op,(0,self.pcs.shape[0]-op.shape[0]),'constant')
            pop = np.asarray(list(plusAndMinusPermutations(op)))
            selfto = np.unique((self.pcs+pop)%self.TET,axis=0)
            outset = []
            for n in range(selfto.shape[0]):
                if minimalNoBijDistance(self.normalOrder(),PCSet(selfto[n]).normalOrder())[0] == opsDistance(name)[1]:
                    outset.append(PCSet(selfto[n]).normalOrder().tolist())
        if self.pcs.shape[0] < op.shape[0]:
            print("increase cardinality by duplicating pc's - program will stop")
            outset = None
        return(Remove(outset))
        
    def Roperator(self,name):
        # operate on the pcs with a normal-ordered voice-leading operator
        
        op = []
        for num in re.findall("[-\d]+", name):
            op.append(int(num))
        op = np.asarray(op)
        selfto = np.unique((self.pcs+op)%self.TET,axis=0)
        return(PCSet(selfto).normalOrder())
        
    def LISVector(self):
        '''
        •	Linear Interval Sequence Vector: sequence of intervals in an ordered pcs
        '''
        return((np.roll(self.normalOrder(),-1)-self.normalOrder())%self.TET)
    
    def LISVectorRow(self):
        '''
        •	Linear Interval Sequence Vector: sequence of intervals in an ordered pcs
        '''
        return((np.roll(self,-1)-self)%self.TET)

    def forteClass(self):
        '''
        •	Name of pcs according to the Forte classification scheme (only for TET=12)
        '''
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
        '''
        •	Name of pcs as chord in a jazz chart (only for TET=12 and cardinalities ≤ 7)
        '''
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
        '''
        •	Display common name of pcs (music21 function - only for TET=12)
        '''
        return(m21.chord.Chord(np.ndarray.tolist(self.normalOrder()[:])).commonName)
    
    def commonNamePrime(self):
        '''
        •	As above, for prime forms
        '''
        return(m21.chord.Chord(np.ndarray.tolist(self.primeForm()[:])).commonName)
    
    def nameWithPitchOrd(self):
        '''
        •	Name of chord with first pitch of pcs in normal order
        '''
        return(m21.note.Note(self.normalOrder()[0]).nameWithOctave+' '+self.commonName())

    def nameWithPitch(self):
        '''
        •	Name of chord with first pitch of pcs
        '''
        return(m21.note.Note(self.pcs[0]).nameWithOctave+' '+self.commonName())

    
    def displayNotes(self,show=True,xml=False,prime=False,chord=False):
        '''
        •	Display pcs in score in musicxml format. If prime is True, display the prime form. If chord is True 
            display the note cluster
        '''
        fac = self.TET/12
        if  not chord:
            s = m21.stream.Stream()
            for i in range(self.pcs.shape[0]):
                if prime: 
                    s.append(m21.note.Note(self.primeForm()[i]/fac+60))
                else:
                    s.append(m21.note.Note(self.pcs[i]/fac+60))
            if show: s.show()
            if xml: s.show('musicxml')
            return(s)
        else:
            ch = []
            if prime: 
                for i in range(self.pcs.shape[0]):
                    ch.append(m21.note.Note(self.primeForm()[i]/fac+60))
            else:
                for i in range(self.pcs.shape[0]):
                    ch.append(m21.note.Note(self.pcs[i]/fac+60))
            c = m21.chord.Chord(ch)
            if show: c.show()
            if xml: c.show('musicxml')
            return(c)

class PCSetR:

    def __init__(self,pcs,TET=12,UNI=False,ORD=False):
        '''
        •	pcs (int)– pitch class set as list or numpy array
        •	TET (int)- number of allowed pitches in the totality of the musical space (temperament).
            Default = 12 tones equal temperament
        •	UNI (logical) – if True, eliminate duplicate pitches (default)
        •   ORD (logical) - if True, sorts the pcs in ascending order
        '''
        if UNI == True:
            self.pcs = np.unique(pcs)
        if ORD == True:
            self.pcs = np.sort(pcs)
        else:
            self.pcs = np.asarray(pcs)
        self.TET = TET

    def normalOrder(self):
        '''
        •	Order the pcs according to the most compact ascending scale
            in pitch-class space that spans less than an octave by cycling permutations.
        '''
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
        return(PCSetR(pcs_norm,TET=self.TET))

    def normal0Order(self):
        '''
        •	As normal order, transposed so that the first pitch is 0
        '''
        return(PCSetR((self.normalOrder().pcs-self.normalOrder().pcs[0])%self.TET))

    def transpose(self,t=0):
        '''
        •	Transposition by t (int) units (modulo TET)
        '''
        return(PCSetR((self.pcs+t)%self.TET,TET=self.TET))
        
    def multiply(self,t=1):
        '''
        •	Multiplication by t (int) units (modulo TET)
        '''
        return(PCSetR(np.unique((self.pcs*t)%self.TET//1,TET=self.TET).astype(int)))
        
    def multiplyBoulez(self,b):
        # Boulez pitch class multiplication of a x b
        ivec = self.LISVector()
        m = []
        for i in range(ivec.shape[0]-1):
            mm = (b+ivec[i])%self.TET
            m.append(mm.tolist())
        return(PCSetR(Remove(flatten(m+b)),TET=self.TET))
    
    def zeroOrder(self):
        '''
        •	transposed so that the first pitch is 0
        '''
        return(PCSetR((self.pcs-self.pcs[0])%self.TET,TET=self.TET))

    def inverse(self,pivot=0):
        '''
        •	inverse operation: (-pcs modulo TET)
        '''
        return(PCSetR((pivot-self.pcs)%self.TET,TET=self.TET))

    def primeForm(self):
        '''
        •	most compact normal 0 order between pcs and its inverse
        '''
        s_orig = self.pcs
        sn = np.sum((self.normalOrder().pcs-self.normalOrder().pcs[0])%self.TET)
        self.pcs = self.inverse().pcs
        si = np.sum((self.normalOrder().pcs-self.normalOrder().pcs[0])%self.TET)
        if sn <= si:
            self.pcs = s_orig
            return(PCSetR((self.normalOrder().pcs-self.normalOrder().pcs[0])%self.TET,TET=self.TET))
        else:
            tmp = (self.normalOrder().pcs-self.normalOrder().pcs[0])%self.TET
            self.pcs = s_orig
            return(PCSetR(tmp,TET=self.TET))

    def intervalVector(self):
        '''
        •	 total interval content of the pcs
        '''
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
        bins = np.linspace(1,int(self.TET/2)+1,int(self.TET/2)+1,dtype=int)
        return(np.histogram(itv,bins)[0])

    def operator(self,name):
        # operate on the pcs with a distance operator O({x})
        
        def plusAndMinusPermutations(items):
            for p in iter.permutations(items):
                for signs in iter.product([-1,1], repeat=len(items)):
                    yield [a*sign for a,sign in zip(p,signs)]

        op = ' '.join(i for i in name if i.isdigit()).split()
        op = np.asarray([list(map(int, x)) for x in op])
        op = np.reshape(op,op.shape[0]*op.shape[1])
        if self.pcs.shape[0] == op.shape[0]:
            pop = np.asarray(list(plusAndMinusPermutations(op)))
            selfto = np.unique((self.pcs+pop)%self.TET,axis=0)
            outset = []
            for n in range(selfto.shape[0]):
                if minimalNoBijDistance(self.normalOrder().pcs,
                                        PCSetR(selfto[n]).normalOrder().pcs)[0] == opsDistance(name)[1]:
                    outset.append(PCSetR(selfto[n]).normalOrder().pcs.tolist())
        if self.pcs.shape[0] > op.shape[0]:
            op = np.pad(op,(0,self.pcs.shape[0]-op.shape[0]),'constant')
            pop = np.asarray(list(plusAndMinusPermutations(op)))
            selfto = np.unique((self.pcs+pop)%self.TET,axis=0)
            outset = []
            for n in range(selfto.shape[0]):
                if minimalNoBijDistance(self.normalOrder().pcs,
                                        PCSetR(selfto[n]).normalOrder().pcs)[0] == opsDistance(name)[1]:
                    outset.append(PCSetR(selfto[n]).normalOrder().pcs.tolist())
        if self.pcs.shape[0] < op.shape[0]:
            print("increase cardinality by duplicating pc's - program will stop")
            outset = None
        return(PCSetR(Remove(outset),TET=self.TET))
        
    def Roperator(self,name):
        # operate on the pcs with a normal-ordered relational operator R({x})
        op = []
        for num in re.findall("[-\d]+", name):
            op.append(int(num))
        op = np.asarray(op)
        selfto = np.unique((self.normalOrder().pcs+op)%self.TET,axis=0)
        return(PCSetR(selfto,TET=self.TET).normalOrder())
    
    def NROperator(self,ops=None):
        x = self.pcs[1]-self.pcs[0]
        y = self.pcs[2]-self.pcs[1]
        if ops == None:
            print("specify NRO ('P','L','R')")
            return
        elif ops == 'P':
            return(PCSetR((x+y-self.pcs)%self.TET,TET=self.TET))
        elif ops == 'L':
            return(PCSetR((x-self.pcs)%self.TET,TET=self.TET))
        elif ops == 'R':
            return(PCSetR((2*x+y-self.pcs)%self.TET,TET=self.TET))

    def opsNameR(self,b,TET=12):
        # given two vectors returns the name of the normal-ordered voice-leading operator R that connects them
        a = self.normalOrder().pcs
        b = b.normalOrder().pcs  
        d = np.zeros((b.shape[0]),dtype=int) 
        for n in range(b.shape[0]):
            c = np.roll(b,n)
            diff = a-c
            for i in range(diff.shape[0]):
                if diff[i] >= int(TET/2):
                    diff[i] -= TET
                if diff[i] < -int(TET/2):
                    diff[i] += TET
            diff = np.abs(diff)
            d[n] = diff.dot(diff)
        nmin = np.argmin(d)
        b = np.roll(b,nmin)
        diff = b-a
        for i in range(diff.shape[0]):
            if diff[i] >= int(TET/2):
                diff[i] -= TET
            if diff[i] < -int(TET/2):
                diff[i] += TET
        return('R('+np.array2string(diff,separator=',').replace(" ","").replace("[","").replace("]","")+')')

    def opsNameO(self,b,TET=12):
        # given two vectors returns the name of the distance 
        # operator O that connects them
        a = np.sort(self.pcs)
        b = np.sort(b.pcs)
        d = np.zeros((b.shape[0]),dtype=int) 
        for n in range(b.shape[0]):
            c = np.roll(b,n)
            diff = a-c
            for i in range(diff.shape[0]):
                if diff[i] >= int(TET/2):
                    diff[i] -= TET
                if diff[i] < -int(TET/2):
                    diff[i] += TET
            diff = np.abs(diff)
            d[n] = diff.dot(diff)
        nmin = np.argmin(d)
        b = np.roll(b,nmin)
        diff = b-a
        for i in range(diff.shape[0]):
            if diff[i] >= int(TET/2):
                diff[i] -= TET
            if diff[i] < -int(TET/2):
                diff[i] += TET
        diff = np.sort(np.abs(diff))
        return('O('+np.array2string(np.trim_zeros(diff),separator=',')\
               .replace(" ","").replace("[","").replace("]","")+')')
    
    def LISVector(self):
        '''
        •	Linear Interval Sequence Vector: sequence of intervals in an ordered pcs
        •	also known as step-interval vector (see Cohn, Neo-Riemannian Operations, 
            Parsimonious Trichords, and Their "Tonnetz" Representations,
            Journal of Music Theory, Vol. 41, No. 1 (Spring, 1997), pp. 1-66)
        '''
        return((np.roll(self.normalOrder().pcs,-1)-self.normalOrder().pcs)%self.TET)
    
    def LISVectorRow(self):
        '''
        •	Linear Interval Sequence Vector: sequence of intervals in an ordered pcs
        •	also known as step-interval vector (see Cohn, Neo-Riemannian Operations, 
            Parsimonious Trichords, and Their "Tonnetz" Representations,
            Journal of Music Theory, Vol. 41, No. 1 (Spring, 1997), pp. 1-66)
        '''
        return((np.roll(self.pcs,-1)-self.pcs)%self.TET)


    def forteClass(self):
        '''
        •	Name of pcs according to the Forte classification scheme (only for TET=12)
        '''
        if self.TET != 12:
            print('Forte class defined only for 12-TET')
            return()
        Fname = m21.chord.Chord(self.primeForm().pcs.tolist()).forteClass
        return(Fname)
        
class PCrow:
#     Helper class for 12-tone rows operations (T,I,R,M,Q)
    def __init__(self,pcs,TET=12):
        self.pcs = np.array(pcs)%TET
        self.TET = TET
    def normalOrder(self):
        self.pcs -= self.pcs[0]
        return(PCrow(self.pcs%12))
    def intervals(self):
        return((np.roll(self.pcs,-1)-self.pcs)%self.TET)
    def T(self,t=0):
        return(PCrow((self.pcs+t)%self.TET,TET=self.TET))
    def I(self,pivot=0):
        return(PCrow((pivot-self.pcs)%self.TET,TET=self.TET))
    def R(self):
        return(PCrow(self.pcs[::-1]).T(6))
    def Q(self):
        lisv = PCrow(self.pcs).intervals()
        lisvQ = np.roll(lisv,np.where(lisv==6)[0][1]-np.where(lisv==6)[0][0])
        Qrow = [0]
        for n in lisvQ:
            Qrow.append((Qrow[-1]+n)%self.TET)
        Qrow.pop()
        return(PCrow(Qrow))
    def M(self):
        return(PCrow((self.pcs*5)%self.TET//1,TET=self.TET))
    def constellation(self):
#         Following Morris and Starr
        reference = []
        entry = ['P',str(self.pcs),str(self.I().pcs),str(self.M().I().pcs),str(self.M().pcs)]
        reference.append(entry)
        entry = ['R',str(self.R().pcs),str(self.I().R().pcs),str(self.M().I().R().pcs),str(self.M().R().pcs)]
        reference.append(entry)
        entry = ['QR',str(self.R().Q().pcs),str(self.I().R().Q().pcs),str(self.M().I().R().Q().pcs),
                 str(self.M().R().Q().pcs)]
        reference.append(entry)
        entry = ['Q',str(self.Q().pcs),str(self.I().Q().pcs),str(self.M().I().Q().pcs),str(self.M().Q().pcs)]
        reference.append(entry)
        star = pd.DataFrame(reference,columns=['','P','I','IM','M'])
        return(star)
    def star(self):
#         star of the row in prime form
        reference = []
        entry = ['P',(self.pcs)]
        reference.append(entry)
        entry = ['I',(self.I().pcs)]
        reference.append(entry)
        entry = ['R',(self.R().pcs)]
        reference.append(entry)
        entry = ['Q',(self.Q().pcs)]
        reference.append(entry)
        entry = ['M',(self.M().pcs)]
        reference.append(entry)
        star = pd.DataFrame(reference,columns=['Op','Row'])
        return(star)
        
########### Network functions ###########

def pcsDictionary(Nc,order=0,TET=12,row=False,a=None):

    '''
    •	Generate the dictionary of all possible pcs of a given cardinality in a generalized musical space of TET pitches
    •	Nc (int)– cardinality
    •	order (logical)– if 0 returns pcs in prime form, if 1 returns pcs in normal order, if 2, returns pcs in normal 0 order
    •	row (logical)– if True build dictionary from tone row, if False, build dictionary from all combinatorial pcs of Nc cardinality given the totality of TET.
    •	if row = True, a is the list of pitches in the tone row (int)
    •	returns the dictionary as pandas DataFrame and the list of all Z-related pcs
    '''
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
    
    '''
    •	generate the network of pcs based on distances between interval vectors
    •	input_csv (str)– file containing the dictionary generated by pcsNetwork
    •	thup, thdw (float)– upper and lower thresholds for edge creation
    •	distance (str)– choice of norm in the musical space, default is 'euclidean'
    •	col = 2 – metric based on interval vector, col = 1 can be used for voice leading networks in spaces of fixed cardinality – NOT RECOMMENDED
    •	prob (float)– if ≠ 1, defines the probability of acceptance of any given edge
    •	in output it writes the nodes.csv and edges.csv as separate files in csv format
    '''

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
    comm.Barrier()
    
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
        dedges.to_csv('edges.csv',index=False)
    elif size == 1:
        os.rename('edges'+str(rank)+'.csv','edges.csv')

    return(dnodes,dedges)

def pcsEgoNetwork(label,input_csv,thup_e=5.0,thdw_e=0.1,thup=1.5,thdw=0.1,TET=12,distance='euclidean'):
    
    '''
    •	network generated from a focal node (ego) and the nodes to whom ego is directly connected to (alters)
    •	label (str)– label of the ego node
    •	thup_e, thdw_e (float) - upper and lower thresholds for edge creation from ego node
    •	thup, thdw (float)– upper and lower thresholds for edge creation among alters
    •	in output it writes the nodes_ego.csv, edges_ego.csv and edges_alters.csv as separate files in csv format
    '''
    
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
    
    return
    
def vLeadNetwork(input_csv,thup=1.5,thdw=0.1,TET=12,w=True,distance='euclidean',prob=1):
    
    '''
    •	generation of the network of all minimal voice leadings in a generalized musical space of TET pitches – based on the minimal distance operators
    •	input_csv (str)– file containing the dictionary generated by pcsNetwork
    •	thup, thdw (float)– upper and lower thresholds for edge creation
    •	w (logical) – if True it writes the nodes.csv and edges.csv files in csv format
    •	returns nodes and edges tables as pandas DataFrames
    '''

    start=time.time()    
    # Create network of minimal voice leadings from the pcsDictionary
    
    df = pd.read_csv(input_csv)
    df = np.asarray(df)

    # write csv for nodes
    dnodes = pd.DataFrame(df[:,0],columns=['Label'])
    if w: dnodes.to_csv('nodes.csv',index=False)
    #dnodes.to_json('nodes.json')
    
    # find edges according to a metric - allows for non-bijective voice leading
    N = df[:,1].shape[0]
    dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
    np.random.seed(int(time.process_time()*10000))
    for i in range(N):
        vector_i  = np.asarray(list(map(int,re.findall('\d+',df[i,1]))))
        for j in range(i,N):
            vector_j  = np.asarray(list(map(int,re.findall('\d+',df[j,1]))))
            if vector_i.shape[0] == vector_j.shape[0]:
                pair,_ = minimalDistance(vector_i,vector_j,TET,distance)
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
                    dist[l],_ = minimalDistance(a,r[l])
                pair = min(dist)
            if pair <= thup and pair >= thdw:
                if prob == 1:
                    tmp = pd.DataFrame([[str(i),str(j),str(1/pair)]],columns=['Source','Target','Weight'])
                    dedges = dedges.append(tmp)
                else:
                    r = np.random.rand()
                    if r <= prob:
                        tmp = pd.DataFrame([[str(i),str(j),str(1/pair)]],columns=['Source','Target','Weight'])
                        dedges = dedges.append(tmp)
                    else:
                        pass

    # write csv for edges
    if w: dedges.to_csv('edges.csv',index=False)

    return(dnodes,dedges)
    
def vLeadNetworkVec(input_csv,thup=1.1,thdw=0.1,TET=12,w=True,distance='euclidean'):
    
    # Create network of minimal voice leadings from the pcsDictionary
    # vector version

    df = pd.read_csv(input_csv)
    df = np.asarray(df)

    # write csv for nodes
    dnodes = pd.DataFrame(df[:,0],columns=['Label'])
    if w: dnodes.to_csv('nodes.csv',index=False)

    # find edges according to a metric
    N = df[:,1].shape[0]
    dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
    vector_i = np.zeros((N,len(list(map(int,re.findall('\d+',df[0,1]))))),dtype=int)
    pair = np.zeros((N,N),dtype=float)
    dis = np.zeros((N,N),dtype=float)
    # vector of pcs
    for i in range(N):
        vector_i[i] = np.asarray(list(map(int,re.findall('\d+',df[i,1]))))

    # vectors of distances
    for i in range(N):
        pair[i,:] = minimalDistanceVec(vector_i,np.roll(vector_i,-i,axis=0),TET,distance)

    for i in range(N):
        dis += np.diag(pair[i,:(N-i)],k=i)

    ix,iy = np.nonzero(dis)
    for n in range(ix.shape[0]):
        if dis[ix[n],iy[n]] < thup and dis[ix[n],iy[n]] > thdw:
            tmp = pd.DataFrame([[str(ix[n]),str(iy[n]),str(1/dis[ix[n],iy[n]])]],columns=['Source','Target','Weight'])
            dedges = dedges.append(tmp)

    # write csv for edges
    if w: dedges.to_csv('edges.csv',index=False)
    
    return(dnodes,dedges)
    
def vLeadNetworkByName(input_csv,name,TET=12,w=True,distance='euclidean',prob=1):
    
    '''
    •	generation of the network of all minimal voice leadings in a generalized musical space of TET pitches – based on the minimal distance operators - selects edges by operator name
    •	input_csv (str)– file containing the dictionary generated by pcsNetwork
    •	name – name of the operator as string: 'O(..l,m,n...)'
    •	w (logical) – if True it writes the nodes.csv and edges.csv files in csv format
    •	returns nodes and edges tables as pandas DataFrames
    '''

    start=time.time()    
    # Create network of minimal voice leadings from the pcsDictionary
    
    df = pd.read_csv(input_csv)
    df = np.asarray(df)

    # write csv for nodes
    dnodes = pd.DataFrame(df[:,0],columns=['Label'])
    if w: dnodes.to_csv('nodes.csv',index=False)
    #dnodes.to_json('nodes.json')
    
    # find edges according to a metric - allows for non-bijective voice leading
    N = df[:,1].shape[0]
    dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
    np.random.seed(int(time.process_time()*10000))
    for i in range(N):
        vector_i  = np.asarray(list(map(int,re.findall('\d+',df[i,1]))))
        for j in range(i,N):
            vector_j  = np.asarray(list(map(int,re.findall('\d+',df[j,1]))))
            if vector_i.shape[0] == vector_j.shape[0]:
                dis,_ = minimalDistance(vector_i,vector_j,TET,distance)
                pair = opsCheckByName(vector_i,vector_j,name,TET)
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
                    dist[l],_=minimalDistance(a,r[l])
                imin = np.argmin(dist)
                pair = opsCheckByName(a,r[imin],name,TET)
                dis = min(dist)
            if pair:
                if prob == 1:
                    tmp = pd.DataFrame([[str(i),str(j),str(1/dis)]],columns=['Source','Target','Weight'])
                    dedges = dedges.append(tmp)
                else:
                    r = np.random.rand()
                    if r <= prob:
                        tmp = pd.DataFrame([[str(i),str(j),str(1/dis)]],columns=['Source','Target','Weight'])
                        dedges = dedges.append(tmp)
                    else:
                        pass

    # write csv for edges
    if w: dedges.to_csv('edges.csv',index=False)

    return(dnodes,dedges)

def vLeadNetworkByNameVec(input_csv,name,TET=12,w=True,distance='euclidean'):
    
    # Create network of minimal voice leadings from the pcsDictionary
    # vector version - only bijective

    df = pd.read_csv(input_csv)
    df = np.asarray(df)

    # write csv for nodes
    dnodes = pd.DataFrame(df[:,0],columns=['Label'])
    if w: dnodes.to_csv('nodes.csv',index=False)

    # find edges according to a metric
    N = df[:,1].shape[0]
    dedges = pd.DataFrame(None,columns=['Source','Target','Weight'])
    vector_i = np.zeros((N,len(list(map(int,re.findall('\d+',df[0,1]))))),dtype=int)
    disx = np.zeros((N,N),dtype=float)
    pairx = np.zeros((N,N),dtype=bool)
    dis = np.zeros((N,N),dtype=float)
    pair = np.zeros((N,N),dtype=bool)
    # vector of pcs
    for i in range(N):
        vector_i[i] = np.asarray(list(map(int,re.findall('\d+',df[i,1]))))
    # matrix of distances
    for i in range(N):
        disx[i,:] = minimalDistanceVec(vector_i,np.roll(vector_i,-i,axis=0),TET,distance)
        pairx[i,:] = opsCheckByNameVec(vector_i,np.roll(vector_i,-i,axis=0),name,TET)

    for i in range(N):
        dis += np.diag(disx[i,:(N-i)],k=i)
        pair += np.diag(pairx[i,:(N-i)],k=i)
        
    ix,iy = np.nonzero(dis)
    for n in range(ix.shape[0]):
        if pair[ix[n],iy[n]]:
            tmp = pd.DataFrame([[str(ix[n]),str(iy[n]),str(1/dis[ix[n],iy[n]])]],columns=['Source','Target','Weight'])
            dedges = dedges.append(tmp)

    # write csv for edges
    if w: dedges.to_csv('edges.csv',index=False)
    
    return(dnodes,dedges)

def scoreNetwork(seq,TET=12,general=False,ntx=False):
    
    ''' 
    •	generates the directional network of chord progressions from any score in musicxml format
    •	seq (int) – list of pcs for each chords extracted from the score
    •	use readScore() to import the score data as sequence
    '''
    # build the directional network of the full progression in the chorale

    dedges = pd.DataFrame(None,columns=['Source','Target','Weight','Label'])
    dnodes = pd.DataFrame(None,columns=['Label'])
    for n in range(len(seq)):
        p = PCSet(np.asarray(seq[n]),TET)
        if TET == 12:
            if p.pcs.shape[0] == 1:
                nn = ''.join(m21.chord.Chord(p.pcs.tolist()).pitchNames)
            else:
                nn = ''.join(m21.chord.Chord(p.normalOrder().tolist()).pitchNames)
            nameseq = pd.DataFrame([[str(nn)]],columns=['Label'])
        elif TET == 24:
            dict24 = {'C':0,'C~':1,'C#':2,'D-':2,'D`':3,'D':4,'D~':5,'D#':6,'E-':6,'E`':7,'E':8,
                                'E~':9,'F`':9,'F':10,'F~':11,'F#':12,'G-':12,'G`':13,'G':14,'G~':15,'G#':16,
                                'A-':16,'A`':17,'A':18,'A~':19,'A#':20,'B-':20,'B`':21,'B':22,'B~':23,'C`':23}
            tmp = []
            for i in p.pcs:
                tmp.append(list(dict24.keys())[list(dict24.values()).index(i)]) 
            nameseq = pd.DataFrame([[''.join(tmp)]],columns=['Label'])
        dnodes = dnodes.append(nameseq)
    df = np.asarray(dnodes)
    dnodes = pd.DataFrame(None,columns=['Label'])
    dcounts = pd.DataFrame(None,columns=['Label','Counts'])
    dff,idx,cnt = np.unique(df,return_inverse=True,return_counts=True)
    for n in range(dff.shape[0]):
        nameseq = pd.DataFrame([[str(dff[n])]],columns=['Label'])
        dnodes = dnodes.append(nameseq)
        namecnt = pd.DataFrame([[str(dff[n]),cnt[n]]],columns=['Label','Counts'])
        dcounts = dcounts.append(namecnt)

    for n in range(1,len(seq)):
        if len(seq[n-1]) == len(seq[n]):
            a = np.asarray(seq[n-1])
            b = np.asarray(seq[n])
            pair,r = minimalDistance(a,b)
        else:
            if len(seq[n-1]) > len(seq[n]):
                a = np.asarray(seq[n-1])
                b = np.asarray(seq[n])
                pair,r = minimalNoBijDistance(a,b)
            else: 
                b = np.asarray(seq[n-1])
                a = np.asarray(seq[n])
                pair,r = minimalNoBijDistance(a,b)
        if pair != 0:
            if general == False:
                tmp = pd.DataFrame([[str(idx[n-1]),str(idx[n]),str(1/pair),opsName(a,r,TET)]],
                                    columns=['Source','Target','Weight','Label'])
            else:
                tmp = pd.DataFrame([[str(idx[n-1]),str(idx[n]),str(1/pair),generalizedOpsName(a,r,TET)[1]]],
                                    columns=['Source','Target','Weight','Label'])
            dedges = dedges.append(tmp)
    
    if ntx:
        # evaluate average degree and modularity
        gbch = nx.from_pandas_edgelist(dedges,'Source','Target',['Weight','Label'],create_using=nx.DiGraph())
        gbch_u = nx.from_pandas_edgelist(dedges,'Source','Target',['Weight','Label'])
        # modularity 
        part = cm.best_partition(gbch_u)
        modul = cm.modularity(part,gbch_u)
        # average degree
        nnodes=gbch.number_of_nodes()
        avg = 0
        for node in gbch.in_degree():
            avg += node[1]
        avgdeg = avg/float(nnodes)
        return(dnodes,dedges,dcounts,avgdeg,modul,gbch,gbch_u)
    else:
        return(dnodes,dedges,dcounts)

def scoreSubNetwork(seq,start=0,end=10,TET=12,general=False,ntx=False,grphtype='directed'):
    
    ''' 
    •	generates the directional network of chord progressions from any score in musicxml format
    •	seq (int) – list of pcs for each chords extracted from the score
    •	use readScore() to import the score data as sequence
    '''
    # build the directional network of the full progression in the chorale

    dedges = pd.DataFrame(None,columns=['Source','Target','Weight','Label'])
    dnodes = pd.DataFrame(None,columns=['Label'])
    for n in range(len(seq)):
        p = PCSet(np.asarray(seq[n]),TET)
        if TET == 12:
            if p.pcs.shape[0] == 1:
                nn = ''.join(m21.chord.Chord(p.pcs.tolist()).pitchNames)
            else:
                nn = ''.join(m21.chord.Chord(p.normalOrder().tolist()).pitchNames)
            nameseq = pd.DataFrame([[str(nn)]],columns=['Label'])
        elif TET == 24:
            dict24 = {'C':0,'C~':1,'C#':2,'D-':2,'D`':3,'D':4,'D~':5,'D#':6,'E-':6,'E`':7,'E':8,
                                'E~':9,'F`':9,'F':10,'F~':11,'F#':12,'G-':12,'G`':13,'G':14,'G~':15,'G#':16,
                                'A-':16,'A`':17,'A':18,'A~':19,'A#':20,'B-':20,'B`':21,'B':22,'B~':23,'C`':23}
            tmp = []
            for i in p.pcs:
                tmp.append(list(dict24.keys())[list(dict24.values()).index(i)]) 
            nameseq = pd.DataFrame([[''.join(tmp)]],columns=['Label'])
        dnodes = dnodes.append(nameseq)
    df = np.asarray(dnodes)
    dnodes = pd.DataFrame(None,columns=['Label'])
    dcounts = pd.DataFrame(None,columns=['Label','Counts'])
    dff,idx,cnt = np.unique(df,return_inverse=True,return_counts=True)
    for n in range(dff.shape[0]):
        nameseq = pd.DataFrame([[str(dff[n])]],columns=['Label'])
        dnodes = dnodes.append(nameseq)
        namecnt = pd.DataFrame([[str(dff[n]),cnt[n]]],columns=['Label','Counts'])
        dcounts = dcounts.append(namecnt)

    for n in range(1,len(seq)):
        if len(seq[n-1]) == len(seq[n]):
            a = np.asarray(seq[n-1])
            b = np.asarray(seq[n])
            pair,r = minimalDistance(a,b)
        else:
            if len(seq[n-1]) > len(seq[n]):
                a = np.asarray(seq[n-1])
                b = np.asarray(seq[n])
                pair,r = minimalNoBijDistance(a,b)
            else: 
                b = np.asarray(seq[n-1])
                a = np.asarray(seq[n])
                pair,r = minimalNoBijDistance(a,b)
        if pair != 0:
            if general == False:
                if n >= start and n < end:
                    tmp = pd.DataFrame([[str(idx[n-1]),str(idx[n]),str(1/pair),opsName(a,r,TET)]],
                                        columns=['Source','Target','Weight','Label'])
#                 else:
#                     tmp = pd.DataFrame([[str(idx[n-1]),str(idx[n]),str(0),opsName(a,r,TET)]],
#                                         columns=['Source','Target','Weight','Label'])
                    dedges = dedges.append(tmp)
            else:
                if n >= start and n < end:
                    tmp = pd.DataFrame([[str(idx[n-1]),str(idx[n]),str(1/pair),generalizedOpsName(a,r,TET)[1]]],
                                        columns=['Source','Target','Weight','Label'])
#                 else:
#                     tmp = pd.DataFrame([[str(idx[n-1]),str(idx[n]),str(0),generalizedOpsName(a,r,TET)[1]]],
#                                         columns=['Source','Target','Weight','Label'])
                    dedges = dedges.append(tmp)
    
    if ntx:
        # evaluate average degree and modularity
        if grphtype == 'directed':
            gbch = nx.from_pandas_edgelist(dedges,'Source','Target',['Weight','Label'],create_using=nx.DiGraph())
        elif grphtype == 'multi':
            gbch = nx.from_pandas_edgelist(dedges,'Source','Target',['Weight','Label'],create_using=nx.MultiDiGraph())
        else:
            print('no graph type specified')
            sys.exit()
        gbch_u = nx.from_pandas_edgelist(dedges,'Source','Target',['Weight','Label'])
        # modularity 
        part = cm.best_partition(gbch_u)
        modul = cm.modularity(part,gbch_u)
        # average degree
        nnodes=gbch.number_of_nodes()
        avg = 0
        for node in gbch.in_degree():
            avg += node[1]
        avgdeg = avg/float(nnodes)
        return(dnodes,dedges,dcounts,avgdeg,modul,gbch,gbch_u)
    else:
        return(dnodes,dedges,dcounts)

def scoreNetworkName(seq,TET=12,general=False,ntx=False):
    
    ''' 
    •	generates the directional network of chord progressions from any score in musicxml format
    •	seq (int) – list of pcs for each chords extracted from the score
    •	use readScore() to import the score data as sequence
    '''
    # build the directional network of the full progression in the chorale

    dedges = pd.DataFrame(None,columns=['Source','Target','Weight','Label'])
    dnodes = pd.DataFrame(None,columns=['Label'])
    for n in range(len(seq)):
        p = PCSet(np.asarray(seq[n]),TET)
        if TET == 12:
            if p.pcs.shape[0] == 1:
                nn = ''.join(m21.chord.Chord(p.pcs.tolist()).pitchNames)
            else:
                nn = ''.join(m21.chord.Chord(p.normalOrder().tolist()).pitchNames)
            nameseq = pd.DataFrame([[str(nn)]],columns=['Label'])
        elif TET == 24:
            dict24 = {'C':0,'C~':1,'C#':2,'D-':2,'D`':3,'D':4,'D~':5,'D#':6,'E-':6,'E`':7,'E':8,
                                'E~':9,'F`':9,'F':10,'F~':11,'F#':12,'G-':12,'G`':13,'G':14,'G~':15,'G#':16,
                                'A-':16,'A`':17,'A':18,'A~':19,'A#':20,'B-':20,'B`':21,'B':22,'B~':23,'C`':23}
            tmp = []
            for i in p.pcs:
                tmp.append(list(dict24.keys())[list(dict24.values()).index(i)]) 
            nameseq = pd.DataFrame([[''.join(tmp)]],columns=['Label'])
        dnodes = dnodes.append(nameseq)
    df = np.asarray(dnodes)
    dnodes = pd.DataFrame(None,columns=['Label'])
    dcounts = pd.DataFrame(None,columns=['Label','Counts'])
    dff,idx,cnt = np.unique(df,return_inverse=True,return_counts=True)
    for n in range(dff.shape[0]):
        nameseq = pd.DataFrame([[str(dff[n])]],columns=['Label'])
        dnodes = dnodes.append(nameseq)
        namecnt = pd.DataFrame([[str(dff[n]),cnt[n]]],columns=['Label','Counts'])
        dcounts = dcounts.append(namecnt)

    for n in range(1,len(seq)):
        if len(seq[n-1]) == len(seq[n]):
            a = np.asarray(seq[n-1])
            b = np.asarray(seq[n])
            pair,r = minimalDistance(a,b)
        else:
            if len(seq[n-1]) > len(seq[n]):
                a = np.asarray(seq[n-1])
                b = np.asarray(seq[n])
                pair,r = minimalNoBijDistance(a,b)
            else: 
                b = np.asarray(seq[n-1])
                a = np.asarray(seq[n])
                pair,r = minimalNoBijDistance(a,b)
        if pair != 0:
            if general == False:
                tmp = pd.DataFrame([[str(seq[n-1]),str(seq[n]),str(1/pair),opsName(a,r,TET)]],
                                    columns=['Source','Target','Weight','Label'])
            else:
                tmp = pd.DataFrame([[str(seq[n-1]),str(seq[n]),str(1/pair),generalizedOpsName(a,r,TET)[1]]],
                                    columns=['Source','Target','Weight','Label'])
            dedges = dedges.append(tmp)
    
    if ntx:
        # evaluate average degree and modularity
        gbch = nx.from_pandas_edgelist(dedges,'Source','Target',['Weight','Label'],create_using=nx.DiGraph())
        gbch_u = nx.from_pandas_edgelist(dedges,'Source','Target',['Weight','Label'])
        # modularity 
        part = cm.best_partition(gbch_u)
        modul = cm.modularity(part,gbch_u)
        # average degree
        nnodes=gbch.number_of_nodes()
        avg = 0
        for node in gbch.in_degree():
            avg += node[1]
        avgdeg = avg/float(nnodes)
        return(dnodes,dedges,dcounts,avgdeg,modul,gbch,gbch_u)
    else:
        return(dnodes,dedges,dcounts)

def scoreDictionary(seq,TET=12):
    '''
    •	build the dictionary of pcs in any score in musicxml format
    •	use readScore() to import the score data as sequence
    '''
    s = Remove(seq)
    v = []
    name = []
    prime = []
    if TET == 12:
        for i in range(len(s)):
            p = PCSet(np.asarray(s[i][:]),TET=TET)
            v.append(p.intervalVector())
            name.append(''.join(m21.chord.Chord(p.normalOrder().tolist()).pitchNames))
            prime.append(np.array2string(p.normalOrder(),separator=',').replace(" ",""))
        vector = np.asarray(v)
        name = np.asarray(name)
    elif TET == 24:
        dict24 = {'C':0,'C~':1,'C#':2,'D-':2,'D`':3,'D':4,'D~':5,'D#':6,'E-':6,'E`':7,'E':8,
                            'E~':9,'F`':9,'F':10,'F~':11,'F#':12,'G-':12,'G`':13,'G':14,'G~':15,'G#':16,
                            'A-':16,'A`':17,'A':18,'A~':19,'A#':20,'B-':20,'B`':21,'B':22,'B~':23,'C`':23}
        for i in range(len(s)):
            p = PCSet(np.asarray(s[i][:]),TET=TET)
            v.append(p.intervalVector())
            tmp = []
            for i in p.pcs:
                tmp.append(list(dict24.keys())[list(dict24.values()).index(i)]) 
            name.append(''.join(tmp))
            prime.append(np.array2string(p.normalOrder(),separator=',').replace(" ",""))
        vector = np.asarray(v)
        name = np.asarray(name)
    else:
        print('temperament needs to be added')
        sys.exit()

    # Create dictionary of pitch class sets
    reference = []
    for n in range(len(name)):
        entry = [name[n],prime[n],
                np.array2string(vector[n,:],separator=',').replace(" ","")]
        reference.append(entry)

    dictionary = pd.DataFrame(reference,columns=['class','pcs','interval'])
    
    return(dictionary)

def readScore(input_xml,TET=12,music21=False,show=False):
    '''
    •	read a score in musicxml format
    •	returns the sequence of chords
    '''
    if TET == 12:
        if music21: 
            score = m21.corpus.parse(input_xml)
            try:
                score = score.mergeScores()
            except:
                pass
        else:
            score = m21.converter.parse(input_xml)
        chords = score.chordify()
        if show: chords.show()
        seq = []
        for c in chords.recurse().getElementsByClass('Chord'):
            seq.append(c.normalOrder)
        return(seq,chords)
    elif TET == 24:
        dict24 = {'C':0,'C~':1,'C#':2,'D-':2,'D`':3,'D':4,'D~':5,'D#':6,'E-':6,'E`':7,'E':8,
                            'E~':9,'F`':9,'F':10,'F~':11,'F#':12,'G-':12,'G`':13,'G':14,'G~':15,'G#':16,
                            'A-':16,'A`':17,'A':18,'A~':19,'A#':20,'B-':20,'B`':21,'B':22,'B~':23,'C`':23} 
                            
        score = m21.converter.parse(input_xml)
        chords = score.chordify()
        if show: chords.show()
        seq = []
        for c in chords.recurse().getElementsByClass('Chord'):
            seq.append(str(c))

        clean = []
        for n in range(len(seq)):
            line = ''.join(i for i in seq[n] if not i.isdigit()).split()[1:]
            c = []
            for l in line:
                c.append(dict24[re.sub('>', '', l)])
            clean.append(PCSet(c,TET=24).normalOrder().tolist())
        return(clean,chords)
    else:
        print('temperament needs to be added')
        sys.exit()
    return

def WRITEscore(file,nseq,rseq,w=None,outxml='./music',outmidi='./music'):
    import pydub as pb
    obj = pb.AudioSegment.from_file(file)
    m = m21.stream.Measure()
    for i in range(nseq.shape[0]):
        n = m21.note.Note(nseq[i])
        n.duration = m21.duration.Duration(4*rseq[i])
        m.append(n)
    m.append(m21.meter.SenzaMisuraTimeSignature('0'))
    t = m21.meter.bestTimeSignature(m)
    bpm = int(np.round(60/(obj.duration_seconds/np.round((t.numerator/t.denominator),0)/4),0))
    m.insert(0,m21.tempo.MetronomeMark(number=bpm))
    if w == 'musicxml':
        m.write('musicxml',outxml+'.xml')
    elif w == 'MIDI':
        m.write('midi',outmidi+'.mid')
    else:
        m.show()

def WRITEscoreNoTime(nseq,rseq,w=None,outxml='./music',outmidi='./music'):
    
    m = m21.stream.Measure()
    for i in range(nseq.shape[0]):
        n = m21.note.Note(nseq[i])
        n.duration = m21.duration.Duration(4*rseq[i])
        m.append(n)
    m.append(m21.meter.SenzaMisuraTimeSignature('0'))
    
    if w == True:
        m.show('musicxml')
    elif w == 'MIDI':
        m.write('midi',outmidi+'.mid')
    else:
        m.show()

def WRITEscoreOps(nseq,w=None,outxml='./music',outmidi='./music',keysig=None,abs=False):
    try:
        ntot = nseq.shape[0]
    except:
        ntot = len(nseq)
    m = m21.stream.Stream()
    m.append(m21.meter.TimeSignature('4/4'))
    for i in range(ntot):
        ch = np.copy(nseq[i])
        for n in range(1,len(ch)):
            if ch[n] < ch[n-1]: ch[n] += 12
        ch += 60
        n = m21.chord.Chord(ch.tolist())
        if i < ntot-1: 
            n.addLyric(str(i)+' '+generalizedOpsName(nseq[i],nseq[i+1])[1])
            if abs:
                if len(nseq[i]) == len(nseq[i+1]):
                    n.addLyric(str(i)+' '+opsName(nseq[i],nseq[i+1]))
                else:
                    r = generalizedOpsName(nseq[i],nseq[i+1])[0]
                    if len(nseq[i]) > len(nseq[i+1]):
                        n.addLyric(str(i)+' '+opsName(nseq[i],r))
                    else:
                        n.addLyric(str(i)+' '+opsName(r,nseq[i+1]))
        if keysig != None:
            rn = m21.roman.romanNumeralFromChord(n, m21.key.Key(keysig))
            n.addLyric(str(rn.figure))
        m.append(n)    
    if w == True:
        m.show('musicxml')
    elif w == 'MIDI':
        m.write('midi',outmidi+'.mid')
    else:
        m.show()

def extractByString(name,label,string):
    '''
    •	extract rows of any dictionary (from csv file or pandas DataFrame) according to a particular string in column 'label'
    •	name (str or pandas DataFrame) – name of the dictionary
    •	string (str) – string to find in column
    •	label (str) – name of column of string
    '''
    if type(name) is str: 
        df = pd.read_csv(name)
    else:
        df = name
    return(df[df[label].str.contains(string)])
    
def minimalDistance(a,b,TET=12,distance='euclidean'):
    '''
    •	calculates the minimal distance between two pcs of same cardinality (bijective)
    •	a,b (int) – pcs as numpy arrays or lists
    '''
    a = np.asarray(a)
    b = np.asarray(b)
    n = a.shape[0]
    if a.shape[0] != b.shape[0]:
        print('dimension of arrays must be equal')
        sys.exit()
    a = np.sort(a)
    iTET = np.vstack([np.identity(n,dtype=int)*TET,-np.identity(n,dtype=int)*TET])
    iTET = np.vstack([iTET,np.zeros(n,dtype=int)])
    diff = np.zeros(2*n+1,dtype=float)
    v = []
    for i in range(2*n+1):
        r = np.sort(b - iTET[i])
        diff[i] = sklm.pairwise_distances(a.reshape(1, -1),r.reshape(1, -1),metric=distance)[0]
        v.append(r)
    imin = np.argmin(diff)
    return(diff.min(),np.asarray(v[imin]).astype(int))
    
def minimalNoBijDistance(a,b,TET=12,distance='euclidean'):
    '''
    •	calculates the minimal distance between two pcs of different cardinality (non bijective) – uses minimalDistance()
    •	a,b (int) – pcs as lists or numpy arrays
    '''
    a = np.asarray(a)
    b = np.asarray(b)
    ndif = np.sort(np.array([a.shape[0],b.shape[0]]))[1] - np.sort(np.array([a.shape[0],b.shape[0]]))[0]
    c = np.asarray(list(iter.combinations_with_replacement(b,ndif)))
    r = np.zeros((c.shape[0],a.shape[0]))
    for l in range(c.shape[0]):
        r[l,:b.shape[0]] = b
        r[l,b.shape[0]:] = c[l]
    dist = np.zeros(r.shape[0])
    for l in range(r.shape[0]):
        dist[l],_=minimalDistance(a,r[l],TET,distance)
    imin = np.argmin(dist)
        
    return(min(dist),r[imin].astype(int))

def minimalDistanceVec(a,b,TET=12,distance='euclidean'):
    '''
    •	calculates the minimal distance between two pcs of same cardinality (bijective)
    •	a,b (int) – pcs as numpy arrays or lists
    •   vector version
    '''
    a = np.asarray(a)
    b = np.asarray(b)
    N = a.shape[0]
    n = a.shape[1]
    if a.shape[0] != b.shape[0]:
        print('dimension of arrays must be equal')
        sys.exit()
    a = np.sort(a)
    iTET = np.vstack([np.identity(n,dtype=int)*TET,-np.identity(n,dtype=int)*TET])
    iTET = np.vstack([iTET,np.zeros(n,dtype=int)])
    diff = np.zeros((N,2*n+1),dtype=float)
    for i in range(2*n+1):
        r = np.sort(b - iTET[i])
        diff[:,i] = np.sqrt(np.sum((a-r)**2,axis=1))

    return(np.amin(diff,axis=1))


def opsDictionary(distance):
    # dictionary of names of distance operators - OBSOLETE
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
    
def opsName(a,b,TET=12):
    # given two vectors returns the name of the minimal distance operator that connects them
    a = np.sort(a)
    b = np.sort(b)   
    d = np.zeros((b.shape[0]),dtype=int) 
    for n in range(b.shape[0]):
        c = np.roll(b,n)
        diff = a-c
        for i in range(diff.shape[0]):
            if diff[i] >= int(TET/2):
                diff[i] -= TET
            if diff[i] < -int(TET/2):
                diff[i] += TET
        diff = np.abs(diff)
        d[n] = diff.dot(diff)
    nmin = np.argmin(d)
    b = np.roll(b,nmin)
    diff = a-b
    for i in range(diff.shape[0]):
        if diff[i] >= int(TET/2):
            diff[i] -= TET
        if diff[i] < -int(TET/2):
            diff[i] += TET
    diff = np.sort(np.abs(diff))

    return('O('+np.array2string(np.trim_zeros(diff),separator=',').replace(" ","").replace("[","").replace("]","")+')')
    
def opsCheckByName(a,b,name,TET=12):
    # given two vectors returns check if the connecting operator is the one sought for
    opname = opsName(a,b,TET)
    if opname != name:
        return(False)
    else:
        return(True)

def opsDistance(name):
    # returns distance for given operator
    opname = np.asarray(' '.join(i for i in name if i.isdigit()).split())
    opdist = np.sqrt(np.sum(np.asarray([list(map(int, x)) for x in opname]).reshape(1,-1)[0]*
        np.asarray([list(map(int, x)) for x in opname]).reshape(1,-1)[0]))
    return(name,opdist)

def opsNameVec(a,b,TET=12):
    # given two arrays of vectors returns the array of the names of the operators that connects them
    # vector version
    a = np.sort(a,axis=1)
    b = np.sort(b,axis=1)
    d = np.zeros((b.shape[0],b.shape[1]),dtype=int) 
    diff = np.zeros((b.shape[0],b.shape[1]),dtype=int)
    nmin = np.zeros((b.shape[0],b.shape[1]),dtype=int)

    for n in range(b.shape[1]):
        c = np.roll(b,n,axis=1)
        diff = a-c
        aux = np.where(diff >= int(TET/2),diff-TET,diff)
        diff = np.abs(np.where(aux < -int(TET/2),aux+TET,aux)) 

        d[:,n] = np.sum(diff*diff,axis=1)
    nmin = np.argmin(d,axis=1)
    for i in range(b.shape[0]):
        b[i] = np.roll(b[i],nmin[i])
        diff[i] = a[i]-b[i]
    aux = np.where(diff >= int(TET/2),diff-TET,diff)
    diff = np.sort(np.abs(np.where(aux < -int(TET/2),aux+TET,aux)))
    name = []
    for i in range(b.shape[0]):
        name.append('O('+np.array2string(np.trim_zeros(diff[i]),separator=',')\
                    .replace(" ","").replace("[","").replace("]","")+')')

    return(np.asarray(name))

def generalizedOpsName(a,b,TET=12):
# generalizes the operator name function for no-bijective chord progression
    if len(a) == len(b):
        return(a,opsNameFull(a,b,TET))
    else:
        if len(a) > len(b):
            pair,r = minimalNoBijDistance(a,b)
            return(r,opsNameFull(a,r,TET))
        else:
            pair,r = minimalNoBijDistance(b,a)
            return(r,opsNameFull(r,b,TET))

def opsNameFull(a,b,TET=12):
    # given two vectors returns the name of the normal ordered distance operator (R) that connects them
    a = PCSet(a,UNI=False).normalOrder()
    b = PCSet(b,UNI=False).normalOrder()   
    d = np.zeros((b.shape[0]),dtype=int) 
    for n in range(b.shape[0]):
        c = np.roll(b,n)
        diff = a-c
        for i in range(diff.shape[0]):
            if diff[i] >= int(TET/2):
                diff[i] -= TET
            if diff[i] < -int(TET/2):
                diff[i] += TET
        diff = np.abs(diff)
        d[n] = diff.dot(diff)
    nmin = np.argmin(d)
    b = np.roll(b,nmin)
    diff = b-a
    for i in range(diff.shape[0]):
        if diff[i] >= int(TET/2):
            diff[i] -= TET
        if diff[i] < -int(TET/2):
            diff[i] += TET

    return('O('+np.array2string(diff,separator=',').replace(" ","").replace("[","").replace("]","")+')')
    
def opsCheckByNameVec(a,b,name,TET=12):
    # given two vectors returns check if the connecting operator is the one sought for
    # vector version
    opname = opsNameVec(a,b,TET)
    opname = np.where(opname == name,True,False)
    return(opname)
    
def opsHistogram(values,counts):
    ops = []
    for i in range(1,6):
        ops.append('O('+str(i)+')')
    for i in range(1,6):
        for j in range(i,6):
            ops.append('O('+str(i)+','+str(j)+')')
    for i in range(1,6):
        for j in range(i,6):
            for k in range(j,6):
                ops.append('O('+str(i)+','+str(j)+','+str(k)+')')
    for i in range(1,6):
        for j in range(i,6):
            for k in range(j,6):
                for l in range(k,6):
                    ops.append('O('+str(i)+','+str(j)+','+str(k)+','+str(l)+')')
    ops = np.array(ops)
    dist = np.zeros(ops.shape[0])
    for i in range(ops.shape[0]):
        dist[i] = opsDistance(ops[i])[1]
    idx = np.argsort(dist)
    ops = ops[idx]

    ops_dict = {}
    for i in range(len(ops)):
        ops_dict.update({ops[i]:0})

    for i in range(len(values)):
        ops_dict.update({values[i]:counts[i]})

    newvalues = np.asarray(list(ops_dict.keys()))
    newcounts = np.asarray(list(ops_dict.values()))
    return(newvalues,newcounts,ops_dict,dist[idx])
    
def generalizedOpsHistogram(values,counts):
    ops = []
    for i in range(-3,3):
        ops.append('O('+str(i)+')')
    for i in range(-3,3):
        for j in range(i,3):
            ops.append('O('+str(i)+','+str(j)+')')
    for i in range(-3,3):
        for j in range(i,3):
            for k in range(j,3):
                ops.append('O('+str(i)+','+str(j)+','+str(k)+')')
    for i in range(-3,3):
        for j in range(i,3):
            for k in range(j,3):
                for l in range(k,3):
                    ops.append('O('+str(i)+','+str(j)+','+str(k)+','+str(l)+')')
    ops = np.array(ops)
    dist = np.zeros(ops.shape[0])
    for i in range(ops.shape[0]):
        dist[i] = opsDistance(ops[i])[1]
    idx = np.argsort(dist)
    ops = ops[idx]

    ops_dict = {}
    for i in range(len(ops)):
        ops_dict.update({ops[i]:0})

    for i in range(len(values)):
        ops_dict.update({values[i]:counts[i]})

    newvalues = np.asarray(list(ops_dict.keys()))
    newcounts = np.asarray(list(ops_dict.values()))
    return(newvalues,newcounts,ops_dict,dist[idx])

    
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
    
def plotHarmonicTable(header,table,dictionary,height=7,width=12,colmap=plt.cm.Reds,coltxt='White',vmin=None,label=True,star=None):
    
    row = header[1:]
    col = header[1:]
    tab = np.array(table)[:,1:]

    norm = 0
    value = np.zeros((len(row),len(col)))
    for i in range(len(row)):
        for j in range(len(col)):
            try:
                value[i,j] = dictionary[tab[i,j]]
#                norm += value[i,j]
            except:
                value[i,j] = 0

#    value /= norm*0.01

    fig, ax = plt.subplots()
    if vmin == None:
        im = ax.imshow(value, aspect='auto',cmap=colmap)
    else:
        im = ax.imshow(value, aspect='auto',cmap=colmap,vmin=vmin)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(row)))
    ax.set_yticks(np.arange(len(col)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(row)
    ax.set_yticklabels(col)
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
             rotation_mode="anchor",fontsize=16)

    plt.setp(ax.get_yticklabels(), rotation=0, ha="right",
             rotation_mode="anchor",fontsize=16)

    ax.set_ylim(len(col)-0.5, -0.5)

    if label == True:
        for i in range(len(row)):
            for j in range(len(col)):
                if value[i,j] > 0:
                    if star != 'x':
                        text = ax.text(j, i, tab[i, j],
                                        ha="center", va="center", color=coltxt, fontsize=16)
                    else:
                        text = ax.text(j, i, 'x',
                                        ha="center", va="center", color=coltxt, fontsize=10)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('probability of progression', rotation=-90, va="center", fontsize=16, labelpad=22)
    _,vscale = np.histogram(Remove(np.sort(np.reshape(value,len(col)*len(row)))),bins=11)
    cbar.ax.set_yticklabels(vscale,fontsize=16)
    cbar.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

    fig.set_figheight(height)
    fig.set_figwidth(width)

    plt.show()

def lookupOps(ops,table,header,Pnumber=None,ch1=None):

    operator = ops
    tab = np.array(table)
    if Pnumber != None:
        try:
            print('Pnumber of operator '+operator+' =',Pnumber[operator],'\n')
        except:
            print('operator not found in Pnumber')
            return
    idx,idy = np.where(tab == operator)
    for n in range(len(idy)):
        if ch1 == None:
            print(str(header[idx[n]+1]).ljust(12),str(' ->\t'+header[idy[n]]).rjust(0))
        else:
            if str(header[idx[n]+1]) == ch1:
                print(str(header[idx[n]+1]).ljust(12),str(' ->\t'+header[idy[n]]).rjust(0))
        
def lookupProgr(ch1,ch2,table,header):

    tab = np.array(table)
    head = np.array(header)
    idx = np.where(head == ch1)
    idy = np.where(head == ch2)
    print(str(ch1).ljust(8),'->',tab[idx[0]-1,idy[0]][0],'->',str(ch2).rjust(8))
    
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
            linlaw = lambda x,a,b,L: a+x*b
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
    
def powerFit(Gx,mode='power_law',xmin=None,xmax=None,linear=False,indeg=True,undir=False,units=None):
    # set-up
    import pylab
    pylab.rcParams['xtick.major.pad']='24'
    pylab.rcParams['ytick.major.pad']='24'
    #pylab.rcParams['font.sans-serif']='Arial'


    from matplotlib import rc
    rc('font', family='sans-serif')
    rc('font', size=14.0)
    rc('text', usetex=False)


    from matplotlib.font_manager import FontProperties

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

def Remove(duplicate): 
    # function to remove duplicates from list
    final_list = [] 
    for num in duplicate: 
        if num not in final_list: 
            final_list.append(num) 
    return final_list 

def flatten(l, ltypes=(list, tuple)):
    # flatten a list of lists
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)
    
def init_list_of_objects(size):
    # initialize a list of list object
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() ) #different object reference each time
    return list_of_objects

def plotCurveY(y):
    stage=vp.canvas()
    f1 = vp.gcurve(color=vp.color.green)
    for n in range(y.shape[0]):
        f1.plot(pos=(n,y[n]))
                
def plotCurveXY(x,y):
    stage=vp.canvas()
    f1 = vp.gcurve(color=vp.color.blue)
    for n in range(y.shape[0]):
        f1.plot(pos=(x[n],y[n]))