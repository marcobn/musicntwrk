#
# MUSIC𝄞NTWRK
#
# A python library for pitch class set and rhythmic sequences classification and manipulation, 
# the generation of networks in generalized musical spaces, and the sonification of arbitrary data 
# See documentation at www.musicntwrk.com
#
# Copyright (C) 2018,2019,2020,2021 Marco Buongiorno Nardelli http://www.materialssoundmusic.com, mbn@unt.edu
# This file is distributed under the terms of the GNU General Public License. See the file `License' 
#in the root directory of the present distribution, or http://www.gnu.org/copyleft/gpl.txt .
#

import re, sys, time
import numpy as np
import music21 as m21
from functools import reduce
import fractions as fr
from math import gcd

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
#        if len(self.pcs) == 2:
#            return(self.pcs)

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

    def T(self,t=0):
        '''
        •	Transposition by t (int) units (modulo TET)
        '''
        return((self.pcs+t)%self.TET)
        
    def M(self,t=1):
        '''
        •	Multiplication by t (int) units (modulo TET)
        '''
        return(np.unique((self.pcs*t)%self.TET//1).astype(int))
        
    def multiplyBoulez(self,b):
        # Boulez pitch class multiplication of a x b
        ivec = self.LISVector()
        m = []
        for i in range(ivec.shape[0]-1):
            mm = (b+ivec[i])%self.TET
            m.append(mm.tolist())
        return(PCSet(Remove(flatten(m+b)),TET).normalOrder())
    
    def zeroOrder(self):
        '''
        •	transposed so that the first pitch is 0
        '''
        return((self.pcs-self.pcs[0])%self.TET)

    def I(self):
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
        self.pcs = self.I()
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

    def Op(self,name):
        # operate on the pcs with a generic distance operator
        
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
        
    def VLOp(self,name):
        # operate on the pcs with a normal-ordered voice-leading operator
        
        op = []
        for num in re.findall("[-\d]+", name):
            op.append(int(num))
        op = np.asarray(op)
        selfto = np.unique((self.pcs+op)%self.TET,axis=0)
        return(PCSet(selfto).normalOrder())
        
    def LISVector(self):
        '''
        •	Linear Interval Sequence Vector: sequence of intervals in an normal ordered pcs
        '''
        return((np.roll(self.normalOrder(),-1)-self.normalOrder())%self.TET)
    
    def intervals(self):
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
        Fname = m21.chord.Chord(self.primeForm().pcs.tolist()).forteClass
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
    
    def commonNamePitched(self):
        '''
        •	As above, for prime forms
        '''
        return(m21.chord.Chord(np.ndarray.tolist(self.normalOrder()[:])).pitchedCommonName)
        
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
            return(PCSetR(self.pcs-self.pcs[0],TET=self.TET))
#        if len(self.pcs) == 2:
#            return(PCSetR(self.pcs,TET=self.TET))

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

    def T(self,t=0):
        '''
        •	Transposition by t (int) units (modulo TET)
        '''
        return(PCSetR((self.pcs+t)%self.TET,TET=self.TET))
        
    def M(self,t=1):
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

    def I(self,pivot=0):
        '''
        •	I operation: (-pcs modulo TET)
        '''
        return(PCSetR((pivot-self.pcs)%self.TET,TET=self.TET))

    def primeForm(self):
        '''
        •	most compact normal 0 order between pcs and its I
        '''
        s_orig = self.pcs
        sn = np.sum((self.normalOrder().pcs-self.normalOrder().pcs[0])%self.TET)
        self.pcs = self.I().pcs
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

    def Op(self,name):
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
        
    def VLOp(self,name,norm=True):
        # operate on the pcs with a (normal-ordered )relational operator R({x})
        op = []
        for num in re.findall("[-\d]+", name):
            op.append(int(num))
        op = np.asarray(op)
        if norm:
            selfto = np.unique((self.normalOrder().pcs+op)%self.TET,axis=0)
            return(PCSetR(selfto,TET=self.TET).normalOrder())
        else:
            selfto = np.unique((self.pcs+op)%self.TET,axis=0)
            return(PCSetR(selfto,TET=self.TET))
    
    def NROp(self,ops=None):
        # operate on the triad with a Neo-Rienmannian Operator
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

    def opsNameVL(self,b,norm=True):
        # given two vectors returns the name of the (normal-ordered) voice-leading operator R that connects them
        if norm:
            a = self.normalOrder().pcs
            b = b.normalOrder().pcs  
        else:
            a = self.pcs
            b = b.pcs 
        d = np.zeros((b.shape[0]),dtype=int) 
        for n in range(b.shape[0]):
            c = np.roll(b,n)
            diff = a-c
            for i in range(diff.shape[0]):
                if diff[i] >= int(self.TET/2):
                    diff[i] -= self.TET
                if diff[i] < -int(self.TET/2):
                    diff[i] += self.TET
            diff = np.abs(diff)
            d[n] = diff.dot(diff)
        nmin = np.argmin(d)
        b = np.roll(b,nmin)
        diff = b-a
        for i in range(diff.shape[0]):
            if diff[i] >= int(self.TET/2):
                diff[i] -= self.TET
            if diff[i] < -int(self.TET/2):
                diff[i] += self.TET
        return('R('+np.array2string(diff,separator=',').replace(" ","").replace("[","").replace("]","")+')')

    def opsNameO(self,b):
        # given two vectors returns the name of the distance 
        # operator O that connects them
        a = np.sort(self.pcs)
        b = np.sort(b.pcs)
        d = np.zeros((b.shape[0]),dtype=int) 
        for n in range(b.shape[0]):
            c = np.roll(b,n)
            diff = a-c
            for i in range(diff.shape[0]):
                if diff[i] >= int(self.TET/2):
                    diff[i] -= self.TET
                if diff[i] < -int(self.TET/2):
                    diff[i] += self.TET
            diff = np.abs(diff)
            d[n] = diff.dot(diff)
        nmin = np.argmin(d)
        b = np.roll(b,nmin)
        diff = b-a
        for i in range(diff.shape[0]):
            if diff[i] >= int(self.TET/2):
                diff[i] -= self.TET
            if diff[i] < -int(self.TET/2):
                diff[i] += self.TET
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
    
    def intervals(self):
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

class PCmidiR:
    
    def __init__(self,midi,UNI=False,ORD=False,TET=12):
        '''
        •	midi –  MIDI number list or string(name+octave) separated by commas ('C4,D4,...')
        •	UNI (logical) – if True, eliminate duplicate pitches (default)
        •   ORD (logical) - if True, sorts the pcs in ascending order
        '''
        
        try:
            midi = midi.tolist()
        except:
            if not isinstance(midi,list):
                midi = [midi]
            else:
                pass
            
        if isinstance(midi[0],str) and len(midi) > 1:
            names = midi.copy()
            for m in range(len(midi)):
                midi[m] = m21.pitch.Pitch(midi[m]).ps
            self.pitches = names 
        elif isinstance(midi[0],str) and len(midi) == 1:
            midi = midi[0].split(',')
            names = midi.copy()
            for m in range(len(midi)):
                midi[m] = m21.pitch.Pitch(midi[m]).ps
            self.pitches = names 
            
        if UNI == True:
            self.midi = np.unique(midi)
        if ORD == True:
            self.midi = np.sort(midi)
        else:
            self.midi = np.asarray(midi)
            
        self.pcs = np.mod(midi,TET)
        
        if isinstance(midi[0],int) or isinstance(midi[0],float):
            pitches = []
            for m in range(len(midi)):
                pitches.append(str(m21.pitch.Pitch(midi[m])))
            self.pitches = pitches
            
        self.TET = TET
        
    def T(self,t=0):
        '''
        •	Transposition by t (int) units
        '''
        return(PCmidiR(self.midi+t))
    
    def zeroOrder(self):
        '''
        •	transposed so that the first pitch is 60 (middle C)
        '''
        return(PCmidiR(self.midi-self.midi[0])+60)
    
    def normalOrder(self):
        '''
        •	Order the pcs according to the most compact ascending scale in pitch-class space that spans 
            less than an octave by cycling permutations.
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
        return(PCmidiR(pcs_norm+60))
    
    
    def I(self,pivot=60):
        '''
        •	I operation
        '''
        return(PCmidiR((pivot-self.midi)+pivot))
    
    def VLOp(self,name):
        # operate on the pcs with a normal-ordered relational operator R({x})
        op = []
        for num in re.findall("[-\d]+", name):
            op.append(int(num))
        op = np.asarray(op)
        selfto = np.unique((self.midi+op),axis=0)
        return(PCmidiR(selfto))
    
    def opsNameVL(self,b):
        # given two vectors returns the name of the normal-ordered voice-leading operator R that connects them
        a = self.midi
        b = b.midi  
        diff = b-a
        return('R('+np.array2string(diff,separator=',').replace(" ","").replace("[","").replace("]","")+')')
    
    def intervals(self):
        '''
        •	Linear Interval Sequence Vector: sequence of intervals in an ordered pcs
        •	also known as step-interval vector (see Cohn, Neo-Riemannian Operations, 
            Parsimonious Trichords, and Their "Tonnetz" Representations,
            Journal of Music Theory, Vol. 41, No. 1 (Spring, 1997), pp. 1-66)
        '''
        return((np.roll(self.midi,-1)-self.midi))
    
    def displayNotes(self,show=True,xml=False,chord=False):
        '''
        •	Display pcs in score in musicxml format. If chord is True 
            displays the note cluster
        '''
        fac = self.TET/12
        if  not chord:
            s = m21.stream.Stream()
            for i in range(self.midi.shape[0]):
                s.append(m21.note.Note(self.midi[i]/fac))
            if show: s.show()
            if xml: s.show('musicxml')
            return(s)
        else:
            ch = []
            for i in range(self.midi.shape[0]):
                ch.append(m21.note.Note(self.midi[i]/fac))
            c = m21.chord.Chord(ch)
            if show: c.show()
            if xml: c.show('musicxml')
            return(c)

class PCSrow:
#     Helper class for 12-tone rows operations (T,I,R,M,Q)
    def __init__(self,pcs,TET=12):
        self.pcs = np.array(pcs)%TET
        self.TET = TET
    def normalOrder(self):
        self.pcs -= self.pcs[0]
        return(PCSrow(self.pcs%self.TET))
    def intervals(self):
        return((np.roll(self.pcs,-1)-self.pcs)%self.TET)
    def T(self,t=0):
        return(PCSrow((self.pcs+t)%self.TET,TET=self.TET))
    def I(self,pivot=0):
        return(PCSrow((pivot-self.pcs)%self.TET,TET=self.TET))
    def R(self,t=0):
        return(PCSrow(self.pcs[::-1]).T(t))
    def Q(self):
        lisv = PCSrow(self.pcs).intervals()
        lisvQ = np.roll(lisv,np.where(lisv==6)[0][1]-np.where(lisv==6)[0][0])
        Qrow = [0]
        for n in lisvQ:
            Qrow.append((Qrow[-1]+n)%self.TET)
        Qrow.pop()
        return(PCSrow(Qrow))
    def M(self):
        return(PCSrow((self.pcs*5)%self.TET//1,TET=self.TET))
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
        •	Order the rhythmic sequence according to the most compact ascending form
        '''
        # 1. cycle to find the most compact ascending order
        nroll = np.linspace(0,len(self.rseq)-1,len(self.rseq),dtype=int)
        dist = np.zeros((len(self.rseq)),dtype=float)
        for i in range(len(self.rseq)):
            dist[i] = np.abs(np.roll(self.rseq,i)[len(self.rseq)-1] - np.roll(self.rseq,i)[0])
        # 2. check for multiple compact orders
        for l in range(1,len(self.rseq)):
            if np.array(np.where(dist == dist.min())).shape[1] != 1:
                indx = np.array(np.where(dist == dist.min()))[0]
                nroll = nroll[indx]
                dist = np.zeros((len(nroll)),dtype=int)
                i = 0
                for n in nroll:
                    dist[i] = np.abs(np.roll(self.rseq,n)[len(self.rseq)-(1+l)] - np.roll(self.rseq,n)[0])
                    i += 1
            else:
                indx = np.array(np.where(dist == dist.min()))[0]
                nroll = nroll[int(indx[0])]
                rseq_norm = np.roll(self.rseq,nroll)
                break
        if np.array(np.where(dist == dist.min())).shape[1] != 1: rseq_norm = self.rseq
        return(rseq_norm)


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
    
    def displayRhythm(self,note=None,xml=False,prime=False):
        '''
        •	Display rhythm sequence in score in musicxml format. If prime is True, display the prime form.
        '''
        if note == None: note=60
        m = m21.stream.Measure()
        if prime: 
            for l in range(self.rseq.shape[0]):
                n = m21.note.Note(note)
                n.duration = m21.duration.Duration(4*self.primeForm()[l])
                n.beams.fill('32nd', type='start')
                m.append(n)   
        else:
             for l in range(self.rseq.shape[0]):
                n = m21.note.Note(note)
                n.duration = m21.duration.Duration(4*self.rseq[l])
                n.beams.fill('32nd', type='start')
                m.append(n)  
        m.append(m21.meter.SenzaMisuraTimeSignature('0'))
        
        if xml: 
            m.show('musicxml')
        else:
            m.show()
        return

class musicntwrk:
    
    def __init__(self,TET=12):
        self.TET = TET
        
    def dictionary(self,space=None,N=None,Nc=None,order=None,row=None,a=None,prob=None,REF=None,scorefil=None,music21=None,show=None):
        '''
        define dictionary in the musical space specified in 'space': pcs, rhythm, rhythmP, score, orch
        '''
        if space == 'pcs':
            from .networks.pcsDictionary import pcsDictionary
            dictionary,ZrelT = pcsDictionary(Nc,row,a,order,prob,TET=self.TET)
            return(dictionary,ZrelT)
        
        if space == 'rhythm':
            from .networks.rhythmDictionary import rhythmDictionary
            dictionary,ZrelT = rhythmDictionary(Nc,a,REF)
            return(dictionary,ZrelT)

        if space == 'rhythmP':
            from .networks.rhythmPDictionary import rhythmPDictionary
            dictionary,ZrelT = rhythmPDictionary(N,Nc,REF)
            return(dictionary,ZrelT)
            
        if space == 'score':
            from .networks.scoreDictionary import scoreDictionary
            from .networks.readScore import readScore
            seq,chords = readScore(scorefil,music21,show,TET=self.TET)
            dictionary = scoreDictionary(seq,TET=self.TET)
            return(seq,chords,dictionary)
            
        if space == 'orch':
            from .networks.orchestralVector import orchestralVector
            score,orch,num = orchestralVector(scorefil)
            return(score,orch,num)

        
    def network(self,space=None,label=None,dictionary=None,thup=None,thdw=None,thup_e=None,thdw_e=None,distance=None,prob=None,write=None,\
                pcslabel=None,vector=None,ops=None,name=None,ntx=None,general=None,seq=None,sub=None,start=None,end=None,grphtype=None,\
                wavefil=None,cepstrum=None,color=None):
        '''
        define networks in the musical space specified in 'space': pcs (reg and ego), vLead (reg, vec, name and nameVec), 
        rhythm, rLead, score (reg, name and sub), timbre, orch
        '''
        if space == 'pcs':
            from .networks.pcsNetwork import pcsNetwork
            nodes, edges = pcsNetwork(dictionary,thup,thdw,distance,prob,write,pcslabel,TET=self.TET)
            return(nodes,edges)
        
        if space == 'pcsEgo':
            from .networks.pcsEgoNetwork import pcsEgoNetwork
            nodes_e, edges_e, edges_a = pcsEgoNetwork(label,dictionary,thup_e,thdw_e,thup,thdw,distance,write,TET=self.TET)
            return(nodes_e,edges_e,edges_a)
        
        if space == 'vLead' and vector != True and ops != True:
            from .networks.vLeadNetwork import vLeadNetwork
            nodes, edges = vLeadNetwork(dictionary,thup,thdw,distance,prob,write,pcslabel,TET=self.TET)
            return(nodes, edges)
        
        if space == 'vLead' and vector and ops != True:
            from .networks.vLeadNetworkVec import vLeadNetworkVec
            nodes, edges = vLeadNetworkVec(dictionary,thup,thdw,distance,prob,write,pcslabel,TET=self.TET)
            return(nodes, edges)
        
        if space == 'vLead' and vector != True and ops:
            from .networks.vLeadNetworkByName import vLeadNetworkByName
            nodes, edges = vLeadNetworkByName(dictionary,name,distance,prob,write,pcslabel,TET=self.TET)
            return(nodes, edges)
        
        if space == 'vLead' and vector and ops:
            from .networks.vLeadNetworkByNameVec import vLeadNetworkByNameVec
            nodes, edges = vLeadNetworkByNameVec(dictionary,name,distance,prob,write,pcslabel,TET=self.TET)
            return(nodes, edges)

        if space == 'rhythm':
            from .networks.rhythmNetwork import rhythmNetwork
            nodes, edges = rhythmNetwork(dictionary,thup,thdw,distance,prob,write)
            return(nodes, edges)
        
        if space == 'rLead':
            from .networks.rLeadNetwork import rLeadNetwork
            nodes, edges = rLeadNetwork(dictionary,thup,thdw,distance,prob,write)
            return(nodes, edges)
        
        if space == 'score' and sub != True:
            from .networks.scoreNetwork import scoreNetwork
            if ntx:
                nodes,edges,counts,deg,modul,Gxfull,Gxufull = scoreNetwork(seq,ntx,general,distance,TET=self.TET)
                return(nodes,edges,counts,deg,modul,Gxfull,Gxufull)
            else:
                nodes, edges, counts = scoreNetwork(seq,ntx,general,distance,TET=self.TET)
                return(nodes, edges, counts)
        
        if space == 'score' and sub:
            from .networks.scoreSubNetwork import scoreSubNetwork
            if ntx:
                nodes,edges,counts,deg,modul,Gxfull,Gxufull = scoreSubNetwork(seq,start,end,ntx,general,distance,grphtype,TET=self.TET)
                return(nodes,edges,counts,deg,modul,Gxfull,Gxufull)
            else:
                nodes, edges, counts = scoreSubNetwork(seq,start,end,ntx,general,distance,grphtype,TET=self.TET)
                return(nodes, edges, counts)

        if space == 'timbre':
            from .networks.timbralNetwork import timbralNetwork
            nodes, edges = timbralNetwork(wavefil,cepstrum,thup,thdw)
            return(nodes, edges)
        
        if space == 'orch':
            from .networks.orchestralNetwork import orchestralNetwork
            nodes,edges,deg,modul,part,_,_ = orchestralNetwork(seq,distance,TET=self.TET)
            return(nodes,edges,deg,modul,part)
        
            
    def timbre(self,descriptor=None,path=None,wavefil=None,standard=None,nmel=None,ncc=None,zero=None,lmax=None,maxi=None,nbins=None,\
                method=None,scnd=None,nstep=None):
        '''
        Define sound descriptor for timbral analysis
        '''
        if descriptor == 'PSCC' and standard != True:
            from .timbre.computePSCC import computePSCC
            waves, cepstrum0, cepstrum = computePSCC(path,wavefil,ncc,zero)
            return(waves, cepstrum0, cepstrum)
        
        if descriptor == 'PSCC' and standard:
            from .timbre.computeStandardizedPSCC import computeStandardizedPSCC
            waves, cepstrum0, lmax = computeStandardizedPSCC(path,wavefil,ncc,lmax,maxi,nbins)
            return(waves, cepstrum0, lmax)
            
        if descriptor == 'MFCC' and standard != True:
            from .timbre.computeMFCC import computeMFCC
            waves, cepstrum0, cepstrum = computeMFCC(path,wavefil,nmel,ncc,zero)
            return(waves, cepstrum0, cepstrum)

        if descriptor == 'MFCC' and standard:
            from .timbre.computeStandardizedMFCC import computeStandardizedMFCC
            waves, cepstrum0, lmax = computeStandardizedMFCC(path,wavefil,nmel,ncc,lmax,maxi,nbins)
            return(waves, cepstrum0, lmax)
            
        if descriptor == 'MFPS' and standard:
            from .timbre.computeStandardizedMFPS import computeStandardizedMFPS
            waves, cepstrum0, lmax = computeStandardizedMFPS(path,wavefil,nmel,lmax,maxi,nbins)
            return(waves, cepstrum0, lmax)
            
        if descriptor == 'ASCBW' and standard != True:
            from .timbre.computeASCBW import computeASCBW
            waves, ascbw = computeASCBW(path,wavefil)
            return(waves, ascbw)

        if descriptor == 'ASCBW' and standard:
            from .timbre.computeModifiedASCBW import computeModifiedASCBW
            waves, ascbw, ascbwu = computeModifiedASCBW(path,wavefil,scnd,method,nstep)
            return(waves, ascbw, ascbwu)

    def harmony(self,descriptor=None,mode=None,x=None,y=None):
        '''
        handler for calculating tonal harmony models, tonnentz and to launch the tonal harmony calculator
        '''
        if descriptor == 'calc':
            from .harmony.tonalHarmonyCalculator import tonalHarmonyCalculator
            tonalHarmonyCalculator()
            
        if descriptor == 'tonnentz':
            from .harmony.tonnentz import tonnentz
            tnz = tonnentz(x,y)
            return(tnz)
            
        if descriptor == 'model':
            from .harmony.tonalHarmonyModels import tonalHarmonyModels
            tonalHarmonyModels(mode)
        
    def sonify(self,descriptor=None,data=None,length=None,midi=None,scalemap=None,ini=None,fin=None,fac=None,dur=None,transp=None,
               col=None,write=None,vnorm=None,plot=None,crm=None,tms=None,xml=None):
        '''
        sonification strategies - simple sound (spectral) or score (melodic progression)
        '''
        if descriptor == 'spectrum':
            from .data.r_1Ddata import r_1Ddata
            from .data.i_spectral_pyo import i_spectral_pyo
            x, y = r_1Ddata(data)
            s,a = i_spectral_pyo(x,y[0])
            s.start()
            time.sleep(length)
            s.stop()
            s.shutdown()
            
        if descriptor == 'melody':
            from .data.r_1Ddata import r_1Ddata
            from .data.scaleMapping import scaleMapping
            from .data.MIDImap import MIDImap
            from .data.MIDIscore import MIDIscore
            from .data.MIDImidi import MIDImidi
            x, y = r_1Ddata(data)
            scale, nnote = scaleMapping(scalemap,ini=ini,fin=fin,fac=fac)
            MIDIscore(MIDImap(y[col],scale,nnote)-transp,dur=dur,w=write)
            if midi: MIDImidi(MIDImap(y[col],scale,nnote)-transp,dur=2*dur,vnorm=vnorm)
        
        if descriptor == 'sound':
            from .data.analyzeSound import analyzeSound
            from .data.WRITEscore import WRITEscore
            nseq,beat,prob = analyzeSound(data,outlist=['nseq','beat','prob'],plot=plot,crm=crm,tms=tms,xml=xml)
            WRITEscore(data,nseq.pcs,beat.rseq,w=write)