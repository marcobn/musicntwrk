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

import pandas as pd

def networkHarmonyGen(mk,descriptor=None,dictionary=None,thup=None,thdw=None,names=None,
                      distance=None,probs=None,write=None,pcslabel=None):
    '''
    generator of harmonic spaces from probabilistic models of complex networks
    - it requires a dictionary of pcs
    
    descriptor = 'pcs' or 'vLead'
    '''
    # this is somewhat redundant since it can be achieved with a single call anyways
    if thup != None and thdw != None:
        
        if descriptor == 'pcs':
            nodes, edges = mk.network(space='pcs',dictionary=dictionary,thup=thup,thdw=thup,
                                      distance=distance,prob=probs[0],write=write,pcslabel=pcslabel) 
    
        elif descriptor == 'vLead':
            nodes, edges = mk.network(space='vLead',dictionary=dictionary,thup=thup,thdw=thdw,
                                      distance=distance,prob=probs[0],write=write,pcslabel=pcslabel)
        
        else:
            print('missing descriptor')
                                  
    # this oine is for building networks from composites (what is really this function for)

    elif thup == None and thdw == None and names != None:
        edges = pd.DataFrame(None,columns=['Source','Target','Weight','Label'])
        # generation of probabilistic complex network with operatorial design
        if len(names) != len(probs):
            print('names not matching probabilities!')
        else:
            for n in range(len(names)):
                nodes,dedges = mk.network(space='vLead',ops=True,name=names[n],
                                           pcslabel=True,dictionary=dictionary,distance=distance,
                                           prob=probs[n],write=False)
                edges = edges.append(dedges)

    return(nodes,edges)