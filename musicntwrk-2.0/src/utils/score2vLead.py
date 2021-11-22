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
import pandas as pd

def score2vLead(nodes,dscore):
    # create nodes dictionary for vLeadNetwork from a score dictionary
    reference = []
    for node in np.array(nodes['Label']):
    #     print(node,np.array(dscore.loc[dscore['class']==node]['pcs'])[0],
    #           np.array(dscore.loc[dscore['class']==node]['interval'])[0])
        tmp = [node,np.array(dscore.loc[dscore['class']==node]['pcs'])[0],
              np.array(dscore.loc[dscore['class']==node]['interval'])[0]]
        reference.append(tmp)
    return(pd.DataFrame(reference,columns=['class','pcs','interval']))