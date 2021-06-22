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
import matplotlib.pyplot as plt

def orchestralVectorColor(orch,dnodes,part,color='coolwarm'):
    '''
    Produces the sequence of the orchestration vectors color-coded according to the modularity class they belong
    Requires the output of orchestralNetwork()
    '''
    cmap = plt.get_cmap(color)
    pdict = pd.DataFrame(None,columns=['vec','part'])
    for n in range(len(part)):
        tmp = pd.DataFrame( [[dnodes.iloc[int(list(part.keys())[n])][0], list(part.values())[n]]], columns=['vec','part'] )
        pdict = pdict.append(tmp)
    dict_vec = pdict.set_index("vec", drop = True)
    orch_color = np.zeros(orch.shape)
    for i in range(orch.shape[0]):
        orch_color[i,:] = orch[i,:] * \
            (dict_vec.loc[np.array2string(orch[i][:]).replace(" ","").replace("[","").replace("]","")][0]+1)

    axprops = dict(xticks=[], yticks=[])
    barprops = dict(aspect='auto', cmap=cmap, interpolation='nearest')
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.1, 3.1, 0.7], **axprops)
    ax1.matshow(orch_color[:].T, **barprops)
    plt.show()

