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
