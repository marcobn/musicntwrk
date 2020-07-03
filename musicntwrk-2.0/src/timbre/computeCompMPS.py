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
def computeCompMPS(input_path,input_file,n_mels=13,barplot=True):
    # read audio files in repository and compute the MPS
    waves = list(glob.glob(os.path.join(input_path,input_file)))
    mps0 = []
    for wav in np.sort(waves):
        y, sr = librosa.load(wav)
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
        # Here we decompose the MPS in a one-dim component and an activation matrix
        comps, acts = librosa.decompose.decompose(S, n_components=1,sort=True)
        comps = np.reshape(comps,comps.shape[0])
        mps0.append(comps)
    mps0 = np.array(mps0)
    if barplot:
        # print the mps0 matrix for all sounds
        axprops = dict(xticks=[], yticks=[])
        barprops = dict(aspect='auto', cmap=plt.cm.coolwarm, interpolation='nearest')
        fig = plt.figure()
        ax1 = fig.add_axes([0.1, 0.1, 3.1, 0.7], **axprops)
        cax = ax1.matshow(np.flip(mps0.T,axis=0), **barprops)
        fig.colorbar(cax)
        plt.show()
    return(np.sort(waves),mps0)
    
