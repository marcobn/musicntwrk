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
def findLengthMax(input_path,input_file):
    # read audio files in repository and compute the number of samples
    waves = list(glob.glob(os.path.join(input_path,input_file)))
    wf = []
    for wav in np.sort(waves):
        y, sr = librosa.load(wav)
        wf.append(y)
    wf = np.asarray(wf)
    # find length of sound for standardization of the number of samples in every file wav
    lwf = []
    for n in range(wf.shape[0]):
        lwf.append(wf[n].shape[0])
    lwf = np.asarray(lwf)
    lmax = np.max(lwf)
    return(np.sort(waves),lwf,lmax)
    
