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
import librosa
import librosa.display

import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

from ..musicntwrk import PCSet
from ..musicntwrk import RHYTHMSeq

def analyzeSound(soundfile,outlist,plot=True,crm=True,tms=True,xml=False):
    var = {}        
    # load soundfile
    y, sr = librosa.load(soundfile)
    var['y'] = y
    var['sr'] = sr
    # analyze onsets
    o_env = librosa.onset.onset_strength(y, sr=sr)
    times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    var['onset_frames'] = onset_frames
    var['times'] = times
    if plot:
        plt.figure(figsize=(18,8))
        ax1 = plt.subplot(2, 1, 1)
        librosa.display.waveplot(y[:])
        plt.title('Waveshape')
        plt.subplot(2, 1, 2, sharex=ax1)
        plt.plot(times, o_env, label='Onset strength')
        plt.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,linestyle='--', label='Onsets')
        plt.axis('tight')
        plt.legend(frameon=True, framealpha=0.75)
    p = None
    if crm:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        var['chroma'] = chroma
        nseq = []
        for i in range(onset_frames.shape[0]-1):
            nseq.append(np.argwhere(chroma[:,onset_frames[i]] == 1.0)[0,0])
        var['nseq'] = PCSet(nseq,UNI=False,ORD=False)
        if plot:
            plt.figure(figsize=(18, 4))
            librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
            plt.colorbar()
            plt.title('Chromagram')
            plt.tight_layout()
        idx = np.argwhere(chroma == 1.0)
        p = np.histogram(idx[:,0],12)
        var['prob'] = np.asarray(p[0]/np.sum(p[0]))
        if plot:
            c = np.array(['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B'])
            plt.figure(figsize=(6, 4))
            plt.bar(c,p[0],width=0.8)
    if plot: plt.show()
    tempo = None
    if tms:
        tempo = librosa.beat.tempo(onset_envelope=o_env, sr=sr)
        beat = librosa.frames_to_time(onset_frames, sr=sr)
        beat = RHYTHMSeq((np.diff(beat)*16).round(0)/16,REF='e')
        var['beat'] = beat
        var['tempo'] = int(tempo[0])
        if plot: beat.displayRhythm(xml)
    output = []
    for out in outlist:
        output.append(var[out])
    return(output)

