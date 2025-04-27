#
# msctools: my collection of composing and performing tools in python
#
# © 2023 Marco Buongiorno Nardelli
#

import glob,time,ast
import numpy as np
import pyo

import threading
import numpy as np
import networkx as nx

from .networks import *
from .decorators import threading_decorator

import musicntwrk.msctools.cfg as cfg

@threading_decorator
def simplePlayerP(clips=None,dur=None,track=0,delay=0.0,offset=1.0,panning=None,gain=1.0,impulse=None,bal=0.25,
            mode='network',external=None,nxmodel='barabasi_albert_graph',*args):
    ''' 
    Play clips in sequence waiting for next clip in following mode
    mode = "network"    : sequence defined by the eulerian path on a network
                        : network models can be found here: 
                        : https://networkx.org/documentation/stable/reference/generators.html
                        : arguments are passed through *args
    mode = "sequential" : plays the clips in descending order
    mode = "random"     : plays clips in random order
    mode = "external"   : plays clip with a user supplied sequence
    '''

    def sleep(sec):
        # internal scope function to pause execution while controlling the termination of the thread
        ntx = int(sec/cfg.TICK)
        for n in range(ntx):
            time.sleep(cfg.TICK)
            if cfg.stop[track]:
                if panout.isPlaying(): 
                    panout.setMul(pyo.SigTo(value=0.0, time=3.0, init=gain))
                    panout.stop(wait=3.0)
                    rev.stop(wait=3.0)
                    snd.stop(wait=3.0)
                    try:
                        pan.stop(wait=3.0)
                    except:
                        pass
                    time.sleep(3.0)
                break
        snd.stop()
        stop = True
        return(stop)
        
    if clips == None:
        return('no clips provided')
    time.sleep(offset)
    while True:
        time.sleep(cfg.TICK)
        if cfg.stop[track]:
            break
        if mode == 'network':
            mynetx = getattr(nx,nxmodel)
            Gx = mynetx(*args)
            chino = chinese_postman(Gx,None,verbose=False)
            seq = [chino[0][0]]
            for s in range(1,len(chino)):
                seq.append(chino[s][1])
                if cfg.stop[track]:
                    break
        elif mode == 'sequential':
            seq = np.linspace(0,len(clips)-1,len(clips),dtype=int).tolist()
        elif mode == 'random':
            seq = np.linspace(0,len(clips)-1,len(clips),dtype=int).tolist()
            np.random.shuffle(seq)
        elif mode == 'external':
            seq = external
        else:
            print('mode not implemented')
        for n in range(len(seq)):
            # set panning
            if panning == 'random':
                pan = np.random.rand()
            elif isinstance(panning,float) or isinstance(panning,int):
                pan = panning
            elif panning == 'LR':
                pan = pyo.SigTo(value=1.0, time=dur[n], init=0.0)
            elif panning == 'RL':
                pan = pyo.SigTo(value=0.0, time=dur[n], init=1.0)
            else:
                print('panning not defined')
            snd = clips[seq[n]].play()
            if impulse != None:
                rev = pyo.CvlVerb(snd,impulse,bal=bal,mul=2*gain)
            else:
                rev = snd
            panout = pyo.SPan(rev,outs=2,pan=pan,mul=gain).out()
            stop = sleep(dur[seq[n]]+delay*np.random.rand())
            panout.stop()
            rev.stop()
            snd.stop()
            try:
                pan.stop()
            except:
                pass
            if stop: break

@threading_decorator
def playerP(clips=None,track=0,delay=0.0,offset=1.0,panning=None,gain=1.0,impulse=None,bal=0.25,
            mode='network',external=None,nxmodel='barabasi_albert_graph',*args):
    ''' 
    Play clips in sequence waiting for next clip in following mode
    mode = "network"    : sequence defined by the eulerian path on a network
                        : network models can be found here: 
                        : https://networkx.org/documentation/stable/reference/generators.html
                        : arguments are passed through *args
    mode = "sequential" : plays the clips in descending order
    mode = "random"     : plays clips in random order
    mode = "external"   : plays clip with a user supplied sequence
    '''
    def sleep(sec):
        # internal scope function to pause execution while controlling the termination of the thread
        ntx = int(sec/cfg.TICK)
        for n in range(ntx):
            time.sleep(cfg.TICK)
            if cfg.stop[track]:
                if panout.isPlaying(): 
                    panout.setMul(pyo.SigTo(value=0.0, time=3.0, init=gain))
                    panout.stop(wait=3.0)
                    rev.stop(wait=3.0)
                    snd.stop(wait=3.0)
                    try:
                        pan.stop(wait=3.0)
                    except:
                        pass
                    #time.sleep(3.0)
                break
        snd.stop()

    if clips == None:
        return('no clips provided')
    time.sleep(offset)
    while True:
        if cfg.stop[track]:
            break
        if mode == 'network':
            mynetx = getattr(nx,nxmodel)
            Gx = mynetx(*args)
            chino = chinese_postman(Gx,None,verbose=False)
            seq = [chino[0][0]]
            for s in range(1,len(chino)):
                seq.append(chino[s][1])
                if cfg.stop[track]:
                    break
        elif mode == 'sequential':
            seq = np.linspace(0,len(clips)-1,len(clips),dtype=int).tolist()
        elif mode == 'random':
            seq = np.linspace(0,len(clips)-1,len(clips),dtype=int).tolist()
            np.random.shuffle(seq)
        elif mode == 'external':
            seq = external
        else:
            print('mode not implemented')
        for n in range(len(seq)):
            # set panning
            if panning == 'random':
                pan = np.random.rand()
            elif isinstance(panning,float) or isinstance(panning,int):
                pan = panning
            elif panning == 'LR':
                pan = pyo.SigTo(value=1.0, time=pyo.sndinfo(clips[n])[1], init=0.0)
            elif panning == 'RL':
                pan = pyo.SigTo(value=0.0, time=pyo.sndinfo(clips[n])[1], init=1.0)
            else:
                print('panning not defined')
            snd = pyo.SfPlayer(clips[seq[n]])
            if impulse != None:
                rev = pyo.CvlVerb(snd,impulse,bal=bal,mul=2*gain)
            else:
                rev = snd
            panout = pyo.SPan(rev,outs=2,pan=pan,mul=gain).out()
            # time.sleep(pyo.sndinfo(clips[seq[n]])[1]+delay*np.random.rand())
            sleep(pyo.sndinfo(clips[seq[n]])[1]+delay*np.random.rand())
            panout.stop()
            rev.stop()
            snd.stop()
            try:
                pan.stop()
            except:
                pass
            if cfg.stop[track]:
                panout.setMul(pyo.SigTo(value=0.0, time=3.0, init=1.0))
                panout.stop(wait=3.0)
                rev.stop(wait=3.0)
                snd.stop(wait=3.0)
                try:
                    pan.stop(wait=3.0)
                except:
                    pass
                break



@threading_decorator
def scorePlayerP(clips,track,score,offset=0,panning=0.5,impulse=None,bal=0.25,gain=1.0,scaledur=1.0,fin=0.1,fout=0.2):
    ''' 
    Play clips in sequence according to a score (pitch + duration)
    score[0] = pitches
    score[1] = durations in seconds (can be scaled with the scaledur parameter)
    '''
    # delay the start of playback - if for any reason is desired

    def sleep(sec):
        # internal scope function to pause execution while controlling the termination of the thread
        ntx = int(sec/cfg.TICK)
        for n in range(ntx):
            time.sleep(cfg.TICK)
            if cfg.stop[track]:
                if panout.isPlaying():
                    panout.setMul(pyo.SigTo(value=0.0, time=3.0, init=gain))
                    panout.stop(wait=3.0)
                    rev.stop(wait=3.0)
                    snd.stop(wait=3.0)
                    try:
                        pan.stop(wait=3.0)
                    except:
                        pass
                    #time.sleep(3.0)
                break
        snd.stop()

    time.sleep(offset)
    while True:
        if cfg.stop[track]:
            break
        seq = score[0]
        dur = score[1]
        for n in range(len(seq)):
            # set panning
            if panning == 'random':
                pan = np.random.rand()
            elif isinstance(panning,float) or isinstance(panning,int):
                pan = panning
            else:
                print('panning not defined')
            fade = pyo.Fader(fadein=fin, fadeout=fout, dur=dur[n]*scaledur).play()
            snd = pyo.SfPlayer(clips[seq[n]],mul=gain)
            if impulse != None:
                rev = pyo.CvlVerb(snd,impulse,bal=bal,mul=2*gain)
            else:
                rev = snd
            panout = pyo.SPan(rev,outs=2,pan=pan,mul=fade).out()
            sleep(dur[n]*scaledur)
            snd.stop()
            rev.stop()
            panout.stop()
            if cfg.stop[track]:
                panout.setMul(pyo.SigTo(value=0.0, time=3.0, init=gain))
                panout.stop(wait=3.0)
                snd.stop(wait=3.0)
                rev.stop(wait=3.0)
                break


@threading_decorator
def playerList(clips=None,track=0,delay=0.0,offset=1.0,panning=None,gain=1.0,impulse=None,bal=0.25,
            external=None,*args):
    ''' 
    Play clips in sequence waiting for next clip - Version for clips in two separate folders
    '''
    def sleep(sec):
        # internal scope function to pause execution while controlling the termination of the thread
        ntx = int(sec/cfg.TICK)
        for n in range(ntx):
            time.sleep(cfg.TICK)
            if cfg.stop[track]:
                if panout.isPlaying(): 
                    panout.setMul(pyo.SigTo(value=0.0, time=3.0, init=gain))
                    panout.stop(wait=3.0)
                    rev.stop(wait=3.0)
                    snd.stop(wait=3.0)
                    time.sleep(3.0)
                break
        snd.stop()

    if clips == None:
        return('no clips provided')
    else:
        assert(len(clips) == 2)
    time.sleep(offset)
    seq = external
    while True:
        if cfg.stop[track]:
            break
        for n in range(len(seq)):
            # set panning
            if panning == 'random':
                pan = np.random.rand()
            elif isinstance(panning,float) or isinstance(panning,int):
                pan = panning
            elif panning == 'LR':
                pan = pyo.SigTo(value=1.0, time=pyo.sndinfo(clips[n])[1], init=0.0)
            elif panning == 'RL':
                pan = pyo.SigTo(value=0.0, time=pyo.sndinfo(clips[n])[1], init=1.0)
            else:
                print('panning not defined')

            if seq[n] == ' ':
                # breaths
                idx = np.random.randint(len(clips[1]))
                snd = pyo.SfPlayer(clips[1][idx])
                cliptime = 2*pyo.sndinfo(clips[1][idx])[1]
                if impulse != None:
                    rev = pyo.CvlVerb(snd,impulse[1],bal=bal,mul=2*gain)
                else:
                    rev = snd
            else:
                # stones
                snd = pyo.SfPlayer(clips[0][seq[n]])
                cliptime = pyo.sndinfo(clips[0][seq[n]])[1]
                if impulse != None:
                    rev = pyo.CvlVerb(snd,impulse[0],bal=bal,mul=2*gain)
                else:
                    rev = snd
            
            panout = pyo.SPan(rev,outs=2,pan=pan,mul=gain).out()
            sleep(cliptime+delay*np.random.rand())
            panout.stop()
            rev.stop()
            snd.stop()
            if cfg.stop[track]:
                panout.setMul(pyo.SigTo(value=0.0, time=3.0, init=1.0))
                panout.stop(wait=3.0)
                rev.stop(wait=3.0)
                snd.stop(wait=3.0)
                break

def pause(sec,last,*args):
    # function to pause execution while controlling the termination of the whole performance
    ntx = int(sec/cfg.TICK)
    for n in range(ntx):
        time.sleep(cfg.TICK)
        # f = open('./MASTER_STOP.txt','r')
        # MASTER_STOP = ast.literal_eval(f.readline())
        # f.close()
        exit = False
        if cfg.MASTER_STOP == True:
            # print('stopping...')
            for i in range(len(cfg.stop)):
                cfg.stop[i] = True
            time.sleep(1.0)
            for i in range(len(args)):
                try:
                    if args[i].isPlaying(): 
                        mulx = [n for n,s in enumerate(args[i].dump().split()) if args[i].dump().split()[n-1] == 'mul:'][0]
                        mul = float(args[i].dump().split()[mulx])
                        args[i].setMul(pyo.SigTo(value=0.0, time=3.0, init=mul))
                        args[i].stop(wait=3.0)
                except:
                    print('Exception in pause - hard stop')
                    args[i].stop()

                pass
            exit = True
            cfg.MASTER_STOP = False 
            return(exit)
            break
    if last:
        # print('stopping...')
        for i in range(len(cfg.stop)):
            cfg.stop[i] = True
        time.sleep(1.0)
        try:
            for i in range(len(args)):
                if args[i].isPlaying(): 
                    mulx = [n for n,s in enumerate(args[i].dump().split()) if args[i].dump().split()[n-1] == 'mul:'][0]
                    mul = float(args[i].dump().split()[mulx])
                    args[i].setMul(pyo.SigTo(value=0.0, time=3.0, init=mul))
                    args[i].stop(wait=3.0)
        except:
            pass
        exit = True
        return(exit)
    

def importSoundfiles(dirpath='./',filepath='./',mult=0.1,gain=1.0,sorted=False):
    # reading wavefiles
    try:
        obj = [None]*len(glob.glob(dirpath+filepath))
        fil = [None]*len(glob.glob(dirpath+filepath))
        if sorted:
            for i,file in enumerate(sorted(glob.glob(dirpath+filepath))):
                fil[i] = pyo.sndinfo(file)[1]
                obj[i] = pyo.SfPlayer(file,mul=mult*gain).stop()
        else:
            for i,file in enumerate(glob.glob(dirpath+filepath)):
                fil[i] = pyo.sndinfo(file)[1]
                obj[i] = pyo.SfPlayer(file,mul=mult*gain).stop()
    except:
        print('error in file reading',dirpath)
        pass
    
    return(obj,fil)

def importSoundfilesT(dirpath='./',filepath='./',mult=0.1,gain=1.0):
    # reading wavefiles
    try:
        obj = [None]*len(glob.glob(dirpath+filepath))
        fil = [None]*len(glob.glob(dirpath+filepath))
        for i,file in enumerate(sorted(glob.glob(dirpath+filepath))):
            fil[i] = pyo.sndinfo(file)[1]
            obj[i] = pyo.SfPlayer(file,mul=mult*gain).stop()
    except:
        print('error in file reading')
        pass
    
    return(obj,fil)

def importSoundfiles2(dirpath='./',filepath='./',mult=0.1,gain=1.0):
    # reading wavefiles
    try:
        obj0 = [None]*len(glob.glob(dirpath+filepath))
        obj1 = [None]*len(glob.glob(dirpath+filepath))
        fil = [None]*len(glob.glob(dirpath+filepath))
        for file in glob.glob(dirpath+filepath):
            i = int(file.split('.')[3])
            fil[i] = file
            obj0[i] = pyo.SfPlayer(file,mul=mult*gain).stop()
            obj1[i] = pyo.SfPlayer(file,mul=mult*gain).stop()
    except:
        print('error in file reading')
        pass
    
    return(obj0,obj1,fil)

def importClips(dirpath='./',filepath='./',sort=True):
    # reading wavefiles
    try:
        clips = [None]*len(glob.glob(dirpath+filepath))
        if sort:
            for i,file in enumerate(sorted(glob.glob(dirpath+filepath))):
                clips[i] = file
            return(clips)
        else:
            for i,file in enumerate(glob.glob(dirpath+filepath)):
                clips[i] = file
            return(clips)
    except:
        print('error in file reading')
        return

### function to distribute sounds in stereo pairs

def panMove(snd0,snd1,fil,nch,mult):
    if snd0.isPlaying() == True:
        pass
    else:
        snd0.play()
        snd1.play()
        ff = float(1/pyo.sndinfo(fil)[1]/4)
        sin = pyo.Sine(freq=ff,phase=0)
        cos = pyo.Sine(freq=ff,phase=0.25)
        ini = np.random.randint(0,nch)
        step = np.random.randint(0,int(nch/2)+1)
        end = (ini+step)%nch
        snd0.out(ini,0).setMul(mult*cos**2)
        snd1.out(end,0).setMul(mult*sin**2)
        snd0.stop(wait=pyo.sndinfo(fil)[1])
        snd1.stop(wait=pyo.sndinfo(fil)[1])
        
### function to distribute sounds in a multichannel environment

def playChannel(snd,fil,nch,mult):
    # play single wav file on channel nch
    if snd.isPlaying() == True:
        pass
    else:
        snd.play()
        snd.out(nch,0).setMul(mult)
        snd.stop(wait=pyo.sndinfo(fil)[1])

def multiEnv(ch,T):
    n = ch-1
    if n == 0:
        env = [np.ones(T)]
    else:
        t = np.linspace(0,T-1,T)
        env = [None]*ch
        for i in range(0,n+1):
            env[i] = np.sin(np.pi/2/T*n*t-(i-1)*np.pi/2)**2
            env[i][t<=(i-1)*T/n] = 0
            env[i][t>=(i+1)*T/n] = 0
    return(env)

def multiTable(sndfile,chpath):
    snd = pyo.SndTable(sndfile)
    T = snd.getSize()[0]
    freq = snd.getRate()
    ch = len(chpath)
    env = multiEnv(ch,snd.getSize()[0])
    wav = snd.getEnvelope(T)
    dur = snd.getDur()
    wave = [None]*ch
    table = [None]*ch
    for i in range(ch):
        tmp = np.array(wav)*env[i]
        wave[i] = tmp.tolist()
    for i,cn in enumerate(chpath):
        table[i] = pyo.DataTable(size=T, init=wave[i])
    return(table,freq,dur)

def chPath(chpath,table,freq,dur,mult):
    a = [None]*len(chpath)
    for i,cn in enumerate(chpath):
        a[i] = pyo.Osc(table=table[i], freq=[freq,freq], mul=mult).out(cn-1,0)
        a[i].stop(wait=dur)
        
class signalGran():
    
    # class written by Connor Scroggins, UNT 2023.
    
    def __init__(self, signal, mul = 1.0, granSize = 0.1, granDens = 30, window = 7):
        self.signal = signal
        self.isPlaying = 0
        self.stopSignalGran = False
        self.mul = mul
        self.granSize = granSize
        self.granDens = granDens
        self.window = window
    
    # granSignal
    #
    # Granulate a given live signal.
    #
    # input:
    # -signal: the live audio signal to granulate
    # -chan: the channel to output the granulator from
    # -granSize: the duration of each grain in seconds
    # -granDens: the density of grains as a percentage
    # -window: window type corresponding with WinTable types
    #
    # output:
    # -1 channel of live granulated audio output
    
    def playSample (self):
    
        # Take the reciprocal of the granSize since the TableRead below will use this value,
        # but TableRead uses the value for playback rate in its "freq" parameter.
        granSizeRecip = 1 / self.granSize
    
        # Generate the grain window including its duration.
        granEnv = pyo.TableRead(table = pyo.WinTable(self.window), freq = granSizeRecip)
    
        # Play the live signal to granulate. This does not directly sound any audio since the
        # grain envelopes attenuate the live signal.
        self.signal.play()
        self.signal.setMul(granEnv)
    
        while True:
            if self.stopSignalGran == True:
                break
    
                # For each possible grain, determine if a grain should play based on density.
                # Each "grain" is not a sound but an envelope of the signal played above.
            granProb = np.random.uniform(0.0, 100.0) # Generate a random value as a percentage
            if granProb <= self.granDens: # Play an if the random value is "within" the density value.
                granEnv.play()
    
            time.sleep(self.granSize) # Do not start the next grain until the current grain envelope ends.
    
    def out(self,n=0,m=1):
    
        # Check if granulator is playing.
        if self.isPlaying == 0:
            self.play()
    
        self.signal.out(n,n+m)
    
    def stop(self):
        self.signal.stop()
        self.isPlaying = 0
        self.stopSignalGran = 1
        
    def play(self):
        threading.Thread(target=self.playSample,args=()).start()
        self.isPlaying = 1
        
        
    def setSignal(self, newSignal):
        self.signal = newSignal
    
    # def setChnl(self, newChnl):
    #   self.chnl = newChnl
        
    def setMul (self, newMul):
        self.mul = newMul
    
    def setGranSize (self, newGranSize):
        self.granSize = newGranSize
        
    def setGranDens (self, newGranDens):
        self.granDens = newGranDens
    
    def setWindow (self, newWindow):
        self.window = newWindow
