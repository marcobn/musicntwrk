## `data`

<span>`data`</span> contains functions for the sonification of data in
multi-column or csv format and produces output as WAV either via csound
(it requires an installation of csound and direct reference to the
<span>ctcsound</span> module), <span>pyo</span> or pure python, and
<span>musicxml</span> or MIDI. Two sonification protocols are available:
spectral - data are mapped to a single sound using subtractive synthesis
(FIR filter); and linear - individual data points are mapped to pitches
in a time-series structure. See Ref.   for a complete description of
this protocol. <span>`data`</span> contains:

<span>__def r\_1Ddata__(path,fileread)</span>  
Read data file in a multicolumn format (csv files can be easily put in
this format using Pandas). Returns the data values as (x,y).

<span><span>path (str)</span> </span>
path to data file

<span><span>fileread (str)</span> </span>
data file

<span>__def i\_spectral2__(xv,yv,itime,path=’./’,instr=’noise’)</span>  
Use subtractive synthesis to sonify data structure. Returns the sound
file.

<span><span>xv,yv (float)</span> </span>
data structure to sonify

<span><span>path (str)</span> </span>
path to data file

<span><span>fileread (str)</span> </span>
data file

<span>__def i\_time\_series__(xv,yv,path=’./’,instr=’csb701’)</span>  
Use csound instruments to sonify data structures as time-series. Returns
the sound file.

<span><span>xv,yv (float)</span> </span>
data structure to sonify

<span><span>path (str)</span> </span>
path to data file

<span><span>fileread (str)</span> </span>
data file

<span><span>instr (str)</span> </span>
csound instrument (it can be modified by user)

<span>__def MIDImap__(pdt,scale,nnote)</span>  
Data to MIDI conversion on a given scale defined in scaleMapping (see
below). Returns the MIDI data structure.

<span><span>pdt (float)</span> </span>
data structure mapped to MIDI numbers

<span><span>scale (float)</span> </span>
scale mapping (from scaleMapping)

<span><span>nnote (int)</span> </span>
number of notes in the scale (from scaleMapping)

<span>__def scaleMapping__(scale)</span>  
Scale definitions for MIDI mapping. Returns: scale, nnote (see above).

<span>__def MIDIscore__(yvf,dur=2,w=None,outxml=’./music’,outmidi=’./music’)</span>  
Display score or writes to file

<span><span>yvf (float)</span> </span>
data structure mapped to MIDI numbers (from MIDImap)

<span><span>dur (int)</span> </span>
reference duration

<span><span>w (logical)</span> </span>
if True writes either musicxml or MIDI file)

<span>__def MIDImidi__(yvf,vnorm=80,dur=4,outmidi=’./music’)</span>  
Display score or writes to file

<span><span>yvf (float)</span> </span>
data structure mapped to MIDI numbers (from MIDImap)

<span><span>vnorm (int)</span> </span>
reference velocity

<span><span>outmidi (str)</span> </span>
MIDI file
