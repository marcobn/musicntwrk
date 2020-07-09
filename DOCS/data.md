`data` contains functions for the sonification of data in multi-column or csv
format and produces output as WAV (it requires an installation of csound and
direct reference to the ctcsound module - ), or musicxml or MIDI. Two sonification
protocols are available: spectral - data are mapped to a single sound using
subtractive synthesis (FIR filter); and linear - individual data points are mapped
to pitches in a time-series structure. sonifiPy contains:

- `def r1Ddata(path,fileread)` -
	 Read data file in a multicolumn format (csv files can be easily put in this
	 format using Pandas). Returns the data values as (x,y).
	- path (str) path to data file
	- fileread (str) data file
- `def ispectral2(xv,yv,itime,path=’./’,instr=’noise’)` -
	 Use subtractive synthesis to sonify data structure. Returns the sound file.
	- xv,yv (float) data structure to sonify
	- path (str) path to data file
	- fileread (str) data file
- `def itimeseries(xv,yv,path=’./’,instr=’csb701’)` -
	 Use csound instruments to sonify data structures as time-series. Returns the
	 sound file.
	- xv,yv (float) data structure to sonify
	- path (str) path to data file
	- fileread (str) data file
	- instr (str) csound instrument (it can be modified by user)
- `def MIDImap(pdt,scale,nnote)` -
	 Data to MIDI conversion on a given scale defined in scaleMapping (see be-
	 low). Returns the MIDI data structure.
	- pdt (float) data structure mapped to MIDI numbers
	- scale (float) scale mapping (from scaleMapping)
	- nnote (int) number of notes in the scale (from scaleMapping)
- `def scaleMapping(scale)` -
	 Scale definitions for MIDI mapping. Returns: scale, nnote (see above).
- `def MIDIscore(yvf,dur=2,w=None,outxml=’./music’,outmidi=’./music’)` -
	 Display score or writes to file
	- yvf (float)data structure mapped to MIDI numbers (from MIDImap)
	- dur (int) reference duration
	- w (logical) if True writes either musicxml or MIDI file)
- `def MIDImidi(yvf,vnorm=80,dur=4,outmidi=’./music’)` -
	 Display score or writes to file
	- yvf (float)data structure mapped to MIDI numbers (from MIDImap)
	- vnorm (int) reference velocity
	- outmidi (str) MIDI file
