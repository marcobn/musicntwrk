# `MUSICNTWRK`

The <span>`MUSICNTWRK`</span> package (www.musicntwrk.com), is a python
library written by the author and available as a PyPi project at
www.pypi.org/project/musicntwrk/ or on GitHub:
https://github.com/marcobn/musicntwrk. <span>`musicntwrk`</span> is the
main module of the project and contains helper classes for pitch class
set classification and manipulation in any arbitrary temperament
(<span>PCSet</span>, <span>PCSetR</span> and <span>PCSrow</span>),
<span>RHYTHMSeq</span> for the manipulation of rhythmic sequences, and
the main class <span>`musicntwrk`</span> that allows the construction of
generalized musical space networks using distances between common
descriptors (interval vectors, voice leadings, rhythm distance, etc.);
the analysis of scores, the sonification of data and the generation of
compositional frameworks. <span>`musicntwrk`</span> acts as a wrapper
for the various functions organized in the following sub-projects:

1.  <span>`networks`</span> - contains all the modules to construct
    dictionaries and networks of pitch class set spaces including voice
    leading, rhythmic spaces, timbral spaces and score network and
    orchestarion analysis

2.  <span>`data`</span> - sonification of arbitrary data structures,
    including automatic score (musicxml) and MIDI generation

3.  <span>`timbre`</span> - analysis and characterization of timbre from
    a (psycho-)acoustical point of view. In particular, it provides: the
    characterization of sound using, among others, Mel Frequency or
    Power Spectrum Cepstrum Coefficients (MFCC or PSCC); the
    construction of timbral networks using descriptors based on MF- or
    PS-CCs

4.  <span>`harmony`</span> - helper functions for harmonic analysis,
    design and autonomous scoring

5.  <span>`ml_utils`</span> - machine learning models for timbre
    recognition through the TensorFlow Keras framework

6.  <span>`plotting`</span> - plotting function including a module for
    automated network drawing

7.  <span>`utils`</span> - utility functions used by other modules

<span>`MUSICNTWRK`</span> is written in python 3 and requires
installation of the following dependencies (done automatically when
<span>pip install musicntwrk</span>):

1.  System modules: <span>sys, re, time, os</span>

2.  Math modules: <span>numpy, scipy</span>

3.  Data modules: <span>pandas, sklearn, networkx, python-louvain,
    tensorflow, powerlaw, ruptures, numba</span>

4.  Music and audio modules: <span>music21, librosa, pyo, pydub</span>

5.  Visualization modules: <span>matplotlib, vpython, PySimpleGUI</span>

6.  Parallelization modules: <span>mpi4py</span> (optional)

The reader is encouraged to consult the documentation of each package to
get acquainted with its purposes and use. In particular,
<span>`MUSICNTWRK`</span> relies heavily on the `music21` package for
all the music theoretical and musicological functions. In what follows
we provide the full API of <span>`MUSICNTWRK`</span> only. The display
of musical examples in <span>musicxml</span> format requires the
installation of a score app like MuseScore (https://musescore.org/). See
Section 08 of the <span>music21</span> documentation for a step by step
guide of installing a <span>musicxml</span> reader.

Finally a full set of examples and application of the library for a
variety of tasks can be downloaded from the
<span>`MUSICNTWRK`</span> repository on GitHub.

#### The `PCSet` class. 

The `PCSet` class deals with the classification and manipulation of
pitch set classes generalized to arbitrary temperament systems
(arbitrary number of pitches). The following methods are available:  
  
`def class PCSet`

<span>def \_\_init\_\_(self,pcs,TET=12,UNI=True,ORD=True)</span>

<span><span>pcs (int)</span> </span>
pitch class set as list or numpy array

<span><span>TET (int)</span> </span>
number of allowed pitches in the totality of the musical space
(temperament). Default = 12 tones equal temperament

<span><span>UNI (logical)</span> </span>
if True, eliminate duplicate pitches (default)

<span><span>ORD (logical)</span> </span>
if True, sorts the pcs in ascending order (default)

_Methods_:

 * <span>def normalOrder(self)</span>  
Order the pcs according to the most compact ascending scale in
pitch-class space that spans less than an octave by cycling
permutations.

 * <span>def normal0Order(self)</span>  
As normal order, transposed so that the first pitch is 0

 * <span>def T(self,t=0)</span>  
Transposition by t (int) units (modulo TET)

 * <span>def zeroOrder(self)</span>  
transposed so that the first pitch is 0

 * <span>def M(self,t=1)</span>  
multiply pcs by an arbitrary scalar mod. 12

 * <span>def multiplyBoulez(self,b)</span>  
pitch class multiplication of self \* b according to P. Boulez

 * <span>def I(self)</span>  
inverse operation: (-pcs modulo TET)

 * <span>def primeForm(self)</span>  
most compact normal 0 order between pcs and its inverse

 * <span>def intervalVector(self)</span>  
total interval content of the pcs

 * <span>def LISVector(self)</span>  
Linear Interval Sequence Vector: sequence of intervals in an ordered pcs

 * <span>def Op(self,name)</span>  
operate on the pcs with a distance operator

   * <span><span>name (str)</span> </span>
name of the operator O(<span>ni</span>)

 * <span>def VLOp(self,name)</span>  
operate on the pcs with a normal-ordered voice-leading operator

   * <span><span>name (str)</span> </span>
name of the operator R(<span>n\(_0\),n\(_1\),\(...\),n\(_{N_c}\)</span>)

 * <span>def forteClass(self)</span>  
Name of pcs according to the Forte classification scheme (only for
TET=12)

 * <span>def commonName(self)</span>  
Display common name of pcs (music21 function - only for TET=12)

 * <span>def commonNamePrime(self)</span>  
As above, for prime forms

 * <span>def commonNamePitched(self)</span>  
Name of chord with first pitch of pcs in normal order

 * <span>def displayNotes(self,xml=False,prime=False)</span>  
Display pcs in score in musicxml format. If prime is True, display the
prime form.

   * <span><span>xml (logical)</span> </span>
write notes on file in musicxml format

   * <span><span>prime (logical)</span> </span>
write pcs in prime form

#### The `PCSetR` class. 

The `PCSetR` class and its methods (listed below) parallels the `PCSet`
class by adding recursive capabilities to it: in practice any method
returns an instance of the class itself. This facilitates the
construction of method chains for compositional or analytical tasks. The
following methods are available:  
  
`def class PCSetR`

<span>def \_\_init\_\_(self,pcs,TET=12,UNI=True,ORD=True)</span>

<span><span>pcs (int)</span> </span>
pitch class set as list or numpy array

<span><span>TET (int)</span> </span>
number of allowed pitches in the totality of the musical space
(temperament). Default = 12 tones equal temperament

<span><span>UNI (logical)</span> </span>
if True, eliminate duplicate pitches (default)

<span><span>ORD (logical)</span> </span>
if True, sorts the pcs in ascending order (default)

_Methods_:

 * <span>def normalOrder(self)</span>  
Order the pcs according to the most compact ascending scale in
pitch-class space that spans less than an octave by cycling
permutations.

 * <span>def normal0Order(self)</span>  
As normal order, transposed so that the first pitch is 0

 * <span>def T(self,t=0)</span>  
Transposition by t (int) units (modulo TET)

 * <span>def M(self,t=1)</span>  
multiply pcs by an arbitrary scalar mod. 12

 * <span>def I(self)</span>  
inverse operation: (-pcs modulo TET)

 * <span>def multiplyBoulez(self,b)</span>  
pitch class multiplication of self \* b according to P. Boulez

 * <span>def zeroOrder(self)</span>  
transposed so that the first pitch is 0

 * <span>def inverse(self,pivot=0)</span> invert pcs around a pivot pitch

 * <span>def primeForm(self)</span>  
most compact normal 0 order between pcs and its inverse

 * <span>def intervalVector(self)</span>  
total interval content of the pcs

 * <span>def LISVector(self)</span>  
Linear Interval Sequence Vector: sequence of intervals in an ordered pcs

 * <span>def Op(self,name)</span>  
operate on the pcs with a distance operator

   * <span><span>name (str)</span> </span>
name of the operator O(<span>n\(_i\)</span>)

 * <span>def VLOp(self,name)</span>  
operate on the pcs with a normal-ordered voice-leading operator

   * <span><span>name (str)</span> </span>
name of the operator R(<span>n\(_0\),n\(_1\),\(...\),n\(_{N_c}\)</span>)

 * <span>def NROp(self,ops=None)</span>  
operate on the pcs with a Neo-Rienmanian operator

   * <span><span>ops (str)</span> </span>
name of the operator, P, L or R

 * <span>def opsNameVL(self,b,TET=12)</span>  
given a pcs returns the name of the normal-ordered voice-leading
operator R that connects self to it

 * <span>def opsNameO(self,b,TET=12)</span>  
given a pcs returns the name of the distance operator O that connects
self to it

#### The <span>PCSrow</span> class. 

<span>PCSrow</span> is a helper class for 12-tone rows operations
(T,I,R,M,Q)

  - <span>def T(self,t=0)</span>  
    Transposition by t (int) units (modulo TET)

  - <span>def I(self)</span>  
    inverse operation: (-pcs modulo TET)

  - <span>def R(self,t=1)</span>  
    Retrograde plus transposition by t (int) units (modulo TET)

  - <span>def M(self,t=1)</span>  
    Multiplication by t (int) units (modulo TET)

  - <span>def Q(self,t=1)</span>  
    cyclic permutation of stride 6 so that the result is an All Interval
    Series in normal form

  - <span>star(self)</span>  
    star of the row in prime form

  - <span>def constellation(self)</span>  
    constellation of the row

#### The `RHYTHMSeq` class. 

The <span>RHYTHMSeq</span> class and its methods (listed below) encode
various functions for rhythmic network manipulations. The
<span>RHYTHMSeq</span> class deals with the classification and
manipulation of rhythmic sequences. The following methods are
available:  
  
`def class RHYTHMSeq`

<span>\_\_init\_\_( self,rseq,REF=’e’,ORD=False)</span>

<span><span>rseq (str/fractions/floats)</span> </span>
rhythm sequence as list of strings or fractions or floats

<span><span>REF (str)</span> </span>
reference duration for prime form – the RHYTHMSeq class contains a
dictionary of common duration notes that uses the fraction module for
the definitions (implies import fraction as fr):
<span>{’w’: fr.Fraction(1,1), ’h’: fr.Fraction(1,2),’q’:
fr.Fraction(1,4), ’e’: fr.Fraction(1,8), ’s’: fr.Fraction(1/16),’t’:
fr.Fraction(1,32), ’wd’: fr.Fraction(3,2), ’hd’: fr.Fraction(3,4),’qd’:
fr.Fraction(3,8), ’ed’: fr.Fraction(3,16), ’sd’: fr.Fraction(3,32),’qt’:
fr.Fraction(1,6), ’et’: fr.Fraction(1,12), ’st’: fr.Fraction(1,24),
’qq’: fr.Fraction(1,5), ’eq’: fr.Fraction(1,10), ’sq’:
fr.Fraction(1,20)}</span>.
This dictionary can be extended by the user on a case by case need.

<span><span>ORD (logical)</span> </span>
if <span>True</span> sort durations in ascending order

_Methods_:

 * <span>def normalOrder(self)</span>  
Order the rhythmic sequence according to the most compact ascending
form.

 * <span>def augment(self,t=’e’)</span>  
Augmentation by t units

   * <span><span>t (str)</span> </span>
duration of augmentation

 * <span>def diminish(self,t=’e’)</span>  
Diminution by t units

   * <span><span>t (str)</span> </span>
duration of diminution

 * <span>def retrograde(self)</span>  
Retrograde operation

 * <span>def isNonRetro(self)</span>  
Check if the sequence is not retrogradable

 * <span>def primeForm(self)</span>  
reduce the series of fractions to prime form

 * <span>def durationVector(self,lseq=None)</span>  
total relative duration ratios content of the sequence

   * <span><span>lseq (list of fractions) </span> </span>
reference list of duration for evaluating interval content; the default
list is:
<span>fr.Fraction(1/8), fr.Fraction(2/8), fr.Fraction(3/8),
fr.Fraction(4/8), fr.Fraction(5/8), fr.Fraction(6/8), fr.Fraction(7/8),
fr.Fraction(8/8), fr.Fraction(9/8)</span>

 * <span>def durationVector(self,lseq=None)</span>  
inter-onset duration interval content of the sequence

   * <span><span>lseq (list of fractions) </span> </span>
reference list of duration for evaluating interval content; the default
list is the same as above.

#### The <span>`musicntwrk`</span> class

Defines wrappers around calls to the main functions of the different
packages. The variables passed in the <span>def</span>s are used to call
the specific function requested. See documentation of the functions in
the separate sections.

<span>__def dictionary__(self, space=None, N=None, Nc=None, order=None,
row=None, a=None, prob=None, REF=None, scorefil=None, music21=None,
show=None)</span>

define dictionary in the musical space specified in ’space’. All other
variables are as defined in <span>pcsDictionary</span>,
<span>rhythmDictionary</span>, <span>orchestralVector</span>, and
<span>scoreDictionary</span>, in Sec. [3.2](#networks) depending on the
choice of ’space’

<span><span>space (string)</span> </span>
\= ’pcs’, pitch class sets dictionary; ’rhythm’ or ’rhythmP’, rhythm
dictionaries; ’score’, score dictionary; and ’orch’ orchestral vector.

<span>*Returns*</span>  
See description in _networks_ for individual functions.

<span>__def network__(self, space=None, label=None, dictionary=None,
thup=None, thdw=None, thup\_e=None, thdw\_e=None, distance=None,
prob=None, write=None, pcslabel=None, vector=None, ops=None, name=None,
ntx=None, general=None, seq=None, sub=None, start=None, end=None,
grphtype=None, wavefil=None, cepstrum=None, color=None)</span>

define networks in the musical space specified in ’space’:

<span><span>space (string)</span> </span>
\= ’pcs’, pitch class sets network, both full and ego network from a
given pcs; ’rhythm’ or ’rhythmP’, rhythm dictionaries (see below for
details); ’score’, score dictionary; and ’orch’ orchestral vector. See
description in _networks_.

<span>*Returns*</span>  
See description in _networks_ for individual functions.

<span>__def timbre__(self, descriptor=None, path=None, wavefil=None,
standard= None, nmel=None, ncc=None, zero=None, lmax=None, maxi= None,
nbins = None, method=None, scnd=None, nstep=None)</span>  
Define sound descriptors for timbral analysis: MFCC, PSCC ASCBW in
regular or standardized form. See description of variables in Sec.
_timbre_.

<span>__def harmony__(self,descriptor=None,mode=None,x=None,y=None)</span>  
handler for calculating tonal harmony models, tonnentz and to launch the
tonal harmony calculator. See description of variables in _harmony_.

<span>__def sonify__(self, descriptor=None, data=None, length=None,
midi=None, scalemap=None, ini=None, fin=None, fac=None, dur=None,
transp=None, col=None, write=None, vnorm=None, plot=None, crm=None,
tms=None, xml=None)</span>

sonification strategies - simple sound (spectral) or score (melodic
progression). See description of variables in _data_