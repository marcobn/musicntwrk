`musicntwrk` - is the main module and contains helper clasess for pitch class set classification and manipulation in any arbitrary temperament (PCSet, PCSetR and PCSrow), and the main class musicntwrk that allows the construction of generalized musical space networks using distances between common descriptors (interval vectors, voice leadings, rhythm distance, etc.); the analysis of scores, the sonification of data and the generation of compositional frameworks.

`the PCSet class` <br>
The PCSet class deals with the classification and manipulation of pitch set classes generalized
to arbitrary temperament systems (arbitrary number of pitches). The following
methods are available:

- `def init (self,pcs,TET=12,UNI=True,ORD=True)`
	- pcs (int) pitch class set as list or numpy array
	- TET (int)number of allowed pitches in the totality of the musical space
	   (temperament). Default = 12 tones equal temperament
	- UNI (logical) if True, eliminate duplicate pitches (default)
	- ORD (logical) if True, sorts the pcs in ascending order (default)
- `def normalOrder(self)` -
	Order the pcs according to the most compact ascending scale in pitch-class
	space that spans less than an octave by cycling permutations.
- `def normal0Order(self)` -
	As normal order, transposed so that the first pitch is 0
- `def T(self,t=0)` -
	Transposition by t (int) units (modulo TET)
- `def zeroOrder(self)` -
	transposed so that the first pitch is 0
- `def I(self)` -
	inverse operation: (-pcs modulo TET)
- `def primeForm(self)` -
	most compact normal 0 order between pcs and its inverse
- `def intervalVector(self)` -
	total interval content of the pcs
- `def LISVector(self)` -
	Linear Interval Sequence Vector: sequence of intervals in a normal ordered pcs
- `def intervals(self)` -
	Linear Interval Sequence Vector: sequence of intervals in an ordered pcs
- `def Op(self,name)` -
	operate on the pcs with a distance operator
	- name (str) name of the operator O(ni)
- `def VLOp(self,name)` -
	operate on the pcs with a voice leading operator
	- name (str) name of the operator O(ni)
- `def M(self,t=1)` -
	Multiplication by t (int) units (modulo TET)
- `def multiplyBoulez(self,b)` -
	Boulez pitch class multiplication of self.pcs x b
- `def forteClass(self)` -
	Name of pcs according to the Forte classification scheme (only for TET=12)
- `def commonName(self)` -
	Display common name of pcs (music21 function - only for TET=12)
- `def commonNamePrime(self)` -
	As above, for prime forms
- `def commonNamePitched(self)` -
	Name of chord with first pitch of pcs in normal order
- `def displayNotes(self,xml=False,prime=False)` -
	Display pcs in score in musicxml format. If prime is True, display the prime
	form.
   - xml (logical) write notes on file in musicxml format
   - prime (logical) write pcs in prime form

`the PCSetR class` <br>
This is a copy of the PCSet class that allows for recursive application of methods. Methods added:

- `def NROp(self,ops=None)` -
	apply Neo Rienmannian Operator to the pcs (works for triads)
- `def opsNameO(self,b)` -
	given two vectors returns the name of the distance operator O that connects them
- `def opsNameVL(self,b)` -
	given two vectors returns the name of the normal-ordered voice-leading operator that connects them
	
`the PCSrow class` <br>
Helper class for 12-tone rows operations (T,I,R,M,Q)

- `def T(self,t=0)` -
	Transposition by t (int) units (modulo TET)
- `def I(self)` -
	inverse operation: (-pcs modulo TET)
- `def R(self,t=1)` -
	Retrograde plus transposition by t (int) units (modulo TET)
- `def M(self,t=1)` -
	Multiplication by t (int) units (modulo TET)
- `def Q(self,t=1)` -
	cyclic permutation of stride 6  so that the result is
	an All Interval Series in normal form
- `star(self)` -
	star of the row in prime form
- `def constellation(self)` -
	constellation of the row (following Morris and Starr)
	
`the RHYTHMSeq class` <br>
The RHYTHMSeq class deals with the classification and manipulation of
rhythmic sequences. The following methods are available:

- `def init ( self,rseq,REF=’e’,ORD=False)`
    - rseq (str/fractions/floats)rhythm sequence as list of strings/fractions/floats
       name
    - REF (str) reference duration for prime form the RHYTHMSeq class
       contains a dictionary of common duration notes that uses the fraction
       module for the definitions (implies import fraction as fr):{’w’:fr.Fraction(1,1),
       ’h’:fr.Fraction(1,2),’q’:fr.Fraction(1,4), ’e’:fr.Fraction(1,8),
       ’s’:fr.Fraction(1/16),’t’:fr.Fraction(1,32), ’wd’:fr.Fraction(3,2),
       ’hd’:fr.Fraction(3,4),’qd’:fr.Fraction(3,8), ’ed’:fr.Fraction(3,16),
       ’sd’:fr.Fraction(3,32),’qt’:fr.Fraction(1,6), ’et’:fr.Fraction(1,12),
       ’st’:fr.Fraction(1,24), ’qq’:fr.Fraction(1,5), ’eq’:fr.Fraction(1,10),
       ’sq’:fr.Fraction(1,20)}. This dictionary can be extended by the user
       on a case by case need.
    - ORD (logical) ifTruesort durations in ascending order
- `def normalOrder(self)` -
    Order the rhythmic sequence according to the most compact ascending form.
- `def augment(self,t=’e’)` -
    Augmentation by t units
    - t (str duration of augmentation
- `def diminish(self,t=’e’)` -
    Diminution by t units
    - t (str) duration of diminution
- `def retrograde(self)` -
    Retrograde operation
- `def isNonRetro(self)` -
    Check if the sequence is not retrogradable
- `def primeForm(self)` -
    reduce the series of fractions to prime form
- `def durationVector(self,lseq=None)` -
    total relative duration ratios content of the sequence
   - lseq (list of fractions) reference list of duration for evaluating
      interval content the default list is: [fr.Fraction(1/8),fr.Fraction(2/8),
      fr.Fraction(3/8), fr.Fraction(4/8),fr.Fraction(5/8), fr.Fraction(6/8),
      fr.Fraction(7/8), fr.Fraction(8/8), fr.Fraction(9/8)]
- `def durationVector(self,lseq=None)` -
    inter-onset duration interval content of the sequence
   - lseq (list of fractions) reference list of duration for evaluating
      interval content the default list is the same as above.

`The musicntwrk class` <br>
Defines wrappers around calls to the main functions of the different packages. The variables passed in the def are used to call the 
specific function requested. See documentation of the functions in the separate sections.

- `def dictionary(self,space=None,N=None,Nc=None,order=None,row=None,a=None,prob=None,REF=None,scorefil=None,music21=None,show=None)` -
	define dictionary in the musical space specified in 'space': pcs, rhythm, rhythmP, score, orch
- `def network(self,space=None,label=None,dictionary=None,thup=None,thdw=None,thup_e=None,thdw_e=None,distance=None,prob=None,write=None,pcslabel=None,vector=None,ops=None,name=None,ntx=None,general=None,seq=None,sub=None,start=None,end=None,grphtype=None,wavefil=None,cepstrum=None,color=None)` -
	define networks in the musical space specified in 'space': pcs (reg and ego), vLead (reg, vec, name and nameVec), 
	rhythm, rLead, score (reg, name and sub), timbre, orch
- `def timbre(self,descriptor=None,path=None,wavefil=None,standard=None,nmel=None,ncc=None,zero=None,lmax=None,maxi=None,nbins=None,method=None,scnd=None,nstep=None)` -
	Define sound descriptor for timbral analysis: MFCC, PSCC ASCBW in regular or standardized form
- `def harmony(self,descriptor=None,mode=None,x=None,y=None)` - 
	handler for calculating tonal harmony models, tonnentz and to launch the tonal harmony calculator
- `def sonify(self,descriptor=None,data=None,length=None,midi=None,scalemap=None,ini=None,fin=None,fac=None,dur=None,transp=None,col=None,write=None,vnorm=None,plot=None,crm=None,tms=None,xml=None)`
	sonification strategies - simple sound (spectral) or score (melodic progression)
