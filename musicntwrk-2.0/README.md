<p></p>
<p align="center">
  <a href="https://www.musicntwrk.com">
    <img src="https://raw.githubusercontent.com/marcobn/musicntwrk/master/IMAGES/logo.png" alt="musicntwrk logo" height="84">
  </a>
</p>

<h3 align="center">music as data, data as music</h3>

<p align="center">
  <em>unleashing data tools for music theory, analysis and composition</em>
  <br>
  <br>
A python library for pitch class set and rhythmic sequences classification and manipulation, the generation of networks in generalized music and sound spaces, deep learning algorithms for timbre recognition, and the sonification of arbitrary data
<br>
</p>

## Table of contents

- [Quick start](#quick-start)
- [What's included](#whats-included)
- [Documentation](#documentation)
- [Author](#author)
- [Citation](#citation)
- [Thanks](#thanks)

## Quick start

## pip install musicntwrk
or
## pip install musicntwrk[with_MPI]
(if there is a pre-existing installation of MPI, pip will automatically install the mpi4pi wrapper)

- [OR download the latest release from GitHub](https://github.com/marcobn/musicntwrk/)
- Clone the repo: `git clone https://github.com/marcobn/musicntwrk.git`
- cd musicntwrk-2.0
- pip install .

## - [Examples and support files can be downloaded here](https://github.com/marcobn/musicntwrk/tree/master/musicntwrk-2.0/examples)
There are three example notebooks: basic, advanced harmony, advanced timbre. See the ipynb files for a full description.

## What's included
**musicntwrk** is a project written for python 3 and comprised of a main module, `musicntwrk`, and many additional helper packages included in the distribution:
- `musicntwrk` - is the main module and contains helper clasess for pitch class set classification and manipulation in any arbitrary temperament (PCSet, PCSetR and PCSrow), and the main class musicntwrk that allows the construction of generalized musical space networks using distances between common descriptors (interval vectors, voice leadings, rhythm distance, etc.); the analysis of scores, the sonification of data and the generation of compositional frameworks. `musicntwrk` acts as a wrapper for the various functions organized in the following sub projects:
  - `networks` - contains all the modules to construct dictionaries and networks of pitch class set spaces including voice leading, rhythmic spaces, timbral spaces and score network and orchestarion analysis 
  - `data` - sonification of arbitrary data structures, including automatic score (musicxml) and MIDI generation
  - `timbre` - analysis and characterization of timbre from a (psycho-)acoustical point of view. In particular, it provides: the characterization of sound using, among others, Mel Frequency or Power Spectrum Cepstrum Coefficients (MFCC or PSCC); the construction of timbral networks using descriptors based on MF- or PS-CCs
  - `harmony` - helper functions for harmonic analysis, design and autonomous scoring
  - `ml_utils` - machine learning models for timbre recognition through the TensorFlow Keras framework
  - `plotting` - plotting function including a module for automated network drawing
  - `utils` - utility functions used by other modules

## Documentation

- [musicntwrk](https://github.com/marcobn/musicntwrk/blob/master/DOCS/musicntwrk.md)
- [networks](https://github.com/marcobn/musicntwrk/blob/master/DOCS/networks.md)
- [data](https://github.com/marcobn/musicntwrk/blob/master/DOCS/data.md)
- [timbre](https://github.com/marcobn/musicntwrk/blob/master/DOCS/timbre.md)
- [harmony](https://github.com/marcobn/musicntwrk/blob/master/DOCS/harmony.md)
- [ml_utils](https://github.com/marcobn/musicntwrk/blob/master/DOCS/ml_utils.md)
- [plotting](https://github.com/marcobn/musicntwrk/blob/master/DOCS/plotting.md)

The most computationally intensive parts of the modules can be run on parallel processors using the MPI (Message Passing Interface) protocol. Communications are handled by two additional modules: `communications` and `load_balancing`. Since the user will never have to interact with these modules, we omit here a detailed description of their functions.

## Author

**Marco Buongiorno Nardelli**

Marco Buongiorno Nardelli is University Distinguished Research Professor at the University of North Texas: composer, flutist, computational materials physicist, and a member of CEMI, the Center for Experimental Music and Intermedia, and iARTA, the Initiative for Advanced Research in Technology and the Arts. He is a Fellow of the American Physical Society and of the Institute of Physics, and a Parma Recordings artist. See [here](https://www.materialssoundmusic.com/long-bio) for a longer bio-sketch.

## Citation

Marco Buongiorno Nardelli, _"musicntwrk, a python library for pitch class set and rhythmic sequences classification and manipulation, the generation of networks in generalized music and sound spaces, deep learning algorithms for timbre recognition, and the sonification of arbitrary data"_, www.musicntwrk.com (2019).

## Thanks

This project has been made possible by contributions from the following institutions:
<p align="center">
  <a href="https://www.unt.edu">
    <img src="https://raw.githubusercontent.com/marcobn/musicntwrk/master/IMAGES//unt.png" alt="UNT logo" height="148" align="bottom">
  </a>&ensp;&ensp;&ensp;
  <a href="https://cemi.music.unt.edu">
    <img src="https://raw.githubusercontent.com/marcobn/musicntwrk/master/IMAGES/cemi.png" alt="CEMI logo" height="84" align="bottom">
  </a>&ensp;&ensp;&ensp;
  <a href="https://www.prism.cnrs.fr">
    <img src="https://raw.githubusercontent.com/marcobn/musicntwrk/master/IMAGES/prism.png" alt="PRISM logo" height="132" align="bottom">
  </a>&ensp;&ensp;&ensp;
  <a href="https://imera.univ-amu.fr">
    <img src="https://raw.githubusercontent.com/marcobn/musicntwrk/master/IMAGES//imera.png" alt="IMeRA logo" height="72" align="bottom">
  </a>
</p>
<p>
<hr>
</p>

<p align="center">
<strong>musicntwrk</strong> is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but <strong>WITHOUT ANY WARRANTY</strong>; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <a href="http://www.gnu.org/licenses/"> http://www.gnu.org/licenses/</a>.
</p>
<p></p>
<p align="center">
Copyright (C) 2018-2020 Marco Buongiorno Nardelli  <br>
<a href="https://www.materialssoundmusic.com"> www.materialssoundmusic.com <br>
<a href="https://www.musicntwrk.com"> www.musicntwrk.com <br>
<a href="mailto:mbn@unt.edu"> mbn@unt.edu <br>
</p>
