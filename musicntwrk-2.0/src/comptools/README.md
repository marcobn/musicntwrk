The modules in this directory are from pyknon, a python library built by 
Pedro Kroger, and documented in the book [Music for Geeks and Nerds](https://pedrokroger.net/mfgan/).
They are included included in musicntwrk under the MIT license and by the author's permission.

What follows is the original README from https://github.com/kroger/pyknon.

***

Pyknon is a simple music library for Python hackers. With Pyknon you
can generate Midi files quickly and reason about musical
properties.

Pyknon works with Python 2.7 and 3.2.

# Usage

Pyknon is very simple to use, here's a basic example to create 4 notes
and save into a MIDI file::

```python
from pyknon.genmidi import Midi
from pyknon.music import NoteSeq

notes1 = NoteSeq("D4 F#8 A Bb4")
midi = Midi(1, tempo=90)
midi.seq_notes(notes1, track=0)
midi.write("demo.mid")
```

See the documentation for more details.

# Documentation

Pyknon needs better documentation. Meanwhile you can download for free my book [Music for Geeks and Nerds](https://pedrokroger.net/mfgan/) that explains how to use Pyknon.

# License

This library is released under a MIT license. See the [LICENSE](LICENSE) file for
more details.

Pyknon's MIDI module is heavily based on Mark Conway Wirt's MIDIUtil.

***
