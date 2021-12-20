import setuptools
from distutils.core import setup
import os,sys
from os import path

data = os.path.join('src','data')
harmony = os.path.join('src','harmony')
ml_utils = os.path.join('src','ml_utils')
networks = os.path.join('src','networks')
plotting = os.path.join('src','plotting')
timbre = os.path.join('src','timbre')
utils = os.path.join('src','utils')

extras = {
	'with_MPI': ['mpi4py']
}

# read the contents of your README file including images
this_directory = './'
#with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
with open(path.join(this_directory, 'README.md')) as f:
	long_description = f.read()


setup(name='musicntwrk',
	version='2.2.14',
	description='music as data, data as music',
	long_description=long_description,
	long_description_content_type='text/markdown',
	author='Marco Buongiorno Nardelli',
	author_email='mbn@unt.edu',
	platforms='OS independent',
	url='https://www.musicntwrk.com',
	packages=['musicntwrk', 'musicntwrk.data','musicntwrk.harmony','musicntwrk.ml_utils','musicntwrk.networks','musicntwrk.plotting',
		'musicntwrk.timbre','musicntwrk.utils','musicntwrk.comptools'],
	package_dir={'musicntwrk':'src'},
	install_requires=['numpy','scipy','pandas','python-louvain','networkx','music21','librosa','numba','pyo',
		'matplotlib','tensorflow','powerlaw','vpython','wget','PySimpleGUI','pydub','ruptures'],
	extras_require=extras
)

