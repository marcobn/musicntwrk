from distutils.core import setup
import os

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

setup(name='musicntwrk',
	version='2.0',
	description='music as data, data as music',
	author='Marco Buongiorno Nardelli',
	author_email='mbn@unt.edu',
	platforms='OS independent',
	url='www.materialssoundmusic.com',
	packages=['musicntwrk', 'musicntwrk.data','musicntwrk.harmony','musicntwrk.ml_utils','musicntwrk.networks','musicntwrk.plotting',
			  'musicntwrk.timbre','musicntwrk.utils'],
	package_dir={'musicntwrk':'src'},
	install_requires=['numpy','scipy','pandas','python-louvain','networkx','music21','librosa','pyo',
					  'matplotlib','tensorflow','powerlaw','vpython','wget','PySimpleGUI'],
	extras_require=extras
	)