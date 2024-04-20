#
# msctools: my collection of composing and performing tools in python
#
# © 2023 Marco Buongiorno Nardelli
#

def panning(tracks,mode=0):
	if mode == 0:
		# reset panning to center
		pan = [0]*len(tracks)
		for n in range(len(tracks)):
			tracks[n].panning(pan=pan[n],mode='set')
	if mode == 1:
		# distribute panning at equal angles
		pan = [-1+2/(len(tracks)-1)*p for p in range(len(tracks))]
		for n in range(len(tracks)):
			tracks[n].panning(pan=pan[n],mode='set')