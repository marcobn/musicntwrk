#
# msctools: my collection of composing and performing tools in python
#
# © 2023 Marco Buongiorno Nardelli
#

# Hand detection for live processing using mediapipe (version 0.9.1)

import cv2
import mediapipe as mp
import time, re, threading
import matplotlib.pyplot as plt
import numpy as np

from pythonosc.udp_client import SimpleUDPClient

class client:
	def __init__(self,address,values,host="127.0.0.1",port=11000):
		self.host = host
		self.port = port
		self.address = address
		self.values = values
		
	def send(self):
		return SimpleUDPClient(self.host,self.port).send_message(self.address,self.values)


# verbose
verbose = False

# annotate
annotate = True

# draw trajectories
draw = False

# send landmark via OSC
oscmessage = True

# stop thread
stop = False

# max number of hands
numhands = 2
normalize = True

# video FPS
fps = 30

# camera ID
camera = 0

# Create a hand landmarker instance with the live stream mode:
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Callback function to map results to a list
def get_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
	global detection_result
	detection_result = result
	try:
		if detection_result.handedness[0] != None:
			n_hands = len(detection_result.handedness)
			tmp = [None]*n_hands
			for n in range(n_hands):
				tmp[n] = [str(detection_result.handedness[n][0]).split("'")[-2]]
				if normalize:
					nlandmrk = int(len(detection_result.hand_landmarks[n]))
					for nl in range(nlandmrk):
						tmp[n].append([re.split('=|,',str(detection_result.hand_landmarks[n][nl]))[1],
										re.split('=|,',str(detection_result.hand_landmarks[n][nl]))[3],
										re.split('=|,',str(detection_result.hand_landmarks[n][nl]))[5]])
				else:
					nlandmrk = int(len(detection_result.hand_world_landmarks[n]))
					for nl in range(nlandmrk):
						tmp[n].append([re.split('=|,',str(detection_result.hand_world_landmarks[n][nl]))[1],
										re.split('=|,',str(detection_result.hand_world_landmarks[n][nl]))[3],
										re.split('=|,',str(detection_result.hand_world_landmarks[n][nl]))[5]])

			detection_result = tmp
	except:
		pass
	return

# print function for troubleshooting
def print_result():
	# print reading of landmarks
	try:
		if len(detection_result) != None:
			n_hands = len(detection_result)
			for n in range(n_hands):
				print(detection_result[n][0],detection_result[n][1])
	except:
		pass
	return

# function to save hand landmark trajectories
def lndmrk_traj(lhand,lnd,trajx,trajy):
	# save trajectory of selected hand/landmark
	while True:
		try:
			if len(detection_result) != None:
				n_hands = len(detection_result)
				for n in range(n_hands):
					if detection_result[n][0] == lhand:
						trajx.append(float(detection_result[n][lnd+1][0]))
						trajy.append(float(detection_result[n][lnd+1][1]))
		except:
			pass
		if stop:
			break
		time.sleep(1/fps)
	return
				
def annotate_hand(h,w):
	try:
		if len(detection_result) != None:
			n_hands = len(detection_result)
			for n in range(n_hands):
				# hand label
				hand = detection_result[n][0]
				hx = w-int(float(detection_result[n][1][0])*w)
				hy = int(float(detection_result[n][1][1])*h)
				cv2.putText(image,detection_result[n][0],(hx-100,hy),cv2.FONT_HERSHEY_DUPLEX,4,(0,200,0),3)
				# fingertips
				for i in [5,9,13,17,21]:
					hx = w-int(float(detection_result[n][i][0])*w)
					hy = int(float(detection_result[n][i][1])*h)
					cv2.circle(image,(hx,hy),20,(200,0,0),-1)
	except:
		pass
		
def sendOSC():
	try:
		if len(detection_result) != None:
			n_hands = len(detection_result)
			for n in range(n_hands):
				# message
				hand = detection_result[n][0]
				for i in [5,9,13,17,21]:
						hx = w-int(float(detection_result[n][i][0])*w)
						hy = int(float(detection_result[n][i][1])*h)
						client("/hand/landmark",[hand,i,hx,hy]).send()
	except:
		pass


# Option for the HandLandmarker
options = HandLandmarkerOptions(
	base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
	running_mode=VisionRunningMode.LIVE_STREAM,
	num_hands = numhands,
	result_callback=get_result)

# Start video capture
img = cv2.VideoCapture(camera)

# Example of threading function that uses the result of the hand detection
# Save trajectory of a chosen landmark in a given hand
Lx = []
Ly = []
threading.Thread(target=lndmrk_traj,args=('Left',0,Lx,Ly)).start()
Rx = []
Ry = []
threading.Thread(target=lndmrk_traj,args=('Right',0,Rx,Ry)).start()


# Main loop
with HandLandmarker.create_from_options(options) as landmarker:
	timestamp = 0.0
	while img.isOpened():
		success, image = img.read()
		h, w, c = image.shape
		timestamp += 1
		
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.flip(image,1))
		
		result = landmarker.detect_async(mp_image, int(timestamp))
		
		if verbose: print_result()
		
		if annotate: annotate_hand(h,w)
		
		if oscmessage: sendOSC()
		
		cv2.imshow('processed image',cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			stop = True
			break

if draw:
	# Plot trajectory
	plt.plot(np.array(Lx)*w,np.array(Ly)*h,'o')
	plt.plot(np.array(Rx)*w,np.array(Ry)*h,'o')
	plt.xlim(0,w)
	plt.ylim(0,h)
	plt.show()
