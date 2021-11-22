import cv2
import time
import numpy as np 
import handtrackingusingmediapipe as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
widthcam,hcam = 640, 480




cap = cv2.VideoCapture(0)
cap.set(3,widthcam)
cap.set(4,hcam)
presentTime = 0

detector = htm.Handetector(detectionCon = 0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]
volbar = 400
vol = 0


while True:
	success, img = cap.read()
	img = detector.findHands(img)
	lmList = detector.findPosition(img, draw = False)
	if len(lmList) != 0:
		#print(lmList[4], lmList[8])

		x1, y1 = lmList[4][1], lmList[4][2]
		x2, y2 = lmList[8][1], lmList[8][2]
		cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
		cv2.circle(img, (x1, y1), 10, (255,0,0), cv2.FILLED)
		cv2.circle(img, (x2, y2), 10, (255,0,0), cv2.FILLED)
		cv2.circle(img, (cx, cy), 10, (255,0,0), cv2.FILLED)
		cv2.line(img, (x1,y1), (x2,y2), (255,0,0), 3)

		length = math.hypot(x2-x1, y2-y1)
		#print(length)

		vol = np.interp(length, [50,150], [minVol, maxVol])
		volbar = np.interp(length, [50,150], [400, 150])
		print(int(length), vol)
		volume.SetMasterVolumeLevel(vol, None)



		if length < 50:
			cv2.circle(img, (cx, cy), 10, (0,255,0), cv2.FILLED)

	cv2.rectangle(img, (50, 150), (85,400), (2555,0,255,0), 3)
	cv2.rectangle(img, (50, int(volbar)), (85,400), (2555,0,255,0), cv2.FILLED)








	#print(lmList)

	currentTime = time.time()
	fps = 1/(currentTime - presentTime)
	presentTime = currentTime

	cv2.putText(img, str(int(fps)), (10,50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3 )
    
	cv2.imshow("Img", img)
  
	cv2.waitKey(1)
