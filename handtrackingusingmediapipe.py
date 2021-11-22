import cv2
import mediapipe as mp 
import time


class Handetector():
	def __init__(self,mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
		self.mode = mode
		self.maxHands = maxHands
		self.detectionCon = detectionCon
		self.trackCon = trackCon

		self.mpHands = mp.solutions.hands
		self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
		self.mpDraw = mp.solutions.drawing_utils

	def findHands(self, img, draw = True ):
			imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			self.result =self.hands.process(imgRGB)
			#print(result.multi_hand_landmarks)


			if self.result.multi_hand_landmarks:
				for handLms in self.result.multi_hand_landmarks:
					if draw:
						self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

			return img
	def findPosition(self, img, handNum = 0, draw = True ):
	    	lmList = []
	    	if self.result.multi_hand_landmarks:
	    		myhand = self.result.multi_hand_landmarks[handNum]

	    		for id, lm in enumerate(myhand.landmark):
	    			h,w,c = img.shape
	    			cx, cy = int(lm.x*w), int(lm.y*h)
	    			lmList.append([id,cx,cy])
	    			if draw:
	    				cv2.circle(img, (cx,cy), 6, (255,0,0,0), cv2.FILLED)
    		return lmList


			  





def main():
	previoustime = 0
	current_time = 0
	cap = cv2.VideoCapture(0)
	detector = Handetector()


	while True:
		success, img = cap.read()
		img = detector.findHands(img)
		lmList = detector.findPosition(img)
		if len(lmList) != 0:
			print(lmList[4])

		current_time = time.time()
		fps = 1/(current_time-previoustime)
		previoustime = current_time
		cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3 )


		cv2.imshow("Image", img)
		cv2.waitKey(1)




if __name__ == "__main__":
	main()

