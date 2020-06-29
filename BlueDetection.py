import cv2
import numpy as np

video_capture=cv2.VideoCapture(0)
while True:
    frame=video_capture.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower=np.array([110,50,50])
    upper=np.array([130,255,255])
    mask=cv2.inRange(hsv,lower,upper)
    blue=cv2.bitwise_and(frame,frame,mask=mask)
    cv2.imshow("original ",frame)
    cv2.imshow("blue ",blue)
    if cv2.waitKey(0):
        break

cv2.destroyAllWindows()
video_capture.release()
