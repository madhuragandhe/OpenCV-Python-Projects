import cv2

face_cascade= cv2.CascadeClassifier('C:/Users/Admin/PycharmProjects/experiments/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
#eye_cascade= cv2.CascadeClassifier('C:/Users/Admin/PycharmProjects/experiments/venv/Lib/site-packages/cv2/data/haarcascade_eye.xml')
smile_cascade= cv2.CascadeClassifier('C:/Users/Admin/PycharmProjects/experiments/venv/Lib/site-packages/cv2/data/haarcascade_smile.xml')

def smile_detector(gray,frame):
    faces= face_cascade.detectMultiScale(gray,1.3,5)
    i=20
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x-i,y-i),((x+w+i),(y+h+i)),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_frame=frame[y:y+h,x:x+w]
        smiles=smile_cascade.detectMultiScale(roi_gray,scaleFactor=1.7,minNeighbors=3,minSize=(15,15))
        if len(smiles) ==0:
            cv2.putText(frame, 'Please Smile:)', (40, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_frame,(sx,sy),((sx+sw),(sy+sh)),(0,0,255),2)
    return frame


video_capture=cv2.VideoCapture(0)
while True:
    ret, frame=video_capture.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas= smile_detector(gray, frame)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
