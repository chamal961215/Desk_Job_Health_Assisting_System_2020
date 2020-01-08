
import cv2
import numpy as np
import time

face_cascade = cv2.CascadeClassifier('C:\\Users\\HP-PC\\Desktop\\python\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\HP-PC\\Desktop\\python\\haarcascade_eye.xml')

video=cv2.VideoCapture(1)

last_time = time.time()
nframes = 0
fps = 0

while True:
    _, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
        
    
    cv2.imshow("Frame", frame)
    key=cv2.waitKey(1)
    
    fps = int(1/(time.time()-last_time))
    print('FPS {}'.format(fps))
    last_time = time.time()
    
    
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()