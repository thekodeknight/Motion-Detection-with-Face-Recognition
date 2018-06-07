import datetime
import Tkinter
import tkMessageBox
import os
import cv2
import numpy as np
from PIL import Image
import mailpeople

now = datetime.datetime.now()
cap = cv2.VideoCapture('video/disha1.mp4')

def diffImg(t0, t1):

    d1 = cv2.absdiff(t1, t0)

    return d1

# Get current width of frame
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
# Get current height of frame
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videofilename = 'output/output_' + str(now.strftime("%Y%m%d_%H%M")) + '.avi'
out = cv2.VideoWriter( videofilename , fourcc, 20.0, (int(width),int(height)))

fh = open("namelist.txt","r")
lines = fh.read().splitlines()
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("trainer/training.yml")

set_names = set()       #added

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)

while( cap.isOpened()):
    ret, frame1 = cap.read()
    ret2, frame2 = cap.read()

    if ret == True:

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        ret3, thresh = cv2.threshold( diffImg(gray1, gray2) ,127,255,0)

        thresh = cv2.GaussianBlur(thresh,(5,5),0)           #Noise reduction
        ret3, thresh = cv2.threshold( thresh ,127,255,0)
        
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        gray1 = cv2.equalizeHist(gray1)    #added

        if (len(contours) > 0):

            id = 0
            faces = faceDetect.detectMultiScale(gray1,1.3,5)
            for(x,y,w,h) in faces:
                cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,0,255),2)
                id, conf = rec.predict(gray1[y:y+h,x:x+w])
                name = lines[id]
                set_names.add(name)

                cv2.putText(frame1,str(conf),(20,20),fontFace, fontScale, fontColor)
                cv2.putText(frame1,str(name),(x,y+h),fontFace, fontScale, color = 255)
            #cv2.imshow("face",frame1)
            out.write(frame1)

        cv2.imshow('frame', frame1)
        cv2.imshow('thresh', thresh)
        cv2.imshow('contour', im2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

f = open("people.txt","w")
fh = open("wanted.txt","r")
lines = fh.read().splitlines()

flag = 0
for line in lines:
    if line in set_names:
        flag = 1
        f.write(line+"\n")
f.close()

if(flag==1):
    mailpeople.main()
cap.release()
out.release()
cv2.destroyAllWindows()
