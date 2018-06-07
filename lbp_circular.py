import Tkinter
import tkMessageBox
import os
import cv2
import numpy as np
from PIL import Image
import mailpeople

#----------------------------------------------------------------------------
top = Tkinter.Tk()

#----------------------------------------------------------------------------
def addDataset():
    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture('video/disha2.mp4')
    window = Tkinter.Toplevel(top)
    
    id = raw_input('Enter user id\n')
    
    path = "dataSet"
    sampleNum = -1
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    for imagepath in imagePaths:
        #print(imagepath)
        #print(os.path.split(imagepath)[-1].split('.')[1])
        if(int(os.path.split(imagepath)[-1].split('.')[1]) == int(id)):
            #print(index)
            index = int(os.path.split(imagepath)[-1].split('.')[2])
            if(index > sampleNum):
                sampleNum = index

    count = 0
    while(True):
        ret,img = cam.read();
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            sampleNum = sampleNum+1
            count = count+1
            cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.waitKey(100)
        cv2.imshow("Face", img)
        cv2.waitKey(1)
        if(count>=100):
            break
    cam.release()
    cv2.destroyAllWindows()

#----------------------------------------------------------------------------
def createDataset():
    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture('video/sample.mp4')
    window = Tkinter.Toplevel(top)
    fh = open("namelist.txt","a")
    mystring = Tkinter.StringVar()

    def getName():
        fh.write("\n"+mystring.get())
        fh.close()

    def closeWindow():
        window.destroy()



    E1 = Tkinter.Entry(window, textvariable = mystring, text ="Create Dataset").place(x=50, y=50)
    B1 = Tkinter.Button(window, text="OK", command = lambda:[getName(),closeWindow()]).place(x=80, y=80)

    id = raw_input('Enter user id\n')
    sampleNum = 0
    while(True):
        ret,img = cam.read();
        #den = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        faces = faceDetect.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            sampleNum = sampleNum+1
            cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.waitKey(100)
        cv2.imshow("Face", img)
        cv2.waitKey(1)
        if(sampleNum>=100):
            break
    cam.release()
    cv2.destroyAllWindows()

#----------------------------------------------------------------------------
def train():
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius = 2, grid_x = 8, grid_y = 8, threshold = 100.0)
    path = "dataSet"

    def getImagesWithID(path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        faces = []
        IDs = []
        for imagePath in imagePaths:
            faceImg = Image.open(imagePath).convert('L')
            faceNp = np.array(faceImg, 'uint8')
            faceNp = cv2.equalizeHist(faceNp)    #added
            ID = int(os.path.split(imagePath)[-1].split('.')[1])
            faces.append(faceNp)
            print ID
            IDs.append(ID)
            cv2.imshow("training", faceNp)
            cv2.waitKey(10)
        return IDs, faces

    Ids, faces = getImagesWithID(path)
    recognizer.train(faces, np.array(Ids))
    recognizer.save('trainer/training1.yml')
    cv2.destroyAllWindows()

#----------------------------------------------------------------------------
def detector():
    fh = open("namelist.txt","r")
    lines = fh.read().splitlines()
    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)
    rec = cv2.face.LBPHFaceRecognizer_create(radius = 2, grid_x = 8, grid_y = 8, threshold = 100.0)
    rec.read("trainer/training1.yml")

    set_names = set()       #added

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 255, 255)
    while(True):
        id = 0
        ret, img = cam.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        gray = cv2.equalizeHist(gray)    #added
        faces = faceDetect.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            id, conf = rec.predict(gray[y:y+h,x:x+w])

            name = lines[id]

            set_names.add(name)

            cv2.putText(img,str(conf),(20,20),fontFace, fontScale, fontColor)
            cv2.putText(img,str(name),(x,y+h),fontFace, fontScale, fontColor)
        cv2.imshow("face",img)
        if(cv2.waitKey(1) == ord('q')):
            break

    flag = 0
    f = open("people.txt","w")
    fh = open("wanted.txt","r")
    lines = fh.read().splitlines()

    for line in lines:
        if line in set_names:
            flag = 1
            f.write(line+"\n")
    f.close()

    if(flag == 1):
        mailpeople.main()
    cam.release()
    cv2.destroyAllWindows()

#----------------------------------------------------------------------------
B1 = Tkinter.Button(top, text ="Create Dataset", command = createDataset)
B1.place(x=10, y=10)

B2 = Tkinter.Button(top, text="Train", command = train)
B2.place(x=75, y=50)

B3 = Tkinter.Button(top, text="Check Results", command = detector)
B3.place(x=100,y=100)

B4 = Tkinter.Button(top, text ="Add Dataset", command = addDataset)
B4.place(x=170, y=170)

top.geometry("280x220")
top.mainloop()
