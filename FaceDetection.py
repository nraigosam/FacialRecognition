import cv2
import numpy as np
import face_recognition
import os
import time
# from datetime import datetime, time
import pickle
# from PIL import ImageGrab
 
class FaceDetection:
    def __init__(self):
        classNames = self.encodeNames()
        encodeListKnown = self.encodeFaces()
        self.detection(encodeListKnown, classNames)


    #### FOR CAPTURING SCREEN RATHER THAN WEBCAM
    # def captureScreen(bbox=(300,300,690+300,530+300)):
    #     capScr = np.array(ImageGrab.grab(bbox))
    #     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
    #     return capScr

    def encodeFaces(self):
        with open('dataset_faces.dat', 'rb') as f:
            encodeListKnown = pickle.load(f)
        return encodeListKnown

    def encodeNames(self):
        with open('dataset_names.dat', 'rb') as f:
            encodeClassNames = pickle.load(f)
        return encodeClassNames

    
    def detection(self, encodeListKnown, classNames):
        self.face_haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(0)
        
        while True:
            _, img = cap.read()
            # img = face_recognition.load_image_file('imagen.jpg')
            # img = captureScreen()
            # print("asdfasf:" , img)
            
            imgS = cv2.resize(img,(0,0),None,0.25,0.25)
            imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # faces_detected = self.face_haar_cascade.detectMultiScale(imgS, scaleFactor=1.05,
            #                                                      minNeighbors=5, minSize=(30, 30),
            #                                                      flags=cv2.CASCADE_SCALE_IMAGE)
        
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS)


            for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
                # matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
                matchIndex = np.argmin(faceDis)
            
                if faceDis[matchIndex]< 0.50:
                    
                    name = classNames[matchIndex].upper()
                    name = name.split('_')[0]

                else: name = 'Unknown'
                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 = y1,x2,y2,x1
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        
            cv2.imshow('Webcam',img)
            cv2.waitKey(1)

        


faceDetection = FaceDetection()