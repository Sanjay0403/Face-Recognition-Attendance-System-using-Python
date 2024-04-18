import cv2
import numpy as np
import face_recognition

imgTestimage = face_recognition.load_image_file('C:/Users/sanjaydeep/majorproject/imagerecognition/ImagesBasics/Hemanth.jpg')
imgTestimage = cv2.cvtColor(imgTestimage, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('C:/Users/sanjaydeep/majorproject/imagerecognition/ImagesBasics/sanjay.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgTestimage)[0]
encodeTestimage = face_recognition.face_encodings(imgTestimage)[0]
cv2.rectangle(imgTestimage,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeTestimage],encodeTest)
faceDis = face_recognition.face_distance([encodeTestimage],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)} ',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)

cv2.imshow('Elon musk', imgTestimage)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)
