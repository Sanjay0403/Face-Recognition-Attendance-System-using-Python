import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'C:/Users/sanjaydeep/majorproject/imagerecognition/ImagesAttendence'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendence(name):
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d')  # Format the date
    time_string = now.strftime('%H:%M:%S')  # Format the time

    # Write date and time to CSV file
    with open('C:/Users/sanjaydeep/majorproject/imagerecognition/attendance.csv', 'a') as f:
        f.write(f'{name},{date_string},{time_string}\n')

    # Print date and time in PyCharm output
    print(f'Attendance marked for {name} at {date_string} {time_string}')

    # Return date and time strings for display or further processing if needed
    return date_string, time_string


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
photo_captured = False  # Flag to control photo capture

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            if not photo_captured:  # Capture photo only once
                markAttendence(name)
                photo_captured = True

    cv2.imshow('webcam', img)
    cv2.waitKey(1)
