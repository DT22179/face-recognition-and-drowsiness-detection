import pandas as pd
import numpy as np
import face_recognition as fr
import streamlit as st 
import os
import cv2
from datetime import datetime
import tensorflow as tf
import tensorflow.keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

##Loading model for drowsiness detection
model = load_model("C:/Users/Dishant Toraskar/Deep Learning Projects/DL + MLE Capstone Project/drowsiness_model_2.h5")
##Streamlit title
st.title("Face Recognition & Drowsiness Detection")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

path = 'Data/Faces'
images = []
studentName = []
studentList = os.listdir(path)
global pred

##Looping through the images and adding the name to the list by splitting out the extension
for cu_img in studentList:
    current_img = cv2.imread(f"{path}/{cu_img}")
    images.append(current_img)
    studentName.append(os.path.splitext(cu_img)[0])


##Face Encodings
def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = faceEncodings(images)

##Starting the camera
camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = fr.face_locations(faces)
    encodeCurrentFrame = fr.face_encodings(faces, facesCurrentFrame)
    
    for encodeFace, faceLoc in zip(encodeCurrentFrame, facesCurrentFrame):
        matches = fr.compare_faces(encodeListKnown, encodeFace)
        faceDist = fr.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = studentName[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 -35), (x2 , y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)



        ret, img = camera.read()
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        # grayscale image for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

        for x, y, w, h in eyes:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) == 0:
                # print("Eyes not detected")
                pass
            else:
                for (ex, ey, ew, eh) in eyes:
                    eyes_roi = roi_color[ey: ey + eh, ex: ex + ew]

                final_image = cv2.resize(eyes_roi, (224, 224))
                final_image = np.expand_dims(final_image, axis=0) ##Expanding to 4 dimensions
                final_image = final_image / 255.0 ##Normalizing
                            
                

                pred = model.predict(final_image)
                if (pred > 0.1):
                    status = "Open Eyes"
                else:
                    status = "Closed Eyes"

                cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                

        ##Marking the log session timing/attendance into csv file.
        df = pd.read_csv(r"C:\Users\Dishant Toraskar\Deep Learning Projects\DL + MLE Capstone Project\Data\attendance.csv")
        if(name not in df['Name'].values):
            now = datetime.now()
            date = now.strftime("%d-%m-%Y")
            time = now.strftime("%H:%M:%S")
            new_row = {'Name': name, 'Date' : date, 'Time' : time}
            new_df = df.append(new_row, ignore_index=True)
            new_df.to_csv(r"C:\Users\Dishant Toraskar\Deep Learning Projects\DL + MLE Capstone Project\Data\attendance.csv", index = False)
        else:
            pass

    FRAME_WINDOW.image(frame)

    

else:
    st.write("Webcam Stopped")
