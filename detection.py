# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detectionImage(uploadpath, filename):
    sourcepath = uploadpath + '/' + filename
    outputpath = uploadpath + '/Detection_' + filename

    # Gray image load
    img = cv2.imread(sourcepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # found face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y),(x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for(ex, ey, ew, eh) in eyes:
        #    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh),(0,255,0), 2)

    cv2.imwrite(outputpath, img)

    cv2.destroyAllWindows()

def tftraceImage(imagepath):
    # Gray image load
    img = cv2.imread(imagepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi_gray = np.zeros(gray.shape)

    # found face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]

    return roi_gray
