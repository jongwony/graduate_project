# -*- coding: UTF-8 -*-

import numpy as np
import cv2
import pdb




face_cascade = cv2.CascadeClassifier('/var/www/flask/haarcascade_frontalface_default.xml')


filename = 'mov_bbb.mp4'


cap = cv2.VideoCapture(filename)





global detect_face
global trackzone
global prev_gray
global roi_gray

global hist_snap




trackzone = np.array([], np.int32)
roi_gray = np.zeros
hist_snap = np.zeros







ret, prev_frame = cap.read()
if ret:
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
else:
    print "Video File %s Not Found" % filename










while(1):
    
    
    ret, frame = cap.read()
    
    
    
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)




        ################### Detection

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
        








        if faces == tuple():
            detect_face = False
        else:
            detect_face = True








        if detect_face:
            #pdb.set_trace()
            trackzone = faces


        




        #################### Tracking

        for x, y, w, h in trackzone:
            
            ################ Histogram
            
            if hist_snap != []:
                hist_snap = cv2.calcHist([prev_gray[y:y+h, x:x+w]], [0], None, [180], [0,180])
            
            
            
            roi_hist = cv2.calcHist([roi_gray], [0], None, [180], [0,180])
         
            
            
            simval = cv2.compareHist(hist_snap, roi_hist, cv2.HISTCMP_CORREL)
            print simval

            if simval < 90.0:
                detect_face = False
                hist_snap = roi_hist
                np.delete(trackzone, 0, 0)
                continue
            ####################### hist_snap 복수 얼굴 인식 시 문제 해결 요망




            ################ Optical Flow

            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 1, max(w,h), 3, 5, 1.2, 0)

            for j in range(h):
                for i in range(w):
                    [ny, nx] = [y, x] + flow[y:y+h, x:x+w][j,i]
            print nx, ny
            nx = int(round(nx))
            ny = int(round(ny))

            
            cv2.rectangle(frame, (nx, ny), (nx+w, ny+h), (0,0,255), 2)







            ################ Trackzone Initialization
            np.append(trackzone, [nx, ny, w, h])
            np.delete(trackzone, 0, 0)



        res = np.vstack((prev_frame, frame))
        cv2.imshow('frame', res)




        
        prev_gray = gray.copy()



    else:
        break
    
    k = cv2.waitKey(25) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
