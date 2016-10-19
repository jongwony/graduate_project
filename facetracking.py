# -*- coding: UTF-8 -*-

import numpy as np
import cv2
import pdb




face_cascade = cv2.CascadeClassifier('/var/www/flask/haarcascade_frontalface_default.xml')


filename = 'gf_na.mp4'


cap = cv2.VideoCapture(filename)





global detect_face
global trackzone
global prev_gray
global roi_hist
global itertrack


trackzone = np.array([],dtype=np.int32) 
roi_hist = np.zeros
detect_face = False 
itertrack = 0





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

        if faces == tuple():
            detect_face = False
        else:
            if trackzone.shape[0] == 0:
                trackzone = faces


        itertrack = 0

        for x, y, w, h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            detect_face = True
            
        
        for a, b in zip(trackzone, faces):
            if ((np.abs(a[0]-b[0]) > 15) or (np.abs(a[1]-b[1]) > 15)) :
                trackzone = np.vstack((trackzone, b))






        #################### Tracking

        itertrack = 0
        
        for x, y, w, h in trackzone:

            j = 0
            itertrack = (itertrack % trackzone.shape[0])



            for nx, ny, nw, nh in trackzone:
                if j == itertrack:
                    continue
                if ((np.abs(nx-x) < 15) and (np.abs(ny-y) < 15)):
                    trackzone = np.delete(trackzone, j, 0)

                j = j + 1


            itertrack = itertrack + 1





        print 'd = %d t = %s' % (detect_face, tuple(trackzone))



        
        itertrack = 0
        
        for x, y, w, h in trackzone:
            itertrack = (itertrack % trackzone.shape[0])
            
            
            
            #print itertrack




            ################ Optical Flow

            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 1, max(w,h), 3, 5, 1.2, 0)

            for j in range(h):
                for i in range(w):
                    [ny, nx] = [y, x] + flow[y:y+h, x:x+w][j,i]
                    if nx < 0:
                        nx = 0
                    if ny < 0:
                        ny = 0

            
            
            
            #print nx, ny
            
            
            
            
            nx = int(round(nx))
            ny = int(round(ny))

            
            cv2.rectangle(frame, (nx, ny), (nx+w, ny+h), (0,0,255), 2)

            # 순서 중요
            trackzone = np.vstack((trackzone, np.array([nx, ny, w, h], dtype=np.int32)))
            trackzone = np.delete(trackzone, itertrack, 0)


            ################ Histogram
            #pdb.set_trace()
        
            prev_hist = cv2.calcHist([prev_gray[ny:ny+h, nx:nx+w]], [0], None, [180], [0,180])
       
            roi_hist = cv2.calcHist([gray[y:y+h, x:x+w]], [0], None, [180], [0,180])
         


                ## if compareHist(calcHist([query]) , roi_hist) < 90:
                ##      continue
         
            
            simval = cv2.compareHist(prev_hist, roi_hist, cv2.HISTCMP_CORREL)
            
            
            
            print simval
            
            
            
            
            if simval < 0.75:
                trackzone = np.delete(trackzone, itertrack, 0)



            itertrack = itertrack + 1




        #res = np.vstack((frame))
        cv2.imshow('frame', frame)




        
        prev_gray = gray.copy()



    else:
        break
    
    k = cv2.waitKey(80) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
