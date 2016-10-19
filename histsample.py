# -*- coding: UTF-8 -*-

import numpy as np
import cv2

cap = cv2.VideoCapture('/var/www/flask/static/uploads/mov_bbb.mp4')

ret, old_frame = cap.read()
old_hsv = cv2.cvtColor(old_frame, cv2.COLOR_BGR2HSV)

face = [(120, 80, 80, 60)]
mask = np.zeros_like(old_hsv[...,0])
for x, y, w, h in face:
    mask[y:y+h, x:x+w] = 255


while(1):
    masked = cv2.bitwise_and(old_hsv[...,0], old_hsv[...,0], mask=mask)
    og_hist = cv2.calcHist([old_hsv], [0], masked, [256], [0,256])
   
    ret, frame = cap.read()
    
    if ret:
        new_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hlist = list()

        for x, y, w, h in face:
            for j in range(frame.shape[0]-h):
                if j % 2 != 0:
                    continue
                for i in range(frame.shape[1]-w):
                    if i % 2 != 0:
                        continue
                    ng_hist = cv2.calcHist([new_hsv[j:j+h,i:i+w,0]], [0], None, [256], [0,256])
                    part_hist = cv2.compareHist(og_hist, ng_hist, cv2.HISTCMP_CORREL)
                    hlist.append((part_hist,[i,j]))
        #max(hlist)
        #print hlist.index(max(hlist))
        #print max(hlist)
        #print max(hlist)[1]

        xy = max(hlist)[1]




        for (x, y, w, h) in face:
            print (x, y)
            cv2.rectangle(old_frame, (x, y), (x+w, y+h), (0,0,255), 2)
            cv2.rectangle(frame, (xy[0], xy[1]), (xy[0]+w, xy[1]+h), (0,255,0),2)
        res = np.vstack((old_frame, frame))
        cv2.imshow('frame', frame)

        
        
        old_hsv = new_hsv.copy()
        for (x, y, w, h) in face:
            face.pop(0)
            face.append((xy[0], xy[1], w, h))



    else:
        break
    

    k = cv2.waitKey(25)
    if k==27:
        break

cv2.destroyAllWindows()
cap.release()
