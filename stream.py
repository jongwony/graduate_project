# -*- coding: UTF-8 -*-
import cv2
import numpy as np


class VideoStream(object):
    def __init__(self, filename):
        
        self.video=cv2.VideoCapture(filename)
        self.face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        
        # snapshot
        self.pre_gr = np.zeros
        self.pre_hsv = np.zeros
        self.trackzone = tuple()

        # detection flag
        self.detectface = False

        # prev frame
        ret, self.pre_fr = self.video.read()
        if ret:
            if self.pre_fr.shape[0] > 800 or self.pre_fr.shape[1] > 800:
                self.pre_fr = cv2.resize(self.pre_fr, (0,0), fx=0.5, fy=0.5)
            self.pre_gr = cv2.cvtColor(self.pre_fr, cv2.COLOR_BGR2GRAY)
            self.pre_hsv = cv2.cvtColor(self.pre_fr, cv2.COLOR_BGR2HSV)
        else:
            print "Video File %s Not Found" % filename

        # query image
        self.queryimg = None
 	self.roi_query = None	
	self.query_hsv_hist = None




    def __del__(self):
        self.video.release()



    def opticalFlow(self, prev, curr, (x, y, w, h)):
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 1, max(w, h), 3, 5, 1.2, 0)

        # out of range
        try:
            for j in range(h):
                for i in range(w):
                    [ny, nx] = [y, x] + (flow[y:y+h, x:x+w][j,i]/w*h)
        except IndexError as e:
            return x, y


        return (int(round(nx)), int(round(ny)))







    def get_frame(self):

        # video frame read
        ret, fr = self.video.read()

        if ret:
            if fr.shape[0] > 800 or fr.shape[1] > 800:
                fr = cv2.resize(fr, (0,0), fx=0.5, fy=0.5)
            # fps calculate
            # sometimes video frame = 0 zero division error occur
            fps = self.video.get(cv2.CAP_PROP_FPS) + 5

            rate = int(round(1000 /fps))

            ######### opencv coding  #########
            gr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)


	    if self.queryimg is not None:
		queryhsv = cv2.cvtColor(self.queryimg, cv2.COLOR_BGR2HSV)
		querygr = cv2.cvtColor(self.queryimg, cv2.COLOR_BGR2GRAY)
	        queryfaces = self.face_cascade.detectMultiScale(querygr, 1.3, 5)		
		for x, y, w, h in queryfaces:
		    self.roi_query = queryhsv[y:y+h, x:x+w]
		
		self.query_hsv_hist = cv2.calcHist([self.roi_query], [0], None, [256], [0,256])

		
		

            faces = self.face_cascade.detectMultiScale(gr, 1.3, 5)
            


            # trackzone refresh
            if faces == tuple():
                self.detectface = False
            else:
                if self.trackzone == tuple():
                    self.trackzone = faces

                if self.trackzone.shape[0] == 0:
                    self.trackzone = faces
                elif self.trackzone.shape[0] > 0:
                    self.trackzone = np.vstack((faces, self.trackzone))



            
            # face detection
            for x, y, w, h in faces:
                detectface = True
                cv2.rectangle(fr, (x, y), (x+w, y+h), (255,0,0), 2)
            
            
            # remove neighborhood
            i = 0
            for x, y, w, h in self.trackzone:
                j = 0
                if self.trackzone.shape[0] == 0:
                    break
                else:
                    i = i % self.trackzone.shape[0]
                for nx, ny, nw, nh in self.trackzone:
                    j = j % self.trackzone.shape[0]
                    if j == i:
                        j = j + 1
                        continue
                    if ((np.abs(nx-x) < int(w/3)) and (np.abs(ny-y) < int(h/3))):
                        self.trackzone = np.delete(self.trackzone, j, 0)

                    j = j + 1
                i = i + 1

            # face tracking
            i = 0
            for x, y, w, h in self.trackzone:
                i = i % self.trackzone.shape[0]



                # Optical Flow
                nx, ny = self.opticalFlow(self.pre_gr, gr, (x, y, w, h))




                cv2.rectangle(fr, (nx, ny), (nx+w, ny+h), (0,0,255), 1)

                self.trackzone = np.vstack((np.array([nx, ny, w, h], dtype=np.int32), self.trackzone))
                self.trackzone = np.delete(self.trackzone, i+1, 0)

		

                # Histogram
                pre_hsv_hist = cv2.calcHist([self.pre_hsv[ny:ny+h, nx:nx+w]], [0], None, [256], [0,256])
                roi_hsv_hist = cv2.calcHist([hsv[y:y+h, x:x+w]], [0], None, [256], [0,256])

                histval = cv2.compareHist(pre_hsv_hist, roi_hsv_hist, cv2.HISTCMP_CORREL)
                if histval < 0.80:
                    self.trackzone = np.delete(self.trackzone, i, 0)
		
                print histval

                if self.roi_query is not None:
                    query_histval = cv2.compareHist(self.query_hsv_hist, roi_hsv_hist, cv2.HISTCMP_CORREL)
                    if query_histval < 0.80:
                        self.trackzone = np.delete(self.trackzone, i, 0)
		    print query_histval
		
		    
                i = i + 1


            # snapshot
            self.pre_gr = gr.copy()
            self.pre_hsv = hsv.copy()

            ##################################
        
            cv2.waitKey(rate)
        

            ret, jpeg = cv2.imencode('.jpg', fr)
            return jpeg.tobytes()
        else:
            return ret
