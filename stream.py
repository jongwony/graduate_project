# -*- coding: UTF-8 -*-
import cv2
import numpy as np


class VideoStream(object):
    def __init__(self, filename):
        
        self.video=cv2.VideoCapture(filename)
        self.face_cascade = cv2.CascadeClassifier('haarcascades_cuda/haarcascade_frontalface_default.xml')
        
        # snapshot
        self.pre_gr = np.zeros
        self.pre_hsv = np.zeros
        self.beforefaces = tuple()
        self.track_db = dict()

        # flag
        self.detectface = False
        self.trackface = 0
        self._label = 0

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
            fps = self.video.get(cv2.CAP_PROP_FPS) + 3
            rate = int(round(1000 /fps))

            ######### opencv coding  #########
            gr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)


	    if self.queryimg is not None:
		queryhsv = cv2.cvtColor(self.queryimg, cv2.COLOR_BGR2HSV)
		querygr = cv2.cvtColor(self.queryimg, cv2.COLOR_BGR2GRAY)
	        queryfaces = self.face_cascade.detectMultiScale(querygr, scaleFactor=1.08, minNeighbors=5)		
		for x, y, w, h in queryfaces:
		    self.roi_query = queryhsv[y:y+h, x:x+w]
		
		self.query_hsv_hist = cv2.calcHist([self.roi_query], [0], None, [256], [0,256])

		
		

            faces = self.face_cascade.detectMultiScale(gr, scaleFactor=1.08, minNeighbors=5)
            

            # face detection
            for x, y, w, h in faces:
                cv2.rectangle(fr, (x, y), (x+w, y+h), (255,0,0), 2)
                
                self.detectface = True


            for x1, y1, w1, h1 in faces:
                for x2, y2, w2, h2 in self.beforefaces:
                    if(np.abs(x1-x2)<w1/2 and np.abs(w1-w2)<15) or (np.abs(y1-y2)<h1/2 and np.abs(h1-h2)<15):
                        self.track_db[self._label] = list((x1, y1, w1, h1))
                        self._label += 1

            for k1, (x1, y1, w1, h1) in self.track_db.items():
                for k2, (x2, y2, w2, h2) in self.track_db.items():
                    if k1 >= k2:
                        continue
                    if(np.abs(x1-x2)<w1/2 and np.abs(w1-w2)<15) or (np.abs(y1-y2)<h1/2 and np.abs(h1-h2)<15):
                        self.track_db[k1] = self.track_db.pop(k2)
                        self._label -= 1
                    elif np.abs(x1-x2)<np.abs(w1-w2) or np.abs(y1-y2)<np.abs(h1-h2):
                        if x1>x2 and y1>y2:
                            self.track_db[k1] = self.track_db.pop(k2)
                            self._label -= 1
            

            self.trackface += 1
            self.trackface %= 2
            if self.trackface == 0:
                self.beforefaces = faces

                        
            for k, (x, y, w, h) in self.track_db.items():
                nx, ny = self.opticalFlow(self.pre_gr, gr, (x, y, w, h))
                cv2.rectangle(fr, (nx, ny), (nx+w, ny+h), (0,0,255), 1)
                self.track_db[k] = list((nx, ny, w, h))

                pre_hsv_hist = cv2.calcHist([self.pre_hsv[ny:ny+h, nx:nx+w]], [0], None, [256], [0,256])
                roi_hsv_hist = cv2.calcHist([hsv[y:y+h, x:x+w]], [0], None, [256], [0,256])

                histval = cv2.compareHist(pre_hsv_hist, roi_hsv_hist, cv2.HISTCMP_CORREL)
                if histval < 0.80:
                    self.track_db.pop(k)
                    self._label -= 1
		

                if self.roi_query is not None:
                    query_histval = cv2.compareHist(self.query_hsv_hist, roi_hsv_hist, cv2.HISTCMP_CORREL)
                    if query_histval < 0.80:
                        self.track_db.pop(k)
                        self._label -= 1
		    print query_histval
		


            # snapshot
            self.pre_gr = gr.copy()
            self.pre_hsv = hsv.copy()

            ##################################
        
            cv2.waitKey(rate)
        

            ret, jpeg = cv2.imencode('.jpg', fr)
            return jpeg.tobytes()
        else:
            return ret
