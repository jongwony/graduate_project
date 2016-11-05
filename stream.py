# -*- coding: UTF-8 -*-
import cv2
import numpy as np

class VideoStream(object):
    def __init__(self, filename):
        self.video=cv2.VideoCapture(filename)
        self.face_cascade = cv2.CascadeClassifier('haarcascades_cuda/haarcascade_frontalface_default.xml')
        
        # snapshot
        self.trackzone = tuple()
        self.pre_gr = np.zeros
        self.pre_hsv = np.zeros

        # detection flag
        self.detectface = False

        # prev frame
        ret, self.pre_fr = self.video.read()
        if ret:
            self.pre_gr = cv2.cvtColor(self.pre_fr, cv2.COLOR_BGR2GRAY)
            self.pre_hsv = cv2.cvtColor(self.pre_fr, cv2.COLOR_BGR2HSV)
        else:
            print "Video File %s Not Found" % filename

        # query image
        self.queryimg = None




    def __del__(self):
        self.video.release()




    def rm_neighborhood(self, track):
        i = 0

        for x, y, w, h in track:
            j = 0

            if track.shape[0] == 0:
                break
            else:
                i = i % track.shape[0]


            for nx, ny, nw, nh in track:
                j = j % track.shape[0]

                if j == i:
                    j = j + 1
                    continue

                if ((np.abs(nx-x) < int(nw/3)) and (np.abs(ny-y) < int(nh/3))):
                    track = np.delete(track, j, 0)

                j = j + 1
            i = i + 1





    def opticalFlow(self, prev, curr, (x, y, w, h)):
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 1, max(w, h), 3, 5, 1.2, 0)

        for j in range(h):
            for i in range(w):
                [ny, nx] = [y, x] + (flow[y:y+h, x:x+w][j,i]/w*h)

        return (int(round(nx)), int(round(ny)))







    def get_frame(self):
        # video frame read
        ret, fr = self.video.read()

        if ret:
            # fps calculate
            # sometimes video frame = 0 zero division error occur
            fps = self.video.get(cv2.CAP_PROP_FPS) + 5
            if int(fps)==0:
                print "Error! fps = 0"
                fps = 25

            rate = int(round(1000 /fps))

            ######### opencv coding  #########
            gr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)


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
            for(x, y, w, h) in faces:
                detectface = True
                cv2.rectangle(fr, (x, y), (x+w, y+h), (255,0,0), 2)
            
            
            
            self.rm_neighborhood(self.trackzone)


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
                if histval < 80:
                    self.trackzone = np.delete(self.trackzone, i, 0)



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
