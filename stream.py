# -*- coding: UTF-8 -*-
import cv2

class VideoStream(object):
    def __init__(self, filename):
        self.video=cv2.VideoCapture(filename)
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        # resolution modify
        #self.video.set(3, 320)
        #self.video.set(4, 240)
        
        # video frame read
        success, frame = self.video.read()

        if success:
            # fps calculate
            # sometimes video frame = 0 zero division error occur
            fps = self.video.get(cv2.CAP_PROP_FPS) + 5
            if int(fps)==0:
                print "Error! fps = 0"
                fps = 25

            rate = int(round(1000 /fps))

            ######### opencv coding  #########
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for(x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
            ##################################
        
            cv2.waitKey(rate)
        

            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        else:
            return success
