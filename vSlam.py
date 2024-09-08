# Trying the experimental Jax-Metal 😎
import jax
import jax.numpy as jnp
import numpy as np
import cv2 as cv
import time
from display import Display

class vSLAM:
    def __init__(self, vidname:str ='highway', fd:str='sift') -> None:
        self.vpath:str = f'./videos/{vidname}.mp4'
        self.fd = self.create_feature_detector(fd.lower())
    
    def create_feature_detector(self, chosen_fd):
        if chosen_fd == 'orb':
            return cv.ORB.create()
        elif chosen_fd == 'sift':
            return cv.SIFT.create()
        else:
            raise ValueError(f"❗ Unsupported feature detector ❗")
    
    def detect_features(self, frame):
        gray= cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        kp = self.fd.detect(gray, None)
        frame=cv.drawKeypoints(gray,kp,frame)
        return frame

    def processVideo(self) -> None:
        vc = cv.VideoCapture(self.vpath)
        if not vc.isOpened():
            print('Error Opening File')
            
        while vc.isOpened():
            _, frame = vc.read()
            time.sleep(0.1)
            cv.imshow('Frame', self.detect_features(frame))
            key = cv.waitKey(100)
            if key == ord('q'):
                break
        vc.release()
        cv.destroyAllWindows()

    def testOpen3D(self):
        window = Display()
        window.runTest()

if __name__ == '__main__':
    vSLAM(vidname='driving').processVideo()