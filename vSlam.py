import os
import sys

# Trying the experimental Jax-Metal 😎
import jax
import jax.numpy as jnp

import open3d as o3d
import numpy as np
import cv2 as cv
import time
from display import Display

class vSLAM:
    def __init__(self, vpath='./videos/main.mp4') -> None:
        self.vpath = vpath
        
    def processFrame(self) -> None:
        vc = cv.VideoCapture(self.vpath)
        if not vc.isOpened():
            print('Error Opening File')
        while vc.isOpened():
            _, frame = vc.read()
            time.sleep(0.1)
            cv.imshow('Frame', frame)
            
            key = cv.waitKey(100)
            if key == ord('q'):
                break
        vc.release()
        cv.destroyAllWindows()

    def testOpen3D(self):
        window = Display()
        window.runTest()


if __name__ == '__main__':
    vSLAM().testOpen3D()