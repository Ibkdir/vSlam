
import numpy as np
import cv2 as cv
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
        elif chosen_fd == 'brisk':
            return cv.BRISK.create()
        else:
            raise ValueError(f"❗ Unsupported feature detector ❗")
    
    def detect_features(self, frame):
        gray= cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        kp = self.fd.detect(gray, None) # This also needs to be changed because I need descriptors as well. Meaning I need to use sift.detectAndCompute
        points = np.array([point.pt for point in kp], dtype=np.float32)
        frame = cv.drawKeypoints(frame,kp,frame)
        return frame, points
    
    def match_features():
        # Use FLANN for fast approximate nearest neighbor search
        pass
    
    def estimate_motion():
        # Use RANSAC for the fundamental matrix
        pass

    def triangulate_points():
        # :Z
        pass
    
    def bundle_adjustment():
        # Oh god
        pass

    def processVideo(self) -> None:
        vc = cv.VideoCapture(self.vpath)
        if not vc.isOpened():
            print('Error Opening File')

        ThreeDisplay = Display()

        while vc.isOpened():
            _, frame = vc.read()
            featureframe, keypoints = self.detect_features(frame)

            ThreeDisplay.run(keypoints)
            
            key = cv.waitKey(100)
            if key == ord('q'):
                break
            
        vc.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    vSLAM(vidname='city', fd='sift').processVideo()