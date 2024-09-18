import os
from jax import jit
import jax.numpy as jnp
import numpy as np
import cv2 as cv
from display import Display

# General Pipeline: 
# Detect Features -> FLANN -> RANSAC -> Estimate Pose -> Triangulation for Map construction -> Bundle adjustment for Graph optimization

class vSLAM:
    def __init__(self, vidname:str ='highway', fd:str='sift') -> None:
        self.vpath:str = f'./videos/{vidname}.mp4'
        if not os.path.exists(self.vpath):
            raise FileNotFoundError(f"Video file not found at path: {self.vpath}")
        
        self.fd, self.matchernorm = self.create_feature_detector(fd.lower())
        self.matcher = cv.BFMatcher(self.matchernorm, crossCheck=True) if self.matchernorm == cv.NORM_HAMMING else cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    def create_feature_detector(self, chosen_fd):
        '''Choose the feature detector for your usecase'''
        if chosen_fd == 'orb':
            return cv.ORB.create(), cv.NORM_HAMMING
        elif chosen_fd == 'sift':
            return cv.SIFT.create(), cv.NORM_L2
        elif chosen_fd == 'brisk':
            return cv.BRISK.create(), cv.NORM_HAMMING
        else:
            raise ValueError(f"❗ Unsupported feature detector ❗")
        
    '''Feature Detection'''
    def detect_features(self, frame):
        gray= cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        kp, descriptors = self.fd.detectAndCompute(gray, None)
        points = np.array([point.pt for point in kp], dtype=np.float32)
        frame = cv.drawKeypoints(frame,kp,frame)
        return frame, points, descriptors

    '''FLANN'''
    def match_features(self, descriptors1, descriptors2):
        if descriptors1 is None or descriptors2 is None:
            raise ValueError("One or both descriptor sets are None.")

        if isinstance(self.matcher, cv.BFMatcher) and self.matchernorm == cv.NORM_HAMMING:
            return self.matcher.match(descriptors1, descriptors2)
        else:
            knn_matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
            return [m for m, n in knn_matches if m.distance < 0.7 * n.distance] # Lowe's Ratio Test
    
    '''RANSAC'''
    @staticmethod
    def compute_fundamental_matrix(pts1, pts2):
        F, mask = cv.findFundamentalMat(jnp.array(pts1), jnp.array(pts2), method=cv.FM_RANSAC)
        return jnp.array(F), jnp.array(mask)
    
    @staticmethod
    @jit
    def select_inliers(pts1:jnp.ndarray, pts2:jnp.ndarray, mask:jnp.ndarray):
        inl1 = pts1[mask.ravel() == 1]
        inl2 = pts2[mask.ravel() == 1]
        return inl1, inl2
    
    '''Main Process'''
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
    
    def triangulate_points(self):
        pass
    
    def bundle_adjustment(self):
        pass

if __name__ == '__main__':
    vSLAM(vidname='city', fd='sift').processVideo()