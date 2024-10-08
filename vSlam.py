import os
import numpy as np
import cv2 as cv
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from display import Display

class vSLAM:
    def __init__(self, vidname: str = 'highway', fd: str = 'sift') -> None:
        self.vpath: str = f'./videos/{vidname}.mp4'
        if not os.path.exists(self.vpath):
            raise FileNotFoundError(f"Video file not found at path: {self.vpath}")

        self.fd, self.matchernorm = self.create_feature_detector(fd.lower())
        self.matcher = cv.BFMatcher(self.matchernorm, crossCheck=True) if self.matchernorm == cv.NORM_HAMMING else cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

        vc = cv.VideoCapture(self.vpath)
        ret, frame = vc.read()
        if not ret:
            raise ValueError("Unable to read video for intrinsic estimation.")
        vc.release()

        image_width = frame.shape[1]
        image_height = frame.shape[0]
        cx = image_width / 2
        cy = image_height / 2
        HFOV = 60
        fx = fy = (image_width) / (2 * np.tan(np.deg2rad(HFOV / 2)))
        self.cameraMat = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        self.distCoeffs = np.zeros((4, 1))

    def create_feature_detector(self, chosen_fd):
        '''Choose the feature detector for your usecase'''
        if chosen_fd == 'orb':
            return cv.ORB.create(), cv.NORM_HAMMING
        elif chosen_fd == 'sift':
            return cv.SIFT_create(), cv.NORM_L2
        elif chosen_fd == 'brisk':
            return cv.BRISK_create(), cv.NORM_HAMMING
        else:
            raise ValueError(f"❗ Unsupported feature detector ❗")

    '''Feature Detection'''
    def detect_features(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        kp, descriptors = self.fd.detectAndCompute(gray, None)
        points = np.array([point.pt for point in kp], dtype=np.float32)

        kp_pts = np.array([kp.pt for kp in kp], dtype=np.float32)
        h, w = frame.shape[:2]
        x = np.clip(kp_pts[:, 0], 0, w - 1).astype(int)
        y = np.clip(kp_pts[:, 1], 0, h - 1).astype(int)
        colors = frame[y, x]

        feature_frame = cv.drawKeypoints(frame, kp, None)
        return feature_frame, points, descriptors, colors

    '''FLANN'''
    def match_features(self, descriptors1, descriptors2):
        if descriptors1 is None or descriptors2 is None:
            raise ValueError("One or both descriptor sets are None.")

        if isinstance(self.matcher, cv.BFMatcher) and self.matchernorm == cv.NORM_HAMMING:
            return self.matcher.match(descriptors1, descriptors2)
        else:
            knn_matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
            return [m for m, n in knn_matches if m.distance < 0.7 * n.distance]  # lowe's ratio test

    '''RANSAC'''
    @staticmethod
    def compute_fundamental_matrix(pts1, pts2):
        F, mask = cv.findFundamentalMat(np.array(pts1), np.array(pts2), method=cv.FM_RANSAC)
        return np.array(F), np.array(mask)

    @staticmethod
    def select_inliers(pts1: np.ndarray, pts2: np.ndarray, mask: np.ndarray):
        inl1 = pts1[mask.ravel() == 1]
        inl2 = pts2[mask.ravel() == 1]
        return inl1, inl2

    '''Estimating Pose'''
    def estimate_pose(self, framepoints1, framepoints2):
        E, mask = cv.findEssentialMat(framepoints1, framepoints2, self.cameraMat, method=cv.RANSAC, threshold=1, prob=0.999)
        if E is None:
            raise ValueError("Essential matrix computation failed.")
        _, R, t, mask_pose = cv.recoverPose(E, framepoints1, framepoints2, cameraMatrix=self.cameraMat, mask=mask)
        return R, t, mask_pose

    '''Triangulation'''

    def reject_outliers(self, points, m=2.0):
        median = np.median(points, axis=0)
        diffs = np.sqrt(np.sum((points - median)**2, axis=1))
        med_diff = np.median(diffs)
        mad = np.median(np.abs(diffs - med_diff))
        if mad == 0: mad = 1e-6
        modified_z_scores = 0.6745 * (diffs - med_diff) / mad
        mask = np.abs(modified_z_scores) < m
        return points[mask], mask

    def triangulate_points(self, R, t, inl1, inl2, colors1):
        P1 = self.cameraMat @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.cameraMat @ np.hstack((R, t))
        points4D_hom = cv.triangulatePoints(P1, P2, inl1.T, inl2.T)

        points3D = points4D_hom[:3] / points4D_hom[3]
        points3D = points3D.T
        valid_idx = points3D[:, 2] > 0
        points3D = points3D[valid_idx]
        inl1 = inl1[valid_idx]
        inl2 = inl2[valid_idx]
        colors = colors1[valid_idx]
        points3D, mask = self.reject_outliers(points3D, m=2.0)
        inl1 = inl1[mask]
        inl2 = inl2[mask]
        colors = colors[mask]

        return points3D, inl1, inl2, colors

    '''Bundle Adjustment'''
    def bundle_adjustment(self, poses, points3D, observations):
        n_cameras = len(poses)
        n_points = len(points3D)
        n_observations = len(observations)

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        for i, obs in enumerate(observations):
            camera_indices[i], point_indices[i], points_2d[i] = obs

        camera_params = np.array([np.hstack((cv.Rodrigues(R)[0].ravel(), t.ravel())) for R, t in poses])
        points_3d = np.array(points3D)

        x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
        f0 = self.reprojection_error(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)

        A = self.bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

        res = least_squares(self.reprojection_error, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                            args=(n_cameras, n_points, camera_indices, point_indices, points_2d))

        x = res.x
        optimized_camera_params = x[:n_cameras * 6].reshape((n_cameras, 6))
        optimized_points_3d = x[n_cameras * 6:].reshape((n_points, 3))

        optimized_poses = [(cv.Rodrigues(params[:3])[0], params[3:6].reshape(3, 1)) for params in optimized_camera_params]

        return optimized_poses, optimized_points_3d.tolist()

    def reprojection_error(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))

        projected = np.empty((len(camera_indices), 2))
        for i, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):
            rvec, tvec = camera_params[cam_idx, :3], camera_params[cam_idx, 3:6]
            pt3d = points_3d[pt_idx]
            proj_pt, _ = cv.projectPoints(pt3d.reshape(1, 1, 3), rvec, tvec, self.cameraMat, None)
            projected[i] = proj_pt.ravel()

        return (projected - points_2d).ravel()

    def bundle_adjustment_sparsity(self, n_cameras, n_points, camera_indices, point_indices):
        m = camera_indices.size * 2
        n = n_cameras * 6 + n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(6):
            A[2 * i, camera_indices * 6 + s] = 1
            A[2 * i + 1, camera_indices * 6 + s] = 1

        for s in range(3):
            A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

        return A

    '''Main Process'''
    def processVideo(self) -> None:
        vc = cv.VideoCapture(self.vpath)
        if not vc.isOpened():
            print('Error Opening File')
            return
        
        ret, prev_frame = vc.read()
        if not ret:
            print('Error reading the first frame.')
            return

        prev_featureframe, prev_points, prev_descriptors, prev_colors = self.detect_features(prev_frame)
        pose = (np.eye(3), np.zeros((3, 1)))
        poses = [pose]
        points3D = []
        point_colors = []
        observations = []

        frame_idx = 1
        ThreeDisplay = Display()

        while vc.isOpened():
            ret, frame = vc.read()
            if not ret:
                break

            featureframe, curr_points, curr_descriptors, curr_colors = self.detect_features(frame)

            matches = self.match_features(prev_descriptors, curr_descriptors)
            if len(matches) < 8:
                print('Not enough matches.')
                prev_frame = frame.copy()
                prev_points = curr_points
                prev_descriptors = curr_descriptors
                prev_colors = curr_colors
                continue

            src_pts = np.float32([prev_points[m.queryIdx] for m in matches])
            dst_pts = np.float32([curr_points[m.trainIdx] for m in matches])
            src_colors = np.array([prev_colors[m.queryIdx] for m in matches])

            try:
                R, t, mask_pose = self.estimate_pose(src_pts, dst_pts)
            except Exception as e:
                print(f'Pose estimation failed: {e}')
                prev_frame = frame.copy()
                prev_points = curr_points
                prev_descriptors = curr_descriptors
                prev_colors = curr_colors
                continue

            inl1, inl2 = self.select_inliers(src_pts, dst_pts, mask_pose)
            inl1_colors = src_colors[mask_pose.ravel() == 1]
            new_points3D, inl1, inl2, new_colors = self.triangulate_points(R, t, inl1, inl2, inl1_colors)

            if new_points3D.shape[0] == 0:
                print('No valid 3D points.')
                prev_frame = frame.copy()
                prev_points = curr_points
                prev_descriptors = curr_descriptors
                prev_colors = curr_colors
                continue

            idx_offset = len(points3D)
            points3D.extend(new_points3D)
            point_colors.extend(new_colors)

            pose = (R, t)
            poses.append(pose)

            for idx in range(new_points3D.shape[0]):
                observations.append((frame_idx - 1, idx_offset + idx, inl1[idx]))
                observations.append((frame_idx, idx_offset + idx, inl2[idx]))

            if frame_idx % 5 == 0:
                poses, points3D = self.bundle_adjustment(poses, points3D, observations)

            if frame_idx % 2 == 0:
                ThreeDisplay.update(np.array(points3D), np.array(point_colors), poses)

            frame_idx += 1
            prev_frame = frame.copy()
            prev_points = curr_points
            prev_descriptors = curr_descriptors
            prev_colors = curr_colors

            key = cv.waitKey(1)
            if key == ord('q'):
                break

        vc.release()
        cv.destroyAllWindows()
        ThreeDisplay.close()

if __name__ == '__main__':
    vSLAM(vidname='city', fd='sift').processVideo()