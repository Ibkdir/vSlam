import open3d as o3d
import numpy as np

color_dict = {'black': [0, 0, 0], 'white': [1, 1, 1]}

class Display:
    def __init__(self, bgcolor='black'):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.get_render_option().background_color = np.array(color_dict[bgcolor])
        self.point_cloud = o3d.geometry.PointCloud()
        self.camera_frames = []
        self.vis.add_geometry(self.point_cloud)
        self.points3D = np.array([])
        self.colors = np.array([])
        self.poses = []

    def update(self, points3D, colors, poses):
        self.points3D = points3D
        self.colors = colors
        self.poses = poses
        self.run()

        if len(self.points3D) > 0:
            max_points_to_visualize = 5000
            points_to_visualize = self.points3D[-max_points_to_visualize:]
            colors_to_visualize = self.colors[-max_points_to_visualize:]

            self.point_cloud.points = o3d.utility.Vector3dVector(points_to_visualize)
            colors_to_visualize = colors_to_visualize / 255.0
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors_to_visualize)
            self.vis.update_geometry(self.point_cloud)

    def run(self):
        for cam in self.camera_frames:
            self.vis.remove_geometry(cam)
        self.camera_frames = []

        max_points_to_visualize = 5000
        if len(self.points3D) > 0:
            if len(self.points3D) > max_points_to_visualize:
                points_to_visualize = self.points3D[-max_points_to_visualize:]
                colors_to_visualize = self.colors[-max_points_to_visualize:]
            else:
                points_to_visualize = self.points3D
                colors_to_visualize = self.colors

            self.point_cloud.points = o3d.utility.Vector3dVector(points_to_visualize)
            colors_to_visualize = colors_to_visualize / 255.0
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors_to_visualize)
            self.vis.update_geometry(self.point_cloud)

        max_poses_to_visualize = 10
        poses_to_visualize = self.poses[-max_poses_to_visualize:] if len(self.poses) > max_poses_to_visualize else self.poses
        for R, t in poses_to_visualize:
            camera_mesh = self.create_camera_mesh(R, t)
            self.vis.add_geometry(camera_mesh)
            self.camera_frames.append(camera_mesh)

        self.vis.poll_events()
        self.vis.update_renderer()

    def create_camera_mesh(self, R, t):
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R
        camera_pose[:3, 3] = (-R.T @ t).flatten()
        camera_frame.transform(camera_pose)
        return camera_frame

    def close(self):
        self.vis.destroy_window()