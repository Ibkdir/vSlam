import open3d as o3d
import time
import numpy as np

color_dict = {'black': [0, 0, 0], 'white': [1,1,1]}

class Display:
    def __init__(self, bgcolor='black'):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.get_render_option().background_color = color_dict[bgcolor]
    
    def runTest(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector([])

        try:
            while True:
                new_points = np.random.rand(1000, 3)
                pcd.points = o3d.utility.Vector3dVector(new_points)
                self.vis.clear_geometries()
                self.vis.add_geometry(pcd)
                
                if not self.vis.poll_events():
                    break
                
                self.vis.update_renderer()
                time.sleep(0.1)
        finally:
            self.vis.destroy_window()