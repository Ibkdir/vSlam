import open3d as o3d
import time
import numpy as np

color_dict = {'black': [0, 0, 0], 'white': [1,1,1]}

class Display:
    def __init__(self, bgcolor='black'):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.get_render_option().background_color = color_dict[bgcolor]
    
    def run(self, keypoints):
        pass