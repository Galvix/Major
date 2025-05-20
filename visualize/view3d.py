# visualize/view3d.py
import open3d as o3d
import numpy as np

def live_view(points_array):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pc = o3d.geometry.PointCloud()
    vis.add_geometry(pc)

    while True:
        if len(points_array)==0: continue
        pc.points = o3d.utility.Vector3dVector(points_array)
        pc.paint_uniform_color([1,0,0])
        vis.update_geometry(pc)
        vis.poll_events(); vis.update_renderer()

if __name__=="__main__":
    # example: generate random cloud
    pts = np.random.randn(200,3)
    live_view(pts)
