import open3d as o3d
import os
import numpy as np
import time
import threading
import cv2
from PIL import Image

def main():
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = ROOT_DIR + "\\data\\final_project_2023_rect"
    SEQ_01 = DATA_DIR + "\\seq_01"
    SEQ_02 = DATA_DIR + "\\seq_01"
    SEQ_03 = DATA_DIR + "\\seq_01"
    

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    view_ctl = vis.get_view_control()


    seq = SEQ_01
    labels = seq + "\\labels.txt"
    frame = 0 # starting frame 0


    # FIXED THE CAMERA
    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_final.json")
    ctr.convert_from_pinhole_camera_parameters(parameters)


    try:
        with open(labels, "r") as f:
            objects = []
            line = f.readline()
            while(line):
                elements = line.split(" ")
                curr = int(elements[0])

                if(curr == frame):
                    # get elements
                    track_id = elements[1]
                    type = elements[2]
                    truncated = elements[3]
                    occluded = elements[4]
                    alpha = elements[5]
                    bbox = elements[6:10]
                    dim = elements[10:13]
                    location = elements[13:16]
                    rotation_y = elements[16]
                    #core = elements[17]

                    # 3D bounding box
                    center = np.array([float(location[0]),float(location[1]), float(location[2])])     # center point of the box
                    # Create a mesh box with the specified dimensions
                    mesh_box = o3d.geometry.TriangleMesh.create_box(width=float(dim[1]), height=float(dim[0]), depth=float(dim[2]))
                    # Translate the mesh box to the specified location
                    
                    mesh_box.translate(center)
                    line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_box)


                    line_set.paint_uniform_color([0, 0, 0]) 
                    objects.append(line_set)

                    # for each line create a bounding box and add it to the list
                else:
                    ctr.convert_from_pinhole_camera_parameters(parameters)
   
                    for o in objects:
                        vis.add_geometry(o)
                    
                    # update the view
                    for i in range(100):
                        vis.update_renderer()
                        vis.poll_events()
                        time.sleep(0.01)

                    for o in objects:
                        vis.remove_geometry(o)
                    
                    objects.clear()
                    frame = frame + 1

                line = f.readline() # read the next line
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down")
        vis.destroy_window()
        exit()

# Define function for event loop
def vis_loop(vis):
    while True:
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)


if __name__ == "__main__":
    main()