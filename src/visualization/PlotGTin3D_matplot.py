import os
import numpy as np
import time
import threading
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def main():
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = ROOT_DIR + "\\data\\final_project_2023_rect"
    SEQ_01 = DATA_DIR + "\\seq_01"
    SEQ_02 = DATA_DIR + "\\seq_01"
    SEQ_03 = DATA_DIR + "\\seq_01"

    seq = SEQ_01
    labels = seq + "\\labels.txt"
    frame = 0  # starting frame 0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    try:
        with open(labels, "r") as f:
            objects = []
            line = f.readline()
            while line:
                elements = line.split(" ")
                curr = int(elements[0])

                if curr == frame:
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
                    # core = elements[17]

                    # 3D bounding box
                    center = np.array(
                        [float(location[0]), float(location[1]), float(location[2])]
                    )  # center point of the box
                    center_x = float(location[0])
                    center_y = float(location[1])
                    center_z = float(location[2])
                    # for each line create a bounding box and add it to the list
                    width = float(dim[1])
                    length = float(dim[0])
                    depth = float(dim[2])

                    # calculate the coordinates of the eight vertices of the box
                    x = [
                        center_x - width / 2,
                        center_x - width / 2,
                        center_x + width / 2,
                        center_x + width / 2,
                        center_x - width / 2,
                        center_x - width / 2,
                        center_x + width / 2,
                        center_x + width / 2,
                    ]

                    y = [
                        center_y - length / 2,
                        center_y + length / 2,
                        center_y + length / 2,
                        center_y - length / 2,
                        center_y - length / 2,
                        center_y + length / 2,
                        center_y + length / 2,
                        center_y - length / 2,
                    ]

                    z = [
                        center_z - depth / 2,
                        center_z - depth / 2,
                        center_z - depth / 2,
                        center_z - depth / 2,
                        center_z + depth / 2,
                        center_z + depth / 2,
                        center_z + depth / 2,
                        center_z + depth / 2,
                    ]

                    # plot the box
                    ax.plot_trisurf(
                        x, y, z, linewidth=0.2, edgecolor="black", alpha=0.5
                    )
                else:
                    # set the axis limits
                    ax.set_xlim(-5, 5)
                    ax.set_ylim(-5, 5)
                    ax.set_zlim(-5, 10)

                    # set the axis labels
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.set_zlabel("Z")

                    # zoom out
                    ax.set_box_aspect([1, 1, 0.5])
                    ax.view_init(elev=30, azim=45)

                    # show the plot
                    plt.show()

                line = f.readline()  # read the next line
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down")
        exit()


if __name__ == "__main__":
    main()
