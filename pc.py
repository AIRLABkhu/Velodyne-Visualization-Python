#!/usr/bin/env python
from cv2 import detail_VoronoiSeamFinder
import rospy
import math
from datetime import datetime
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import numpy as np
import os
import open3d 
import matplotlib

dirname = './pointclouds'
vis = open3d.visualization.Visualizer()
pcd = open3d.geometry.PointCloud()
geom_added = False

def vis_open3d_pcl(pc):
    global geom_added, pcd, vis

    pc_intensity = 1 - (pc[:,3] / np.max(pc[:,3]))


    if pc.shape[1] == 4:
        pc = pc[:,:3]

    cmap = matplotlib.cm.get_cmap('jet')

    c = np.array(cmap(pc_intensity)) # RGBA
    c = c[:,:3]

    pcd.clear()
    pcd.points = open3d.utility.Vector3dVector(pc)
    pcd.colors = open3d.utility.Vector3dVector(c)

    if geom_added == False:
        vis.add_geometry(pcd)
        geom_added = True

    vis.update_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()


def callback(input_ros_msg):
    pointcloud = []
    for data in pc2.read_points(input_ros_msg, skip_nans=True):
        # x, y, z, intensity, ring, timestamp
        pointcloud.append(data[:4])
    pointcloud = np.array(pointcloud)
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S:%f')
    vis_open3d_pcl(pointcloud)

    # filename = f'{dirname}/{timestamp}'
    # np.save(filename, pointcloud)
    # print(f'File saved: \'{filename}.npy\' containing {pointcloud.shape[0]} points.')

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('/velodyne_points', PointCloud2, callback)
    # rospy.sleep(10000)
    rospy.spin()

if __name__ == '__main__':

    vis.create_window("Pointcloud", width=640, height=480)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    listener()
