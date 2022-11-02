import numpy as np
from BEV import *

if __name__ =="__main__":
    pcl = np.load('./lidar.npy')
    print(pcl.shape)
    vis_open3d_pcl(pcl)