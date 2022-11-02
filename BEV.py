from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import open3d
import numpy as np
import matplotlib.cm

def vis_open3d_pcl(pc):

    pc_intensity = 1 - (pc[:,3] / np.max(pc[:,3]))


    if pc.shape[1] == 4:
        pc = pc[:,:3]

    cmap = matplotlib.cm.get_cmap('jet')

    c = np.array(cmap(pc_intensity)) # RGBA
    c = c[:,:3]
    # c = c.astype(np.int64)

    print(c)
    #
    # pc_intensity = pc_intensity[:, np.newaxis]
    # intensity_color = cv2.applyColorMap(pc_intensity, cv2.COLORMAP_JET)
    # intensity_color = np.squeeze(intensity_color)
    #
    # array_of_colors = np.zeros(shape=pc.shape)
    # array_of_colors[:] = [0, 0, 0]  # Points with label 0 are colored in Green.

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)
    pcd.colors = open3d.utility.Vector3dVector(c)
    open3d.visualization.draw_geometries([pcd])

def matplotlib_3d_ptcloud(output_pcl):
    xdata = output_pcl[:,0]
    ydata = output_pcl[:,1]
    zdata = output_pcl[:,2]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    ax.scatter3D(xdata, ydata, zdata, marker='o', s=10)
    plt.show()

# ==============================================================================
#                                                                   SCALE_TO_255
# ==============================================================================
def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


# ==============================================================================
#                                                          BIRDS_EYE_POINT_CLOUD
# ==============================================================================
def birds_eye_point_cloud(points,
                          side_range=(-10, 10),
                          fwd_range=(-10, 10),
                          all_xmin=0,
                          all_ymin=0,
                          res=0.001,
                          saveto=None):

    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]

    indices = np.argwhere(x_lidar).flatten()

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (x_lidar[indices]/res).astype(np.int32) # x axis is -y in LIDAR
    y_img = (-y_lidar[indices]/res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img -= int(np.floor(all_xmin/res))
    y_img -= int(np.floor(all_ymin/res))

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(z_lidar[indices], min=z_lidar[indices].min(), max=z_lidar[indices].max())

    # FILL PIXEL VALUES IN IMAGE ARRAY
    x_max = int((abs(all_xmin) * 2)/res)
    y_max = int((abs(all_ymin) * 2)/res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] = pixel_values # -y because images start from top left

    return im

def birds_eye_point_cloud_Z(points,
                          all_xmin=0,
                          all_ymin=0,
                          res=0.001):

    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2] # Z는 카메라 쪽이 0 정면

    indices = np.argwhere(y_lidar).flatten()

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (x_lidar[indices]/res).astype(np.int32)
    y_img = (z_lidar[indices]/res).astype(np.int32)

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img -= int(np.floor(all_xmin/res))
    y_img -= int(np.floor(all_ymin/res))

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    if points.shape[0]!=0:
        pixel_values = scale_to_255(y_lidar[indices], min=y_lidar[indices].min(), max=y_lidar[indices].max())
    else:
        x_max = int((abs(all_xmin) * 2) / res)
        y_max = int((abs(all_ymin) * 2) / res)

        im = np.zeros([y_max, x_max], dtype=np.uint8)
        return im


    # FILL PIXEL VALUES IN IMAGE ARRAY
    x_max = int((abs(all_xmin) * 2)/res)
    y_max = int((abs(all_ymin) * 2)/res)

    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] = pixel_values # -y because images start from top left

    return im

def lidar_to_bird_view_img(lidar, x_min, x_max, y_min, y_max, factor=1):
    # Input:
    #   lidar: (N', 4)
    # Output:
    #   birdview: (w, l, 3)

    Y_MIN = y_min
    Y_MAX = y_max
    X_MIN = x_min
    X_MAX = x_max
    VOXEL_X_SIZE = 0.1
    VOXEL_Y_SIZE = 0.1

    INPUT_WIDTH = int((X_MAX - X_MIN) / VOXEL_X_SIZE)
    INPUT_HEIGHT = int((Y_MAX - Y_MIN) / VOXEL_Y_SIZE)

    birdview = np.zeros(
        (INPUT_HEIGHT * factor, INPUT_WIDTH * factor, 1))
    for point in lidar:
        x, y = point[0:2]
        if X_MIN < x < X_MAX and Y_MIN < y < Y_MAX:
            x, y = int((x - X_MIN) / VOXEL_X_SIZE *
                       factor), int((y - Y_MIN) / VOXEL_Y_SIZE * factor)
            birdview[y, x] += 1
    birdview = birdview - np.min(birdview)
    divisor = np.max(birdview) - np.min(birdview)
    # TODO: adjust this factor
    birdview = np.clip((birdview / divisor * 255) *
                       5 * factor, a_min=0, a_max=255)
    birdview = np.tile(birdview, 3).astype(np.uint8)

    birdview_reshape = cv2.resize(birdview, (600,600))
    # cv2.imshow("test",birdview_reshape)
    # cv2.waitKey(100000)
    return birdview

if __name__ =="__main__":
    pc = np.load('./result/pointcloud.npy')

    vis_open3d_pcl(pc)

    # Z는 빼고 X와 Y의 min 과 max 확인
    print("X_min: %d, X_max: %d"%( math.floor(pc[:,0].min()), math.ceil(pc[:,0].max())))
    print("Y_min: %d, Y_max: %d"%(pc[:,1].min(), pc[:, 1].max()))

    # lidar_to_bird_view_img(pc, math.floor(pc[:,0].min()), math.ceil(pc[:,0].max()), math.floor(pc[:,1].min()), math.ceil(pc[:,1].max()))

    # View a Square that is 10m on all sides of the car
    # img = birds_eye_point_cloud(pc,side_range=(math.floor(pc[:, 0].min()) - 2, math.ceil(pc[:, 0].max()) + 2),
    #                                     fwd_range=(math.floor(pc[:, 1].min()) - 2, math.ceil(pc[:, 1].max()) + 2),
    #                                     all_xmin=math.floor(pc[:, 0].min()) - 2, all_ymin=math.floor(pc[:, 1].min())-2
    #                             ,res=0.01, saveto="lidar_pil_01.png")


    img = birds_eye_point_cloud_Z(pc,
                                    all_xmin=math.floor(pc[:, 1].min()) - 2, all_ymin=math.floor(pc[:, 2].min()),
                                    res=0.02,
                                    saveto="lidar_pil_01.png")


    cv2.imshow("BEV", img)
    cv2.waitKey(10000000)
    #
    # # View a Square that is 10m on either side of the car and 20m in front
    # birds_eye_point_cloud(pc, side_range=(-3, 3), fwd_range=(0, 4), res=0.1, saveto="lidar_pil_02.png")
    #
    # # View a rectangle that is 5m on either side of the car and 20m in front
    # birds_eye_point_cloud(pc, side_range=(-3, 3), fwd_range=(-1, 1), res=0.1, saveto="lidar_pil_03.png")
