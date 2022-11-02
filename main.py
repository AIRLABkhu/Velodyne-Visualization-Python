import time

import pyrealsense2 as rs
from datetime import datetime
from BEV import *

if __name__ == "__main__":
    # 5M 까지만 depth 값을 얻음 그 이상으로는 depth 값을 5로 고정
    Z_parameter = 5

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    # Different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, 5)
    # depth_sensor.set_option(rs.option.depth_units, 0.0001)
    depth_scale = depth_sensor.get_depth_scale()

    print("Depth Scale is: ", depth_scale)

    # Color Settings
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.visual_preset, 0)
    colorizer.set_option(rs.option.color_scheme, 0)

    # We will be removing the background of objects more than clipping_distance_in_meters meters away
    clipping_distance_in_meters = Z_parameter  # meter 단위
    clipping_distance = clipping_distance_in_meters / depth_scale

    ####################################################
    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)
    ####################################################

    # Streaming loop
    try:
        vis = open3d.visualization.Visualizer()


        vis.create_window("Pointcloud", width=320, height=120)
        point_cloud = open3d.geometry.PointCloud()
        geom_added = False

        # Declare filters (interpolation method)
        dec_filter = rs.decimation_filter()  # Decimation - reduces depth frame density. Value range [2-8]. Default is 2.
        spat_filter = rs.spatial_filter()  # Spatial    - edge-preserving spatial smoothing
        temp_filter = rs.temporal_filter()  # Temporal   - reduces temporal noise
        hole_filter = rs.hole_filling_filter()  # Hole Filling

        x_min_all = 99999
        x_max_all = 0
        y_min_all = 99999
        y_max_all = 0
        z_min_all = 99999
        z_max_all = 0


        while True:
            d_start = datetime.now()
            # vis.add_geometry(point_cloud)
            point_cloud.clear()

            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # filter - interpolation
            aligned_depth_frame = spat_filter.process(aligned_depth_frame)
            aligned_depth_frame = temp_filter.process(aligned_depth_frame)

            depth_color_frame = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
            depth_color_frame = cv2.resize(depth_color_frame, (320, 240), interpolation=cv2.INTER_AREA)
            depth_color_frame = depth_color_frame[60:180, :]
            cv2.imshow("depth_color_frame", depth_color_frame)

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())


            depth_image = cv2.resize(depth_image, (320, 240), interpolation=cv2.INTER_AREA)

            color_image  = cv2.resize(color_image, (320, 240), interpolation=cv2.INTER_AREA)
            color_image = color_image[60:180, :]

            ### depth cliping
            depth_image[depth_image>clipping_distance] = clipping_distance

            # Remove background - Set pixels further than clipping_distance to grey
            # grey_color = 150
            # depth_image_3d = np.dstack(
            #     (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
            # Color image background remove
            # color_image = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # depth_image_half_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),
            #                                            cv2.COLORMAP_JET)

            """
            Real sense Intrinsic 320 * 240
            K = np.array([[193.072, 0.0, 161.149],
                          [0.0, 193.072, 119.491],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
            """
            depth_image[:60,:] = 0
            depth_image[180:, :] = 0
            pinhole_camera_intrinsic = open3d.camera.PinholeCameraIntrinsic(320, 240, 193.072, 193.072, 161.149, 119.491)
            img_depth = open3d.geometry.Image(depth_image)
            pcd = open3d.geometry.PointCloud.create_from_depth_image(img_depth, pinhole_camera_intrinsic,
                                                                     depth_scale=1000.0, depth_trunc=1000.0, stride=1)
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            bev = np.asarray(pcd.points)
            # np.save('./pc_test.npy', bev)

            # x_range, y_range, z_range, scale = (-20, 20), (-20, 20), (-2, 2), 10
            # topview_img = velo.velo_2_topview_frame(x_range=x_range, y_range=y_range, z_range=z_range)


            x_min = math.floor(bev[:, 0].min())
            x_max = math.ceil(bev[:, 0].max())
            y_min = math.floor(bev[:, 1].min())
            y_max = math.ceil(bev[:, 1].max())
            z_min = math.floor(bev[:, 2].min())
            z_max = math.ceil(bev[:, 2].max())

            if(x_min_all>x_min): x_min_all = x_min
            if(y_min_all>y_min): y_min_all = y_min
            if(x_max_all<x_max): x_max_all = x_max
            if(y_max_all<y_max): y_max_all = y_max
            if (z_min_all > z_min): z_min_all = z_min
            if (z_max_all < z_max): z_max_all = z_max

            # print("X_min: %d, X_max: %d" % (math.floor(bev[:, 0].min()) - 2, math.ceil(bev[:, 0].max()) + 2))
            # print("Y_min: %d, Y_max: %d" % (math.floor(bev[:, 1].min()) - 2, math.ceil(bev[:, 1].max()) + 2))

            print("X_min: %d, X_max: %d, Y_min: %d, Y_max: %d, Z_min: %d, Z_max: %d"
                  % (x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all))

            # img = birds_eye_point_cloud(bev,side_range=(math.floor(bev[:, 0].min()) - 2, math.ceil(bev[:, 0].max()) + 2),
            #                             fwd_range=(math.floor(bev[:, 1].min()) - 2, math.ceil(bev[:, 1].max()) + 2),
            #                             all_xmin=x_min_all-2, all_ymin=y_min_all-2,res=0.01, saveto="lidar_pil_01.png")
            # img_re = cv2.resize(img, (depth_image.shape[1], depth_image.shape[0]))
            # img_re = cv2.cvtColor(img_re, cv2.COLOR_GRAY2BGR)

            img_z = birds_eye_point_cloud_Z(bev,
                                        all_xmin=y_min_all-2, all_ymin=z_min_all-2, res=0.025)

            cv2.imshow('Bird eye View', img_z)

            # img_z1 = birds_eye_point_cloud_Z(bev,
            #                             side_range=(math.floor(bev[:, 1].min()) - 2, math.ceil(bev[:, 1].max()) + 2),
            #                             fwd_range=(math.floor(bev[:, 2].min()) - 2, math.ceil(bev[:, 2].max()) + 2),
            #                             all_xmin=y_max, all_ymin=z_min_all-3, res=0.005,
            #                             saveto="lidar_pil_01.png")


            # cv2.imshow('Bird eye View1', img_z1)

            # img_re_z = cv2.resize(img_z, (depth_image.shape[1], depth_image.shape[0]))
            # img_re_z = cv2.cvtColor(img_re_z, cv2.COLOR_GRAY2BGR)
            #
            #
            # images = np.hstack((color_image, depth_colormap, img_re_z))
            # cv2.namedWindow('Show', cv2.WINDOW_NORMAL)
            # cv2.imshow('Show', images)
            # cv2.imshow('Depth', depth_colormap)
            cv2.imshow('Image', color_image)


            key = cv2.waitKey(1)


            point_cloud.points = pcd.points
            point_cloud.colors = pcd.colors

            #array_of_colors = np.zeros(shape=bev.shape)
            #array_of_colors[:] = [0, 0, 0]  # Points with label 0 are colored in Green.
            #point_cloud.colors = open3d.utility.Vector3dVector(array_of_colors)

            if geom_added == False:
                vis.add_geometry(point_cloud)
                geom_added = True

            vis.update_geometry(point_cloud)

            # vis_opt = vis.get_render_option()
            # a = vis_opt.point_color_option.color()

            vis.poll_events()
            vis.update_renderer()

            process_time = datetime.now() - d_start
            # print("FPS = {0}".format(1 / process_time.total_seconds()))

            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            elif key & 0xFF == ord('s'):
                img_path = './result/raw.png'
                depth_img_path = './result/depth_colorization.png'
                depth_path = './result/depth_raw.npy'
                BEV_path = './result/BEV.png'

                cv2.imwrite(img_path, color_image)
                cv2.imwrite(depth_img_path, depth_color_frame)
                cv2.imwrite(BEV_path, img_z)
                np.save(depth_path, depth_image)
                np.save('./pointcloud.npy', bev)
                print('Saved images!')



    finally:
        pipeline.stop()