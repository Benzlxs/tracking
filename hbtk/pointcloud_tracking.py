# Copyright 2018, Xuesong LI, (email: benzlee08@gmail.com). All Rights Reserved.

import os
import sys
import shutil
import pickle
from pathlib import Path
import fire
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from matplotlib import gridspec
from skimage import io


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
#sys.path.append(os.path.join( os.path.dirname(ROOT_DIR), 'hbtk'))
#sys.path.append(os.path.join( os.path.dirname(ROOT_DIR), 'hbtk', 'protos'))


from google.protobuf import text_format
from hbtk.protos import pipeline_pb2
from hbtk.detectors.two_merge_one import HybridDetector
from hbtk.trackers.sort_3d import Sort_3d
from hbtk.utils.pc_plot import point_cloud_2_birdseye, convert_xyz_to_img
from hbtk.dataset.kitti_dataset import Kitti_dataset
from hbtk.utils import box_np_ops

colours = {'Bg':[0.5,1,0.5],
            'Car': [1,1,1],
           'Pedestrian':[1,1,0],
           'Cyclist':[1,0,1]}

lw = {'Bg':2,
      'Car': 3,
      'Pedestrian':2,
       'Cyclist':2}

class_type={0:'Bg', 1:'Car', 2:'Pedestrain', 3:'Cyclist'}

def fig_initialization():
    cmap = matplotlib.cm.Spectral
    cmap.set_under(color='black')
    cmap.set_bad(color='black')
    plt.ion()
    #fig = plt.figure(figsize=(10, 8))
    fig = plt.figure(num=1, figsize=(12,11))
    spec2 = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1, 2]) #[1, 2.2]
    ax1 = fig.add_subplot(spec2[0,0])
    ax2 = fig.add_subplot(spec2[1,0])
    return fig, ax1, ax2, cmap

def pointcloud_visulization(config_path=None,
             output_dir='./results',
             display = True,
             ):
    """ main function for tracking

    we run different detectors and combine them together to improve efficiency
    without decreasing perfromance too much. Detection-->Tracking

    Args:
        config_path:
        output_dir:

    Returns:
        sss

    Raises:
        IOError: An error occurred accesssing hte bigtable. Table of object

    """
    #display function first
    dataset_root_path='/home/ben/Dataset/KITTI/2011_09_26/2011_09_26_drive_0084_sync'
    dataset_path = Path(dataset_root_path)
    if display:
        cmap = matplotlib.cm.Spectral
        cmap.set_under(color='black')
        cmap.set_bad(color='black')
        plt.ion()
        #fig = plt.figure(figsize=(10, 8))
        fig = plt.figure()

    if display:
        res=0.1
        side_range=(-60., 60.)
        fwd_range = (5.,  65.)
        height_range=(-1.8, 1.)
        spec2 = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1, 2.5]) #[1, 2.2]
        ax1 = fig.add_subplot(spec2[0,0])
        ax2 = fig.add_subplot(spec2[1,0])

        image_path = list(sorted(dataset_path.glob('image_02/data/*.png')))
        pointcloud_path = list(sorted(dataset_path.glob('velodyne_points/data/*.bin')))
        gt_dets_path = list(sorted(dataset_path.glob('detection/gt/*.txt')))
        assert len(image_path) == len(pointcloud_path), "the image and point cloud files should be the same"
        for i in range(len(image_path)):
            # print("No frame:{}".format(i))
            img_path = str(image_path[i])
            im =io.imread(img_path)
            # display image
            ax1.cla()
            ax1.imshow(im)
            ax1.set_title('image')
            ax1.set_xticks([])
            ax1.set_yticks([])

            # display point cloud
            pc_path = str(pointcloud_path[i])
            points_v = np.fromfile(pc_path, dtype=np.float32, count=-1).reshape([-1, 4])
            pc_img = point_cloud_2_birdseye(points_v, res=res, side_range=side_range, fwd_range=fwd_range)
            # read the bounding boxes
            det_path = str(gt_dets_path[i])
            with open(det_path, 'r') as f:
                lines = f.readlines()
            content = [line.strip().split(',') for line in lines]

            ax2.cla()
            ax2.imshow(pc_img,cmap=cmap, vmin = 1, vmax=255)

            for det in content:
                if det[0] in ['Car', 'Pedestrian', 'Cyclist']:
                    bb = convert_xyz_to_img( np.array(det[1:], dtype=np.float), res=res, side_range=side_range, fwd_range=fwd_range)
                    t2 = matplotlib.transforms.Affine2D().rotate_around((bb[2]+bb[0])/2, (bb[3]+bb[1])/2, -np.float(det[7])) + ax2.transData
                    # _box = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], angle= -np.float(det[7]), fill=False, lw=lw[det[0]], ec=colours[det[0]])
                    _box = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], fill=False, lw=lw[det[0]], ec=colours[det[0]], transform=t2)
                    ax2.add_patch(_box)
            ax2.set_title('point cloud')
            ax2.set_xticks([])
            ax2.set_yticks([])


            #plt.title('Tracking')
            fig.canvas.flush_events()
            #plt.draw()
            plt.show()
            # ax1.cla()
            time.sleep(0.1)

def pointcloud_tracking_gt(config_path=None,
                        output_dir='./results',
                        display = False,
                        display_trajectory = False,
                        downsample_num = 400,

             ):
    """ main function for tracking

    we run different detectors and combine them together to improve efficiency
    without decreasing perfromance too much. Detection-->Tracking

    Args:
        config_path:
        output_dir:

    Returns:
        sss

    Raises:
        IOError: An error occurred accesssing hte bigtable. Table of object

    """
    # read configuration file
    config = pipeline_pb2.TrackingPipeline()
    output_folder_dir = Path(output_dir)
    output_folder_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "r") as f:
        protos_str = f.read()
        text_format.Merge(protos_str, config)
    shutil.copyfile(config_path, str(output_dir+"/"+"pipeline.config"))
    detector_config = config.detector
    filter_config = config.filter
    tracker_config = config.tracker
    dataset_config = config.dataset


    # figure initialization
    if display: fig,ax1,ax2, cmap = fig_initialization()
    if display_trajectory: plt.ion();fig2=plt.figure(num=2, figsize=(12,11));ax2_1=fig2.add_subplot(1,1,1);

    # initialization
    Dataset = Kitti_dataset(dataset_config)
    mot_tracker = Sort_3d(config=tracker_config, data_association=filter_config.data_association)

    # read calibration data
    calib_data_velo_2_cam = Dataset.calib_data_velo_2_cam
    calib_data_cam_2_cam = Dataset.calib_data_cam_2_cam
    robot_poses = []
    global_maps = None
    for i in range(0, Dataset.__len__()):
        # robot positions
        dets = Dataset.get_gt(i)
        local_points = np.zeros((4, len(dets)+1), dtype=np.float32)
        local_points[3,:] = 1
        dets = np.array(dets, dtype=np.float32)
        local_points[0,1:] = dets[:,1]
        local_points[1,1:] = dets[:,2]
        local_points[2,1:] = dets[:,3]
        # convert into global for robot
        global_points = Dataset.pose[i].dot(local_points)
        cur_robot_pose = [global_points[0,0], global_points[1,0], Dataset.yaw[i]]
        robot_poses.append(cur_robot_pose)
        # objects global position
        dets[:,1] = global_points[0,1:]
        dets[:,2] = global_points[1,1:]
        dets[:,3] = global_points[2,1:]
        dets[:,7] += cur_robot_pose[2] # theta

        # prediction step
        # data associations
        # updating step
        trackers = mot_tracker.update(dets)



        # plotting all objects
        if display_trajectory:
            ax2_1.cla()
            # prepare the point cloud
            pc_path = str(Dataset.pc_list[i])
            points_v = np.fromfile(pc_path, dtype=np.float32, count=-1).reshape([-1, 4])
            # xx
            _idx_dist = np.where( (points_v[:,1]>3 ) | (points_v[:,1]<-3 ) )
            points_v = points_v[_idx_dist[0],:]
            # yy
            _idx_dist = np.where( (points_v[:,0]>10 ) | (points_v[:,1]<- 10) )
            points_v = points_v[_idx_dist[0],:]


            _ind = np.random.randint(points_v.shape[0], size=downsample_num )
            points_v = points_v[_ind,:]
            points_v[:,3] = 1
            global_points_v = Dataset.pose[i].dot( points_v.T )
            if global_maps is None:
                global_maps = global_points_v
            else:
                global_maps = np.hstack((global_maps, global_points_v))
            ax2_1.plot(global_maps[0,:], global_maps[1,:], '.', color=[0.7,0.7,0.7], markersize=0.5)
            #ax2_1.plot(global_points_v[0,:], global_points_v[1,:], '.', color=[0.5,0.5,0.5], markersize=0.5)


            # plot the robots itself trajectory and vehicle
            plot_poses = np.array(robot_poses)
            ax2_1.plot(plot_poses[:,0], plot_poses[:,1], '-.', color='red')
            # ax2_1.plot(plot_poses[-1,0], plot_poses[-1,1], 'bs', color='red', markersize=15)
            t2 = matplotlib.transforms.Affine2D().rotate_around(cur_robot_pose[0], cur_robot_pose[1], np.float(cur_robot_pose[2])) + ax2_1.transData
            _box = patches.Rectangle((cur_robot_pose[0]-2, cur_robot_pose[1]-1), 4 , 2, lw=lw['Car'], ec=[1,0,0], fc=[1,0,0], transform=t2, zorder=0)
            ax2_1.add_patch(_box)

            # plot all trackters
            for t_idx in range(len(mot_tracker.trackers)):
                trk_type = class_type[np.int(mot_tracker.trackers[t_idx].category)]
                if trk_type in ['Car','Cyclist','Pedestrian']:
                    trace = mot_tracker.trackers[t_idx].history
                    trace = np.array(trace)
                    x = trace[-1,0]
                    y = trace[-1,1]
                    h = mot_tracker.trackers[t_idx].height
                    l = mot_tracker.trackers[t_idx].length
                    w = mot_tracker.trackers[t_idx].width
                    theta = trace[-1,2]
                    category = mot_tracker.trackers[t_idx].category
                    _cc = mot_tracker.trackers[t_idx].color
                    ax2_1.plot(trace[:,0], trace[:,1], '-.', color=_cc, markersize=4)
                    # ax2_1.plot(trace[-1,0], trace[-1,1], 'bs', color=_cc, markersize=15)
                    t2 = matplotlib.transforms.Affine2D().rotate_around(x, y, np.float(theta)) + ax2_1.transData
                    _box = patches.Rectangle((x-l/2, y-h/2), l , h, lw=lw['Car'], ec=_cc, fc=_cc, transform=t2, zorder=t_idx)
                    ax2_1.add_patch(_box)
                    # velocity and type class
                    _v_ = mot_tracker.trackers[t_idx].X[3]
                    ax2_1.text(trace[-1,0], trace[-1,1], '{},V:{:.1f}'.format(trk_type, _v_), color=[0,0,0], fontsize=8)

            ax2_1.set_xlim([-20 ,200])
            ax2_1.set_ylim([-200,50])

            fig2.canvas.draw()
            name_img = 'global_img/%06d.png'% i
            plt.savefig((output_folder_dir/name_img))
            # time.sleep(0.1)
            fig2.canvas.flush_events()

        if display:
            res=0.1
            side_range=(-60., 60.)
            fwd_range = (5.,  65.)
            height_range=(-1.8, 1.)
            # read the bounding boxes
            det_path = str(Dataset.gt_list[i])
            with open(det_path, 'r') as f:
                lines = f.readlines()
            content = [line.strip().split(',') for line in lines]

            img_path = str( Dataset.img_list[i])
            im =io.imread(img_path)
            # display image
            ax1.cla()
            ax1.imshow(im)
            for det in content:
                if det[0] in ['Bg','Car', 'Pedestrian', 'Cyclist']:
                    _r_ = calib_data_velo_2_cam['R'].reshape(3,3)
                    _t_ = calib_data_velo_2_cam['T'].reshape(3,1)
                    Trev2c = np.vstack((np.hstack([_r_, _t_]), [0, 0, 0, 1]))
                    # Trev2c = np.hstack([_r_, _t_])
                    Rect = calib_data_cam_2_cam['R_rect_02'].reshape(3,3)
                    Rect = np.vstack((np.hstack((Rect, [[0],[0],[0]])), [0, 0, 0, 1]))
                    P2 = calib_data_cam_2_cam['P_rect_02'].reshape(3,4)
                    P2 = np.vstack((P2, [0, 0, 0, 1]))
                    # def
                    # Trev2c=np.array([[ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],[ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],[ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],[ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])
                    # Rect=np.array([[ 0.9999239 ,  0.00983776, -0.00744505,  0.        ],[-0.0098698 ,  0.9999421 , -0.00427846,  0.        ],[ 0.00740253,  0.00435161,  0.9999631 ,  0.        ],[ 0.        ,  0.        ,  0.        ,  1.        ]])
                    # P2=np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],[0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],[0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03],[0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]])
                    # from lidar to camer
                    det_order = [det[1], det[2], det[3], det[4], det[5], det[6], det[7]]
                    box_camera = box_np_ops.box_lidar_to_camera(np.array(det_order, dtype=np.float).reshape(1,7), Rect, Trev2c)
                    locs = box_camera[:, :3]
                    dims = box_camera[:, 3:6]
                    angles =box_camera[:, 6]
                    camera_box_origin = [0.5, 1.0, 0.5]
                    box_corners = box_np_ops.center_to_corner_box3d(
                        locs, dims, angles, camera_box_origin, axis=1)
                    box_corners_in_image = box_np_ops.project_to_image(
                        box_corners, P2)
                    # project into image
                    # convert 3D bounding boxes into 2D image plane
                    # bb = convert_xyz_to_img( np.array(det[1:], dtype=np.float), res=res, side_range=side_range, fwd_range=fwd_range)
                    minxy = np.min(box_corners_in_image, axis=1)[0]
                    minxy[0] = 0 if minxy[0]<0 else minxy[0]
                    minxy[1] = 0 if minxy[1]<0 else minxy[1]
                    maxxy = np.max(box_corners_in_image, axis=1)[0]
                    maxxy[0] = im.shape[1] if maxxy[0]> im.shape[1] else maxxy[0]
                    maxxy[1] = im.shape[0] if maxxy[1]> im.shape[0] else maxxy[1]

                    bb = [ minxy[0], minxy[1], maxxy[0] - minxy[0],   maxxy[1] - minxy[1]]
                    #t2 = matplotlib.transforms.Affine2D().rotate_around((bb[2]+bb[0])/2, (bb[3]+bb[1])/2, -np.float(det[7])) + ax2.transData
                    # _box = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], angle= -np.float(det[7]), fill=False, lw=lw[det[0]], ec=colours[det[0]])
                    _box = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], fill=False, lw=lw[det[0]], ec=colours[det[0]])
                    ax1.add_patch(_box)

            ax1.set_title('image')
            ax1.set_xticks([])
            ax1.set_yticks([])

            # display point cloud
            pc_path = str(Dataset.pc_list[i])
            points_v = np.fromfile(pc_path, dtype=np.float32, count=-1).reshape([-1, 4])
            pc_img = point_cloud_2_birdseye(points_v, res=res, side_range=side_range, fwd_range=fwd_range)
            ax2.cla()
            ax2.imshow(pc_img,cmap=cmap, vmin = 1, vmax=255)

            for det in content:
                if det[0] in ['Bg','Car', 'Pedestrian', 'Cyclist']:
                    bb = convert_xyz_to_img( np.array(det[1:], dtype=np.float), res=res, side_range=side_range, fwd_range=fwd_range)
                    t2 = matplotlib.transforms.Affine2D().rotate_around((bb[2]+bb[0])/2, (bb[3]+bb[1])/2, -np.float(det[7])) + ax2.transData
                    # t2 = matplotlib.transforms.Affine2D().rotate_around((bb[2]+bb[0])/2, (bb[3]+bb[1])/2, 0) + ax2.transData
                    # _box = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], angle= -np.float(det[7]), fill=False, lw=lw[det[0]], ec=colours[det[0]])
                    _box = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], fill=False, lw=lw[det[0]], ec=colours[det[0]], transform=t2)
                    ax2.add_patch(_box)
            ax2.set_title('point cloud')
            ax2.set_xticks([])
            ax2.set_yticks([])
            fig.canvas.draw()
            name_img = 'bv_img/%06d.png'% i
            plt.savefig((output_folder_dir/name_img))
            #time.sleep(0.1)
            fig.canvas.flush_events()

    # leave the plotting there
    wait = input("PRESS ENTER TO CLOSE.")

def pointcloud_tracking_det(config_path=None,
                        output_dir='./results',
                        display = False,
                        display_trajectory = False,
                        downsample_num = 400,

             ):
    """ main function for tracking

    we run different detectors and combine them together to improve efficiency
    without decreasing perfromance too much. Detection-->Tracking

    Args:
        config_path:
        output_dir:

    Returns:
        sss

    Raises:
        IOError: An error occurred accesssing hte bigtable. Table of object

    """
    # read configuration file
    config = pipeline_pb2.TrackingPipeline()
    output_folder_dir = Path(output_dir)
    output_folder_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "r") as f:
        protos_str = f.read()
        text_format.Merge(protos_str, config)
    shutil.copyfile(config_path, str(output_dir+"/"+"pipeline.config"))
    detector_config = config.detector
    filter_config = config.filter
    tracker_config = config.tracker
    dataset_config = config.dataset


    # figure initialization
    if display: fig,ax1,ax2, cmap = fig_initialization()
    if display_trajectory: plt.ion();fig2=plt.figure(num=2, figsize=(12,11));ax2_1=fig2.add_subplot(1,1,1);

    # initialization
    Dataset = Kitti_dataset(dataset_config)
    mot_tracker = Sort_3d(config=tracker_config, data_association=filter_config.data_association)

    # read calibration data
    calib_data_velo_2_cam = Dataset.calib_data_velo_2_cam
    calib_data_cam_2_cam = Dataset.calib_data_cam_2_cam
    robot_poses = []
    global_maps = None
    for i in range(0, Dataset.__len__()):
        # robot positions
        dets = Dataset.get_detection_gt(i)
        local_points = np.zeros((4, len(dets)+1), dtype=np.float32)
        local_points[3,:] = 1
        dets = np.array(dets, dtype=np.float32)
        local_points[0,1:] = dets[:,1]
        local_points[1,1:] = dets[:,2]
        local_points[2,1:] = dets[:,3]
        # convert into global for robot
        global_points = Dataset.pose[i].dot(local_points)
        cur_robot_pose = [global_points[0,0], global_points[1,0], Dataset.yaw[i]]
        robot_poses.append(cur_robot_pose)
        # objects global position
        dets[:,1] = global_points[0,1:]
        dets[:,2] = global_points[1,1:]
        dets[:,3] = global_points[2,1:]
        dets[:,7] += cur_robot_pose[2] # theta

        # prediction step
        # data associations
        # updating step
        trackers = mot_tracker.update(dets)

        print("Frame_id:{}".format(i))

        # plotting all objects
        if display_trajectory:
            ax2_1.cla()
            # prepare the point cloud
            pc_path = str(Dataset.pc_list[i])
            points_v = np.fromfile(pc_path, dtype=np.float32, count=-1).reshape([-1, 4])
            # xx
            _idx_dist = np.where( (points_v[:,1]>3 ) | (points_v[:,1]<-3 ) )
            points_v = points_v[_idx_dist[0],:]
            # yy
            _idx_dist = np.where( (points_v[:,0]>10 ) | (points_v[:,1]<- 10) )
            points_v = points_v[_idx_dist[0],:]


            _ind = np.random.randint(points_v.shape[0], size=downsample_num )
            points_v = points_v[_ind,:]
            points_v[:,3] = 1
            global_points_v = Dataset.pose[i].dot( points_v.T )
            if global_maps is None:
                global_maps = global_points_v
            else:
                global_maps = np.hstack((global_maps, global_points_v))
            ax2_1.plot(global_maps[0,:], global_maps[1,:], '.', color=[0.7,0.7,0.7], markersize=0.5)
            #ax2_1.plot(global_points_v[0,:], global_points_v[1,:], '.', color=[0.5,0.5,0.5], markersize=0.5)


            # plot the robots itself trajectory and vehicle
            plot_poses = np.array(robot_poses)
            ax2_1.plot(plot_poses[:,0], plot_poses[:,1], '-.', color='red')
            # ax2_1.plot(plot_poses[-1,0], plot_poses[-1,1], 'bs', color='red', markersize=15)
            t2 = matplotlib.transforms.Affine2D().rotate_around(cur_robot_pose[0], cur_robot_pose[1], np.float(cur_robot_pose[2])) + ax2_1.transData
            _box = patches.Rectangle((cur_robot_pose[0]-2, cur_robot_pose[1]-1), 4 , 2, lw=lw['Car'], ec=[1,0,0], fc=[1,0,0], transform=t2, zorder=0)
            ax2_1.add_patch(_box)

            # plot all trackters
            for t_idx in range(len(mot_tracker.trackers)):
                trk_type = class_type[np.int(mot_tracker.trackers[t_idx].category)]
                if trk_type in ['Car','Cyclist','Pedestrian']:
                    trace = mot_tracker.trackers[t_idx].history
                    trace = np.array(trace)
                    x = trace[-1,0]
                    y = trace[-1,1]
                    h = mot_tracker.trackers[t_idx].height
                    l = mot_tracker.trackers[t_idx].length
                    w = mot_tracker.trackers[t_idx].width
                    theta = trace[-1,2]
                    category = mot_tracker.trackers[t_idx].category
                    _cc = mot_tracker.trackers[t_idx].color
                    ax2_1.plot(trace[:,0], trace[:,1], '-.', color=_cc, markersize=4)
                    # ax2_1.plot(trace[-1,0], trace[-1,1], 'bs', color=_cc, markersize=15)
                    t2 = matplotlib.transforms.Affine2D().rotate_around(x, y, np.float(theta)) + ax2_1.transData
                    _box = patches.Rectangle((x-l/2, y-h/2), l , h, lw=lw['Car'], ec=_cc, fc=_cc, transform=t2, zorder=t_idx)
                    ax2_1.add_patch(_box)
                    # velocity and type class
                    _v_ = mot_tracker.trackers[t_idx].X[3]
                    ax2_1.text(trace[-1,0], trace[-1,1], '{},V:{:.1f}'.format(trk_type, _v_), color=[0,0,0], fontsize=8)

            ax2_1.set_xlim([-20 ,200])
            ax2_1.set_ylim([-200,50])

            fig2.canvas.draw()
            name_img = 'global_img/%06d.png'% i
            plt.savefig((output_folder_dir/name_img))
            # time.sleep(0.1)
            fig2.canvas.flush_events()

        if display:
            res=0.1
            side_range=(-60., 60.)
            fwd_range = (5.,  65.)
            height_range=(-1.8, 1.)
            # read the bounding boxes
            det_path = str(Dataset.det_gt_list[i])
            with open(det_path, 'r') as f:
                lines = f.readlines()
            content = [line.strip().split(',') for line in lines]

            img_path = str( Dataset.img_list[i])
            im =io.imread(img_path)
            # display image
            ax1.cla()
            ax1.imshow(im)
            for det in content:
                if det[0] in ['Bg','Car', 'Pedestrian', 'Cyclist']:
                    _r_ = calib_data_velo_2_cam['R'].reshape(3,3)
                    _t_ = calib_data_velo_2_cam['T'].reshape(3,1)
                    Trev2c = np.vstack((np.hstack([_r_, _t_]), [0, 0, 0, 1]))
                    # Trev2c = np.hstack([_r_, _t_])
                    Rect = calib_data_cam_2_cam['R_rect_02'].reshape(3,3)
                    Rect = np.vstack((np.hstack((Rect, [[0],[0],[0]])), [0, 0, 0, 1]))
                    P2 = calib_data_cam_2_cam['P_rect_02'].reshape(3,4)
                    P2 = np.vstack((P2, [0, 0, 0, 1]))
                    # def
                    # Trev2c=np.array([[ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],[ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],[ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],[ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])
                    # Rect=np.array([[ 0.9999239 ,  0.00983776, -0.00744505,  0.        ],[-0.0098698 ,  0.9999421 , -0.00427846,  0.        ],[ 0.00740253,  0.00435161,  0.9999631 ,  0.        ],[ 0.        ,  0.        ,  0.        ,  1.        ]])
                    # P2=np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],[0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],[0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03],[0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]])
                    # from lidar to camer
                    det_order = [det[1], det[2], det[3], det[4], det[5], det[6], det[7]]
                    box_camera = box_np_ops.box_lidar_to_camera(np.array(det_order, dtype=np.float).reshape(1,7), Rect, Trev2c)
                    locs = box_camera[:, :3]
                    dims = box_camera[:, 3:6]
                    angles =box_camera[:, 6]
                    camera_box_origin = [0.5, 1.0, 0.5]
                    box_corners = box_np_ops.center_to_corner_box3d(
                        locs, dims, angles, camera_box_origin, axis=1)
                    box_corners_in_image = box_np_ops.project_to_image(
                        box_corners, P2)
                    # project into image
                    # convert 3D bounding boxes into 2D image plane
                    # bb = convert_xyz_to_img( np.array(det[1:], dtype=np.float), res=res, side_range=side_range, fwd_range=fwd_range)
                    minxy = np.min(box_corners_in_image, axis=1)[0]
                    minxy[0] = 0 if minxy[0]<0 else minxy[0]
                    minxy[1] = 0 if minxy[1]<0 else minxy[1]
                    maxxy = np.max(box_corners_in_image, axis=1)[0]
                    maxxy[0] = im.shape[1] if maxxy[0]> im.shape[1] else maxxy[0]
                    maxxy[1] = im.shape[0] if maxxy[1]> im.shape[0] else maxxy[1]

                    bb = [ minxy[0], minxy[1], maxxy[0] - minxy[0],   maxxy[1] - minxy[1]]
                    #t2 = matplotlib.transforms.Affine2D().rotate_around((bb[2]+bb[0])/2, (bb[3]+bb[1])/2, -np.float(det[7])) + ax2.transData
                    # _box = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], angle= -np.float(det[7]), fill=False, lw=lw[det[0]], ec=colours[det[0]])
                    _box = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], fill=False, lw=lw[det[0]], ec=colours[det[0]])
                    ax1.add_patch(_box)

            ax1.set_title('image')
            ax1.set_xticks([])
            ax1.set_yticks([])

            # display point cloud
            pc_path = str(Dataset.pc_list[i])
            points_v = np.fromfile(pc_path, dtype=np.float32, count=-1).reshape([-1, 4])
            pc_img = point_cloud_2_birdseye(points_v, res=res, side_range=side_range, fwd_range=fwd_range)
            ax2.cla()
            ax2.imshow(pc_img,cmap=cmap, vmin = 1, vmax=255)

            for det in content:
                if det[0] in ['Bg','Car', 'Pedestrian', 'Cyclist']:
                    bb = convert_xyz_to_img( np.array(det[1:], dtype=np.float), res=res, side_range=side_range, fwd_range=fwd_range)
                    t2 = matplotlib.transforms.Affine2D().rotate_around((bb[2]+bb[0])/2, (bb[3]+bb[1])/2, -np.float(det[7])) + ax2.transData
                    # _box = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], angle= -np.float(det[7]), fill=False, lw=lw[det[0]], ec=colours[det[0]])
                    _box = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], fill=False, lw=lw[det[0]], ec=colours[det[0]], transform=t2)
                    ax2.add_patch(_box)
            ax2.set_title('point cloud')
            ax2.set_xticks([])
            ax2.set_yticks([])
            fig.canvas.draw()
            name_img = 'bv_img/%06d.png'% i
            plt.savefig((output_folder_dir/name_img))
            #time.sleep(0.1)
            fig.canvas.flush_events()

    # leave the plotting there
    wait = input("PRESS ENTER TO CLOSE.")

def pointcloud_tracking_classification(config_path=None,
                        output_dir='./results',
                        display = False,
                        display_trajectory = False,
                        downsample_num = 400,

             ):
    """ main function for tracking

    we run different detectors and combine them together to improve efficiency
    without decreasing perfromance too much. Detection-->Tracking

    Args:
        config_path:
        output_dir:

    Returns:
        sss

    Raises:
        IOError: An error occurred accesssing hte bigtable. Table of object

    """
    # read configuration file
    config = pipeline_pb2.TrackingPipeline()
    output_folder_dir = Path(output_dir)
    output_folder_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "r") as f:
        protos_str = f.read()
        text_format.Merge(protos_str, config)
    shutil.copyfile(config_path, str(output_dir+"/"+"pipeline.config"))
    detector_config = config.detector
    filter_config = config.filter
    tracker_config = config.tracker
    dataset_config = config.dataset


    # figure initialization
    if display: fig,ax1,ax2, cmap = fig_initialization()
    if display_trajectory: plt.ion();fig2=plt.figure(num=2, figsize=(12,11));ax2_1=fig2.add_subplot(1,1,1);

    # initialization
    Dataset = Kitti_dataset(dataset_config)
    mot_tracker = Sort_3d(config=tracker_config, data_association=filter_config.data_association)

    # read calibration data
    calib_data_velo_2_cam = Dataset.calib_data_velo_2_cam
    calib_data_cam_2_cam = Dataset.calib_data_cam_2_cam
    robot_poses = []
    global_maps = None
    for i in range(0, Dataset.__len__()):
        # robot positions
        dets = Dataset.get_detection_class(i)
        local_points = np.zeros((4, len(dets)+1), dtype=np.float32)
        local_points[3,:] = 1
        dets = np.array(dets, dtype=np.float32)
        local_points[0,1:] = dets[:,1]
        local_points[1,1:] = dets[:,2]
        local_points[2,1:] = dets[:,3]
        # convert into global for robot
        global_points = Dataset.pose[i].dot(local_points)
        cur_robot_pose = [global_points[0,0], global_points[1,0], Dataset.yaw[i]]
        robot_poses.append(cur_robot_pose)
        # objects global position
        dets[:,1] = global_points[0,1:]
        dets[:,2] = global_points[1,1:]
        dets[:,3] = global_points[2,1:]
        dets[:,7] += cur_robot_pose[2] # theta


        # prediction step
        # data associations
        # updating step
        trackers = mot_tracker.update(dets)

        print("Frame_id:{}".format(i))

        # plotting all objects
        if display_trajectory:
            ax2_1.cla()
            # prepare the point cloud
            pc_path = str(Dataset.pc_list[i])
            points_v = np.fromfile(pc_path, dtype=np.float32, count=-1).reshape([-1, 4])
            # xx
            _idx_dist = np.where( (points_v[:,1]>3 ) | (points_v[:,1]<-3 ) )
            points_v = points_v[_idx_dist[0],:]
            # yy
            _idx_dist = np.where( (points_v[:,0]>10 ) | (points_v[:,1]<- 10) )
            points_v = points_v[_idx_dist[0],:]


            _ind = np.random.randint(points_v.shape[0], size=downsample_num )
            points_v = points_v[_ind,:]
            points_v[:,3] = 1
            global_points_v = Dataset.pose[i].dot( points_v.T )
            if global_maps is None:
                global_maps = global_points_v
            else:
                global_maps = np.hstack((global_maps, global_points_v))
            ax2_1.plot(global_maps[0,:], global_maps[1,:], '.', color=[0.7,0.7,0.7], markersize=0.5)
            #ax2_1.plot(global_points_v[0,:], global_points_v[1,:], '.', color=[0.5,0.5,0.5], markersize=0.5)
            # plot the robots itself trajectory and vehicle
            plot_poses = np.array(robot_poses)
            ax2_1.plot(plot_poses[:,0], plot_poses[:,1], '-.', color='red')
            # ax2_1.plot(plot_poses[-1,0], plot_poses[-1,1], 'bs', color='red', markersize=15)
            t2 = matplotlib.transforms.Affine2D().rotate_around(cur_robot_pose[0], cur_robot_pose[1], np.float(cur_robot_pose[2])) + ax2_1.transData
            _box = patches.Rectangle((cur_robot_pose[0]-2, cur_robot_pose[1]-1), 4 , 2, lw=lw['Car'], ec=[1,0,0], fc=[1,0,0], transform=t2, zorder=0)
            ax2_1.add_patch(_box)

            # plot all trackters
            for t_idx in range(len(mot_tracker.trackers)):
                trk_type = class_type[np.int(mot_tracker.trackers[t_idx].category)]
                if trk_type in ['Car','Cyclist','Pedestrian']:
                    trace = mot_tracker.trackers[t_idx].history
                    trace = np.array(trace)
                    x = trace[-1,0]
                    y = trace[-1,1]
                    h = mot_tracker.trackers[t_idx].height
                    l = mot_tracker.trackers[t_idx].length
                    w = mot_tracker.trackers[t_idx].width
                    theta = trace[-1,2]
                    category = mot_tracker.trackers[t_idx].category
                    _cc = mot_tracker.trackers[t_idx].color
                    ax2_1.plot(trace[:,0], trace[:,1], '-.', color=_cc, markersize=4)
                    # ax2_1.plot(trace[-1,0], trace[-1,1], 'bs', color=_cc, markersize=15)
                    t2 = matplotlib.transforms.Affine2D().rotate_around(x, y, np.float(theta)) + ax2_1.transData
                    _box = patches.Rectangle((x-l/2, y-h/2), l , h, lw=lw['Car'], ec=_cc, fc=_cc, transform=t2, zorder=t_idx)
                    ax2_1.add_patch(_box)
                    # velocity and type class
                    _v_ = mot_tracker.trackers[t_idx].X[3]
                    ax2_1.text(trace[-1,0], trace[-1,1], '{},V:{:.1f}'.format(trk_type, _v_), color=[0,0,0], fontsize=8)

            ax2_1.set_xlim([-20 ,200])
            ax2_1.set_ylim([-200,50])

            fig2.canvas.draw()
            name_img = 'global_img/%06d.png'% i
            plt.savefig((output_folder_dir/name_img))
            # time.sleep(0.1)
            fig2.canvas.flush_events()

        if display:
            res=0.1
            side_range=(-60., 60.)
            fwd_range = (5.,  65.)
            height_range=(-1.8, 1.)
            # read the bounding boxes
            det_path = str(Dataset.det_classification_list[i])
            with open(det_path, 'r') as f:
                lines = f.readlines()
            content = [line.strip().split(',') for line in lines]

            img_path = str( Dataset.img_list[i])
            im =io.imread(img_path)
            # display image
            ax1.cla()
            ax1.imshow(im)
            for det in content:
                if det[0] in ['Bg','Car', 'Pedestrian', 'Cyclist']:
                    _r_ = calib_data_velo_2_cam['R'].reshape(3,3)
                    _t_ = calib_data_velo_2_cam['T'].reshape(3,1)
                    Trev2c = np.vstack((np.hstack([_r_, _t_]), [0, 0, 0, 1]))
                    # Trev2c = np.hstack([_r_, _t_])
                    Rect = calib_data_cam_2_cam['R_rect_02'].reshape(3,3)
                    Rect = np.vstack((np.hstack((Rect, [[0],[0],[0]])), [0, 0, 0, 1]))
                    P2 = calib_data_cam_2_cam['P_rect_02'].reshape(3,4)
                    P2 = np.vstack((P2, [0, 0, 0, 1]))
                    # def
                    # Trev2c=np.array([[ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],[ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],[ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],[ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])
                    # Rect=np.array([[ 0.9999239 ,  0.00983776, -0.00744505,  0.        ],[-0.0098698 ,  0.9999421 , -0.00427846,  0.        ],[ 0.00740253,  0.00435161,  0.9999631 ,  0.        ],[ 0.        ,  0.        ,  0.        ,  1.        ]])
                    # P2=np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],[0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],[0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03],[0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]])
                    # from lidar to camer
                    det_order = [det[1], det[2], det[3], det[4], det[5], det[6], det[7]]
                    box_camera = box_np_ops.box_lidar_to_camera(np.array(det_order, dtype=np.float).reshape(1,7), Rect, Trev2c)
                    locs = box_camera[:, :3]
                    dims = box_camera[:, 3:6]
                    angles =box_camera[:, 6]
                    camera_box_origin = [0.5, 1.0, 0.5]
                    box_corners = box_np_ops.center_to_corner_box3d(
                        locs, dims, angles, camera_box_origin, axis=1)
                    box_corners_in_image = box_np_ops.project_to_image(
                        box_corners, P2)
                    # project into image
                    # convert 3D bounding boxes into 2D image plane
                    # bb = convert_xyz_to_img( np.array(det[1:], dtype=np.float), res=res, side_range=side_range, fwd_range=fwd_range)
                    minxy = np.min(box_corners_in_image, axis=1)[0]
                    minxy[0] = 0 if minxy[0]<0 else minxy[0]
                    minxy[1] = 0 if minxy[1]<0 else minxy[1]
                    maxxy = np.max(box_corners_in_image, axis=1)[0]
                    maxxy[0] = im.shape[1] if maxxy[0]> im.shape[1] else maxxy[0]
                    maxxy[1] = im.shape[0] if maxxy[1]> im.shape[0] else maxxy[1]

                    bb = [ minxy[0], minxy[1], maxxy[0] - minxy[0],   maxxy[1] - minxy[1]]
                    #t2 = matplotlib.transforms.Affine2D().rotate_around((bb[2]+bb[0])/2, (bb[3]+bb[1])/2, -np.float(det[7])) + ax2.transData
                    # _box = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], angle= -np.float(det[7]), fill=False, lw=lw[det[0]], ec=colours[det[0]])
                    _box = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], fill=False, lw=lw[det[0]], ec=colours[det[0]])
                    ax1.add_patch(_box)
                    # ax1.text(trace[-1,0], trace[-1,1], '{},V:{:.1f}'.format(trk_type, _v_), color=[0,0,0], fontsize=8)

            ax1.set_title('image')
            ax1.set_xticks([])
            ax1.set_yticks([])

            # display point cloud
            pc_path = str(Dataset.pc_list[i])
            points_v = np.fromfile(pc_path, dtype=np.float32, count=-1).reshape([-1, 4])
            pc_img = point_cloud_2_birdseye(points_v, res=res, side_range=side_range, fwd_range=fwd_range)
            ax2.cla()
            ax2.imshow(pc_img,cmap=cmap, vmin = 1, vmax=255)

            for det in content:
                if det[0] in ['Bg','Car', 'Pedestrian', 'Cyclist']:
                    bb = convert_xyz_to_img( np.array(det[1:-1], dtype=np.float), res=res, side_range=side_range, fwd_range=fwd_range)
                    t2 = matplotlib.transforms.Affine2D().rotate_around((bb[2]+bb[0])/2, (bb[3]+bb[1])/2, -np.float(det[7])) + ax2.transData
                    # _box = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], angle= -np.float(det[7]), fill=False, lw=lw[det[0]], ec=colours[det[0]])
                    _box = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], fill=False, lw=lw[det[0]], ec=colours[det[0]], transform=t2)
                    ax2.add_patch(_box)
                    ax2.text(bb[0], bb[1], 'B:%0.2f,C:%0.2f,P:%0.2f,CY:%0.2f'%(np.float(det[8]), np.float(det[9])+np.float(det[11]), np.float(det[10]), np.float(det[12])), color=[1,0,0], fontsize=8)
            ax2.set_title('point cloud')
            ax2.set_xticks([])
            ax2.set_yticks([])
            fig.canvas.draw()
            name_img = 'bv_img/%06d.png'% i
            plt.savefig((output_folder_dir/name_img))
            #time.sleep(0.1)
            fig.canvas.flush_events()

    # leave the plotting there
    wait = input("PRESS ENTER TO CLOSE.")

def pointcloud_tracking_within_ranges(config_path=None,
                                      output_dir='./results/',
                                      phases = ['2011_09_26_drive_0001_sync', '2011_09_26_drive_0020_sync',
                                                '2011_09_26_drive_0035_sync', '2011_09_26_drive_0084_sync'],
                                      ranges = [10,20,30,40,50,60,70,80],):
    # read config file
    config = pipeline_pb2.TrackingPipeline()
    if output_dir is not None:
        output_folder_dir = Path(output_dir)
        output_folder_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r") as f:
        protos_str = f.read()
        text_format.Merge(protos_str, config)
    text_file_path = output_folder_dir / 'efficiency_saving.txt'
    text_file = open(str(text_file_path),'w')
    for pha in phases:
        for ran in ranges:
            print("Processing phase:{}, range:{}".format(pha, ran))
            config.dataset.phase = pha
            config.tracker.interest_range = ran
            num_frames, num_full_detection, num_hybrid_detection = \
                                        _tracking_within_one_range(config)
            text_file.write('%s,%d,%d,%d,%d \n'%(pha, ran, num_frames, num_full_detection, num_hybrid_detection))

    text_file.close()

def _tracking_within_one_range(config):
    """
    Object tracking with hybrid detection method, 3D oject segmentation is used as low-level
    detection method, and segment+pointNet classification is used as high-level detection method,
    Pipeline: current frame ----> segmentation ---> data association ----> unmathced proposals
           ----> classification model ----> next frame
    """

    detector_config = config.detector
    filter_config = config.filter
    tracker_config = config.tracker
    dataset_config = config.dataset

    # initialization
    Dataset = Kitti_dataset(dataset_config)
    mot_tracker = Sort_3d(config=tracker_config, data_association=filter_config.data_association)

    # read calibration data
    calib_data_velo_2_cam = Dataset.calib_data_velo_2_cam
    calib_data_cam_2_cam = Dataset.calib_data_cam_2_cam
    robot_poses = []
    global_maps = None

    # tracking
    num_full_detection  = 0
    num_hybrid_detection  = 0
    for i in range(0, Dataset.__len__()):
        # robot positions
        dets = Dataset.get_detection_gt(i)
        local_points = np.zeros((4, len(dets)+1), dtype=np.float32)
        local_points[3,:] = 1
        dets = np.array(dets, dtype=np.float32)
        # local_points[:,0] is position for robot
        local_points[0,1:] = dets[:,1]
        local_points[1,1:] = dets[:,2]
        local_points[2,1:] = dets[:,3]
        # convert into global for robot
        global_points = Dataset.pose[i].dot(local_points)
        cur_robot_pose = [global_points[0,0], global_points[1,0], Dataset.yaw[i]]
        robot_poses.append(cur_robot_pose)
        # objects global position
        dets[:,1] = global_points[0,1:]
        dets[:,2] = global_points[1,1:]
        dets[:,3] = global_points[2,1:]
        dets[:,7] += cur_robot_pose[2] # theta

        # prediction step
        # data associations
        # updating step
        trackers, num_classification_run = mot_tracker.update_range(dets, cur_robot_pose)


        num_full_detection += dets.shape[0]
        num_hybrid_detection += num_classification_run
    # print("Nume of frames:{}".format(Dataset.__len__()))
    # print("Full detectors:{}".format(num_full_detection))
    # print("Efficient detectors with tracking:{}".format(num_hybrid_detection))
    # print("Ratio:{}".format(num_hybrid_detection/num_full_detection))
    return Dataset.__len__(), num_full_detection, num_hybrid_detection


def pointcloud_tracking_within_one_range(config_path=None,
                                          output_dir=None,
                                          display = False,
                                          display_trajectory = False,
                                          downsample_num = 400,):
    """
    Object tracking with hybrid detection method, 3D oject segmentation is used as low-level
    detection method, and segment+pointNet classification is used as high-level detection method,
    Pipeline: current frame ----> segmentation ---> data association ----> unmathced proposals
           ----> classification model ----> next frame
    """
    # read configuration file
    config = pipeline_pb2.TrackingPipeline()
    if output_dir is not None:
        output_folder_dir = Path(output_dir)
        output_folder_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r") as f:
        protos_str = f.read()
        text_format.Merge(protos_str, config)
    shutil.copyfile(config_path, str(output_dir+"/"+"pipeline.config"))
    detector_config = config.detector
    filter_config = config.filter
    tracker_config = config.tracker
    dataset_config = config.dataset

    # initialization
    Dataset = Kitti_dataset(dataset_config)
    mot_tracker = Sort_3d(config=tracker_config, data_association=filter_config.data_association)

    # read calibration data
    calib_data_velo_2_cam = Dataset.calib_data_velo_2_cam
    calib_data_cam_2_cam = Dataset.calib_data_cam_2_cam
    robot_poses = []
    global_maps = None

    # tracking
    num_full_detection  = 0
    num_hybrid_detection  = 0
    for i in range(0, Dataset.__len__()):
        # robot positions
        dets = Dataset.get_detection_gt(i)
        local_points = np.zeros((4, len(dets)+1), dtype=np.float32)
        local_points[3,:] = 1
        dets = np.array(dets, dtype=np.float32)
        # local_points[:,0] is position for robot
        local_points[0,1:] = dets[:,1]
        local_points[1,1:] = dets[:,2]
        local_points[2,1:] = dets[:,3]
        # convert into global for robot
        global_points = Dataset.pose[i].dot(local_points)
        cur_robot_pose = [global_points[0,0], global_points[1,0], Dataset.yaw[i]]
        robot_poses.append(cur_robot_pose)
        # objects global position
        dets[:,1] = global_points[0,1:]
        dets[:,2] = global_points[1,1:]
        dets[:,3] = global_points[2,1:]
        dets[:,7] += cur_robot_pose[2] # theta

        # prediction step
        # data associations
        # updating step
        trackers, num_classification_run = mot_tracker.update_range(dets, cur_robot_pose)


        num_full_detection += dets.shape[0]
        num_hybrid_detection += num_classification_run
    print("Nume of frames:{}".format(Dataset.__len__()))
    print("Full detectors:{}".format(num_full_detection))
    print("Efficient detectors with tracking:{}".format(num_hybrid_detection))
    print("Ratio:{}".format(num_hybrid_detection/num_full_detection))


def pointcloud_tracking_within_one_range_with_fusion(config_path=None,
                                                     output_dir =None,
                                                     display  = False,
                                                     display_trajectory = False,
                                                     fusion_confidence  = 0.98,
                                                     downsample_num = 400,):
    """
    Object tracking with hybrid detection method, 3D oject segmentation is used as low-level
    detection method, and segment+pointNet classification is used as high-level detection method,
    Pipeline: current frame ----> segmentation ---> data association ----> unmathced proposals
           ----> classification model ----> next frame
    """
    # read configuration file
    config = pipeline_pb2.TrackingPipeline()
    if output_dir is not None:
        output_folder_dir = Path(output_dir)
        output_folder_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r") as f:
        protos_str = f.read()
        text_format.Merge(protos_str, config)
    shutil.copyfile(config_path, str(output_dir+"/"+"pipeline.config"))
    detector_config = config.detector
    filter_config = config.filter
    tracker_config = config.tracker
    dataset_config = config.dataset

    # initialization
    Dataset = Kitti_dataset(dataset_config)
    mot_tracker = Sort_3d(config=tracker_config, data_association=filter_config.data_association, fusion_confidence=fusion_confidence)

    # figure initialization
    if display: fig,ax1,ax2, cmap = fig_initialization()
    if display_trajectory: plt.ion();fig2=plt.figure(num=2, figsize=(12,11));ax2_1=fig2.add_subplot(1,1,1);


    # read calibration data
    calib_data_velo_2_cam = Dataset.calib_data_velo_2_cam
    calib_data_cam_2_cam = Dataset.calib_data_cam_2_cam
    robot_poses = []
    global_maps = None

    # tracking
    num_full_detection  = 0
    num_hybrid_detection  = 0
    for i in range(0, Dataset.__len__()):
        # robot positions
        dets = Dataset.get_detection_class(i)
        local_points = np.zeros((4, len(dets)+1), dtype=np.float32)
        local_points[3,:] = 1
        dets = np.array(dets, dtype=np.float32)
        # local_points[:,0] is position for robot
        local_points[0,1:] = dets[:,1]
        local_points[1,1:] = dets[:,2]
        local_points[2,1:] = dets[:,3]
        # convert into global for robot
        global_points = Dataset.pose[i].dot(local_points)
        cur_robot_pose = [global_points[0,0], global_points[1,0], Dataset.yaw[i]]
        robot_poses.append(cur_robot_pose)
        # objects global position
        dets[:,1] = global_points[0,1:]
        dets[:,2] = global_points[1,1:]
        dets[:,3] = global_points[2,1:]
        dets[:,7] += cur_robot_pose[2] # theta

        # prediction step
        # data associations
        # updating step
        trackers, num_classification_run = mot_tracker.update_range_fusion(dets, cur_robot_pose)

        # plotting all objects
        if display_trajectory:
            ax2_1.cla()
            # prepare the point cloud
            pc_path = str(Dataset.pc_list[i])
            points_v = np.fromfile(pc_path, dtype=np.float32, count=-1).reshape([-1, 4])
            # xx
            _idx_dist = np.where( (points_v[:,1]>3 ) | (points_v[:,1]<-3 ) )
            points_v = points_v[_idx_dist[0],:]
            # yy
            _idx_dist = np.where( (points_v[:,0]>10 ) | (points_v[:,1]<- 10) )
            points_v = points_v[_idx_dist[0],:]


            _ind = np.random.randint(points_v.shape[0], size=downsample_num )
            points_v = points_v[_ind,:]
            points_v[:,3] = 1
            global_points_v = Dataset.pose[i].dot( points_v.T )
            if global_maps is None:
                global_maps = global_points_v
            else:
                global_maps = np.hstack((global_maps, global_points_v))
            ax2_1.plot(global_maps[0,:], global_maps[1,:], '.', color=[0.7,0.7,0.7], markersize=0.5)
            #ax2_1.plot(global_points_v[0,:], global_points_v[1,:], '.', color=[0.5,0.5,0.5], markersize=0.5)
            # plot the robots itself trajectory and vehicle
            plot_poses = np.array(robot_poses)
            ax2_1.plot(plot_poses[:,0], plot_poses[:,1], '-.', color='red')
            # ax2_1.plot(plot_poses[-1,0], plot_poses[-1,1], 'bs', color='red', markersize=15)
            t2 = matplotlib.transforms.Affine2D().rotate_around(cur_robot_pose[0], cur_robot_pose[1], np.float(cur_robot_pose[2])) + ax2_1.transData
            _box = patches.Rectangle((cur_robot_pose[0]-2, cur_robot_pose[1]-1), 4 , 2, lw=lw['Car'], ec=[1,0,0], fc=[1,0,0], transform=t2, zorder=0)
            ax2_1.add_patch(_box)

            # plot all trackters
            for t_idx in range(len(mot_tracker.trackers)):
                trk_type = class_type[np.int(mot_tracker.trackers[t_idx].category)]
                if trk_type in ['Car','Cyclist','Pedestrian']:
                    trace = mot_tracker.trackers[t_idx].history
                    trace = np.array(trace)
                    x = trace[-1,0]
                    y = trace[-1,1]
                    h = mot_tracker.trackers[t_idx].height
                    l = mot_tracker.trackers[t_idx].length
                    w = mot_tracker.trackers[t_idx].width
                    theta = trace[-1,2]
                    category = mot_tracker.trackers[t_idx].category
                    _cc = mot_tracker.trackers[t_idx].color
                    ax2_1.plot(trace[:,0], trace[:,1], '-.', color=_cc, markersize=4)
                    # ax2_1.plot(trace[-1,0], trace[-1,1], 'bs', color=_cc, markersize=15)
                    t2 = matplotlib.transforms.Affine2D().rotate_around(x, y, np.float(theta)) + ax2_1.transData
                    _box = patches.Rectangle((x-l/2, y-h/2), l , h, lw=lw['Car'], ec=_cc, fc=_cc, transform=t2, zorder=t_idx)
                    ax2_1.add_patch(_box)
                    # velocity and type class
                    _v_ = mot_tracker.trackers[t_idx].X[3]
                    ax2_1.text(trace[-1,0], trace[-1,1]    ,'{},V:{:.1f}'.format(trk_type, _v_), color=[0,0,0], fontsize=8)
                    ax2_1.text(trace[-1,0], trace[-1,1]+2.8,'B:{:.2f},C:{:.2f},P:{:.2f},CY:{:.2f}'.format(mot_tracker.trackers[t_idx].confid[0],mot_tracker.trackers[t_idx].confid[1],
                                                                                                          mot_tracker.trackers[t_idx].confid[2],mot_tracker.trackers[t_idx].confid[3]),
                               color=[1,0,0], fontsize=8)

            ax2_1.set_xlim([-20 ,200])
            ax2_1.set_ylim([-200,50])

            fig2.canvas.draw()
            name_img = 'global_img/%06d.png'% i
            plt.savefig((output_folder_dir/name_img))
            # time.sleep(0.1)
            fig2.canvas.flush_events()

        if display:
            res=0.1
            side_range=(-60., 60.)
            fwd_range = (5.,  65.)
            height_range=(-1.8, 1.)
            # read the bounding boxes
            det_path = str(Dataset.det_classification_list[i])
            with open(det_path, 'r') as f:
                lines = f.readlines()
            content = [line.strip().split(',') for line in lines]

            img_path = str( Dataset.img_list[i])
            im =io.imread(img_path)
            # display image
            ax1.cla()
            ax1.imshow(im)
            for det in content:
                if det[0] in ['Bg','Car', 'Pedestrian', 'Cyclist']:
                    _r_ = calib_data_velo_2_cam['R'].reshape(3,3)
                    _t_ = calib_data_velo_2_cam['T'].reshape(3,1)
                    Trev2c = np.vstack((np.hstack([_r_, _t_]), [0, 0, 0, 1]))
                    # Trev2c = np.hstack([_r_, _t_])
                    Rect = calib_data_cam_2_cam['R_rect_02'].reshape(3,3)
                    Rect = np.vstack((np.hstack((Rect, [[0],[0],[0]])), [0, 0, 0, 1]))
                    P2 = calib_data_cam_2_cam['P_rect_02'].reshape(3,4)
                    P2 = np.vstack((P2, [0, 0, 0, 1]))
                    # from lidar to camer
                    det_order = [det[1], det[2], det[3], det[4], det[5], det[6], det[7]]
                    box_camera = box_np_ops.box_lidar_to_camera(np.array(det_order, dtype=np.float).reshape(1,7), Rect, Trev2c)
                    locs = box_camera[:, :3]
                    dims = box_camera[:, 3:6]
                    angles =box_camera[:, 6]
                    camera_box_origin = [0.5, 1.0, 0.5]
                    box_corners = box_np_ops.center_to_corner_box3d(
                        locs, dims, angles, camera_box_origin, axis=1)
                    box_corners_in_image = box_np_ops.project_to_image(
                        box_corners, P2)
                    # project into image
                    # convert 3D bounding boxes into 2D image plane
                    # bb = convert_xyz_to_img( np.array(det[1:], dtype=np.float), res=res, side_range=side_range, fwd_range=fwd_range)
                    minxy = np.min(box_corners_in_image, axis=1)[0]
                    minxy[0] = 0 if minxy[0]<0 else minxy[0]
                    minxy[1] = 0 if minxy[1]<0 else minxy[1]
                    maxxy = np.max(box_corners_in_image, axis=1)[0]
                    maxxy[0] = im.shape[1] if maxxy[0]> im.shape[1] else maxxy[0]
                    maxxy[1] = im.shape[0] if maxxy[1]> im.shape[0] else maxxy[1]

                    bb = [ minxy[0], minxy[1], maxxy[0] - minxy[0],   maxxy[1] - minxy[1]]
                    #t2 = matplotlib.transforms.Affine2D().rotate_around((bb[2]+bb[0])/2, (bb[3]+bb[1])/2, -np.float(det[7])) + ax2.transData
                    # _box = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], angle= -np.float(det[7]), fill=False, lw=lw[det[0]], ec=colours[det[0]])
                    _box = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], fill=False, lw=lw[det[0]], ec=colours[det[0]])
                    ax1.add_patch(_box)
                    # ax1.text(trace[-1,0], trace[-1,1], '{},V:{:.1f}'.format(trk_type, _v_), color=[0,0,0], fontsize=8)

            ax1.set_title('image')
            ax1.set_xticks([])
            ax1.set_yticks([])

            # display point cloud
            pc_path = str(Dataset.pc_list[i])
            points_v = np.fromfile(pc_path, dtype=np.float32, count=-1).reshape([-1, 4])
            pc_img = point_cloud_2_birdseye(points_v, res=res, side_range=side_range, fwd_range=fwd_range)
            ax2.cla()
            ax2.imshow(pc_img,cmap=cmap, vmin = 1, vmax=255)

            for det in content:
                if det[0] in ['Bg','Car', 'Pedestrian', 'Cyclist']:
                    bb = convert_xyz_to_img( np.array(det[1:-1], dtype=np.float), res=res, side_range=side_range, fwd_range=fwd_range)
                    t2 = matplotlib.transforms.Affine2D().rotate_around((bb[2]+bb[0])/2, (bb[3]+bb[1])/2, -np.float(det[7])) + ax2.transData
                    # _box = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], angle= -np.float(det[7]), fill=False, lw=lw[det[0]], ec=colours[det[0]])
                    _box = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], fill=False, lw=lw[det[0]], ec=colours[det[0]], transform=t2)
                    ax2.add_patch(_box)
                    ax2.text(bb[0], bb[1], 'B:%0.2f,C:%0.2f,P:%0.2f,CY:%0.2f'%(np.float(det[8]), np.float(det[9])+np.float(det[11]), np.float(det[10]), np.float(det[12])), color=[1,0,0], fontsize=8)
            ax2.set_title('point cloud')
            ax2.set_xticks([])
            ax2.set_yticks([])
            fig.canvas.draw()
            name_img = 'bv_img/%06d.png'% i
            plt.savefig((output_folder_dir/name_img))
            #time.sleep(0.1)
            fig.canvas.flush_events()



        num_full_detection += dets.shape[0]
        num_hybrid_detection += num_classification_run
    print("Nume of frames:{}".format(Dataset.__len__()))
    print("Full detectors:{}".format(num_full_detection))
    print("Efficient detectors with tracking:{}".format(num_hybrid_detection))
    print("Ratio:{}".format(num_hybrid_detection/num_full_detection))





if __name__=='__main__':
    fire.Fire()


