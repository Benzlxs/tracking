# Copyright 2018, Xuesong LI, (email: benzlee08@gmail.com). All Rights Reserved.

import os
import sys
import shutil
import pickle
import collections
from pathlib import Path
import fire
import time
#import shutil
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

class_type={0:'Bg', 1:'Car', 2:'Pedestrian', 3:'Cyclist'}
LABEL_NUM = collections.namedtuple('LABEL_NUM',['unknow_object_label', 'need_more', 'good_enough'])
label_to_num = LABEL_NUM(unknow_object_label=256, need_more=4, good_enough=8)


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

def pointcloud_tracking_det(config_path='/home/ben/projects/tracking/hbtk/config/kitti_tracking.config',
                        output_dir='./results',
                        display = True,
                        display_trajectory = True,
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

            #ax2_1.set_xlim([-20 ,200]) # for dataset 0084
            #ax2_1.set_ylim([-200,50])  # for dataset 0084
            ax2_1.set_xlim([0 , 250])
            ax2_1.set_ylim([-85,85])


            fig2.canvas.draw()
            name_img = 'global_img/%06d.png'% i
            fig2.savefig((output_folder_dir/name_img), dpi='figure', quality=95)
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

            ax1.set_title('Image')
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
            ax2.set_title('Point cloud')
            ax2.set_xticks([])
            ax2.set_yticks([])
            fig.canvas.draw()
            name_img = 'bv_img/%06d.png'% i
            fig.savefig((output_folder_dir/name_img),dpi='figure', quality=95)
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
                    ax2.text(bb[0], bb[1], 'B:%0.2f,C:%0.2f,P:%0.2f,CY:%0.2f'%(np.float(det[8]), np.float(det[9]), np.float(det[10]), np.float(det[11])), color=[1,0,0], fontsize=8)
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


def __pointcloud_tracking_classification_tracklets__(config, save_directory):
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
    for i in range(0, Dataset.__len__()):
        # robot positions
        dets = Dataset.get_detection_class(i)
        # read the car, pedestrain, cyclist only.
        if len(dets) == 0:
            continue
        local_points = np.zeros((4, len(dets)+1), dtype=np.float32)
        local_points[3,:] = 1
        dets = np.array(dets, dtype=np.float32)
        # copy the array to save the detection results in local coordinate
        dets_local_coordinate = dets.copy()

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
        trackers = mot_tracker.update(dets, tracklet_save_dir=save_directory, dets_local=dets_local_coordinate)
        # save the rest of tracking
        print("Frame_id:{}".format(i))

    mot_tracker.save_all_trk(save_directory)


def pointcloud_tracking_classification_with_saving_tracklets(config_path='/home/ben/projects/tracking/hbtk/config/kitti_tracking.config',
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
    with open(config_path, "r") as f:
        protos_str = f.read()
        text_format.Merge(protos_str, config)
    phases = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0020_sync',
              '2011_09_26_drive_0035_sync','2011_09_26_drive_0084_sync',
              '2011_09_26_drive_0005_sync','2011_09_26_drive_0014_sync',
              '2011_09_26_drive_0019_sync','2011_09_26_drive_0059_sync',]

    for phase in phases:
        print("Phase name: {}".format(phase))
        config.dataset.phase = phase
        save_directory = Path(config.dataset.database_dir) / phase / 'detection/tracklet_det'
        # save_directory.rmdir()
        shutil.rmtree(str(save_directory))
        save_directory.mkdir(parents=True, exist_ok=True)
        # os.remove(str(save_directory/'*'))
        __pointcloud_tracking_classification_tracklets__(config, save_directory)


def pointcloud_tracking_within_ranges(config_path=None,
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
                                                     save_trk_results = False,
                                                     save_det_results = True,
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

        # save PDFs of detectiors which are updated with that of trackers.
        if save_det_results:
            print('save_results in #{} tracker'.format(i))
            result_folder = Path(dataset_config.database_dir)/dataset_config.phase/'detection/dets_trk'
            result_folder.mkdir(parents=True, exist_ok=True)
            result_file_path = str(Dataset.det_gt_list[i]).replace('/dets_gt','/dets_trk')

            with open(result_file_path, 'w') as f:
                for kk in range(len(dets)):
                    cc = np.int(dets[kk,0])
                    if cc >= label_to_num.unknow_object_label:
                        cc = cc - label_to_num.unknow_object_label
                    trk_type = class_type[cc]
                    x = dets[kk,1]
                    y = dets[kk,2]
                    z = dets[kk,3]
                    l = dets[kk,4]
                    w = dets[kk,5]
                    h = dets[kk,6]
                    theta = dets[kk,7]
                    confid_bg = dets[kk,8]
                    confid_car= dets[kk,9]
                    confid_ped= dets[kk,10]
                    confid_cyc= dets[kk,11]
                    num_points= dets[kk,12]
                    f.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%d\n'%(trk_type,
                                 x, y, z, l, w, h, theta, confid_bg, confid_car, confid_ped, confid_cyc, num_points))



        if save_trk_results:
            print('save_results in #{} tracker'.format(i))
            result_folder = Path(dataset_config.database_dir)/dataset_config.phase/'detection/dets_trk'
            result_folder.mkdir(parents=True, exist_ok=True)
            result_file_path = str(Dataset.det_gt_list[i]).replace('/dets_gt','/dets_trk')

            with open(result_file_path, 'w') as f:
                for t_idx in range(len(mot_tracker.trackers)):
                    cc = np.int(mot_tracker.trackers[t_idx].category)
                    if cc >= label_to_num.unknow_object_label:
                        cc = cc - label_to_num.unknow_object_label
                    trk_type = class_type[cc]
                    x = mot_tracker.trackers[t_idx].X[0]
                    y = mot_tracker.trackers[t_idx].X[1]
                    z = mot_tracker.trackers[t_idx].z_s
                    w = mot_tracker.trackers[t_idx].height
                    l = mot_tracker.trackers[t_idx].length
                    h = mot_tracker.trackers[t_idx].width
                    theta = mot_tracker.trackers[t_idx].X[2]
                    confid_bg = mot_tracker.trackers[t_idx].confid[0]
                    confid_car= mot_tracker.trackers[t_idx].confid[1]
                    confid_ped= mot_tracker.trackers[t_idx].confid[2]
                    confid_cyc= mot_tracker.trackers[t_idx].confid[3]
                    f.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f, \n'%(trk_type,
                                 x, y, z, l, w, h, theta, confid_bg, confid_car, confid_ped, confid_cyc))

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
                    ax2.text(bb[0], bb[1], 'B:%0.2f,C:%0.2f,P:%0.2f,CY:%0.2f'%(np.float(det[8]), np.float(det[9]), np.float(det[10]), np.float(det[11])), color=[1,0,0], fontsize=8)
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

def __pointcloud_tracking__fusion__(config,
                                    display  = False,
                                    display_trajectory = False,
                                    save_trk_results = False,
                                    save_det_results = True,
                                    save_all_track_det_confidence=True,
                                    fusion_confidence  = 0.96,
                                    downsample_num = 400,):
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

    # save the history of confidence change with respect to detector and
    # trackers.
    if save_all_track_det_confidence:
        result_trk_folder = Path(dataset_config.database_dir)/dataset_config.phase/'detection/dets_trk_confidence'
        result_trk_folder.mkdir(parents=True, exist_ok=True)


    # initialization
    Dataset = Kitti_dataset(dataset_config)
    mot_tracker = Sort_3d(config=tracker_config, data_association=filter_config.data_association,
                          fusion_confidence=fusion_confidence, result_trk_folder=result_trk_folder)

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
        trackers, num_classification_run = mot_tracker.update_range_fusion(dets, cur_robot_pose, i)

        #if np.any(dets[:,0]==2.):
        #    import pudb
        #    pudb.set_trace()

        check_nan = np.argwhere(np.isnan(dets))
        if check_nan.size!=0:
            import pudb; pudb.set_trace()
        # save PDFs of detectiors which are updated with that of trackers.
        if save_det_results:
            # print('save_results in #{} detectors'.format(i))
            result_folder = Path(dataset_config.database_dir)/dataset_config.phase/'detection/dets_trk'
            result_folder.mkdir(parents=True, exist_ok=True)
            result_file_path = str(Dataset.det_gt_list[i]).replace('/dets_gt','/dets_trk')

            with open(result_file_path, 'w') as f:
                for kk in range(len(dets)):
                    cc = np.int(dets[kk,0])
                    if cc >= label_to_num.unknow_object_label:
                        cc = cc - label_to_num.unknow_object_label
                    trk_type = class_type[cc]
                    x = dets[kk,1]
                    y = dets[kk,2]
                    z = dets[kk,3]
                    l = dets[kk,4]
                    w = dets[kk,5]
                    h = dets[kk,6]
                    theta = dets[kk,7]
                    confid_bg = dets[kk,8]
                    confid_car= dets[kk,9]
                    confid_ped= dets[kk,10]
                    confid_cyc= dets[kk,11]
                    num_points= dets[kk,12]
                    f.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%d\n'%(trk_type,
                                 x, y, z, l, w, h, theta, confid_bg, confid_car, confid_ped, confid_cyc, num_points))



        if save_trk_results:
            print('save_results in #{} tracker'.format(i))
            result_folder = Path(dataset_config.database_dir)/dataset_config.phase/'detection/dets_trk'
            result_folder.mkdir(parents=True, exist_ok=True)
            result_file_path = str(Dataset.det_gt_list[i]).replace('/dets_gt','/dets_trk')

            with open(result_file_path, 'w') as f:
                for t_idx in range(len(mot_tracker.trackers)):
                    # only save the new trackers or ones that were updated last
                    # time
                    if mot_tracker.trackers[t_idx].time_since_update == 0:
                        cc = np.int(mot_tracker.trackers[t_idx].category)
                        if cc >= label_to_num.unknow_object_label:
                            cc = cc - label_to_num.unknow_object_label
                        trk_type = class_type[cc]
                        x = mot_tracker.trackers[t_idx].X[0]
                        y = mot_tracker.trackers[t_idx].X[1]
                        z = mot_tracker.trackers[t_idx].z_s
                        w = mot_tracker.trackers[t_idx].height
                        l = mot_tracker.trackers[t_idx].length
                        h = mot_tracker.trackers[t_idx].width
                        theta = mot_tracker.trackers[t_idx].X[2]
                        confid_bg = mot_tracker.trackers[t_idx].confid[0]
                        confid_car= mot_tracker.trackers[t_idx].confid[1]
                        confid_ped= mot_tracker.trackers[t_idx].confid[2]
                        confid_cyc= mot_tracker.trackers[t_idx].confid[3]
                        f.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f, \n'%(trk_type,
                                     x, y, z, l, w, h, theta, confid_bg, confid_car, confid_ped, confid_cyc))

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
                    ax2.text(bb[0], bb[1], 'B:%0.2f,C:%0.2f,P:%0.2f,CY:%0.2f'%(np.float(det[8]), np.float(det[9]), np.float(det[10]), np.float(det[11])), color=[1,0,0], fontsize=8)
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

def pointcloud_tracking_within_one_range_with_fusion_multiple(config_path=None,
                                                              output_dir =None,
                                                              display  = False,
                                                              display_trajectory = False,
                                                              save_trk_results = True,
                                                              save_det_results = False,
                                                              save_all_track_det_confidence=True,
                                                              phases = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0020_sync',
                                                                        '2011_09_26_drive_0035_sync','2011_09_26_drive_0084_sync',
                                                                        '2011_09_26_drive_0005_sync','2011_09_26_drive_0014_sync',
                                                                        '2011_09_26_drive_0019_sync','2011_09_26_drive_0059_sync',]):
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

    for phase in phases:
        print("Phase name: {}".format(phase))
        config.dataset.phase = phase
        __pointcloud_tracking__fusion__(config, display=display, display_trajectory=display_trajectory,
                                        save_trk_results=save_trk_results, save_det_results=save_det_results,
                                        save_all_track_det_confidence=save_all_track_det_confidence)

def pointcloud_tracking_within_one_range_with_fusion_bk(config_path=None,
                                                     output_dir =None,
                                                     display  = False,
                                                     display_trajectory = False,
                                                     save_trk_results = True,
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

        # save the tracking results
        if save_trk_results:
            print('save_results in #{} tracker'.format(i))
            result_folder = Path(dataset_config.database_dir)/dataset_config.phase/'detection/dets_trk'
            result_folder.mkdir(parents=True, exist_ok=True)
            result_file_path = str(Dataset.det_gt_list[i]).replace('/dets_gt','/dets_trk')

            with open(result_file_path, 'w') as f:
                for t_idx in range(len(mot_tracker.trackers)):
                    cc = np.int(mot_tracker.trackers[t_idx].category)
                    if cc >= label_to_num.unknow_object_label:
                        cc = cc - label_to_num.unknow_object_label
                    trk_type = class_type[cc]
                    x = mot_tracker.trackers[t_idx].X[0]
                    y = mot_tracker.trackers[t_idx].X[1]
                    z = mot_tracker.trackers[t_idx].z_s
                    w = mot_tracker.trackers[t_idx].height
                    l = mot_tracker.trackers[t_idx].length
                    h = mot_tracker.trackers[t_idx].width
                    theta = mot_tracker.trackers[t_idx].X[2]
                    confid_bg = mot_tracker.trackers[t_idx].confid[0]
                    confid_car= mot_tracker.trackers[t_idx].confid[1]
                    confid_ped= mot_tracker.trackers[t_idx].confid[2]
                    confid_cyc= mot_tracker.trackers[t_idx].confid[3]
                    f.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f, \n'%(trk_type,
                                 x, y, z, l, w, h, theta, confid_bg, confid_car, confid_ped, confid_cyc))

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
                    ax2.text(bb[0], bb[1], 'B:%0.2f,C:%0.2f,P:%0.2f,CY:%0.2f'%(np.float(det[8]), np.float(det[9]), np.float(det[10]), np.float(det[11])), color=[1,0,0], fontsize=8)
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


### function testing and tuning for tracking
def __pointcloud_tracking__fusion_tracker_shaking_test__(config,
                                    display  = False,
                                    display_trajectory = False,
                                    save_trk_results = False,
                                    save_det_results = True,
                                    save_all_track_det_confidence=True,
                                    fusion_confidence  = 0.96,
                                    downsample_num = 400,):
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

    # save the history of confidence change with respect to detector and
    # trackers.
    if save_all_track_det_confidence:
        result_trk_folder = Path(dataset_config.database_dir)/dataset_config.phase/'detection/dets_trk_confidence'
        result_trk_folder.mkdir(parents=True, exist_ok=True)


    # initializatioN
    Dataset = Kitti_dataset(dataset_config)
    mot_tracker = Sort_3d(config=tracker_config, data_association=filter_config.data_association,
                          fusion_confidence=fusion_confidence, result_trk_folder=result_trk_folder)

    # figure initialization
    if display: fig,ax1,ax2, cmap = fig_initialization()
    if display_trajectory: plt.ion();fig2=plt.figure(num=2, figsize=(12,11));ax2_1=fig2.add_subplot(1,1,1);


    # read calibration data
    calib_data_velo_2_cam = Dataset.calib_data_velo_2_cam
    calib_data_cam_2_cam = Dataset.calib_data_cam_2_cam
    robot_poses = []
    global_maps = None

    ## plotting position with detector............
    fig, ax = plt.subplots()
    all_tracking_ids = []
    distance_streams = {}
    frame_streams = {}
    # tracking
    num_full_detection  = 0
    num_hybrid_detection  = 0
    for i in range(0, Dataset.__len__()):
        # robot positions
        dets = Dataset.get_detection_class(i)
        if len(dets)==0:
            continue
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

        x_pos       = global_points[0,1:]
        y_pos       = global_points[1,1:]
        p_range = np.sqrt(y_pos**2 + x_pos**2)
        tracking_id = dets[:,13]

        for j in range(tracking_id.shape[0]):
            _id_ = int(tracking_id[j])
            if _id_ not in all_tracking_ids:
                all_tracking_ids.append(_id_)
                distance_streams[_id_] = [p_range[j]]
                frame_streams[_id_] = [i]
            else:
                distance_streams[_id_].append(p_range[j])
                frame_streams[_id_].append(i)
    for i in all_tracking_ids:
        plt.plot(frame_streams[i], distance_streams[i])

    plt.xlabel('Number of frames')
    plt.ylabel('distance detector')
    fig_name = 'utils/shaking/{}_detector.png'.format(config.dataset.phase)
    plt.savefig(fig_name)


    # plotting the details with tracking ....................
    fig, ax = plt.subplots()
    all_tracking_ids = []
    distance_streams = {}
    frame_streams = {}
    velocity_streams = {}
    # tracking
    num_full_detection  = 0
    num_hybrid_detection  = 0
    for i in range(0, len(Dataset.det_tracklet_det_list)):
        # robot positions
        start_frame_id = int(Dataset.det_tracklet_det_list[i].name.split('_')[1].split('.')[0])
        dets = Dataset.get_detection_tracklet_det(i)
        if len(dets)==0:
            continue

        dets = np.array(dets, dtype=np.float32)
        # local_points[:,0] is position for robot
        tracking_id = dets[:,8]
        velocities = dets[:,11]

        frame_streams[i] = [start_frame_id]

        velocity_streams[i] = [velocities[0]]
        p_range = np.sqrt(dets[0,9]**2 + dets[0,10]**2)
        distance_streams[i] = [p_range]

        for j in range(1, tracking_id.shape[0]):
            # _id_ = int(tracking_id[j])
            frame_streams[i].append( start_frame_id-j )

            velocity_streams[i].append(velocities[j])
            p_range = np.sqrt(dets[j,9]**2 + dets[j,10]**2)
            distance_streams[i].append(p_range)
            # if _id_ not in all_tracking_ids:
            #     all_tracking_ids.append(_id_)
            #     # distance_streams[_id_] = [p_range[j]]
            #     frame_streams[_id_] = [start_frame_id]
            #     velocity_streams[_id_] = [velocities[j]]

            #     local_points = np.zeros((4, 1), dtype=np.float32)
            #     local_points[0,0] = dets[j,9]
            #     local_points[1,0] = dets[j,10]
            #     local_points[2,0] = -1.63
            #     local_points[3,0] = 1
            #     #if frame_streams[_id_][-1]>=len(Dataset.pose):
            #     #    global_points = Dataset.pose[-1].dot(local_points)
            #     #else:
            #     #    global_points = Dataset.pose[frame_streams[_id_][-1]].dot(local_points)
            #     # p_range = np.sqrt(global_points[0,0]**2 + global_points[1,0]**2)
            #     p_range = np.sqrt(local_points[0,0]**2 + local_points[1,0]**2)
            #     distance_streams[_id_] = [p_range]

            # else:
            #     # distance_streams[_id_].append(p_range[j])
            #     frame_streams[_id_].append( frame_streams[_id_][-1]-1)
            #     velocity_streams[_id_].append(velocities[j])
            #     # calculate the pose

            #     local_points = np.zeros((4, 1), dtype=np.float32)
            #     local_points[0,0] = dets[j,9]
            #     local_points[1,0] = dets[j,10]
            #     local_points[2,0] = -1.63
            #     local_points[3,0] = 1
            #     # if frame_streams[_id_][-1]>=len(Dataset.pose):
            #     #     global_points = Dataset.pose[-1].dot(local_points)
            #     # else:
            #     #     global_points = Dataset.pose[frame_streams[_id_][-1]].dot(local_points)
            #     # p_range = np.sqrt(global_points[0,0]**2 + global_points[1,0]**2)
            #     p_range = np.sqrt(local_points[0,0]**2 + local_points[1,0]**2)
            #     distance_streams[_id_].append(p_range)

    # for i in all_tracking_ids:
    #     plt.plot(frame_streams[i].reverse(), distance_streams[i])
    for i in range(len(distance_streams)):
        plt.plot(frame_streams[i][::-1], distance_streams[i])
    plt.xlabel('Number of frames')
    plt.ylabel('distance estimation')
    fig_name = 'utils/shaking/{}_position_estimation.png'.format(config.dataset.phase)
    plt.savefig(fig_name)


    # plotting the details with tracking ....................
    fig, ax = plt.subplots()
    for i in range(len(distance_streams)):
        plt.plot(frame_streams[i][::-1], distance_streams[i])
        # adding the speed
        plt.plot(frame_streams[i][::-1], distance_streams[i][0]+velocity_streams[i],Color='y')

    plt.xlabel('Number of frames')
    plt.ylabel('distance estimation')
    fig_name = 'utils/shaking/{}_position_with_velocity_estimation.png'.format(config.dataset.phase)
    plt.savefig(fig_name)



    # plotting the details with tracking ....................
    fig, ax = plt.subplots()
    for i in range(len(distance_streams)):# all_tracking_ids:
        plt.plot(frame_streams[i][::-1], velocity_streams[i])

    plt.xlabel('Number of frames')
    plt.ylabel('Velocity')
    fig_name = 'utils/shaking/{}_velocity.png'.format(config.dataset.phase)
    plt.savefig(fig_name)


def pointcloud_tracking_within_one_range_with_fusion_multiple_shaking_testing(config_path='/home/ben/projects/tracking/hbtk/config/kitti_tracking.config',
                                                              output_dir =None,
                                                              display  = False,
                                                              display_trajectory = False,
                                                              save_trk_results = True,
                                                              save_det_results = False,
                                                              save_all_track_det_confidence=True,
                                                              phases = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0020_sync',
                                                                        '2011_09_26_drive_0035_sync','2011_09_26_drive_0084_sync',
                                                                        '2011_09_26_drive_0005_sync','2011_09_26_drive_0014_sync',
                                                                        '2011_09_26_drive_0019_sync','2011_09_26_drive_0059_sync',]):
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

    for phase in phases:
        print("Phase name: {}".format(phase))
        config.dataset.phase = phase
        __pointcloud_tracking__fusion_tracker_shaking_test__(config, display=display, display_trajectory=display_trajectory,
                                                             save_trk_results=save_trk_results, save_det_results=save_det_results,
                                                             save_all_track_det_confidence=save_all_track_det_confidence)



if __name__=='__main__':
    fire.Fire()


