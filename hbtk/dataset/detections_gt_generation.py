"""
    File Description:
        Generate gt results of efficient detection with known ground truth objects, so the all classification
        confidence for object is 1
    Porcedures:
        1. save the foreground objects firstly;
        2. crop off the points of these foreground objects;
        3. run the segmentation algorithm;
        4. save the background segments;

"""
import pykitti
import fire
import os
import sys
import open3d

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(ROOT_DIR))
sys.path.append(os.path.join( os.path.dirname(ROOT_DIR), 'hbtk'))
#sys.path.append(os.path.join( os.path.dirname(os.path.dirname(ROOT_DIR)), 'hbtk', 'protos'))


from google.protobuf import text_format
from hbtk.protos import pipeline_pb2, pipeline_det_pb2
from hbtk.dataset.kitti_dataset import Kitti_dataset
from hbtk.utils import box_np_ops
from pathlib import Path
from source import parseTrackletXML as xmlParser
import numpy as np
from skimage import io
from hbtk.detectors.efficient_det.cluster_m_jit import PointCluster
from hbtk.detectors.efficient_det.points_filtering import PointFilter
from hbtk.detectors.efficient_det.clusters_fltering_jit import ClustersFiltering
from hbtk.detectors.efficient_det.clusters_orientation import find_optimal_bbox3d
from hbtk.detectors.efficient_det.classificaiton_pointnet  import Classification_Pointnet
from hbtk.utils.visualization_pc import *


def display_one_pc(points):
    pcl = open3d.PointCloud()
    pcl.points = open3d.Vector3dVector(points[:,:3])
    open3d.draw_geometries([pcl])

def read_detection_file(one_file):
    if isinstance(one, str):
        file_path = one_file
    else:
        file_path = str(one_file)
    with open(file_path, 'r') as f:
        lines = f.readlines()
    _dets_ = [line.strip().split(',') for line in lines]
    for _det in _dets_:
        if _det[0] in ['Car', 'Pedestrian', 'Cyclist'] :
            if _det[0] in ['Car'] :
                _det[0] = 0
            if _det[0] in ['Pedestrian'] :
                _det[0] = 1
            if _det[0] in ['Cyclist'] :
                _det[0] = 2
            dets.append(_det)

    return dets

def points_inside_box(points, xyzlwh):
    x,y,z,l,w,h = xyzlwh[0], xyzlwh[1], xyzlwh[2], xyzlwh[3], xyzlwh[4], xyzlwh[5]
    x_up   = x + l/2.
    x_down = x - l/2.
    y_up   = y + w/2.
    y_down = y - w/2.
    z_down = z

    indices = np.logical_and.reduce((points[:,0]<x_up, points[:,0]>x_down,
                                     points[:,1]<y_up, points[:,1]>y_down,
                                     points[:,2]>z_down))
    return indices

def create_reduced_point_cloud(config_path):
    # read configuration file
    config = pipeline_pb2.TrackingPipeline()
    with open(config_path, "r") as f:
        protos_str = f.read()
        text_format.Merge(protos_str, config)
    #shutil.copyfile(config_path, str(output_dir+"/"+"pipeline.config"))
    detector_config = config.detector
    filter_config = config.filter
    tracker_config = config.tracker
    dataset_config = config.dataset

    output_folder_dir = Path(dataset_config.database_dir)/dataset_config.phase/'reduced_points/data'
    output_folder_dir.mkdir(parents=True, exist_ok=True)

    Dataset = Kitti_dataset(dataset_config)
    calib_data_velo_2_cam = Dataset.calib_data_velo_2_cam
    calib_data_cam_2_cam = Dataset.calib_data_cam_2_cam

    print("there are {} files".format(Dataset.__len__()))
    for i in range(0, Dataset.__len__()):
        points_v = np.fromfile(str(Dataset.pc_list[i]), dtype=np.float32, count=-1).reshape([-1, 4])
        # read the calibration info
        _r_ = calib_data_velo_2_cam['R'].reshape(3,3)
        _t_ = calib_data_velo_2_cam['T'].reshape(3,1)
        Trv2c = np.vstack((np.hstack([_r_, _t_]), [0, 0, 0, 1]))
        # Trev2c = np.hstack([_r_, _t_])
        Rect = calib_data_cam_2_cam['R_rect_02'].reshape(3,3)
        Rect = np.vstack((np.hstack((Rect, [[0],[0],[0]])), [0, 0, 0, 1]))
        P2 = calib_data_cam_2_cam['P_rect_02'].reshape(3,4)
        P2 = np.vstack((P2, [0, 0, 0, 1]))
        # image read
        img = np.array(io.imread(str(Dataset.img_list[i])), dtype=np.int32)
        img_shape = img.shape

        points_v = box_np_ops.remove_outside_points(points_v, Rect, Trv2c, P2,img_shape)

        points_v = points_v.astype(np.float32)

        save_filename = output_folder_dir/str(Dataset.pc_list[i].name)
        with open(str(save_filename), 'w') as f:
            points_v.tofile(f)

    print("Finish!!!!")

def create_negative_samples(tracking_config_path,
                            detection_config_path,):
    """
    1 prepare the path for all the file, point, img and calibiration
    2 read the ground truth and add them to dets
    3 crop off points of foreground objects
    4 read the image and calibration, crop all the point within the image view
    5 run segmentation algorithm to save the background objects
    """
    #  ********* tracking configuration ***********
    # read configuration file
    config = pipeline_pb2.TrackingPipeline()
    with open(tracking_config_path, "r") as f:
        protos_str = f.read()
        text_format.Merge(protos_str, config)
    #shutil.copyfile(config_path, str(output_dir+"/"+"pipeline.config"))
    detector_config = config.detector
    filter_config = config.filter
    tracker_config = config.tracker
    dataset_config = config.dataset

    output_folder_dir = Path(dataset_config.database_dir)/dataset_config.phase/'negative_samples'
    output_folder_dir.mkdir(parents=True, exist_ok=True)

    object_folder_dir = Path(dataset_config.database_dir)/dataset_config.phase/'object_samples'
    object_folder_dir.mkdir(parents=True, exist_ok=True)

    Dataset = Kitti_dataset(dataset_config)

    # *************** detection configuration **********
    config = pipeline_det_pb2.DetectionPipeline()
    with open(detection_config_path, "r") as f:
        protos_str = f.read()
        text_format.Merge(protos_str, config)
    points_filter   = PointFilter(config.pointsfilter)
    points_cluster  = PointCluster(config.pointscluster)
    clusters_filter = ClustersFiltering(config.clustersfilter)



    # ********* inialization *************
    for i in range(0, Dataset.__len__()):
        print("Block No.{}".format(i))
        object_types = []
        xyz_lwhr_confid = []
        object_points_clusters = []

        # reading detections
        dets = Dataset.get_gt(i) #read detections
        dets = np.array(dets, dtype=np.float32)
        # read cropped points
        points = np.fromfile(str(Dataset.reduce_pc_list[i]),dtype=np.float32, count=-1).reshape([-1, 4])
        for j in range(0, dets.shape[0]):
            # xyzlwh = dets[j][1:]
            large_ratio = 1.2
            xyzlwhr_lidar = np.array([[dets[j][1], dets[j][2], dets[j][3]+0.1, dets[j][4]*large_ratio, dets[j][5]*large_ratio, dets[j][6]*1.5, -dets[j][7]]])
            # indices  = points_inside_box(points, xyzlwhr, axis=2, origin=(0.5,0.5,0))
            # get points within the 3D bounding boxes, xyzlwhr_lidar
            indices = box_np_ops.points_in_rbbox(points, xyzlwhr_lidar, z_axis=2, origin=(0.5,0.5,0))
            object_points = points[indices[:,0],:]
            # running the classificaiton model to generate PDF
            object_points_clusters.append(object_points)
            if object_points.shape[0]==0:
                continue

            # read the class label
            j_type = int(dets[j][0])
            if j_type == 1:
                object_types.append('Car')
            if j_type == 2:
                object_types.append('Pedestrian')
            if j_type == 3:
                object_types.append('Cyclist')


            filename = '%4d_'%(Dataset.__len__()) + '%s'%object_types[-1] + '_%d%d'%(i,j)+'.bin'
            filepath = object_folder_dir/filename
            with open(filepath, 'w') as f:
                object_points.tofile(f)


            # read the size of bounding box
            _x_min = min(object_points[:,0])
            _x_max = max(object_points[:,0])
            _y_min = min(object_points[:,1])
            _y_max = max(object_points[:,1])
            _z_min = min(object_points[:,2])
            _z_max = max(object_points[:,2])
            _x_c = (_x_min+_x_max)/2
            _y_c = (_y_min+_y_max)/2
            _xy_c = object_points[:,:2] - np.array([[_x_c, _y_c]])
            heading, wl = find_optimal_bbox3d(_xy_c)
            # saving the results
            confidence = 1.0
            # heading = dets[j][7]
            xyz_lwhr_confid.append([ _x_c, _y_c, _z_min, wl[1], wl[0], dets[j][6], heading, confidence])
            # remove the objectness points
            points = points[~indices[:,0],:]

        # do efficient detection
        road_parameter_path = str(Dataset.gt_list[i]).replace('detection/gt','ground_plane/data')
        points = points_filter._filtering_(points, road_parameter_path)
        clusters, num_points = points_cluster._cluster_(points)
        clusters = clusters_filter._clusterfiltering_(clusters)
        # save the background segments
        if True or i%2==0:
            for k in range(len(clusters)):
                bg_points = clusters[k]
                file_name = '%d'%(Dataset.__len__()) +'%04d'%i+'_%d'%k+'.bin'
                file_path = output_folder_dir/file_name
                with open(file_path, 'w') as f:
                    bg_points.tofile(f)

def save_detection_gt(tracking_config_path='/home/ben/projects/tracking/hbtk/config/kitti_tracking.config',
                      detection_config_path='/home/ben/projects/tracking/hbtk/config/detection.config',):
    """
    1 prepare the path for all the file, point, img and calibiration
    2 read the ground truth and add them to dets
    3 crop off points of foreground objects
    4 read the image and calibration, crop all the point within the image view
    5 run segmentation algorithm to save the background objects
    """
    #  ********* tracking configuration ***********
    # read configuration file
    config = pipeline_pb2.TrackingPipeline()
    with open(tracking_config_path, "r") as f:
        protos_str = f.read()
        text_format.Merge(protos_str, config)
    #shutil.copyfile(config_path, str(output_dir+"/"+"pipeline.config"))
    detector_config = config.detector
    filter_config = config.filter
    tracker_config = config.tracker
    dataset_config = config.dataset

    output_folder_dir = Path(dataset_config.database_dir)/dataset_config.phase/'detection/dets_gt'
    output_folder_dir.mkdir(parents=True, exist_ok=True)

    Dataset = Kitti_dataset(dataset_config)

    # *************** detection configuration **********
    config = pipeline_det_pb2.DetectionPipeline()
    with open(detection_config_path, "r") as f:
        protos_str = f.read()
        text_format.Merge(protos_str, config)
    points_filter   = PointFilter(config.pointsfilter)
    points_cluster  = PointCluster(config.pointscluster)
    clusters_filter = ClustersFiltering(config.clustersfilter)



    # ********* inialization *************
    for i in range(0, Dataset.__len__()):
        print("Block No.{}".format(i))
        object_types = []
        xyz_lwhr_confid = []
        object_points_clusters = []

        # reading detections
        dets = Dataset.get_gt(i) #read detections
        dets = np.array(dets, dtype=np.float32)
        # read cropped points
        points = np.fromfile(str(Dataset.reduce_pc_list[i]),dtype=np.float32, count=-1).reshape([-1, 4])
        for j in range(0, dets.shape[0]):
            # xyzlwh = dets[j][1:]
            large_ratio = 1.2
            xyzlwhr_lidar = np.array([[dets[j][1], dets[j][2], dets[j][3]+0.1, dets[j][4]*large_ratio, dets[j][5]*large_ratio, dets[j][6]*1.5, -dets[j][7]]])
            # indices  = points_inside_box(points, xyzlwhr, axis=2, origin=(0.5,0.5,0))
            # get points within the 3D bounding boxes, xyzlwhr_lidar
            indices = box_np_ops.points_in_rbbox(points, xyzlwhr_lidar, z_axis=2, origin=(0.5,0.5,0))
            object_points = points[indices[:,0],:]
            # running the classificaiton model to generate PDF
            object_points_clusters.append(object_points)
            if object_points.shape[0]==0:
                continue
            # read the class label
            j_type = int(dets[j][0])
            if j_type == 1:
                object_types.append('Car')
            if j_type == 2:
                object_types.append('Pedestrian')
            if j_type == 3:
                object_types.append('Cyclist')
            # read the size of bounding box
            _x_min = min(object_points[:,0])
            _x_max = max(object_points[:,0])
            _y_min = min(object_points[:,1])
            _y_max = max(object_points[:,1])
            _z_min = min(object_points[:,2])
            _z_max = max(object_points[:,2])
            _x_c = (_x_min+_x_max)/2
            _y_c = (_y_min+_y_max)/2
            _xy_c = object_points[:,:2] - np.array([[_x_c, _y_c]])
            heading, wl = find_optimal_bbox3d(_xy_c)
            # saving the results
            confidence = 1.0
            # heading = dets[j][7]
            xyz_lwhr_confid.append([ _x_c, _y_c, _z_min, wl[1], wl[0], dets[j][6], heading, confidence])
            # remove the objectness points
            points = points[~indices[:,0],:]

        # do efficient detection
        road_parameter_path = str(Dataset.gt_list[i]).replace('detection/gt','ground_plane/data')
        points = points_filter._filtering_(points, road_parameter_path)
        clusters, num_points = points_cluster._cluster_(points)
        clusters = clusters_filter._clusterfiltering_(clusters)
        # save the background segments
        for k in range(len(clusters)):
            bg_points = clusters[k]
            _x_min = min(bg_points[:,0])
            _x_max = max(bg_points[:,0])
            _y_min = min(bg_points[:,1])
            _y_max = max(bg_points[:,1])
            _z_min = min(bg_points[:,2])
            _z_max = max(bg_points[:,2])
            # get the bounding boxes information
            _x_c = (_x_min+_x_max)/2
            _y_c = (_y_min+_y_max)/2
            _xy_c = bg_points[:,:2] - np.array([[_x_c, _y_c]])
            heading, wl = find_optimal_bbox3d(_xy_c)
            confidence = 1.0
            # heading = 0
            xyz_lwhr_confid.append([ _x_c, _y_c, _z_min, wl[1], wl[0], 1.1*(_z_max - _z_min), heading, confidence])
            object_types.append('Bg')

        # save the segment results
        result_file_path = str(Dataset.gt_list[i]).replace('/gt/','/dets_gt/')
        with open(result_file_path, 'w') as f:
            for m in range(len(object_types)):
                ty = object_types[m]
                x     = xyz_lwhr_confid[m][0]
                y     = xyz_lwhr_confid[m][1]
                z     = xyz_lwhr_confid[m][2]
                l     = xyz_lwhr_confid[m][3]
                w     = xyz_lwhr_confid[m][4]
                h     = xyz_lwhr_confid[m][5]
                theta = xyz_lwhr_confid[m][6]
                confid= xyz_lwhr_confid[m][7]
                f.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f \n'%(ty, x, y, z, l, w, h, theta, confid))

def save_detection_classification(tracking_config_path,
                                  detection_config_path,):
    """
    1 prepare the path for all the file, point, img and calibiration
    2 read the ground truth and add them to dets
    3 crop off points of foreground objects, run classification model to generate PDF
    4 read the image and calibration, crop all the point within the image view
    5 run segmentation algorithm to save the background objects
    """
    #  ********* tracking configuration ***********
    # read configuration file
    config = pipeline_pb2.TrackingPipeline()
    with open(tracking_config_path, "r") as f:
        protos_str = f.read()
        text_format.Merge(protos_str, config)
    #shutil.copyfile(config_path, str(output_dir+"/"+"pipeline.config"))
    detector_config = config.detector
    filter_config = config.filter
    tracker_config = config.tracker
    dataset_config = config.dataset

    output_folder_dir = Path(dataset_config.database_dir)/dataset_config.phase/'detection/dets_class'
    output_folder_dir.mkdir(parents=True, exist_ok=True)

    Dataset = Kitti_dataset(dataset_config)

    # *************** detection configuration **********
    config = pipeline_det_pb2.DetectionPipeline()
    with open(detection_config_path, "r") as f:
        protos_str = f.read()
        text_format.Merge(protos_str, config)
    points_filter = PointFilter(config.pointsfilter)
    points_cluster = PointCluster(config.pointscluster)
    clusters_filter = ClustersFiltering(config.clustersfilter)
    classifier      = Classification_Pointnet(config.clustertodetection)

    # ********* inialization *************
    for i in range(0, Dataset.__len__()):
        print("Block No.{}".format(i))
        object_types = []
        xyz_lwhr_confid = []
        object_points_clusters = []

        # reading detections
        dets = Dataset.get_gt(i) #read detections
        dets = np.array(dets, dtype=np.float32)
        # read cropped points
        points = np.fromfile(str(Dataset.reduce_pc_list[i]),dtype=np.float32, count=-1).reshape([-1, 4])
        for j in range(0, dets.shape[0]):
            # xyzlwh = dets[j][1:]
            large_ratio = 1.2
            xyzlwhr_lidar = np.array([[dets[j][1], dets[j][2], dets[j][3]+0.1, dets[j][4]*large_ratio, dets[j][5]*large_ratio, dets[j][6]*1.5, -dets[j][7]]])
            # indices  = points_inside_box(points, xyzlwhr, axis=2, origin=(0.5,0.5,0))
            # get points within the 3D bounding boxes, xyzlwhr_lidar
            indices = box_np_ops.points_in_rbbox(points, xyzlwhr_lidar, z_axis=2, origin=(0.5,0.5,0))
            object_points = points[indices[:,0],:]
            object_points_clusters.append(object_points)
            if object_points.shape[0]==0:
                continue
            # read the class label
            j_type = int(dets[j][0])
            if j_type == 1:
                object_types.append('Car')
            if j_type == 2:
                object_types.append('Pedestrian')
            if j_type == 3:
                object_types.append('Cyclist')
            # read the size of bounding box
            _x_min = min(object_points[:,0])
            _x_max = max(object_points[:,0])
            _y_min = min(object_points[:,1])
            _y_max = max(object_points[:,1])
            _z_min = min(object_points[:,2])
            _z_max = max(object_points[:,2])
            _x_c = (_x_min+_x_max)/2
            _y_c = (_y_min+_y_max)/2
            _xy_c = object_points[:,:2] - np.array([[_x_c, _y_c]])
            heading, wl = find_optimal_bbox3d(_xy_c)
            # saving the results
            confidence = classifier.classification(object_points)
            # heading = dets[j][7]
            xyz_lwhr_confid.append([ _x_c, _y_c, _z_min, wl[1], wl[0], dets[j][6], heading] + confidence)
            # remove the objectness points
            points = points[~indices[:,0],:]

        # do efficient detection
        road_parameter_path = str(Dataset.gt_list[i]).replace('detection/gt','ground_plane/data')
        points = points_filter._filtering_(points, road_parameter_path)
        clusters, num_points = points_cluster._cluster_(points)
        clusters = clusters_filter._clusterfiltering_(clusters)
        # save the background segments
        for k in range(len(clusters)):
            bg_points = clusters[k]
            _x_min = min(bg_points[:,0])
            _x_max = max(bg_points[:,0])
            _y_min = min(bg_points[:,1])
            _y_max = max(bg_points[:,1])
            _z_min = min(bg_points[:,2])
            _z_max = max(bg_points[:,2])
            # get the bounding boxes information
            _x_c = (_x_min+_x_max)/2
            _y_c = (_y_min+_y_max)/2
            _xy_c = bg_points[:,:2] - np.array([[_x_c, _y_c]])
            heading, wl = find_optimal_bbox3d(_xy_c)
            # confidence = 1.0
            confidence = classifier.classification(bg_points)
            # heading = 0
            xyz_lwhr_confid.append([ _x_c, _y_c, _z_min, wl[1], wl[0], 1.1*(_z_max - _z_min), heading]+confidence)
            object_types.append('Bg')

        # save the segment results
        result_file_path = str(Dataset.gt_list[i]).replace('/gt/','/dets_class/')
        with open(result_file_path, 'w') as f:
            for m in range(len(object_types)):
                ty = object_types[m]
                x     = xyz_lwhr_confid[m][0]
                y     = xyz_lwhr_confid[m][1]
                z     = xyz_lwhr_confid[m][2]
                l     = xyz_lwhr_confid[m][3]
                w     = xyz_lwhr_confid[m][4]
                h     = xyz_lwhr_confid[m][5]
                theta = xyz_lwhr_confid[m][6]
                confid_bg =  xyz_lwhr_confid[m][7]
                confid_car = xyz_lwhr_confid[m][8]
                confid_ped = xyz_lwhr_confid[m][9]
                # confid_van = xyz_lwhr_confid[m][10]
                confid_cyc = xyz_lwhr_confid[m][11]

                # f.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f, \n'%(ty,
                #               x, y, z, l, w, h, theta, confid_bg, confid_car, confid_ped, confid_van, confid_cyc))
                f.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f, \n'%(ty,
                               x, y, z, l, w, h, theta, confid_bg, confid_car, confid_ped, confid_cyc))


def __save_detection_classification_one_phases( tracking_config,
                                                detection_config_path,
                                                bg_id = 100000. ):
    """
    1 prepare the path for all the file, point, img and calibiration
    2 read the ground truth and add them to dets
    3 crop off points of foreground objects, run classification model to generate PDF
    4 read the image and calibration, crop all the point within the image view
    5 run segmentation algorithm to save the background objects
    """
    #  ********* tracking configuration ***********
    detector_config = tracking_config.detector
    filter_config   = tracking_config.filter
    tracker_config  = tracking_config.tracker
    dataset_config  = tracking_config.dataset

    output_folder_dir = Path(dataset_config.database_dir)/dataset_config.phase/'detection/dets_class'
    output_folder_dir.mkdir(parents=True, exist_ok=True)

    Dataset = Kitti_dataset(dataset_config)

    # *************** detection configuration **********
    config = pipeline_det_pb2.DetectionPipeline()
    with open(detection_config_path, "r") as f:
        protos_str = f.read()
        text_format.Merge(protos_str, config)
    points_filter = PointFilter(config.pointsfilter)
    points_cluster = PointCluster(config.pointscluster)
    clusters_filter = ClustersFiltering(config.clustersfilter)
    classifier      = Classification_Pointnet(config.clustertodetection)

    num_car = 0
    num_ped = 0
    num_cyc = 0
    # ********* inialization *************
    for i in range(0, Dataset.__len__()):
        #print("Block No.{}".format(i))
        object_types = []
        # xyz_lwhr_confid = []
        xyz_lwhr_confid_id = []
        object_points_clusters = []

        # reading detections
        dets = Dataset.get_gt(i) #read detections
        dets = np.array(dets, dtype=np.float32)
        # read cropped points
        points = np.fromfile(str(Dataset.reduce_pc_list[i]),dtype=np.float32, count=-1).reshape([-1, 4])
        # ***************************************************
        # generate the ground truth bounding boxes
        for j in range(0, dets.shape[0]):
            # xyzlwh = dets[j][1:]
            large_ratio = 1.2 # 1.2
            xyzlwhr_lidar = np.array([[dets[j][1], dets[j][2], dets[j][3]+0.1, dets[j][4]*large_ratio, dets[j][5]*large_ratio, dets[j][6]*1.5, -dets[j][7]]])
            # indices  = points_inside_box(points, xyzlwhr, axis=2, origin=(0.5,0.5,0))
            # get points within the 3D bounding boxes, xyzlwhr_lidar
            indices = box_np_ops.points_in_rbbox(points, xyzlwhr_lidar, z_axis=2, origin=(0.5,0.5,0))
            object_points = points[indices[:,0],:]
            object_points_clusters.append(object_points)
            _num_points = object_points.shape[0]
            if object_points.shape[0]==0:
                print("one empty")
                continue
            # read the class label
            j_type = int(dets[j][0])
            if j_type == 1:
                object_types.append('Car')
            if j_type == 2:
                object_types.append('Pedestrian')
            if j_type == 3:
                object_types.append('Cyclist')
            # read the size of bounding box
            _x_min = min(object_points[:,0])
            _x_max = max(object_points[:,0])
            _y_min = min(object_points[:,1])
            _y_max = max(object_points[:,1])
            _z_min = min(object_points[:,2])
            _z_max = max(object_points[:,2])
            _x_c = (_x_min+_x_max)/2
            _y_c = (_y_min+_y_max)/2
            _xy_c = object_points[:,:2] - np.array([[_x_c, _y_c]])
            heading, wl = find_optimal_bbox3d(_xy_c)
            # saving the results
            confidence = classifier.classification(object_points)
            # heading = dets[j][7]
            # xyz_lwhr_confid.append([ _x_c, _y_c, _z_min, wl[1], wl[0], dets[j][6], heading] + confidence+[_num_points])
            xyz_lwhr_confid_id.append([ _x_c, _y_c, _z_min, wl[1], wl[0], dets[j][6], heading] + confidence+[_num_points] + [dets[j][8]])
            # remove the objectness points
            # points = points[~indices[:,0],:]
        num_car += object_types.count('Car')
        num_ped += object_types.count('Pedestrian')
        num_cyc += object_types.count('Cyclist')
        # **************************************
        # removing the object points
        for j in range(0, dets.shape[0]):
            # xyzlwh = dets[j][1:]
            large_ratio = 1.2 # 1.2
            xyzlwhr_lidar = np.array([[dets[j][1], dets[j][2], dets[j][3]+0.1, dets[j][4]*large_ratio, dets[j][5]*large_ratio, dets[j][6]*1.5, -dets[j][7]]])
            # indices  = points_inside_box(points, xyzlwhr, axis=2, origin=(0.5,0.5,0))
            # get points within the 3D bounding boxes, xyzlwhr_lidar
            indices = box_np_ops.points_in_rbbox(points, xyzlwhr_lidar, z_axis=2, origin=(0.5,0.5,0))
            obddject_points = points[indices[:,0],:]
            if object_points.shape[0]==0:
                continue
            # remove the objectness points
            points = points[~indices[:,0],:]

        # do efficient detection
        road_parameter_path = str(Dataset.gt_list[i]).replace('detection/gt','ground_plane/data')
        points = points_filter._filtering_(points, road_parameter_path)
        clusters, num_points = points_cluster._cluster_(points)
        clusters = clusters_filter._clusterfiltering_(clusters)
        # save the background segments
        for k in range(len(clusters)):
            bg_points = clusters[k]
            _num_points = bg_points.shape[0]
            _x_min = min(bg_points[:,0])
            _x_max = max(bg_points[:,0])
            _y_min = min(bg_points[:,1])
            _y_max = max(bg_points[:,1])
            _z_min = min(bg_points[:,2])
            _z_max = max(bg_points[:,2])
            # get the bounding boxes information
            _x_c = (_x_min+_x_max)/2
            _y_c = (_y_min+_y_max)/2
            _xy_c = bg_points[:,:2] - np.array([[_x_c, _y_c]])
            heading, wl = find_optimal_bbox3d(_xy_c)
            # confidence = 1.0
            confidence = classifier.classification(bg_points)
            # heading = 0
            # xyz_lwhr_confid.append([ _x_c, _y_c, _z_min, wl[1], wl[0], 1.1*(_z_max - _z_min), heading]+confidence+[_num_points])
            xyz_lwhr_confid_id.append([ _x_c, _y_c, _z_min, wl[1], wl[0], 1.1*(_z_max - _z_min), heading]+confidence+[_num_points]+[bg_id])
            object_types.append('Bg')

        # save the segment results
        result_file_path = str(Dataset.gt_list[i]).replace('/gt/','/dets_class/')
        with open(result_file_path, 'w') as f:
            for m in range(len(object_types)):
                ty = object_types[m]
                x     = xyz_lwhr_confid_id[m][0]
                y     = xyz_lwhr_confid_id[m][1]
                z     = xyz_lwhr_confid_id[m][2]
                l     = xyz_lwhr_confid_id[m][3]
                w     = xyz_lwhr_confid_id[m][4]
                h     = xyz_lwhr_confid_id[m][5]
                theta = xyz_lwhr_confid_id[m][6]
                confid_bg =  xyz_lwhr_confid_id[m][7]
                confid_car = xyz_lwhr_confid_id[m][8]
                confid_ped = xyz_lwhr_confid_id[m][9]
                # confid_van = xyz_lwhr_confid_id[m][10]
                confid_cyc = xyz_lwhr_confid_id[m][10]
                _num_points = xyz_lwhr_confid_id[m][11]
                track_id    = xyz_lwhr_confid_id[m][12]
                # f.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f, \n'%(ty,
                #               x, y, z, l, w, h, theta, confid_bg, confid_car, confid_ped, confid_van, confid_cyc))
                f.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%d,%.2f\n'%(ty,
                               x, y, z, l, w, h, theta, confid_bg, confid_car, confid_ped, confid_cyc, _num_points,track_id))

    print("The number of car: {}".format(num_car))
    print("The number of pedestrian: {}".format(num_ped))
    print("The number of cyclist: {}".format(num_cyc))


def save_detection_classification_multi_phases(tracking_config_path='/home/ben/projects/tracking/hbtk/config/kitti_tracking.config',
                                               detection_config_path='/home/ben/projects/tracking/hbtk/config/detection.config',
                                               phases = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0020_sync',
                                                        '2011_09_26_drive_0035_sync','2011_09_26_drive_0084_sync',
                                                        '2011_09_26_drive_0005_sync','2011_09_26_drive_0014_sync',
                                                        '2011_09_26_drive_0019_sync','2011_09_26_drive_0059_sync']):
                                               # phases = ['2011_09_26_drive_0001_sync']):
    """
    1 prepare the path for all the file, point, img and calibiration
    2 read the ground truth and add them to dets
    3 crop off points of foreground objects, run classification model to generate PDF
    4 read the image and calibration, crop all the point within the image view
    5 run segmentation algorithm to save the background objects
    """
    #  ********* tracking configuration ***********
    # read configuration file
    config = pipeline_pb2.TrackingPipeline()
    with open(tracking_config_path, "r") as f:
        protos_str = f.read()
        text_format.Merge(protos_str, config)
    #shutil.copyfile(config_path, str(output_dir+"/"+"pipeline.config"))
    for phase in phases:
        print(phase)
        config.dataset.phase = phase
        __save_detection_classification_one_phases(config, detection_config_path)



if __name__=='__main__':
    fire.Fire()
