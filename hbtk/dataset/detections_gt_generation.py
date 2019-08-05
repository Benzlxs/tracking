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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(ROOT_DIR))
#sys.path.append(os.path.join( os.path.dirname(os.path.dirname(ROOT_DIR)), 'hbtk'))
#sys.path.append(os.path.join( os.path.dirname(os.path.dirname(ROOT_DIR)), 'hbtk', 'protos'))


from google.protobuf import text_format
from hbtk.protos import pipeline_pb2
from hbtk.dataset.kitti_dataset import Kitti_dataset
from hbtk.utils import box_np_ops
from pathlib import Path
from source import parseTrackletXML as xmlParser
import numpy as np
from skimage import io

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





def save_detection_gt(config_path):
    """
    1 prepare the path for all the file, point, img and calibiration
    2 read the ground truth and add them to dets
    3 crop off points of foreground objects
    4 read the image and calibration, crop all the point within the image view
    5 run segmentation algorithm to save the background objects
    """
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

    output_folder_dir = Path(dataset_config.database_dir)/dataset_config.phase/'detection/dets_gt'
    output_folder_dir.mkdir(parents=True, exist_ok=True)

    Dataset = Kitti_dataset(dataset_config)

    object_types = []
    xyz_lwh_confid = []
    for i in range(0, Dataset.__len__()):
        dets = Dataset.get_detection(i)
        points = np.fromfile(str(Dataset.pc_list[i]),dtype=np.float32, count=-1).reshape([-1, 4])
        for j in range(0,len(dets)):
            j_type = dets[j][0]
            if j_type == 0:
                object_types.append(['Car'])
            else:
                if j_type == 1:
                    object_types.append(['Pedestrian'])
                else:
                    object_types.append(['Cyclist'])
            xyzlwh = dets[j][1:]
            indices  = points_inside_box(points, xyzlwh)
            object_points = points[indices[:],:]
            _x_min = min(object_points[:,0])
            _x_max = max(object_points[:,0])
            _y_min = min(object_points[:,1])
            _y_max = max(object_points[:,1])
            _z_min = min(object_points[:,2])
            _z_max = max(object_points[:,2])

            xyz_lwh_confid.append([(_x_min+_x_max)/2,  (_y_min+_y_max)/2, _z_min,
                                   _x_max - _x_min, _y_max - _y_min, _z_max - _z_min])

            points = points[~indices[:],:]



if __name__=='__main__':
    fire.Fire()
