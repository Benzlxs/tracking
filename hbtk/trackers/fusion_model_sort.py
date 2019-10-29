"""
    File Description:
        Fusion model testing with the tracking results from sort_3d
    Output:
        1. the tracking detection files

"""
import pykitti
import fire
import os
import sys
import open3d
import torch
import time
import shutil
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(ROOT_DIR))
sys.path.append(os.path.join( os.path.dirname(ROOT_DIR), 'hbtk'))
#sys.path.append(os.path.join( os.path.dirname(os.path.dirname(ROOT_DIR)), 'hbtk', 'protos'))


from google.protobuf import text_format
from hbtk.protos import pipeline_pb2, pipeline_det_pb2
from hbtk.dataset.kitti_dataset import Kitti_dataset
from hbtk.utils import box_np_ops
from pathlib import Path
#from source import parseTrackletXML as xmlParser
import numpy as np
from skimage import io
from hbtk.detectors.pointnet.model import PointNetCls, feature_transform_regularizer
from hbtk.detectors.efficient_det.classificaiton_pointnet import Classification_Pointnet
from hbtk.utils.visualization_pc import *


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


def fuse_probability(trk_c, det_c):
    assert len(trk_c) == len(det_c), "the variables shold have the same shape"
    fuse_confid = []
    normalization = sum([trk_c[i]*det_c[i] for i in range(len(trk_c))])
    if normalization == 0.0:
        normalization = sum([(trk_c[i]+very_very_small_number)*(det_c[i]+very_very_small_number) for i in range(len(trk_c))])
        for i in range(len(trk_c)):
            fuse_confid.append((trk_c[i]+very_very_small_number)*(det_c[i]+very_very_small_number)/normalization)
    else:
        for i in range(len(trk_c)):
            fuse_confid.append(trk_c[i]*det_c[i]/normalization)

    return fuse_confid


def __fusion_model_with_sort__(config_path, dataset_path):
    # fetch all tracklets created by sort_3d
    tracklet_path = dataset_path / 'detection/tracklet_det'
    all_seqs = list(tracklet_path.glob('seq_*.txt'))

    # make the directory to save the results of fusion model
    tracking_results = dataset_path / 'detection/tracklet_trk'
    shutil.rmtree(str(tracking_results))
    tracking_results.mkdir(parents=True, exist_ok=True)

    # processing with classifier to generate the detection and tracking
    # confidence
    for one_seq in all_seqs:
        _name_seq = one_seq.name

        _dets_ = None
        with open(str(one_seq), 'r') as f:
            lines = f.readlines()

        _names_type = [line.strip().split(',')[0] for line in lines] # ignore the first class

        _dets_ = [line.strip().split(',')[1:] for line in lines] # ignore the first class
        _dets_ = np.array(_dets_, dtype=np.float32)

        # all sub-point cloud
        all_frame_ids = []

        trk_one = None
        object_types = []
        trk_confidence = []
        previous_num_points = 0
        max_num_points = 0
        ratio = 0.16
        start_frame_count = 3
        for _ind  in range(_dets_.shape[0]):
            # adding the order checking, make sure that sequence order is right

            det_one = _dets_[_ind, :-1]
            num_points = _dets_[_ind, -1]

            object_types.append(_names_type[_ind])

            if _ind < start_frame_count:
                trk_confidence.append(det_one)
            else:
                if trk_one is None:
                    # first frame
                    trk_one = det_one
                    previous_num_points = num_points
                    max_num_points = num_points
                else:
                    #if abs(num_points - previous_num_points)/float(previous_num_points) > ratio:
                    best_confidence = max(trk_one)
                    if (num_points - previous_num_points)/float(previous_num_points) > ratio :
                    # if (num_points - max_num_points)/float(max_num_points) > ratio and _count_ >= start_frame_count:
                    # if (num_points - previous_num_points)/float(previous_num_points) > ratio and _count_ >= start_frame_count and best_confidence < 0.999:  # # get more and more points without abs
                        # fuse the confidence
                        trk_one = fuse_probability(trk_one, det_one)
                        previous_num_points = num_points
                        # get the maximum points
                        if num_points > max_num_points:
                            max_num_points = num_points

                trk_confidence.append(trk_one)
                # trk_confidence.append(det_one)

        _trk_file = tracking_results / '{}.txt'.format(_name_seq)
        with open(str(_trk_file), 'w') as f:
            for m in range(len(object_types)):
                cla = object_types[m]
                confid_bg  = trk_confidence[m][0]
                confid_car = trk_confidence[m][1]
                confid_ped = trk_confidence[m][2]
                confid_cyc = trk_confidence[m][3]
                f.write('%s,%.4f,%.4f,%.4f,%.4f\n'%(cla, confid_bg, confid_car, confid_ped, confid_cyc))


def fusion_model_with_sort_results(dataset_root='/home/ben/Dataset/KITTI',
                                         ):

    dataset_root = '/home/ben/Dataset/KITTI/2011_09_26'
    phases = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0020_sync',
              '2011_09_26_drive_0035_sync','2011_09_26_drive_0084_sync',
              '2011_09_26_drive_0005_sync','2011_09_26_drive_0014_sync',
              '2011_09_26_drive_0019_sync','2011_09_26_drive_0059_sync']
    config_path = '/home/ben/projects/tracking/hbtk/config/detection.config'

    for phase in phases:
        print('Process the dataset {}'.format(phase))
        dataset_path = Path(dataset_root) / phase
        __fusion_model_with_sort__(config_path, dataset_path)






if __name__=='__main__':
    fire.Fire()
