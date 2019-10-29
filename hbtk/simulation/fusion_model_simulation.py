"""
    File Description:
        Fusion model simulation run without running real tracker algorithm

"""
import pykitti
import fire
import os
import sys
import open3d
import torch
import time

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
from hbtk.detectors.pointnet.model import PointNetCls, feature_transform_regularizer
from hbtk.detectors.efficient_det.classificaiton_pointnet import Classification_Pointnet
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


def _save_detection_tracking_(config_path, dataset_path):
    #
    tracklet_path = dataset_path / 'tracklets_pc'
    all_seqs = list(tracklet_path.glob('seq_*'))

    # create the results folder for detection and tracking
    detection_results = dataset_path / 'tracklets_pc/dets_conf'
    detection_results.mkdir(parents=True, exist_ok=True)
    tracking_results = dataset_path / 'tracklets_pc/trk_conf'
    tracking_results.mkdir(parents=True, exist_ok=True)

    config = pipeline_det_pb2.DetectionPipeline()
    with open(config_path, "r") as f:
        protos_str = f.read()
        text_format.Merge(protos_str, config)

    classifier = Classification_Pointnet(config.clustertodetection)

    # classifier = Classification_Pointnet(config.clustertodetection)
    # processing with classifier to generate the detection and tracking
    # confidence
    for one_seq in all_seqs:
        _name_seq = one_seq.name

        # all sub-point cloud
        all_frame_ids = []
        _all_pc = list(sorted(one_seq.glob('*.bin')))
        for _pc_ in _all_pc:
            _pc_name = _pc_.name
            _frame_id = int(_pc_name.split('_')[1][:-4])
            o_type = _pc_name.split('_')[0]
            all_frame_ids.append(_frame_id)
        all_frame_ids = sorted(all_frame_ids)
        previous_id = None
        det_one = None
        trk_one = None
        object_types = []
        det_confidence = []
        trk_confidence = []
        previous_num_points = 0
        ratio = 0.1
        start_frame_count = 0
        _count_ = 0
        for one_id in all_frame_ids:
            # adding the order checking, make sure that sequence order is right
            _pc_name = _pc_.name
            # _frame_id = int(_pc_name.split('_')[1][:-4])
            # o_type = _pc_name.split('_')[0]
            _frame_id = one_id
            pc_file_path = one_seq / '{}_{}.bin'.format(o_type, one_id)

            _count_ += 1
            if previous_id is None:
                previous_id = _frame_id
            else:
                assert _frame_id > previous_id, 'the order of detected objects are not correct.{}_{}_{}'.format(one_seq, _frame_id, previous_id)
                previous_id = _frame_id

            points = np.fromfile(str(pc_file_path) ,dtype=np.float32, count=-1).reshape([-1, 4])
            num_points = points.shape[0]
            det_one = classifier.classification(points)
            object_types.append(o_type)

            det_confidence.append(det_one)

            if trk_one is None:
                # first frame
                trk_one = det_one
                previous_num_points = num_points
            else:
                #if abs(num_points - previous_num_points)/float(previous_num_points) > ratio:
                if (num_points - previous_num_points)/float(previous_num_points) > ratio: #and _count_ >= start_frame_count:  # # get more and more points without abs
                    # fuse the confidence
                    trk_one = fuse_probability(trk_one, det_one)
                    previous_num_points = num_points

            if _count_ < start_frame_count:
                trk_confidence.append(det_one)
                from pudb import set_trace; set_trace()
            else:
                trk_confidence.append(trk_one)



        # save the resultant files
        _det_file = detection_results / '{}.txt'.format(_name_seq)
        with open(str(_det_file), 'w') as f:
            for m in range(len(object_types)):
                cla = object_types[m]
                confid_bg  = det_confidence[m][0]
                confid_car = det_confidence[m][1]
                confid_ped = det_confidence[m][2]
                confid_cyc = det_confidence[m][3]
                f.write('%s,%.4f,%.4f,%.4f,%.4f\n'%(cla, confid_bg, confid_car, confid_ped, confid_cyc))

        _trk_file = tracking_results / '{}.txt'.format(_name_seq)
        with open(str(_trk_file), 'w') as f:
            for m in range(len(object_types)):
                cla = object_types[m]
                confid_bg  = trk_confidence[m][0]
                confid_car = trk_confidence[m][1]
                confid_ped = trk_confidence[m][2]
                confid_cyc = trk_confidence[m][3]
                f.write('%s,%.4f,%.4f,%.4f,%.4f\n'%(cla, confid_bg, confid_car, confid_ped, confid_cyc))

def save_detection_tracking_multi_phases(dataset_root='/home/ben/Dataset/KITTI',
                                         ):

    dataset_root = '/home/ben/Dataset/KITTI/2011_09_26'
    phases = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0020_sync',
              '2011_09_26_drive_0035_sync','2011_09_26_drive_0084_sync',
              '2011_09_26_drive_0005_sync','2011_09_26_drive_0014_sync',
              '2011_09_26_drive_0019_sync','2011_09_26_drive_0059_sync']
    config_path = '/home/ben/projects/tracking/hbtk/config/detection.config'

    t_s = time.time()
    for phase in phases:
        print('Process the dataset {}'.format(phase))
        dataset_path = Path(dataset_root) / phase
        _save_detection_tracking_(config_path, dataset_path)

    print("Finish time:{}".format(time.time() - t_s))



def _save_detection_tracking_without_detection(config_path, dataset_path):
    #
    tracklet_path = dataset_path / 'tracklets_pc'
    all_seqs = list(tracklet_path.glob('seq_*'))

    # create the results folder for detection and tracking
    detection_results = dataset_path / 'tracklets_pc/dets_conf'
    # detection_results.mkdir(parents=True, exist_ok=True)
    tracking_results = dataset_path / 'tracklets_pc/trk_conf'
    tracking_results.mkdir(parents=True, exist_ok=True)

    # processing with classifier to generate the detection and tracking
    # confidence
    for one_seq in all_seqs:
        _name_seq = one_seq.name

        _dets_ = None
        with open(str(detection_results / (_name_seq+'.txt')), 'r') as f:
            lines = f.readlines()
        _dets_ = [line.strip().split(',')[1:] for line in lines] # ignore the first class
        _dets_ = np.array(_dets_, dtype=np.float32)

        # all sub-point cloud
        all_frame_ids = []
        _all_pc = list(sorted(one_seq.glob('*.bin')))
        for _pc_ in _all_pc:
            _pc_name = _pc_.name
            _frame_id = int(_pc_name.split('_')[1][:-4])
            o_type = _pc_name.split('_')[0]
            all_frame_ids.append(_frame_id)
        # sorting ids is critical step
        all_frame_ids = sorted(all_frame_ids)
        previous_id = None
        trk_one = None
        object_types = []
        trk_confidence = []
        previous_num_points = 0
        max_num_points = 0
        ratio = 0.16
        start_frame_count = 16
        _count_ = 0
        for one_id in all_frame_ids:
            # adding the order checking, make sure that sequence order is right
            _pc_name = _pc_.name
            # _frame_id = int(_pc_name.split('_')[1][:-4])
            # o_type = _pc_name.split('_')[0]
            _frame_id = one_id
            pc_file_path = one_seq / '{}_{}.bin'.format(o_type, one_id)


            if previous_id is None:
                previous_id = _frame_id
            else:
                assert _frame_id > previous_id, 'the order of detected objects are not correct.{}_{}_{}'.format(one_seq, _frame_id, previous_id)
                previous_id = _frame_id

            points = np.fromfile(str(pc_file_path) ,dtype=np.float32, count=-1).reshape([-1, 4])
            num_points = points.shape[0]

            det_one = _dets_[_count_, :]
            _count_ += 1

            object_types.append(o_type)

            if _count_ < start_frame_count:
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


def save_detection_tracking_multi_phases_without_detection(dataset_root='/home/ben/Dataset/KITTI',
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
        _save_detection_tracking_without_detection(config_path, dataset_path)






if __name__=='__main__':
    fire.Fire()
