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
import torch
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(ROOT_DIR))
sys.path.append(os.path.join( os.path.dirname(ROOT_DIR), 'hbtk'))
#sys.path.append(os.path.join( os.path.dirname(os.path.dirname(ROOT_DIR)), 'hbtk', 'protos'))

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature

from sklearn import metrics
from scipy.interpolate import interp1d

from google.protobuf import text_format
from hbtk.protos import pipeline_pb2, pipeline_det_pb2
from hbtk.dataset.kitti_dataset import Kitti_dataset
from hbtk.utils import box_np_ops
from pathlib import Path
import numpy as np
from skimage import io
from hbtk.detectors.pointnet.model import PointNetCls, feature_transform_regularizer
from hbtk.detectors.efficient_det.classificaiton_pointnet import Classification_Pointnet
#from hbtk.utils.visualization_pc import *


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


def _save_detection_tracking_without_detection(ratio, start_frame_count, config_path, dataset_path):
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
        # ratio = 0.16
        # start_frame_count = 3
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


def save_detection_tracking_multi_phases_without_detection(ratio=0.10, start_frame_count=3, dataset_root='/home/ben/Dataset/KITTI'):
    dataset_root = '/home/ben/Dataset/KITTI/2011_09_26'
    phases = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0020_sync',
              '2011_09_26_drive_0035_sync','2011_09_26_drive_0084_sync',
              '2011_09_26_drive_0005_sync','2011_09_26_drive_0014_sync',
              '2011_09_26_drive_0019_sync','2011_09_26_drive_0059_sync']
    config_path = '/home/ben/projects/tracking/hbtk/config/detection.config'

    for phase in phases:
        print('Process the dataset {}'.format(phase))
        dataset_path = Path(dataset_root) / phase
        _save_detection_tracking_without_detection(ratio, start_frame_count, config_path, dataset_path)


### running the simulation method
def read_the_detection_simulation(path):
    """
    select the some detections
    """
    dets = []
    with open(str(path), 'r') as f:
        lines = f.readlines()
    _dets_ = [line.strip().split(',') for line in lines]
    for _det in _dets_:
        if _det[0] in ['Bg','Car', 'Van', 'Pedestrian', 'Cyclist']:
            if _det[0] in ['Bg']:
                _det[0] = 0
            if _det[0] in ['Car', 'Van'] :
                _det[0] = 1
            if _det[0] in ['Pedestrian'] :
                _det[0] = 2
            if _det[0] in ['Cyclist'] :
                _det[0] = 3
            dets.append(_det)
    dets = np.array( dets, dtype=np.float32)
    return dets

# read background objects
def read_tracklet_bg(path):
    """
    background
    """
    dets = []
    with open(str(path), 'r') as f:
        lines = f.readlines()
    _dets_ = [line.strip().split(',') for line in lines]
    for _det in _dets_:
        if _det[0] in ['Bg']:
        # if _det[0] in ['Car', 'Van', 'Pedestrian', 'Cyclist']:
            if _det[0] in ['Bg']:
                 _det[0] = 0
            if _det[0] in ['Car', 'Van'] :
                _det[0] = 1
            if _det[0] in ['Pedestrian'] :
                _det[0] = 2
            if _det[0] in ['Cyclist'] :
                _det[0] = 3
            #dets.append([_det[0], _det[3], _det[4], _det[5], _det[6]])
            dets.append([_det[0], _det[8], _det[9], _det[10], _det[11]])

    dets = np.array( dets, dtype=np.float32)
    return dets


# generating the experimental results of detection accuracy in C. Experiments with real detector and ideal tracker
def run_det_trk_result_in_simulation_with_background_objects():
    """
    Evaluation of one dataset in simulation
    """

    # configuration here
    dataset_dir = '/home/ben/Dataset/KITTI/2011_09_26'
    phase_names = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0020_sync','2011_09_26_drive_0084_sync','2011_09_26_drive_0035_sync',
                   '2011_09_26_drive_0005_sync','2011_09_26_drive_0014_sync','2011_09_26_drive_0019_sync','2011_09_26_drive_0059_sync']

    current_class=['Car','Pedestrian','Cyclist']
    ROC_plot = True
    PRC_plot = True

    dataset_dir = Path(dataset_dir)
    if not isinstance(phase_names, list):
        phase_names = [phase_names]

    bg_annos = []
    for phase_name in phase_names:
        bg_dir = dataset_dir/phase_name/'detection/'
        # det_annos_list = list(sorted(det_dir.glob('tracklet_det/*.txt')))
        bg_annos_list = list(sorted(bg_dir.glob('dets_class/*.txt')))
        # bg_annos_list = list(sorted(bg_dir.glob('dets_trk/*.txt')))
        # det_dir = dataset_dir/phase_name
        # det_annos_list = list(sorted(det_dir.glob('tracklets_pc/dets_conf/*.txt')))
        for i in range(len(bg_annos_list)):
            # det_annos.append(read_tracklet_det(det_annos_list[i]))
            test_m=read_tracklet_bg(bg_annos_list[i])
            if test_m.shape[0] > 0:
                bg_annos.append(read_tracklet_bg(bg_annos_list[i]))

    trk_annos = []
    num_tracking_streak = 0
    for phase_name in phase_names:
        trk_dir = dataset_dir/phase_name/'tracklets_pc/'
        trk_annos_list = list(sorted(trk_dir.glob('trk_conf/*.txt')))
        num_tracking_streak += len(trk_annos_list)
        for i in range(len(trk_annos_list)):
            trk_annos.append(read_the_detection_simulation(trk_annos_list[i]))

    trk_annos = np.concatenate(trk_annos+bg_annos, axis=0)
    # trk_annos = np.concatenate(trk_annos, axis=0)
    print("The number of tracking streak: {}".format(num_tracking_streak))

    det_annos = []
    for phase_name in phase_names:
        det_dir = dataset_dir/phase_name/'tracklets_pc/'
        det_annos_list = list(sorted(det_dir.glob('dets_conf/*.txt')))
        for i in range(len(det_annos_list)):
            det_annos.append(read_the_detection_simulation(det_annos_list[i]))

    det_annos = np.concatenate(det_annos+bg_annos, axis=0)
    # det_annos = np.concatenate(det_annos, axis=0)
    #assert det_annos.shape[0]==trk_annos.shape[0], "The number of det and trk results should be the same!"

    class_to_name = {
        0:'Bg',
        1: 'Car',
        2: 'Pedestrian',
        3: 'Cyclist',
    }
    name_to_class = {v:n for n, v in class_to_name.items()}
    mAP_to_class = {}

    for class_name in current_class:
        if class_name == 'Bg':
            y_gt_det = np.where(det_annos[:,0]==0., 1. , 0. )
            y_pred_det = det_annos[:,1]
            y_gt_trk = np.where(trk_annos[:,0]==0., 1. , 0. )
            y_pred_trk = trk_annos[:,1]

        if class_name == 'Car':
            y_gt_det = np.where(det_annos[:,0]==1., 1. , 0. )
            y_pred_det = det_annos[:,2]
            y_gt_trk = np.where(trk_annos[:,0]==1., 1. , 0. )
            y_pred_trk = trk_annos[:,2]

        if class_name == 'Pedestrian':
            y_gt_det = np.where(det_annos[:,0]==2., 1. , 0. )
            y_pred_det = det_annos[:,3]
            y_gt_trk = np.where(trk_annos[:,0]==2., 1. , 0. )
            y_pred_trk = trk_annos[:,3]

        if class_name == 'Cyclist':
            y_gt_det = np.where(det_annos[:,0]==3., 1. , 0. )
            y_pred_det = det_annos[:,4]
            y_gt_trk = np.where(trk_annos[:,0]==3., 1. , 0. )
            y_pred_trk = trk_annos[:,4]


        mAP_det = average_precision_score(y_gt_det, y_pred_det)
        mAP_to_class[class_name] = mAP_det

        mAP_trk = average_precision_score(y_gt_trk, y_pred_trk)
        mAP_to_class[class_name] = mAP_trk


        precision_det, recall_det, thresholds_det = precision_recall_curve(y_gt_det, y_pred_det)
        fpr_det, tpr_det, _ = metrics.roc_curve(y_gt_det, y_pred_det)

        precision_trk, recall_trk, thresholds_trk = precision_recall_curve(y_gt_trk, y_pred_trk)
        fpr_trk, tpr_trk, _ = metrics.roc_curve(y_gt_trk, y_pred_trk)

        if class_name == 'Car':
            car_det_mAP = mAP_det
            car_trk_mAP = mAP_trk
        if class_name == 'Pedestrian':
            ped_det_mAP = mAP_det
            ped_trk_mAP = mAP_trk
        if class_name == 'Cyclist':
            cyc_det_mAP = mAP_det
            cyc_trk_mAP = mAP_trk

    return car_det_mAP, car_trk_mAP, ped_det_mAP, ped_trk_mAP, cyc_det_mAP, cyc_trk_mAP



if __name__=='__main__':
    fire.Fire()
