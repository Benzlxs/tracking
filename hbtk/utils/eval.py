import io as sysio
import time

import numba
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
import os
import sys
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature

from sklearn import metrics

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# select the useful information
def read_the_detection_bk(path, class_type=None):
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
            dets.append(_det[:-1])
    if class_type == None:
        dets = np.array( dets, dtype=np.float32)
        return dets
    else:
        dets = np.array( dets, dtype=np.float32)
        if class_type=='Car':
            dets[:,8] = dets[:,9] + dets[:,11]
            return dets[:,:9]
        if class_type == 'Pedestrian':
            return dets[:,[0,1,2,3,4,5,6,7,10]]
        if class_type == 'Cyclist':
            return dets[:,[0,1,2,3,4,5,6,7,12]]

def read_the_detection(path):
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
            dets.append(_det[:-1])
    dets = np.array( dets, dtype=np.float32)
    return dets

def evaluate_dataset(dataset_dir,
                     phase_names,
                     current_class=['Bg','Car','Pedestrian','Cyclist'],
                     ROC_plot=True,
                     PRC_plot=True,
                     save_path='../results/'):
    """
    Evaluation of one dataset
    """
    dataset_dir = Path(dataset_dir)
    if not isinstance(phase_names, list):
        phase_names = [phase_names]

    det_annos = []
    for phase_name in phase_names:
        det_dir = dataset_dir/phase_name/'detection/'
        det_annos_list = list(sorted(det_dir.glob('dets_class/*.txt')))
        for i in range(len(det_annos_list)):
            det_annos.append(read_the_detection(det_annos_list[i]))

    det_annos = np.concatenate(det_annos, axis=0)

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
            y_gt = np.where(det_annos[:,0]==0., 1. , 0. )
            y_pred = det_annos[:,8]

        if class_name == 'Car':
            y_gt = np.where(det_annos[:,0]==1., 1. , 0. )
            y_pred = det_annos[:,9] + det_annos[:,11]

        if class_name == 'Pedestrian':
            y_gt = np.where(det_annos[:,0]==2., 1. , 0. )
            y_pred = det_annos[:,10]

        if class_name == 'Cyclist':
            y_gt = np.where(det_annos[:,0]==3., 1. , 0. )
            y_pred = det_annos[:,12]

        mAP = average_precision_score(y_gt, y_pred)
        mAP_to_class[class_name] = mAP

        precision, recall, thresholds = precision_recall_curve(y_gt, y_pred)
        fpr, tpr, _ = metrics.roc_curve(y_gt, y_pred)

        if ROC_plot:
            plt.figure()
            plt.plot(fpr, tpr)

            plt.xlabel('Speciality')
            plt.ylabel('Sensitivity')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Roc curve of {}: AP={}%'.format(class_name ,int(mAP*100)))
            plt.show()


        if PRC_plot:
            plt.figure()
            plt.plot(recall, precision)

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('PRC curve of {}: AP={}%'.format(class_name ,int(100*mAP)))
            plt.show()

    return mAP_to_class

if __name__=="__main__":
    dataset_dir = '/home/ben/Dataset/KITTI/2011_09_26'
    phase_name = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0020_sync','2011_09_26_drive_0084_sync','2011_09_26_drive_0035_sync']
    mAPs = evaluate_dataset(dataset_dir, phase_name)
    dataset_dir = '/home/ben/Dataset/KITTI/2011_09_26/2011_09_26_drive_0084_sync/detection/'
    dataset_dir = Path(dataset_dir)
    # read the gt_annos, type as list
    # gt_annos_list = list(sorted(dataset_dir.glob('dets_gt/*.txt')))
    # read the det_annos
    det_annos_list = list(sorted(dataset_dir.glob('dets_class/*.txt')))
    # assert len(gt_annos_list) == len(det_annos_list), "The detection results and ground truth should have the same length"
    # gt_annos = []
    det_annos = []
    for i in range(len(det_annos_list)):
        # assert gt_annos_list[i].name() == det_annos_list[i].name(), "The ground truth and detection should have the same name"
        # gt_annos.append(read_the_detection(gt_annos_list[i]))
        det_annos.append(read_the_detection(det_annos_list[i]))
    det_annos = np.concatenate(det_annos, axis=0)

    y_gt = np.where(det_annos[:,0]==1.,1. ,0. )
    y_pred = det_annos[:,9] + det_annos[:,11]
    # y_pred = det_annos[:,10]

    mAP = average_precision_score(y_gt, y_pred)

    # precision, recall, thresholds = precision_recall_curve(y_gt, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_gt, y_pred)

    plt.figure()
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    #plt.step(recall, precision, color='b', alpha=0.2, where='post')
    # plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    # plt.plot(recall, precision)
    plt.plot(fpr, tpr)

    #plt.xlabel('Recall')
    #plt.ylabel('Precision')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(mAP))
    # convert them into the specific format
    # call the evaluation function
    plt.show()

    class_to_name = {
        0:'Bg',
        1: 'Car',
        2: 'Pedestrian',
        3: 'Cyclist',
    }

    eval_classes = ['Car', 'Pedestrian', 'Cyclist']


