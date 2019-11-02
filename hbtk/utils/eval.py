import io as sysio
import time
import fire
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

def evaluate_dataset_det(dataset_dir,
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
        #det_annos_list = list(sorted(det_dir.glob('dets_trk/*.txt')))
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
            y_pred = det_annos[:,9]

        if class_name == 'Pedestrian':
            y_gt = np.where(det_annos[:,0]==2., 1. , 0. )
            y_pred = det_annos[:,10]

        if class_name == 'Cyclist':
            y_gt = np.where(det_annos[:,0]==3., 1. , 0. )
            y_pred = det_annos[:,11]

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

def evaluate_dataset_trk_det(dataset_dir,
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

    trk_annos = []
    for phase_name in phase_names:
        trk_dir = dataset_dir/phase_name/'detection/'
        # det_annos_list = list(sorted(det_dir.glob('dets_class/*.txt')))
        trk_annos_list = list(sorted(trk_dir.glob('dets_trk/*.txt')))
        # trk_annos_list = list(sorted(trk_dir.glob('dets_class/*.txt')))
        for i in range(len(trk_annos_list)):
            trk_annos.append(read_the_detection(trk_annos_list[i]))

    trk_annos = np.concatenate(trk_annos, axis=0)

    det_annos = []
    for phase_name in phase_names:
        det_dir = dataset_dir/phase_name/'detection/'
        det_annos_list = list(sorted(det_dir.glob('dets_class/*.txt')))
        # det_annos_list = list(sorted(det_dir.glob('dets_class_small_scale/*.txt')))
        for i in range(len(det_annos_list)):
            det_annos.append(read_the_detection(det_annos_list[i]))
    det_annos = np.concatenate(det_annos, axis=0)

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
            y_pred_det = det_annos[:,8]
            y_gt_trk = np.where(trk_annos[:,0]==0., 1. , 0. )
            y_pred_trk = trk_annos[:,8]

        if class_name == 'Car':
            y_gt_det = np.where(det_annos[:,0]==1., 1. , 0. )
            y_pred_det = det_annos[:,9]
            y_gt_trk = np.where(trk_annos[:,0]==1., 1. , 0. )
            y_pred_trk = trk_annos[:,9]

        if class_name == 'Pedestrian':
            y_gt_det = np.where(det_annos[:,0]==2., 1. , 0. )
            y_pred_det = det_annos[:,10]
            y_gt_trk = np.where(trk_annos[:,0]==2., 1. , 0. )
            y_pred_trk = trk_annos[:,10]

        if class_name == 'Cyclist':
            y_gt_det = np.where(det_annos[:,0]==3., 1. , 0. )
            y_pred_det = det_annos[:,11]
            y_gt_trk = np.where(trk_annos[:,0]==3., 1. , 0. )
            y_pred_trk = trk_annos[:,11]


        mAP_det = average_precision_score(y_gt_det, y_pred_det)
        mAP_to_class[class_name] = mAP_det

        mAP_trk = average_precision_score(y_gt_trk, y_pred_trk)
        mAP_to_class[class_name] = mAP_trk


        precision_det, recall_det, thresholds_det = precision_recall_curve(y_gt_det, y_pred_det)
        fpr_det, tpr_det, _ = metrics.roc_curve(y_gt_det, y_pred_det)

        precision_trk, recall_trk, thresholds_trk = precision_recall_curve(y_gt_trk, y_pred_trk)
        fpr_trk, tpr_trk, _ = metrics.roc_curve(y_gt_trk, y_pred_trk)

        if ROC_plot:
            fig, ax = plt.subplots()
            plt.plot(fpr_det, tpr_det, linestyle='dashed', linewidth=4, label='det_only')
            plt.plot(fpr_trk, tpr_trk, linestyle='dashed', linewidth=4, label='with_trk')

            ax.legend()
            plt.xlabel('Speciality')
            plt.ylabel('Sensitivity')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            #plt.title('Roc curve of {}: AP={}%'.format(class_name ,int(mAP*100)))
            plt.title("ROC")
            # plt.show()
            plt.savefig("Roc_{}_det_{}_trk_{}.png".format(class_name, mAP_det, mAP_trk))


        if PRC_plot:
            # fig, ax = plt.subplots()
            plt.figure()
            # plt.plot(recall, precision)
            plt.plot(recall_det, precision_det,linestyle='dashed', linewidth=4, label='det_only')
            plt.plot(recall_trk, precision_trk,linestyle='dashed', linewidth=4, label='with_trk')

            plt.legend()
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            #plt.title('PRC curve of {}: AP={}%'.format(class_name ,int(100*mAP)))
            plt.title("PRC")
            # plt.show()
            plt.savefig("PRC_{}_det_{}_trk_{}.png".format(class_name, mAP_det, mAP_trk))


    return mAP_to_class

def evaluate_dataset_trk(dataset_dir,
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
        # det_annos_list = list(sorted(det_dir.glob('dets_class/*.txt')))
        det_annos_list = list(sorted(det_dir.glob('dets_trk/*.txt')))
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
            y_pred = det_annos[:,9]

        if class_name == 'Pedestrian++':
            y_gt = np.where(det_annos[:,0]==2., 1. , 0. )
            y_pred = det_annos[:,10]

        if class_name == 'Cyclist':
            y_gt = np.where(det_annos[:,0]==3., 1. , 0. )
            y_pred = det_annos[:,11]

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

def run_det_results():
    dataset_dir = '/home/ben/Dataset/KITTI/2011_09_26'
    phase_name = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0020_sync','2011_09_26_drive_0084_sync','2011_09_26_drive_0035_sync']
    mAPs = evaluate_dataset_det(dataset_dir, phase_name)

def run_trk_results():
    dataset_dir = '/home/ben/Dataset/KITTI/2011_09_26'
    phase_name = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0020_sync','2011_09_26_drive_0084_sync','2011_09_26_drive_0035_sync']
    mAPs = evaluate_dataset_trk(dataset_dir, phase_name)

def run_det_trk_results():
    dataset_dir = '/home/ben/Dataset/KITTI/2011_09_26'
    #phase_name = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0020_sync','2011_09_26_drive_0084_sync','2011_09_26_drive_0035_sync',
    #              '2011_09_26_drive_0005_sync','2011_09_26_drive_0014_sync','2011_09_26_drive_0019_sync','2011_09_26_drive_0059_sync']
    phase_name = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0020_sync','2011_09_26_drive_0084_sync','2011_09_26_drive_0035_sync']
    #              '2011_09_26_drive_0005_sync','2011_09_26_drive_0014_sync','2011_09_26_drive_0019_sync','2011_09_26_drive_0059_sync']

    mAPs = evaluate_dataset_trk_det(dataset_dir, phase_name)

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

def run_det_trk_result_in_simulation():
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

    trk_annos = []
    for phase_name in phase_names:
        trk_dir = dataset_dir/phase_name/'tracklets_pc/'
        trk_annos_list = list(sorted(trk_dir.glob('trk_conf/*.txt')))
        for i in range(len(trk_annos_list)):
            trk_annos.append(read_the_detection_simulation(trk_annos_list[i]))

    trk_annos = np.concatenate(trk_annos, axis=0)

    det_annos = []
    for phase_name in phase_names:
        det_dir = dataset_dir/phase_name/'tracklets_pc/'
        det_annos_list = list(sorted(det_dir.glob('dets_conf/*.txt')))
        for i in range(len(det_annos_list)):
            det_annos.append(read_the_detection_simulation(det_annos_list[i]))
    det_annos = np.concatenate(det_annos, axis=0)

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

        if ROC_plot:
            fig, ax = plt.subplots()
            plt.plot(fpr_det, tpr_det, linestyle='dashed', linewidth=4, label='det_only')
            plt.plot(fpr_trk, tpr_trk, linestyle='dashed', linewidth=4, label='with_trk')

            ax.legend()
            plt.xlabel('Speciality')
            plt.ylabel('Sensitivity')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            #plt.title('Roc curve of {}: AP={}%'.format(class_name ,int(mAP*100)))
            plt.title("ROC")
            # plt.show()
            plt.savefig("tmp/Roc_{}_det_{}_trk_{}.png".format(class_name, mAP_det, mAP_trk))


        if PRC_plot:
            # fig, ax = plt.subplots()
            plt.figure()
            # plt.plot(recall, precision)
            plt.plot(recall_det, precision_det,linestyle='dashed', linewidth=4, label='det_only')
            plt.plot(recall_trk, precision_trk,linestyle='dashed', linewidth=4, label='with_trk')

            plt.legend()
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            #plt.title('PRC curve of {}: AP={}%'.format(class_name ,int(100*mAP)))
            plt.title("PRC")
            # plt.show()
            plt.savefig("tmp/PRC_{}_det_{}_trk_{}.png".format(class_name, mAP_det, mAP_trk))

#####*********************************#######
### running experiments with real trackers
# reading the tracking results in tracklet
def read_tracklet_trk(path):
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

# reading the detection results in tracklet
def read_tracklet_det(path):
    """
    select the some detections
    """
    dets = []
    with open(str(path), 'r') as f:
        lines = f.readlines()
    _dets_ = [line.strip().split(',') for line in lines]
    for _det in _dets_:
        # if _det[0] in ['Bg','Car', 'Van', 'Pedestrian', 'Cyclist']:
        if _det[0] in ['Car', 'Van', 'Pedestrian', 'Cyclist']:
            if _det[0] in ['Bg']:
                 _det[0] = 0
            if _det[0] in ['Car', 'Van'] :
                _det[0] = 1
            if _det[0] in ['Pedestrian'] :
                _det[0] = 2
            if _det[0] in ['Cyclist'] :
                _det[0] = 3
            # dets.append([_det[0], _det[3], _det[4], _det[5], _det[6]])
            dets.append([_det[0], _det[8], _det[9], _det[10], _det[11]])

    dets = np.array( dets, dtype=np.float32)
    return dets

# check the accuracy difference
def run_det_trk_result_with_real_tracker():
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

    trk_annos = []
    for phase_name in phase_names:
        trk_dir = dataset_dir/phase_name/'detection/'
        trk_annos_list = list(sorted(trk_dir.glob('tracklet_trk/*.txt')))
        for i in range(len(trk_annos_list)):
            trk_annos.append(read_tracklet_trk(trk_annos_list[i]))
    trk_annos = np.concatenate(trk_annos, axis=0)

    det_annos = []
    for phase_name in phase_names:
        det_dir = dataset_dir/phase_name/'detection/'
        # det_annos_list = list(sorted(det_dir.glob('tracklet_det/*.txt')))
        det_annos_list = list(sorted(det_dir.glob('dets_class/*.txt')))
        # det_dir = dataset_dir/phase_name
        # det_annos_list = list(sorted(det_dir.glob('tracklets_pc/dets_conf/*.txt')))
        for i in range(len(det_annos_list)):
            # det_annos.append(read_tracklet_det(det_annos_list[i]))
            test_m=read_tracklet_det(det_annos_list[i])
            if test_m.shape[0] > 0:
                det_annos.append(read_tracklet_det(det_annos_list[i]))
            # det_annos.append(read_the_detection_simulation(det_annos_list[i]))
    det_annos = np.concatenate(det_annos, axis=0)

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

        if ROC_plot:
            fig, ax = plt.subplots()
            plt.plot(fpr_det, tpr_det, linestyle='dashed', linewidth=4, label='det_only')
            plt.plot(fpr_trk, tpr_trk, linestyle='dashed', linewidth=4, label='with_trk')

            ax.legend()
            plt.xlabel('Speciality')
            plt.ylabel('Sensitivity')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            #plt.title('Roc curve of {}: AP={}%'.format(class_name ,int(mAP*100)))
            plt.title("ROC")
            # plt.show()
            plt.savefig("real_tracker_tmp/Roc_{}_det_{}_trk_{}.png".format(class_name, mAP_det, mAP_trk))


        if PRC_plot:
            # fig, ax = plt.subplots()
            plt.figure()
            # plt.plot(recall, precision)
            plt.plot(recall_det, precision_det,linestyle='dashed', linewidth=4, label='det_only')
            plt.plot(recall_trk, precision_trk,linestyle='dashed', linewidth=4, label='with_trk')

            plt.legend()
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            #plt.title('PRC curve of {}: AP={}%'.format(class_name ,int(100*mAP)))
            plt.title("PRC")
            # plt.show()
            plt.savefig("real_tracker_tmp/PRC_{}_det_{}_trk_{}.png".format(class_name, mAP_det, mAP_trk))


#### *****************########
### unit test function, to test every dataset separately
def unit_test_realtracker():
    """
    Evaluation of one dataset in simulation
    """
    # configuration here
    dataset_dir = '/home/ben/Dataset/KITTI/2011_09_26'
    phase_names = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0020_sync','2011_09_26_drive_0084_sync','2011_09_26_drive_0035_sync',
                   '2011_09_26_drive_0005_sync','2011_09_26_drive_0014_sync','2011_09_26_drive_0019_sync','2011_09_26_drive_0059_sync']
    # phase_names = ['2011_09_26_drive_0001_sync']

    current_class=['Car','Pedestrian','Cyclist']
    ROC_plot = True
    PRC_plot = True

    dataset_dir = Path(dataset_dir)
    if not isinstance(phase_names, list):
        phase_names = [phase_names]

    # the detection results of extract ground truth firstly, then do detection
    # using the result in /detection/det_class
    trk_annos = []
    for phase_name in phase_names:
        trk_dir = dataset_dir/phase_name/'detection/'
        trk_annos_list = list(sorted(trk_dir.glob('dets_class/*.txt')))
        for i in range(len(trk_annos_list)):
            tem_m = read_tracklet_det(trk_annos_list[i])
            if tem_m.shape[0] > 0:
                trk_annos.append(read_tracklet_det(trk_annos_list[i]))
    trk_annos = np.concatenate(trk_annos, axis=0)

    # the detection results of cropping the points firstly
    # using the confidence scores in /tracklets/dets_conf
    det_annos = []
    for phase_name in phase_names:
        # det_dir = dataset_dir/phase_name/'detection/'
        # det_annos_list = list(sorted(det_dir.glob('tracklet_det/*.txt')))
        # det_annos_list = list(sorted(det_dir.glob('dets_class/*.txt')))
        det_dir = dataset_dir/phase_name
        det_annos_list = list(sorted(det_dir.glob('tracklets_pc/dets_conf/*.txt')))
        for i in range(len(det_annos_list)):
            # det_annos.append(read_tracklet_det(det_annos_list[i]))
            # test_m=read_tracklet_det(det_annos_list[i])
            # if test_m.shape[0] > 0:
            #     det_annos.append(read_tracklet_det(det_annos_list[i]))
            det_annos.append(read_the_detection_simulation(det_annos_list[i]))
    det_annos = np.concatenate(det_annos, axis=0)

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

        if ROC_plot:
            fig, ax = plt.subplots()
            plt.plot(fpr_det, tpr_det, linestyle='dashed', linewidth=4, label='det_only')
            plt.plot(fpr_trk, tpr_trk, linestyle='dashed', linewidth=4, label='with_trk')

            ax.legend()
            plt.xlabel('Speciality')
            plt.ylabel('Sensitivity')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            #plt.title('Roc curve of {}: AP={}%'.format(class_name ,int(mAP*100)))
            plt.title("ROC")
            # plt.show()
            plt.savefig("real_tracker_tmp/Roc_{}_det_{}_trk_{}.png".format(class_name, mAP_det, mAP_trk))


        if PRC_plot:
            # fig, ax = plt.subplots()
            plt.figure()
            # plt.plot(recall, precision)
            plt.plot(recall_det, precision_det,linestyle='dashed', linewidth=4, label='det_only')
            plt.plot(recall_trk, precision_trk,linestyle='dashed', linewidth=4, label='with_trk')

            plt.legend()
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            #plt.title('PRC curve of {}: AP={}%'.format(class_name ,int(100*mAP)))
            plt.title("PRC")
            # plt.show()
            plt.savefig("real_tracker_tmp/PRC_{}_det_{}_trk_{}.png".format(class_name, mAP_det, mAP_trk))




## Obtain the average number of tracking frames
def read_num_frame_per_tracklet(path, car_num, cyc_num, ped_num):
    """
    select the some detections
    """
    dets = []
    with open(str(path), 'r') as f:
        lines = f.readlines()
    _dets_ = [line.strip().split(',') for line in lines]
    if _dets_[0][0] in ['Car', 'Van'] :
        car_num.append(len(_dets_))
    if _dets_[0][0] in ['Pedestrian'] :
        ped_num.append(len(_dets_))
    if _dets_[0][0] in ['Cyclist'] :
        cyc_num.append(len(_dets_))

def obtain_number_tracking_frame():
    """
    Evaluation of one dataset in simulation
    """

    # configuration here
    dataset_dir = '/home/ben/Dataset/KITTI/2011_09_26'
    phase_names = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0020_sync','2011_09_26_drive_0084_sync','2011_09_26_drive_0035_sync',
                   '2011_09_26_drive_0005_sync','2011_09_26_drive_0014_sync','2011_09_26_drive_0019_sync','2011_09_26_drive_0059_sync']

    current_class=['Car','Pedestrian','Cyclist']

    car_num = []
    cyc_num = []
    ped_num = []

    dataset_dir = Path(dataset_dir)
    if not isinstance(phase_names, list):
        phase_names = [phase_names]

    det_annos = []
    for phase_name in phase_names:
        det_dir = dataset_dir/phase_name/'tracklets_pc/'
        det_annos_list = list(sorted(det_dir.glob('dets_conf/*.txt')))
        for i in range(len(det_annos_list)):
            read_num_frame_per_tracklet(det_annos_list[i],  car_num, cyc_num, ped_num)

    # car_num.remove(min(car_num))
    # car_num.remove(max(car_num))
    car_num = sorted(car_num)
    print("Car: the number of sequence:{}, the number of tracking frames:{}".format(len(car_num) , np.mean(car_num[2:-2])))

    # ped_num.remove(min(ped_num))
    # ped_num.remove(max(ped_num))
    ped_num = sorted(ped_num)
    print("Pedestrian: the number of sequence:{}, the number of tracking frames:{}".format(len(ped_num) , np.mean(ped_num[2:-2])))
    # print(np.mean(ped_num[2:-2]))

    # cyc_num.remove(min(cyc_num))
    # cyc_num.remove(max(cyc_num))
    cyc_num = sorted(cyc_num)
    # print(np.mean(cyc_num[2:-2]))
    print("Cyclist: the number of sequence:{}, the number of tracking frames:{}".format(len(cyc_num) , np.mean(cyc_num[2:-2])))


## the phases analysis
def _confidence_analysis_one_phase(dataset_dir, phase_name):
    """
    Description of method:
        1. check the consistency of detection class;
        2. if data association is correct, calculate the mean confidence of one class;
        3. compare the mean confidence of detection and that of tracker;
        4. then, get the good instance or bad instance;
    """
    print('Process the phase: {}'.format(phase_name))
    dataset_dir = Path(dataset_dir)
    confidence_dir = dataset_dir/phase_name/'detection/dets_trk_confidence'
    file_list = list(sorted(confidence_dir.glob('*.txt')))
    assert len(file_list)%5 == 0, "The number of file list should be divided by 5"
    num_files = int(len(file_list)/5)

    one_det = 0
    num_good_match = 0
    num_mis_match = 0
    good_instance = 0
    bad_instance  = 0

    num_tracking_all = 0
    _count_ = 0

    frame_num = 5
    less_frame_num = 0

    for id_f in range(num_files):
        # checking mis-association
        detection_class_dir = confidence_dir/ str('detection_class_%06d.txt'% int(id_f))
        detection_class_file = np.loadtxt(detection_class_dir)
        detection_class_file = detection_class_file.astype(np.int).tolist()

        if isinstance(detection_class_file, int):
            one_det += 1
            continue

        if len(detection_class_file) <= frame_num:
            less_frame_num += 1
            continue

        class_gt = detection_class_file[0]
        if detection_class_file != [class_gt ]*len(detection_class_file):
            num_mis_match += 1
            continue
        else:
            num_good_match += 1

        detection_confidence_dir = confidence_dir/ str('detection_confidecne_%06d.txt'% int(id_f))
        detection_confidence_file = np.loadtxt(detection_confidence_dir)
        averge_detection_confidence = np.mean(detection_confidence_file[frame_num:, class_gt])

        num_tracking_all += detection_confidence_file.shape[0]
        _count_ += 1

        tracking_confidence_dir = confidence_dir/ str('tracker_confidence_%06d.txt'% int(id_f))
        tracking_confidence_file = np.loadtxt(tracking_confidence_dir)
        averge_tracking_confidence = np.mean(tracking_confidence_file[frame_num:, class_gt])

        if averge_detection_confidence < averge_tracking_confidence:
            good_instance += 1
        else:
            bad_instance += 1

    # print(one_det)
    print('average tracking No:{}'.format(num_tracking_all/_count_))
    return num_good_match, num_mis_match, good_instance, bad_instance

def confidence_analysis_tracking_detection():
    """
    For a tracklet, the detection and tracker confidence can be found, and should be compared to check whether the tracker helps the
    detection or not. The category, frame_ids, detection confidence, and tracking confidence are all known.

    Objectives: get the stastics about good instances and good instances, and record the corresponding Track_ID

    Definition of good instance: the tracker makes the right class more confident, otherwise, it is the bad instace.

    Mistassociation number: the category changes of one tracklet.

    Confidence histogram: [Bg, Car, Pedestrian, Cyclist]
    """

    dataset_dir = '/home/ben/Dataset/KITTI/2011_09_26'
    # phase_names = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0020_sync','2011_09_26_drive_0035_sync','2011_09_26_drive_0084_sync']
    phase_names = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0020_sync','2011_09_26_drive_0084_sync','2011_09_26_drive_0035_sync',
                   '2011_09_26_drive_0005_sync','2011_09_26_drive_0014_sync','2011_09_26_drive_0019_sync','2011_09_26_drive_0059_sync']


    for phase_name in phase_names:
        num_good_match, num_mis_match, good_instance, bad_instance =  _confidence_analysis_one_phase(dataset_dir, phase_name)
        print("num_good_match:{}, num_mis_match:{}, good_instance:{}, bad_instance:{}".format(num_good_match, num_mis_match, good_instance, bad_instance))


## the efficiency analysis
def efficiency_analysis_all_phases():
    """
    1. read all data;
    1. the efficiency ratio against different range
    """
    all_dists = [10*i for i in range(1,8)]
    dataset_dir = '/home/ben/Dataset/KITTI/2011_09_26'
    # phase_names = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0020_sync','2011_09_26_drive_0035_sync','2011_09_26_drive_0084_sync']
    phase_names = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0005_sync','2011_09_26_drive_0014_sync','2011_09_26_drive_0019_sync',
                   '2011_09_26_drive_0020_sync','2011_09_26_drive_0035_sync','2011_09_26_drive_0059_sync','2011_09_26_drive_0084_sync']

    fig, ax = plt.subplots()
    for phase_name in phase_names:
        ratios = []
        tracklet_path = Path(dataset_dir) / phase_name / 'tracklets_pc'
        all_seqs = list(tracklet_path.glob('seq_*'))

        for dist in all_dists:
            # inialization
            num_det = 0
            num_trk = 0
            # for one distance and all sequence data
            for one_seq in all_seqs:
                _all_pc = list(sorted(one_seq.glob('*.bin')))
                #pre_num_det = num_det
                num_det += len(_all_pc)
                # go through all the point cloud in one sequence
                for _pc_ in _all_pc:
                    points = np.fromfile(str(_pc_) ,dtype=np.float32, count=-1).reshape([-1, 4])
                    p_range = np.sqrt(np.mean(points[:,0])**2 + np.mean(points[:,1])**2)
                    if p_range < dist:
                        #num_det += 1
                        num_trk += 1
                        break

                # if num_det > pre_num_det:
                #    num_trk += 1
            ratios.append(float(num_trk)/float(num_det))
        plt.plot(all_dists, ratios, marker='o', linestyle='dashed', markersize=12, linewidth=2, label='{}'.format(phase_name[17:21]))

    plt.xlabel('Range of interest (m)')
    plt.ylabel('Ratio')
    # plt.ylim(0, 0.04)
    plt.xlim(0, 90)
    ax.legend()
    fig_name = 'efficiency_results.png'
    plt.savefig(fig_name)


def efficiency_analysis_all_phases_with_fusion_model():
    """
    1. read all data;
    1. the efficiency ratio against different range
    """
    all_dists = [10*i for i in range(1,8)]
    dataset_dir = '/home/ben/Dataset/KITTI/2011_09_26'
    # phase_names = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0020_sync','2011_09_26_drive_0035_sync','2011_09_26_drive_0084_sync']
    phase_names = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0005_sync','2011_09_26_drive_0014_sync','2011_09_26_drive_0019_sync',
                   '2011_09_26_drive_0020_sync','2011_09_26_drive_0035_sync','2011_09_26_drive_0059_sync','2011_09_26_drive_0084_sync']

    fig, ax = plt.subplots()
    confusion_ratio = 0.16
    for phase_name in phase_names:
        ratios = []
        tracklet_path = Path(dataset_dir) / phase_name / 'tracklets_pc'
        all_seqs = list(tracklet_path.glob('seq_*'))

        for dist in all_dists:
            # inialization
            num_det = 0
            num_trk = 0
            # for one distance and all sequence data
            for one_seq in all_seqs:
                all_frame_ids = []
                _all_pc = list(sorted(one_seq.glob('*.bin')))
                for _pc_ in _all_pc:
                    _pc_name = _pc_.name
                    _frame_id = int(_pc_name.split('_')[1][:-4])
                    o_type = _pc_name.split('_')[0]
                    all_frame_ids.append(_frame_id)
                # sortting the ids is very critical step
                all_frame_ids = sorted(all_frame_ids)
                #pre_num_det = num_det
                num_det += len(_all_pc)
                # go through all the point cloud in one sequence
                previous_num_points = None

                for one_id in all_frame_ids:
                    pc_file_path = one_seq / '{}_{}.bin'.format(o_type, one_id)
                    points = np.fromfile(str(pc_file_path) ,dtype=np.float32, count=-1).reshape([-1, 4])
                    num_points = points.shape[0]
                    p_range = np.sqrt(np.mean(points[:,0])**2 + np.mean(points[:,1])**2)
                    if p_range < dist:
                        #num_det += 1
                        if previous_num_points is None:
                            previous_num_points = num_points
                            num_trk += 1
                        if (num_points - previous_num_points)/float(previous_num_points) > confusion_ratio :
                            num_trk += 1
                            previous_num_points = num_points

                # if num_det > pre_num_det:
                #    num_trk += 1
            ratios.append(float(num_trk)/float(num_det))
        plt.plot(all_dists, ratios, marker='o', linestyle='dashed', markersize=12, linewidth=2, label='{}'.format(phase_name[17:21]))

    plt.xlabel('Range of interest (m)')
    plt.ylabel('Ratio')
    # plt.ylim(0, 0.04)
    plt.xlim(0, 90)
    ax.legend()
    fig_name = 'efficiency_with_fusion_results.png'
    plt.savefig(fig_name)




if __name__=='__main__':
    fire.Fire()
    """
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
    y_pred = det_annos[:,9]
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
    """

