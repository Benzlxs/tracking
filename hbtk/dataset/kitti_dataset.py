# the dataset class to read KITTI

import numpy as np
from pathlib import Path
from hbtk.utils.utils import load_one_oxts_packets_and_poses

class Kitti_dataset(object):
    """
    pre-read all pose information,
    pre-read list of all point cloud
    list of image
    list of gt_detection
    """
    def __init__(self, config):
        self.database_dir = config.database_dir
        dataset_dir = Path(self.database_dir) / config.phase
        # get dataset list
        self.img_list = list(sorted(dataset_dir.glob('image_02/data/*.png')))
        self.pc_list  = list(sorted(dataset_dir.glob('velodyne_points/data/*.bin')))
        self.reduce_pc_list  = list(sorted(dataset_dir.glob('reduced_points/data/*.bin')))
        self.gt_list = list(sorted(dataset_dir.glob('detection/gt/*.txt')))
        self.det_gt_list = list(sorted(dataset_dir.glob('detection/dets_gt/*.txt')))
        self.det_classification_list = list(sorted(dataset_dir.glob('detection/dets_class/*.txt')))
        self.oxts_list = list(sorted(dataset_dir.glob('oxts/data/*.txt')))
        assert len(self.img_list) == len(self.pc_list), "the image and point cloud numenbr should be the same"
        assert len(self.pc_list) == len(self.gt_list), "the numebr of detection and that of point cloud should be the same"
        ###

        ##  pre-fetch all the pose
        self.pose = []
        self.yaw = []
        self.pose.append(np.eye(4,dtype=np.float32))
        # set the first frame the orignal point
        oxts_raw_file, scale, origin, yaw_0 = load_one_oxts_packets_and_poses(self.oxts_list[0])
        origin_pose_inv = np.linalg.inv(oxts_raw_file.T_w_imu)
        self.yaw.append(0)
        for i in range(1, len(self.oxts_list)):
            oxts_raw_file, yaw = load_one_oxts_packets_and_poses(self.oxts_list[i] ,scale=scale, origin=origin)
            next_pose = oxts_raw_file.T_w_imu
            self.pose.append(origin_pose_inv.dot(next_pose).astype(np.float32))
            self.yaw.append(yaw - yaw_0)


        # read calibration data
        self.calib_data_path = dataset_dir/ 'calibration'
        _velo_to_camera_txt = self.calib_data_path/'calib_velo_to_cam.txt'
        self.calib_data_velo_2_cam = {}
        with open(str(_velo_to_camera_txt),'r') as f:
            for line in f.readlines():
                 key, value = line.split(':', 1)
                 try:
                    self.calib_data_velo_2_cam[key] = np.array([float(x) for x in value.split()])
                 except ValueError:
                    pass
        _cam_to_cam_txt = self.calib_data_path/'calib_cam_to_cam.txt'
        self.calib_data_cam_2_cam = {}
        with open(str(_cam_to_cam_txt),'r') as f:
            for line in f.readlines():
                 key, value = line.split(':', 1)
                 try:
                    self.calib_data_cam_2_cam[key] = np.array([float(x) for x in value.split()])
                 except ValueError:
                    pass

    def __len__(self):
        return len(self.pc_list)

    def get_gt(self, frame_id):
        """
        fetch detection data
        format: class, x, y, z, l, w, h
        """
        assert frame_id<len(self.gt_list),"number of detection is larger than that of point blocks"
        dets = []
        with open(str(self.gt_list[frame_id]), 'r') as f:
            lines = f.readlines()
        _dets_ = [line.strip().split(',') for line in lines]
        for _det in _dets_:
            if _det[0] in ['Bg','Car', 'Van', 'Pedestrian', 'Cyclist'] :
                if _det[0] in ['Bg']:
                    _det[0] = 0
                if _det[0] in ['Car', 'Van'] :
                    _det[0] = 1
                if _det[0] in ['Pedestrian'] :
                    _det[0] = 2
                if _det[0] in ['Cyclist'] :
                    _det[0] = 3
                dets.append(_det)

        return dets


    def get_detection_gt(self, frame_id):
        """
        fetch detection data
        format: class, x, y, z, l, w, h
        """
        assert frame_id<len(self.det_gt_list),"number of detection is larger than that of point blocks"
        dets = []
        with open(str(self.det_gt_list[frame_id]), 'r') as f:
            lines = f.readlines()
        _dets_ = [line.strip().split(',') for line in lines]
        for _det in _dets_:
            if _det[0] in ['Bg','Car', 'Van', 'Pedestrian', 'Cyclist'] :
                if _det[0] in ['Bg']:
                    _det[0] = 0
                if _det[0] in ['Car', 'Van'] :
                    _det[0] = 1
                if _det[0] in ['Pedestrian'] :
                    _det[0] = 2
                if _det[0] in ['Cyclist'] :
                    _det[0] = 3
                dets.append(_det)

        return dets

    def get_detection_class(self, frame_id):
        """
        fetch detection data
        format: class, x, y, z, l, w, h
        """
        assert frame_id<len(self.det_gt_list),"number of detection is larger than that of point blocks"
        dets = []
        with open(str(self.det_classification_list[frame_id]), 'r') as f:
            lines = f.readlines()
        _dets_ = [line.strip().split(',') for line in lines]
        for _det in _dets_:
            if _det[0] in ['Bg','Car', 'Van', 'Pedestrian', 'Cyclist'] :
                if _det[0] in ['Bg']:
                    _det[0] = 0
                if _det[0] in ['Car', 'Van'] :
                    _det[0] = 1
                if _det[0] in ['Pedestrian'] :
                    _det[0] = 2
                if _det[0] in ['Cyclist'] :
                    _det[0] = 3
                dets.append(_det[:-1])

        return dets

