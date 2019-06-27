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
        self.det_list = list(sorted(dataset_dir.glob('detection/gt/*.txt')))
        self.oxts_list = list(sorted(dataset_dir.glob('oxts/data/*.txt')))
        assert len(self.img_list) == len(self.pc_list), "the image and point cloud numenbr should be the same"
        assert len(self.pc_list) == len(self.det_list), "the numebr of detection and that of point cloud should be the same"
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

    def __len__(self):
        return len(self.pc_list)

    def get_detection(self, frame_id):
        """
        fetch detection data
        format: class, x, y, z, w, l, h
        """
        assert frame_id<len(self.det_list),"number of detection is larger than that of point blocks"
        dets = []
        with open(str(self.det_list[frame_id]), 'r') as f:
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









