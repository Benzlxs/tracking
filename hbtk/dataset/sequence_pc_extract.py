# extract the ground truth data from tracklets

from pathlib import Path
from source import parseTrackletXML as xmlParser
import numpy as np
import pykitti
import fire
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(ROOT_DIR))
sys.path.append(os.path.join( os.path.dirname(ROOT_DIR), 'hbtk'))

from hbtk.utils import box_np_ops


def _save_one_sequence_(xml_path, sequence_path, mini_tracking_count = 3):
    """
    1. read all the tracklet;
    2. save the sequence of point cloud into corresponding name folder;
    """
    tracklets = xmlParser.parseXML(xml_path)
    print('There are {} trackers'.format(len(tracklets)))

    for i, tracklet in enumerate(tracklets):
        h, w, l = tracklet.size

        if tracklet.nFrames < mini_tracking_count:
            continue

        if tracklet.objectType not in ['Car', 'Cyclist', 'Pedestrian']:
            continue

        one_seq_folder = sequence_path / 'seq_{}'.format(i)
        one_seq_folder.mkdir(parents=True, exist_ok=True)

        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:

            yaw    = rotation[2]
            o_type = tracklet.objectType
            xyz = translation

            # read the point cloud
            _pc_path = str(sequence_path).replace('tracklets_pc','reduced_points/data/') + '%010d.bin'%absoluteFrameNumber
            points = np.fromfile( _pc_path, dtype=np.float32, count=-1).reshape([-1, 4])

            large_ratio = 1.2
            xyzlwhr_lidar = np.array([[xyz[0], xyz[1], xyz[2]+0.1, l*large_ratio, w*large_ratio, h*1.5, -yaw]])
            indices = box_np_ops.points_in_rbbox(points, xyzlwhr_lidar, z_axis=2, origin=(0.5,0.5,0))

            object_points = points[indices[:,0],:]
            if object_points.shape[0]==0:
                continue

            # save the point cloud with name_format = type_framenumber
            _one_pc_file = one_seq_folder / '{}_{}.bin'.format(o_type, absoluteFrameNumber)

            with open(str(_one_pc_file), 'w') as f:
                object_points.tofile(f)

    print('Finish...')


# save all ground truth detection
def save_sequence_pc_frames(dataset_root='/home/ben/Dataset/KITTI',
                     date='2011_09_26',
                     drive='0001'):
    # read all detections
    dataset_root = '/home/ben/Dataset/KITTI/2011_09_26'
    phases = ['2011_09_26_drive_0001_sync','2011_09_26_drive_0020_sync',
              '2011_09_26_drive_0035_sync','2011_09_26_drive_0084_sync',
              '2011_09_26_drive_0005_sync','2011_09_26_drive_0014_sync',
              '2011_09_26_drive_0019_sync','2011_09_26_drive_0059_sync']

    #phases = ['2011_09_26_drive_0001_sync']
    dataset_root = Path(dataset_root)
    for phase in phases:
        print('Process the dataset {}'.format(phase))
        xml_path = dataset_root / phase / 'tracklet_labels.xml'
        sequence_path = dataset_root / phase / 'tracklets_pc'
        sequence_path.mkdir(parents=True, exist_ok=True)
        _save_one_sequence_(xml_path, sequence_path)


if __name__=='__main__':
    fire.Fire()
