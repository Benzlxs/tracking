# extract the ground truth data from tracklets

from pathlib import Path
from source import parseTrackletXML as xmlParser
import numpy as np
import pykitti
import fire
from collections import OrderedDict


"""
read all gt boxes
save them into the folder as txt: name bounding boxes
"""


def load_dataset(root_dir, date, drive, calibrated=False, frame_range=None):
    """
    Loads the dataset with `date` and `drive`.

    Parameters
    ----------
    date        : Dataset creation date.
    drive       : Dataset drive.
    calibrated  : Flag indicating if we need to parse calibration data. Defaults to `False`.
    frame_range : Range of frames. Defaults to `None`.

    Returns
    -------
    Loaded dataset of type `raw`.
    """
    dataset = pykitti.raw(root_dir, date, drive)

    # Load the data
    if calibrated:
        dataset.load_calib()  # Calibration data are accessible as named tuples

    np.set_printoptions(precision=4, suppress=True)
    print('\nDrive: ' + str(dataset.drive))
    print('\nFrame range: ' + str(dataset.frames))

    if calibrated:
        print('\nIMU-to-Velodyne transformation:\n' + str(dataset.calib.T_velo_imu))
        print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
        print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

    return dataset

def load_tracklets_for_frames(n_frames, xml_path):
    """
    Loads dataset labels also referred to as tracklets, saving them individually for each frame.

    Parameters
    ----------
    n_frames    : Number of frames in the dataset.
    xml_path    : Path to the tracklets XML.

    Returns
    -------
    Tuple of dictionaries with integer keys corresponding to absolute frame numbers and arrays as values. First array
    contains coordinates of bounding box vertices for each object in the frame, and the second array contains objects
    types as strings.
    """
    tracklets = xmlParser.parseXML(xml_path)

    frame_tracklets = {}
    frame_tracklets_types = {}
    frame_xyz = {}
    frame_lwh = {}
    frame_occlusion = {}
    frame_theta = {}
    frame_track_id = {}

    num_car = 0
    num_ped = 0
    num_cyc = 0
    for i in range(n_frames):
        frame_tracklets[i] = []
        frame_tracklets_types[i] = []
        frame_xyz[i] = []
        frame_occlusion[i] = []
        frame_lwh[i] = []
        frame_theta[i] = []
        frame_track_id[i] = []
    # loop over tracklets
    for i, tracklet in enumerate(tracklets):
        # this part is inspired by kitti object development kit matlab code: computeBox3D
        h, w, l = tracklet.size
        # in velodyne coordinates around zero point and without orientation yet
        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]
        ])
        # loop over all data in tracklet
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:
            # determine if object is in the image; otherwise continue
            # if truncation not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
            #    continue
            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]  # other rotations are supposedly 0
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rotMat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0]
            ])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
            frame_tracklets[absoluteFrameNumber] = frame_tracklets[absoluteFrameNumber] + [cornerPosInVelo]
            frame_tracklets_types[absoluteFrameNumber] = frame_tracklets_types[absoluteFrameNumber] + [
                tracklet.objectType]
            frame_xyz[absoluteFrameNumber] = frame_xyz[absoluteFrameNumber] + [translation]
            frame_lwh[absoluteFrameNumber] = frame_lwh[absoluteFrameNumber] + [[l, w, h]]
            frame_theta[absoluteFrameNumber] = frame_theta[absoluteFrameNumber] + [yaw]
            frame_track_id[absoluteFrameNumber] = frame_track_id[absoluteFrameNumber] + [i]
            frame_occlusion[absoluteFrameNumber] = frame_occlusion[absoluteFrameNumber] + [occlusion]

            if tracklet.objectType in ['Car']:
                num_car += 1
            if tracklet.objectType in ['Cyclist']:
                num_cyc += 1
            if tracklet.objectType in ['Pedestrian']:
                num_ped += 1
    print("The number of car: {}".format(num_car))
    print("The number of cyc: {}".format(num_cyc))
    print("The number of ped: {}".format(num_ped))

    return (frame_tracklets, frame_tracklets_types, frame_xyz, frame_lwh, frame_theta, frame_track_id, frame_occlusion)


# save all ground truth detection
def save_dets(dataset_root='/home/ben/Dataset/KITTI',
              date='2011_09_26',
              drive='0084'):
    # read all detections
    dataset = load_dataset(dataset_root, date, drive)
    tracklet_rects, tracklet_types, frame_xyz, frame_lwh, frame_theta, frame_track_id = load_tracklets_for_frames(len(list(dataset.velo)),
                                                               '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(dataset_root, date, date, drive))

    det_path = Path(dataset_root)/ '{}/{}/{}_drive_{}_sync/detection/gt'.format(dataset_root, date, date, drive)
    det_path.mkdir(parents=True, exist_ok=True)
    for i in range(len(list(dataset.velo))):
        file_path = det_path/ "{:06d}.txt".format(i)
        assert len(tracklet_rects[i]) == len(tracklet_types[i]), "number of bounding boxes should be the same to number of types"
        with open(file_path, 'w') as f:
            for k in range(len(tracklet_rects[i])):
                #trackletBox = np.array([
                #    [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                #    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                #    [0.0, 0.0, 0.0, 0.0, h, h, h, h]])
                #ty= tracklet_types[i][k]
                #x = (tracklet_rects[i][k][0,0] + tracklet_rects[i][k][0,2])/2
                #y = (tracklet_rects[i][k][1,0] + tracklet_rects[i][k][1,1])/2
                #z = (tracklet_rects[i][k][2,0] + tracklet_rects[i][k][2,4])/2
                #l = tracklet_rects[i][k][0,0] - tracklet_rects[i][k][0,2]
                #w = tracklet_rects[i][k][1,1] - tracklet_rects[i][k][1,0]
                #h = tracklet_rects[i][k][2,4] - tracklet_rects[i][k][2,0]
                ty= tracklet_types[i][k]
                x = frame_xyz[i][k][0]
                y = frame_xyz[i][k][1]
                z = frame_xyz[i][k][2]
                l = frame_lwh[i][k][0]
                w = frame_lwh[i][k][1]
                h = frame_lwh[i][k][2]
                theta = frame_theta[i][k]
                track_id = frame_track_id[i][k]
                f.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n'%(ty, x, y, z, l, w, h, theta,track_id))



def save_dets_kitti_format(dataset_root='/home/ben/Dataset/KITTI',
              date='2011_09_26',
              drive='0084'):
    # read all detections
    dataset = load_dataset(dataset_root, date, drive)
    tracklet_rects, tracklet_types, frame_xyz, frame_lwh, frame_theta, frame_track_id, frame_occlusion = load_tracklets_for_frames(len(list(dataset.velo)),
                                                               '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(dataset_root, date, date, drive))

    det_path = Path(dataset_root)/ '{}/{}/{}_drive_{}_sync/detection/gt'.format(dataset_root, date, date, drive)
    det_path.mkdir(parents=True, exist_ok=True)
    for i in range(len(list(dataset.velo))):
        file_path = det_path/ "{:06d}.txt".format(i)
        assert len(tracklet_rects[i]) == len(tracklet_types[i]), "number of bounding boxes should be the same to number of types"
        result_lines = []
        for k in range(len(tracklet_rects[i])):
            #trackletBox = np.array([
            #    [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            #    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            #    [0.0, 0.0, 0.0, 0.0, h, h, h, h]])
            #ty= tracklet_types[i][k]
            #x = (tracklet_rects[i][k][0,0] + tracklet_rects[i][k][0,2])/2
            #y = (tracklet_rects[i][k][1,0] + tracklet_rects[i][k][1,1])/2
            #z = (tracklet_rects[i][k][2,0] + tracklet_rects[i][k][2,4])/2
            #l = tracklet_rects[i][k][0,0] - tracklet_rects[i][k][0,2]
            #w = tracklet_rects[i][k][1,1] - tracklet_rects[i][k][1,0]
            #h = tracklet_rects[i][k][2,4] - tracklet_rects[i][k][2,0]
            ty= tracklet_types[i][k]
            x = frame_xyz[i][k][0]
            y = frame_xyz[i][k][1]
            z = frame_xyz[i][k][2]
            l = frame_lwh[i][k][0]
            w = frame_lwh[i][k][1]
            h = frame_lwh[i][k][2]
            occlusion = frame_occlusion[i][k][0]
            theta = frame_theta[i][k]
            track_id = frame_track_id[i][k]
            result_dict = {
                'name': ty,
                'alpha': theta,
                'occluded': int(occlusion),
                'bbox': [0,0,100,100] ,
                'location': [x,y,z],
                'dimensions': [l,w,h],
                'rotation_y': theta,
                'score':1.0,
            }
            result_line = kitti_result_line(result_dict)
            # f.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n'%(ty, x, y, z, l, w, h, theta,track_id))
            result_lines.append(result_line)
        result_str = '\n'.join(result_lines)
        with open(file_path, 'w') as f:
            f.write(result_str)
        # with open(file_path, 'w') as f:




def kitti_result_line(result_dict, precision=4):
    prec_float = "{" + ":.{}f".format(precision) + "}"
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', 0.0),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError("you must specify a value for {}".format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError("unknown key. supported key:{}".format(
                res_dict.keys()))
    return ' '.join(res_line)



if __name__=='__main__':
    fire.Fire()
