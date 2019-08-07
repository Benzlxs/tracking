# find the orientation of cluster
import sys
import torch
import numpy as np
from numpy import linalg as npla
from pointnet.model import PointNetCls, feature_transform_regularizer

def find_optimal_heading(xy, ang_reso=90, angle_range = (180-1)*np.pi/180):
    """
    return the optimal heading of fitted bounding boxes
    """
    angle_samples = np.linspace(-angle_range/2, angle_range/2, num=ang_reso) # n
    sin_samples = np.sin(angle_samples)  # n
    sin_samples = np.expand_dims(sin_samples, axis=0)
    cos_samples = np.cos(angle_samples)  # n
    cos_samples = np.expand_dims(cos_samples, axis=0)
    x_sin = np.matmul(np.expand_dims(xy[:,0],axis=1),sin_samples)
    y_sin = np.matmul(np.expand_dims(xy[:,1],axis=1),sin_samples)
    x_cos = np.matmul(np.expand_dims(xy[:,0],axis=1),cos_samples)
    y_cos = np.matmul(np.expand_dims(xy[:,1],axis=1),cos_samples)
    rotated_x = x_cos - y_sin # m*n
    rotated_y = x_sin + y_cos
    x_min = np.amin(rotated_x, axis=0) # n*1
    x_max = np.amax(rotated_x, axis=0)
    y_min = np.amin(rotated_y, axis=0)
    y_max = np.amax(rotated_y, axis=0)

    area = (x_max - x_min)*(y_max - y_min)
    ind = np.argmin(area)

    return -angle_samples[ind]


def find_optimal_bbox3d(xy, ang_reso=90, angle_range = (180-1)*np.pi/180):
    """
    return the optimal 3D fitted bounding boxes
    """
    angle_samples = np.linspace(-angle_range/2, angle_range/2, num=ang_reso) # n
    sin_samples = np.sin(angle_samples)  # n
    sin_samples = np.expand_dims(sin_samples, axis=0)
    cos_samples = np.cos(angle_samples)  # n
    cos_samples = np.expand_dims(cos_samples, axis=0)
    x_sin = np.matmul(np.expand_dims(xy[:,0],axis=1),sin_samples)
    y_sin = np.matmul(np.expand_dims(xy[:,1],axis=1),sin_samples)
    x_cos = np.matmul(np.expand_dims(xy[:,0],axis=1),cos_samples)
    y_cos = np.matmul(np.expand_dims(xy[:,1],axis=1),cos_samples)
    rotated_x = x_cos - y_sin # m*n
    rotated_y = x_sin + y_cos
    x_min = np.amin(rotated_x, axis=0) # n*1
    x_max = np.amax(rotated_x, axis=0)
    y_min = np.amin(rotated_y, axis=0)
    y_max = np.amax(rotated_y, axis=0)

    area = (x_max - x_min)*(y_max - y_min)
    ind = np.argmin(area)

    _ls = x_max - x_min
    _ws = y_max - y_min

    _l = _ls[ind]
    _w = _ws[ind]

    # long edge should be l
    if _l > _w:
        l = _l
        w = _w
    else:
        l = _w
        w = _l

    return -angle_samples[ind], [w, l]


def find_optimal_bbox3d_without_heading(xy, ang_reso=90, angle_range = (180-1)*np.pi/180):
    """
    return the optimal 3D fitted bounding boxes
    """
    angle_samples = np.linspace(0, np.pi/2, num=2) # n
    sin_samples = np.sin(angle_samples)  # n
    sin_samples = np.expand_dims(sin_samples, axis=0)
    cos_samples = np.cos(angle_samples)  # n
    cos_samples = np.expand_dims(cos_samples, axis=0)
    x_sin = np.matmul(np.expand_dims(xy[:,0],axis=1),sin_samples)
    y_sin = np.matmul(np.expand_dims(xy[:,1],axis=1),sin_samples)
    x_cos = np.matmul(np.expand_dims(xy[:,0],axis=1),cos_samples)
    y_cos = np.matmul(np.expand_dims(xy[:,1],axis=1),cos_samples)
    rotated_x = x_cos - y_sin # m*n
    rotated_y = x_sin + y_cos
    x_min = np.amin(rotated_x, axis=0) # n*1
    x_max = np.amax(rotated_x, axis=0)
    y_min = np.amin(rotated_y, axis=0)
    y_max = np.amax(rotated_y, axis=0)

    area = (x_max - x_min)*(y_max - y_min)
    ind = np.argmin(area)

    _ls = x_max - x_min
    _ws = y_max - y_min

    _l = _ls[ind]
    _w = _ws[ind]

    # long edge should be l
    if _l > _w:
        l = _l
        w = _w
    else:
        l = _w
        w = _l

    return -angle_samples[ind], [w, l]


def pca_points(points, correlation = False, sort = True):
    """
    Using the PCA to find the normal of all points
    """
    mean = np.mean(points, axis=0)
    point_adjust = points - mean

    # covert into square matrix
    if correlation:
        matrix = np.corrcoef(data_adjust.T)
    else:
        matrix = np.cov(data_adjust.T)

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if sort:
        #: sort eigenvalues and eigenvectors
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:,sort]
    return eigenvalues, eigenvectors


def points_sampling_to_torch(points, num_points):
    """
    sampling into the fixed number of points
    """
    num_v = points.shape[0]
    if num_v < num_points:
        _gap = num_points - num_v
        _ind = np.random.choice(num_v,_gap,replace=True)
        _gap_points = points[_ind, :]
        points = np.vstack((points, _gap_points))
    else:
        _ind = np.random.choice(num_v, num_points,replace=False)
        points = points[_ind, :]

    points =  points - np.expand_dims(np.mean(points, axis=0), 0)
    dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
    points = points / dist  # scale

    points = np.expand_dims(points, 0)

    points = torch.from_numpy(points.astype(np.float32))

    return points


class ClustersToDets(object):
    """
    To find the orientation and class of every cluster, there are two methdos for orientation:
        1. brute-forcely enumerate all the angles and selecte the best one;
        2. use the matrix decomposition and eigen vector
    The pointnet is used to do classification.
    """
    # LABEL_DICT = {'bg':0, 'Car':1, 'Pedestrian':2, 'Van':3, 'Cyclist':4}
    LABEL_DICT = {0:'bg', 1:'Car', 2:'Pedestrian', 3:'Van', 4:'Cyclist'}

    def __init__(self, config):
        """
        """
        self.angle_resolution = config.angle_resolution

        clustersdetection_dict = {
            "cluster2det":self.clusters_to_dets,
            }
        self.model_path = config.model_path
        # self.class_name = config.class_name
        self.num_sample = config.num_sample
        self.num_classes = config.num_classes
        self.feature_transform = config.feature_transform
        # prepare the model net for classification
        self.classifier = PointNetCls(k=self.num_classes, feature_transform=self.feature_transform)
        self.GPU = config.GPU
        if self.GPU:
            self.classifier.cuda()
        else:
            self.classifier.cpu()
        self.classifier.eval()
        self.classifier.load_state_dict(torch.load(self.model_path))
        self._orientation_ = clustersdetection_dict[config.detection_method]

    def clusters_to_dets(self, clusters, meta, hight=2.4):
        """
        using the enumeration method to find the optimal orientation
        """
        theta = []
        xyz = []
        wlh = []
        name = []
        confidence = []
        if not len(clusters):
            bboxes = np.array([[0., 0., 0., 0., 0., 0., 0.]])
            confidence = 1
            name = 'bg'
            one_det = {
                    "box3d_lidar": bboxes,
                    "scores": confidence,
                    "label_preds":name,
                    "metadata":meta,
                    }
            return [one_det]
        for c in clusters:
            center_xyz = np.mean(c[:,:3], axis=0)
            # center_xyz[2] -= hight/3  # move the center to the bottom
            xyz.append(center_xyz)
            _xy_c = c[:,:2] - center_xyz[:2]
            # heading = find_optimal_heading(_xy_c, ang_reso =self.angle_resolution)
            _heading, _wl = find_optimal_bbox3d(_xy_c, ang_reso =self.angle_resolution)
            _wlh = _wl + [hight]
            theta.append(_heading)
            wlh.append(_wlh)
            # smaple the points
            input_points = points_sampling_to_torch(c, self.num_sample)
            input_points = input_points.transpose(2, 1)
            if self.GPU:
                input_points = input_points.cuda()
            else:
                input_points = input_points.cup()
            # do classification
            pred, _, _ = self.classifier(input_points)
            # do classification
            pred_choice = pred.data.max(1)[1]
            pred_confid = pred.data.max(1)[0]
            pred_choice = int(pred_choice.cpu().numpy())
            name.append(self.LABEL_DICT[pred_choice])
            confidence.append(float(pred_confid.cpu().numpy()))

        wlh = np.array(wlh)
        theta = np.array(theta)
        theta = np.expand_dims(theta, axis=1)
        xyz = np.array(xyz)


        bboxes = np.hstack((xyz, wlh, theta))
        one_det = {
            "box3d_lidar": bboxes,
            "scores": confidence,
            "label_preds":name,
            "metadata":meta,
        }

        return [one_det]


