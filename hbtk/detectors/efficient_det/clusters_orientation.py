# find the orientation of cluster
import sys
import numpy as np
from numpy import linalg as npla

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


class ClustersOrientation(object):
    """
    To find the orientation of every cluster, there are two methdos:
        1. brute-forcely enumerate all the angles and selecte the best one;
        2. use the matrix decomposition and eigen vector
    """
    def __init__(self, config):
        """
        """
        self.angle_resolution = config.angle_resolution

        clustersorientation_dict = {
            "orientation_enumeration": self.orientation_enumeration,
            "clusters_bbox":self.clusters_bbox,
            "clusters_bbox_without_heading":self.clusters_bbox_without_heading,
            "orientation_eigen_vector": self.orientation_eigen_vector,
            }

        self._orientation_ = clustersorientation_dict[config.orientation_method]

    def clusters_bbox(self, clusters, hight=1):
        """
        using the enumeration method to find the optimal orientation
        """
        theta = []
        xyz = []
        wlh = []
        if not len(clusters):
            return np.array([[0., 0., 0., 0., 0., 0., 0.]])
        for c in clusters:
            center_xyz = np.mean(c[:,:3], axis=0)
            xyz.append(center_xyz)
            _xy_c = c[:,:2] - center_xyz[:2]
            # heading = find_optimal_heading(_xy_c, ang_reso =self.angle_resolution)
            _heading, _wl = find_optimal_bbox3d(_xy_c, ang_reso =self.angle_resolution)
            _wlh = _wl + [hight]
            theta.append(_heading)
            wlh.append(_wlh)
        wlh = np.array(wlh)
        theta = np.array(theta)
        theta = np.expand_dims(theta, axis=1)
        xyz = np.array(xyz)

        bboxes = np.hstack((xyz, wlh, theta))
        return bboxes


    def clusters_bbox_without_heading(self, clusters, hight=1):
        """
        using the enumeration method to find the optimal orientation
        """
        theta = []
        xyz = []
        wlh = []
        if not len(clusters):
            return np.array([[0., 0., 0., 0., 0., 0., 0.]])
        for c in clusters:
            center_xyz = np.mean(c[:,:3], axis=0)
            xyz.append(center_xyz)
            _xy_c = c[:,:2] - center_xyz[:2]
            # heading = find_optimal_heading(_xy_c, ang_reso =self.angle_resolution)
            _heading, _wl = find_optimal_bbox3d_without_heading(_xy_c, ang_reso =self.angle_resolution)
            _wlh = _wl + [hight]
            theta.append(_heading)
            wlh.append(_wlh)
        wlh = np.array(wlh)
        theta = np.array(theta)
        theta = np.expand_dims(theta, axis=1)
        xyz = np.array(xyz)

        bboxes = np.hstack((xyz, wlh, theta))
        return bboxes


    def orientation_enumeration(self, clusters, hight = 1):
        """
        using the enumeration method to find the optimal orientation
        """
        theta = []
        for c in clusters:
            center_xy = np.mean(c[:,:2], axis=0)
            _xy_c = c[:,:2] - center_xy
            heading = find_optimal_heading(_xy_c, ang_reso =self.angle_resolution)
            theta.append(heading)

        return theta

    def orientation_eigen_vector(self, clusters):
        """
        using the matrix decomposition to calculate the orientation of all points
        """
        theta = []
        for c in clusters:
            w, v = pca_points(c)
            # convert the normal into heading
            normal = v[:,2]
            heading = np.arctan2(normal[0], normal[1])
            theta.append(theta)

        return theta
