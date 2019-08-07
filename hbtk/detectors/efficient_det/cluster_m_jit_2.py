## here is a bunch of clustering methods to do clustering

import numpy as np
import numba
import time
from numba import types
from numba.typed import Dict
from utils.geometry_utils import dist_points_to_points
small_number = np.finfo(np.float32).tiny


@numba.njit
def dist_points_to_points_jit(points_1, points_2, h_dist, min_connect_points, FLAG):
    """
    calculate the euclidean distance between two sets of points
    """
    points_2 = np.expand_dims(points_2, axis=1)
    all_dist = (points_1[:, :, 0] - points_2[:, :, 0])**2 + (
        points_1[:, :, 1] - points_2[:, :, 1])**2 + (points_1[:, :, 2] -
                                                     points_2[:, :, 2])**2
    all_dist = all_dist + small_number
    all_dist = np.sqrt(all_dist)
    _ind_con = np.where(all_dist < h_dist)
    if _ind_con[0].shape[0] > min_connect_points:
        FLAG = True
    #return _ind_con


@numba.njit
def find_all_scans(points, angle_dif, min_scan_points):
    """
    convert the cartesian coordinate into polar then extract all scans
    """
    polar_theta = np.arccos(
        points[:, 1] /
        np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2))
    num_points = polar_theta.size
    diff = polar_theta[1:num_points] - polar_theta[:num_points - 1]
    break_index = np.where(diff > angle_dif)[0]
    # num_scans should be one more than num of breaks
    num_scans = break_index.size + 1
    # start point of the first scan is the 0
    scans = np.zeros((num_scans, 2), dtype=numba.types.uint32)
    # the end point of last scan is the last point
    scans[num_scans - 1, 1] = num_points

    scans[:num_scans - 1, 1] = break_index
    scans[1:num_scans, 0] = break_index + 1

    # check the every scan points number
    scan_points_num = scans[:, 1] - scans[:, 0]

    abandoned_scans = np.where(scan_points_num < min_scan_points)[0]
    if abandoned_scans.size > 0:
        #scans = np.delete(scans, abandoned_scans, 0)
        scans = delete_workaround(scans, abandoned_scans)

    return scans


#@numba.jit('float32[:, :], float32, uint32, uint32, uint32[:,:]', nopython=True)
@numba.njit
def find_segs_one_scan_sub(points, H_dist, min_seg_point, num_points):
    adjacent_dist = np.sqrt( (points[0:num_points-1,0] - points[1:num_points,0])**2 + \
                             (points[0:num_points-1,1] - points[1:num_points,1])**2 + \
                             (points[0:num_points-1,2] - points[1:num_points,2])**2 + \
                             small_number )
    _b_ind = np.where(adjacent_dist > H_dist[0])
    break_index = _b_ind[0]
    num_segs = break_index.size + 1
    # start point of the first scan is the 0
    segs = np.zeros((num_segs, 2), dtype=numba.types.uint32)
    # the end point of last scan is the last point
    segs[num_segs - 1, 1] = num_points - 1
    segs[:num_segs - 1, 1] = break_index
    segs[1:num_segs, 0] = break_index + 1

    # check the every seg points number
    seg_points_num = segs[:, 1] - segs[:, 0]
    _abandoned_segs = np.where(seg_points_num < min_seg_point)
    abandoned_segs = _abandoned_segs[0]

    return abandoned_segs, segs, num_segs


@numba.njit
def delete_workaround(arr, num):
    mask = np.zeros(arr.shape[0], dtype=np.int64) == 0
    mask[num] = False
    return arr[mask]


#@numba.njit
def find_segs_one_scan(points, H_dist, min_seg_point, start_point):
    """
    find all the segments in one scan
    """
    num_points = points.shape[0]
    ret = find_segs_one_scan_sub(points, H_dist, min_seg_point, num_points)
    abandoned_segs, segs, num_segs = ret
    if abandoned_segs.size > 0:
        #segs = np.delete(segs, abandoned_segs, 0)
        segs = delete_workaround(segs, abandoned_segs)
        num_segs = num_segs - abandoned_segs.size

    segs = segs + start_point
    #seg_dict = {
    #    x: points[segs[x, 0]:(segs[x, 1] + 1), :3] for x in range(num_segs)
    #}
    # seg_dict = Dict.empty( key_type=types.uint32, value_type=types.float64[:], )
    # for x in range(num_segs):
    #    seg_dict[x] = points[segs[x, 0]:(segs[x, 1] + 1), :3].astype(np.float64)
    if num_segs == 0:
        return [[0, 0]], 0
    else:
        seg_list = []
        #label_list = []
        for i in range(num_segs):
            seg_list.append([segs[i, 0], segs[i, 1] + 1])
            #label_list.append(x)

        return seg_list, num_segs


#@numba.njit
def find_connection(one_seg, all_segs, above_label_list,h_dist=0.5, min_connect_points=2):
    """
    calculate the connections
    """
    connects = []
    points_1 = np.expand_dims(one_seg, axis=0)
    for i in range(len(above_label_list)):
        #points_2 = np.expand_dims( all_segs[label], axis=1)
        #all_dist = dist_points_to_points(points_1, points_2)
        #_ind_con = np.where(all_dist<h_dist)
        #_ind_con = dist_points_to_points_jit(points_1, points_2, h_dist)
        FLAG = False
        dist_points_to_points_jit(points_1, all_segs[i], h_dist,
                                         min_connect_points, FLAG)
        if FLAG:
            connects.append(above_label_list[i])
    return connects


class PointCluster(object):
    """
    To cluster the points
    """

    def __init__(self, config):
        self.clustering_method = config.clustering_method
        self.dist_threshold = config.dist_threshold
        self.H_dist = config.H_dist
        self.V_dist = config.V_dist
        self.min_connect_points = config.min_connect_points
        self.angle_dif_threshold = config.angle_dif_threshold
        self.min_points_per_scan = config.min_points_per_scan
        self.min_seg_point = config.min_seg_point
        clustering_dict = {
            "euclidean_dist_": self.euclidean_dist_,
            "scan_run_": self.scan_run_,
        }
        self._cluster_ = clustering_dict[config.clustering_method]

    def euclidean_dist_(self, points):
        """
        enclidean distance based clustering method
        """
        T_dist = self.dist_threshold
        points_0 = points
        points_1 = np.expand_dims(points, axis=0)
        points_2 = np.expand_dims(points, axis=1)
        all_dist = dist_points_to_points(points_1, points_2)
        clusters = []
        num_points = []
        # stop until the all_dist is empty
        while all_dist.shape[0]:
            STOP = False
            c_1 = []
            c_1.append(0)
            index_new = np.array([0])
            while not STOP:
                index_new = np.where(all_dist[index_new[:], :] < T_dist)
                index_new = set(index_new[1]) - set(c_1)
                index_new = list(index_new)
                if index_new == []:
                    STOP = True
                else:
                    c_1 = c_1 + index_new
                    index_new = np.array(index_new)
            c_1 = np.array(c_1)
            clusters.append(points_0[c_1[:], :])
            num_points.append(c_1.shape[0])

            # delete these points
            all_dist = np.delete(all_dist, c_1, 0)
            all_dist = np.delete(all_dist, c_1, 1)
            points_0 = np.delete(points_0, c_1, 0)

        return clusters, num_points


    def scan_run_(self, points):
        """
        clustering methods based on laser scan order, which more accurate
        arguments:
                points: 3D point cloud N*3
                H_dist: distance threshold inside one scan, horizontal direction.
                V_dist: distance thresold between two scans, vertical direction.
        """
        angle_dif = self.angle_dif_threshold
        min_scan_points = self.min_points_per_scan
        min_seg_point = self.min_seg_point
        min_connect_points = self.min_connect_points
        V_dist = self.V_dist,
        H_dist = self.H_dist
        # the start and end points of all scan
        all_scans = find_all_scans(points[:, :3], angle_dif, min_scan_points)
        num_scans = all_scans.shape[0]
        # initializaton --> class or parameters
        label_vector_global = np.full((points.shape[0]),-1,np.int32)

        # global_segs = {}

        global_label = 0
        start_point = all_scans[0, 0]
        end_point = all_scans[0, 1]
        above_index_list, num_segs = find_segs_one_scan(
            points[start_point:end_point, :], V_dist, min_seg_point, start_point)
        global_label = num_segs - 1
        global_index_list = above_index_list.copy()
        global_label_list = [i for i in range(num_segs)]
        above_label_list = global_label_list.copy()

        above_points_list = []
        for i in range(num_segs):
            above_points_list.append(points[above_index_list[i][0]:above_index_list[i][1], :])
            label_vector_global[above_index_list[i][0]:above_index_list[i][1]] = global_label_list[i]
        # main loop
        for i in range(1, num_scans):
            start_point = all_scans[i, 0]
            end_point = all_scans[i, 1] + 1
            current_index_list,  num_segs = find_segs_one_scan(
                points[start_point:end_point, :], V_dist, min_seg_point, start_point)

            # _temp_dict = {}
            _temp_points_list = []
            _temp_label_list = []
            if num_segs == 0:
                continue
            for j in range(num_segs):

                points_cur = points[current_index_list[j][0]:current_index_list[j][1], :]
                if  len(above_label_list)>0:
                    connect_index = find_connection(points_cur, above_points_list, above_label_list,
                                                H_dist, min_connect_points)
                    num_connect = len(connect_index)
                else:
                    num_connect = 0
                    # break

                # not connection, create new label
                if num_connect == 0:
                    global_label = global_label + 1
                    # global_segs[global_label] = current_segs[j]
                    global_label_list.append(global_label)
                    global_index_list.append(current_index_list[j])
                    label_vector_global[current_index_list[j][0]:current_index_list[j][1]] = global_label

                    _temp_points_list.append(points_cur)
                    _temp_label_list.append(global_label)
                # only one connection, merge it into global segs and check _temp_dict
                if num_connect == 1:
                    label_vector_global[current_index_list[j][0]:current_index_list[j][1]] = connect_index[0]
                    # check whether min_label has been added to _temp_dict or not
                    if connect_index[0] in _temp_label_list:
                        _loc_local_ = _temp_label_list.index(connect_index[0])
                        _temp_points_list[_loc_local_] = np.vstack(
                            (_temp_points_list[_loc_local_], points_cur))
                    else:
                        #_temp_dict[connect_index[0]] = current_segs[j]
                        _temp_points_list.append(points_cur)
                        _temp_label_list.append(connect_index[0])

                    _loc_global = global_label_list.index(connect_index[0])
                    global_index_list[_loc_global] +=current_index_list[j]


                # more two connection, merge the global segments and above_segs, or check duplication in _temp_dict
                if num_connect >= 2:
                    min_label = min(connect_index)
                    connect_index.remove(min_label)
                    # check whether min_label has been added to _temp_dict or not
                    if min_label in _temp_label_list:
                        _loc_local_ = _temp_label_list.index(min_label)
                        _temp_points_list[_loc_local_] = np.vstack(
                            (_temp_points_list[_loc_local_], points_cur))
                    else:
                        _temp_points_list.append(points_cur)
                        _temp_label_list.append(min_label)
                    # merge the current seg points into global segments
                    # global_segs[min_label] = np.vstack(
                        #(global_segs[min_label], current_segs[j]))
                    label_vector_global[current_index_list[j][0]:current_index_list[j][1]] = min_label
                    _loc_global_ = global_label_list.index(min_label)
                    global_index_list[_loc_global_]+=current_index_list[j]
                    ## merge and delete global label and segment
                    for m in connect_index:
                        # merge globally
                        #global_segs[min_label] = np.vstack(
                        #    (global_segs[min_label], global_segs[m]))
                        _loc_global_1 = global_label_list.index(min_label)
                        _loc_global_2 = global_label_list.index(m)
                        lst_index_2 = global_index_list[_loc_global_2]
                        global_index_list[_loc_global_1] += lst_index_2
                        # the index in list in list of pairs
                        for i in range((len(lst_index_2)/2)):
                            l_i = 2*i
                            label_vector_global[lst_index_2[l_i]:lst_index_2[l_i+1]]=min_label
                        # delete globally
                        global_label_list.pop(_loc_global_2)
                        global_index_list.pop(_loc_global_2)

                        # merge locally
                        _loc_local_1 = above_label_list.index(min_label)
                        _loc_local_2 = above_label_list.index(m)
                        above_points_list[_loc_local_1] = np.vstacka(
                        (above_points_list[_loc_local_1], above_points_list[_loc_local_2] )
                        )
                        # delete locally
                        above_label_list.pop(_loc_local_2)
                        above_points_list.pop(_loc_local_2)

                        if m in _temp_label_list:
                            _loc_local_1 = _temp_label_list.index(min_label)
                            _loc_local_2 = _temp_label_list.index(m)
                            _temp_points_list[_loc_local_1] = np.vstack(
                            (_temp_points_list[_loc_local_1], _temp_points_list[_loc_local_2])
                            )
                            # delete in temporaral vector
                            _temp_label_list.pop(_loc_local_2)
                            _temp_points_list.pop(_loc_local_2)

            above_points_list = _temp_points_list.copy()
            above_label_list = _temp_label_list.copy()

        # convert the clusters from dictionary into list
        # segs_list = [global_segs[i] for i in global_segs.keys()]
        segs_list = []
        for i in global_label_list:
            _ind_ = np.where(label_vector_global==i)[0]
            segs_list.append(points[_ind_[:],:3])

        #return global_label_list, label_vector_global
        return segs_list, len(segs_list)
