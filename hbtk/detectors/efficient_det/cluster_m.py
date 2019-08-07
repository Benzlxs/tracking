## here is a bunch of clustering methods to do clustering

import numpy as np
from utils.geometry_utils import dist_points_to_points

def find_all_scans(points, angle_dif, min_scan_points):
    """
    convert the cartesian coordinate into polar then extract all scans
    """
    polar_theta = np.arccos(points[:,1]/np.sqrt(points[:,0]**2 + points[:,1]**2 + points[:,2]**2))
    num_points = polar_theta.size
    diff = polar_theta[1:num_points] - polar_theta[:num_points-1]
    break_index = np.where(diff>angle_dif)[0]
    # num_scans should be one more than num of breaks
    num_scans = break_index.size + 1
    # start point of the first scan is the 0
    scans = np.zeros([num_scans,2],dtype=np.uint32)
    # the end point of last scan is the last point
    scans[num_scans-1,1] = num_points

    scans[:num_scans-1, 1] = break_index
    scans[1:num_scans,0] = break_index + 1

    # check the every scan points number
    scan_points_num = scans[:,1] - scans[:,0]

    abandoned_scans = np.where(scan_points_num<min_scan_points)[0]
    if abandoned_scans.size>0:
        scans = np.delete(scans, abandoned_scans, 0)

    return scans

def find_segs_one_scan(points, H_dist, min_seg_point):
    """
    find all the segments in one scan
    """
    num_points = points.shape[0]
    adjacent_dist = np.sqrt( (points[0:num_points-1,0] - points[1:num_points,0])**2 + \
                             (points[0:num_points-1,1] - points[1:num_points,1])**2 + \
                             (points[0:num_points-1,2] - points[1:num_points,2])**2 + \
                             np.finfo(np.float32).tiny )
    break_index = np.where(adjacent_dist > H_dist)[0]
    num_segs = break_index.size + 1
    # start point of the first scan is the 0
    segs = np.zeros([num_segs,2],dtype=np.uint32)
    # the end point of last scan is the last point
    segs[num_segs-1,1] = num_points-1

    segs[:num_segs-1, 1] = break_index
    segs[1:num_segs,0] = break_index + 1

    # check the every seg points number
    seg_points_num = segs[:,1] - segs[:,0]
    abandoned_segs = np.where(seg_points_num<min_seg_point)[0]
    if abandoned_segs.size>0:
        segs = np.delete(segs, abandoned_segs, 0)
        num_segs = num_segs - abandoned_segs.size

    # pack segs into dictionary
    seg_dict = {x: points[segs[x,0]:(segs[x,1]+1) ,:3] for x in range(num_segs)}

    return seg_dict, num_segs

def find_connection(one_seg, all_segs, h_dist = 0.5, min_connect_points=2):
    """
    calculate the connections
    """
    connects = []
    for label in all_segs.keys():
        points_1 = np.expand_dims(one_seg, axis=0)
        points_2 = np.expand_dims( all_segs[label], axis=1)
        all_dist = dist_points_to_points(points_1, points_2)
        _ind_con = np.where(all_dist<h_dist)
        if _ind_con[0].shape[0] > min_connect_points:
            connects.append(label)
    return connects

class PointCluster(object):
    """
    To cluster the points
    """
    def __init__(self, config):
        self.clustering_method = config.clustering_method
        self.dist_threshold    = config.dist_threshold
        self.H_dist = config.H_dist
        self.V_dist = config.V_dist
        self.min_connect_points = config.min_connect_points
        self.angle_dif_threshold  = config.angle_dif_threshold
        self.min_points_per_scan = config.min_points_per_scan
        self.min_seg_point = config.min_seg_point
        clustering_dict = {
            "euclidean_dist_": self.euclidean_dist_,
            "scan_run_":self.scan_run_,

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
            index_new=np.array([0])
            while not STOP:
                index_new = np.where(all_dist[index_new[:],:]<T_dist)
                index_new = set(index_new[1]) - set(c_1)
                index_new = list(index_new)
                if index_new == []:
                    STOP = True
                else:
                    c_1 = c_1 + index_new
                    index_new = np.array(index_new)
            c_1 = np.array(c_1)
            clusters.append(points_0[c_1[:],:])
            num_points.append(c_1.shape[0])

            # delete these points
            all_dist = np.delete(all_dist,c_1,0)
            all_dist = np.delete(all_dist,c_1,1)
            points_0 = np.delete(points_0,c_1,0)

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
        all_scans = find_all_scans(points[:,:3], angle_dif, min_scan_points)
        num_scans = all_scans.shape[0]
        # initializaton --> class or parameters
        global_segs = {}
        global_label = 0
        start_point = all_scans[0,0]
        end_point = all_scans[0,1]
        above_segs, num_segs = find_segs_one_scan(points[start_point:end_point,:], V_dist, min_seg_point)
        global_label = num_segs - 1
        global_segs = above_segs.copy()
        for i in range(1,num_scans):
            start_point = all_scans[i,0]
            end_point = all_scans[i,1]+1
            current_segs, num_segs = find_segs_one_scan( points[start_point:end_point,:], V_dist, min_seg_point)
            _temp_dict = {}
            for j in range(num_segs):
                connect_index = find_connection(current_segs[j], above_segs, H_dist, min_connect_points)
                num_connect = len(connect_index)
                # not connection, create new label
                if num_connect == 0:
                    global_label = global_label + 1
                    global_segs[global_label] = current_segs[j]
                    _temp_dict[global_label] = current_segs[j]
                # only one connection, merge it into global segs and check _temp_dict
                if num_connect == 1:
                    global_segs[connect_index[0]] = np.vstack((global_segs[connect_index[0]], current_segs[j]))
                    # check whether min_label has been added to _temp_dict or not
                    if connect_index[0] in _temp_dict.keys():
                        _temp_dict[connect_index[0]] = np.vstack((_temp_dict[connect_index[0]], current_segs[j] ))
                    else:
                        _temp_dict[connect_index[0]] = current_segs[j]

                # more two connection, merge the global segments and above_segs, or check duplication in _temp_dict
                if num_connect >= 2:
                    min_label = min(connect_index)
                    connect_index.remove(min_label)
                    # check whether min_label has been added to _temp_dict or not
                    if min_label in _temp_dict.keys():
                        _temp_dict[min_label] = np.vstack((_temp_dict[min_label], current_segs[j] ))
                    else:
                        _temp_dict[min_label] = current_segs[j]
                    # merge the current seg points into global segments
                    global_segs[min_label] = np.vstack((global_segs[min_label], current_segs[j]))
                    ## merge and delete global label and segment
                    for m in connect_index:
                        # merge globally
                        global_segs[min_label] = np.vstack((global_segs[min_label], global_segs[m]))
                        # delete globally
                        global_segs.pop(m) # this bugs

                        # merge locally
                        above_segs[min_label] = np.vstack((above_segs[min_label], above_segs[m]))
                        # delete locally
                        above_segs.pop(m)

                        if m in _temp_dict.keys():
                            _temp_dict[min_label] = np.vstack((_temp_dict[min_label], _temp_dict[m]))
                            # delete in temporaral vector
                            _temp_dict.pop(m)
            above_segs = {}
            above_segs = _temp_dict.copy()


        # convert the clusters from dictionary into list
        segs_list = [global_segs[i] for i in global_segs.keys()]

        return segs_list, len(segs_list)







