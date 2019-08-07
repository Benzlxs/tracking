# a combination of clustering filtering methods

import numpy as np
import numba

def calculate_ang_overlap(lab_ang):
    """
    calculate the angle overlap matrix
            ang_overlap = np.max(0, np.min(rightmost1, rightmost2 ) - np.max(leftmost1, leftmost2));
            construct the matrix using above equations
    angle calculation is in the vehicle coordinate, from left to right, and strart from positive y-axis to negative y-axis
    """
    lab_ang = np.array(lab_ang)
    h_ang_left, v_ang_left = np.meshgrid(lab_ang[:, 0], lab_ang[:, 0])
    h_ang_left = np.expand_dims(h_ang_left, axis=2)
    v_ang_left = np.expand_dims(v_ang_left, axis=2)

    h_ang_right, v_ang_right = np.meshgrid(lab_ang[:, 1], lab_ang[:, 1])
    h_ang_right = np.expand_dims(h_ang_right, axis=2)
    v_ang_right = np.expand_dims(v_ang_right, axis=2)

    hv_ang_left = np.concatenate((h_ang_left, v_ang_left), axis=2)
    hv_ang_right = np.concatenate((h_ang_right, v_ang_right), axis=2)

    ang_diff = np.amin(hv_ang_right, axis=2) - np.amax(hv_ang_left, axis=2)
    ang_diff = np.expand_dims(ang_diff, axis=2)
    zeros_ = np.zeros_like(ang_diff)

    ang_diff_zeros = np.concatenate((ang_diff, zeros_), axis=2)
    # zero denotes no overlap, positive means overlap
    overlap_matrix = np.amax(ang_diff_zeros, axis=2)

    return overlap_matrix


def calculate_range_overlap(lab_rng):
    """
    calculate the range diff overlap matrix
    """
    lab_rng = np.array(lab_rng)
    h_rng, v_rng = np.meshgrid(lab_rng, lab_rng)

    diff_rng = v_rng - h_rng  #h_rng - v_rng
    diff_rng = np.expand_dims(diff_rng, axis=2)

    zeros_ = np.zeros_like(diff_rng)
    diff_rng_zeros = np.concatenate((diff_rng, zeros_), axis=2)

    rng_overlap_matrix = np.amax(diff_rng_zeros, axis=2)

    return rng_overlap_matrix


def check_min_num_points(rng, a=1074.0, b=0.1220, c=-14.7):
    """
    exponential function relationship
    """
    return a * np.exp(-b * rng) + c


def find_all_angle(c):
    """
    Calculate the angles for every points
    The angle calculation is from left to right, the positive y-axis to negative y-axis, x is always positive.
    Here is the x and y-axis under vehicle convention are swapped off.
    """
    y = c[:, 0]
    x = c[:, 1]
    angles = np.arctan2(y, x)
    return angles


def find_all_range(c):
    """
    calculate all the range
    argument:
        c: np.array of all points inside
    """
    return np.sqrt(c[:, 0]**2 + c[:, 1]**2 + c[:, 2]**2)


def get_overlap_max(mm_xyz):
    """
    calculate the overlap matrix

            x_overlap = Math.max(0, Math.min(rect1.rightmost, rect2.rightmost) - Math.max(rect1.leftmost, rect2.leftmost));
            y_overlap = Math.max(0, Math.min(rect1.bottom, rect2.bottom) - Math.max(rect1.top, rect2.top));
            overlapArea = x_overlap * y_overlap;

            construct the matrix using above equations

    """
    s_mm_xyz = np.array(mm_xyz)  # n*6
    h_x_min, v_x_min = np.meshgrid(s_mm_xyz[:, 0], s_mm_xyz[:, 0])  # n*n*1
    h_x_min = np.expand_dims(h_x_min, axis=2)
    v_x_min = np.expand_dims(v_x_min, axis=2)

    h_x_max, v_x_max = np.meshgrid(s_mm_xyz[:, 1], s_mm_xyz[:, 1])
    h_x_max = np.expand_dims(h_x_max, axis=2)
    v_x_max = np.expand_dims(v_x_max, axis=2)

    h_y_min, v_y_min = np.meshgrid(s_mm_xyz[:, 2], s_mm_xyz[:, 2])
    h_y_min = np.expand_dims(h_y_min, axis=2)
    v_y_min = np.expand_dims(v_y_min, axis=2)

    h_y_max, v_y_max = np.meshgrid(s_mm_xyz[:, 3], s_mm_xyz[:, 3])
    h_y_max = np.expand_dims(h_y_max, axis=2)
    v_y_max = np.expand_dims(v_y_max, axis=2)

    hv_x_min = np.concatenate((h_x_min, v_x_min), axis=2)  # n*n*2
    hv_x_max = np.concatenate((h_x_max, v_x_max), axis=2)
    hv_y_min = np.concatenate((h_y_min, v_y_min), axis=2)
    hv_y_max = np.concatenate((h_y_max, v_y_max), axis=2)

    dx = np.amin(hv_x_max, axis=2) - np.amax(hv_x_min, axis=2)  # n*n*1
    dx = np.expand_dims(dx, axis=2)
    dy = np.amin(hv_y_max, axis=2) - np.amax(hv_y_min, axis=2)
    dy = np.expand_dims(dy, axis=2)

    z_ = np.zeros_like(dx)

    dx = np.concatenate((dx, z_), axis=2)  # n*n*2
    dy = np.concatenate((dy, z_), axis=2)

    dx = np.amax(dx, axis=2)
    dx = np.expand_dims(dx, axis=2)
    dy = np.amax(dy, axis=2)
    dy = np.expand_dims(dy, axis=2)

    #overlap_matrix = dx*dy  # n*n*1
    overlap_matrix = np.multiply(dx, dy)

    return overlap_matrix


@numba.njit
def filtering_with_xyz_jit(clusters, max_length_cluster, max_width_cluster):
    mm_xyz = [np.array([np.min(clusters[i][:,0]), np.max(clusters[i][:,0]), np.min(clusters[i][:,1]), np.max(clusters[i][:,1])])   for i in range(len(clusters))]
    count = 0
    for _c in mm_xyz:
        x_min = _c[0]
        x_max = _c[1]
        y_min = _c[2]
        y_max = _c[3]
        if ((x_max - x_min) > max_length_cluster) or \
                ((y_max - y_min) > max_width_cluster):
            clusters.pop(count)
            mm_xyz.pop(count)
        count = count + 1

    return mm_xyz

@numba.njit
def filtering_with_z_jit(clusters, max_hight_cluster):
    """
    filter out clustering with hight threshold (z-axis)
    """
    count = 0
    for _clu in clusters:
        z_min = np.min(_clu[:, 2])
        z_max = np.max(_clu[:, 2])
        if (z_max - z_min) < max_hight_cluster:
            clusters.pop(count)
        count += 1



class ClustersFiltering(object):
    """
    To remove unwanted clustering and merge the cluster
    descriptions: 1. filtering the clustering first, e.g. the size of clusters, minimum number of points;
        2. merge the clusters if there is overlap in vertical direction.
    """

    def __init__(self, config):
        self.clusterfiltering_method = config.clusterfiltering_method
        self.min_points_cluster = config.min_points_cluster
        self.max_length_cluster = config.max_length_cluster
        self.max_width_cluster = config.max_width_cluster
        self.max_hight_cluster = config.max_hight_cluster
        self.angle_threshold = config.angle_threshold
        # decide the selected frustum region
        self.frustum_offset = config.frustum_offset
        self.frustum_ratio = config.frustum_ratio
        self.frustum_max_x = config.frustum_max_x

        self._clusterfiltering_ = self.filtering_clusters

        _teniques_names = config.clusterfiltering_method.split('_')

        if 'xyzsize' in _teniques_names:
            self.xyzsize = True
        else:
            self.xyzsize = False

        if 'merge' in _teniques_names:
            self.merge = True
        else:
            self.merge = False

        if 'zh' in _teniques_names:
            self.zh = True
        else:
            self.zh = False

        if 'numpoints' in _teniques_names:
            self.numpoints = True
        else:
            self.numpoints =False


    def filtering_clusters(self, clusters):
        """
        filtering the cluster
        1. remove extreme large clusters
        2. merge clusters
        3. use self.max_hight_cluster to filter out clusters
        arguements:
            clusters is a list;
            mm_xyz is a list of np.array([x_min, x_max, y_min, y_max, z_min, z_max])
        """
        if not isinstance(clusters, list):
            raise ValueError("The data type of clusters should be list")


        if self.xyzsize:
            mm_xyz = filtering_with_xyz_jit(clusters, self.max_length_cluster, self.max_width_cluster)
        else:
            mm_xyz = [np.array([np.min(clusters[i][:,0]), np.max(clusters[i][:,0]), \
                                np.min(clusters[i][:,1]), np.max(clusters[i][:,1])])   for i in range(len(clusters))]


        # merge the clusters with overlap in vertical direction
        if self.merge:
            clusters = self.merge_clusters(clusters, mm_xyz)

        # remove clusters with small hight
        # clusters = self.filtering_with_z(clusters)
        if self.zh:
            filtering_with_z_jit(clusters, self.max_hight_cluster)

        # remove clusters with too small number of points
        if self.numpoints:
            self.filtering_with_num_points(clusters)

        return clusters

    def filtering_with_xy(self, clusters, mm_xyz):
        """
        use size to filter out clusters, width and length (x and y axis)
        """
        selected_clusters = []
        selected_mm_xyz = []
        count = 0
        for _c in mm_xyz:
            x_min = _c[0]
            x_max = _c[1]
            y_min = _c[2]
            y_max = _c[3]
            if ((x_max - x_min) < self.max_length_cluster) and \
                    ((y_max - y_min) < self.max_width_cluster):
                selected_clusters.append(clusters[count])
                selected_mm_xyz.append(_c)
            count = count + 1

        return selected_clusters, selected_mm_xyz

    def merge_clusters(self, clusters, mm_xyz):
        """
        merge two clusters
        """
        overlap_matrix = get_overlap_max(mm_xyz)
        overlap_matrix = np.squeeze(overlap_matrix, axis=2)
        # initial
        s_clusters = []
        # stop unitl the overlap_matrix is empty
        while overlap_matrix.shape[0]:
            STOP = False
            c_1 = []
            c_1.append(0)
            index_new = np.array([0])
            while not STOP:
                index_new = np.where(overlap_matrix[index_new[:], :] > 0)
                index_new = set(index_new[1]) - set(c_1)
                index_new = list(index_new)
                if index_new == []:
                    STOP = True
                else:
                    c_1 = c_1 + index_new
                    index_new = np.array(index_new)
            _c_ = np.array(c_1)
            overlap_matrix = np.delete(overlap_matrix, _c_, 0)
            overlap_matrix = np.delete(overlap_matrix, _c_, 1)
            _cb = c_1.copy()
            if len(c_1) == 1:
                s_clusters.append(clusters[c_1[0]])
                clusters.pop(c_1[0])
            else:
                _clus = clusters[c_1[0]]
                c_1.remove(c_1[0])  # remove value from list
                for i in c_1:
                    _clus = np.vstack((_clus, clusters[i]))
                s_clusters.append(_clus)
                for i in sorted(_cb, reverse=True):
                    clusters.pop(i)  # remve by index

        return s_clusters

    def filtering_with_z(self, clusters):
        """
        filter out clustering with hight threshold (z-axis)
        """
        s_clusters = []
        for _clu in clusters:
            z_min = np.min(_clu[:, 2])
            z_max = np.max(_clu[:, 2])
            if (z_max - z_min) > self.max_hight_cluster:
                s_clusters.append(_clu)

        return s_clusters

    def filtering_with_num_points(self, clusters):
        """
        Filtering clusters with number of inner points, and the selectd region should be in front of car, and the boundary
        truncation shouold not happen.
        The selected region is a frustum
        The filtering procedures:
            1. give the occlusion level for every clustering within the selected region
            2. use the number of points to filter out most of useless region
        Angle calculation is from left to right and +y-axis to -y-axis under kITTI convention
        """
        lab_clusters = []  # the clusters are used to label the occlusion level
        lab_ang = []
        lab_rng = []
        for c in clusters:
            _x_mean = np.mean(c[:, 0])
            _y_mean = np.mean(c[:, 1])
            #if (_y_mean < self.frustum_ratio*(_x_mean - self.frustum_offset)) and \
            #        ( -_y_mean < self.frustum_ratio*(_x_mean - self.frustum_offset) ) and \
            #            (_x_mean < self.frustum_max_x):
            if _x_mean < self.frustum_max_x:
                all_ang = find_all_angle(c)
                leftmost_ang = np.min(all_ang) - self.angle_threshold
                rightmost_ang = np.max(all_ang) + self.angle_threshold
                # all_range = find_all_range(c)
                mid_rng = np.mean(find_all_range(c))
                lab_clusters.append(c)
                lab_ang.append([leftmost_ang, rightmost_ang])
                lab_rng.append(mid_rng)

        # no cluster is located in specified region
        if len(lab_ang) > 0:
            ang_overlap_matrix = calculate_ang_overlap(lab_ang)
            rng_overlap_matrix = calculate_range_overlap(lab_rng)

            occlusion_matrix = ang_overlap_matrix * rng_overlap_matrix

            occlusion_matrix = np.sum(
                occlusion_matrix,
                axis=1)  # n*1 vector to indicate the occlusion level
            count = 0
            for _cc in lab_clusters:
                if occlusion_matrix[count] == 0:  # occluson level = 0
                    # mean_rng = np.mean(np.sqrt(lab_clusters[i][:,0]**2 + lab_clusters[i][:,1]**2 + lab_clusters[i][:,2]**2))
                    _x_mean = np.mean(_cc[:, 0])
                    _y_mean = np.mean(_cc[:, 1])
                    # located in region of interest
                    if (_y_mean < self.frustum_ratio*(_x_mean - self.frustum_offset)) and \
                            (-_y_mean < self.frustum_ratio*(_x_mean - self.frustum_offset)):
                        mean_rng = np.mean(find_all_range(_cc))
                        min_num_points = check_min_num_points(mean_rng)
                        if _cc.shape[0] < min_num_points:
                            clusters.remove(_cc)
                count += 1

