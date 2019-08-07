# filtering the point cloud

import numpy as np
from numba import njit, jit
from utils.box_np_ops import remove_outside_points, lidar_to_camera, project_to_image


def get_road_plane(plane_path):
    """
    Read the road plane from file
    """
    with open(plane_path, 'r') as input_file:
        lines = input_file.readlines()
        input_file.close()
    # Plane coefficients stored in 4th row
    lines = lines[3].split()

    # Convert str to float
    lines = [float(i) for i in lines]
    plane = np.asarray(lines)

    # In Kitti's frame of reference, +y is down
    # if plane[1] > 0:
    #     plane = -plane

    norm = np.linalg.norm(plane[0:3])
    plane = plane / norm

    return plane

def remove_semantic_road(points,
                         semantic_img,
                         rect,
                         Trv2c,
                         P2,
                         semantic_conf=0.005):
    """
    using the road semantic image to filter out points
    """
    # get the location of point in the image
    xyz_camera = lidar_to_camera(points[:, 0:3], rect, Trv2c)
    xy_camera = project_to_image(xyz_camera, P2)
    xy_camera = xy_camera.astype(np.int32)

    [h, w] = semantic_img.shape
    index_y = np.where(xy_camera[:, 1] > (h - 1))
    xy_camera[index_y, 1] = h - 1
    index_x = np.where(xy_camera[:, 0] > (w - 1))
    xy_camera[index_x, 0] = w - 1

    semantic_points = semantic_img[xy_camera[:, 1], xy_camera[:, 0]]

    _index = np.where(semantic_points[:] < semantic_conf * 255)

    return points[_index[0][:], :]


#@jit('float32[:, :], float32, float32', nopython=True)
@njit
def remove_out_of_range_jit(
        points,
        ranges,
        near_ranges,
):
    """
    removing points out-of-range
    """
    i = 0  # x
    _index = np.where((points[:, i] > ranges[0,i]) &
                      (points[:, i] < ranges[1,i]))
    points = points[_index[0][:], :]
    i = 1  # y
    _index = np.where((points[:, i] > ranges[0,i]) &
                      (points[:, i] < ranges[1,i]))
    points = points[_index[0][:], :]
    i = 2  # z
    _index = np.where((points[:, i] > ranges[0,i]) &
                      (points[:, i] < ranges[1,i]))
    points = points[_index[0][:], :]

    _index = np.where((points[:, 0] > near_ranges[0]) |
                          (points[:, 2] > near_ranges[1]))
    points = points[_index[0][:], :]

    return points


def remove_out_of_range(
        points,
        ranges,
        near_ranges=None,
):
    """
    removing points out-of-range
    """
    if len(ranges) != 2:
        raise ValueError("Points have the wrong shape: {}".format(len(ranges)))
    i = 0  # x
    _index = np.where((points[:, i] > ranges[0][i]) &
                      (points[:, i] < ranges[1][i]))
    points = points[_index[0][:], :]
    i = 1  # y
    _index = np.where((points[:, i] > ranges[0][i]) &
                      (points[:, i] < ranges[1][i]))
    points = points[_index[0][:], :]
    i = 2  # z
    _index = np.where((points[:, i] > ranges[0][i]) &
                      (points[:, i] < ranges[1][i]))
    points = points[_index[0][:], :]

    if near_ranges is not None:
        _index = np.where((points[:, 0] > near_ranges[0]) |
                          (points[:, 2] > near_ranges[1]))
        points = points[_index[0][:], :]

    return points


def remove_road_plane(
        points_v,
        ground_plane,
        offset_dist,
        max_dist=None,
):
    """
    remove points with the road plane
    """
    points = np.asarray(points_v[:, :3]).transpose()
    ones_col = -np.ones(points.shape[1])

    padded_points = np.vstack([points, ones_col])
    offset_plane = ground_plane + [0, 0, 0, offset_dist]

    # create plane filter
    dot_prod = np.dot(offset_plane, padded_points)
    dot_prod_far = np.dot(ground_plane, padded_points)

    if max_dist is None:
        point_filter = np.where(dot_prod > 0)
    else:
        point_filter = np.where(((dot_prod > 0) | (points[0, :] > max_dist)) &
                                (dot_prod_far > 0))

    points = np.asarray(points).transpose()

    return points_v[point_filter[0][:], :]


def remove_gound_points_multiple_planes(points,road_parameter,
                                        offset_dist, h_sub, w_sub,H, W):
    """
    removing the points under ground
    """
    h_mat, w_mat = road_parameter.shape
    selected_points = None
    for h_i in range(h_mat):
        for w_j in range(w_mat):
            height_z = road_parameter[h_i, w_j]
            # the invaild sub region is 100 high
            if height_z < 20:
                selected_1 = np.logical_and((points[:,0]>(h_i*h_sub + H[0])), (points[:,0]<((h_i+1)*h_sub + H[0])))
                selected_2 = np.logical_and((points[:,1]>(w_j*w_sub + W[0])), (points[:,1]<((w_j+1)*w_sub + W[0])))
                selects = np.logical_and(selected_1, selected_2)
                subregion_points = points[selects,:]
                selected_3 = subregion_points[:,2] > (height_z+offset_dist)
                subregion_points = subregion_points[selected_3,:]

                if selected_points is None:
                    selected_points = subregion_points.copy()
                else:
                    selected_points = np.vstack((selected_points, subregion_points))

    return selected_points


class PointFilter(object):
    """
    To removing the uesless points
    """

    def __init__(self, config):
        """
        get the inital
        """
        self.filtering_method = config.filtering_method
        self.offset_dist = config.offset_dist
        self.max_filtering_dist = config.max_filtering_dist
        self.semantic_road_conf = config.semantic_road_conf
        self.region_ranges      = [[config.region_ranges[0], config.region_ranges[1], config.region_ranges[2]],\
                                   [config.region_ranges[3], config.region_ranges[4], config.region_ranges[5]]]
        self.close_ranges = config.close_ranges

        self.W = [config.region_ranges[1], config.region_ranges[4]]
        self.H = [config.region_ranges[0], config.region_ranges[3]]
        self.h_sub = config.h_sub
        self.w_sub = config.w_sub

        filtering_method_dict = {
            "semantic_raod_range": self.semantic_ranges_roadplane,
            "Multiregion_planes":self.multiregion_roadplanes,
            "Road_plane":self.road_plane,
        }

        self._filtering_ = filtering_method_dict[self.filtering_method]


    def multiregion_roadplanes(self, points, semantic_img, road_parameter,
                               image_shape, calib):
        """
        using the multiple road region plane to remove the road points.
        """
        rect = calib['R0_rect']
        P2 = calib['P2']
        Trv2c = calib['Tr_velo_to_cam']

        # crop the frustum of image view, using the cropped points
        # points = remove_outside_points(points, rect, Trv2c, P2, image_shape)

        # removing road semantic points

        # change it back after finishing the plane testing
        if True:
            points = remove_semantic_road(points,
                                      semantic_img,
                                      rect,
                                      Trv2c,
                                      P2,
                                      semantic_conf=self.semantic_road_conf)

        # use the multiple sub-region planes to remove ground points
        points = remove_gound_points_multiple_planes(points,
                                                     road_parameter,
                                                     self.offset_dist,
                                                     self.h_sub,
                                                     self.w_sub,
                                                     self.H,
                                                     self.W)

        return points


    def road_plane(self, points, road_parameter_dir):
        """
        filter methods
        """
        ground_ori = get_road_plane(road_parameter_dir)
        road_parameter = [ground_ori[2], -ground_ori[0], -ground_ori[1], -ground_ori[3]]
        road_parameter = np.array(road_parameter)

        points = remove_out_of_range(points,
                                         self.region_ranges,
                                         near_ranges=self.close_ranges)

        # remvoing the road plane points
        points = remove_road_plane(points,
                                   road_parameter,
                                   self.offset_dist,
                                   max_dist=self.close_ranges[0])

        return points
    def semantic_ranges_roadplane(self, points, semantic_img, road_parameter,
                                  image_shape, calib, speedup=False):
        """
        filter methods
        """
        rect = calib['R0_rect']
        P2 = calib['P2']
        Trv2c = calib['Tr_velo_to_cam']

        # crop the frustum of image view, using the cropped points
        # points = remove_outside_points(points, rect, Trv2c, P2, image_shape)

        # removing road semantic points
        points = remove_semantic_road(points,
                                  semantic_img,
                                  rect,
                                  Trv2c,
                                  P2,
                                  semantic_conf=self.semantic_road_conf)

        #  removing out-of-range points
        if speedup:
            _region_rg = np.array(self.region_ranges)
            _close_rg = np.array(self.close_ranges)
            points = remove_out_of_range_jit(points,
                                         _region_rg,
                                         _close_rg)
        else:
            points = remove_out_of_range(points,
                                         self.region_ranges,
                                         near_ranges=self.close_ranges)

        # remvoing the road plane points
        points = remove_road_plane(points,
                                   road_parameter,
                                   self.offset_dist,
                                   max_dist=self.close_ranges[0])

        return points
