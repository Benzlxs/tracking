### from http://ronny.rest/tutorials/module/pointclouds_01/point_cloud_birdseye/

import numpy as np

def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


# ==============================================================================
#               Convert xyz into coordinates in image
# ==============================================================================
def convert_xyz_to_img(det,
                       res=0.1,
                       side_range=(-60., 60.),  # left-most to right-most
                       fwd_range = (-10., 70.), # back-most to forward-most
                       height_range=(-1.8, 1.),  # bottom-most to upper-most
                       ):
    """
    args:
        det: [class, x, y, z, l, w, h]
    return:
        [right_up, left_bottom] ([x0,y0, x1, y1])
    """
    # checking bounding points
    x_lidar = det[0]
    y_lidar = det[1]
    l_lidar = det[3]
    w_lidar = det[4]

    side_range = np.array(side_range)
    fwd_range = np.array(fwd_range)
    height_range = np.array(height_range)
    if x_lidar > fwd_range[1]:
        x_lidar = fwd_range[1]
    if x_lidar < fwd_range[0]:
        x_lidar = fwd_range[0]

    if y_lidar > side_range[1]:
        y_lidar = side_range[1]
    if y_lidar < side_range[0]:
        y_lidar = side_range[0]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_lidar / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_lidar / res).astype(np.int32)  # y axis is -x in LIDAR
    l_img_half = ( l_lidar / res / 2).astype(np.int32)
    w_img_half = ( w_lidar / res / 2).astype(np.int32)

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    return [x_img - w_img_half, y_img - l_img_half, x_img + w_img_half, y_img + l_img_half]


# ==============================================================================
#                                                         POINT_CLOUD_2_BIRDSEYE
# ==============================================================================
def point_cloud_2_birdseye(points,
                           res=0.1,
                           side_range=(-60., 60.),  # left-most to right-most
                           fwd_range = (-10., 70.), # back-most to forward-most
                           height_range=(-1.8, 1.),  # bottom-most to upper-most
                           ):
    """ Creates an 2D birds eye view representation of the point cloud data.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        2D numpy array representing an image of the birds eye view.
    """
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    z_filt = np.logical_and((z_points > height_range[0]), (z_points < height_range[1]))
    filter = np.logical_and(f_filt, s_filt, z_filt)
    indices = np.argwhere(filter).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(pixel_values,
                                min=height_range[0],
                                max=height_range[1])

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img] = pixel_values

    return im
