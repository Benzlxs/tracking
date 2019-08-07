# point cloud visualization
import numpy as np
#from open3d import *
import open3d
from hbtk.utils.bbox3d_ops import Bbox3D

_lines_vids = np.array([[0,1],[0,3],[1,2],[2,3],
                        [4,5],[4,7],[5,6],[6,7],
                        [0,4],[1,5],[2,6],[3,7]] )

def get_corners_rotation(points, angle):
    """
    fetch all the corner points along the given heading
    x= x*cos - y*sin
    y= x*sin + y*cos
    """
    mean = np.mean(points, axis=0)
    point_adjust = points - mean

    _sin = np.sin(-angle)
    _cos = np.cos(-angle)
    x_sin = point_adjust[:,0]*_sin
    x_cos = point_adjust[:,0]*_cos
    y_sin = point_adjust[:,1]*_sin
    y_cos = point_adjust[:,1]*_cos

    rotat_x = x_cos - y_sin
    rotat_y = x_sin + y_cos

    x0 = np.min(rotat_x)
    x1 = np.max(rotat_x)
    y0 = np.min(rotat_y)
    y1 = np.max(rotat_y)
    z0 = np.min(point_adjust[:,2])
    z1 = np.max(point_adjust[:,2])
    corners = np.array([[x0, y0, z0],
                        [x0, y1, z0],
                        [x1, y1, z0],
                        [x1, y0, z0],
                        [x0, y0, z1],
                        [x0, y1, z1],
                        [x1, y1, z1],
                        [x1, y0, z1]])
    # rotate -angle (clockwise) to go back to original coordinate
    _sin = np.sin(angle)
    _cos = np.cos(angle)
    corners_back = corners.copy()
    corners_back[:,0] = corners[:,0]*_cos - corners[:,1]*_sin
    corners_back[:,1] = corners[:,0]*_sin + corners[:,1]*_cos

    corners_back = corners_back + mean

    return corners_back

def plot_all_clusters_with_rotation(clusters, rotation):
    """
    clusters is lists of point cloud in every cluster
    """
    # get all the cluster
    colors  = []
    points  = []
    corners = []
    for i in range(len(clusters)):
        # min x, y, z
        _cr = get_corners_rotation(clusters[i], rotation[i])
        corners.append(_cr)
        # random generate colors
        colors.append([np.random.rand(),
                       np.random.rand(),
                       np.random.rand() ])

        # get points
        points.append(clusters[i])


    # draw points
    pcl = [get_points_pcl(points[i], colors[i]) for i in range(len(points))]

    # get linesets

    bboxes_lineset_ls = [lineset_rotation_cor(corners[i], color = colors[i]) for i in range(len(corners))]

    # plot
    open3d.draw_geometries(bboxes_lineset_ls + pcl)


def plot_all_clusters(clusters):
    """
    clusters is lists of point cloud in every cluster
    """
    # get all the cluster
    min_cor = []
    max_cor = []
    colors  = []
    points  = []
    for i in range(len(clusters)):
        # min x, y, z
        min_cor.append([np.min(clusters[i][:,0]),
                        np.min(clusters[i][:,1]),
                        np.min(clusters[i][:,2])])
        # max x, y, z
        max_cor.append([np.max(clusters[i][:,0]),
                        np.max(clusters[i][:,1]),
                        np.max(clusters[i][:,2])])
        # random generate colors
        colors.append([np.random.rand(),
                       np.random.rand(),
                       np.random.rand() ])

        # get points
        points.append(clusters[i])


    # draw points
    pcl = [get_points_pcl(points[i], colors[i]) for i in range(len(points))]

    # get linesets

    bboxes_lineset_ls = [lineset_min_max_cor(min_cor[i], max_cor[i], color = colors[i]) for i in range(len(max_cor))]

    # plot
    open3d.draw_geometries(bboxes_lineset_ls + pcl)

def plot_all_clusters_labels(clusters, labels):
    """
    clusters is lists of point cloud in every cluster
    """
    # get all the cluster
    min_cor = []
    max_cor = []
    colors  = []
    points  = []
    for i in range(len(clusters)):
        # min x, y, z
        min_cor.append([np.min(clusters[i][:,0]),
                        np.min(clusters[i][:,1]),
                        np.min(clusters[i][:,2])])
        # max x, y, z
        max_cor.append([np.max(clusters[i][:,0]),
                        np.max(clusters[i][:,1]),
                        np.max(clusters[i][:,2])])
        # random generate colors
        if labels[i]==0:
            colors.append([1,1,0])
        else:
            colors.append([0,0,1])
        #colors.append([np.random.rand(),
        #               np.random.rand(),
        #               np.random.rand() ])

        # get points
        points.append(clusters[i])





    # draw points
    pcl = [get_points_pcl(points[i], colors[i]) for i in range(len(points))]

    # get linesets

    bboxes_lineset_ls = [lineset_min_max_cor(min_cor[i], max_cor[i], color = colors[i]) for i in range(len(max_cor))]

    # plot
    open3d.draw_geometries(bboxes_lineset_ls + pcl)


def plot_all_clusters_dic(clusters_dic):
    """
    clusters is lists of point cloud in every cluster
    """
    clusters = [clusters_dic[i] for i in clusters_dic.keys()]
    # get all the cluster
    min_cor = []
    max_cor = []
    colors  = []
    points  = []
    for i in range(len(clusters)):
        # min x, y, z
        min_cor.append([np.min(clusters[i][:,0]),
                        np.min(clusters[i][:,1]),
                        np.min(clusters[i][:,2])])
        # max x, y, z
        max_cor.append([np.max(clusters[i][:,0]),
                        np.max(clusters[i][:,1]),
                        np.max(clusters[i][:,2])])
        # random generate colors
        colors.append([np.random.rand(),
                       np.random.rand(),
                       np.random.rand() ])

        # get points
        points.append(clusters[i])





    # draw points
    pcl = [get_points_pcl(points[i], colors[i]) for i in range(len(points))]

    # get linesets

    bboxes_lineset_ls = [lineset_min_max_cor(min_cor[i], max_cor[i], color = colors[i]) for i in range(len(max_cor))]

    # plot
    open3d.draw_geometries(bboxes_lineset_ls + pcl)

def draw_points_open3d(points, color=[0,1,1], show=True):
    points = points[:,0:3]
    pcl = open3d.PointCloud()
    pcl.points = open3d.Vector3dVector(points[:,:3])
    pcl.colors = open3d.Vector3dVector(color)
    #pcl.paint_uniform_color(color)
    if show:
      open3d.draw_geometries([pcl])
    return pcl



def get_points_pcl(points, color=[0,1,1], show=False):
    points = points[:,0:3]
    pcl = open3d.PointCloud()
    pcl.points = open3d.Vector3dVector(points[:,0:3])
    pcl.paint_uniform_color(color)
    return pcl

def lineset_rotation_cor(corners, color=[0,1,1]):
    """
    get the lineset according min corner and max corner
    """
    # colors = np.random.rand(1,3)
    lineset = get_lineset(corners, color)

    return lineset



def lineset_min_max_cor(min_corner, max_corner, color=[0,1,1]):
    """
    get the lineset according min corner and max corner
    """
    x0,y0,z0 = min_corner
    x1,y1,z1 = max_corner
    corners = np.array([[x0, y0, z0],
                        [x0, y1, z0],
                        [x1, y1, z0],
                        [x1, y0, z0],
                        [x0, y0, z1],
                        [x0, y1, z1],
                        [x1, y1, z1],
                        [x1, y0, z1]])
    # colors = np.random.rand(1,3)
    lineset = get_lineset(corners, color)

    return lineset



def get_lineset(corners, color=[1,0,0]):
    """
    get the lineset using all corners
    """
    line_set = open3d.LineSet()
    line_set.points = open3d.Vector3dVector(corners)
    line_set.lines = open3d.Vector2iVector(_lines_vids)
    colors = [color for i in range(corners.shape[0])]
    line_set.colors = open3d.Vector3dVector(colors)
    return line_set








