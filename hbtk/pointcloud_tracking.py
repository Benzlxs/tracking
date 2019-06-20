# Copyright 2019, Xuesong LI, (email: benzlee08@gmail.com). All Rights Reserved.

import os
import sys
import shutil
import pickle
from pathlib import Path
import fire
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from matplotlib import gridspec
from skimage import io


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join( os.path.dirname(ROOT_DIR), 'hbtk'))
sys.path.append(os.path.join( os.path.dirname(ROOT_DIR), 'hbtk', 'protos'))


from google.protobuf import text_format
from hbtk.protos import pipeline_pb2
from hbtk.detectors.two_merge_one import HybridDetector
from hbtk.trackers.sort import Sort
from utils.pc_plot import point_cloud_2_birdseye

def pointcloud_tracking(config_path=None,
             output_dir='./results',
             display = True,
             ):
    """ main function for tracking

    we run different detectors and combine them together to improve efficiency
    without decreasing perfromance too much. Detection-->Tracking

    Args:
        config_path:
        output_dir:

    Returns:
        sss

    Raises:
        IOError: An error occurred accesssing hte bigtable. Table of object

    """
    #display function first
    dataset_root_path='/home/ben/Dataset/KITTI/2011_09_26/2011_09_26_drive_0001_sync'
    dataset_path = Path(dataset_root_path)
    if display:
        cmap = matplotlib.cm.Spectral
        cmap.set_under(color='black')
        cmap.set_bad(color='black')
        plt.ion()
        #fig = plt.figure(figsize=(10, 8))
        fig = plt.figure()

    if display:
        spec2 = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1, 2.2])
        ax1 = fig.add_subplot(spec2[0,0])
        ax2 = fig.add_subplot(spec2[1,0])

        image_path = list(sorted(dataset_path.glob('image_02/data/*.png')))
        pointcloud_path = list(sorted(dataset_path.glob('velodyne_points/data/*.bin')))
        assert len(image_path) == len(pointcloud_path), "the image and point cloud files should be the same"
        for i in range(len(image_path)):
            img_path = str(image_path[i])
            im =io.imread(img_path)
            # display image
            ax1.cla()
            ax1.imshow(im)
            ax1.set_title('image')
            ax1.set_xticks([])
            ax1.set_yticks([])

            # display point cloud
            pc_path = str(pointcloud_path[i])
            points_v = np.fromfile(pc_path, dtype=np.float32, count=-1).reshape([-1, 4])
            pc_img = point_cloud_2_birdseye(points_v)
            ax2.cla()
            ax2.imshow(pc_img,cmap=cmap, vmin = 1, vmax=255)
            ax2.set_title('point cloud')
            ax2.set_xticks([])
            ax2.set_yticks([])


            #plt.title('Tracking')
            fig.canvas.flush_events()
            #plt.draw()
            plt.show()
            # ax1.cla()
            time.sleep(0.2)





if __name__=='__main__':
    fire.Fire()


