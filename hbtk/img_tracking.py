# Copyright 2019, Xuesong LI, (email: benzlee08@gmail.com). All Rights Reserved.

import os
import sys
import shutil
import pickle
import pathlib
import fire
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join( os.path.dirname(ROOT_DIR), 'hbtk'))
sys.path.append(os.path.join( os.path.dirname(ROOT_DIR), 'hbtk', 'protos'))


from google.protobuf import text_format
from hbtk.protos import pipeline_pb2
from hbtk.detectors.two_merge_one import HybridDetector
from hbtk.trackers.sort import Sort

def img_tracking(config_path,
             output_dir='./results',
             display = False,
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
    config = pipeline_pb2.TrackingPipeline()
    output_folder_dir = pathlib.Path(output_dir)
    output_folder_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "r") as f:
        protos_str = f.read()
        text_format.Merge(protos_str, config)
    shutil.copyfile(config_path, str(output_dir+"/"+"pipeline.config"))
    detector_config = config.detector
    filter_config = config.filter
    tracker_config = config.tracker
    dataset_config = config.dataset

    # build method
    hbtk_detectors = HybridDetector(detector_config)


    # initialization of parameters
    total_frames = 0
    total_time = 0.0

    # initializatio for plotting
    if(display):
        colours = np.random.rand(32,3) #used only for display
        plt.ion()
        fig = plt.figure()

    # run method
    for subset in hbtk_detectors.data_subset:
        print("Performing tracking in dataset %s"%(subset))
        # creating the tracker
        mot_tracker = Sort( config = tracker_config, data_association = filter_config.data_association )

        # file to save the detection results
        det_folder_dir = output_dir+ '/detection/' + 'interval_%d'%(detector_config.interval_num)
        if not os.path.exists(det_folder_dir):
            os.makedirs(det_folder_dir)
        det_file = open(det_folder_dir + '/%s.txt'%(subset), 'w' )

        with open(output_dir+'/%s.txt'%(subset),'w') as out_file:
            for frame in range(1,(hbtk_detectors.get_num_frames(subset)+1)):
                object_dets = hbtk_detectors.fetch_detection(subset, frame)[:,2:7]

                # saving the new detection combination
                for d in object_dets:
                    det_file.write('%d,-1,%.2f,%.2f,%.2f,%.2f,%.2f,-1,-1,-1\n'%(frame, d[0], d[1], \
                                                                                d[2],d[3],d[4] ))

                object_dets[:,2:4] +=object_dets[:,0:2]  # convert to [x1, y1, w, h] to [x1, y1, x2, y2]

                if display:
                    ax1 = fig.add_subplot(111, aspect='equal')
                    fn = dataset_config.database_dir + '/%s/%s-FRCNN/img1/%06d.jpg'%(dataset_config.phase,subset,frame)
                    im =io.imread(fn)
                    ax1.imshow(im)
                    plt.title(subset+' Tracked Targets')

                start_time = time.time()

                if frame % detector_config.interval_num==1:
                    reset_confid = True
                else:
                    reset_confid = True

                trackers_results = mot_tracker.update(object_dets, reset_confid=reset_confid)

                # middle parameters to indicate the running program
                cycle_time = time.time() - start_time
                total_frames += 1
                total_time += cycle_time

                # saving the results
                for d in trackers_results:
                    # saving the detection results with confidence
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1],d[5]),file=out_file)
                    if display:
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))
                        ax1.set_adjustable('box-forced')

                if display:
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()

        print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))
        det_file.close()


if __name__=='__main__':
    fire.Fire()


